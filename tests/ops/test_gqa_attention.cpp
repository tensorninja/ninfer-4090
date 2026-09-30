#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHeadDim       = 256;
constexpr std::int32_t kQuantGroup    = 64;
constexpr std::int32_t kQuantGroups   = kHeadDim / kQuantGroup;
constexpr float kAttentionScale       = 0.0625f;
constexpr std::uint16_t kOutputCanary = 0x7fc1u;

// The Op has two registered compute profiles. A1 and A3 use the same criterion for a given
// profile; token count, geometry, execution envelope, and private launch route do not select it.
constexpr ReductionCriterion kAttentionBf16Criterion{
    /*relative_l2*/ 2.8e-3,
    /*gross_absolute*/ 1.0e-3,
    /*gross_relative_to_max_reference*/ 2.7e-3,
};

constexpr ReductionCriterion kAttentionInt8Criterion{
    /*relative_l2*/ 3.15e-3,
    /*gross_absolute*/ 1.1e-3,
    /*gross_relative_to_max_reference*/ 2.2e-3,
};

struct Geometry {
    const char* name;
    std::int32_t q_heads;
    std::int32_t kv_heads;

    [[nodiscard]] std::int32_t query_group() const { return q_heads / kv_heads; }
};

constexpr Geometry kGeometries[] = {
    {"qwen3_8_27b", 24, 4},
    {"qwen3_6_35b_a3b", 16, 2},
};

struct AttentionCase {
    std::int32_t tokens;
    std::int32_t base;
    std::uint32_t envelope_max;
    std::uint32_t seed;
};

enum class MappingPattern { Identity, Offset, Fragmented };

const char* mapping_name(MappingPattern pattern) {
    switch (pattern) {
    case MappingPattern::Identity:
        return "identity";
    case MappingPattern::Offset:
        return "offset";
    case MappingPattern::Fragmented:
        return "fragmented";
    }
    return "unknown";
}

std::int32_t align_up_page(std::int32_t value) {
    constexpr std::int32_t kFixtureAlignment = 2 * kPagedKVPageSize;
    return ((value + kFixtureAlignment - 1) / kFixtureAlignment) * kFixtureAlignment;
}

std::int32_t physical_page_count(std::int32_t logical_pages, MappingPattern pattern) {
    switch (pattern) {
    case MappingPattern::Identity:
        return logical_pages;
    case MappingPattern::Offset:
        return logical_pages + 2;
    case MappingPattern::Fragmented:
        return 2 * logical_pages + 1;
    }
    return 0;
}

std::vector<std::int32_t> make_block_table(std::int32_t logical_pages, MappingPattern pattern) {
    std::vector<std::int32_t> table(static_cast<std::size_t>(logical_pages));
    switch (pattern) {
    case MappingPattern::Identity:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = page; }
        break;
    case MappingPattern::Offset:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = page + 1; }
        break;
    case MappingPattern::Fragmented:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = 2 * page + 1; }
        break;
    }
    return table;
}

std::size_t q_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                    std::int32_t token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(geometry.q_heads) * static_cast<std::size_t>(token));
}

std::size_t kv_input_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                           std::int32_t token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(geometry.kv_heads) * static_cast<std::size_t>(token));
}

std::size_t cache_index(const Geometry& geometry, std::int32_t padded_context, std::int32_t head,
                        std::int32_t position, std::int32_t d) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

std::size_t scale_index(const Geometry& geometry, std::int32_t padded_context, std::int32_t head,
                        std::int32_t position, std::int32_t group) {
    (void)geometry;
    return static_cast<std::size_t>(group) +
           static_cast<std::size_t>(kQuantGroups) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

std::size_t cache_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

std::size_t scale_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kQuantGroups) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

std::size_t paged_index(std::int32_t leading_extent, const Geometry& geometry,
                        std::int32_t physical_page, std::int32_t head, std::int32_t position,
                        std::int32_t leading) {
    return static_cast<std::size_t>(leading) +
           static_cast<std::size_t>(leading_extent) *
               (static_cast<std::size_t>(position % kPagedKVPageSize) +
                static_cast<std::size_t>(kPagedKVPageSize) *
                    (static_cast<std::size_t>(head) + static_cast<std::size_t>(geometry.kv_heads) *
                                                          static_cast<std::size_t>(physical_page)));
}

template <typename T>
std::vector<T> scatter_paged(const std::vector<T>& logical, std::int32_t leading_extent,
                             const Geometry& geometry, std::int32_t logical_capacity,
                             std::span<const std::int32_t> block_table,
                             std::int32_t physical_pages) {
    std::vector<T> physical(static_cast<std::size_t>(leading_extent) * kPagedKVPageSize *
                            static_cast<std::size_t>(geometry.kv_heads) *
                            static_cast<std::size_t>(physical_pages));
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t source = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                physical[paged_index(leading_extent, geometry, page, head, position, leading)] =
                    logical[source];
            }
        }
    }
    return physical;
}

template <typename T>
void scatter_paged_into(const std::vector<T>& logical, std::int32_t leading_extent,
                        const Geometry& geometry, std::int32_t logical_capacity,
                        std::span<const std::int32_t> block_table, std::vector<T>& physical) {
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t source = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                physical[paged_index(leading_extent, geometry, page, head, position, leading)] =
                    logical[source];
            }
        }
    }
}

template <typename T>
std::vector<T> gather_paged(std::span<const T> physical, std::int32_t leading_extent,
                            const Geometry& geometry, std::int32_t logical_capacity,
                            std::span<const std::int32_t> block_table) {
    std::vector<T> logical(static_cast<std::size_t>(leading_extent) * logical_capacity *
                           static_cast<std::size_t>(geometry.kv_heads));
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t target = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                logical[target] =
                    physical[paged_index(leading_extent, geometry, page, head, position, leading)];
            }
        }
    }
    return logical;
}

std::vector<float> make_bf16_values(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(count);
    fill_uniform(values, seed, lo, hi);
    round_to_bf16(values);
    return values;
}

std::vector<std::uint16_t> to_bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> bf16_bits_to_double(const std::vector<std::uint16_t>& bits) {
    std::vector<double> values(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        values[i] = static_cast<double>(bf16_to_f32(bits[i]));
    }
    return values;
}

std::uint16_t f32_to_f16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exp  = (bits >> 23) & 0xffu;
    std::uint32_t mantissa   = bits & 0x007fffffu;
    if (exp == 0xffu) {
        return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
    }

    const int half_exp = static_cast<int>(exp) - 127 + 15;
    if (half_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
    if (half_exp <= 0) {
        if (half_exp < -10) { return static_cast<std::uint16_t>(sign); }
        mantissa |= 0x00800000u;
        const int shift             = 14 - half_exp;
        std::uint32_t half_mantissa = mantissa >> shift;
        const std::uint32_t halfway = 1u << (shift - 1);
        const std::uint32_t tail    = mantissa & ((1u << shift) - 1u);
        if (tail > halfway || (tail == halfway && (half_mantissa & 1u) != 0u)) { ++half_mantissa; }
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }

    std::uint32_t half_mantissa = mantissa >> 13;
    const std::uint32_t tail    = mantissa & 0x1fffu;
    std::uint32_t rounded_exp   = static_cast<std::uint32_t>(half_exp);
    if (tail > 0x1000u || (tail == 0x1000u && (half_mantissa & 1u) != 0u)) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0;
            ++rounded_exp;
            if (rounded_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
        }
    }
    return static_cast<std::uint16_t>(sign | (rounded_exp << 10) | half_mantissa);
}

float f16_bits_to_f32(std::uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0;
    const int exp       = (bits >> 10) & 0x1f;
    const int mantissa  = bits & 0x03ff;
    float magnitude     = 0.0f;
    if (exp == 0) {
        magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exp == 31) {
        magnitude = mantissa == 0 ? std::numeric_limits<float>::infinity()
                                  : std::numeric_limits<float>::quiet_NaN();
    } else {
        magnitude = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, exp - 15);
    }
    return negative ? -magnitude : magnitude;
}

std::int32_t round_even_to_i32(float value) {
    const float lower_f  = std::floor(value);
    const float fraction = value - lower_f;
    std::int32_t lower   = static_cast<std::int32_t>(lower_f);
    if (fraction < 0.5f) return lower;
    if (fraction > 0.5f) return lower + 1;
    return (lower & 1) == 0 ? lower : lower + 1;
}

struct HostCache {
    Geometry geometry;
    DType dtype;
    std::int32_t max_context;
    std::int32_t logical_capacity;
    std::vector<std::uint16_t> k_bf16;
    std::vector<std::uint16_t> v_bf16;
    std::vector<std::int8_t> k_i8;
    std::vector<std::int8_t> v_i8;
    std::vector<std::uint16_t> k_scale;
    std::vector<std::uint16_t> v_scale;
};

void encode_group(const std::vector<float>& source, std::size_t source_base,
                  std::vector<std::int8_t>& codes, std::size_t code_base,
                  std::vector<std::uint16_t>& scales, std::size_t scale_offset) {
    float absmax = 0.0f;
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        absmax = std::max(absmax, std::abs(source[source_base + static_cast<std::size_t>(i)]));
    }

    const float unrounded_scale    = absmax / 127.0f;
    const std::uint16_t scale_bits = f32_to_f16_bits(unrounded_scale);
    const float stored_scale       = f16_bits_to_f32(scale_bits);
    const float inverse_scale      = stored_scale == 0.0f ? 0.0f : 1.0f / stored_scale;
    scales[scale_offset]           = scale_bits;
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        std::int32_t code = 0;
        if (stored_scale != 0.0f) {
            const float scaled = source[source_base + static_cast<std::size_t>(i)] * inverse_scale;
            code               = std::clamp(round_even_to_i32(scaled), -127, 127);
        }
        codes[code_base + static_cast<std::size_t>(i)] = static_cast<std::int8_t>(code);
    }
}

HostCache make_cache(const Geometry& geometry, DType dtype, std::int32_t max_context,
                     std::uint32_t seed) {
    const std::int32_t logical_capacity = align_up_page(max_context);
    const std::size_t elements          = cache_elements(geometry, logical_capacity);
    std::vector<float> logical_k        = make_bf16_values(elements, seed, -0.25f, 0.25f);
    std::vector<float> logical_v        = make_bf16_values(elements, seed + 1u, -1.0f, 1.0f);

    HostCache cache{geometry, dtype, max_context, logical_capacity};
    if (dtype == DType::BF16) {
        cache.k_bf16 = to_bf16_bits(logical_k);
        cache.v_bf16 = to_bf16_bits(logical_v);
        return cache;
    }

    cache.k_i8.assign(elements, 0);
    cache.v_i8.assign(elements, 0);
    const std::size_t scales = scale_elements(geometry, logical_capacity);
    cache.k_scale.assign(scales, 0);
    cache.v_scale.assign(scales, 0);
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                const std::int32_t d   = group * kQuantGroup;
                const std::size_t code = cache_index(geometry, logical_capacity, head, position, d);
                const std::size_t scale =
                    scale_index(geometry, logical_capacity, head, position, group);
                encode_group(logical_k, code, cache.k_i8, code, cache.k_scale, scale);
                encode_group(logical_v, code, cache.v_i8, code, cache.v_scale, scale);
            }
        }
    }
    return cache;
}

void append_cache(HostCache& cache, const std::vector<float>& k, const std::vector<float>& v,
                  const std::vector<std::int32_t>& positions) {
    const Geometry& geometry = cache.geometry;
    for (std::int32_t token = 0; token < static_cast<std::int32_t>(positions.size()); ++token) {
        const std::int32_t position = positions[static_cast<std::size_t>(token)];
        for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
            if (cache.dtype == DType::BF16) {
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const std::size_t source = kv_input_index(geometry, head, d, token);
                    const std::size_t target =
                        cache_index(geometry, cache.logical_capacity, head, position, d);
                    cache.k_bf16[target] = f32_to_bf16(k[source]);
                    cache.v_bf16[target] = f32_to_bf16(v[source]);
                }
                continue;
            }

            for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                const std::int32_t d     = group * kQuantGroup;
                const std::size_t source = kv_input_index(geometry, head, d, token);
                const std::size_t target =
                    cache_index(geometry, cache.logical_capacity, head, position, d);
                const std::size_t scale =
                    scale_index(geometry, cache.logical_capacity, head, position, group);
                encode_group(k, source, cache.k_i8, target, cache.k_scale, scale);
                encode_group(v, source, cache.v_i8, target, cache.v_scale, scale);
            }
        }
    }
}

double cache_value(const HostCache& cache, bool key, std::int32_t head, std::int32_t position,
                   std::int32_t d) {
    const std::size_t code = cache_index(cache.geometry, cache.logical_capacity, head, position, d);
    if (cache.dtype == DType::BF16) {
        return static_cast<double>(bf16_to_f32(key ? cache.k_bf16[code] : cache.v_bf16[code]));
    }

    const std::size_t scale =
        scale_index(cache.geometry, cache.logical_capacity, head, position, d / kQuantGroup);
    const auto& codes   = key ? cache.k_i8 : cache.v_i8;
    const auto& scales  = key ? cache.k_scale : cache.v_scale;
    const float decoded = static_cast<float>(codes[code]) * f16_bits_to_f32(scales[scale]);
    return static_cast<double>(decoded);
}

std::vector<double> ideal_attention(const std::vector<float>& q, const HostCache& cache,
                                    const std::vector<std::int32_t>& positions) {
    const Geometry& geometry  = cache.geometry;
    const std::int32_t tokens = static_cast<std::int32_t>(positions.size());
    std::vector<double> output(static_cast<std::size_t>(kHeadDim) *
                               static_cast<std::size_t>(geometry.q_heads) *
                               static_cast<std::size_t>(tokens));

    std::vector<double> scores(static_cast<std::size_t>(positions.back()) + 1);
    std::vector<double> probabilities(scores.size());
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::int32_t visible = positions[static_cast<std::size_t>(token)] + 1;
        for (std::int32_t q_head = 0; q_head < geometry.q_heads; ++q_head) {
            const std::int32_t kv_head = q_head / geometry.query_group();
            double max_score           = -std::numeric_limits<double>::infinity();
            for (std::int32_t position = 0; position < visible; ++position) {
                double dot = 0.0;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    dot += static_cast<double>(q[q_index(geometry, q_head, d, token)]) *
                           cache_value(cache, true, kv_head, position, d);
                }
                const double score = dot * static_cast<double>(kAttentionScale);
                scores[static_cast<std::size_t>(position)] = score;
                max_score                                  = std::max(max_score, score);
            }

            double sum = 0.0;
            for (std::int32_t position = 0; position < visible; ++position) {
                const double probability =
                    std::exp(scores[static_cast<std::size_t>(position)] - max_score);
                probabilities[static_cast<std::size_t>(position)] = probability;
                sum += probability;
            }
            for (std::int32_t position = 0; position < visible; ++position) {
                probabilities[static_cast<std::size_t>(position)] /= sum;
            }

            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                double value = 0.0;
                for (std::int32_t position = 0; position < visible; ++position) {
                    value += probabilities[static_cast<std::size_t>(position)] *
                             cache_value(cache, false, kv_head, position, d);
                }
                output[q_index(geometry, q_head, d, token)] = value;
            }
        }
    }
    return output;
}

template <typename T>
std::vector<T> copy_from_guarded(const GuardedDeviceBuffer& buffer, std::size_t count) {
    std::vector<T> values(count);
    buffer.copy_to_host(values.data(), values.size() * sizeof(T));
    return values;
}

class DeviceCache {
public:
    DeviceCache(const HostCache& cache, MappingPattern mapping)
        : geometry_(cache.geometry), dtype_(cache.dtype), max_context_(cache.max_context),
          logical_capacity_(cache.logical_capacity),
          logical_pages_(logical_capacity_ / kPagedKVPageSize),
          physical_pages_(physical_page_count(logical_pages_, mapping)),
          block_table_host_(make_block_table(logical_pages_, mapping)),
          code_elements_(static_cast<std::size_t>(kHeadDim) * kPagedKVPageSize *
                         geometry_.kv_heads * physical_pages_),
          scale_elements_(static_cast<std::size_t>(kQuantGroups) * kPagedKVPageSize *
                          geometry_.kv_heads * physical_pages_),
          k_(code_elements_ *
             (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::int8_t))),
          v_(code_elements_ *
             (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::int8_t))),
          k_scale_(dtype_ == DType::I8 ? scale_elements_ * sizeof(std::uint16_t) : 1),
          v_scale_(dtype_ == DType::I8 ? scale_elements_ * sizeof(std::uint16_t) : 1),
          block_table_(block_table_host_.size() * sizeof(std::int32_t)) {
        block_table_.copy_from_host(block_table_host_.data(),
                                    block_table_host_.size() * sizeof(std::int32_t));
        if (dtype_ == DType::BF16) {
            const auto k_physical =
                scatter_paged(cache.k_bf16, kHeadDim, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_bf16, kHeadDim, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::uint16_t));
            v_.copy_from_host(v_physical.data(), v_physical.size() * sizeof(std::uint16_t));
        } else {
            const auto k_physical =
                scatter_paged(cache.k_i8, kHeadDim, geometry_, logical_capacity_, block_table_host_,
                              physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_i8, kHeadDim, geometry_, logical_capacity_, block_table_host_,
                              physical_pages_);
            const auto ks_physical =
                scatter_paged(cache.k_scale, kQuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto vs_physical =
                scatter_paged(cache.v_scale, kQuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::int8_t));
            v_.copy_from_host(v_physical.data(), v_physical.size() * sizeof(std::int8_t));
            k_scale_.copy_from_host(ks_physical.data(), ks_physical.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(vs_physical.data(), vs_physical.size() * sizeof(std::uint16_t));
        }
    }

    PagedKVLayerView view() {
        PagedKVLayerView result;
        result.k_pages      = Tensor(k_.data(), dtype_,
                                     {kHeadDim, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.v_pages      = Tensor(v_.data(), dtype_,
                                     {kHeadDim, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.block_table  = Tensor(block_table_.data(), DType::I32, {logical_pages_});
        result.num_kv_heads = geometry_.kv_heads;
        result.head_dim     = kHeadDim;
        result.dtype        = dtype_;
        if (dtype_ == DType::I8) {
            result.k_scale_pages =
                Tensor(k_scale_.data(), DType::FP16,
                       {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.v_scale_pages =
                Tensor(v_scale_.data(), DType::FP16,
                       {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.quant_group = kQuantGroup;
        }
        return result;
    }

    PagedKVBatchLayerView batch_view() {
        const PagedKVLayerView direct = view();
        return {
            .k_pages       = direct.k_pages,
            .v_pages       = direct.v_pages,
            .k_scale_pages = direct.k_scale_pages,
            .v_scale_pages = direct.v_scale_pages,
            .block_tables  = direct.block_table.view({logical_pages_, 1}),
            .head_dim      = direct.head_dim,
            .num_kv_heads  = direct.num_kv_heads,
            .dtype         = direct.dtype,
            .quant_group   = direct.quant_group,
        };
    }

    HostCache snapshot() const {
        HostCache cache{geometry_, dtype_, max_context_, logical_capacity_};
        if (dtype_ == DType::BF16) {
            const auto k_physical = copy_from_guarded<std::uint16_t>(k_, code_elements_);
            const auto v_physical = copy_from_guarded<std::uint16_t>(v_, code_elements_);
            cache.k_bf16          = gather_paged<std::uint16_t>(k_physical, kHeadDim, geometry_,
                                                                logical_capacity_, block_table_host_);
            cache.v_bf16          = gather_paged<std::uint16_t>(v_physical, kHeadDim, geometry_,
                                                                logical_capacity_, block_table_host_);
        } else {
            const auto k_physical  = copy_from_guarded<std::int8_t>(k_, code_elements_);
            const auto v_physical  = copy_from_guarded<std::int8_t>(v_, code_elements_);
            const auto ks_physical = copy_from_guarded<std::uint16_t>(k_scale_, scale_elements_);
            const auto vs_physical = copy_from_guarded<std::uint16_t>(v_scale_, scale_elements_);
            cache.k_i8             = gather_paged<std::int8_t>(k_physical, kHeadDim, geometry_,
                                                               logical_capacity_, block_table_host_);
            cache.v_i8             = gather_paged<std::int8_t>(v_physical, kHeadDim, geometry_,
                                                               logical_capacity_, block_table_host_);
            cache.k_scale = gather_paged<std::uint16_t>(ks_physical, kQuantGroups, geometry_,
                                                        logical_capacity_, block_table_host_);
            cache.v_scale = gather_paged<std::uint16_t>(vs_physical, kQuantGroups, geometry_,
                                                        logical_capacity_, block_table_host_);
        }
        return cache;
    }

    int verify_guards(const std::string& label) const {
        int failures = 0;
        failures += k_.verify_guards((label + " cache-k").c_str());
        failures += v_.verify_guards((label + " cache-v").c_str());
        if (dtype_ == DType::I8) {
            failures += k_scale_.verify_guards((label + " cache-k-scale").c_str());
            failures += v_scale_.verify_guards((label + " cache-v-scale").c_str());
        }
        failures += block_table_.verify_guards((label + " block-table").c_str());
        failures +=
            verify_exact((label + " block-table unchanged").c_str(),
                         copy_from_guarded<std::int32_t>(block_table_, block_table_host_.size()),
                         block_table_host_);
        return failures;
    }

private:
    Geometry geometry_;
    DType dtype_;
    std::int32_t max_context_;
    std::int32_t logical_capacity_;
    std::int32_t logical_pages_;
    std::int32_t physical_pages_;
    std::vector<std::int32_t> block_table_host_;
    std::size_t code_elements_;
    std::size_t scale_elements_;
    GuardedDeviceBuffer k_;
    GuardedDeviceBuffer v_;
    GuardedDeviceBuffer k_scale_;
    GuardedDeviceBuffer v_scale_;
    GuardedDeviceBuffer block_table_;
};

class BatchDeviceCache {
public:
    BatchDeviceCache(std::span<const HostCache> rows, MappingPattern mapping)
        : geometry_(rows.front().geometry), dtype_(rows.front().dtype), rows_(rows.size()),
          logical_capacity_(rows.front().logical_capacity),
          logical_pages_(logical_capacity_ / kPagedKVPageSize),
          physical_pages_(mapping == MappingPattern::Fragmented
                              ? 2 * static_cast<std::int32_t>(rows_) * logical_pages_ + 1
                              : static_cast<std::int32_t>(rows_) * logical_pages_),
          block_tables_host_(rows_ * static_cast<std::size_t>(logical_pages_)),
          code_elements_(static_cast<std::size_t>(kHeadDim) * kPagedKVPageSize *
                         geometry_.kv_heads * physical_pages_),
          scale_elements_(static_cast<std::size_t>(kQuantGroups) * kPagedKVPageSize *
                          geometry_.kv_heads * physical_pages_),
          k_(code_elements_ *
             (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::int8_t))),
          v_(code_elements_ *
             (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::int8_t))),
          k_scale_(dtype_ == DType::I8 ? scale_elements_ * sizeof(std::uint16_t) : 1),
          v_scale_(dtype_ == DType::I8 ? scale_elements_ * sizeof(std::uint16_t) : 1),
          block_tables_(block_tables_host_.size() * sizeof(std::int32_t)) {
        for (std::size_t row = 0; row < rows_; ++row) {
            const HostCache& cache = rows[row];
            if (cache.geometry.q_heads != geometry_.q_heads ||
                cache.geometry.kv_heads != geometry_.kv_heads || cache.dtype != dtype_ ||
                cache.logical_capacity != logical_capacity_) {
                throw std::invalid_argument("batch cache rows must share one physical geometry");
            }
            for (std::int32_t logical = 0; logical < logical_pages_; ++logical) {
                const std::int32_t linear =
                    static_cast<std::int32_t>(row) * logical_pages_ + logical;
                block_tables_host_[row * static_cast<std::size_t>(logical_pages_) + logical] =
                    mapping == MappingPattern::Fragmented ? 2 * linear + 1 : linear;
            }
        }
        block_tables_.copy_from_host(block_tables_host_.data(),
                                     block_tables_host_.size() * sizeof(std::int32_t));
        upload_rows(rows);
    }

    PagedKVBatchLayerView view() {
        PagedKVBatchLayerView result;
        result.k_pages      = Tensor(k_.data(), dtype_,
                                     {kHeadDim, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.v_pages      = Tensor(v_.data(), dtype_,
                                     {kHeadDim, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.block_tables = Tensor(block_tables_.data(), DType::I32,
                                     {logical_pages_, static_cast<std::int32_t>(rows_)});
        result.num_kv_heads = geometry_.kv_heads;
        result.head_dim     = kHeadDim;
        result.dtype        = dtype_;
        if (dtype_ == DType::I8) {
            result.k_scale_pages =
                Tensor(k_scale_.data(), DType::FP16,
                       {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.v_scale_pages =
                Tensor(v_scale_.data(), DType::FP16,
                       {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.quant_group = kQuantGroup;
        }
        return result;
    }

    int verify(const std::string& label, std::span<const HostCache> expected) const {
        if (expected.size() != rows_) {
            std::cerr << label << ": expected cache row count mismatch\n";
            return 1;
        }
        int failures = 0;
        if (dtype_ == DType::BF16) {
            std::vector<std::uint16_t> expected_k(code_elements_, 0);
            std::vector<std::uint16_t> expected_v(code_elements_, 0);
            scatter_bf16_rows(expected, expected_k, expected_v);
            failures +=
                verify_exact((label + " cache-k").c_str(),
                             copy_from_guarded<std::uint16_t>(k_, code_elements_), expected_k);
            failures +=
                verify_exact((label + " cache-v").c_str(),
                             copy_from_guarded<std::uint16_t>(v_, code_elements_), expected_v);
        } else {
            std::vector<std::int8_t> expected_k(code_elements_, 0);
            std::vector<std::int8_t> expected_v(code_elements_, 0);
            std::vector<std::uint16_t> expected_ks(scale_elements_, 0);
            std::vector<std::uint16_t> expected_vs(scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(expected[row].k_i8, kHeadDim, geometry_, logical_capacity_,
                                   table, expected_k);
                scatter_paged_into(expected[row].v_i8, kHeadDim, geometry_, logical_capacity_,
                                   table, expected_v);
                scatter_paged_into(expected[row].k_scale, kQuantGroups, geometry_,
                                   logical_capacity_, table, expected_ks);
                scatter_paged_into(expected[row].v_scale, kQuantGroups, geometry_,
                                   logical_capacity_, table, expected_vs);
            }
            failures +=
                verify_exact((label + " cache-k-code").c_str(),
                             copy_from_guarded<std::int8_t>(k_, code_elements_), expected_k);
            failures +=
                verify_exact((label + " cache-v-code").c_str(),
                             copy_from_guarded<std::int8_t>(v_, code_elements_), expected_v);
            failures += verify_exact((label + " cache-k-scale").c_str(),
                                     copy_from_guarded<std::uint16_t>(k_scale_, scale_elements_),
                                     expected_ks);
            failures += verify_exact((label + " cache-v-scale").c_str(),
                                     copy_from_guarded<std::uint16_t>(v_scale_, scale_elements_),
                                     expected_vs);
        }
        failures +=
            verify_exact((label + " block tables unchanged").c_str(),
                         copy_from_guarded<std::int32_t>(block_tables_, block_tables_host_.size()),
                         block_tables_host_);
        failures += k_.verify_guards((label + " cache-k guard").c_str());
        failures += v_.verify_guards((label + " cache-v guard").c_str());
        if (dtype_ == DType::I8) {
            failures += k_scale_.verify_guards((label + " cache-k-scale guard").c_str());
            failures += v_scale_.verify_guards((label + " cache-v-scale guard").c_str());
        }
        failures += block_tables_.verify_guards((label + " block tables guard").c_str());
        return failures;
    }

private:
    [[nodiscard]] std::span<const std::int32_t> row_table(std::size_t row) const {
        return std::span<const std::int32_t>(block_tables_host_.data() +
                                                 row * static_cast<std::size_t>(logical_pages_),
                                             static_cast<std::size_t>(logical_pages_));
    }

    void scatter_bf16_rows(std::span<const HostCache> rows, std::vector<std::uint16_t>& k,
                           std::vector<std::uint16_t>& v) const {
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(rows[row].k_bf16, kHeadDim, geometry_, logical_capacity_, table, k);
            scatter_paged_into(rows[row].v_bf16, kHeadDim, geometry_, logical_capacity_, table, v);
        }
    }

    void upload_rows(std::span<const HostCache> rows) {
        if (dtype_ == DType::BF16) {
            std::vector<std::uint16_t> physical_k(code_elements_, 0);
            std::vector<std::uint16_t> physical_v(code_elements_, 0);
            scatter_bf16_rows(rows, physical_k, physical_v);
            k_.copy_from_host(physical_k.data(), physical_k.size() * sizeof(std::uint16_t));
            v_.copy_from_host(physical_v.data(), physical_v.size() * sizeof(std::uint16_t));
            return;
        }
        std::vector<std::int8_t> physical_k(code_elements_, 0);
        std::vector<std::int8_t> physical_v(code_elements_, 0);
        std::vector<std::uint16_t> physical_ks(scale_elements_, 0);
        std::vector<std::uint16_t> physical_vs(scale_elements_, 0);
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(rows[row].k_i8, kHeadDim, geometry_, logical_capacity_, table,
                               physical_k);
            scatter_paged_into(rows[row].v_i8, kHeadDim, geometry_, logical_capacity_, table,
                               physical_v);
            scatter_paged_into(rows[row].k_scale, kQuantGroups, geometry_, logical_capacity_, table,
                               physical_ks);
            scatter_paged_into(rows[row].v_scale, kQuantGroups, geometry_, logical_capacity_, table,
                               physical_vs);
        }
        k_.copy_from_host(physical_k.data(), physical_k.size() * sizeof(std::int8_t));
        v_.copy_from_host(physical_v.data(), physical_v.size() * sizeof(std::int8_t));
        k_scale_.copy_from_host(physical_ks.data(), physical_ks.size() * sizeof(std::uint16_t));
        v_scale_.copy_from_host(physical_vs.data(), physical_vs.size() * sizeof(std::uint16_t));
    }

    Geometry geometry_;
    DType dtype_;
    std::size_t rows_;
    std::int32_t logical_capacity_;
    std::int32_t logical_pages_;
    std::int32_t physical_pages_;
    std::vector<std::int32_t> block_tables_host_;
    std::size_t code_elements_;
    std::size_t scale_elements_;
    GuardedDeviceBuffer k_;
    GuardedDeviceBuffer v_;
    GuardedDeviceBuffer k_scale_;
    GuardedDeviceBuffer v_scale_;
    GuardedDeviceBuffer block_tables_;
};

int verify_cache(const std::string& label, const HostCache& got, const HostCache& expected) {
    int failures = 0;
    if (expected.dtype == DType::BF16) {
        failures += verify_exact((label + " cache-k").c_str(), got.k_bf16, expected.k_bf16);
        failures += verify_exact((label + " cache-v").c_str(), got.v_bf16, expected.v_bf16);
    } else {
        failures += verify_exact((label + " cache-k-code").c_str(), got.k_i8, expected.k_i8);
        failures += verify_exact((label + " cache-v-code").c_str(), got.v_i8, expected.v_i8);
        failures += verify_exact((label + " cache-k-scale").c_str(), got.k_scale, expected.k_scale);
        failures += verify_exact((label + " cache-v-scale").c_str(), got.v_scale, expected.v_scale);
    }
    return failures;
}

int verify_input(const std::string& label, const GuardedDeviceBuffer& device,
                 const std::vector<std::uint16_t>& expected) {
    int failures = verify_exact(
        label.c_str(), copy_from_guarded<std::uint16_t>(device, expected.size()), expected);
    failures += device.verify_guards((label + " guard").c_str());
    return failures;
}

int verify_positions(const std::string& label, const GuardedDeviceBuffer& device,
                     const std::vector<std::int32_t>& expected) {
    int failures = verify_exact(label.c_str(),
                                copy_from_guarded<std::int32_t>(device, expected.size()), expected);
    failures += device.verify_guards((label + " guard").c_str());
    return failures;
}

const char* cache_name(DType dtype) { return dtype == DType::BF16 ? "bf16" : "int8-g64"; }

ReductionCriterion attention_criterion(DType dtype) {
    return dtype == DType::BF16 ? kAttentionBf16Criterion : kAttentionInt8Criterion;
}

int verify_attention(const std::string& label, const std::vector<double>& actual,
                     const std::vector<double>& reference, const ReductionCriterion& criterion) {
    return verify_reduction(label.c_str(), actual, reference, criterion);
}

std::string case_label(const char* entry, const Geometry& geometry, DType dtype,
                       const AttentionCase& test_case, MappingPattern mapping) {
    return std::string(entry) + " " + geometry.name + " " + cache_name(dtype) +
           " mapping=" + mapping_name(mapping) + " T=" + std::to_string(test_case.tokens) +
           " keys=" + std::to_string(test_case.base + test_case.tokens) +
           " envelope_max=" + std::to_string(test_case.envelope_max);
}

void inject_codec_edges(const Geometry& geometry, std::int32_t tokens, std::vector<float>& k,
                        std::vector<float>& v) {
    if (tokens == 0) return;
    for (std::int32_t d = 0; d < kQuantGroup; ++d) {
        k[kv_input_index(geometry, 0, d, 0)]               = 0.0f;
        v[kv_input_index(geometry, 0, kQuantGroup + d, 0)] = 0.0f;
    }
    k[kv_input_index(geometry, geometry.kv_heads - 1, 0, tokens - 1)] = -1.0f;
    v[kv_input_index(geometry, geometry.kv_heads - 1, 0, tokens - 1)] = 1.0f;
}

int run_append_case(const Geometry& geometry, DType dtype, MappingPattern mapping,
                    std::uint32_t seed, std::int32_t tokens = 3, std::int32_t base = 63) {
    const std::int32_t max_context = base + tokens + 4;
    const std::size_t elements =
        static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(geometry.kv_heads) * tokens;
    std::vector<float> k = make_bf16_values(elements, seed, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(elements, seed + 1u, -1.0f, 1.0f);
    inject_codec_edges(geometry, tokens, k, v);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = base + token;
    }

    const HostCache initial = make_cache(geometry, dtype, max_context, seed + 10u);
    HostCache expected      = initial;
    append_cache(expected, k, v, positions);
    DeviceCache cache(initial, mapping);

    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dpositions(positions.size() * sizeof(std::int32_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dpositions.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tp(dpositions.data(), DType::I32, {tokens});

    ops::gqa_kv_append(tk, tv, tp, cache.view(), nullptr);
    cuda_synchronize();

    const std::string label = std::string("gqa_kv_append ") + geometry.name + " " +
                              cache_name(dtype) + " mapping=" + mapping_name(mapping);
    int failures = verify_cache(label, cache.snapshot(), expected);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dpositions, positions);
    failures += cache.verify_guards(label);
    return failures;
}

int run_a1_case(const Geometry& geometry, DType dtype, const AttentionCase& test_case,
                MappingPattern mapping) {
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    const std::size_t kv_elements = static_cast<std::size_t>(kHeadDim) *
                                    static_cast<std::size_t>(geometry.kv_heads) *
                                    static_cast<std::size_t>(test_case.tokens);
    std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    std::vector<float> k = make_bf16_values(kv_elements, test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(kv_elements, test_case.seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(geometry, test_case.tokens, k, v);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }

    const HostCache initial = make_cache(geometry, dtype, max_context, test_case.seed + 10u);
    HostCache expected      = initial;
    append_cache(expected, k, v, positions);
    const std::vector<double> reference = ideal_attention(q, expected, positions);
    DeviceCache cache(initial, mapping);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_row(sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    const std::int32_t table_row = 0;
    dtable_row.copy_from_host(&table_row, sizeof(table_row));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor ttable_row(dtable_row.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                             test_case.envelope_max};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, 1, test_case.tokens, test_case.tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::gqa_attention(tq, tk, tv, tp, Tensor{}, ttable_row, kAttentionScale, cache.batch_view(),
                       envelope, workspace, tout, nullptr);
    cuda_synchronize();

    const std::string label = case_label("gqa_attention", geometry, dtype, test_case, mapping);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(dtype));
    failures += verify_cache(label, cache.snapshot(), expected);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += verify_positions(label + " table row unchanged", dtable_row, {table_row});
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    return failures;
}

int run_a3_case(const Geometry& geometry, DType dtype, const AttentionCase& test_case,
                MappingPattern mapping) {
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }

    const HostCache cache_host = make_cache(geometry, dtype, max_context, test_case.seed + 10u);
    const std::vector<double> reference = ideal_attention(q, cache_host, positions);
    DeviceCache cache(cache_host, mapping);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                             test_case.envelope_max};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, 1, test_case.tokens, test_case.tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::gqa_attention_cached(tq, tp, kAttentionScale, cache.view(), envelope, workspace, tout,
                              nullptr);
    cuda_synchronize();

    const std::string label =
        case_label("gqa_attention_cached", geometry, dtype, test_case, mapping);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(dtype));
    failures += verify_cache(label + " cache unchanged", cache.snapshot(), cache_host);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    return failures;
}

struct BatchAttentionCase {
    std::int32_t width;
    std::vector<std::int32_t> contexts;
    std::vector<std::int32_t> valid_columns;
    std::vector<std::int32_t> table_rows;
    MappingPattern mapping;
    std::uint32_t seed;
};

std::vector<float> extract_request_columns(const std::vector<float>& source,
                                           std::size_t column_elements, std::int32_t width,
                                           std::int32_t request, std::int32_t valid) {
    const std::size_t begin = static_cast<std::size_t>(request) * width * column_elements;
    std::vector<float> result(static_cast<std::size_t>(valid) * column_elements);
    std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(begin), result.size(), result.begin());
    return result;
}

void insert_request_columns(const std::vector<double>& source, std::size_t column_elements,
                            std::int32_t width, std::int32_t request,
                            std::vector<double>& destination) {
    const std::size_t begin = static_cast<std::size_t>(request) * width * column_elements;
    std::copy(source.begin(), source.end(),
              destination.begin() + static_cast<std::ptrdiff_t>(begin));
}

int verify_invalid_columns_zero(const std::string& label, std::span<const std::uint16_t> output,
                                const Geometry& geometry, std::int32_t width,
                                std::span<const std::int32_t> valid_columns) {
    int failures                      = 0;
    const std::size_t column_elements = static_cast<std::size_t>(kHeadDim) * geometry.q_heads;
    for (std::size_t batch = 0; batch < valid_columns.size(); ++batch) {
        for (std::int32_t token = valid_columns[batch]; token < width; ++token) {
            const std::size_t begin =
                (batch * static_cast<std::size_t>(width) + token) * column_elements;
            for (std::size_t element = 0; element < column_elements; ++element) {
                if (output[begin + element] != 0) {
                    if (failures == 0) {
                        std::cerr << label << ": invalid output column is not BF16 zero at row "
                                  << batch << " column " << token << '\n';
                    }
                    ++failures;
                }
            }
        }
    }
    return failures;
}

int run_batch_case(const Geometry& geometry, DType dtype, const BatchAttentionCase& test_case) {
    const std::int32_t batch = static_cast<std::int32_t>(test_case.contexts.size());
    if (batch <= 0 || test_case.valid_columns.size() != static_cast<std::size_t>(batch) ||
        test_case.table_rows.size() != static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("invalid GQA batch test profile");
    }

    std::int32_t maximum_visible = 1;
    for (std::int32_t row = 0; row < batch; ++row) {
        maximum_visible =
            std::max(maximum_visible, test_case.contexts[static_cast<std::size_t>(row)] +
                                          test_case.valid_columns[static_cast<std::size_t>(row)]);
    }
    const std::int32_t max_context       = maximum_visible + 3;
    const std::size_t q_column_elements  = static_cast<std::size_t>(kHeadDim) * geometry.q_heads;
    const std::size_t kv_column_elements = static_cast<std::size_t>(kHeadDim) * geometry.kv_heads;
    const std::size_t columns            = static_cast<std::size_t>(test_case.width) * batch;
    std::vector<float> q =
        make_bf16_values(q_column_elements * columns, test_case.seed, -0.25f, 0.25f);
    std::vector<float> k =
        make_bf16_values(kv_column_elements * columns, test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v =
        make_bf16_values(kv_column_elements * columns, test_case.seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(geometry, static_cast<std::int32_t>(columns), k, v);

    std::vector<std::int32_t> positions(columns, 0);
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t valid = test_case.valid_columns[static_cast<std::size_t>(row)];
        for (std::int32_t token = 0; token < valid; ++token) {
            positions[static_cast<std::size_t>(row) * test_case.width + token] =
                test_case.contexts[static_cast<std::size_t>(row)] + token;
        }
        const std::int32_t padding_position =
            valid == 0 ? 0 : test_case.contexts[static_cast<std::size_t>(row)] + valid - 1;
        for (std::int32_t token = valid; token < test_case.width; ++token) {
            positions[static_cast<std::size_t>(row) * test_case.width + token] = padding_position;
        }
    }

    std::vector<HostCache> initial;
    initial.reserve(static_cast<std::size_t>(batch));
    for (std::int32_t row = 0; row < batch; ++row) {
        initial.push_back(
            make_cache(geometry, dtype, max_context, test_case.seed + 20u + 3u * row));
    }
    std::vector<HostCache> expected = initial;
    std::vector<double> reference(q_column_elements * columns, 0.0);
    for (std::int32_t request = 0; request < batch; ++request) {
        const std::int32_t valid = test_case.valid_columns[static_cast<std::size_t>(request)];
        if (valid == 0) { continue; }
        const std::int32_t table_row = test_case.table_rows[static_cast<std::size_t>(request)];
        std::vector<std::int32_t> row_positions(static_cast<std::size_t>(valid));
        std::copy_n(positions.begin() + static_cast<std::ptrdiff_t>(request * test_case.width),
                    valid, row_positions.begin());
        const std::vector<float> row_q =
            extract_request_columns(q, q_column_elements, test_case.width, request, valid);
        const std::vector<float> row_k =
            extract_request_columns(k, kv_column_elements, test_case.width, request, valid);
        const std::vector<float> row_v =
            extract_request_columns(v, kv_column_elements, test_case.width, request, valid);
        append_cache(expected[static_cast<std::size_t>(table_row)], row_k, row_v, row_positions);
        insert_request_columns(
            ideal_attention(row_q, expected[static_cast<std::size_t>(table_row)], row_positions),
            q_column_elements, test_case.width, request, reference);
    }

    BatchDeviceCache cache(initial, test_case.mapping);
    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dvalid(test_case.valid_columns.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_rows(test_case.table_rows.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    dvalid.copy_from_host(test_case.valid_columns.data(),
                          test_case.valid_columns.size() * sizeof(std::int32_t));
    dtable_rows.copy_from_host(test_case.table_rows.data(),
                               test_case.table_rows.size() * sizeof(std::int32_t));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.width, batch});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.width, batch});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.width, batch});
    Tensor tp(dp.data(), DType::I32, {test_case.width, batch});
    Tensor tvalid(dvalid.data(), DType::I32, {batch});
    Tensor ttable_rows(dtable_rows.data(), DType::I32, {batch});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.width, batch});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(maximum_visible),
                                             static_cast<std::uint32_t>(maximum_visible)};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, batch, test_case.width, test_case.width);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    const bool masked = std::any_of(test_case.valid_columns.begin(), test_case.valid_columns.end(),
                                    [&](std::int32_t valid) { return valid != test_case.width; });
    ops::gqa_attention(tq, tk, tv, tp, masked ? tvalid : Tensor{}, ttable_rows, kAttentionScale,
                       cache.view(), envelope, workspace, tout, nullptr);
    cuda_synchronize();

    const std::string label = std::string("gqa_attention batch ") + geometry.name + " " +
                              cache_name(dtype) + " mapping=" + mapping_name(test_case.mapping) +
                              " B=" + std::to_string(batch) +
                              " W=" + std::to_string(test_case.width);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(dtype));
    failures += verify_invalid_columns_zero(label, output_bits, geometry, test_case.width,
                                            test_case.valid_columns);
    failures += cache.verify(label, expected);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    if (masked) {
        failures +=
            verify_positions(label + " valid columns unchanged", dvalid, test_case.valid_columns);
    }
    failures +=
        verify_positions(label + " table rows unchanged", dtable_rows, test_case.table_rows);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_batch_cases() {
    int failures = 0;
    failures += run_batch_case(kGeometries[0], DType::I8,
                               {6, {127}, {3}, {0}, MappingPattern::Identity, 499u});
    failures += run_batch_case(kGeometries[0], DType::BF16,
                               {16, {49}, {7}, {0}, MappingPattern::Identity, 500u});
    failures += run_batch_case(kGeometries[0], DType::BF16,
                               {1, {63, 2048}, {1, 1}, {1, 0}, MappingPattern::Fragmented, 501u});
    failures += run_batch_case(kGeometries[1], DType::I8,
                               {1,
                                {0, 31, 63, 127, 511, 1023, 2047, 4095},
                                {1, 1, 1, 1, 1, 1, 1, 1},
                                {7, 0, 5, 2, 6, 1, 4, 3},
                                MappingPattern::Identity,
                                502u});
    failures +=
        run_batch_case(kGeometries[0], DType::I8,
                       {6, {61, 127, 511}, {6, 3, 0}, {2, 0, 1}, MappingPattern::Fragmented, 503u});
    failures += run_batch_case(kGeometries[1], DType::BF16,
                               {16, {49, 2041}, {16, 7}, {1, 0}, MappingPattern::Identity, 504u});
    return failures;
}

int run_geometry(const Geometry& geometry) {
    int failures = 0;
    for (const DType dtype : {DType::BF16, DType::I8}) {
        for (const MappingPattern mapping :
             {MappingPattern::Identity, MappingPattern::Offset, MappingPattern::Fragmented}) {
            failures += run_append_case(geometry, dtype, mapping, 100u + geometry.q_heads);
            failures += run_a1_case(geometry, dtype, {6, 61, 67, 190u}, mapping);
            failures += run_a3_case(geometry, dtype, {1, 128, 129, 191u}, mapping);
        }
        if (dtype == DType::I8) {
            failures += run_append_case(geometry, dtype, MappingPattern::Fragmented,
                                        150u + geometry.q_heads, 129, 61);
        }

        const AttentionCase a1_cases[] = {
            {1, 0, 1, 201u},    {6, 17, 23, 202u},   {7, 17, 512, 203u},
            {17, 31, 48, 204u}, {66, 63, 129, 205u},
        };
        for (const AttentionCase& test_case : a1_cases) {
            failures += run_a1_case(geometry, dtype, test_case, MappingPattern::Identity);
        }

        const AttentionCase a3_cases[] = {
            {1, 31, 32, 301u},
            {7, 17, 512, 302u},
            {17, 31, 48, 303u},
        };
        for (const AttentionCase& test_case : a3_cases) {
            failures += run_a3_case(geometry, dtype, test_case, MappingPattern::Identity);
        }

        if (geometry.q_heads == 16) {
            // Loose execution envelopes straddle the two registered host-resource frontiers.
            // Device positions, not these bounds, continue to define the oracle result.
            failures += run_a1_case(geometry, dtype, {7, 17, 513, 401u}, MappingPattern::Identity);
            failures += run_a3_case(geometry, dtype, {7, 17, 513, 402u}, MappingPattern::Identity);
            failures +=
                run_a3_case(geometry, dtype, {16, 17, 1024, 403u}, MappingPattern::Identity);
            failures +=
                run_a3_case(geometry, dtype, {16, 17, 1025, 404u}, MappingPattern::Identity);
        }
    }
    return failures;
}

int verify_workspace_capacity_contract() {
    int failures = 0;
    for (const DType dtype : {DType::BF16, DType::I8}) {
        constexpr ops::GqaExecutionEnvelope envelope{1, 1025};
        const std::size_t interval =
            ops::gqa_attention_workspace_capacity_bytes(16, dtype, envelope, 1, 1, 17);
        std::size_t witness = 0;
        for (std::int32_t tokens = 1; tokens <= 17; ++tokens) {
            witness = std::max(witness, ops::gqa_attention_workspace_capacity_bytes(
                                            16, dtype, envelope, 1, tokens, tokens));
        }
        if (interval != witness) {
            std::cerr << "gqa_attention interval capacity has no exact route witness\n";
            ++failures;
        }
    }
    try {
        (void)ops::gqa_attention_workspace_capacity_bytes(
            16, DType::BF16, {1, ops::kGqaAttentionMaximumVisibleKeys}, 1, 1, 1);
    } catch (const std::invalid_argument&) {
        std::cerr << "gqa_attention rejected its maximum visible-key envelope\n";
        ++failures;
    }
    try {
        (void)ops::gqa_attention_workspace_capacity_bytes(
            16, DType::BF16, {1, ops::kGqaAttentionMaximumVisibleKeys + 1}, 1, 1, 1);
        std::cerr << "gqa_attention accepted an envelope outside the launcher domain\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    return failures;
}

// A4 shared-prefix segmented attention; only the 27B group-6 geometry is registered. Every case
// populates [0, prefix) through A2 from representative BF16 K/V and leaves every other cache row
// NaN, so reading any key the Op must not see poisons the output. The attention is checked
// against the independent FP64 oracle below with the named criterion of the cache dtype.

struct KvCodec {
    const char* name;
    DType dtype;
    bool packed_k;
    bool packed_v;
    bool rotated;
    bool e8_lattice;
    bool e8_root;

    [[nodiscard]] bool quantized() const { return dtype == DType::I8; }

    [[nodiscard]] std::int32_t k_row_bytes() const {
        if (!quantized()) return kHeadDim * 2;
        return e8_root ? kHeadDim / 4 : (packed_k ? kHeadDim / 2 : kHeadDim);
    }

    [[nodiscard]] std::int32_t v_row_bytes() const {
        if (!quantized()) return kHeadDim * 2;
        return packed_v ? kHeadDim / 2 : kHeadDim;
    }
};

constexpr KvCodec kBf16Kv{"bf16", DType::BF16, false, false, false, false, false};
constexpr KvCodec kInt8Kv{"int8-g64", DType::I8, false, false, false, false, false};
constexpr KvCodec kRk8v4Kv{"rk8v4", DType::I8, false, true, true, false, false};
constexpr KvCodec kRk4v4Kv{"rk4v4", DType::I8, true, true, true, false, false};
constexpr KvCodec kRk4v4E8Kv{"rk4v4-e8", DType::I8, true, true, true, true, false};
constexpr KvCodec kRk2v4E8Kv{"rk2v4-e8", DType::I8, false, true, true, false, true};

constexpr Geometry kSegmentedGeometry   = kGeometries[0];
constexpr std::int32_t kSegmentedGroup  = 6;
constexpr std::int32_t kScaleRowBytes   = kQuantGroups * 2;
constexpr std::int32_t kSegmentedRow    = 1;
constexpr std::int32_t kPrefixFillChunk = 8192;

struct CacheBytes {
    std::vector<std::uint8_t> k;
    std::vector<std::uint8_t> v;
    std::vector<std::uint8_t> k_scale;
    std::vector<std::uint8_t> v_scale;
};

// Paged planes of one codec with a two-row table matrix: row 0 is a decoy over the even physical
// pages and row 1, the row under test, maps logical page p to physical page 2p + 1.
class SegmentedCache {
public:
    SegmentedCache(const KvCodec& codec, std::int32_t logical_pages)
        : codec_(codec), logical_pages_(logical_pages), physical_pages_(2 * logical_pages + 1),
          k_(plane_bytes(codec.k_row_bytes())), v_(plane_bytes(codec.v_row_bytes())),
          k_scale_(codec.quantized() ? plane_bytes(kScaleRowBytes) : 1),
          v_scale_(codec.quantized() ? plane_bytes(kScaleRowBytes) : 1),
          tables_(static_cast<std::size_t>(2 * logical_pages) * sizeof(std::int32_t)) {
        tables_host_.resize(static_cast<std::size_t>(2 * logical_pages));
        for (std::int32_t page = 0; page < logical_pages; ++page) {
            tables_host_[static_cast<std::size_t>(page)]                 = 2 * page;
            tables_host_[static_cast<std::size_t>(logical_pages + page)] = 2 * page + 1;
        }
        tables_.copy_from_host(tables_host_.data(), tables_host_.size() * sizeof(std::int32_t));
        // All-ones bytes are NaN BF16 values and NaN FP16 scales.
        k_.fill(0xff);
        v_.fill(0xff);
        k_scale_.fill(0xff);
        v_scale_.fill(0xff);
    }

    [[nodiscard]] const KvCodec& codec() const { return codec_; }

    [[nodiscard]] std::size_t row_offset(std::int32_t row_bytes, std::int32_t position,
                                         std::int32_t head) const {
        const std::int32_t page = tables_host_[static_cast<std::size_t>(
            logical_pages_ * kSegmentedRow + position / kPagedKVPageSize)];
        return static_cast<std::size_t>(row_bytes) *
               (static_cast<std::size_t>(position % kPagedKVPageSize) +
                static_cast<std::size_t>(kPagedKVPageSize) *
                    (static_cast<std::size_t>(head) +
                     static_cast<std::size_t>(kSegmentedGeometry.kv_heads) * page));
    }

    PagedKVBatchLayerView batch_view() {
        PagedKVBatchLayerView view;
        bind_planes(view);
        view.block_tables = Tensor(tables_.data(), DType::I32, {logical_pages_, 2});
        return view;
    }

    PagedKVLayerView row_view() {
        PagedKVLayerView view;
        bind_planes(view);
        view.block_table = Tensor(static_cast<std::int32_t*>(tables_.data()) +
                                      static_cast<std::ptrdiff_t>(logical_pages_) * kSegmentedRow,
                                  DType::I32, {logical_pages_});
        return view;
    }

    [[nodiscard]] CacheBytes snapshot() const {
        CacheBytes bytes;
        bytes.k = copy_from_guarded<std::uint8_t>(k_, k_.bytes());
        bytes.v = copy_from_guarded<std::uint8_t>(v_, v_.bytes());
        if (codec_.quantized()) {
            bytes.k_scale = copy_from_guarded<std::uint8_t>(k_scale_, k_scale_.bytes());
            bytes.v_scale = copy_from_guarded<std::uint8_t>(v_scale_, v_scale_.bytes());
        }
        return bytes;
    }

    int verify_guards(const std::string& label) const {
        int failures = k_.verify_guards(label + " cache-k");
        failures += v_.verify_guards(label + " cache-v");
        failures += k_scale_.verify_guards(label + " cache-k-scale");
        failures += v_scale_.verify_guards(label + " cache-v-scale");
        failures += tables_.verify_guards(label + " block-tables");
        failures += verify_exact((label + " block tables unchanged").c_str(),
                                 copy_from_guarded<std::int32_t>(tables_, tables_host_.size()),
                                 tables_host_);
        return failures;
    }

private:
    std::size_t plane_bytes(std::int32_t row_bytes) const {
        return static_cast<std::size_t>(row_bytes) * kPagedKVPageSize *
               static_cast<std::size_t>(kSegmentedGeometry.kv_heads) *
               static_cast<std::size_t>(physical_pages_);
    }

    template <typename View>
    void bind_planes(View& view) {
        const DType k_dtype = !codec_.quantized() ? DType::BF16
                              : (codec_.packed_k || codec_.e8_root) ? DType::U8
                                                                     : DType::I8;
        const DType v_dtype = !codec_.quantized() ? DType::BF16
                              : codec_.packed_v   ? DType::U8
                                                  : DType::I8;
        const std::int32_t k_elements = codec_.quantized() ? codec_.k_row_bytes() : kHeadDim;
        const std::int32_t v_elements = codec_.quantized() ? codec_.v_row_bytes() : kHeadDim;
        const std::int32_t heads      = kSegmentedGeometry.kv_heads;
        view.k_pages = Tensor(k_.data(), k_dtype, {k_elements, kPagedKVPageSize, heads, physical_pages_});
        view.v_pages = Tensor(v_.data(), v_dtype, {v_elements, kPagedKVPageSize, heads, physical_pages_});
        if (codec_.quantized()) {
            view.k_scale_pages = Tensor(k_scale_.data(), DType::FP16,
                                        {kQuantGroups, kPagedKVPageSize, heads, physical_pages_});
            view.v_scale_pages = Tensor(v_scale_.data(), DType::FP16,
                                        {kQuantGroups, kPagedKVPageSize, heads, physical_pages_});
            view.quant_group = kQuantGroup;
        }
        view.head_dim     = kHeadDim;
        view.num_kv_heads = heads;
        view.dtype        = codec_.dtype;
        view.packed_v     = codec_.packed_v;
        view.rotate_k     = codec_.rotated;
        view.rotate_v     = codec_.rotated;
        view.packed_k     = codec_.packed_k;
        view.e8_lattice   = codec_.e8_lattice;
        view.e8_root      = codec_.e8_root;
    }

    KvCodec codec_;
    std::int32_t logical_pages_;
    std::int32_t physical_pages_;
    std::vector<std::int32_t> tables_host_;
    GuardedDeviceBuffer k_;
    GuardedDeviceBuffer v_;
    GuardedDeviceBuffer k_scale_;
    GuardedDeviceBuffer v_scale_;
    GuardedDeviceBuffer tables_;
};

template <typename Body>
void run_threads(Body&& body) {
    const unsigned count = std::max(1U, std::thread::hardware_concurrency());
    std::vector<std::thread> workers;
    workers.reserve(count);
    for (unsigned thread = 0; thread < count; ++thread) {
        workers.emplace_back([&body, thread, count] { body(thread, count); });
    }
    for (std::thread& worker : workers) { worker.join(); }
}

// E8 root codec value of one (root, radius/axis) code pair. Roots are the 240 E8 roots scaled by
// four: codes 0..111 put +-4 on the lexicographic dimension pair code / 4, bit 1 giving the first
// sign and bit 0 the second; codes 112..239 put +-2 on every dimension, bits 0..6 of code - 112
// giving the signs of dimensions 0..6 and dimension 7 completing an even number of negatives;
// codes 240..255 are zero. Axis a adds +1 (a even) or -1 (a odd) to dimension a / 2. The sum is
// scaled by the log-radius multiplier and rounded half-even; radius index 0 is the zero vector.
constexpr float kE8RadiusScale[16] = {0.0000f, 0.0992f, 0.1250f, 0.1575f, 0.1984f, 0.2500f,
                                      0.3150f, 0.3969f, 0.5000f, 0.6300f, 0.7937f, 1.0000f,
                                      1.2599f, 1.5874f, 2.0000f, 2.5198f};

void decode_e8_root(std::uint8_t root, std::uint8_t radius_axis, float* out) {
    const std::int32_t radius = radius_axis >> 4;
    std::int32_t vector[8]    = {};
    if (root < 112) {
        const std::int32_t pair = root / 4;
        std::int32_t index      = 0;
        for (std::int32_t first = 0; first < 8; ++first) {
            for (std::int32_t second = first + 1; second < 8; ++second, ++index) {
                if (index != pair) continue;
                vector[first]  = (root & 2) != 0 ? 4 : -4;
                vector[second] = (root & 1) != 0 ? 4 : -4;
            }
        }
    } else if (root < 240) {
        const std::int32_t signs = root - 112;
        std::int32_t negatives   = 0;
        for (std::int32_t d = 0; d < 7; ++d) {
            const bool positive = ((signs >> d) & 1) != 0;
            vector[d]           = positive ? 2 : -2;
            negatives += positive ? 0 : 1;
        }
        vector[7] = negatives % 2 == 0 ? 2 : -2;
    }
    const std::int32_t axis = radius_axis & 15;
    vector[axis / 2] += (axis & 1) != 0 ? -1 : 1;
    for (std::int32_t d = 0; d < 8; ++d) {
        out[d] = radius == 0 ? 0.0f
                             : static_cast<float>(round_even_to_i32(
                                   static_cast<float>(vector[d]) * kE8RadiusScale[radius]));
    }
}

// Stored-domain value of one cache row: BF16 values, or each decoded code times its FP16 group
// scale. Rotated codecs store the Hadamard rotation of the logical row.
void decode_cache_row(const SegmentedCache& cache, const CacheBytes& bytes, bool key,
                      std::int32_t position, std::int32_t head, float* out) {
    const KvCodec& codec        = cache.codec();
    const std::int32_t row_size = key ? codec.k_row_bytes() : codec.v_row_bytes();
    const std::uint8_t* row = (key ? bytes.k : bytes.v).data() + cache.row_offset(row_size, position, head);
    if (!codec.quantized()) {
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, row + 2 * d, sizeof(bits));
            out[d] = bf16_to_f32(bits);
        }
        return;
    }
    const std::uint8_t* scale_row = (key ? bytes.k_scale : bytes.v_scale).data() +
                                    cache.row_offset(kScaleRowBytes, position, head);
    float scales[kQuantGroups];
    for (std::int32_t group = 0; group < kQuantGroups; ++group) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, scale_row + 2 * group, sizeof(bits));
        scales[group] = f16_bits_to_f32(bits);
    }
    if (key && codec.e8_root) {
        for (std::int32_t block = 0; block < kHeadDim / 8; ++block) {
            float decoded[8];
            decode_e8_root(row[2 * block], row[2 * block + 1], decoded);
            for (std::int32_t i = 0; i < 8; ++i) {
                out[8 * block + i] = decoded[i] * scales[(8 * block) / kQuantGroup];
            }
        }
    } else if (key ? codec.packed_k : codec.packed_v) {
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            const std::int32_t nibble = (row[d / 2] >> (4 * (d % 2))) & 15;
            out[d] = static_cast<float>(nibble >= 8 ? nibble - 16 : nibble) * scales[d / kQuantGroup];
        }
    } else {
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            out[d] = static_cast<float>(static_cast<std::int8_t>(row[d])) * scales[d / kQuantGroup];
        }
    }
}

// Exact natural-order Sylvester-Hadamard transform of each 64-element group scaled by 1/8: the
// orthonormal, self-inverse rotation that the rotated codecs apply before quantization.
void hadamard64_groups(double* values) {
    for (std::int32_t group = 0; group < kQuantGroups; ++group) {
        double* x = values + group * kQuantGroup;
        for (std::int32_t half = 1; half < kQuantGroup; half <<= 1) {
            for (std::int32_t base = 0; base < kQuantGroup; base += 2 * half) {
                for (std::int32_t i = base; i < base + half; ++i) {
                    const double a = x[i];
                    const double b = x[i + half];
                    x[i]           = a + b;
                    x[i + half]    = a - b;
                }
            }
        }
        for (std::int32_t i = 0; i < kQuantGroup; ++i) { x[i] *= 0.125; }
    }
}

double dot_fp64(const double* query, const float* key) {
    double s0 = 0.0;
    double s1 = 0.0;
    double s2 = 0.0;
    double s3 = 0.0;
    for (std::int32_t d = 0; d < kHeadDim; d += 4) {
        s0 += query[d] * static_cast<double>(key[d]);
        s1 += query[d + 1] * static_cast<double>(key[d + 1]);
        s2 += query[d + 2] * static_cast<double>(key[d + 2]);
        s3 += query[d + 3] * static_cast<double>(key[d + 3]);
    }
    return (s0 + s1) + (s2 + s3);
}

// FP64 oracle of the A4 formula over the cache state after the append. A rotated row's logical
// value is H * (stored value); because H is orthonormal and symmetric, the oracle rotates q and the
// reduced value instead of every row, which evaluates the same formula.
std::vector<double> segmented_attention_oracle(const SegmentedCache& cache, const CacheBytes& state,
                                               const std::vector<float>& q, std::int32_t prefix,
                                               const std::vector<std::int32_t>& starts) {
    const Geometry& geometry   = kSegmentedGeometry;
    const bool rotated         = cache.codec().rotated;
    const std::int32_t columns = static_cast<std::int32_t>(starts.size());
    const std::int32_t keys    = prefix + columns;
    std::vector<double> output(static_cast<std::size_t>(kHeadDim) * geometry.q_heads * columns);
    std::vector<float> k_rows(static_cast<std::size_t>(keys) * kHeadDim);
    std::vector<float> v_rows(static_cast<std::size_t>(keys) * kHeadDim);
    for (std::int32_t kv_head = 0; kv_head < geometry.kv_heads; ++kv_head) {
        run_threads([&](unsigned thread, unsigned threads) {
            for (std::int32_t key = static_cast<std::int32_t>(thread); key < keys;
                 key += static_cast<std::int32_t>(threads)) {
                const std::size_t row = static_cast<std::size_t>(key) * kHeadDim;
                decode_cache_row(cache, state, true, key, kv_head, &k_rows[row]);
                decode_cache_row(cache, state, false, key, kv_head, &v_rows[row]);
            }
        });
        run_threads([&](unsigned thread, unsigned threads) {
            std::vector<double> scores(static_cast<std::size_t>(keys));
            double query[kHeadDim];
            double value[kHeadDim];
            for (std::int32_t unit = static_cast<std::int32_t>(thread);
                 unit < columns * kSegmentedGroup; unit += static_cast<std::int32_t>(threads)) {
                const std::int32_t column = unit / kSegmentedGroup;
                const std::int32_t q_head = kv_head * kSegmentedGroup + unit % kSegmentedGroup;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    query[d] = static_cast<double>(q[q_index(geometry, q_head, d, column)]);
                    value[d] = 0.0;
                }
                if (rotated) { hadamard64_groups(query); }
                const std::int32_t lo = prefix + starts[static_cast<std::size_t>(column)];
                const std::int32_t hi = prefix + column;
                const auto visit      = [&](auto&& body) {
                    for (std::int32_t key = 0; key < prefix; ++key) { body(key); }
                    for (std::int32_t key = lo; key <= hi; ++key) { body(key); }
                };
                double maximum = -std::numeric_limits<double>::infinity();
                visit([&](std::int32_t key) {
                    const double score = dot_fp64(query, &k_rows[static_cast<std::size_t>(key) * kHeadDim]) *
                                         static_cast<double>(kAttentionScale);
                    scores[static_cast<std::size_t>(key)] = score;
                    maximum                               = std::max(maximum, score);
                });
                double sum = 0.0;
                visit([&](std::int32_t key) {
                    const double weight = std::exp(scores[static_cast<std::size_t>(key)] - maximum);
                    const float* row    = &v_rows[static_cast<std::size_t>(key) * kHeadDim];
                    sum += weight;
                    for (std::int32_t d = 0; d < kHeadDim; ++d) {
                        value[d] += weight * static_cast<double>(row[d]);
                    }
                });
                for (std::int32_t d = 0; d < kHeadDim; ++d) { value[d] /= sum; }
                if (rotated) { hadamard64_groups(value); }
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    output[q_index(geometry, q_head, d, column)] = value[d];
                }
            }
        });
    }
    return output;
}

// Writes [0, prefix) of table row 1 through A2 from representative BF16 K/V.
void populate_prefix(SegmentedCache& cache, std::int32_t prefix, std::uint32_t seed) {
    const Geometry& geometry = kSegmentedGeometry;
    for (std::int32_t begin = 0; begin < prefix; begin += kPrefixFillChunk) {
        const std::int32_t tokens = std::min(kPrefixFillChunk, prefix - begin);
        const std::size_t elements =
            static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * static_cast<std::size_t>(tokens);
        const std::uint32_t chunk_seed = seed + 2u * static_cast<std::uint32_t>(begin / kPrefixFillChunk);
        const std::vector<std::uint16_t> k = to_bf16_bits(make_bf16_values(elements, chunk_seed, -0.25f, 0.25f));
        const std::vector<std::uint16_t> v = to_bf16_bits(make_bf16_values(elements, chunk_seed + 1u, -1.0f, 1.0f));
        std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) {
            positions[static_cast<std::size_t>(token)] = begin + token;
        }
        const DeviceBuffer dk = to_device(k);
        const DeviceBuffer dv = to_device(v);
        const DeviceBuffer dp = to_device(positions);
        const Tensor tk(dk.p, DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
        const Tensor tv(dv.p, DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
        const Tensor tp(dp.p, DType::I32, {tokens});
        ops::gqa_kv_append(tk, tv, tp, cache.row_view(), nullptr);
        cuda_synchronize();
    }
}

// Copies the rows of [begin, end) from `source` into `target` for every plane.
void copy_rows(const SegmentedCache& cache, const CacheBytes& source, std::int32_t begin,
               std::int32_t end, CacheBytes& target) {
    const KvCodec& codec = cache.codec();
    const auto copy      = [&](const std::vector<std::uint8_t>& from, std::vector<std::uint8_t>& to,
                          std::int32_t row_bytes, std::int32_t position, std::int32_t head) {
        const std::size_t offset = cache.row_offset(row_bytes, position, head);
        std::memcpy(to.data() + offset, from.data() + offset, static_cast<std::size_t>(row_bytes));
    };
    for (std::int32_t position = begin; position < end; ++position) {
        for (std::int32_t head = 0; head < kSegmentedGeometry.kv_heads; ++head) {
            copy(source.k, target.k, codec.k_row_bytes(), position, head);
            copy(source.v, target.v, codec.v_row_bytes(), position, head);
            if (codec.quantized()) {
                copy(source.k_scale, target.k_scale, kScaleRowBytes, position, head);
                copy(source.v_scale, target.v_scale, kScaleRowBytes, position, head);
            }
        }
    }
}

int verify_cache_bytes(const std::string& label, const CacheBytes& got, const CacheBytes& expected) {
    int failures = verify_exact((label + " cache-k").c_str(), got.k, expected.k);
    failures += verify_exact((label + " cache-v").c_str(), got.v, expected.v);
    failures += verify_exact((label + " cache-k-scale").c_str(), got.k_scale, expected.k_scale);
    failures += verify_exact((label + " cache-v-scale").c_str(), got.v_scale, expected.v_scale);
    return failures;
}

// Two implementations that each meet a reduction criterion against the same oracle differ by at
// most the sum of their envelopes.
ReductionCriterion pairwise_criterion(const ReductionCriterion& criterion) {
    return {2.0 * criterion.relative_l2, 2.0 * criterion.gross_absolute,
            2.0 * criterion.gross_relative_to_max_reference};
}

struct SegmentedCase {
    const KvCodec* codec;
    std::int32_t prefix;
    const char* mix;
    std::vector<std::int32_t> lengths;
    bool per_segment_a1;
    std::uint32_t seed;
};

// The supplementary consistency check: each segment runs A1 alone behind the same prefix, which
// appends it at [prefix, prefix + T_s). It rewrites the cache, so it runs last.
int verify_per_segment_a1(const std::string& label, const SegmentedCase& test_case,
                          SegmentedCache& cache, const GuardedDeviceBuffer& dq,
                          const GuardedDeviceBuffer& dk, const GuardedDeviceBuffer& dv,
                          GuardedDeviceBuffer& dtable_row, const std::vector<double>& segmented) {
    const Geometry& geometry = kSegmentedGeometry;
    const std::int32_t columns =
        static_cast<std::int32_t>(segmented.size() / (static_cast<std::size_t>(kHeadDim) * geometry.q_heads));
    GuardedDeviceBuffer dout(static_cast<std::size_t>(kHeadDim) * geometry.q_heads * columns * 2);
    std::int32_t begin = 0;
    for (const std::int32_t length : test_case.lengths) {
        std::vector<std::int32_t> positions(static_cast<std::size_t>(length));
        for (std::int32_t token = 0; token < length; ++token) {
            positions[static_cast<std::size_t>(token)] = test_case.prefix + token;
        }
        const DeviceBuffer dp = to_device(positions);
        const auto q_offset   = static_cast<std::ptrdiff_t>(kHeadDim) * geometry.q_heads * begin;
        const auto kv_offset  = static_cast<std::ptrdiff_t>(kHeadDim) * geometry.kv_heads * begin;
        const Tensor tq(const_cast<std::uint16_t*>(static_cast<const std::uint16_t*>(dq.data())) + q_offset,
                        DType::BF16, {kHeadDim, geometry.q_heads, length});
        const Tensor tk(const_cast<std::uint16_t*>(static_cast<const std::uint16_t*>(dk.data())) + kv_offset,
                        DType::BF16, {kHeadDim, geometry.kv_heads, length});
        const Tensor tv(const_cast<std::uint16_t*>(static_cast<const std::uint16_t*>(dv.data())) + kv_offset,
                        DType::BF16, {kHeadDim, geometry.kv_heads, length});
        const Tensor tp(dp.p, DType::I32, {length});
        const Tensor ttable_row(dtable_row.data(), DType::I32, {1});
        Tensor tout(static_cast<std::uint16_t*>(dout.data()) + q_offset, DType::BF16,
                    {kHeadDim, geometry.q_heads, length});
        const auto visible = static_cast<std::uint32_t>(test_case.prefix + length);
        const ops::GqaExecutionEnvelope envelope{visible, visible};
        const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
            geometry.q_heads, test_case.codec->dtype, envelope, 1, length, length);
        DeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
        WorkspaceArena workspace(DeviceSpan{workspace_buffer.p, workspace_buffer.bytes});
        ops::gqa_attention(tq, tk, tv, tp, Tensor{}, ttable_row, kAttentionScale, cache.batch_view(),
                           envelope, workspace, tout, nullptr);
        cuda_synchronize();
        begin += length;
    }
    const std::vector<double> per_segment = bf16_bits_to_double(
        copy_from_guarded<std::uint16_t>(dout, static_cast<std::size_t>(kHeadDim) * geometry.q_heads * columns));
    return verify_reduction(label + " vs per-segment A1", segmented, per_segment,
                            pairwise_criterion(attention_criterion(test_case.codec->dtype)));
}

int run_segmented_case(const SegmentedCase& test_case) {
    const Geometry& geometry = kSegmentedGeometry;
    const KvCodec& codec     = *test_case.codec;
    const std::int32_t prefix   = test_case.prefix;
    const std::int32_t segments = static_cast<std::int32_t>(test_case.lengths.size());
    std::vector<std::int32_t> table(static_cast<std::size_t>(2 * segments));
    std::vector<std::int32_t> starts;
    for (std::int32_t segment = 0; segment < segments; ++segment) {
        const std::int32_t start  = static_cast<std::int32_t>(starts.size());
        const std::int32_t length = test_case.lengths[static_cast<std::size_t>(segment)];
        table[static_cast<std::size_t>(2 * segment)]     = start;
        table[static_cast<std::size_t>(2 * segment + 1)] = length;
        starts.insert(starts.end(), static_cast<std::size_t>(length), start);
    }
    const std::int32_t columns = static_cast<std::int32_t>(starts.size());
    const std::string label = std::string("gqa_attention_segmented ") + geometry.name + " " +
                              codec.name + " prefix=" + std::to_string(prefix) + " mix=" +
                              test_case.mix;

    // One logical page past the call stays unwritten, so reads beyond prefix + N meet NaN rows.
    SegmentedCache cache(codec, (prefix + columns) / kPagedKVPageSize + 2);
    populate_prefix(cache, prefix, test_case.seed + 100u);

    const std::size_t q_elements =
        static_cast<std::size_t>(kHeadDim) * geometry.q_heads * static_cast<std::size_t>(columns);
    const std::size_t kv_elements =
        static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * static_cast<std::size_t>(columns);
    const std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits =
        to_bf16_bits(make_bf16_values(kv_elements, test_case.seed + 1u, -0.25f, 0.25f));
    const std::vector<std::uint16_t> v_bits =
        to_bf16_bits(make_bf16_values(kv_elements, test_case.seed + 2u, -1.0f, 1.0f));
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dsegments(table.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_row(sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dsegments.copy_from_host(table.data(), table.size() * sizeof(std::int32_t));
    dtable_row.copy_from_host(&kSegmentedRow, sizeof(kSegmentedRow));
    const std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    const Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, columns});
    const Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, columns});
    const Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, columns});
    const Tensor tsegments(dsegments.data(), DType::I32, {2, segments});
    const Tensor ttable_row(dtable_row.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, columns});
    const std::size_t workspace_bytes = ops::gqa_attention_segmented_workspace_capacity_bytes(columns);
    GuardedDeviceBuffer workspace_buffer(workspace_bytes);
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    const auto run = [&] {
        ops::gqa_attention_segmented(tq, tk, tv, tsegments, ttable_row, prefix, kAttentionScale,
                                     cache.batch_view(), workspace, tout, nullptr);
        cuda_synchronize();
        return copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    };
    const CacheBytes before              = cache.snapshot();
    const std::vector<std::uint16_t> out = run();
    const CacheBytes after               = cache.snapshot();

    const std::vector<double> output = bf16_bits_to_double(out);
    int failures = verify_attention(label, output,
                                    segmented_attention_oracle(cache, after, q, prefix, starts),
                                    attention_criterion(codec.dtype));
    // Only the appended rows of the selected table row may change.
    CacheBytes expected = before;
    copy_rows(cache, after, prefix, prefix + columns, expected);
    failures += verify_cache_bytes(label + " rows outside [prefix, prefix + N)", after, expected);

    failures += verify_exact((label + " repeated output").c_str(), run(), out);
    failures += verify_cache_bytes(label + " repeated append", cache.snapshot(), after);

    // The appended rows are exactly the A2 encoding of k/v at prefix + i.
    {
        std::vector<std::int32_t> positions(static_cast<std::size_t>(columns));
        for (std::int32_t column = 0; column < columns; ++column) {
            positions[static_cast<std::size_t>(column)] = prefix + column;
        }
        const DeviceBuffer dp = to_device(positions);
        ops::gqa_kv_append(tk, tv, Tensor(dp.p, DType::I32, {columns}), cache.row_view(), nullptr);
        cuda_synchronize();
        failures += verify_cache_bytes(label + " A2 encoding", cache.snapshot(), after);
    }

    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " segments unchanged", dsegments, table);
    failures += verify_positions(label + " table row unchanged", dtable_row, {kSegmentedRow});
    failures += dout.verify_guards(label + " output");
    failures += workspace_buffer.verify_guards(label + " workspace");
    if (workspace.used() != 0 || workspace.peak_used() > workspace_bytes) {
        std::cerr << label << ": workspace exceeded its capacity query\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    if (test_case.per_segment_a1) {
        failures += verify_per_segment_a1(label, test_case, cache, dq, dk, dv, dtable_row, output);
    }
    return failures;
}

int run_segmented_cases() {
    const std::vector<std::int32_t> single{1};
    const std::vector<std::int32_t> six_by_40(6, 40);
    const std::vector<std::int32_t> wide{1024};
    const std::vector<std::int32_t> many(255, 3);
    // Every codec, prefix, and segment mix appears; the long-prefix and wide cases are spread over
    // codecs instead of forming the full product. Prefixes straddle the page (63/64) and split
    // (1000, 8192, 65535) boundaries; prefix 0 has no shared history.
    const SegmentedCase cases[] = {
        {&kBf16Kv, 1, "1x1024", wide, false, 601u},
        {&kBf16Kv, 1000, "6x40", six_by_40, true, 602u},
        {&kInt8Kv, 0, "6x40", six_by_40, false, 603u},
        {&kInt8Kv, 63, "255x3", many, true, 604u},
        {&kInt8Kv, 8192, "1x1", single, false, 605u},
        {&kRk8v4Kv, 64, "1x1024", wide, false, 606u},
        {&kRk8v4Kv, 65535, "1x1", single, false, 607u},
        {&kRk4v4Kv, 1000, "255x3", many, false, 608u},
        {&kRk4v4Kv, 63, "6x40", six_by_40, false, 609u},
        {&kRk4v4E8Kv, 65535, "6x40", six_by_40, true, 610u},
        {&kRk4v4E8Kv, 63, "1x1024", wide, false, 611u},
        {&kRk4v4E8Kv, 1, "1x1", single, false, 612u},
        {&kRk2v4E8Kv, 8192, "6x40", six_by_40, false, 613u},
        {&kRk2v4E8Kv, 64, "255x3", many, false, 614u},
    };
    int failures = 0;
    for (const SegmentedCase& test_case : cases) { failures += run_segmented_case(test_case); }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += verify_workspace_capacity_contract();
    for (const Geometry& geometry : kGeometries) { failures += run_geometry(geometry); }
    failures += run_batch_cases();
    failures += run_segmented_cases();
    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " gqa_attention public-contract correctness\n";
    return failures == 0 ? 0 : 1;
}
