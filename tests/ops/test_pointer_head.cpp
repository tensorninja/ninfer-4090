#include "ninfer/ops/pointer_head.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHidden  = 5120; // D: registered 27B hidden width
constexpr std::int32_t kPointer = 256;  // P: pointer-head width
constexpr double kUnitRoundoff  = 0x1p-24;

// Acceptance criteria.
//
// Projection. A BF16 x BF16 product is exact in FP32 (8+8 significand bits), so the only error is
// FP32 accumulation of the D products and the bias. For zero-mean data the partial sums grow like
// sqrt(k), and the rounding error of any accumulation order scales as u*sqrt(D) relative to the
// result (Higham & Mary's probabilistic bound; plain sequential summation sits near
// 0.4*u*sqrt(D)). Every output column must satisfy ||y_hat - y||_2 <= 2*u*sqrt(D)*||y||_2
// (8.5e-6 at D=5120) with a gross pointwise cap of 8*u*sqrt(D)*max|y| per column. Rounding any
// stage to BF16 (2^-9 relative) would exceed both by more than two orders of magnitude.
//
// Probabilities. Perturbing every logit of a question by at most Delta changes each softmax
// probability by at most p_i*(exp(2*Delta) - 1). FP32 evaluation of the softmax itself adds at
// most (|z_i - max z| + K + 8)*u relative error: argument rounding, expf's 2 ulp, the K-term
// positive sum, and the division. The logit bound Delta is
//   score alone, from represented FP32 queries and keys: the FP32 dot of P products and the
//     scale multiply, s*4*sqrt(P)*u*sum_p|k_p*q_p| + u*|z| (probabilistic, lambda=4);
//   projections then score, from the BF16 inputs: the projection criterion bounds each projected
//     vector's relative error by eps = 2*u*sqrt(D), so by Cauchy-Schwarz the dot moves by at most
//     (2*eps + eps^2)*||k||*||q||, plus the same FP32 score term.
// FLT_MIN absorbs denormal exponentials of negligible probabilities.
double projection_relative_l2_limit(std::int32_t hidden) {
    return 2.0 * std::sqrt(static_cast<double>(hidden)) * kUnitRoundoff;
}

double projection_gross_limit(std::int32_t hidden) {
    return 8.0 * std::sqrt(static_cast<double>(hidden)) * kUnitRoundoff;
}

double score_dot_factor(std::int32_t pointer) {
    return 4.0 * std::sqrt(static_cast<double>(pointer)) * kUnitRoundoff;
}

constexpr std::uint32_t kSentinelBits = 0x5a5a5a5aU;

struct HostHead {
    std::int32_t hidden  = kHidden;
    std::int32_t pointer = kPointer;
    std::vector<std::uint16_t> weight; // [D,P]: column p is row p of the projection matrix
    std::vector<std::uint16_t> bias;   // [P]
};

struct DeviceHead {
    explicit DeviceHead(const HostHead& head)
        : host(head), weight(to_device(head.weight)), bias(to_device(head.bias)) {}

    Tensor weight_tensor() const {
        return Tensor(weight.p, DType::BF16, {host.hidden, host.pointer});
    }

    Tensor bias_tensor() const { return Tensor(bias.p, DType::BF16, {host.pointer}); }

    int verify_preserved(const std::string& label) const {
        int failures =
            verify_exact((label + " preserves weight").c_str(),
                         from_device<std::uint16_t>(weight, host.weight.size()), host.weight);
        failures += verify_exact((label + " preserves bias").c_str(),
                                 from_device<std::uint16_t>(bias, host.bias.size()), host.bias);
        return failures;
    }

    HostHead host;
    DeviceBuffer weight;
    DeviceBuffer bias;
};

HostHead make_head(std::uint32_t seed, std::int32_t hidden = kHidden,
                   std::int32_t pointer = kPointer) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> normal(0.0F, 0.02F);
    HostHead head;
    head.hidden  = hidden;
    head.pointer = pointer;
    head.weight.resize(static_cast<std::size_t>(hidden) * pointer);
    head.bias.resize(static_cast<std::size_t>(pointer));
    for (auto& value : head.weight) { value = f32_to_bf16(normal(generator)); }
    for (auto& value : head.bias) { value = f32_to_bf16(normal(generator)); }
    return head;
}

// Final-norm output columns: N(0,1) times a per-channel gain in [0.5,2), with a few large-gain
// outlier channels.
std::vector<std::uint16_t> make_hidden(std::int32_t columns, std::uint32_t seed,
                                       std::int32_t width = kHidden) {
    std::mt19937 gain_generator(7U);
    std::uniform_real_distribution<float> uniform(0.5F, 2.0F);
    std::vector<float> gain(static_cast<std::size_t>(width));
    for (auto& value : gain) { value = uniform(gain_generator); }
    for (std::int32_t channel = 311; channel < width; channel += 641) { gain[channel] = 9.0F; }

    std::mt19937 generator(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(width) * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        for (std::int32_t channel = 0; channel < width; ++channel) {
            hidden[static_cast<std::size_t>(column) * width + channel] =
                f32_to_bf16(gain[channel] * normal(generator));
        }
    }
    return hidden;
}

// y[p,r] = bias[p] + sum_d weight[d,p]*hidden[d,r], naive FP64 from the represented BF16 values.
std::vector<double> project_oracle(const HostHead& head, const std::vector<std::uint16_t>& hidden,
                                   std::int32_t columns) {
    const std::int32_t width   = head.hidden;
    const std::int32_t pointer = head.pointer;
    std::vector<double> weight(head.weight.size());
    for (std::size_t index = 0; index < weight.size(); ++index) {
        weight[index] = bf16_to_f32(head.weight[index]);
    }
    std::vector<double> values(static_cast<std::size_t>(width));
    std::vector<double> result(static_cast<std::size_t>(pointer) * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        for (std::int32_t channel = 0; channel < width; ++channel) {
            values[channel] =
                bf16_to_f32(hidden[static_cast<std::size_t>(column) * width + channel]);
        }
        for (std::int32_t row = 0; row < pointer; ++row) {
            const double* weight_row = weight.data() + static_cast<std::size_t>(row) * width;
            double sum               = bf16_to_f32(head.bias[row]);
            for (std::int32_t channel = 0; channel < width; ++channel) {
                sum += weight_row[channel] * values[channel];
            }
            result[static_cast<std::size_t>(column) * pointer + row] = sum;
        }
    }
    return result;
}

std::vector<float> filled(std::size_t count, std::uint32_t bits) {
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return std::vector<float>(count, value);
}

bool same_bits(float lhs, float rhs) { return std::memcmp(&lhs, &rhs, sizeof(float)) == 0; }

// got/reference are [P,R] column blocks of the head's geometry.
int verify_projection(const std::string& label, const HostHead& head, std::span<const float> got,
                      std::span<const double> reference, std::int32_t columns) {
    const double relative_l2_limit = projection_relative_l2_limit(head.hidden);
    const double gross_limit       = projection_gross_limit(head.hidden);
    double worst_relative_l2       = 0.0;
    double worst_gross_ratio       = 0.0;
    double worst_absolute          = 0.0;
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * head.pointer;
        double error_squared = 0.0, reference_squared = 0.0, maximum_reference = 0.0,
               maximum_error = 0.0;
        for (std::int32_t row = 0; row < head.pointer; ++row) {
            const double actual   = got[base + row];
            const double expected = reference[base + row];
            if (!std::isfinite(actual)) {
                std::cerr << label << ": non-finite output at column " << column << " row " << row
                          << '\n';
                return 1;
            }
            const double error = actual - expected;
            error_squared += error * error;
            reference_squared += expected * expected;
            maximum_reference = std::max(maximum_reference, std::abs(expected));
            maximum_error     = std::max(maximum_error, std::abs(error));
        }
        worst_relative_l2 =
            std::max(worst_relative_l2, std::sqrt(error_squared / reference_squared));
        worst_gross_ratio = std::max(worst_gross_ratio, maximum_error / maximum_reference);
        worst_absolute    = std::max(worst_absolute, maximum_error);
    }
    if (error_stats_enabled()) {
        std::printf("OP_ERROR_STATS kind=pointer_projection columns=%d max_column_rel_l2=%.3e "
                    "rel_l2_limit=%.3e max_gross_ratio=%.3e gross_limit=%.3e max_abs=%.3e "
                    "case=%s\n",
                    columns, worst_relative_l2, relative_l2_limit, worst_gross_ratio, gross_limit,
                    worst_absolute, label.c_str());
    }
    if (worst_relative_l2 <= relative_l2_limit && worst_gross_ratio <= gross_limit) { return 0; }
    std::cerr << label << ": projection criterion failed: column rel_l2=" << worst_relative_l2
              << " gross ratio=" << worst_gross_ratio << '\n';
    return 1;
}

struct Question {
    std::int32_t query;
    std::int32_t key_begin;
    std::int32_t key_count;
};

struct ScoreReference {
    std::vector<double> probability; // [Nk]; NaN where no question owns the key
    std::vector<double> limit;       // [Nk] admissible absolute error
};

// queries [P,Nq] and keys [P,Nk] are the represented operand values. vector_relative_error is
// the admitted per-vector relative error of projected operands, zero for exactly represented ones.
ScoreReference score_oracle(std::span<const double> queries, std::span<const double> keys,
                            std::int32_t pointer, std::int32_t key_columns,
                            std::span<const Question> questions, double scale,
                            double vector_relative_error) {
    ScoreReference result;
    result.probability.assign(static_cast<std::size_t>(key_columns),
                              std::numeric_limits<double>::quiet_NaN());
    result.limit.assign(static_cast<std::size_t>(key_columns), 0.0);
    const double eps = vector_relative_error;
    for (const Question& question : questions) {
        const double* query = queries.data() + static_cast<std::size_t>(question.query) * pointer;
        double query_norm   = 0.0;
        for (std::int32_t row = 0; row < pointer; ++row) { query_norm += query[row] * query[row]; }
        query_norm = std::sqrt(query_norm);

        std::vector<double> logits(static_cast<std::size_t>(question.key_count));
        double delta = 0.0;
        for (std::int32_t index = 0; index < question.key_count; ++index) {
            const double* key =
                keys.data() + static_cast<std::size_t>(question.key_begin + index) * pointer;
            double dot = 0.0, absolute = 0.0, key_norm = 0.0;
            for (std::int32_t row = 0; row < pointer; ++row) {
                dot += key[row] * query[row];
                absolute += std::abs(key[row] * query[row]);
                key_norm += key[row] * key[row];
            }
            key_norm      = std::sqrt(key_norm);
            logits[index] = scale * dot;
            const double logit_error =
                std::abs(scale) * (score_dot_factor(pointer) * absolute +
                                   (2.0 * eps + eps * eps) * key_norm * query_norm) +
                kUnitRoundoff * std::abs(logits[index]);
            delta = std::max(delta, logit_error);
        }
        const double maximum = *std::max_element(logits.begin(), logits.end());
        double sum           = 0.0;
        for (const double logit : logits) { sum += std::exp(logit - maximum); }
        for (std::int32_t index = 0; index < question.key_count; ++index) {
            const double probability = std::exp(logits[index] - maximum) / sum;
            const double evaluation =
                (std::abs(logits[index] - maximum) + question.key_count + 8.0) * kUnitRoundoff;
            const auto slot          = static_cast<std::size_t>(question.key_begin + index);
            result.probability[slot] = probability;
            result.limit[slot]       = probability * (std::expm1(2.0 * delta) + evaluation) +
                                 static_cast<double>(std::numeric_limits<float>::min());
        }
    }
    return result;
}

int verify_probabilities(const std::string& label, std::span<const float> got,
                         const ScoreReference& reference, std::span<const float> before) {
    double worst_absolute = 0.0;
    double worst_ratio    = 0.0;
    for (std::size_t key = 0; key < got.size(); ++key) {
        if (std::isnan(reference.probability[key])) {
            if (!same_bits(got[key], before[key])) {
                std::cerr << label << ": key " << key << " has no question but was written\n";
                return 1;
            }
            continue;
        }
        const double error = std::abs(static_cast<double>(got[key]) - reference.probability[key]);
        if (!std::isfinite(static_cast<double>(got[key])) || !(error <= reference.limit[key])) {
            std::cerr << label << ": probability mismatch at key " << key << " got=" << got[key]
                      << " reference=" << reference.probability[key]
                      << " limit=" << reference.limit[key] << '\n';
            return 1;
        }
        worst_absolute = std::max(worst_absolute, error);
        worst_ratio    = std::max(worst_ratio, error / reference.limit[key]);
    }
    if (error_stats_enabled()) {
        std::printf("OP_ERROR_STATS kind=pointer_probability max_abs=%.3e max_limit_ratio=%.3e "
                    "case=%s\n",
                    worst_absolute, worst_ratio, label.c_str());
    }
    return 0;
}

// Lays question key blocks out in a scrambled order with unowned gap columns between some blocks;
// question q scores against query column (q*5+3) % Qn.
struct Layout {
    std::vector<Question> questions;
    std::int32_t key_columns = 0;
};

Layout make_layout(std::span<const std::int32_t> key_counts, std::int32_t key_offset) {
    const auto count = static_cast<std::int32_t>(key_counts.size());
    std::vector<std::int32_t> order(key_counts.size());
    for (std::int32_t index = 0; index < count; ++index) { order[index] = index; }
    std::mt19937 generator(static_cast<std::uint32_t>(count) * 7919U);
    std::shuffle(order.begin(), order.end(), generator);
    if (count > 1 && order.front() == 0) { std::swap(order.front(), order.back()); }

    Layout layout;
    layout.questions.resize(key_counts.size());
    std::int32_t cursor = key_offset;
    for (std::int32_t position = 0; position < count; ++position) {
        const std::int32_t question = order[position];
        layout.questions[question]  = {(question * 5 + 3) % count, cursor, key_counts[question]};
        cursor += key_counts[question] + (position % 2);
    }
    layout.key_columns = cursor + 1;
    return layout;
}

DeviceBuffer questions_to_device(std::span<const Question> questions) {
    std::vector<std::int32_t> table;
    for (const Question& question : questions) {
        table.push_back(question.query);
        table.push_back(question.key_begin);
        table.push_back(question.key_count);
    }
    return to_device(table);
}

// Projection into columns [offset, offset+R) of a sentinel-filled [P,capacity] buffer.
int project_case(const DeviceHead& head, std::int32_t columns, std::int32_t offset,
                 std::int32_t capacity, std::uint32_t seed) {
    const std::int32_t width   = head.host.hidden;
    const std::int32_t pointer = head.host.pointer;
    const std::string label    = "pointer_head_project D=" + std::to_string(width) +
                              " P=" + std::to_string(pointer) + " R=" + std::to_string(columns) +
                              " slice=" + std::to_string(offset) + "/" + std::to_string(capacity);
    const auto hidden                = make_hidden(columns, seed, width);
    const std::vector<double> ref    = project_oracle(head.host, hidden, columns);
    const DeviceBuffer device_hidden = to_device(hidden);
    const auto elements              = static_cast<std::size_t>(pointer) * capacity;
    const auto initial               = filled(elements, kSentinelBits);
    GuardedDeviceBuffer buffer(elements * sizeof(float));
    buffer.copy_from_host(initial.data(), elements * sizeof(float));

    const Tensor hidden_tensor(device_hidden.p, DType::BF16, {width, columns});
    Tensor out = Tensor(buffer.data(), DType::FP32, {pointer, capacity}).slice(1, offset, columns);
    ops::pointer_head_project(hidden_tensor, head.weight_tensor(), head.bias_tensor(), out,
                              nullptr);
    cuda_synchronize();
    const auto first = from_device<float>(buffer.data(), elements);
    ops::pointer_head_project(hidden_tensor, head.weight_tensor(), head.bias_tensor(), out,
                              nullptr);
    cuda_synchronize();
    const auto second = from_device<float>(buffer.data(), elements);

    const auto begin = static_cast<std::size_t>(offset) * pointer;
    const auto end   = begin + static_cast<std::size_t>(columns) * pointer;
    int failures     = verify_projection(
        label, head.host, std::span<const float>(first).subspan(begin, end - begin), ref, columns);
    for (std::size_t index = 0; index < elements; ++index) {
        if ((index < begin || index >= end) && !same_bits(first[index], initial[index])) {
            std::cerr << label << ": column outside the slice was written\n";
            ++failures;
            break;
        }
    }
    if (std::memcmp(first.data(), second.data(), elements * sizeof(float)) != 0) {
        std::cerr << label << ": repeated call is not bit-identical\n";
        ++failures;
    }
    failures += verify_exact((label + " preserves hidden").c_str(),
                             from_device<std::uint16_t>(device_hidden, hidden.size()), hidden);
    failures += buffer.verify_guards(label + " out");
    return failures;
}

// Scoring alone from represented FP32 queries and keys.
int score_case(const std::string& name, std::span<const std::int32_t> key_counts, float scale,
               std::uint32_t seed, std::int32_t pointer = kPointer) {
    const std::string label  = "pointer_head_score P=" + std::to_string(pointer) + " " + name;
    const Layout layout      = make_layout(key_counts, 0);
    const auto query_columns = static_cast<std::int32_t>(key_counts.size());
    std::mt19937 generator(seed);
    std::normal_distribution<float> normal(0.0F, 1.9F);
    std::vector<float> queries(static_cast<std::size_t>(pointer) * query_columns);
    std::vector<float> keys(static_cast<std::size_t>(pointer) * layout.key_columns);
    for (auto& value : queries) { value = normal(generator); }
    for (auto& value : keys) { value = normal(generator); }
    const std::vector<double> queries_fp64(queries.begin(), queries.end());
    const std::vector<double> keys_fp64(keys.begin(), keys.end());
    const ScoreReference reference = score_oracle(queries_fp64, keys_fp64, pointer,
                                                  layout.key_columns, layout.questions, scale, 0.0);

    const DeviceBuffer device_queries   = to_device(queries);
    const DeviceBuffer device_keys      = to_device(keys);
    const DeviceBuffer device_questions = questions_to_device(layout.questions);
    const auto initial = filled(static_cast<std::size_t>(layout.key_columns), kSentinelBits);
    GuardedDeviceBuffer probabilities(initial.size() * sizeof(float));
    probabilities.copy_from_host(initial.data(), initial.size() * sizeof(float));

    const Tensor queries_tensor(device_queries.p, DType::FP32, {pointer, query_columns});
    const Tensor keys_tensor(device_keys.p, DType::FP32, {pointer, layout.key_columns});
    const Tensor questions_tensor(device_questions.p, DType::I32, {3, query_columns});
    Tensor probabilities_tensor(probabilities.data(), DType::FP32, {layout.key_columns});
    ops::pointer_head_score(queries_tensor, keys_tensor, questions_tensor, scale,
                            probabilities_tensor, nullptr);
    cuda_synchronize();
    const auto first = from_device<float>(probabilities.data(), initial.size());
    ops::pointer_head_score(queries_tensor, keys_tensor, questions_tensor, scale,
                            probabilities_tensor, nullptr);
    cuda_synchronize();
    const auto second = from_device<float>(probabilities.data(), initial.size());

    int failures = verify_probabilities(label, first, reference, initial);
    if (std::memcmp(first.data(), second.data(), first.size() * sizeof(float)) != 0) {
        std::cerr << label << ": repeated call is not bit-identical\n";
        ++failures;
    }
    failures += verify_exact((label + " preserves keys").c_str(),
                             from_device<float>(device_keys, keys.size()), keys);
    failures += probabilities.verify_guards(label + " probabilities");
    return failures;
}

struct ChainOutputs {
    std::vector<float> queries;
    std::vector<float> keys;
    std::vector<float> probabilities;
};

// Both projections and the scoring from BF16 hidden states, keys written into a column range of
// a larger key buffer, eagerly and through a captured CUDA Graph.
int head_case(const std::string& name, const DeviceHead& query_head, const DeviceHead& key_head,
              std::span<const std::int32_t> key_counts, std::int32_t key_offset,
              std::uint32_t seed) {
    const std::string label   = "pointer_head " + name;
    const float scale         = 1.0F / (16.0F * 1.38F);
    const auto question_count = static_cast<std::int32_t>(key_counts.size());
    const Layout layout       = make_layout(key_counts, key_offset);
    std::int32_t key_count    = 0;
    for (const std::int32_t count : key_counts) { key_count += count; }
    // Option hidden column j feeds key column key_offset + j; gaps between question blocks stay
    // unowned, so the slice holds key_count + gaps columns.
    const std::int32_t slice_columns = layout.key_columns - 1 - key_offset;
    const std::int32_t capacity      = layout.key_columns + 2;

    const auto decide_hidden = make_hidden(question_count, seed);
    const auto option_hidden = make_hidden(slice_columns, seed + 1U);
    const auto query_ref     = project_oracle(query_head.host, decide_hidden, question_count);
    const auto key_ref       = project_oracle(key_head.host, option_hidden, slice_columns);
    std::vector<double> keys_fp64(static_cast<std::size_t>(kPointer) * capacity, 0.0);
    std::copy(key_ref.begin(), key_ref.end(),
              keys_fp64.begin() + static_cast<std::ptrdiff_t>(key_offset) * kPointer);
    const ScoreReference reference =
        score_oracle(query_ref, keys_fp64, kPointer, capacity, layout.questions, scale,
                     projection_relative_l2_limit(kHidden));

    const DeviceBuffer device_decide    = to_device(decide_hidden);
    const DeviceBuffer device_option    = to_device(option_hidden);
    const DeviceBuffer device_questions = questions_to_device(layout.questions);
    const auto query_elements           = static_cast<std::size_t>(kPointer) * question_count;
    const auto key_elements             = static_cast<std::size_t>(kPointer) * capacity;
    const auto key_initial              = filled(key_elements, kSentinelBits);
    const auto prob_initial             = filled(static_cast<std::size_t>(capacity), kSentinelBits);
    GuardedDeviceBuffer queries(query_elements * sizeof(float));
    GuardedDeviceBuffer keys(key_elements * sizeof(float));
    GuardedDeviceBuffer probabilities(prob_initial.size() * sizeof(float));

    const Tensor decide_tensor(device_decide.p, DType::BF16, {kHidden, question_count});
    const Tensor option_tensor(device_option.p, DType::BF16, {kHidden, slice_columns});
    const Tensor questions_tensor(device_questions.p, DType::I32, {3, question_count});
    Tensor queries_tensor(queries.data(), DType::FP32, {kPointer, question_count});
    Tensor keys_tensor(keys.data(), DType::FP32, {kPointer, capacity});
    Tensor key_slice = keys_tensor.slice(1, key_offset, slice_columns);
    Tensor probabilities_tensor(probabilities.data(), DType::FP32, {capacity});

    const auto reset = [&] {
        queries.fill(0xa5);
        keys.copy_from_host(key_initial.data(), key_elements * sizeof(float));
        probabilities.copy_from_host(prob_initial.data(), prob_initial.size() * sizeof(float));
    };
    const auto enqueue = [&](cudaStream_t stream) {
        ops::pointer_head_project(decide_tensor, query_head.weight_tensor(),
                                  query_head.bias_tensor(), queries_tensor, stream);
        ops::pointer_head_project(option_tensor, key_head.weight_tensor(), key_head.bias_tensor(),
                                  key_slice, stream);
        ops::pointer_head_score(queries_tensor, keys_tensor, questions_tensor, scale,
                                probabilities_tensor, stream);
    };
    const auto read = [&] {
        return ChainOutputs{from_device<float>(queries.data(), query_elements),
                            from_device<float>(keys.data(), key_elements),
                            from_device<float>(probabilities.data(), prob_initial.size())};
    };
    const auto identical = [](const ChainOutputs& lhs, const ChainOutputs& rhs) {
        return lhs.queries.size() == rhs.queries.size() && lhs.keys.size() == rhs.keys.size() &&
               lhs.probabilities.size() == rhs.probabilities.size() &&
               std::memcmp(lhs.queries.data(), rhs.queries.data(),
                           lhs.queries.size() * sizeof(float)) == 0 &&
               std::memcmp(lhs.keys.data(), rhs.keys.data(), lhs.keys.size() * sizeof(float)) ==
                   0 &&
               std::memcmp(lhs.probabilities.data(), rhs.probabilities.data(),
                           lhs.probabilities.size() * sizeof(float)) == 0;
    };

    reset();
    enqueue(nullptr);
    cuda_synchronize();
    const ChainOutputs eager = read();

    const auto key_begin = static_cast<std::size_t>(key_offset) * kPointer;
    const auto key_end   = key_begin + static_cast<std::size_t>(slice_columns) * kPointer;
    int failures = verify_projection(label + " queries", query_head.host, eager.queries, query_ref,
                                     question_count);
    failures += verify_projection(
        label + " keys", key_head.host,
        std::span<const float>(eager.keys).subspan(key_begin, key_end - key_begin), key_ref,
        slice_columns);
    for (std::size_t index = 0; index < key_elements; ++index) {
        if ((index < key_begin || index >= key_end) &&
            !same_bits(eager.keys[index], key_initial[index])) {
            std::cerr << label << ": key column outside the slice was written\n";
            ++failures;
            break;
        }
    }
    failures += verify_probabilities(label + " probabilities", eager.probabilities, reference,
                                     prob_initial);

    reset();
    enqueue(nullptr);
    cuda_synchronize();
    if (!identical(eager, read())) {
        std::cerr << label << ": repeated eager chain is not bit-identical\n";
        ++failures;
    }

    cudaStream_t stream  = nullptr;
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    reset();
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "cudaStreamBeginCapture");
    enqueue(stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    cuda_check(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate");
    for (int replay = 0; replay < 2; ++replay) {
        cuda_check(cudaGraphLaunch(exec, stream), "cudaGraphLaunch");
    }
    cuda_synchronize(stream);
    if (!identical(eager, read())) {
        std::cerr << label << ": CUDA Graph replay differs from eager execution\n";
        ++failures;
    }
    cuda_check(cudaGraphExecDestroy(exec), "cudaGraphExecDestroy");
    cuda_check(cudaGraphDestroy(graph), "cudaGraphDestroy");
    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");

    failures += query_head.verify_preserved(label + " query head");
    failures += key_head.verify_preserved(label + " key head");
    failures += queries.verify_guards(label + " queries");
    failures += keys.verify_guards(label + " keys");
    failures += probabilities.verify_guards(label + " probabilities");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    const DeviceHead head(make_head(101U));
    int failures = 0;
    // Column extents put both sides of every private route boundary (32, 96, 256, 416) and full
    // and partial column tiles of each route through the registered D=5120, P=256 geometry.
    for (const std::int32_t columns :
         {1, 2, 3, 15, 16, 17, 32, 33, 48, 49, 96, 97, 129, 256, 257, 416, 417, 600}) {
        failures +=
            project_case(head, columns, 0, columns, 1000U + static_cast<std::uint32_t>(columns));
    }
    failures += project_case(head, 3, 1, 5, 2001U);
    failures += project_case(head, 255, 7, 270, 2002U);
    // D=1056 splits its 33 chunks unevenly over every route's warps, and P=100 leaves a partial
    // weight-row tile in every route.
    const DeviceHead odd_head(make_head(102U, 1056, 100));
    for (const std::int32_t columns : {5, 40, 130, 300, 450}) {
        failures += project_case(odd_head, columns, 0, columns,
                                 1500U + static_cast<std::uint32_t>(columns));
    }

    const std::int32_t single[] = {1};
    const std::int32_t widest[] = {255};
    const std::int32_t six[]    = {3, 1, 255, 2, 3, 1};
    std::vector<std::int32_t> sixty_four(64);
    for (std::size_t question = 0; question < sixty_four.size(); ++question) {
        sixty_four[question] = question == 41 ? 255 : static_cast<std::int32_t>(question % 3) + 1;
    }
    const float calibrated = 1.0F / (16.0F * 1.38F);
    // Key tables of at least 32 columns per question (K=255, Q=6) take the wide scoring CTA; the
    // others take the narrow one, which also meets the K=255 question among Q=64.
    failures += score_case("Q=1 K=1", single, calibrated, 3001U);
    failures += score_case("Q=1 K=255", widest, calibrated, 3002U);
    failures += score_case("Q=6 mixed K", six, calibrated, 3003U);
    failures += score_case("Q=64 mixed K", sixty_four, calibrated, 3004U);
    // Logits of several hundred: exp overflows without the max subtraction.
    failures += score_case("Q=6 large logits", six, 8.0F, 3005U);
    // 25 float4 vectors per column leave most lanes of every dot product idle.
    failures += score_case("Q=6 mixed K", six, calibrated, 3006U, 100);
    failures += score_case("Q=64 mixed K", sixty_four, calibrated, 3007U, 100);

    const DeviceHead query_head(make_head(201U));
    const DeviceHead key_head(make_head(202U));
    const std::int32_t three[] = {3};
    failures += head_case("Q=1 K=3", query_head, key_head, three, 0, 4001U);
    failures += head_case("Q=6 mixed K key slice", query_head, key_head, six, 5, 4002U);
    failures += head_case("Q=64 mixed K key slice", query_head, key_head, sixty_four, 2, 4003U);

    std::cout << (failures ? "FAIL" : "OK") << " pointer_head_project and pointer_head_score\n";
    return failures ? 1 : 0;
}
