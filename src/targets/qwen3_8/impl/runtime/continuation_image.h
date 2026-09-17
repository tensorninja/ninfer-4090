#pragma once

#include "core/cyclic_kv_cache.h"
#include "artifact/sha256.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "core/pinned_transfer.h"
#include "runtime/cache/continuation_cache.h"
#include "targets/qwen3_8/impl/runtime/prefix_identity.h"
#include <ninfer/targets/qwen3_8/runtime.h>

#include <bit>
#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <span>
#include <type_traits>
#include <vector>

namespace ninfer::targets::qwen3_8::detail::continuation {

inline constexpr std::uint32_t kTargetImageVersion = 3;
class Writer {
public:
    void u8(std::uint8_t value) { bytes_.push_back(value); }

    void u32(std::uint32_t value) {
        for (unsigned shift = 0; shift != 32; shift += 8) {
            u8(static_cast<std::uint8_t>(value >> shift));
        }
    }

    void u64(std::uint64_t value) {
        for (unsigned shift = 0; shift != 64; shift += 8) {
            u8(static_cast<std::uint8_t>(value >> shift));
        }
    }

    void i32(std::int32_t value) { u32(std::bit_cast<std::uint32_t>(value)); }
    void f64(double value) { u64(std::bit_cast<std::uint64_t>(value)); }

    void raw(std::span<const std::uint8_t> bytes) {
        bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
    }

    void blob(std::span<const std::uint8_t> bytes) {
        u64(bytes.size());
        raw(bytes);
    }

    void string(std::string_view value) {
        blob(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(value.data()),
                                           value.size()));
    }

    [[nodiscard]] cache::Bytes finish() && { return std::move(bytes_); }

private:
    cache::Bytes bytes_;
};

class Reader {
public:
    explicit Reader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

    [[nodiscard]] std::uint8_t u8() {
        require(1);
        return bytes_[cursor_++];
    }

    [[nodiscard]] std::uint32_t u32() {
        std::uint32_t value = 0;
        for (unsigned shift = 0; shift != 32; shift += 8) {
            value |= static_cast<std::uint32_t>(u8()) << shift;
        }
        return value;
    }

    [[nodiscard]] std::uint64_t u64() {
        std::uint64_t value = 0;
        for (unsigned shift = 0; shift != 64; shift += 8) {
            value |= static_cast<std::uint64_t>(u8()) << shift;
        }
        return value;
    }

    [[nodiscard]] std::int32_t i32() { return std::bit_cast<std::int32_t>(u32()); }
    [[nodiscard]] double f64() { return std::bit_cast<double>(u64()); }

    [[nodiscard]] std::size_t count(
        std::size_t maximum, std::uint64_t element_bytes = 1) {
        const std::uint64_t value = u64();
        if (value > maximum ||
            (element_bytes != 0 && value > remaining() / element_bytes) ||
            value > std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument("continuation image container is out of bounds");
        }
        return static_cast<std::size_t>(value);
    }

    [[nodiscard]] cache::Bytes blob(std::size_t maximum) {
        const std::size_t size = count(maximum);
        require(size);
        cache::Bytes out(bytes_.begin() + static_cast<std::ptrdiff_t>(cursor_),
                         bytes_.begin() + static_cast<std::ptrdiff_t>(cursor_ + size));
        cursor_ += size;
        return out;
    }

    [[nodiscard]] cache::Bytes blob_exact(std::size_t expected) {
        const std::size_t size = count(expected);
        if (size != expected) {
            throw std::invalid_argument("continuation image blob has an incompatible extent");
        }
        require(size);
        cache::Bytes out(bytes_.begin() + static_cast<std::ptrdiff_t>(cursor_),
                         bytes_.begin() + static_cast<std::ptrdiff_t>(cursor_ + size));
        cursor_ += size;
        return out;
    }

    void finish() const {
        if (cursor_ != bytes_.size()) {
            throw std::invalid_argument("continuation image segment has trailing bytes");
        }
    }

private:
    [[nodiscard]] std::size_t remaining() const noexcept { return bytes_.size() - cursor_; }

    void require(std::size_t size) const {
        if (size > remaining()) {
            throw std::invalid_argument("continuation image segment is truncated");
        }
    }

    std::span<const std::uint8_t> bytes_;
    std::size_t cursor_ = 0;
};

inline void write_header(Writer& out, std::string_view kind) {
    out.string(kind);
    out.u32(kTargetImageVersion);
}

inline void read_header(Reader& in, std::string_view kind) {
    const cache::Bytes encoded = in.blob_exact(kind.size());
    if (encoded.size() != kind.size() ||
        !std::equal(encoded.begin(), encoded.end(), kind.begin()) ||
        in.u32() != kTargetImageVersion) {
        throw std::invalid_argument("continuation image segment has an incompatible header");
    }
}

template <class T>
void write_i32_vector(Writer& out, const std::vector<T>& values) {
    static_assert(sizeof(T) == sizeof(std::int32_t));
    out.u64(values.size());
    for (const T value : values) { out.i32(static_cast<std::int32_t>(value)); }
}

template <class T>
std::vector<T> read_i32_vector(Reader& in,
                               std::size_t maximum = std::numeric_limits<std::size_t>::max()) {
    static_assert(sizeof(T) == sizeof(std::int32_t));
    const std::size_t count = in.count(maximum, sizeof(std::int32_t));
    std::vector<T> out(count);
    for (T& value : out) { value = static_cast<T>(in.i32()); }
    return out;
}

inline cache::Bytes encode_prefix(const std::vector<TokenId>& ledger,
                                  const ResidentPrefixIdentitySnapshot& identity) {
    Writer out;
    write_header(out, "qwen-prefix");
    write_i32_vector(out, ledger);
    out.u64(identity.token_types.size());
    out.raw(identity.token_types);
    for (const auto& axis : identity.positions) { write_i32_vector(out, axis); }
    out.u64(identity.vision_items.size());
    for (const VisionItem& item : identity.vision_items) {
        out.u8(static_cast<std::uint8_t>(item.modality));
        out.i32(item.grid.temporal);
        out.i32(item.grid.height);
        out.i32(item.grid.width);
        out.u64(item.patch_begin);
        out.u64(item.patch_count);
        out.raw(item.content_digest);
        out.u64(item.timestamps.size());
        for (const double timestamp : item.timestamps) { out.f64(timestamp); }
        out.u64(item.token_spans.size());
        for (const TokenSpan& span : item.token_spans) {
            out.u64(span.begin);
            out.u64(span.count);
        }
    }
    return std::move(out).finish();
}

struct PrefixData {
    std::vector<TokenId> ledger;
    ResidentPrefixIdentitySnapshot identity;
};

inline PrefixData decode_prefix(
    std::span<const std::uint8_t> bytes,
    std::size_t maximum_tokens = std::numeric_limits<std::size_t>::max()) {
    Reader in(bytes);
    read_header(in, "qwen-prefix");
    PrefixData out;
    out.ledger = read_i32_vector<TokenId>(in, maximum_tokens);
    const std::size_t token_types = in.count(maximum_tokens);
    out.identity.token_types.resize(token_types);
    for (std::uint8_t& type : out.identity.token_types) { type = in.u8(); }
    for (auto& axis : out.identity.positions) {
        axis = read_i32_vector<std::int32_t>(in, maximum_tokens);
    }
    const std::size_t item_count = in.count(maximum_tokens);
    out.identity.vision_items.resize(item_count);
    for (VisionItem& item : out.identity.vision_items) {
        const std::uint8_t modality = in.u8();
        if (modality != static_cast<std::uint8_t>(PromptModality::Image) &&
            modality != static_cast<std::uint8_t>(PromptModality::Video)) {
            throw std::invalid_argument("continuation prefix has an invalid modality");
        }
        item.modality      = static_cast<PromptModality>(modality);
        item.grid.temporal = in.i32();
        item.grid.height   = in.i32();
        item.grid.width    = in.i32();
        item.patch_begin   = in.u64();
        item.patch_count   = in.u64();
        for (std::uint8_t& byte : item.content_digest) { byte = in.u8(); }
        item.timestamps.resize(in.count(maximum_tokens, sizeof(double)));
        for (double& timestamp : item.timestamps) { timestamp = in.f64(); }
        item.token_spans.resize(in.count(maximum_tokens, 2 * sizeof(std::uint64_t)));
        for (TokenSpan& span : item.token_spans) {
            span.begin = in.u64();
            span.count = in.u64();
        }
    }
    in.finish();
    if (out.identity.token_types.size() != out.ledger.size()) {
        throw std::invalid_argument("continuation prefix ledger and identity differ in size");
    }
    for (const auto& axis : out.identity.positions) {
        if (axis.size() != out.ledger.size()) {
            throw std::invalid_argument("continuation prefix positions have an invalid shape");
        }
    }
    return out;
}

inline cache::Bytes prefix_filter_digest(const std::vector<TokenId>& ledger,
                                         ResidentPrefixIdentitySnapshot identity,
                                         std::uint32_t depth) {
    if (depth == 0 || depth > ledger.size()) {
        throw std::invalid_argument("continuation prefix filter depth is out of bounds");
    }
    ResidentPrefixIdentity resident;
    resident.restore(std::move(identity));
    resident.truncate(depth);
    std::vector<TokenId> tokens(ledger.begin(),
                                ledger.begin() + static_cast<std::ptrdiff_t>(depth));
    const cache::Bytes exact = encode_prefix(tokens, resident.export_prefix(depth));
    artifact::Sha256 hash;
    hash.update(std::as_bytes(std::span(exact)));
    const artifact::Sha256Digest digest = hash.finish();
    return cache::Bytes(digest.begin(), digest.end());
}

inline cache::Bytes prefix_filter_digest(const PreparedPromptData& prompt, std::uint32_t depth) {
    if (depth == 0 || depth > prompt.token_ids.size()) {
        throw std::invalid_argument("prompt prefix filter depth is out of bounds");
    }
    ResidentPrefixIdentity identity;
    identity.assign(prompt);
    return prefix_filter_digest(prompt.token_ids, identity.export_prefix(prompt.token_ids.size()),
                                depth);
}

// Content-addressed alias of the exact prompt prefix through `depth`: the tokens, their
// types and MRoPE positions, under one compatibility key. Any prompt sharing that prefix
// derives the same name, which is what lets a boundary published by one conversation serve
// another. The canonical form is versioned; bumping it retires every alias derived before.
inline std::optional<std::string> boundary_alias(std::span<const std::uint8_t> compatibility_key,
                                                 const PreparedPromptData& prompt,
                                                 std::uint32_t depth) {
    if (!prompt.identity.reusable || depth == 0 || depth > prompt.token_ids.size() ||
        prompt.token_types.size() != prompt.token_ids.size() ||
        prompt.positions.size() != 3 * prompt.token_ids.size()) {
        return std::nullopt;
    }

    ResidentPrefixIdentity identity;
    identity.assign(prompt);
    identity.truncate(depth);
    std::vector<TokenId> tokens(prompt.token_ids.begin(),
                                prompt.token_ids.begin() + static_cast<std::ptrdiff_t>(depth));
    const cache::Bytes exact = encode_prefix(tokens, identity.export_prefix(depth));
    Writer canonical;
    canonical.string("ninfer/qwen3.8/stable-prefix-alias");
    canonical.u32(2);
    canonical.blob(compatibility_key);
    canonical.u32(depth);
    canonical.blob(exact);
    const cache::Bytes bytes = std::move(canonical).finish();
    artifact::Sha256 hash;
    hash.update(std::as_bytes(std::span(bytes)));
    const artifact::Sha256Digest digest = hash.finish();
    constexpr char hex[] = "0123456789abcdef";
    // The cache treats an alias name as opaque and carries what it means in AliasKind, so this
    // namespace exists only to keep the target's own alias space distinct from a routing hint.
    std::string alias("@stable/v2/");
    alias.reserve(alias.size() + 64);
    for (const std::uint8_t byte : digest) {
        alias.push_back(hex[byte >> 4]);
        alias.push_back(hex[byte & 0x0f]);
    }
    return alias;
}

// Turn openers older than this many are not looked up: each lookup hashes the prefix through
// its depth and costs a cache probe, and a conversation's reusable state is at its recent
// openers. System/tools and explicit boundaries are always looked up.
constexpr std::size_t kTurnOpenerLookupWindow = 8;

inline std::vector<PromptBoundaryAlias> boundary_aliases(
    std::span<const std::uint8_t> compatibility_key, const PreparedPromptData& prompt) {
    std::vector<PromptBoundaryAlias> result;
    if (!prompt.identity.reusable) return result;
    std::size_t openers = 0;
    for (const PromptBoundary& boundary : prompt.identity.boundaries) {
        if (boundary.kind == PromptBoundaryKind::TurnOpener) ++openers;
    }
    const std::size_t skipped_openers =
        openers > kTurnOpenerLookupWindow ? openers - kTurnOpenerLookupWindow : 0;
    std::size_t opener_index = 0;
    result.reserve(prompt.identity.boundaries.size());
    for (const PromptBoundary& boundary : prompt.identity.boundaries) {
        if (boundary.kind == PromptBoundaryKind::TurnOpener && opener_index++ < skipped_openers) {
            continue;
        }
        std::optional<std::string> alias = boundary_alias(compatibility_key, prompt, boundary.depth);
        if (!alias) continue;
        result.push_back(PromptBoundaryAlias{
            .depth            = boundary.depth,
            .kind             = boundary.kind,
            .publish          = boundary.publish,
            .rewrite_frontier = prompt.identity.turn_rewrite_boundary == boundary.depth,
            .alias            = std::move(*alias)});
    }
    return result;
}

// The nearest frontier past the cursor among the lane's capture points, so every prefill chunk
// ends exactly on a frontier the lane must snapshot. `boundaries` is ascending.
inline std::optional<std::uint32_t>
next_prefill_checkpoint(std::uint32_t cursor, std::span<const std::uint32_t> boundaries,
                        std::optional<std::uint32_t> turn,
                        std::optional<std::uint32_t> user_turn) noexcept {
    std::optional<std::uint32_t> next;
    for (const std::uint32_t boundary : boundaries) {
        if (boundary > cursor) {
            next = boundary;
            break;
        }
    }
    if (user_turn && *user_turn > cursor && (!next || *user_turn < *next)) next = user_turn;
    if (turn && *turn > cursor && (!next || *turn < *next)) next = turn;
    return next;
}

struct FrontierMetadata {
    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier = 0;
    std::int32_t rope_delta = 0;
    std::uint32_t text_kv_valid = 0;
    std::uint32_t backend_kv_valid = 0;
    std::vector<TokenId> mtp_drafts;
};

inline bool valid_ledger_frontier(std::uint32_t execution_frontier,
                                  std::uint32_t ledger_frontier) noexcept {
    return execution_frontier != 0 &&
           (ledger_frontier == execution_frontier ||
            (execution_frontier != std::numeric_limits<std::uint32_t>::max() &&
             ledger_frontier == execution_frontier + 1U));
}

inline cache::Bytes encode_frontier(const FrontierMetadata& metadata) {
    Writer out;
    write_header(out, "qwen-frontier");
    out.u32(metadata.execution_frontier);
    out.u32(metadata.ledger_frontier);
    out.i32(metadata.rope_delta);
    out.u32(metadata.text_kv_valid);
    out.u32(metadata.backend_kv_valid);
    write_i32_vector(out, metadata.mtp_drafts);
    return std::move(out).finish();
}

inline FrontierMetadata decode_frontier(std::span<const std::uint8_t> bytes) {
    Reader in(bytes);
    read_header(in, "qwen-frontier");
    FrontierMetadata out;
    out.execution_frontier = in.u32();
    out.ledger_frontier    = in.u32();
    out.rope_delta         = in.i32();
    out.text_kv_valid      = in.u32();
    out.backend_kv_valid   = in.u32();
    out.mtp_drafts =
        read_i32_vector<TokenId>(in, qwen3_8::kMtpDecodeMaximumDrafts);
    in.finish();
    return out;
}

struct BoundaryMetadata {
    bool valid = false;
    std::uint32_t frontier = 0;
};

inline cache::Bytes encode_boundary(BoundaryMetadata metadata) {
    Writer out;
    write_header(out, "qwen-boundary");
    out.u8(metadata.valid ? 1 : 0);
    out.u32(metadata.frontier);
    return std::move(out).finish();
}

inline BoundaryMetadata decode_boundary(std::span<const std::uint8_t> bytes) {
    Reader in(bytes);
    read_header(in, "qwen-boundary");
    const std::uint8_t valid = in.u8();
    if (valid > 1) { throw std::invalid_argument("continuation boundary validity is invalid"); }
    BoundaryMetadata out{.valid = valid != 0, .frontier = in.u32()};
    in.finish();
    if (!out.valid && out.frontier != 0) {
        throw std::invalid_argument("invalid continuation boundary has a nonzero frontier");
    }
    return out;
}

inline void write_plane_spec(Writer& out, const PagedKVPlaneSpec& spec) {
    out.u8(static_cast<std::uint8_t>(spec.dtype));
    out.i32(spec.leading_extent);
    out.i32(spec.head_extent);
    out.u64(spec.alignment);
}

inline PagedKVPlaneSpec read_plane_spec(Reader& in) {
    return PagedKVPlaneSpec{.dtype = static_cast<DType>(in.u8()),
                            .leading_extent = in.i32(),
                            .head_extent = in.i32(),
                            .alignment = static_cast<std::size_t>(in.u64())};
}

using Segments = std::map<std::string, cache::Bytes>;

// Paged KV is emitted as one small descriptor segment plus one segment per plane, rather than as a
// single blob. Keeping a plane in its own segment is what lets the cache's content-addressed
// chunking share bytes with the image published one turn earlier: concatenated into one blob,
// every plane after the first is displaced as soon as an earlier plane grows, and nothing below
// the frontier is ever recognised again.
//
// For a PageMajor pool - which is what the text and MTP caches use - a plane payload is dense in
// logical page order, so pages [0, P) are a byte prefix of pages [0, P+k) and the whole unchanged
// prefix is recognised. A HeadMajor payload interleaves heads at a stride that grows with the page
// count, so only its first head keeps its offsets; those planes dedup when they are unchanged and
// otherwise fall back to a private copy. Only the DFlash full pool is HeadMajor.
[[nodiscard]] inline std::string paged_plane_segment(std::string_view base, std::size_t plane) {
    char suffix[16];
    std::snprintf(suffix, sizeof(suffix), ".p%03zu", plane);
    return std::string(base) + suffix;
}

[[nodiscard]] inline std::string paged_descriptor_segment(std::string_view base) {
    return std::string(base) + ".kv";
}

inline void emit_paged(Segments& segments, std::string_view base, PagedKVLogicalImage&& image) {
    Writer out;
    write_header(out, "paged-kv");
    out.u32(image.valid_tokens);
    out.u8(static_cast<std::uint8_t>(image.plane_order));
    out.u64(image.planes.size());
    for (const auto& spec : image.planes) { write_plane_spec(out, spec); }
    out.u64(image.payloads.size());
    for (const auto& payload : image.payloads) { out.u64(payload.size()); }
    segments.emplace(paged_descriptor_segment(base), std::move(out).finish());
    for (std::size_t index = 0; index < image.payloads.size(); ++index) {
        segments.emplace(paged_plane_segment(base, index), std::move(image.payloads[index]));
    }
}

inline void paged_segment_names(std::string_view base, std::size_t plane_count,
                                std::set<std::string>& out) {
    out.emplace(paged_descriptor_segment(base));
    for (std::size_t index = 0; index < plane_count; ++index) {
        out.emplace(paged_plane_segment(base, index));
    }
}

[[nodiscard]] inline PagedKVLogicalImage decode_paged(const Segments& segments,
                                                      std::string_view base,
                                                      const PagedKVPool& pool,
                                                      std::uint32_t expected_valid_tokens) {
    const auto descriptor = segments.find(paged_descriptor_segment(base));
    if (descriptor == segments.end()) {
        throw std::invalid_argument("paged KV image is missing its descriptor");
    }
    Reader in(descriptor->second);
    read_header(in, "paged-kv");
    PagedKVLogicalImage out;
    out.valid_tokens = in.u32();
    if (out.valid_tokens != expected_valid_tokens) {
        throw std::invalid_argument("paged KV image has an incompatible token extent");
    }
    const std::uint8_t order = in.u8();
    if (order > static_cast<std::uint8_t>(PagedKVPlaneOrder::HeadMajor)) {
        throw std::invalid_argument("paged KV image has an invalid plane order");
    }
    out.plane_order = static_cast<PagedKVPlaneOrder>(order);
    if (out.plane_order != pool.plane_order()) {
        throw std::invalid_argument("paged KV image plane order differs from pool");
    }
    out.planes.resize(in.count(pool.plane_count()));
    if (out.planes.size() != pool.plane_count()) {
        throw std::invalid_argument("paged KV image has an incompatible plane inventory");
    }
    for (std::size_t index = 0; index < out.planes.size(); ++index) {
        out.planes[index] = read_plane_spec(in);
        const Tensor& plane = pool.plane(index);
        const PagedKVPlaneSpec& expected = pool.plane_spec(index);
        const std::int32_t heads = out.plane_order == PagedKVPlaneOrder::PageMajor
                                       ? plane.ne[2]
                                       : plane.ne[3];
        if (out.planes[index].dtype != plane.dtype ||
            out.planes[index].leading_extent != plane.ne[0] ||
            out.planes[index].head_extent != heads ||
            out.planes[index].alignment != expected.alignment) {
            throw std::invalid_argument("paged KV image plane geometry differs from pool");
        }
    }
    const std::size_t payload_count = in.count(pool.plane_count());
    if (payload_count != pool.plane_count()) {
        throw std::invalid_argument("paged KV image has an incompatible payload inventory");
    }
    std::vector<std::uint64_t> declared(payload_count);
    for (auto& size : declared) { size = in.u64(); }
    in.finish();

    const std::size_t pages = expected_valid_tokens == 0
                                  ? 0
                                  : 1U + (expected_valid_tokens - 1U) / kPagedKVPageSize;
    out.payloads.resize(payload_count);
    for (std::size_t index = 0; index < payload_count; ++index) {
        const Tensor& plane = pool.plane(index);
        const std::size_t page_bytes =
            out.plane_order == PagedKVPlaneOrder::PageMajor
                ? static_cast<std::size_t>(plane.nb[3])
                : static_cast<std::size_t>(plane.nb[2]) * static_cast<std::size_t>(plane.ne[3]);
        if (pages != 0 && page_bytes > std::numeric_limits<std::size_t>::max() / pages) {
            throw std::invalid_argument("paged KV image payload extent overflows size_t");
        }
        const std::size_t expected_bytes = page_bytes * pages;
        const auto payload = segments.find(paged_plane_segment(base, index));
        if (payload == segments.end() || payload->second.size() != expected_bytes ||
            declared[index] != expected_bytes) {
            throw std::invalid_argument("paged KV plane payload is missing or missized");
        }
        out.payloads[index] = payload->second;
    }
    if (out.planes.size() != out.payloads.size()) {
        throw std::invalid_argument("paged KV image has inconsistent planes");
    }
    return out;
}

inline void write_linear_header(Writer& out, const LinearAttentionStatePoolSpec& spec) {
    write_header(out, "linear-state");
    out.u32(spec.layers);
    out.i32(spec.conv_channels);
    out.i32(spec.conv_width);
    out.i32(spec.value_heads);
    out.i32(spec.value_head_dim);
    out.i32(spec.key_head_dim);
    out.u8(static_cast<std::uint8_t>(spec.conv_dtype));
}

inline cache::Bytes encode_linear(const LinearAttentionStateImage& image) {
    Writer out;
    write_linear_header(out, LinearAttentionStatePoolSpec{
                                 .layers         = image.layers,
                                 .conv_channels  = image.conv_channels,
                                 .conv_width     = image.conv_width,
                                 .value_heads    = image.value_heads,
                                 .value_head_dim = image.value_head_dim,
                                 .key_head_dim   = image.key_head_dim,
                                 .conv_dtype     = image.conv_dtype,
                             });
    out.u64(image.conv.size());
    for (const auto& payload : image.conv) { out.blob(payload); }
    out.u64(image.recurrent.size());
    for (const auto& payload : image.recurrent) { out.blob(payload); }
    return std::move(out).finish();
}

// The "linear-state" segment of one pool slot, byte-identical to
// encode_linear(export_linear_attention_state(pool, slot, ...)). The wire layout is laid out
// first and every layer's device copy lands directly at its blob offset, so the segment costs
// one host allocation and one device-to-host pass instead of a per-layer image that is then
// serialized again.
inline cache::Bytes export_linear_segment(const LinearAttentionStatePool& pool, std::int32_t slot,
                                          PinnedTransferBuffer& transfer, cudaStream_t stream) {
    const std::uint32_t layers = pool.layer_count();
    Writer header;
    write_linear_header(header, pool.spec);
    cache::Bytes out               = std::move(header).finish();
    const std::size_t header_bytes = out.size();
    std::size_t total              = header_bytes + 2 * sizeof(std::uint64_t);
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        total += sizeof(std::uint64_t) + pool.conv_slot(layer, slot).bytes();
        total += sizeof(std::uint64_t) + pool.recurrent_slot(layer, slot).bytes();
    }
    out.resize(total);
    std::size_t cursor = header_bytes;
    const auto put_u64 = [&](std::uint64_t value) {
        for (unsigned shift = 0; shift != 64; shift += 8) {
            out[cursor++] = static_cast<std::uint8_t>(value >> shift);
        }
    };
    std::vector<DeviceToHostTransfer> copies;
    copies.reserve(2 * static_cast<std::size_t>(layers));
    const auto emit_slots = [&](auto&& slot_tensor) {
        put_u64(layers);
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            const Tensor source = slot_tensor(layer);
            if (!source.is_contiguous()) {
                throw std::logic_error("linear attention state slot is not contiguous");
            }
            put_u64(source.bytes());
            copies.push_back({&out, cursor, source.data, source.bytes()});
            cursor += source.bytes();
        }
    };
    emit_slots([&](std::uint32_t layer) { return pool.conv_slot(layer, slot); });
    emit_slots([&](std::uint32_t layer) { return pool.recurrent_slot(layer, slot); });
    if (cursor != out.size()) { throw std::logic_error("linear-state segment layout mismatch"); }
    transfer.copy_device_to_host(copies, stream);
    return out;
}

inline LinearAttentionStateImage decode_linear(std::span<const std::uint8_t> bytes,
                                                const LinearAttentionStatePool& pool) {
    Reader in(bytes);
    read_header(in, "linear-state");
    LinearAttentionStateImage out;
    out.layers         = in.u32();
    out.conv_channels  = in.i32();
    out.conv_width     = in.i32();
    out.value_heads    = in.i32();
    out.value_head_dim = in.i32();
    out.key_head_dim   = in.i32();
    out.conv_dtype     = static_cast<DType>(in.u8());
    if (out.layers != pool.layer_count() || out.conv_channels != pool.spec.conv_channels ||
        out.conv_width != pool.spec.conv_width || out.value_heads != pool.spec.value_heads ||
        out.value_head_dim != pool.spec.value_head_dim ||
        out.key_head_dim != pool.spec.key_head_dim || out.conv_dtype != pool.spec.conv_dtype) {
        throw std::invalid_argument("linear state image geometry differs from pool");
    }
    out.conv.resize(in.count(out.layers));
    if (out.conv.size() != out.layers) {
        throw std::invalid_argument("linear state image has an incompatible layer inventory");
    }
    for (std::uint32_t layer = 0; layer < out.layers; ++layer) {
        out.conv[layer] = in.blob_exact(pool.conv_slot(layer, 0).bytes());
    }
    out.recurrent.resize(in.count(out.layers));
    if (out.recurrent.size() != out.layers) {
        throw std::invalid_argument("linear state image has an incompatible layer inventory");
    }
    for (std::uint32_t layer = 0; layer < out.layers; ++layer) {
        out.recurrent[layer] = in.blob_exact(pool.recurrent_slot(layer, 0).bytes());
    }
    in.finish();
    if (out.layers != out.conv.size() || out.layers != out.recurrent.size()) {
        throw std::invalid_argument("linear state image has inconsistent layers");
    }
    return out;
}

inline cache::Bytes encode_cyclic(const CyclicKVCacheImage& image) {
    Writer out;
    write_header(out, "cyclic-kv");
    out.u32(image.layers);
    out.u32(image.capacity);
    out.u32(image.padded_capacity);
    out.i32(image.num_kv_heads);
    out.i32(image.head_dim);
    out.u64(image.k.size());
    for (const auto& payload : image.k) { out.blob(payload); }
    out.u64(image.v.size());
    for (const auto& payload : image.v) { out.blob(payload); }
    return std::move(out).finish();
}

inline CyclicKVCacheImage decode_cyclic(std::span<const std::uint8_t> bytes,
                                        const CyclicKVCache& cache) {
    Reader in(bytes);
    read_header(in, "cyclic-kv");
    CyclicKVCacheImage out;
    out.layers          = in.u32();
    out.capacity        = in.u32();
    out.padded_capacity = in.u32();
    out.num_kv_heads    = in.i32();
    out.head_dim        = in.i32();
    if (out.layers != cache.layer_count() || out.capacity != cache.capacity() ||
        out.padded_capacity != cache.padded_capacity() ||
        out.num_kv_heads != cache.num_kv_heads() || out.head_dim != cache.head_dim()) {
        throw std::invalid_argument("cyclic KV image geometry differs from cache");
    }
    out.k.resize(in.count(out.layers));
    if (out.k.size() != out.layers) {
        throw std::invalid_argument("cyclic KV image has an incompatible layer inventory");
    }
    for (std::uint32_t layer = 0; layer < out.layers; ++layer) {
        out.k[layer] = in.blob_exact(cache.layer_view(layer).k.slice(3, 0, 1).bytes());
    }
    out.v.resize(in.count(out.layers));
    if (out.v.size() != out.layers) {
        throw std::invalid_argument("cyclic KV image has an incompatible layer inventory");
    }
    for (std::uint32_t layer = 0; layer < out.layers; ++layer) {
        out.v[layer] = in.blob_exact(cache.layer_view(layer).v.slice(3, 0, 1).bytes());
    }
    in.finish();
    if (out.layers != out.k.size() || out.layers != out.v.size()) {
        throw std::invalid_argument("cyclic KV image has inconsistent layers");
    }
    return out;
}

inline cache::Bytes encode_tensor(std::span<const std::uint8_t> payload) {
    Writer out;
    write_header(out, "tensor");
    out.blob(payload);
    return std::move(out).finish();
}

inline cache::Bytes decode_tensor(std::span<const std::uint8_t> bytes, std::size_t expected) {
    Reader in(bytes);
    read_header(in, "tensor");
    cache::Bytes out = in.blob_exact(expected);
    in.finish();
    if (out.size() != expected) {
        throw std::invalid_argument("continuation tensor has an incompatible extent");
    }
    return out;
}

} // namespace ninfer::targets::qwen3_8::detail::continuation

namespace ninfer::targets::qwen3_8 {

// The host-side product of decoding a continuation image, owned and pointer-free.
//
// Decoding materialises the whole image on the host and is the largest CPU term in a restore. It
// reads only startup-fixed pool geometry, touches no lane state, no device memory and no CUDA API,
// so it is separated from the import step and may be produced on a preparation thread while the
// GPU executor runs. Everything that follows decoding - the KV reservation, the device transfers
// and the sequence-state writes - remains the executor's exclusive work.
struct DecodedContinuation {
    PagedKVLogicalImage text_kv;
    LinearAttentionStateImage current_gdn;
    cache::Bytes tail_hidden;
    std::optional<LinearAttentionStateImage> checkpoint_gdn;
    std::optional<cache::Bytes> checkpoint_hidden;
    std::optional<PagedKVLogicalImage> backend_kv;
    std::optional<CyclicKVCacheImage> dflash_local;
    std::optional<CyclicKVCacheImage> dflash_checkpoint_local;

    // Resident host cost, used by the engine to bound how much decoding it runs ahead.
    [[nodiscard]] std::uint64_t host_bytes() const noexcept {
        std::uint64_t total = tail_hidden.size();
        const auto paged = [](const PagedKVLogicalImage& image) {
            std::uint64_t bytes = 0;
            for (const auto& payload : image.payloads) { bytes += payload.size(); }
            return bytes;
        };
        const auto linear = [](const LinearAttentionStateImage& image) {
            std::uint64_t bytes = 0;
            for (const auto& layer : image.conv) { bytes += layer.size(); }
            for (const auto& layer : image.recurrent) { bytes += layer.size(); }
            return bytes;
        };
        const auto cyclic = [](const CyclicKVCacheImage& image) {
            std::uint64_t bytes = 0;
            for (const auto& layer : image.k) { bytes += layer.size(); }
            for (const auto& layer : image.v) { bytes += layer.size(); }
            return bytes;
        };
        total += paged(text_kv) + linear(current_gdn);
        if (checkpoint_gdn) { total += linear(*checkpoint_gdn); }
        if (checkpoint_hidden) { total += checkpoint_hidden->size(); }
        if (backend_kv) { total += paged(*backend_kv); }
        if (dflash_local) { total += cyclic(*dflash_local); }
        if (dflash_checkpoint_local) { total += cyclic(*dflash_checkpoint_local); }
        return total;
    }
};

} // namespace ninfer::targets::qwen3_8
