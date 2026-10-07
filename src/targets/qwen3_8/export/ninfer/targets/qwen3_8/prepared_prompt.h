#pragma once

#include <ninfer/types.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_8 {

class PreparedPrompt;

enum class PromptModality : std::uint8_t {
    Image = 1,
    Video = 2,
};

struct VisionGrid {
    std::int32_t temporal = 0;
    std::int32_t height   = 0;
    std::int32_t width    = 0;
};

struct TokenSpan {
    std::size_t begin = 0;
    std::size_t count = 0;
};

struct VisionItem {
    PromptModality modality = PromptModality::Image;
    VisionGrid grid;
    std::size_t patch_begin = 0;
    std::size_t patch_count = 0;
    // SHA-256 of the owned encoded media bytes. Grid/modality/span identity is carried
    // separately so this digest binds the content without retaining the request payload.
    std::array<std::uint8_t, 32> content_digest{};
    std::array<std::uint8_t, 32> preprocessing_digest{};
    std::vector<double> timestamps;
    std::vector<TokenSpan> token_spans;
};

// A content-addressed prefix boundary. Every boundary names a continuation alias that any
// later prompt sharing the exact tokens through `depth` can restore from; the ones marked
// `publish` are captured during prefill and published under that alias.
enum class PromptBoundaryKind : std::uint8_t {
    // End of the initial system/tools block: the prefix independent conversations share.
    SystemTools,
    // An `<|im_start|>assistant\n` opener. Every one is a lookup candidate; the one that equals
    // the turn-rewrite frontier publishes.
    TurnOpener,
    // A client breakpoint. Always looked up and always published.
    Explicit,
};

struct PromptBoundary {
    std::uint32_t depth     = 0;
    PromptBoundaryKind kind = PromptBoundaryKind::SystemTools;
    bool publish            = false;
};

struct PromptIdentity {
    bool reusable = true;
    // The policy-selected rewrite frontier: the lane ends the request holding its turn
    // checkpoint here.
    std::optional<std::uint32_t> turn_rewrite_boundary;
    // Opener of the last real user query; sits before that message's content so it survives a
    // client rewriting the message tail (floating synthetic reminders).
    std::optional<std::uint32_t> user_turn_boundary;
    // Strictly ascending by depth. None lies past `turn_rewrite_boundary` when it exists.
    std::vector<PromptBoundary> boundaries;
};

struct PrepareStats {
    double seconds                = 0.0;
    std::size_t media_items       = 0;
    std::uint64_t raw_patches     = 0;
    std::uint64_t vision_tokens   = 0;
    std::uint64_t attention_pairs = 0;
    std::size_t patch_bytes       = 0;
};

struct PreparedPromptData {
    std::vector<TokenId> token_ids;
    std::vector<std::uint8_t> token_types;
    std::vector<std::int32_t> positions;
    std::int32_t rope_delta = 0;
    std::vector<float> patches;
    std::vector<VisionItem> vision_items;
    PromptIdentity identity;
    bool starts_in_reasoning = false;
    PrepareStats prepare;

    [[nodiscard]] std::span<const std::int32_t> position_axis(int axis) const;

    [[nodiscard]] bool has_media() const noexcept { return !vision_items.empty(); }

    void release_media_payload() noexcept { std::vector<float>().swap(patches); }
};

class PreparedPromptAccess {
public:
    [[nodiscard]] static const PreparedPromptData& view(const PreparedPrompt& prompt);
    [[nodiscard]] static PreparedPromptData take(PreparedPrompt&& prompt);
};

} // namespace ninfer::targets::qwen3_8
