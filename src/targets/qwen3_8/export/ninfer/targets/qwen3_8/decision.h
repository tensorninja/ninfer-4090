#pragma once

#include "ninfer/types.h"
#include <ninfer/targets/qwen3_8/prepared_prompt.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_8 {

// System One decision layout (kev model.encode with its serving limits). A decision is the state
// row [<|fim_prefix|>, state...] followed, per question in request order, by the branch
// [<|fim_middle|>, instructions..., (<|box_start|>, option..., <|box_end|>)..., <|fim_suffix|>].
// Every branch sees the state and itself only; its RoPE positions continue the state's MRoPE.
// Readouts are the final-norm hidden states at each <|box_end|> (options) and at <|fim_suffix|>
// (decide).

// The state row holds at most this many tokens including its delimiter. A state plus one branch
// holds at most kDecisionMaxRowTokens.
inline constexpr std::uint32_t kDecisionMaxStateTokens = 65536;
inline constexpr std::uint32_t kDecisionMaxRowTokens   = kDecisionMaxStateTokens + 8192;

// Delimiter names in kev's order: state, question, option open, option close, decide.
inline constexpr std::array<std::string_view, 5> kDecisionDelimiterNames = {
    "<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"};

struct DecisionDelimiters {
    TokenId state        = -1;
    TokenId question     = -1;
    TokenId option_open  = -1;
    TokenId option_close = -1;
    TokenId decide       = -1;
};

using ::ninfer::DecisionBranch;

struct DecisionPrompt {
    PreparedPromptData state;
    std::vector<TokenId> branch_tokens;
    std::vector<DecisionBranch> branches;
    DecisionSummary summary;
    double prepare_seconds = 0.0;

    [[nodiscard]] std::uint32_t state_tokens() const noexcept { return summary.state_tokens; }
};

// What one completed decision produced.
struct DecisionOutcome {
    // Per question in request order, the option probabilities in option order.
    std::vector<std::vector<float>> probabilities;
    // State tokens restored from the lane instead of computed.
    std::uint32_t reused_state_tokens = 0;
    std::uint32_t branch_passes       = 0;
    std::uint32_t long_branch_chunks  = 0;
    double state_seconds              = 0.0;
    double vision_seconds             = 0.0;
    double branch_seconds             = 0.0;
};

// Context a decision occupies: every branch continues the state at positions state, state + 1,
// ..., so the lane must hold the state plus its longest branch.
[[nodiscard]] inline std::uint64_t decision_context_tokens(const DecisionPrompt& prompt) noexcept {
    std::uint32_t longest = 0;
    for (const DecisionBranch& branch : prompt.branches) {
        longest = std::max(longest, branch.length);
    }
    return static_cast<std::uint64_t>(prompt.state_tokens()) + longest;
}

} // namespace ninfer::targets::qwen3_8
