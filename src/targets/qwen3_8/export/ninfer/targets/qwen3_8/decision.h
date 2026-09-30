#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_8 {

// System One decision layout (kev model.encode with its serving limits). A decision is the state
// row [<|fim_prefix|>, state...] followed, per question in request order, by the branch
// [<|fim_middle|>, instructions..., (<|box_start|>, option..., <|box_end|>)..., <|fim_suffix|>].
// Every branch sees the state and itself only; its positions restart at the state length.
// Readouts are the final-norm hidden states at each <|box_end|> (options) and at <|fim_suffix|>
// (decide).

// The state row holds at most this many tokens including its delimiter; a longer state text is
// cut to its head silently. A state plus one branch holds at most kDecisionMaxRowTokens.
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

// A laid-out decision: the state row followed by every branch in request order (kev's packed
// token sequence).
struct DecisionPrompt {
    std::vector<TokenId> tokens;
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
    double branch_seconds             = 0.0;
};

// Branch scratch a decision lane holds above its state: packed short branches occupy at most one
// pass of `pass_columns`, and a branch longer than a pass occupies its whole length.
[[nodiscard]] inline std::uint32_t decision_scratch_extent(const DecisionPrompt& prompt,
                                                           std::uint32_t pass_columns) noexcept {
    std::uint64_t short_tokens = 0;
    std::uint32_t long_branch  = 0;
    for (const DecisionBranch& branch : prompt.branches) {
        if (branch.length <= pass_columns) {
            short_tokens += branch.length;
        } else {
            long_branch = std::max(long_branch, branch.length);
        }
    }
    return std::max(static_cast<std::uint32_t>(
                        std::min<std::uint64_t>(short_tokens, pass_columns)),
                    long_branch);
}

} // namespace ninfer::targets::qwen3_8
