#pragma once

#include <ninfer/targets/qwen3_8/decision.h>

#include "targets/qwen3_8/impl/frontend/tokenizer.h"

#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_8::frontend_internal {

// kev's user-text rule: every `<|name|>` whose name is one or more ASCII word characters is
// rewritten to `<¦name¦>` (U+00A6) before tokenization, so caller text can never produce the
// delimiters or any other `<|...|>` control token. The remaining added tokens keep their ids.
[[nodiscard]] std::string escape_decision_text(std::string_view text);

// Resolves the five delimiters by name. Throws std::out_of_range when one is absent.
[[nodiscard]] DecisionDelimiters resolve_decision_delimiters(const Tokenizer& tokenizer);

void layout_decision_branches(const Tokenizer& tokenizer, const DecisionDelimiters& delimiters,
                              const DecisionInput& input, DecisionPrompt& prompt);

} // namespace ninfer::targets::qwen3_8::frontend_internal
