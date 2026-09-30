#include "targets/qwen3_8/impl/frontend/decision_template.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_8::frontend_internal {
namespace {

constexpr std::string_view kBrokenBar = "\xC2\xA6";

bool is_word_byte(char c) noexcept {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
}

void append_text(std::vector<TokenId>& tokens, const Tokenizer& tokenizer, std::string_view text) {
    const std::vector<int> ids = tokenizer.encode(escape_decision_text(text));
    tokens.insert(tokens.end(), ids.begin(), ids.end());
}

std::uint32_t checked_u32(std::size_t value) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw DecisionInputError("decision exceeds 4294967295 tokens");
    }
    return static_cast<std::uint32_t>(value);
}

} // namespace

std::string escape_decision_text(std::string_view text) {
    std::string escaped;
    escaped.reserve(text.size());
    std::size_t pos = 0;
    while (pos < text.size()) {
        // Python re.sub(r"<\|([A-Za-z0-9_]+)\|>") scans left to right for non-overlapping matches;
        // the name class excludes '|', so a failed candidate never hides a later match.
        if (text[pos] == '<' && pos + 1 < text.size() && text[pos + 1] == '|') {
            std::size_t end = pos + 2;
            while (end < text.size() && is_word_byte(text[end])) { ++end; }
            if (end > pos + 2 && end + 1 < text.size() && text[end] == '|' && text[end + 1] == '>') {
                escaped.push_back('<');
                escaped.append(kBrokenBar);
                escaped.append(text.substr(pos + 2, end - pos - 2));
                escaped.append(kBrokenBar);
                escaped.push_back('>');
                pos = end + 2;
                continue;
            }
        }
        escaped.push_back(text[pos]);
        ++pos;
    }
    return escaped;
}

DecisionDelimiters resolve_decision_delimiters(const Tokenizer& tokenizer) {
    return DecisionDelimiters{.state        = tokenizer.token_id(kDecisionDelimiterNames[0]),
                              .question     = tokenizer.token_id(kDecisionDelimiterNames[1]),
                              .option_open  = tokenizer.token_id(kDecisionDelimiterNames[2]),
                              .option_close = tokenizer.token_id(kDecisionDelimiterNames[3]),
                              .decide       = tokenizer.token_id(kDecisionDelimiterNames[4])};
}

DecisionPrompt layout_decision(const Tokenizer& tokenizer, const DecisionDelimiters& delimiters,
                               const DecisionInput& input) {
    if (input.questions.empty()) {
        throw std::invalid_argument("a decision needs at least one question");
    }
    DecisionPrompt prompt;
    std::vector<TokenId>& tokens = prompt.tokens;

    const std::vector<int> state = tokenizer.encode(escape_decision_text(input.state));
    const std::size_t kept       = std::min<std::size_t>(state.size(), kDecisionMaxStateTokens - 1);
    tokens.reserve(1 + kept);
    tokens.push_back(delimiters.state);
    tokens.insert(tokens.end(), state.begin(), state.begin() + static_cast<std::ptrdiff_t>(kept));
    const auto state_tokens = static_cast<std::uint32_t>(tokens.size());

    DecisionSummary& summary = prompt.summary;
    summary.state_tokens     = state_tokens;
    summary.state_truncated  = state.size() + 1 > kDecisionMaxStateTokens;
    summary.questions        = checked_u32(input.questions.size());
    prompt.branches.reserve(input.questions.size());

    const std::uint32_t row_budget = kDecisionMaxRowTokens - state_tokens;
    for (const DecisionQuestion& question : input.questions) {
        if (question.options.empty() || question.options.size() > kMaximumDecisionOptions) {
            throw std::invalid_argument("a decision question needs 1 to 255 options");
        }
        DecisionBranch branch;
        const std::size_t begin = tokens.size();
        branch.begin            = checked_u32(begin);
        tokens.push_back(delimiters.question);
        append_text(tokens, tokenizer, question.instructions);
        branch.option_readouts.reserve(question.options.size());
        for (const std::string& option : question.options) {
            tokens.push_back(delimiters.option_open);
            append_text(tokens, tokenizer, option);
            tokens.push_back(delimiters.option_close);
            branch.option_readouts.push_back(checked_u32(tokens.size() - 1 - begin));
        }
        tokens.push_back(delimiters.decide);
        const std::size_t length = tokens.size() - begin;
        if (length > row_budget) {
            throw DecisionInputError("branch too long: " + std::to_string(length) +
                                     " tokens with a " + std::to_string(state_tokens) +
                                     "-token state (row limit " +
                                     std::to_string(kDecisionMaxRowTokens) + ")");
        }
        branch.length = static_cast<std::uint32_t>(length);
        summary.options += static_cast<std::uint32_t>(question.options.size());
        summary.longest_branch = std::max(summary.longest_branch, branch.length);
        prompt.branches.push_back(std::move(branch));
    }
    summary.branch_tokens = checked_u32(tokens.size() - state_tokens);
    return prompt;
}

} // namespace ninfer::targets::qwen3_8::frontend_internal
