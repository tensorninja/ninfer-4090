// Exact kev parity of the System One decision layout with the served checkpoint's tokenizer.
// kev's user_tokens and encode outputs at its serving limits (tests/fixtures/systemone/
// kev_layout.json) are compared with Frontend::prepare_decision, and kev's usage.output_tokens
// (the answers cases of tests/fixtures/systemone/kev_golden.json) with
// Frontend::count_text_tokens. tools/parity/qwen3_8_27b/decision.py captures both from kev.
// Opt-in through NINFER_QWEN3_8_TOKENIZER_DIR, a directory holding the checkpoint's six frontend
// resources: tokenizer.json, tokenizer_config.json, chat_template.jinja, generation_config.json,
// preprocessor_config.json and video_preprocessor_config.json.
#include <ninfer/targets/qwen3_8/decision.h>
#include <ninfer/targets/qwen3_8/frontend.h>
#include <ninfer/targets/qwen3_8/frontend_resources.h>
#include <ninfer/types.h>

#include "artifact/sha256.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace q8 = ninfer::targets::qwen3_8;
using Json   = nlohmann::json;
using ninfer::TokenId;

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path.string()); }
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

Json read_fixture(const char* name) {
    return Json::parse(
        read_file(std::filesystem::path(NINFER_SOURCE_DIR) / "tests/fixtures/systemone" / name));
}

q8::Frontend load_frontend(const std::filesystem::path& root) {
    q8::FrontendResources resources;
    resources.tokenizer_json                 = read_file(root / "tokenizer.json");
    resources.tokenizer_config_json          = read_file(root / "tokenizer_config.json");
    resources.chat_template_jinja            = read_file(root / "chat_template.jinja");
    resources.generation_config_json         = read_file(root / "generation_config.json");
    resources.preprocessor_config_json       = read_file(root / "preprocessor_config.json");
    resources.video_preprocessor_config_json = read_file(root / "video_preprocessor_config.json");
    return q8::make_frontend(resources, /*vision_enabled=*/false);
}

class Failures {
public:
    void check(bool condition, const std::string& label, const std::string& message) {
        if (condition) { return; }
        ++count_;
        std::cerr << label << ": " << message << '\n';
    }

    [[nodiscard]] int count() const noexcept { return count_; }

private:
    int count_ = 0;
};

// A fixture text: a string, or {"repeat": unit, "times": n} meaning unit * n.
std::string fixture_text(const Json& value) {
    if (value.is_string()) { return value.get<std::string>(); }
    const auto unit  = value.at("repeat").get<std::string>();
    const auto times = value.at("times").get<std::size_t>();
    std::string text;
    text.reserve(unit.size() * times);
    for (std::size_t index = 0; index < times; ++index) { text += unit; }
    return text;
}

ninfer::DecisionInput decision_input(const Json& input) {
    ninfer::DecisionInput decision;
    decision.state = fixture_text(input.at("state"));
    for (const Json& question : input.at("questions")) {
        ninfer::DecisionQuestion& target = decision.questions.emplace_back();
        target.instructions              = fixture_text(question.at("instructions"));
        for (const Json& option : question.at("options")) {
            target.options.push_back(fixture_text(option));
        }
    }
    return decision;
}

// The fixture's ids_sha256: SHA-256 of the ids as little-endian int32.
std::string ids_sha256(std::span<const TokenId> ids) {
    std::vector<std::byte> bytes;
    bytes.reserve(ids.size() * 4);
    for (const TokenId id : ids) {
        const auto value = static_cast<std::uint32_t>(id);
        for (int shift = 0; shift < 32; shift += 8) {
            bytes.push_back(static_cast<std::byte>((value >> shift) & 0xFFU));
        }
    }
    ninfer::artifact::Sha256 hash;
    hash.update(bytes);
    constexpr char kHex[] = "0123456789abcdef";
    std::string hex;
    for (const std::uint8_t byte : hash.finish()) {
        hex.push_back(kHex[byte >> 4]);
        hex.push_back(kHex[byte & 0x0FU]);
    }
    return hex;
}

std::string difference(std::span<const TokenId> actual, std::span<const TokenId> expected) {
    const auto [got, want] =
        std::mismatch(actual.begin(), actual.end(), expected.begin(), expected.end());
    std::ostringstream out;
    out << actual.size() << " ids, kev " << expected.size() << "; first difference at index "
        << (got - actual.begin()) << ':';
    if (got != actual.end()) { out << " got " << *got; }
    if (want != expected.end()) { out << " kev " << *want; }
    return out.str();
}

// The ids in full, or for a long layout the count, the digest and the first and last ids.
void check_ids(const Json& item, std::span<const TokenId> actual, const std::string& label,
               Failures& failures) {
    if (item.contains("ids")) {
        const auto expected = item.at("ids").get<std::vector<TokenId>>();
        failures.check(std::ranges::equal(actual, expected), label, difference(actual, expected));
        return;
    }
    const auto length = item.at("ids_length").get<std::size_t>();
    const auto head   = item.at("ids_head").get<std::vector<TokenId>>();
    const auto tail   = item.at("ids_tail").get<std::vector<TokenId>>();
    if (actual.size() != length || actual.size() < head.size() || actual.size() < tail.size()) {
        failures.check(false, label,
                       std::to_string(actual.size()) + " ids, kev " + std::to_string(length));
        return;
    }
    const std::span<const TokenId> first = actual.first(head.size());
    const std::span<const TokenId> last  = actual.last(tail.size());
    failures.check(std::ranges::equal(first, head), label, "head: " + difference(first, head));
    failures.check(std::ranges::equal(last, tail), label, "tail: " + difference(last, tail));
    failures.check(ids_sha256(actual) == item.at("ids_sha256").get<std::string>(), label,
                   "the ids differ from kev's (SHA-256)");
}

// kev tokenizes the state, the instructions and every option alone with user_tokens, so one text
// in all three places lays out as [state] ids [question] ids [open] ids [close] [decide].
void check_user_tokens(const q8::Frontend& frontend, const std::vector<TokenId>& delimiters,
                       const Json& item, const std::string& label, Failures& failures) {
    const auto text = item.at("text").get<std::string>();
    const auto ids  = item.at("ids").get<std::vector<TokenId>>();
    ninfer::DecisionInput input;
    input.state = text;
    input.questions.push_back({.instructions = text, .options = {text}});

    std::vector<TokenId> expected;
    for (std::size_t index = 0; index < 3; ++index) {
        expected.push_back(delimiters[index]);
        expected.insert(expected.end(), ids.begin(), ids.end());
    }
    expected.push_back(delimiters[3]);
    expected.push_back(delimiters[4]);

    const q8::DecisionPrompt prompt = frontend.prepare_decision(input);
    failures.check(std::ranges::equal(prompt.tokens, expected), label,
                   difference(prompt.tokens, expected));
}

void check_encode(const q8::Frontend& frontend, const Json& item, const std::string& label,
                  Failures& failures) {
    const ninfer::DecisionInput input = decision_input(item.at("input"));
    if (item.contains("error")) {
        // kev's ContextOverflow text, which kev serves as the 422 detail.
        const auto expected = item.at("error").get<std::string>();
        try {
            (void)frontend.prepare_decision(input);
            failures.check(false, label, "laid out; kev rejects it: " + expected);
        } catch (const ninfer::DecisionInputError& error) {
            failures.check(error.what() == expected, label,
                           "rejected with \"" + std::string(error.what()) + "\", kev with \"" +
                               expected + "\"");
        }
        return;
    }
    const q8::DecisionPrompt prompt = frontend.prepare_decision(input);
    check_ids(item, prompt.tokens, label, failures);

    const ninfer::DecisionSummary& summary = prompt.summary;
    const auto state_tokens                = item.at("state_tokens").get<std::uint32_t>();
    const auto decide  = item.at("decide_idx").get<std::vector<std::uint32_t>>();
    const auto options = item.at("opt_idx").get<std::vector<std::vector<std::uint32_t>>>();
    failures.check(summary.state_tokens == state_tokens, label,
                   "state of " + std::to_string(summary.state_tokens) + " tokens, kev " +
                       std::to_string(state_tokens));
    failures.check(summary.state_truncated == item.at("state_truncated").get<bool>(), label,
                   "state truncation flag differs from kev's");
    if (prompt.branches.size() != decide.size()) {
        failures.check(false, label,
                       std::to_string(prompt.branches.size()) + " branches, kev " +
                           std::to_string(decide.size()));
        return;
    }
    // kev packs the branches back to back after the state; each ends at its decide token.
    std::uint32_t begin   = state_tokens;
    std::uint32_t longest = 0;
    std::uint32_t count   = 0;
    for (std::size_t k = 0; k < decide.size(); ++k) {
        const q8::DecisionBranch& branch = prompt.branches[k];
        const std::string where          = label + " branch " + std::to_string(k);
        const std::uint32_t length       = decide[k] + 1 - begin;
        failures.check(branch.begin == begin && branch.length == length, where,
                       "spans [" + std::to_string(branch.begin) + ", +" +
                           std::to_string(branch.length) + "), kev [" + std::to_string(begin) +
                           ", +" + std::to_string(length) + ")");
        std::vector<std::uint32_t> readouts;
        for (const std::uint32_t offset : branch.option_readouts) {
            readouts.push_back(branch.begin + offset);
        }
        failures.check(readouts == options[k], where, "option readouts differ from kev's");
        longest = std::max(longest, length);
        count += static_cast<std::uint32_t>(options[k].size());
        begin = decide[k] + 1;
    }
    failures.check(summary.questions == decide.size() && summary.options == count &&
                       summary.branch_tokens == begin - state_tokens &&
                       summary.longest_branch == longest &&
                       summary.input_tokens() == prompt.tokens.size(),
                   label, "summary counts differ from kev's layout");
}

int check_kev_parity(const std::filesystem::path& dir) {
    const q8::Frontend frontend = load_frontend(dir);
    const Json layout           = read_fixture("kev_layout.json");
    const Json golden           = read_fixture("kev_golden.json");
    Failures failures;

    failures.check(layout.at("format") == "ninfer_systemone_kev_layout_v1", "fixture",
                   "unknown layout fixture format");
    failures.check(
        layout.at("serve_max_state").get<std::uint32_t>() == q8::kDecisionMaxStateTokens &&
            layout.at("serve_max_branch").get<std::uint32_t>() == q8::kDecisionMaxRowTokens,
        "limits", "SERVE_MAX_STATE or SERVE_MAX_BRANCH differs from kev's");
    // kev's delimiter order is the layout's: state, question, option open, option close, decide.
    std::vector<TokenId> delimiters;
    for (const Json& delimiter : layout.at("delimiters")) {
        const auto name = delimiter.at("token").get<std::string>();
        failures.check(delimiters.size() < q8::kDecisionDelimiterNames.size() &&
                           name == q8::kDecisionDelimiterNames[delimiters.size()],
                       "delimiters",
                       "kev's delimiter " + std::to_string(delimiters.size()) + " is " + name);
        delimiters.push_back(delimiter.at("id").get<TokenId>());
    }
    if (delimiters.size() != q8::kDecisionDelimiterNames.size()) {
        std::cerr << "kev has " << delimiters.size() << " delimiters\n";
        return 1;
    }

    const auto guarded = [&](const std::string& label, const auto& check) {
        try {
            check();
        } catch (const std::exception& error) {
            failures.check(false, label, std::string("threw: ") + error.what());
        }
    };
    const Json& user_tokens = layout.at("user_tokens");
    for (std::size_t index = 0; index < user_tokens.size(); ++index) {
        const std::string label = "user_tokens case " + std::to_string(index);
        guarded(label, [&] {
            check_user_tokens(frontend, delimiters, user_tokens[index], label, failures);
        });
    }
    const Json& encode = layout.at("encode");
    for (std::size_t index = 0; index < encode.size(); ++index) {
        const std::string label = "encode case " + std::to_string(index);
        guarded(label, [&] { check_encode(frontend, encode[index], label, failures); });
    }
    const Json& answers = golden.at("answers");
    for (std::size_t index = 0; index < answers.size(); ++index) {
        const std::string label = "output_tokens case " + std::to_string(index);
        guarded(label, [&] {
            const std::uint32_t count =
                frontend.count_text_tokens(answers[index].at("answers").get<std::string>());
            const auto expected = answers[index].at("output_tokens").get<std::uint32_t>();
            failures.check(count == expected, label,
                           std::to_string(count) + " tokens, kev " + std::to_string(expected));
        });
    }

    if (failures.count() != 0) {
        std::cerr << failures.count() << " mismatches with kev\n";
        return 1;
    }
    std::cout << "kev parity: " << user_tokens.size() << " user_tokens, " << encode.size()
              << " encode and " << answers.size() << " output_tokens cases\nok\n";
    return 0;
}

} // namespace

int main() {
    const char* dir = std::getenv("NINFER_QWEN3_8_TOKENIZER_DIR");
    if (dir == nullptr) {
        std::cout << "skip: NINFER_QWEN3_8_TOKENIZER_DIR is not set\n";
        return 77;
    }
    try {
        return check_kev_parity(dir);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
