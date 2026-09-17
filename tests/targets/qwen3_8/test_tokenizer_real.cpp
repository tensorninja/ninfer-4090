// Real-vocabulary tokenizer check: exact ids on a fixed mixed-script sample, encode throughput
// over a large natural-text corpus, and rendered-chat boundary encoding against a re-tokenizing
// reference. Opt-in through NINFER_QWEN3_8_TOKENIZER_DIR, a directory holding tokenizer.json,
// tokenizer_config.json, generation_config.json and chat_template.jinja.
//
// NINFER_TOKENIZER_GOLDEN=<path> compares the corpus ids against that file when it exists
// and writes it otherwise, which is how an implementation change is proven token-exact.
#include "targets/qwen3_8/impl/frontend/chat_template.h"
#include "targets/qwen3_8/impl/frontend/processor.h"
#include "targets/qwen3_8/impl/frontend/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fi = ninfer::targets::qwen3_8::frontend_internal;
using fi::EncodeOptions;
using fi::Tokenizer;
using fi::TokenizerResources;

double seconds_since(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

// Slice `index` of `count` equal corpus slices, moved to code point boundaries.
std::string corpus_slice(const std::string& corpus, std::size_t index, std::size_t count) {
    const auto align = [&](std::size_t offset) {
        while (offset < corpus.size() &&
               (static_cast<unsigned char>(corpus[offset]) & 0xC0) == 0x80) {
            ++offset;
        }
        return offset;
    };
    const std::size_t slice = corpus.size() / count;
    const std::size_t begin = align(index * slice);
    const std::size_t end   = align((index + 1) * slice);
    return corpus.substr(begin, end - begin);
}

// A multi-turn chat whose turns slice the corpus, with a client breakpoint inside one message
// and a system prompt, which is the boundary shape prepare sees on every session turn.
std::vector<fi::ChatMessage> build_chat(const std::string& corpus, std::size_t turns) {
    std::vector<fi::ChatMessage> messages;
    fi::ChatMessage system;
    system.role = "system";
    system.parts.push_back(fi::ChatPart::text_part("You are a careful assistant."));
    messages.push_back(std::move(system));
    for (std::size_t turn = 0; turn < turns; ++turn) {
        fi::ChatMessage user;
        user.role = "user";
        user.parts.push_back(fi::ChatPart::text_part(corpus_slice(corpus, 2 * turn, 2 * turns)));
        if (turn == turns / 2) {
            user.parts.push_back(fi::ChatPart::text_part("\nSummarize the section above."));
            user.parts.front().cache_breakpoint = true;
        }
        messages.push_back(std::move(user));
        if (turn + 1 == turns) { break; }
        fi::ChatMessage assistant;
        assistant.role = "assistant";
        assistant.parts.push_back(
            fi::ChatPart::text_part(corpus_slice(corpus, 2 * turn + 1, 2 * turns)));
        messages.push_back(std::move(assistant));
    }
    return messages;
}

// The boundary contract stated directly: every template cut is an exact token prefix of the
// complete encoding, and an explicit cut snaps to the longest common prefix of its own encoding.
fi::EncodedChat reference_encode(const Tokenizer& tokenizer, const fi::RenderedChat& rendered) {
    fi::EncodedChat encoded;
    encoded.input_ids   = tokenizer.encode(rendered.text);
    const auto relation = [&](std::size_t byte_offset) {
        const std::vector<int> prefix =
            tokenizer.encode(std::string_view(rendered.text).substr(0, byte_offset));
        const auto divergence = std::mismatch(prefix.begin(), prefix.end(),
                                              encoded.input_ids.begin(), encoded.input_ids.end());
        return std::pair<std::size_t, bool>{
            static_cast<std::size_t>(divergence.first - prefix.begin()),
            divergence.first == prefix.end()};
    };
    if (rendered.turn_rewrite_byte_offset) {
        encoded.turn_rewrite_boundary =
            static_cast<std::uint32_t>(relation(*rendered.turn_rewrite_byte_offset).first);
    }
    if (rendered.user_turn_byte_offset && *rendered.user_turn_byte_offset > 0) {
        encoded.user_turn_boundary =
            static_cast<std::uint32_t>(relation(*rendered.user_turn_byte_offset).first);
    }
    for (const fi::PromptBoundaryByteHint& hint : rendered.boundaries) {
        const auto [common, exact] = relation(hint.byte_offset);
        const bool explicit_cut    = hint.kind == ninfer::targets::qwen3_8::PromptBoundaryKind::Explicit;
        if (!explicit_cut && !exact) { throw std::logic_error("template cut is not exact"); }
        const auto depth = static_cast<std::uint32_t>(common);
        if (depth == 0) { continue; }
        if (!encoded.boundaries.empty() && depth <= encoded.boundaries.back().depth) {
            ninfer::targets::qwen3_8::PromptBoundary& last = encoded.boundaries.back();
            if (depth == last.depth) {
                last.publish = last.publish || hint.publish;
                if (last.kind == ninfer::targets::qwen3_8::PromptBoundaryKind::TurnOpener && explicit_cut) {
                    last.kind = ninfer::targets::qwen3_8::PromptBoundaryKind::Explicit;
                }
            }
            continue;
        }
        encoded.boundaries.push_back({.depth = depth, .kind = hint.kind, .publish = hint.publish});
    }
    return encoded;
}

bool same_boundaries(const fi::EncodedChat& lhs, const fi::EncodedChat& rhs) {
    if (lhs.input_ids != rhs.input_ids || lhs.turn_rewrite_boundary != rhs.turn_rewrite_boundary ||
        lhs.user_turn_boundary != rhs.user_turn_boundary ||
        lhs.boundaries.size() != rhs.boundaries.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.boundaries.size(); ++i) {
        if (lhs.boundaries[i].depth != rhs.boundaries[i].depth ||
            lhs.boundaries[i].kind != rhs.boundaries[i].kind ||
            lhs.boundaries[i].publish != rhs.boundaries[i].publish) {
            return false;
        }
    }
    return true;
}

int check_rendered_chat(const Tokenizer& tokenizer, const fi::CompiledChatTemplate& chat_template,
                        const std::string& corpus) {
    constexpr std::size_t kTurns = 24;
    const std::vector<fi::ChatMessage> messages = build_chat(corpus, kTurns);
    fi::ChatRenderOptions options;
    options.enable_thinking = false;
    const fi::RenderedChat rendered = chat_template.render(messages, options);

    auto started                 = std::chrono::steady_clock::now();
    const std::vector<int> plain = tokenizer.encode(rendered.text);
    const double plain_seconds   = seconds_since(started);
    started                      = std::chrono::steady_clock::now();
    const fi::EncodedChat fast   = fi::encode_rendered_chat(tokenizer, rendered);
    const double fast_seconds    = seconds_since(started);
    started                         = std::chrono::steady_clock::now();
    const fi::EncodedChat reference = reference_encode(tokenizer, rendered);
    const double reference_seconds  = seconds_since(started);
    std::cout << "chat " << rendered.text.size() << " bytes -> " << fast.input_ids.size()
              << " tokens, " << rendered.boundaries.size() << " byte hints -> "
              << fast.boundaries.size() << " boundaries; encode " << plain_seconds
              << " s, encode_rendered_chat " << fast_seconds << " s, re-tokenizing reference "
              << reference_seconds << " s\n";
    if (fast.input_ids != plain) {
        std::cerr << "segmented encoding differs from plain encode\n";
        return 1;
    }
    if (!same_boundaries(fast, reference)) {
        std::cerr << "rendered-chat boundaries differ from the re-tokenizing reference\n";
        return 1;
    }
    if (fast.boundaries.size() < kTurns) {
        std::cerr << "expected at least one boundary per turn\n";
        return 1;
    }
    // Boundary resolution must not re-tokenize the prompt per boundary.
    if (fast_seconds > 2.0 * plain_seconds + 0.01) {
        std::cerr << "encode_rendered_chat costs " << fast_seconds / plain_seconds
                  << "x a plain encode\n";
        return 1;
    }
    return 0;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path.string()); }
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

std::string load_corpus() {
    if (const char* path = std::getenv("NINFER_TOKENIZER_CORPUS")) { return read_file(path); }
    std::string corpus;
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(std::string(NINFER_SOURCE_DIR) + "/docs")) {
        if (entry.is_regular_file() && entry.path().extension() == ".md") {
            corpus += read_file(entry.path());
            corpus += "\n<|im_end|>\n<|im_start|>user\n";
        }
    }
    return corpus;
}

// Mixed English, Chinese, code, digits, punctuation runs, special tokens and whitespace shapes.
constexpr std::string_view kSample =
    "<|im_start|>system\nYou are Qwen.<|im_end|>\n<|im_start|>user\n"
    "NInfer 是一个从零实现的 C++/CUDA 推理引擎，目标是单卡最高性能。\n"
    "def prefill(chunk: int = 1024) -> None:\n    return chunk * 2  # 4090 sm_89\n\n"
    "Prices rose 12.5% in 2026; see https://example.com/a_b?c=d&e=f!!  Tabs\t\tand   spaces.\n"
    "<tool_call>{\"name\":\"x\",\"arguments\":{\"k\":[1,2,3]}}</tool_call>"
    "<|im_end|>\n<|im_start|>assistant\n<think>\n";

constexpr int kSampleIds[] = {
    248045,   8678,    198,   2523,    513,   1167,  16451,     13, 248046,    198, 248045,    846,
       198,     45,    623,    776,    220,  98267, 134195, 116033,    351,    992,     14,  77517,
       220, 111892, 104170,   3709, 116096,  96132,  96736,  98644,  99454,   1710,    198,    727,
       829,   7320,  40203,     25,    514,    283,    220,     16,     15,     17,     19,      8,
      1411,   2168,     25,    198,    262,    460,  11540,    348,    220,     17,    220,    653,
       220,     19,     15,     24,     15,   1471,     62,     23,     24,    271,  60605,  15537,
       220,     16,     17,     13,     20,      4,    303,    220,     17,     15,     17,     21,
        26,   1436,   3577,   1074,   8422,    877,  13780,    853,     30,     66,  24600,  65772,
     17572,   2834,    220,  50842,    197,  50711,    256,  12258,     13,    198, 248058,   4754,
       591,   3147,     87,   2129,  15889,  21624,     74,   8631,     16,     11,     17,     11,
        18,     60,   3307, 248059, 248046,    198, 248045,  74455,    198, 248068,    198,
};

} // namespace

int main() {
    const char* dir = std::getenv("NINFER_QWEN3_8_TOKENIZER_DIR");
    if (dir == nullptr) {
        std::cout << "skip: NINFER_QWEN3_8_TOKENIZER_DIR is not set\n";
        return 77;
    }
    const std::filesystem::path root(dir);
    const std::string tokenizer_json  = read_file(root / "tokenizer.json");
    const std::string tokenizer_cfg   = read_file(root / "tokenizer_config.json");
    const std::string generation_json = read_file(root / "generation_config.json");
    const Tokenizer tokenizer(TokenizerResources{tokenizer_json, tokenizer_cfg, generation_json});
    const fi::CompiledChatTemplate chat_template =
        fi::CompiledChatTemplate::resolve(read_file(root / "chat_template.jinja"));

    EncodeOptions options;
    options.parse_added_tokens = true;

    const std::vector<int> sample_ids = tokenizer.encode(kSample, options);
    const std::vector<int> expected(std::begin(kSampleIds), std::end(kSampleIds));
    if (sample_ids != expected) {
        std::cerr << "sample ids differ from the golden sequence (" << sample_ids.size() << " vs "
                  << expected.size() << " ids):";
        for (const int id : sample_ids) { std::cerr << ' ' << id; }
        std::cerr << '\n';
        return 1;
    }
    if (tokenizer.decode(sample_ids) != kSample) {
        std::cerr << "sample does not round-trip through decode\n";
        return 1;
    }

    const std::string corpus = load_corpus();
    std::vector<int> ids;
    double best_seconds = 1e30;
    for (int repeat = 0; repeat < 3; ++repeat) {
        const auto started = std::chrono::steady_clock::now();
        ids                = tokenizer.encode(corpus, options);
        const double secs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        best_seconds = std::min(best_seconds, secs);
    }
    std::cout << "corpus " << corpus.size() << " bytes -> " << ids.size() << " tokens, best "
              << best_seconds << " s = " << static_cast<double>(ids.size()) / best_seconds
              << " tok/s\n";
    if (tokenizer.decode(ids) != corpus) {
        std::cerr << "corpus does not round-trip through decode\n";
        return 1;
    }

    if (const char* golden = std::getenv("NINFER_TOKENIZER_GOLDEN")) {
        const std::filesystem::path path(golden);
        if (std::filesystem::exists(path)) {
            std::ifstream in(path);
            std::vector<int> reference;
            for (int id; in >> id;) { reference.push_back(id); }
            if (reference != ids) {
                std::size_t first = 0;
                while (first < reference.size() && first < ids.size() &&
                       reference[first] == ids[first]) {
                    ++first;
                }
                std::cerr << "corpus ids differ from golden at index " << first << " (golden "
                          << reference.size() << " ids, got " << ids.size() << ")\n";
                return 1;
            }
            std::cout << "corpus ids match golden " << path << '\n';
        } else {
            std::ofstream out(path);
            for (const int id : ids) { out << id << '\n'; }
            std::cout << "wrote golden " << path << '\n';
        }
    }
    if (const int status = check_rendered_chat(tokenizer, chat_template, corpus); status != 0) {
        return status;
    }
    std::cout << "ok\n";
    return 0;
}
