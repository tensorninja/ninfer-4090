#include <ninfer/targets/qwen3_8/frontend.h>
#include <ninfer/targets/qwen3_8/frontend_resources.h>

#include "targets/qwen3_8/impl/frontend/chat_template.h"
#include "targets/qwen3_8/impl/frontend/processor.h"
#include "targets/qwen3_8/impl/frontend/test_access.h"
#include "targets/qwen3_8/impl/frontend/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using Frontend          = ninfer::targets::qwen3_8::Frontend;
using FrontendFactory   = ninfer::targets::qwen3_8::FrontendTestAccess;
using FrontendResources = ninfer::targets::qwen3_8::FrontendResources;
using PublishedOutput   = ninfer::targets::qwen3_8::PublishedOutput;
namespace fi            = ninfer::targets::qwen3_8::frontend_internal;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::string read_file(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error(std::string("failed to open test resource: ") + path); }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

std::string read_template_fixture(const char* path) {
    std::string source = read_file(path);
    if (!source.empty() && source.back() == '\n') { source.pop_back(); }
    return source;
}

const std::string& thinking_toggle_template_source() {
    static const std::string source = read_template_fixture(
        NINFER_SOURCE_DIR "/tests/fixtures/frontend/thinking_toggle_chat_template.jinja");
    return source;
}

const std::string& reasoning_effort_template_source() {
    static const std::string source = read_template_fixture(
        NINFER_SOURCE_DIR "/tests/fixtures/frontend/reasoning_effort_chat_template.jinja");
    return source;
}

const fi::CompiledChatTemplate& thinking_toggle_template() {
    static const fi::CompiledChatTemplate value =
        fi::CompiledChatTemplate::resolve(thinking_toggle_template_source());
    return value;
}

const fi::CompiledChatTemplate& reasoning_effort_template() {
    static const fi::CompiledChatTemplate value =
        fi::CompiledChatTemplate::resolve(reasoning_effort_template_source());
    return value;
}

nlohmann::json added(int id, std::string content, bool special = false) {
    return nlohmann::json{{"id", id},
                          {"content", std::move(content)},
                          {"single_word", false},
                          {"lstrip", false},
                          {"rstrip", false},
                          {"normalized", false},
                          {"special", special}};
}

nlohmann::json decoder_added(std::string content, bool special = false) {
    nlohmann::json value = added(0, std::move(content), special);
    value.erase("id");
    return value;
}

FrontendResources resources(const std::string& chat_template = thinking_toggle_template_source()) {
    FrontendResources result;
    result.chat_template_jinja  = chat_template;
    const nlohmann::json tokens =
        nlohmann::json::array({added(1, "helloST"),
                               added(2, "OPtail"),
                               added(3, "thought</thi"),
                               added(4, "nk>\n\nanswer"),
                               added(6, "<eos>", true),
                               added(7, "<0.0 seconds>"),
                               added(30, "user\n"),
                               added(31, "assistant\n"),
                               added(32, "\n"),
                               added(33, "system\n"),
                               added(40, "plan<tool_"),
                               added(41, "call>\n{\"name\":\"go\",\"arguments\":{}}\n</tool_call>"),
                               added(42, "</think>\n\ndone"),
                               added(248045, "<|im_start|>", true),
                               added(248046, "<|im_end|>", true),
                               added(248053, "<|vision_start|>", true),
                               added(248054, "<|vision_end|>", true),
                               added(248056, "<|image_pad|>", true),
                               added(248057, "<|video_pad|>", true),
                               added(248068, "<think>"),
                               added(248069, "</think>")});
    result.tokenizer_json = nlohmann::json{
        {"model",
         {{"type", "BPE"},
          {"vocab", {{"x", 0}, {"ä", 10}, {"¸", 11}, {"Ń", 12}}},
          {"merges", nlohmann::json::array()}}},
        {"added_tokens",
         tokens}}.dump();

    nlohmann::json decoder = nlohmann::json::object();
    for (const nlohmann::json& token : tokens) {
        nlohmann::json value = token;
        const std::string id = std::to_string(value.at("id").get<int>());
        value.erase("id");
        decoder[id] = std::move(value);
    }
    decoder["248070"]            = decoder_added("<|audio_start|>", true);
    decoder["248071"]            = decoder_added("<|audio_end|>", true);
    decoder["248072"]            = decoder_added("<tts_pad>", true);
    decoder["248073"]            = decoder_added("<tts_text_bos>", true);
    decoder["248074"]            = decoder_added("<tts_text_eod>", true);
    decoder["248075"]            = decoder_added("<tts_text_bos_single>", true);
    decoder["248076"]            = decoder_added("<|audio_pad|>", true);
    result.tokenizer_config_json = nlohmann::json{
        {"add_bos_token", false},
        {"add_prefix_space", false},
        {"pad_token", "<|endoftext|>"},
        {"chat_template", result.chat_template_jinja},
        {"added_tokens_decoder",
         std::move(decoder)}}.dump();
    result.generation_config_json = R"({"eos_token_id":[6]})";
    result.preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":16777216}})";
    result.video_preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":25165824}})";
    return result;
}

std::vector<std::uint8_t> gradient_ppm() {
    std::vector<std::uint8_t> ppm;
    const std::string header = "P6\n64 64\n255\n";
    for (const char byte : header) {
        ppm.push_back(static_cast<std::uint8_t>(static_cast<unsigned char>(byte)));
    }
    for (int index = 0; index < 64 * 64; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

ninfer::PromptInput image_input() {
    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = gradient_ppm();
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "inline.ppm";
    ninfer::ChatMessage message;
    message.role = "user";
    message.parts.push_back(std::move(image));
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    return input;
}

bool near(float actual, float expected) { return std::abs(actual - expected) < 1.0e-6F; }

constexpr std::array<std::uint8_t, 32> kGradientDigest{
    0x1e, 0x8c, 0xd9, 0x22, 0x40, 0xfa, 0x10, 0x62, 0x7b, 0x60, 0x86, 0x8e, 0xe9, 0x66, 0x41, 0xa2,
    0x4d, 0x21, 0xff, 0xc7, 0xe9, 0xa2, 0x2b, 0x34, 0xc0, 0xec, 0x99, 0x84, 0x6c, 0xa9, 0xa4, 0x8a,
};

std::string channel_text(const PublishedOutput& output, ninfer::OutputChannel channel) {
    std::string result;
    for (const ninfer::OutputDelta& delta : output) {
        if (delta.channel == channel) { result += delta.text; }
    }
    return result;
}

fi::ChatMessage chat_message(std::string role, std::string content) {
    fi::ChatMessage message;
    message.role = std::move(role);
    message.parts.push_back(fi::ChatPart::text_part(std::move(content)));
    return message;
}

fi::RenderedChat render_chat(std::vector<fi::ChatMessage> messages,
                             fi::ChatRenderOptions options = {}) {
    return thinking_toggle_template().render(messages, std::move(options));
}

std::string render_chat_text(std::vector<fi::ChatMessage> messages,
                             fi::ChatRenderOptions options = {}) {
    return render_chat(std::move(messages), std::move(options)).text;
}

template <class Callable>
bool throws_invalid_argument(Callable&& callable) {
    try {
        callable();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

template <class Callable>
bool throws_logic_error(Callable&& callable) {
    try {
        callable();
    } catch (const std::logic_error&) { return true; }
    return false;
}

int test_official_tokenizer_merge() {
    const char* configured_root = std::getenv("NINFER_QWEN3_8_27B_HF_DIR");
    if (configured_root == nullptr || *configured_root == '\0') {
        std::cout << "skip: NINFER_QWEN3_8_27B_HF_DIR is not set\n";
        return 0;
    }
    const std::filesystem::path root(configured_root);
    const std::string tokenizer_json =
        read_file((root / "tokenizer.json").string().c_str());
    const std::string tokenizer_config_json =
        read_file((root / "tokenizer_config.json").string().c_str());
    const std::string generation_config_json =
        read_file((root / "generation_config.json").string().c_str());
    const fi::Tokenizer tokenizer({.tokenizer_json         = tokenizer_json,
                                   .tokenizer_config_json  = tokenizer_config_json,
                                   .generation_config_json = generation_config_json});

    constexpr std::array<std::pair<const char*, int>, 7> appended = {{
        {"<|audio_start|>", 248070},
        {"<|audio_end|>", 248071},
        {"<tts_pad>", 248072},
        {"<tts_text_bos>", 248073},
        {"<tts_text_eod>", 248074},
        {"<tts_text_bos_single>", 248075},
        {"<|audio_pad|>", 248076},
    }};
    int failures = check(tokenizer.has_exact_token_domain(248077),
                         "official tokenizer merge left a hole in the token domain");
    for (const auto& [text, id] : appended) {
        const std::vector<int> encoded = tokenizer.encode(text);
        failures += check(encoded == std::vector<int>{id} && tokenizer.is_special_token(id) &&
                              tokenizer.decode_token_bytes(id) == text,
                          "official tokenizer_config.json token did not merge exactly");
    }

    // A client breakpoint inside a merged token snaps back to the last exact token prefix.
    fi::ChatMessage glued = chat_message("user", "hel");
    glued.parts.back().cache_breakpoint = true;
    glued.parts.push_back(fi::ChatPart::text_part("lo world"));
    fi::ChatRenderOptions explicit_options;
    explicit_options.prompt_cache_mode  = ninfer::PromptCacheMode::Explicit;
    const fi::RenderedChat glued_chat   = render_chat({glued}, explicit_options);
    const fi::EncodedChat glued_encoded = fi::encode_rendered_chat(tokenizer, glued_chat);
    const std::size_t cut               = glued_chat.boundaries.front().byte_offset;
    const std::vector<int> cut_prefix =
        tokenizer.encode(std::string_view(glued_chat.text).substr(0, cut));
    const std::vector<int> opener_prefix = tokenizer.encode("<|im_start|>user\n");
    failures += check(glued_chat.boundaries.size() == 1 && glued_encoded.boundaries.size() == 1 &&
                          glued_encoded.boundaries.front().depth == opener_prefix.size() &&
                          glued_encoded.boundaries.front().depth < cut_prefix.size() &&
                          std::equal(opener_prefix.begin(), opener_prefix.end(),
                                     glued_encoded.input_ids.begin()),
                      "explicit cut inside a merged token did not snap to the exact prefix");

    FrontendResources conflicting = resources();
    nlohmann::json config         = nlohmann::json::parse(conflicting.tokenizer_config_json);
    config["added_tokens_decoder"]["248045"]["special"] = false;
    conflicting.tokenizer_config_json                   = config.dump();
    failures += check(
        throws_invalid_argument([&] {
            fi::Tokenizer invalid({.tokenizer_json         = conflicting.tokenizer_json,
                                   .tokenizer_config_json  = conflicting.tokenizer_config_json,
                                   .generation_config_json = conflicting.generation_config_json});
        }),
        "conflicting tokenizer/tokenizer_config added-token definitions were accepted");
    return failures;
}

int test_official_chat_template() {
    int failures = 0;
    failures += check(render_chat_text({chat_message("user", "hello")}) ==
                          "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n",
                      "ordinary user prompt differs from the official template");

    fi::ChatRenderOptions no_generation;
    no_generation.add_generation_prompt = false;
    failures += check(
        render_chat_text({chat_message("system", "  be concise  "), chat_message("user", "hello")},
                         no_generation) == "<|im_start|>system\nbe concise<|im_end|>\n"
                                           "<|im_start|>user\nhello<|im_end|>\n",
        "leading system prompt differs from the official template");
    failures += check(
        render_chat_text({chat_message("system", "first"), chat_message("system", "second"),
                          chat_message("user", "hello")},
                         no_generation) == "<|im_start|>system\nfirst\n\nsecond<|im_end|>\n"
                                           "<|im_start|>user\nhello<|im_end|>\n",
        "contiguous leading system prompts were not merged in order");
    failures += check(render_chat_text({chat_message("system", ""), chat_message("user", "hello")},
                                        no_generation) ==
                          "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n",
                      "empty leading system prompt differs from the official template");

    fi::ChatMessage tool_assistant = chat_message("assistant", "");
    tool_assistant.tool_calls.push_back(
        {.id = "", .name = "f", .arguments_json = R"({"flag":true,"nested":{"x":[1,2]}})"});
    failures +=
        check(render_chat_text({chat_message("user", "hi"), tool_assistant}, no_generation) ==
                  "<|im_start|>user\nhi<|im_end|>\n"
                  "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                  "<tool_call>\n<function=f>\n<parameter=flag>\ntrue\n</parameter>\n"
                  "<parameter=nested>\n{\"x\": [1, 2]}\n</parameter>\n"
                  "</function>\n</tool_call><|im_end|>\n",
              "nested or boolean tool arguments differ from official JSON rendering");

    fi::ChatRenderOptions no_thinking;
    no_thinking.enable_thinking = false;
    failures += check(
        render_chat_text({chat_message("user", "q1"),
                          chat_message("assistant", "<think>\nold thought\n</think>\n\nold answer"),
                          chat_message("user", "q2")},
                         no_thinking) == "<|im_start|>user\nq1<|im_end|>\n"
                                         "<|im_start|>assistant\nold answer<|im_end|>\n"
                                         "<|im_start|>user\nq2<|im_end|>\n"
                                         "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "thinking history differs from the official template");

    fi::ChatMessage lookup = chat_message("assistant", "");
    lookup.tool_calls.push_back(
        {.id = "", .name = "lookup", .arguments_json = R"({"city":"Paris"})"});
    failures += check(
        render_chat_text({chat_message("user", "weather?"), lookup, chat_message("tool", "sunny"),
                          chat_message("tool", "20C"), chat_message("user", "thanks")},
                         no_generation) ==
            "<|im_start|>user\nweather?<|im_end|>\n"
            "<|im_start|>assistant\n<tool_call>\n<function=lookup>\n"
            "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
            "<|im_start|>user\n<tool_response>\nsunny\n</tool_response>\n"
            "<tool_response>\n20C\n</tool_response><|im_end|>\n"
            "<|im_start|>user\nthanks<|im_end|>\n",
        "tool-response grouping differs from the official template");

    fi::ChatRenderOptions tools = no_generation;
    tools.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object","properties":{"flag":{"type":"boolean"}}}}})");
    const std::string tools_rendered =
        render_chat_text({chat_message("system", "be exact"), chat_message("user", "hi")}, tools);
    failures += check(
        tools_rendered.find("\n{\"type\": \"function\", \"function\": {\"name\": \"f\", "
                            "\"description\": \"d\", \"parameters\": {\"type\": \"object\", "
                            "\"properties\": {\"flag\": {\"type\": \"boolean\"}}}}}\n</tools>") !=
                std::string::npos &&
            tools_rendered.ends_with(
                "</IMPORTANT>\n\nbe exact<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"),
        "tools system block differs from official tojson rendering");

    failures += check(throws_invalid_argument([&] {
                          (void)render_chat(
                              {chat_message("developer", "policy"), chat_message("user", "hi")},
                              no_generation);
                      }),
                      "direct developer role was accepted by the model frontend");
    failures +=
        check(throws_invalid_argument([&] {
                  (void)render_chat({chat_message("user", "hi"), chat_message("system", "late")},
                                    no_generation);
              }),
              "late system role was accepted by the model frontend");
    failures += check(throws_invalid_argument([&] {
                          (void)render_chat({chat_message("system", "only")}, no_generation);
                      }),
                      "message history without a user query was accepted");
    failures +=
        check(throws_invalid_argument([&] {
                  (void)render_chat({chat_message("user", "hi"), chat_message("unexpected", "bad")},
                                    no_generation);
              }),
              "unexpected chat role was accepted");
    return failures;
}

int test_reasoning_effort_chat_template() {
    constexpr std::string_view low_instructions =
        "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly "
        "to the conclusion without unnecessary elaboration.";
    constexpr std::string_view xhigh_instructions =
        "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
        "assumptions, consider plausible alternatives, and prioritize correctness, consistency, "
        "and clarity in the final answer.";

    const ninfer::PromptCapabilities toggle_capabilities =
        thinking_toggle_template().capabilities();
    const ninfer::PromptCapabilities effort_capabilities =
        reasoning_effort_template().capabilities();
    int failures = check(toggle_capabilities.enable_thinking &&
                             !toggle_capabilities.reasoning_effort.default_effort &&
                             !toggle_capabilities.reasoning_effort.low &&
                             !toggle_capabilities.reasoning_effort.medium &&
                             !toggle_capabilities.reasoning_effort.xhigh,
                         "thinking-toggle template advertised reasoning effort");
    failures += check(
        effort_capabilities.enable_thinking && effort_capabilities.reasoning_effort.low &&
            effort_capabilities.reasoning_effort.medium &&
            effort_capabilities.reasoning_effort.xhigh &&
            effort_capabilities.reasoning_effort.default_effort == ninfer::ReasoningEffort::XHigh,
        "reasoning-effort template did not advertise its complete capability set");

    const auto render_effort = [](ninfer::ReasoningEffort effort) {
        fi::ChatRenderOptions options;
        options.reasoning_effort = effort;
        return reasoning_effort_template().render({chat_message("user", "hello")}, options).text;
    };
    const std::string tail = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n";
    failures +=
        check(reasoning_effort_template().render({chat_message("user", "hello")}).text ==
                  "<|im_start|>system\n" + std::string(xhigh_instructions) + "<|im_end|>\n" + tail,
              "reasoning-effort template did not apply its xhigh default");
    failures +=
        check(render_effort(ninfer::ReasoningEffort::Low) ==
                  "<|im_start|>system\n" + std::string(low_instructions) + "<|im_end|>\n" + tail,
              "low reasoning effort did not render the official instruction");
    failures += check(render_effort(ninfer::ReasoningEffort::Medium) == tail,
                      "medium reasoning effort injected an instruction");

    fi::ChatRenderOptions disabled;
    disabled.enable_thinking = false;
    failures +=
        check(reasoning_effort_template()
                      .render({chat_message("system", ""), chat_message("user", "hello")}, disabled)
                      .text == "<|im_start|>user\nhello<|im_end|>\n"
                               "<|im_start|>assistant\n<think>\n\n</think>\n\n",
              "disabled thinking did not suppress effort and an empty system turn");
    disabled.reasoning_effort = ninfer::ReasoningEffort::Low;
    failures += check(throws_invalid_argument([&] {
                          (void)reasoning_effort_template().render({chat_message("user", "hello")},
                                                                   disabled);
                      }),
                      "reasoning effort and disabled thinking were accepted together");

    fi::ChatRenderOptions unsupported;
    unsupported.reasoning_effort = ninfer::ReasoningEffort::Low;
    failures += check(throws_invalid_argument([&] {
                          (void)thinking_toggle_template().render({chat_message("user", "hello")},
                                                                  unsupported);
                      }),
                      "thinking-toggle template accepted reasoning effort");

    fi::ChatMessage previous   = chat_message("assistant", "old answer");
    previous.reasoning_content = "old thought";
    fi::ChatRenderOptions no_generation;
    no_generation.add_generation_prompt = false;
    no_generation.reasoning_effort      = ninfer::ReasoningEffort::Medium;
    const std::string preserved =
        reasoning_effort_template()
            .render({chat_message("user", "q1"), previous, chat_message("user", "q2")},
                    no_generation)
            .text;
    failures += check(
        preserved.find("<|im_start|>assistant\n<think>\nold thought\n</think>\n\nold answer") !=
            std::string::npos,
        "reasoning-effort template did not preserve prior thinking by default");
    no_generation.preserve_thinking = false;
    failures +=
        check(reasoning_effort_template()
                      .render({chat_message("user", "q1"), previous, chat_message("user", "q2")},
                              no_generation)
                      .text.find("old thought") == std::string::npos,
              "explicit preserve_thinking=false did not remove prior thinking");

    fi::ChatMessage empty_arguments = chat_message("assistant", "");
    empty_arguments.tool_calls.push_back({.id = "", .name = "f", .arguments_json = ""});
    failures += check(reasoning_effort_template()
                          .render({chat_message("user", "call"), empty_arguments}, no_generation)
                          .text.ends_with("<tool_call>\n<function=f>\n</function>\n"
                                          "</tool_call><|im_end|>\n"),
                      "empty tool arguments did not follow the reasoning-effort template");
    return failures;
}

int test_turn_rewrite_trace() {
    const std::string assistant_header = "<|im_start|>assistant\n";
    fi::ChatMessage first              = chat_message("assistant", "");
    first.reasoning_content            = "first thought";
    first.parts.front().text           = "first answer";
    fi::ChatMessage second             = chat_message("assistant", "");
    second.reasoning_content           = "second thought";
    second.parts.front().text          = "second answer";

    const std::vector<fi::ChatMessage> tool_loop{chat_message("user", "question"), first,
                                                 chat_message("tool", "result one"), second,
                                                 chat_message("tool", "result two")};
    fi::ChatRenderOptions stable;
    stable.prefix_checkpoint_policy = ninfer::PrefixCheckpointPolicy::StableTurn;
    const fi::RenderedChat open     = render_chat(tool_loop, stable);
    const std::size_t first_header  = open.text.find(assistant_header);
    int failures =
        check(first_header != std::string::npos && open.turn_rewrite_byte_offset &&
                  *open.turn_rewrite_byte_offset == first_header + assistant_header.size(),
              "tool loop did not retain its first assistant rewrite boundary");

    fi::ChatRenderOptions preserve;
    preserve.preserve_thinking        = true;
    preserve.prefix_checkpoint_policy = ninfer::PrefixCheckpointPolicy::StableTurn;
    const fi::RenderedChat preserved  = render_chat(tool_loop, preserve);
    failures += check(preserved.turn_rewrite_byte_offset == open.turn_rewrite_byte_offset,
                      "preserve_thinking changed the turn rewrite boundary");

    fi::ChatRenderOptions rolling;
    rolling.prefix_checkpoint_policy = ninfer::PrefixCheckpointPolicy::RollingTool;
    const fi::RenderedChat rolled    = render_chat(tool_loop, rolling);
    const std::size_t rolling_header = rolled.text.rfind(assistant_header);
    failures +=
        check(rolling_header != std::string::npos && rolled.turn_rewrite_byte_offset &&
                  *rolled.turn_rewrite_byte_offset == rolling_header + assistant_header.size() &&
                  *rolled.turn_rewrite_byte_offset > *open.turn_rewrite_byte_offset,
              "rolling tool checkpoint did not advance to the generation opener");
    // Every assistant opener is a lookup boundary; only the one at the rewrite frontier publishes.
    failures += check(
        rolled.boundaries.size() == 3 &&
            std::all_of(rolled.boundaries.begin(), rolled.boundaries.end(),
                        [](const fi::PromptBoundaryByteHint& hint) {
                            return hint.kind == ninfer::targets::qwen3_8::PromptBoundaryKind::TurnOpener;
                        }) &&
            rolled.boundaries[0].byte_offset < rolled.boundaries[1].byte_offset &&
            rolled.boundaries[1].byte_offset < rolled.boundaries[2].byte_offset &&
            !rolled.boundaries[0].publish && !rolled.boundaries[1].publish &&
            rolled.boundaries[2].publish &&
            rolled.boundaries[2].byte_offset == *rolled.turn_rewrite_byte_offset,
        "rolling tool loop did not expose every turn opener with the frontier publishing");
    failures += check(open.boundaries.size() == 1 &&
                          open.boundaries.front().byte_offset == *open.turn_rewrite_byte_offset &&
                          open.boundaries.front().publish,
                      "stable-turn policy retained boundaries past its rewrite frontier");

    const fi::RenderedChat first_roll = render_chat(
        {chat_message("user", "question"), first, chat_message("tool", "result one")}, rolling);
    failures += check(first_roll.turn_rewrite_byte_offset &&
                          *first_roll.turn_rewrite_byte_offset < *rolled.turn_rewrite_byte_offset,
                      "rolling tool checkpoint did not advance with completed tool history");

    std::vector<fi::ChatMessage> next_turn = tool_loop;
    next_turn.push_back(chat_message("user", "next question"));
    const fi::RenderedChat next    = render_chat(next_turn);
    const std::size_t final_header = next.text.rfind(assistant_header);
    failures += check(final_header != std::string::npos && next.turn_rewrite_byte_offset &&
                          *next.turn_rewrite_byte_offset == final_header + assistant_header.size(),
                      "new user turn did not move the rewrite boundary to its generation opener");

    fi::ChatRenderOptions no_generation;
    no_generation.add_generation_prompt = false;
    const fi::RenderedChat no_assistant =
        render_chat({chat_message("user", "question")}, no_generation);
    failures += check(!no_assistant.turn_rewrite_byte_offset,
                      "boundary-less prompt unexpectedly published a rewrite boundary");

    const fi::RenderedChat wrapped =
        render_chat({chat_message("user", "question"), first,
                     chat_message("user", "<tool_response>compat result</tool_response>"), second},
                    no_generation);
    const std::size_t wrapped_first = wrapped.text.find(assistant_header);
    failures +=
        check(wrapped.turn_rewrite_byte_offset &&
                  *wrapped.turn_rewrite_byte_offset == wrapped_first + assistant_header.size(),
              "bare tool-response wrapper incorrectly advanced the real user turn");
    return failures;
}

int test_official_resource_guards() {
    FrontendResources stale_pad     = resources();
    nlohmann::json tokenizer_config = nlohmann::json::parse(stale_pad.tokenizer_config_json);
    tokenizer_config["pad_token"]   = "<|vision_pad|>";
    stale_pad.tokenizer_config_json = tokenizer_config.dump();
    int failures =
        check(throws_invalid_argument([&] { (void)FrontendFactory::create_component(stale_pad); }),
              "stale Unsloth pad-token policy was accepted");

    FrontendResources mismatched       = resources();
    nlohmann::json mismatched_config   = nlohmann::json::parse(mismatched.tokenizer_config_json);
    mismatched_config["chat_template"] = reasoning_effort_template_source();
    mismatched.tokenizer_config_json   = mismatched_config.dump();
    failures +=
        check(throws_invalid_argument([&] { (void)FrontendFactory::create_component(mismatched); }),
              "different standalone and tokenizer-config chat templates were accepted");

    FrontendResources unknown = resources("{{ messages }}");
    failures +=
        check(throws_invalid_argument([&] { (void)FrontendFactory::create_component(unknown); }),
              "unknown chat template was accepted");

    const Frontend effort_frontend =
        FrontendFactory::create_component(resources(reasoning_effort_template_source()), false);
    const ninfer::PromptCapabilities capabilities = effort_frontend.prompt_capabilities();
    failures +=
        check(capabilities.reasoning_effort.low && capabilities.reasoning_effort.medium &&
                  capabilities.reasoning_effort.xhigh &&
                  capabilities.reasoning_effort.default_effort == ninfer::ReasoningEffort::XHigh,
              "Frontend did not expose capabilities from its loaded chat template");

    return failures;
}

int test_text_and_image_prepare(const Frontend& frontend) {
    ninfer::ChatMessage text_message;
    text_message.role = "user";
    text_message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput text_input;
    text_input.messages.push_back(std::move(text_message));
    auto text             = frontend.prepare(std::move(text_input));
    const auto& text_data = FrontendFactory::inspect(text);
    const std::vector<ninfer::TokenId> expected{248045, 30, 0, 248046, 32, 248045, 31, 248068, 32};
    int failures =
        check(text_data.token_ids == expected, "text frontend did not render/tokenize chat");
    failures += check(text_data.identity.turn_rewrite_boundary == 7 &&
                          text_data.starts_in_reasoning && !text_data.has_media(),
                      "text frontend did not preserve prefix/thinking identity");
    failures +=
        check(text_data.position_axis(0).back() == 8 && text_data.position_axis(1).back() == 8 &&
                  text_data.position_axis(2).back() == 8,
              "text frontend did not construct axis-major positions");

    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = gradient_ppm();
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "inline.ppm";
    ninfer::ChatMessage image_message;
    image_message.role = "user";
    image_message.parts.push_back(std::move(image));
    ninfer::PromptInput image_input;
    image_input.messages.push_back(std::move(image_message));
    auto prepared             = frontend.prepare(std::move(image_input));
    const auto& prepared_data = FrontendFactory::inspect(prepared);
    failures += check(prepared_data.has_media() && prepared_data.vision_items.size() == 1,
                      "image frontend did not retain one Vision item");
    if (!prepared_data.vision_items.empty()) {
        const auto& item = prepared_data.vision_items.front();
        failures +=
            check(item.grid.temporal == 1 && item.grid.height == 4 && item.grid.width == 4 &&
                      item.patch_count == 16 && item.content_digest == kGradientDigest &&
                      item.token_spans.size() == 1 && item.token_spans.front().count == 4,
                  "image frontend grid/patch/placeholder geometry is incorrect");
        if (!item.token_spans.empty()) {
            const std::size_t span = item.token_spans.front().begin;
            failures += check(
                prepared_data.position_axis(0)[span] == prepared_data.position_axis(1)[span] &&
                    prepared_data.position_axis(1)[span] == prepared_data.position_axis(2)[span] &&
                    prepared_data.position_axis(1)[span + 2] ==
                        prepared_data.position_axis(1)[span] + 1 &&
                    prepared_data.position_axis(2)[span + 1] ==
                        prepared_data.position_axis(2)[span] + 1,
                "image frontend MRoPE positions are incorrect");
        }
    }
    failures += check(
        prepared_data.patches.size() == 16 * 1536 && prepared_data.prepare.raw_patches == 16 &&
            prepared_data.prepare.vision_tokens == 4 && prepared_data.identity.reusable &&
            prepared_data.identity.turn_rewrite_boundary &&
            *prepared_data.identity.turn_rewrite_boundary < prepared_data.token_ids.size(),
        "image frontend did not own the expected patch payload and identity");
    if (prepared_data.patches.size() == 16 * 1536) {
        failures += check(near(prepared_data.patches[0], -1.0F) &&
                              near(prepared_data.patches[1], 1.0F / 127.5F - 1.0F) &&
                              near(prepared_data.patches[256], -1.0F) &&
                              near(prepared_data.patches[1536], 16.0F / 127.5F - 1.0F),
                          "image frontend patch normalization/order is incorrect");
    }
    return failures;
}

int test_stable_prefix_identity(const Frontend& frontend) {
    using ninfer::targets::qwen3_8::PromptBoundary;
    using ninfer::targets::qwen3_8::PromptBoundaryKind;
    const auto make_input = [](std::string user, std::vector<std::string> tools = {}) {
        ninfer::PromptInput input;
        ninfer::ChatMessage system;
        system.role = "system";
        system.parts.push_back(
            ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text,
                                .text = "x",
                                .media = {}});
        input.messages.push_back(std::move(system));
        ninfer::ChatMessage message;
        message.role = "user";
        message.parts.push_back(ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text,
                                                    .text = std::move(user),
                                                    .media = {}});
        input.messages.push_back(std::move(message));
        input.options.tool_jsons = std::move(tools);
        return input;
    };
    const auto system_tools_boundary = [](const auto& data) -> std::optional<PromptBoundary> {
        for (const PromptBoundary& boundary : data.identity.boundaries) {
            if (boundary.kind == PromptBoundaryKind::SystemTools) { return boundary; }
        }
        return std::nullopt;
    };

    const auto first = frontend.prepare(make_input("x"));
    const auto second = frontend.prepare(make_input("xx"));
    const auto& first_data = FrontendFactory::inspect(first);
    const auto& second_data = FrontendFactory::inspect(second);
    const auto first_stable  = system_tools_boundary(first_data);
    const auto second_stable = system_tools_boundary(second_data);
    int failures = check(first_stable && second_stable && first_stable->publish &&
                             first_stable->depth == second_stable->depth,
                         "same system block did not produce a stable prefix boundary");
    if (first_stable) {
        const std::size_t boundary = first_stable->depth;
        failures += check(boundary < first_data.token_ids.size() &&
                              boundary < second_data.token_ids.size() &&
                              std::equal(first_data.token_ids.begin(),
                                         first_data.token_ids.begin() + boundary,
                                         second_data.token_ids.begin()),
                          "varying user messages changed stable prefix tokens");
        failures += check(first_data.identity.boundaries.front().depth == boundary &&
                              first_data.identity.boundaries.size() == 2 &&
                              first_data.identity.boundaries.back().kind ==
                                  PromptBoundaryKind::TurnOpener &&
                              first_data.identity.boundaries.back().publish &&
                              first_data.identity.boundaries.back().depth ==
                                  first_data.identity.turn_rewrite_boundary &&
                              std::is_sorted(first_data.identity.boundaries.begin(),
                                             first_data.identity.boundaries.end(),
                                             [](const auto& lhs, const auto& rhs) {
                                                 return lhs.depth < rhs.depth;
                                             }),
                          "prompt boundaries omitted the stable prefix or the generation opener");
    }

    ninfer::PromptInput bare;
    ninfer::ChatMessage bare_user;
    bare_user.role = "user";
    bare_user.parts.push_back(ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text,
                                                   .text = "x",
                                                  .media = {}});
    bare.messages.push_back(std::move(bare_user));
    const auto no_stable = frontend.prepare(std::move(bare));
    failures += check(!system_tools_boundary(FrontendFactory::inspect(no_stable)),
                      "prompt without a leading block published a stable prefix");

    const std::vector<std::string> tool_a = {
        R"({"type":"function","function":{"name":"alpha","parameters":{"type":"object"}}})"};
    const std::vector<std::string> tool_b = {
        R"({"type":"function","function":{"name":"beta","parameters":{"type":"object"}}})"};
    fi::ChatRenderOptions tool_a_options;
    tool_a_options.tool_jsons = tool_a;
    fi::ChatRenderOptions tool_b_options;
    tool_b_options.tool_jsons = tool_b;
    const fi::RenderedChat with_tool_a = render_chat(
        {chat_message("system", "x"), chat_message("user", "x")}, tool_a_options);
    const fi::RenderedChat with_tool_b = render_chat(
        {chat_message("system", "x"), chat_message("user", "x")}, tool_b_options);
    const auto stable_bytes = [](const fi::RenderedChat& chat) -> std::optional<std::size_t> {
        if (chat.boundaries.empty() ||
            chat.boundaries.front().kind != PromptBoundaryKind::SystemTools) {
            return std::nullopt;
        }
        return chat.boundaries.front().byte_offset;
    };
    failures += check(stable_bytes(with_tool_a) && stable_bytes(with_tool_b) &&
                          with_tool_a.text.substr(0, *stable_bytes(with_tool_a)) !=
                              with_tool_b.text.substr(0, *stable_bytes(with_tool_b)),
                      "tool definition change did not change stable prefix tokens");

    const fi::RenderedChat rendered = render_chat(
        {chat_message("system", "x"), chat_message("user", "x")});
    const fi::Tokenizer tokenizer({.tokenizer_json = resources().tokenizer_json,
                                   .tokenizer_config_json = resources().tokenizer_config_json,
                                   .generation_config_json = resources().generation_config_json});
    const fi::EncodedChat encoded = fi::encode_rendered_chat(tokenizer, rendered);
    failures += check(stable_bytes(rendered).has_value() &&
                          encoded.boundaries.size() == rendered.boundaries.size() &&
                          std::equal(encoded.boundaries.begin(), encoded.boundaries.end(),
                                     rendered.boundaries.begin(),
                                     [&](const PromptBoundary& token_hint,
                                         const fi::PromptBoundaryByteHint& byte_hint) {
                                         const std::vector<int> prefix = tokenizer.encode(
                                             std::string_view(rendered.text)
                                                 .substr(0, byte_hint.byte_offset));
                                         return token_hint.kind == byte_hint.kind &&
                                                token_hint.publish == byte_hint.publish &&
                                                token_hint.depth == prefix.size() &&
                                                std::equal(prefix.begin(), prefix.end(),
                                                           encoded.input_ids.begin());
                                     }),
                      "template byte boundaries were not independently exact-prefix validated");

    fi::RenderedChat unstable = rendered;
    unstable.boundaries.front().byte_offset = 1;
    failures += check(throws_logic_error(
                          [&] { (void)fi::encode_rendered_chat(tokenizer, unstable); }),
                      "tokenizer-unstable template byte boundary was accepted");

    fi::RenderedChat media_boundary;
    media_boundary.text = "<|image_pad|>suffix";
    media_boundary.turn_rewrite_byte_offset = std::string_view("<|image_pad|>").size();
    media_boundary.boundaries.push_back(
        {PromptBoundaryKind::TurnOpener, *media_boundary.turn_rewrite_byte_offset, true});
    const std::size_t old_boundary = *media_boundary.turn_rewrite_byte_offset;
    fi::adjust_rendered_boundaries_for_replacement(media_boundary, 0, old_boundary,
                                                   old_boundary * 2);
    failures += check(media_boundary.turn_rewrite_byte_offset == old_boundary * 2 &&
                          media_boundary.boundaries.front().byte_offset == old_boundary * 2,
                      "media expansion did not adjust every boundary");
    media_boundary.boundaries.front().byte_offset = 1;
    failures += check(throws_logic_error([&] {
                          fi::adjust_rendered_boundaries_for_replacement(media_boundary, 0,
                                                                         old_boundary,
                                                                         old_boundary * 2);
                      }),
                      "boundary intersecting a media placeholder was accepted");

    fi::ChatRenderOptions low_options;
    low_options.reasoning_effort = ninfer::ReasoningEffort::Low;
    fi::ChatRenderOptions xhigh_options;
    xhigh_options.reasoning_effort = ninfer::ReasoningEffort::XHigh;
    const fi::RenderedChat low = reasoning_effort_template().render(
        {chat_message("user", "hello")}, low_options);
    const fi::RenderedChat xhigh = reasoning_effort_template().render(
        {chat_message("user", "hello")}, xhigh_options);
    failures += check(stable_bytes(low) && stable_bytes(xhigh) &&
                          low.text.substr(0, *stable_bytes(low)) !=
                              xhigh.text.substr(0, *stable_bytes(xhigh)),
                      "reasoning instruction change did not change stable block identity");
    return failures;
}

int test_explicit_breakpoints() {
    using ninfer::targets::qwen3_8::PromptBoundary;
    using ninfer::targets::qwen3_8::PromptBoundaryKind;
    const auto marked = [](fi::ChatMessage message, std::size_t part) {
        message.parts.at(part).cache_breakpoint = true;
        return message;
    };
    // Built from the fixture vocabulary so the rendering also encodes.
    fi::ChatMessage split = chat_message("user", "x");
    split.parts.push_back(fi::ChatPart::text_part("OPtail"));
    const std::vector<fi::ChatMessage> conversation{
        chat_message("system", "x"), marked(split, 0), chat_message("assistant", "helloST"),
        marked(chat_message("user", "OPtail"), 0)};

    const fi::RenderedChat implicit = render_chat(conversation);
    const std::string first_user    = "user\nxOPtail";
    const std::size_t x_end = implicit.text.find(first_user) + std::string_view("user\nx").size();
    // A breakpoint on the final message sits at the generation opener, merging with the
    // implicit turn-opener boundary there.
    const std::size_t next_end = *implicit.turn_rewrite_byte_offset;
    const auto kinds = [](const fi::RenderedChat& chat) {
        std::vector<PromptBoundaryKind> out;
        for (const auto& hint : chat.boundaries) { out.push_back(hint.kind); }
        return out;
    };
    int failures = check(
        kinds(implicit) == std::vector<PromptBoundaryKind>{
                               PromptBoundaryKind::SystemTools, PromptBoundaryKind::Explicit,
                               PromptBoundaryKind::TurnOpener, PromptBoundaryKind::Explicit} &&
            implicit.boundaries[1].byte_offset == x_end && implicit.boundaries[1].publish &&
            implicit.boundaries[3].byte_offset == next_end && implicit.boundaries[3].publish &&
            !implicit.boundaries[2].publish &&
            implicit.text.compare(next_end - std::string_view("<|im_start|>assistant\n").size(),
                                  std::string_view("<|im_start|>assistant\n").size(),
                                  "<|im_start|>assistant\n") == 0,
        "implicit mode did not interleave client breakpoints with template boundaries");
    // Without a generation prompt the final message's breakpoint sits after its close.
    fi::ChatRenderOptions no_generation;
    no_generation.add_generation_prompt = false;
    const fi::RenderedChat closed       = render_chat(conversation, no_generation);
    failures += check(!closed.boundaries.empty() &&
                          closed.boundaries.back().kind == PromptBoundaryKind::Explicit &&
                          closed.boundaries.back().byte_offset == closed.text.size() &&
                          closed.text.ends_with("OPtail<|im_end|>\n"),
                      "final-message breakpoint without a generation prompt left the close");

    fi::ChatRenderOptions explicit_options;
    explicit_options.prompt_cache_mode = ninfer::PromptCacheMode::Explicit;
    const fi::RenderedChat explicit_only = render_chat(conversation, explicit_options);
    failures += check(kinds(explicit_only) == std::vector<PromptBoundaryKind>{
                                                  PromptBoundaryKind::Explicit,
                                                  PromptBoundaryKind::Explicit} &&
                          explicit_only.boundaries[0].byte_offset == x_end &&
                          explicit_only.boundaries[1].byte_offset == next_end &&
                          explicit_only.text == implicit.text,
                      "explicit mode emitted template boundaries or changed the rendering");

    // A breakpoint on a system message reaches the system/tools prefix in explicit mode.
    const fi::RenderedChat system_marked = render_chat(
        {marked(chat_message("system", "x"), 0), chat_message("user", "x")}, explicit_options);
    failures += check(system_marked.boundaries.size() == 1 &&
                          system_marked.boundaries.front().kind == PromptBoundaryKind::Explicit &&
                          system_marked.boundaries.front().byte_offset ==
                              implicit.boundaries.front().byte_offset,
                      "system breakpoint did not land on the system/tools boundary");

    // A breakpoint past the rewrite frontier would leave the lane nothing to rewind to.
    fi::ChatRenderOptions stable;
    stable.prefix_checkpoint_policy = ninfer::PrefixCheckpointPolicy::StableTurn;
    fi::ChatMessage call            = chat_message("assistant", "");
    call.tool_calls.push_back({.id = "c", .name = "f", .arguments_json = "{}"});
    const fi::RenderedChat clipped = render_chat(
        {chat_message("user", "q"), call, marked(chat_message("tool", "result"), 0)}, stable);
    failures += check(std::none_of(clipped.boundaries.begin(), clipped.boundaries.end(),
                                   [&](const fi::PromptBoundaryByteHint& hint) {
                                       return hint.byte_offset > *clipped.turn_rewrite_byte_offset;
                                   }) &&
                          std::none_of(clipped.boundaries.begin(), clipped.boundaries.end(),
                                       [](const fi::PromptBoundaryByteHint& hint) {
                                           return hint.kind == PromptBoundaryKind::Explicit;
                                       }),
                      "breakpoint past the stable-turn rewrite frontier was retained");
    fi::ChatRenderOptions rolling;
    rolling.prefix_checkpoint_policy = ninfer::PrefixCheckpointPolicy::RollingTool;
    const fi::RenderedChat rolled = render_chat(
        {chat_message("user", "q"), call, marked(chat_message("tool", "result"), 0)}, rolling);
    failures += check(!rolled.boundaries.empty() &&
                          rolled.boundaries.back().kind == PromptBoundaryKind::Explicit &&
                          rolled.boundaries.back().publish &&
                          rolled.boundaries.back().byte_offset ==
                              *rolled.turn_rewrite_byte_offset &&
                          std::count_if(rolled.boundaries.begin(), rolled.boundaries.end(),
                                        [](const fi::PromptBoundaryByteHint& hint) {
                                            return hint.kind == PromptBoundaryKind::Explicit;
                                        }) == 1,
                      "final tool result breakpoint did not merge into the rolling frontier");

    const fi::Tokenizer tokenizer({.tokenizer_json = resources().tokenizer_json,
                                   .tokenizer_config_json = resources().tokenizer_config_json,
                                   .generation_config_json = resources().generation_config_json});
    const fi::EncodedChat encoded = fi::encode_rendered_chat(tokenizer, explicit_only);
    const std::vector<int> x_prefix =
        tokenizer.encode(std::string_view(explicit_only.text).substr(0, x_end));
    const std::vector<int> next_prefix =
        tokenizer.encode(std::string_view(explicit_only.text).substr(0, next_end));
    failures += check(encoded.boundaries.size() == 2 &&
                          encoded.boundaries[0].kind == PromptBoundaryKind::Explicit &&
                          encoded.boundaries[0].publish &&
                          encoded.boundaries[0].depth == x_prefix.size() &&
                          encoded.boundaries[1].depth == next_prefix.size() &&
                          std::equal(next_prefix.begin(), next_prefix.end(),
                                     encoded.input_ids.begin()),
                      "token-aligned explicit cuts did not encode as exact prefixes");

    // A cut inside a token has no exact prefix at all with this vocabulary: the boundary is
    // dropped rather than reported at a depth the tokens do not support.
    fi::ChatMessage glued = chat_message("user", "hello");
    glued.parts.push_back(fi::ChatPart::text_part("ST"));
    const fi::RenderedChat glued_chat = render_chat({marked(glued, 0)}, explicit_options);
    const fi::EncodedChat glued_encoded = fi::encode_rendered_chat(tokenizer, glued_chat);
    failures += check(glued_chat.boundaries.size() == 1 && glued_encoded.boundaries.empty() &&
                          glued_encoded.input_ids.size() > 2,
                      "unencodable explicit cut was not dropped");
    return failures;
}

int test_video_prepare(const Frontend& frontend) {
    ninfer::MessagePart video;
    video.kind              = ninfer::MessagePartKind::Media;
    video.media.kind        = ninfer::MediaKind::Video;
    video.media.bytes       = gradient_ppm();
    video.media.media_type  = "image/x-portable-pixmap";
    video.media.source_name = "single-frame.ppm";
    ninfer::ChatMessage message;
    message.role = "user";
    message.parts.push_back(std::move(video));
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));

    auto prepared             = frontend.prepare(std::move(input));
    const auto& prepared_data = FrontendFactory::inspect(prepared);
    int failures = check(prepared_data.vision_items.size() == 1 && prepared_data.has_media(),
                         "video frontend did not retain one Vision item");
    if (!prepared_data.vision_items.empty()) {
        const auto& item = prepared_data.vision_items.front();
        failures +=
            check(item.modality == ninfer::targets::qwen3_8::PromptModality::Video &&
                      item.grid.temporal == 1 && item.grid.height == 4 && item.grid.width == 4 &&
                      item.patch_count == 16 && item.content_digest == kGradientDigest &&
                      item.timestamps.size() == 1 && item.timestamps.front() == 0.0 &&
                      item.token_spans.size() == 1 && item.token_spans.front().count == 4,
                  "video frontend temporal/grid/placeholder metadata is incorrect");
    }
    failures +=
        check(prepared_data.patches.size() == 16 * 1536 &&
                  near(prepared_data.patches[0], prepared_data.patches[256]) &&
                  prepared_data.prepare.raw_patches == 16 &&
                  prepared_data.prepare.vision_tokens == 4 && prepared_data.identity.reusable,
              "video frontend did not duplicate the odd temporal frame correctly");
    return failures;
}

int test_cross_round_stop(const Frontend& frontend) {
    auto prompt = frontend.prepare_tokens({0});
    ninfer::StopPolicy stop;
    stop.strings.push_back(ninfer::StopString{.text = "STOP"});
    auto session = frontend.make_output_session(prompt, stop);

    const auto first_decision =
        session.preview(std::array<ninfer::TokenId, 1>{1}, 2, ninfer::FinishReason::OutputLimit);
    int failures     = check(first_decision.accepted_tokens == 1 && !first_decision.finished(),
                             "cross-round stop ended before the stop string was complete");
    const auto first = session.commit_preview();
    failures += check(channel_text(first, ninfer::OutputChannel::Content) == "hello",
                      "cross-round stop did not retain the ambiguous suffix");

    const auto second_decision =
        session.preview(std::array<ninfer::TokenId, 1>{2}, 1, ninfer::FinishReason::OutputLimit);
    failures += check(second_decision.accepted_tokens == 1 &&
                          second_decision.finish_reason == ninfer::FinishReason::StopString,
                      "cross-round stop did not select the exact terminal token prefix");
    const auto second = session.commit_preview();
    failures += check(second.empty(), "stop marker or same-token suffix leaked to output");
    return failures;
}

int test_same_token_stop_priority(const Frontend& frontend) {
    auto prompt = frontend.prepare_tokens({0});
    ninfer::StopPolicy stop;
    stop.strings = {
        ninfer::StopString{.text = "tail", .include_in_output = true},
        ninfer::StopString{.text = "OPtail"},
        ninfer::StopString{.text = "OP", .include_in_output = true},
    };
    auto session = frontend.make_output_session(prompt, stop);
    const auto decision =
        session.preview(std::array<ninfer::TokenId, 1>{2}, 2, ninfer::FinishReason::OutputLimit);
    int failures      = check(decision.accepted_tokens == 1 &&
                                  decision.finish_reason == ninfer::FinishReason::StopString,
                              "same-token stop strings did not select a terminal prefix");
    const auto output = session.commit_preview();
    failures += check(output.empty(),
                      "same-token stops did not prefer the earliest byte and declaration order");
    return failures;
}

int test_terminal_flush(const Frontend& frontend) {
    auto prompt = frontend.prepare_tokens({0});
    ninfer::StopPolicy stop;
    stop.strings.push_back(ninfer::StopString{.text = "STOP"});
    auto session = frontend.make_output_session(prompt, stop);

    const auto first_decision =
        session.preview(std::array<ninfer::TokenId, 1>{1}, 2, ninfer::FinishReason::OutputLimit);
    int failures     = check(first_decision.accepted_tokens == 1 && !first_decision.finished(),
                             "terminal flush setup unexpectedly finished");
    const auto first = session.commit_preview();
    failures += check(channel_text(first, ninfer::OutputChannel::Content) == "hello",
                      "terminal flush setup did not retain the possible stop suffix");

    const auto terminal = session.preview_terminal(ninfer::FinishReason::Cancelled);
    failures += check(terminal.accepted_tokens == 0 &&
                          terminal.finish_reason == ninfer::FinishReason::Cancelled,
                      "between-round terminal preview returned the wrong decision");
    const auto flushed = session.commit_preview();
    failures += check(channel_text(flushed, ninfer::OutputChannel::Content) == "ST",
                      "between-round terminal preview lost the pending stop suffix");
    return failures;
}

int test_reasoning_split(const Frontend& frontend) {
    ninfer::ChatMessage message;
    message.role = "user";
    message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.add_generation_prompt = true;
    input.options.enable_thinking       = true;
    auto prompt                         = frontend.prepare(std::move(input));
    auto session                        = frontend.make_output_session(prompt, {});
    const std::array<ninfer::TokenId, 2> tokens{3, 4};
    const auto decision = session.preview(tokens, 2, ninfer::FinishReason::OutputLimit);
    int failures        = check(decision.accepted_tokens == 2 &&
                                    decision.finish_reason == ninfer::FinishReason::OutputLimit,
                                "reasoning output did not finish at the requested token limit");
    const auto output   = session.commit_preview();
    failures += check(channel_text(output, ninfer::OutputChannel::Reasoning) == "thought",
                      "reasoning channel did not remove the close marker");
    failures += check(channel_text(output, ninfer::OutputChannel::Content) == "answer",
                      "content channel did not strip the post-thinking separator");
    failures += check(session.reasoning_tokens() == 2,
                      "reasoning token usage did not count accepted reasoning tokens exactly");
    return failures;
}

// A tripped repetition guard licenses the round against its own extent with RepetitionCycle, the
// executor route that first retired a live engine when the decoder rejected that reason.
int test_repetition_cycle_limit(const Frontend& frontend) {
    ninfer::ChatMessage message;
    message.role = "user";
    message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.add_generation_prompt = true;
    input.options.enable_thinking       = true;
    auto prompt                         = frontend.prepare(std::move(input));
    auto session                        = frontend.make_output_session(prompt, {});
    const std::array<ninfer::TokenId, 2> tokens{3, 4};
    const auto decision = session.preview(tokens, 2, ninfer::FinishReason::RepetitionCycle);
    int failures        = check(decision.accepted_tokens == 2 && decision.finished() &&
                                    decision.finish_reason == ninfer::FinishReason::RepetitionCycle,
                                "repetition-cycle round did not terminalize at its own extent");
    const auto output   = session.commit_preview();
    failures += check(channel_text(output, ninfer::OutputChannel::Reasoning) == "thought" &&
                          channel_text(output, ninfer::OutputChannel::Content) == "answer",
                      "repetition-cycle terminal did not flush the decoder channels");
    return failures;
}

ninfer::PromptInput thinking_prompt_input() {
    ninfer::ChatMessage message;
    message.role = "user";
    message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.add_generation_prompt = true;
    input.options.enable_thinking       = true;
    return input;
}

constexpr std::string_view kToolRegion =
    "<tool_call>\n{\"name\":\"go\",\"arguments\":{}}\n</tool_call>";

// A reasoning delta is on the wire as soon as it is appended, so the tool
// region has to be withheld until the end of the generation decides what it
// was. Publishing it as reasoning would make the call unrecoverable.
int test_unclosed_thinking_tool_call_becomes_content(const Frontend& frontend) {
    auto prompt  = frontend.prepare(thinking_prompt_input());
    auto session = frontend.make_output_session(prompt, {});

    const auto opened =
        session.preview(std::array<ninfer::TokenId, 1>{40}, 2, ninfer::FinishReason::OutputLimit);
    int failures    = check(opened.accepted_tokens == 1 && !opened.finished(),
                            "opening the tool region ended generation early");
    const auto held = session.commit_preview();
    failures += check(channel_text(held, ninfer::OutputChannel::Reasoning) == "plan",
                      "reasoning before the tool marker was not published");
    failures += check(channel_text(held, ninfer::OutputChannel::Content).empty(),
                      "a partial tool marker was published before its outcome was known");

    const auto closed =
        session.preview(std::array<ninfer::TokenId, 1>{41}, 1, ninfer::FinishReason::OutputLimit);
    failures += check(closed.accepted_tokens == 1 && closed.finished(),
                      "the tool region did not finish at the token limit");
    const auto output = session.commit_preview();
    failures += check(channel_text(output, ninfer::OutputChannel::Reasoning).empty(),
                      "the withheld tool region leaked onto the reasoning channel");
    failures += check(channel_text(output, ninfer::OutputChannel::Content) == kToolRegion,
                      "an unclosed thought did not hand its tool call to the content channel");
    return failures;
}

// The mirror case, and the reason the region is held rather than promoted on
// sight: a model that merely writes about a call and then closes its thought
// must keep every byte of it as reasoning.
int test_closed_thinking_keeps_tool_call_as_reasoning(const Frontend& frontend) {
    auto prompt  = frontend.prepare(thinking_prompt_input());
    auto session = frontend.make_output_session(prompt, {});

    const std::array<ninfer::TokenId, 3> tokens{40, 41, 42};
    const auto decision = session.preview(tokens, 3, ninfer::FinishReason::OutputLimit);
    int failures        = check(decision.accepted_tokens == 3 && decision.finished(),
                                "closed thinking did not finish at the token limit");
    const auto output   = session.commit_preview();
    failures += check(channel_text(output, ninfer::OutputChannel::Reasoning) ==
                          std::string("plan") + std::string(kToolRegion),
                      "a closed thought did not keep its tool region as reasoning");
    failures += check(channel_text(output, ninfer::OutputChannel::Content) == "done",
                      "content after a closed thought was lost");
    return failures;
}

int test_utf8_and_hidden_eos(const Frontend& frontend) {
    auto prompt             = frontend.prepare_tokens({0});
    auto session            = frontend.make_output_session(prompt, {});
    int failures            = 0;
    std::uint32_t remaining = 4;
    for (const ninfer::TokenId token : {10, 11}) {
        const auto decision = session.preview(std::array<ninfer::TokenId, 1>{token}, remaining,
                                              ninfer::FinishReason::OutputLimit);
        failures += check(decision.accepted_tokens == 1 && !decision.finished(),
                          "partial UTF-8 token unexpectedly ended generation");
        const auto output = session.commit_preview();
        remaining -= decision.accepted_tokens;
        failures += check(output.empty(), "partial UTF-8 codepoint was published");
    }
    const auto complete_decision = session.preview(std::array<ninfer::TokenId, 1>{12}, remaining,
                                                   ninfer::FinishReason::OutputLimit);
    failures += check(complete_decision.accepted_tokens == 1 && !complete_decision.finished(),
                      "complete UTF-8 token unexpectedly ended generation");
    const auto complete = session.commit_preview();
    failures += check(channel_text(complete, ninfer::OutputChannel::Content) == "中",
                      "UTF-8 codepoint was not published when complete");

    auto eos_prompt         = frontend.prepare_tokens({0});
    auto eos_session        = frontend.make_output_session(eos_prompt, {});
    const auto eos_decision = eos_session.preview(std::array<ninfer::TokenId, 1>{6}, 2,
                                                  ninfer::FinishReason::OutputLimit);
    failures += check(eos_decision.accepted_tokens == 1 &&
                          eos_decision.finish_reason == ninfer::FinishReason::StopToken,
                      "default EOS token did not end generation");
    const auto eos = eos_session.commit_preview();
    failures += check(eos.empty(), "default EOS token was published");

    auto raw_prompt  = frontend.prepare_tokens({0});
    auto raw_session = frontend.make_output_session(
        raw_prompt, {}, ninfer::OutputOptions{.raw = true, .preserve_special_tokens = false});
    const auto raw_eos_decision = raw_session.preview(std::array<ninfer::TokenId, 1>{6}, 2,
                                                      ninfer::FinishReason::OutputLimit);
    failures += check(raw_eos_decision.accepted_tokens == 1 &&
                          raw_eos_decision.finish_reason == ninfer::FinishReason::StopToken,
                      "raw EOS token did not end generation");
    const auto raw_eos = raw_session.commit_preview();
    failures += check(channel_text(raw_eos, ninfer::OutputChannel::Content) == "<eos>",
                      "raw output did not preserve the terminal special token");
    return failures;
}

// GPT-2's byte-level alphabet, which the checkpoint tokenizer uses: printable Latin-1 bytes stand
// for themselves, and every other byte for the next code point from U+0100 on.
std::string byte_level_symbol(unsigned char byte) {
    std::uint32_t codepoint = 0;
    std::uint32_t shifted   = 0x100;
    for (std::uint32_t value = 0; value <= byte; ++value) {
        const bool printable =
            (value >= 0x21 && value <= 0x7e) || (value >= 0xa1 && value <= 0xac) || value >= 0xae;
        codepoint = printable ? value : shifted++;
    }
    if (codepoint < 0x80) { return std::string(1, static_cast<char>(codepoint)); }
    return {static_cast<char>(0xc0 | (codepoint >> 6)),
            static_cast<char>(0x80 | (codepoint & 0x3f))};
}

struct GeneratedBytesCase {
    const char* name;
    std::vector<unsigned char> bytes;
    std::string expected;
};

constexpr ninfer::TokenId kByteToken = 1000; // + the byte
constexpr ninfer::TokenId kCaseToken = 2000; // + the case index: all of its bytes in one token

std::string replacements(std::size_t count) {
    std::string text;
    for (std::size_t index = 0; index < count; ++index) { text += "\xef\xbf\xbd"; }
    return text;
}

// The fixture tokenizer with a vocabulary of one token per byte and one token per case.
FrontendResources generated_bytes_resources(const std::vector<GeneratedBytesCase>& cases) {
    FrontendResources result = resources();
    nlohmann::json tokenizer = nlohmann::json::parse(result.tokenizer_json);
    nlohmann::json vocab     = nlohmann::json::object();
    for (std::uint32_t byte = 0; byte < 256; ++byte) {
        vocab[byte_level_symbol(static_cast<unsigned char>(byte))] =
            kByteToken + static_cast<ninfer::TokenId>(byte);
    }
    for (std::size_t index = 0; index < cases.size(); ++index) {
        std::string symbols;
        for (const unsigned char byte : cases[index].bytes) { symbols += byte_level_symbol(byte); }
        vocab[symbols] = kCaseToken + static_cast<ninfer::TokenId>(index);
    }
    tokenizer["model"]["vocab"] = std::move(vocab);
    result.tokenizer_json       = tokenizer.dump();
    return result;
}

// Content published for `tokens` in rounds of `round_size`, under a budget that outlasts them so
// that termination flushes nothing; nullopt when a round is not accepted whole or throws.
std::optional<std::string> streamed_content(const Frontend& frontend,
                                            const std::vector<ninfer::TokenId>& tokens,
                                            std::size_t round_size) {
    try {
        const auto prompt = frontend.prepare_tokens({kByteToken + 'x'});
        auto session      = frontend.make_output_session(prompt, {});
        auto remaining    = static_cast<std::uint32_t>(tokens.size() + 1);
        std::string content;
        for (std::size_t begin = 0; begin < tokens.size(); begin += round_size) {
            const std::size_t count = std::min(round_size, tokens.size() - begin);
            const auto decision =
                session.preview(std::span<const ninfer::TokenId>(tokens).subspan(begin, count),
                                remaining, ninfer::FinishReason::OutputLimit);
            if (decision.accepted_tokens != count || decision.finished()) { return std::nullopt; }
            content += channel_text(session.commit_preview(), ninfer::OutputChannel::Content);
            remaining -= static_cast<std::uint32_t>(count);
        }
        return content;
    } catch (const std::exception& error) {
        std::cerr << "generated bytes threw: " << error.what() << '\n';
        return std::nullopt;
    }
}

// The model may emit any byte-level tokens, so the generated bytes need not be well-formed UTF-8.
// Every maximal subpart of an ill-formed sequence is published as one U+FFFD. The cases are the
// examples of the Unicode Standard, section 3.9; their expected text is CPython's
// bytes.decode("utf-8", "replace"). The text must not depend on whether the bytes arrive one per
// round, one per token of a single round, or all in one token.
int test_ill_formed_utf8_is_replaced() {
    const std::vector<GeneratedBytesCase> cases{
        {"non-shortest forms",
         {0xc0, 0xaf, 0xe0, 0x80, 0xbf, 0xf0, 0x81, 0x82, 0x41},
         replacements(8) + "A"},
        {"surrogates",
         {0xed, 0xa0, 0x80, 0xed, 0xbf, 0xbf, 0xed, 0xaf, 0x41},
         replacements(8) + "A"},
        {"other ill-formed sequences",
         {0xf4, 0x91, 0x92, 0x93, 0xff, 0x41, 0x80, 0xbf, 0x42},
         replacements(5) + "A" + replacements(2) + "B"},
        {"truncated sequences",
         {0xe1, 0x80, 0xe2, 0xf0, 0x91, 0x92, 0xf1, 0xbf, 0x41},
         replacements(4) + "A"},
        {"ill-formed between ASCII",
         {0x61, 0xf1, 0x80, 0x80, 0xe1, 0x80, 0xc2, 0x62, 0x80, 0x63, 0x80, 0xbf, 0x64},
         "a" + replacements(3) + "b" + replacements(1) + "c" + replacements(2) + "d"},
        {"well-formed around ill-formed",
         {0xe4, 0xb8, 0xad, 0x80, 0xf0, 0x9f, 0x98, 0x80, 0xe4, 0xe4, 0xb8, 0xad},
         "\xe4\xb8\xad" + replacements(1) + "\xf0\x9f\x98\x80" + replacements(1) + "\xe4\xb8\xad"},
    };
    const Frontend frontend = FrontendFactory::create_component(generated_bytes_resources(cases));
    int failures            = 0;
    for (std::size_t index = 0; index < cases.size(); ++index) {
        const GeneratedBytesCase& item = cases[index];
        std::vector<ninfer::TokenId> byte_tokens;
        for (const unsigned char byte : item.bytes) { byte_tokens.push_back(kByteToken + byte); }
        const std::vector<ninfer::TokenId> case_token{kCaseToken +
                                                      static_cast<ninfer::TokenId>(index)};
        const std::array<std::pair<const char*, std::optional<std::string>>, 3> splits{{
            {"one byte per round", streamed_content(frontend, byte_tokens, 1)},
            {"one round", streamed_content(frontend, byte_tokens, byte_tokens.size())},
            {"one token", streamed_content(frontend, case_token, 1)},
        }};
        for (const auto& [split, content] : splits) {
            if (content != item.expected) {
                std::cerr << item.name << ", " << split
                          << ": the published text is not the U+FFFD substitution\n";
                ++failures;
            }
        }
    }

    // A budget that ends inside a code point: 41 E1 80 decodes to "A" and one U+FFFD.
    const auto prompt       = frontend.prepare_tokens({kByteToken + 'x'});
    auto session            = frontend.make_output_session(prompt, {});
    std::uint32_t remaining = 3;
    std::string content;
    ninfer::FinishReason reason = ninfer::FinishReason::None;
    for (const int byte : {0x41, 0xe1, 0x80}) {
        const auto decision = session.preview(std::array<ninfer::TokenId, 1>{kByteToken + byte},
                                              remaining, ninfer::FinishReason::OutputLimit);
        content += channel_text(session.commit_preview(), ninfer::OutputChannel::Content);
        remaining -= decision.accepted_tokens;
        reason = decision.finish_reason;
    }
    failures +=
        check(reason == ninfer::FinishReason::OutputLimit && content == "A" + replacements(1),
              "a code point cut off by the budget was not published as one U+FFFD");
    return failures;
}

int test_disabled_vision() {
    const Frontend frontend = FrontendFactory::create_component(resources(), false);
    int failures = check(throws_invalid_argument([&] { (void)frontend.prepare(image_input()); }),
                         "Vision-disabled frontend accepted media during prepare");
    failures += check(throws_invalid_argument([&] { (void)frontend.count_tokens(image_input()); }),
                      "Vision-disabled frontend accepted media during token counting");

    ninfer::ChatMessage message;
    message.role = "user";
    message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    failures += check(frontend.prepare(std::move(input)).summary().prompt_tokens != 0,
                      "Vision-disabled frontend rejected a text prompt");
    return failures;
}

} // namespace

int main() {
    const FrontendResources owned = resources();
    const Frontend frontend       = FrontendFactory::create_component(owned);
    int failures                  = 0;
    failures += test_official_tokenizer_merge();
    failures += test_official_chat_template();
    failures += test_reasoning_effort_chat_template();
    failures += test_turn_rewrite_trace();
    failures += test_official_resource_guards();
    failures += test_text_and_image_prepare(frontend);
    failures += test_stable_prefix_identity(frontend);
    failures += test_explicit_breakpoints();
    failures += test_video_prepare(frontend);
    failures += test_cross_round_stop(frontend);
    failures += test_same_token_stop_priority(frontend);
    failures += test_terminal_flush(frontend);
    failures += test_reasoning_split(frontend);
    failures += test_repetition_cycle_limit(frontend);
    failures += test_unclosed_thinking_tool_call_becomes_content(frontend);
    failures += test_closed_thinking_keeps_tool_call_as_reasoning(frontend);
    failures += test_utf8_and_hidden_eos(frontend);
    failures += test_ill_formed_utf8_is_replaced();
    failures += test_disabled_vision();
    return failures == 0 ? 0 : 1;
}
