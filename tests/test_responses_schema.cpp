// Contract tests for the implemented OpenAI Responses Core surface: typed
// input Items, explicit capability rejection, terminal objects, and semantic
// SSE event sequencing.

#include "serve/generation_service.h"
#include "serve/responses_schema.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;
using namespace ninfer::serve;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

RequestLimits limits() {
    RequestLimits value;
    value.default_max_tokens = 256;
    return value;
}

ninfer::PromptCapabilities effort_capabilities() {
    ninfer::PromptCapabilities capabilities;
    capabilities.enable_thinking                 = true;
    capabilities.reasoning_effort.low            = true;
    capabilities.reasoning_effort.medium         = true;
    capabilities.reasoning_effort.xhigh          = true;
    capabilities.reasoning_effort.default_effort = ninfer::ReasoningEffort::XHigh;
    return capabilities;
}

bool throws_api(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const ApiException&) { return true; } catch (...) {
        return false;
    }
    return false;
}

std::string api_code(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const ApiException& error) { return error.error().code; } catch (...) {
        return "wrong_exception";
    }
    return {};
}

Json parse_event(const std::string& event) {
    const std::size_t newline = event.find('\n');
    if (!event.starts_with("event: ") || newline == std::string::npos || !event.ends_with("\n\n")) {
        throw std::runtime_error("invalid SSE framing");
    }
    const std::string type   = event.substr(7, newline - 7);
    const std::string prefix = "data: ";
    if (event.compare(newline + 1, prefix.size(), prefix) != 0) {
        throw std::runtime_error("missing SSE data field");
    }
    const std::size_t begin = newline + 1 + prefix.size();
    Json payload            = Json::parse(event.substr(begin, event.size() - begin - 2));
    if (payload.at("type") != type) { throw std::runtime_error("SSE type mismatch"); }
    return payload;
}

int test_basic_request() {
    const Json body                = {{"model", "qwen3.8-27b"},
                                      {"input", "hello"},
                                      {"instructions", "be concise"},
                                      {"previous_response_id", "resp_previous"},
                                      {"prompt_cache_key", "session_123"},
                                      {"max_output_tokens", 64},
                                      {"temperature", 0.3},
                                      {"top_p", 0.8},
                                      {"reasoning", Json{{"effort", "medium"}, {"summary", "detailed"}}},
                                      {"metadata", Json{{"trace", "abc"}}}};
    const ResponsesRequest request = parse_responses_request(body, limits());
    int failures                   = 0;
    failures += check(request.generation.model == "qwen3.8-27b", "model parsed");
    failures += check(request.input_turns.size() == 1 && request.input_turns[0].role == "user" &&
                          request.input_turns[0].content[0].text == "hello",
                      "string input normalized to a user turn");
    failures += check(request.input_items[0].at("type") == "message" &&
                          request.input_items[0].at("content")[0].at("type") == "input_text" &&
                          request.input_items[0].contains("id"),
                      "canonical input Item built with an id");
    failures += check(request.instructions && *request.instructions == "be concise",
                      "instructions parsed separately");
    failures +=
        check(request.previous_response_id && *request.previous_response_id == "resp_previous",
              "previous response id parsed");
    failures += check(request.generation.max_tokens == 64 && request.generation.max_tokens_set,
                      "max_output_tokens reaches generation request");
    failures += check(request.generation.reasoning_effort == RequestedReasoningEffort::Medium,
                      "medium reasoning effort was not parsed");
    failures +=
        check(request.reasoning_summary == "detailed" && request.prompt_cache_key == "session_123",
               "reasoning summary and prompt cache key retained");
    failures += check(request.generation.prompt_cache_routing_hint == "session_123" &&
                          to_request_options(request.generation, ServeOptions{})
                                  .execution.routing_hint == "session_123",
                      "Responses prompt cache key did not reach execution routing metadata");
    failures += check(request.store && !request.stream, "Responses defaults applied");
    ResponsesRequest composed = request;
    ChatTurn previous;
    previous.role = "assistant";
    ContentPart previous_text;
    previous_text.kind = ContentKind::Text;
    previous_text.text = "old answer";
    previous.content.push_back(std::move(previous_text));
    compose_responses_generation_messages(composed, {previous});
    failures += check(composed.generation.messages.size() == 3 &&
                          composed.generation.messages[0].role == "developer" &&
                          composed.generation.messages[1].content[0].text == "old answer" &&
                          composed.generation.messages[2].content[0].text == "hello",
                      "instructions, previous context, and current input composed in order");
    return failures;
}

int test_reasoning_effort() {
    const Json base = {{"model", "m"}, {"input", "hello"}, {"max_output_tokens", 32}};
    int failures    = 0;

    for (const auto& [wire, expected] :
         std::array<std::pair<const char*, RequestedReasoningEffort>, 7>{
             {{"none", RequestedReasoningEffort::None},
              {"minimal", RequestedReasoningEffort::Minimal},
              {"low", RequestedReasoningEffort::Low},
              {"medium", RequestedReasoningEffort::Medium},
              {"high", RequestedReasoningEffort::High},
              {"xhigh", RequestedReasoningEffort::XHigh},
              {"max", RequestedReasoningEffort::Max}}}) {
        Json body         = base;
        body["reasoning"] = Json{{"effort", wire}};
        failures +=
            check(parse_responses_request(body, limits()).generation.reasoning_effort == expected,
                  std::string("Responses did not accept protocol effort ") + wire);
    }

    Json low                            = base;
    low["reasoning"]                    = Json{{"effort", "low"}};
    const GenerationRequest low_request = parse_responses_request(low, limits()).generation;
    const ResolvedPromptSemantics low_semantics =
        resolve_prompt_semantics(low_request, ServeOptions{}, effort_capabilities());
    failures += check(low_semantics.enable_thinking &&
                          low_semantics.reasoning_effort == ninfer::ReasoningEffort::Low,
                      "Responses low effort did not resolve through template capabilities");

    Json none                                    = base;
    none["reasoning"]                            = Json{{"effort", "none"}};
    const ResolvedPromptSemantics none_semantics = resolve_prompt_semantics(
        parse_responses_request(none, limits()).generation, ServeOptions{}, effort_capabilities());
    failures += check(!none_semantics.enable_thinking && !none_semantics.reasoning_effort,
                      "Responses none effort did not disable thinking");

    Json high                            = base;
    high["reasoning"]                    = Json{{"effort", "high"}};
    const GenerationRequest high_request = parse_responses_request(high, limits()).generation;
    failures += check(api_code([&] {
                          (void)resolve_prompt_semantics(high_request, ServeOptions{},
                                                         effort_capabilities());
                      }) == "reasoning_effort_not_supported",
                      "Responses high effort bypassed template capability validation");

    Json invalid         = base;
    invalid["reasoning"] = Json{{"effort", "ultra"}};
    failures += check(throws_api([&] { (void)parse_responses_request(invalid, limits()); }),
                      "unknown Responses reasoning effort was accepted");
    invalid["reasoning"] = Json{{"effort", 1}};
    failures += check(throws_api([&] { (void)parse_responses_request(invalid, limits()); }),
                      "non-string Responses reasoning effort was accepted");
    return failures;
}

int test_preserve_thinking_options_and_inheritance() {
    const Json base = {{"model", "m"}, {"input", "hello"}};
    int failures    = 0;

    Json kwargs                    = base;
    kwargs["chat_template_kwargs"] = Json{{"preserve_thinking", true}};
    ResponsesRequest request       = parse_responses_request(kwargs, limits());
    failures += check(request.generation.preserve_thinking == true,
                      "Responses chat_template_kwargs preserve_thinking parsed");

    ResponsesRequest inherited = parse_responses_request(base, limits());
    inherit_responses_preserve_thinking(inherited, true);
    failures += check(inherited.generation.preserve_thinking == true &&
                          !inherited.generation.preserve_thinking_semantic_change,
                      "Responses child did not inherit parent preserve_thinking");

    ResponsesRequest unchanged = request;
    inherit_responses_preserve_thinking(unchanged, true);
    failures += check(!unchanged.generation.preserve_thinking_semantic_change,
                      "equal explicit preserve value marked a semantic change");

    Json explicit_false                 = base;
    explicit_false["preserve_thinking"] = false;
    ResponsesRequest changed            = parse_responses_request(explicit_false, limits());
    inherit_responses_preserve_thinking(changed, true);
    failures += check(changed.generation.preserve_thinking == false &&
                          changed.generation.preserve_thinking_semantic_change,
                      "explicit Responses preserve branch was not marked");

    Json conflict                 = kwargs;
    conflict["preserve_thinking"] = false;
    failures += check(api_code([&] { (void)parse_responses_request(conflict, limits()); }) ==
                          "conflicting_template_option",
                      "Responses conflicting preserve values were accepted");

    Json unknown                    = base;
    unknown["chat_template_kwargs"] = Json{{"unknown", 1}};
    failures += check(api_code([&] { (void)parse_responses_request(unknown, limits()); }) ==
                          "chat_template_option_not_supported",
                      "Responses unknown chat template option was accepted");
    return failures;
}

int test_typed_items_and_tools() {
    const Json function = {{"type", "function"},
                           {"name", "weather"},
                           {"description", "Get weather"},
                           {"parameters", Json{{"type", "object"}, {"properties", Json::object()}}},
                           {"strict", false}};
    const Json body     = {
        {"model", "qwen3.8-27b"},
        {"input",
             Json::array(
             {Json{{"id", "rs_old"},
                       {"type", "reasoning"},
                       {"summary",
                        Json::array({Json{{"type", "summary_text"}, {"text", "need tools"}}})}},
                  Json{{"id", "fc_old_1"},
                       {"type", "function_call"},
                       {"call_id", "call_1"},
                       {"name", "weather"},
                       {"arguments", R"({"city":"Paris"})"}},
                  Json{{"id", "fc_old_2"},
                       {"type", "function_call"},
                       {"call_id", "call_2"},
                       {"name", "weather"},
                       {"arguments", R"({"city":"Rome"})"}},
                  Json{{"id", "fco_old"},
                       {"type", "function_call_output"},
                       {"call_id", "call_1"},
                       {"output", Json::array({Json{{"type", "input_text"}, {"text", R"({"temp":20})"}},
                                               Json{{"type", "input_image"},
                                                    {"image_url", "data:image/png;base64,AA=="}}})}},
                  Json{{"type", "message"},
                       {"role", "user"},
                       {"content",
                        Json::array({Json{{"type", "input_image"},
                                          {"image_url", "data:image/png;base64,AA=="},
                                          {"detail", "auto"}},
                                     Json{{"type", "input_text"}, {"text", "describe"}}})}}})},
        {"tools", Json::array({function})},
        {"tool_choice", "auto"},
        {"parallel_tool_calls", true},
        {"max_output_tokens", 32}};

    const ResponsesRequest request = parse_responses_request(body, limits());
    int failures                   = 0;
    failures += check(request.input_turns.size() == 3, "typed Items grouped into three turns");
    failures += check(request.input_turns[0].role == "assistant" &&
                          request.input_turns[0].reasoning_content == "need tools" &&
                          request.input_turns[0].tool_calls.size() == 2,
                      "reasoning and adjacent function calls grouped into one assistant turn");
    failures += check(request.input_turns[1].role == "tool" &&
                          request.input_turns[1].tool_call_id == "call_1" &&
                          request.input_turns[1].content.size() == 2 &&
                          request.input_turns[1].content[0].kind == ContentKind::Text &&
                          request.input_turns[1].content[1].kind == ContentKind::Image,
                      "rich function output preserved in tool turn order");
    failures += check(request.input_items[3].at("output").is_array() &&
                          request.input_items[3].at("output")[0].at("type") == "input_text" &&
                          request.input_items[3].at("output")[1].at("detail") == "auto",
                      "rich function output canonicalized without reordering");
    failures += check(request.input_turns[2].content[0].kind == ContentKind::Image,
                      "input image translated to media part");
    failures += check(request.generation.tools.size() == 1 &&
                          request.generation.tools[0].name == "weather" &&
                          !request.generation.tools[0].strict,
                      "flat Responses function converted to internal tool");
    const Json nested = Json::parse(request.generation.tools[0].definition_json);
    failures += check(nested.at("function").at("name") == "weather",
                      "Qwen prompt receives normalized nested function definition");
    return failures;
}

int test_explicit_rejections() {
    const Json base = {{"model", "qwen3.8-27b"}, {"input", "hello"}, {"max_output_tokens", 32}};
    int failures    = 0;

    Json strict     = base;
    strict["tools"] = Json::array({Json{
        {"type", "function"}, {"name", "f"}, {"parameters", Json::object()}, {"strict", true}}});
    failures += check(api_code([&] { (void)parse_responses_request(strict, limits()); }) ==
                          "strict_tools_not_supported",
                      "strict tools rejected explicitly");

    Json required           = base;
    required["tools"]       = Json::array({Json{{"type", "function"}, {"name", "f"}}});
    required["tool_choice"] = "required";
    failures += check(api_code([&] { (void)parse_responses_request(required, limits()); }) ==
                          "tool_choice_not_supported",
                      "required tool choice rejected explicitly");

    Json structured    = base;
    structured["text"] = Json{{"format", Json{{"type", "json_schema"}}}};
    failures += check(api_code([&] { (void)parse_responses_request(structured, limits()); }) ==
                          "structured_outputs_not_supported",
                      "structured output rejected");

    Json background          = base;
    background["background"] = true;
    failures += check(api_code([&] { (void)parse_responses_request(background, limits()); }) ==
                          "background_not_supported",
                      "background rejected");

    Json invalid_cache_key                = base;
    invalid_cache_key["prompt_cache_key"] = 42;
    failures +=
        check(throws_api([&] { (void)parse_responses_request(invalid_cache_key, limits()); }),
              "non-string prompt cache key rejected");

    Json file_output     = base;
    file_output["input"] = Json::array(
        {Json{{"type", "function_call_output"},
              {"call_id", "call_1"},
              {"output", Json::array({Json{{"type", "input_file"}, {"file_id", "file_1"}}})}}});
    failures += check(api_code([&] { (void)parse_responses_request(file_output, limits()); }) ==
                          "file_inputs_not_supported",
                      "files in function output rejected explicitly");

    Json unknown       = base;
    unknown["made_up"] = 1;
    failures += check(api_code([&] { (void)parse_responses_request(unknown, limits()); }) ==
                          "unknown_parameter",
                      "unknown parameter rejected");

    Json too_small                 = base;
    too_small["max_output_tokens"] = 15;
    failures += check(api_code([&] { (void)parse_responses_request(too_small, limits()); }) ==
                          "invalid_value",
                      "OpenAI minimum max_output_tokens enforced");

    const auto rejects_nested = [&](Json input, const std::string& message) {
        Json body     = base;
        body["input"] = Json::array({std::move(input)});
        failures += check(api_code([&] { (void)parse_responses_request(body, limits()); }) ==
                              "unknown_parameter",
                          message);
    };
    rejects_nested(
        Json{{"type", "message"},
             {"role", "user"},
             {"content",
              Json::array({Json{{"type", "input_text"}, {"text", "hello"}, {"semantic", true}}})}},
        "unknown message content field was dropped");
    rejects_nested(
        Json{{"type", "reasoning"},
             {"summary", Json::array({Json{
                             {"type", "summary_text"}, {"text", "thought"}, {"semantic", true}}})}},
        "unknown reasoning summary field was dropped");
    rejects_nested(
        Json{{"type", "function_call_output"},
             {"call_id", "call_1"},
             {"output",
              Json::array({Json{{"type", "input_text"}, {"text", "result"}, {"semantic", true}}})}},
        "unknown tool-output content field was dropped");

    Json replay     = base;
    replay["input"] = Json::array({
        Json{{"type", "message"},
             {"role", "assistant"},
             {"content", Json::array({Json{{"type", "output_text"},
                                           {"text", "answer"},
                                           {"annotations", Json::array()},
                                           {"logprobs", Json::array()},
                                           {"provider_extension", nullptr}}})}},
        Json{{"type", "message"}, {"role", "user"}, {"content", "continue"}},
    });
    failures += check(!throws_api([&] { (void)parse_responses_request(replay, limits()); }),
                      "common AI SDK replay defaults were rejected");

    Json annotated = replay;
    annotated["input"][0]["content"][0]["annotations"] =
        Json::array({Json{{"type", "url_citation"}}});
    failures += check(throws_api([&] { (void)parse_responses_request(annotated, limits()); }),
                      "nonempty dropped output annotations were accepted");
    return failures;
}

int test_prompt_cache_breakpoints() {
    const Json marker = Json{{"mode", "explicit"}};
    const auto text   = [&](const char* type, const char* value, bool marked) {
        Json part = Json{{"type", type}, {"text", value}};
        if (marked) { part["prompt_cache_breakpoint"] = marker; }
        return part;
    };
    Json body = {{"model", "qwen3.8-27b"},
                 {"max_output_tokens", 32},
                 {"prompt_cache_options", marker},
                 {"input",
                  Json::array(
                      {Json{{"type", "message"},
                            {"role", "developer"},
                            {"content", Json::array({text("input_text", "rules", true)})}},
                       Json{{"type", "message"},
                            {"role", "user"},
                            {"content", Json::array({text("input_text", "first", true),
                                                     text("input_text", "second", false)})}},
                       Json{{"type", "message"},
                            {"role", "assistant"},
                            {"content", Json::array({text("output_text", "reply", true)})}},
                       Json{{"type", "function_call"},
                            {"call_id", "call_1"},
                            {"name", "weather"},
                            {"arguments", "{}"}},
                       Json{{"type", "function_call_output"},
                            {"call_id", "call_1"},
                            {"output", Json::array({text("input_text", "sunny", true)})}}})}};
    int failures = 0;
    const ResponsesRequest request = parse_responses_request(body, limits());
    failures += check(request.generation.prompt_cache_mode == ninfer::PromptCacheMode::Explicit,
                      "prompt_cache_options.mode explicit not carried");
    // developer, user, assistant text, assistant call, tool
    failures += check(request.input_turns.size() == 5 &&
                          request.input_turns[0].content[0].cache_breakpoint &&
                          request.input_turns[1].content[0].cache_breakpoint &&
                          !request.input_turns[1].content[1].cache_breakpoint &&
                          request.input_turns[2].content[0].cache_breakpoint &&
                          request.input_turns[4].content[0].cache_breakpoint,
                      "prompt_cache_breakpoint markers not carried onto their parts");
    failures += check(request.input_items[1].at("content")[0].at("prompt_cache_breakpoint") ==
                              marker &&
                          !request.input_items[1].at("content")[1].contains(
                              "prompt_cache_breakpoint") &&
                          request.input_items[4].at("output")[0].at("prompt_cache_breakpoint") ==
                              marker,
                      "canonical input Items did not round-trip the markers");
    const ninfer::PromptInput prompt = to_prompt_input(
        request.generation, resolve_prompt_semantics(request.generation, ServeOptions{},
                                                     effort_capabilities()),
        nullptr);
    failures += check(prompt.options.prompt_cache_mode == ninfer::PromptCacheMode::Explicit &&
                          prompt.messages[1].parts.front().cache_breakpoint &&
                          !prompt.messages[1].parts.back().cache_breakpoint,
                      "breakpoints did not reach the prompt input");

    Json fifth = body;
    fifth["input"][1]["content"][1]["prompt_cache_breakpoint"] = marker;
    failures += check(api_code([&] { (void)parse_responses_request(fifth, limits()); }) ==
                          "too_many_prompt_cache_breakpoints",
                      "fifth prompt_cache_breakpoint accepted");

    Json implicit                    = body;
    implicit["prompt_cache_options"] = Json{{"mode", "implicit"}};
    failures += check(parse_responses_request(implicit, limits()).generation.prompt_cache_mode ==
                          ninfer::PromptCacheMode::Implicit,
                      "prompt_cache_options.mode implicit not carried");
    Json unknown_mode                    = body;
    unknown_mode["prompt_cache_options"] = Json{{"mode", "sticky"}};
    failures += check(api_code([&] { (void)parse_responses_request(unknown_mode, limits()); }) ==
                          "invalid_value",
                      "unknown prompt_cache_options.mode accepted");
    Json retention                    = body;
    retention["prompt_cache_options"] = Json{{"mode", "explicit"}, {"retention", "24h"}};
    failures += check(api_code([&] { (void)parse_responses_request(retention, limits()); }) ==
                          "parameter_not_supported",
                      "prompt_cache_options.retention accepted");
    Json implicit_marker = body;
    implicit_marker["input"][1]["content"][0]["prompt_cache_breakpoint"] =
        Json{{"mode", "implicit"}};
    failures += check(api_code([&] { (void)parse_responses_request(implicit_marker, limits()); }) ==
                          "invalid_value",
                      "implicit block marker accepted");
    Json bare_marker = body;
    bare_marker["input"][1]["content"][0]["prompt_cache_breakpoint"] = true;
    failures += check(throws_api([&] { (void)parse_responses_request(bare_marker, limits()); }),
                      "non-object block marker accepted");
    return failures;
}

GenerationOutcome sample_outcome() {
    GenerationOutcome outcome;
    outcome.text                            = "answer";
    outcome.reasoning                       = "thought";
    outcome.prompt_tokens                   = 11;
    outcome.completion_tokens               = 7;
    outcome.reasoning_tokens                = 3;
    outcome.finish_reason                   = ninfer::FinishReason::StopToken;
    outcome.metrics.prefix_cache_hit_tokens = 4;
    // Deeper than the prompt leaves room for: clamped to prompt - cached in usage.
    outcome.metrics.continuation.cache_write_tokens = 9;
    return outcome;
}

int test_response_object() {
    ResponsesRequest request =
        parse_responses_request(Json{{"model", "qwen3.8-27b"},
                                     {"input", "hello"},
                                     {"max_output_tokens", 32},
                                     {"reasoning", Json{{"effort", "low"}, {"summary", "auto"}}},
                                     {"prompt_cache_key", "session_123"},
                                     {"store", false}},
                                limits());
    ResponsesRuntimeValues runtime;
    runtime.temperature = 0.6F;
    runtime.top_p       = 0.95F;
    const BuiltResponse built =
        make_response_object("resp_test", 123, request, runtime, sample_outcome());
    const Json& response = built.body;
    int failures         = 0;
    failures += check(response.at("object") == "response" && response.at("status") == "completed",
                      "completed response envelope");
    failures += check(response.at("output").size() == 2 &&
                          response.at("output")[0].at("type") == "reasoning" &&
                          response.at("output")[1].at("type") == "message",
                      "typed reasoning and message output Items");
    failures += check(response.at("output")[0].at("summary")[0].at("text") == "thought",
                      "reasoning is exposed as a native summary_text part");
    failures += check(!response.at("output")[0].contains("status"),
                      "reasoning Item does not expose a nonstandard status");
    failures += check(!response.contains("output_text"),
                      "SDK-only output_text helper absent from wire body");
    failures += check(response.at("reasoning").at("effort") == "low",
                      "Responses object did not echo reasoning effort");
    failures += check(response.at("reasoning").at("summary") == "auto" &&
                          response.at("prompt_cache_key") == "session_123",
                      "Responses object did not echo summary and cache key");
    failures +=
        check(response.at("usage").at("input_tokens_details").at("cached_tokens") == 4 &&
                  response.at("usage").at("input_tokens_details").at("cache_write_tokens") == 7 &&
                  response.at("usage").at("output_tokens_details").at("reasoning_tokens") == 3 &&
                  response.at("usage").at("total_tokens") == 18,
              "Responses usage details serialized");
    failures += check(built.output_history.size() == 1 &&
                          built.output_history[0].reasoning_content == "thought" &&
                          built.output_history[0].content[0].text == "answer",
                      "terminal output converted to continuation history");

    GenerationOutcome incomplete = sample_outcome();
    incomplete.finish_reason     = ninfer::FinishReason::OutputLimit;
    const Json limited = make_response_object("resp_limit", 123, request, runtime, incomplete).body;
    failures +=
        check(limited.at("status") == "incomplete" && limited.at("completed_at").is_null() &&
                  limited.at("incomplete_details").at("reason") == "max_output_tokens",
              "output limit mapped to incomplete response");

    GenerationOutcome tools = sample_outcome();
    tools.text.clear();
    tools.tool_calls.push_back(ToolCall{"call_weather", "weather", R"({"city":"Paris"})"});
    const Json tool_response = make_response_object("resp_tool", 123, request, runtime, tools).body;
    failures += check(tool_response.at("output").back().at("type") == "function_call" &&
                          tool_response.at("output").back().at("call_id") == "call_weather" &&
                          !tool_response.at("output").back().at("id").get<std::string>().empty(),
                      "function call has distinct Item id and call_id");

    // Items carry their own status: the message ended when the model began calling, a closed
    // call completed, and the call the output cut short is incomplete. Only a closed call with
    // object arguments can be replayed as continuation history.
    GenerationOutcome cut = sample_outcome();
    cut.text              = "Checking.";
    cut.finish_reason     = ninfer::FinishReason::OutputLimit;
    cut.tool_calls        = {ToolCall{"call_a", "weather", R"({"city":"Paris"})"},
                             ToolCall{"call_b", "weather", R"(["Paris"])"},
                             ToolCall{"call_c", "weather", R"({"city": "Ber)",
                                      ToolCallState::Incomplete}};
    const BuiltResponse cut_built = make_response_object("resp_cut", 123, request, runtime, cut);
    const Json& cut_output        = cut_built.body.at("output");
    failures += check(cut_built.body.at("status") == "incomplete" && cut_output.size() == 5 &&
                          cut_output[1].at("type") == "message" &&
                          cut_output[1].at("status") == "completed" &&
                          cut_output[2].at("status") == "completed" &&
                          cut_output[3].at("status") == "completed" &&
                          cut_output[4].at("status") == "incomplete" &&
                          cut_output[4].at("arguments") == R"({"city": "Ber)",
                      "message and call Items carry their own completion status");
    failures += check(cut_built.output_history.size() == 1 &&
                          cut_built.output_history[0].tool_calls.size() == 1 &&
                          cut_built.output_history[0].tool_calls[0].id == "call_a" &&
                          cut_built.output_history[0].content[0].text == "Checking.",
                      "history replays only closed calls with object arguments");
    return failures;
}

int test_public_reasoning_and_include_hint() {
    const Json base = {
        {"model", "qwen3.8-27b"}, {"input", "hello"}, {"max_output_tokens", 32}, {"store", false}};
    const GenerationOutcome outcome = sample_outcome();
    int failures                    = 0;

    const ResponsesRequest public_request = parse_responses_request(base, limits());
    const BuiltResponse public_response =
        make_response_object("resp_public", 123, public_request, {}, outcome);
    const Json& public_item = public_response.body.at("output")[0];
    failures += check(public_response.body.at("reasoning").at("summary") == "auto" &&
                          public_response.body.at("output").size() == 2 &&
                          public_item.at("type") == "reasoning" &&
                          public_item.at("summary")[0].at("text") == "thought",
                      "omitted summary did not use the public auto default");
    failures += check(public_response.output_history[0].reasoning_content == "thought",
                      "reasoning was lost from internal continuation history");

    Json hinted_body                      = base;
    hinted_body["store"]                  = true;
    hinted_body["include"]                = Json::array({"reasoning.encrypted_content"});
    const ResponsesRequest hinted_request = parse_responses_request(hinted_body, limits());
    const BuiltResponse hinted_response =
        make_response_object("resp_hinted", 123, hinted_request, {}, outcome);
    const Json& hinted_item = hinted_response.body.at("output")[0];
    failures += check(hinted_item.at("summary")[0].at("text") == "thought" &&
                          !hinted_item.contains("encrypted_content"),
                      "ignored AI SDK include hint changed public reasoning output");

    Json replay = {
        {"model", "qwen3.8-27b"},
        {"input",
         Json::array({public_item,
                      Json{{"type", "message"}, {"role", "assistant"}, {"content", "answer"}},
                      Json{{"type", "message"}, {"role", "user"}, {"content", "continue"}}})}};
    const ResponsesRequest replayed = parse_responses_request(replay, limits());
    failures += check(replayed.input_turns[0].reasoning_content == "thought",
                      "public summary_text was not translated into reasoning history");

    replay["input"][0]["encrypted_content"] = nullptr;
    failures += check(!throws_api([&] { (void)parse_responses_request(replay, limits()); }),
                      "null encrypted_content SDK compatibility field was rejected");
    replay["input"][0]["encrypted_content"] = "opaque-envelope";
    failures += check(api_code([&] { (void)parse_responses_request(replay, limits()); }) ==
                          "encrypted_reasoning_not_supported",
                      "non-null encrypted reasoning was not rejected explicitly");

    Json unsupported_include       = base;
    unsupported_include["include"] = Json::array({"message.output_text.logprobs"});
    failures += check(api_code([&] {
                          (void)parse_responses_request(unsupported_include, limits());
                      }) == "include_not_supported",
                      "unsupported include value was accepted");
    return failures;
}

int test_default_reasoning_stream() {
    ResponsesRequest request =
        parse_responses_request(Json{{"model", "qwen3.8-27b"},
                                     {"input", "hello"},
                                     {"include", Json::array({"reasoning.encrypted_content"})},
                                     {"stream", true}},
                                limits());
    ResponsesEventStream encoder("resp_public_stream", 123, request, {});
    (void)encoder.start();
    int failures                    = 0;
    std::vector<std::string> events = encoder.reasoning_delta("thought");
    failures += check(!events.empty(), "default streaming reasoning omitted public events");
    std::vector<std::string> content_events = encoder.content_delta("answer");
    events.insert(events.end(), content_events.begin(), content_events.end());
    const ResponsesStreamFinish finish = encoder.finish(sample_outcome());
    events.insert(events.end(), finish.events_before_terminal.begin(),
                  finish.events_before_terminal.end());
    bool saw_summary_done = false;
    for (const std::string& wire : events) {
        const Json event = parse_event(wire);
        if (event.at("type") == "response.output_item.done" &&
            event.at("item").at("type") == "reasoning") {
            saw_summary_done = event.at("item").at("summary")[0].at("text") == "thought";
            failures += check(!event.at("item").contains("encrypted_content"),
                              "stream emitted unsupported encrypted_content");
        }
    }
    failures +=
        check(saw_summary_done, "terminal reasoning stream Item omitted its public summary");
    return failures;
}

int test_sse_sequence() {
    ResponsesRequest request =
        parse_responses_request(Json{{"model", "qwen3.8-27b"},
                                     {"input", "hello"},
                                     {"max_output_tokens", 32},
                                     {"reasoning", Json{{"summary", "auto"}}},
                                     {"stream", true}},
                                limits());
    ResponsesEventStream encoder("resp_stream", 123, request, {});
    std::vector<std::string> wire = encoder.start();
    std::vector<std::string> more = encoder.reasoning_delta("thought");
    wire.insert(wire.end(), more.begin(), more.end());
    more = encoder.content_delta("ans");
    wire.insert(wire.end(), more.begin(), more.end());
    more = encoder.content_delta("wer");
    wire.insert(wire.end(), more.begin(), more.end());
    GenerationOutcome outcome    = sample_outcome();
    ResponsesStreamFinish finish = encoder.finish(outcome);
    wire.insert(wire.end(), finish.events_before_terminal.begin(),
                finish.events_before_terminal.end());
    wire.push_back(encoder.terminal(finish.response));

    int failures                    = 0;
    std::uint64_t expected_sequence = 0;
    std::string text_deltas;
    std::string reasoning_deltas;
    bool saw_reasoning_part_done = false;
    for (const std::string& event : wire) {
        failures += check(event.find("[DONE]") == std::string::npos,
                          "Responses stream must not emit Chat [DONE]");
        const Json payload = parse_event(event);
        failures += check(payload.at("sequence_number") == expected_sequence++,
                          "sequence_number is contiguous");
        if (payload.at("type") == "response.output_text.delta") {
            text_deltas += payload.at("delta").get<std::string>();
        }
        if (payload.at("type") == "response.reasoning_summary_text.delta") {
            reasoning_deltas += payload.at("delta").get<std::string>();
        }
        if (payload.at("type") == "response.reasoning_summary_part.done") {
            saw_reasoning_part_done = true;
        }
    }
    failures += check(parse_event(wire.front()).at("type") == "response.created",
                      "stream starts with response.created");
    failures += check(parse_event(wire.back()).at("type") == "response.completed",
                      "stream ends with response.completed");
    failures += check(text_deltas == "answer", "text deltas reconstruct terminal output");
    failures += check(reasoning_deltas == "thought", "native reasoning summary deltas emitted");
    failures += check(saw_reasoning_part_done, "native reasoning summary completion emitted");
    return failures;
}

int test_sse_function_call() {
    ResponsesRequest request = parse_responses_request(Json{{"model", "qwen3.8-27b"},
                                                            {"input", "weather"},
                                                            {"max_output_tokens", 32},
                                                            {"stream", true}},
                                                       limits());
    // Reasoning, a message, a call the model closed, and a call the output cut short, each
    // published as the parser decided it.
    ResponsesEventStream encoder("resp_tool_stream", 123, request, {});
    std::vector<std::string> wire = encoder.start();
    auto append = [&wire](const std::vector<std::string>& events) {
        wire.insert(wire.end(), events.begin(), events.end());
    };
    const ToolCall weather{"call_weather", "weather", R"({"city": "Paris"})"};
    const ToolCall clock{"call_time", "time", R"({"zone": "Eur)", ToolCallState::Incomplete};
    append(encoder.reasoning_delta("look it up"));
    append(encoder.content_delta("Checking."));
    append(encoder.tool_call_begin(0, ToolCall{weather.id, weather.name, ""}));
    append(encoder.tool_call_arguments(0, R"({"city": ")"));
    append(encoder.tool_call_arguments(0, R"(Paris"})"));
    append(encoder.tool_call_end(0, weather));
    append(encoder.tool_call_begin(1, ToolCall{clock.id, clock.name, ""}));
    append(encoder.tool_call_arguments(1, clock.arguments_json));
    GenerationOutcome outcome;
    outcome.text              = "Checking.";
    outcome.reasoning         = "look it up";
    outcome.prompt_tokens     = 8;
    outcome.completion_tokens = 40;
    outcome.finish_reason     = ninfer::FinishReason::OutputLimit;
    outcome.tool_calls        = {weather, clock};
    ResponsesStreamFinish finish = encoder.finish(outcome);
    append(finish.events_before_terminal);
    wire.push_back(encoder.terminal(finish.response));

    int failures                    = 0;
    std::uint64_t expected_sequence = 0;
    int next_index                  = 0;
    int open_index                  = -1;
    bool items_contiguous           = true;
    std::vector<Json> added;
    std::vector<Json> done;
    std::vector<std::string> arguments(2);
    std::vector<std::string> arguments_done(2);
    for (const std::string& event : wire) {
        const Json payload = parse_event(event);
        failures += check(payload.at("sequence_number") == expected_sequence++,
                          "sequence_number is contiguous across function call events");
        const std::string type = payload.at("type").get<std::string>();
        // Output Items are contiguous: each is added, streamed and done before the next opens.
        if (payload.contains("output_index")) {
            const int index = payload.at("output_index").get<int>();
            if (type == "response.output_item.added") {
                items_contiguous = items_contiguous && open_index == -1 && index == next_index;
                open_index       = next_index++;
                added.push_back(payload.at("item"));
            } else {
                items_contiguous = items_contiguous && index == open_index;
            }
            if (type == "response.output_item.done") {
                open_index = -1;
                done.push_back(payload.at("item"));
            }
        }
        if (type == "response.function_call_arguments.delta") {
            arguments.at(static_cast<std::size_t>(payload.at("output_index").get<int>() - 2)) +=
                payload.at("delta").get<std::string>();
        }
        if (type == "response.function_call_arguments.done") {
            arguments_done.at(static_cast<std::size_t>(payload.at("output_index").get<int>() - 2)) =
                payload.at("arguments").get<std::string>();
        }
    }
    failures += check(items_contiguous && open_index == -1 && added.size() == 4 &&
                          done.size() == 4,
                      "reasoning, message and calls stream as contiguous Items");
    failures += check(added[2] == Json({{"id", added[2].at("id")},
                                        {"type", "function_call"},
                                        {"status", "in_progress"},
                                        {"call_id", "call_weather"},
                                        {"name", "weather"},
                                        {"arguments", ""}}) &&
                          added[2].at("id") != "call_weather",
                      "function_call Item is announced with its name and empty arguments");
    failures += check(done[1].at("type") == "message" && done[1].at("status") == "completed" &&
                          done[1].at("content")[0].at("text") == "Checking.",
                      "message Item completes when the first call begins");
    failures += check(arguments[0] == R"({"city": "Paris"})" &&
                          arguments_done[0] == arguments[0] &&
                          done[2].at("status") == "completed" &&
                          done[2].at("arguments") == arguments[0],
                      "closed call streams argument deltas and completes");
    failures += check(arguments[1] == R"({"zone": "Eur)" && arguments_done[1] == arguments[1] &&
                          done[3].at("status") == "incomplete",
                      "cut-short call keeps exactly its streamed arguments and is incomplete");

    const Json terminal = parse_event(wire.back());
    const Json& output  = terminal.at("response").at("output");
    failures += check(terminal.at("type") == "response.incomplete" && output.size() == 4 &&
                          output[2].at("id") == added[2].at("id") &&
                          output[3].at("id") == added[3].at("id") &&
                          output[3].at("status") == "incomplete",
                      "terminal response repeats the streamed Items and their ids");
    failures += check(finish.response.output_history[0].tool_calls.size() == 1 &&
                          finish.response.output_history[0].tool_calls[0].id == "call_weather",
                      "cut-short call is not replayed as continuation history");
    return failures;
}

int test_cancelled_sse_terminal() {
    ResponsesRequest request = parse_responses_request(
        Json{{"model", "qwen3.8-27b"}, {"input", "hello"}, {"stream", true}}, limits());
    ResponsesEventStream encoder("resp_cancelled", 123, request, {});
    (void)encoder.start();
    GenerationOutcome outcome;
    outcome.finish_reason              = ninfer::FinishReason::Cancelled;
    const ResponsesStreamFinish finish = encoder.finish(outcome);
    const Json terminal                = parse_event(encoder.terminal(finish.response));
    int failures                       = 0;
    failures += check(terminal.at("type") == "response.failed",
                      "cancelled stream uses AI SDK response.failed terminal");
    failures += check(terminal.at("response").at("status") == "failed" &&
                          terminal.at("response").at("error").at("code") == "request_cancelled",
                      "cancelled terminal carries request_cancelled error");
    return failures;
}

int test_input_tokens_schema() {
    const ResponsesRequest request = parse_response_input_tokens_request(
        Json{{"model", "qwen3.8-27b"}, {"input", "hello"}}, limits());
    int failures = 0;
    failures += check(!request.store && !request.stream, "input_tokens request is stateless");
    failures += check(Json::parse(make_response_input_tokens_body(9)) ==
                          Json{{"object", "response.input_tokens"}, {"input_tokens", 9}},
                      "input_tokens response shape");
    failures +=
        check(api_code([&] {
                  (void)parse_response_input_tokens_request(
                      Json{{"model", "qwen3.8-27b"}, {"input", "hello"}, {"instructions", "x"}},
                      limits());
              }) == "unknown_parameter",
              "input_tokens accepts only model and input");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_basic_request();
    failures += test_preserve_thinking_options_and_inheritance();
    failures += test_reasoning_effort();
    failures += test_typed_items_and_tools();
    failures += test_explicit_rejections();
    failures += test_prompt_cache_breakpoints();
    failures += test_response_object();
    failures += test_public_reasoning_and_include_hint();
    failures += test_default_reasoning_stream();
    failures += test_sse_sequence();
    failures += test_sse_function_call();
    failures += test_cancelled_sse_terminal();
    failures += test_input_tokens_schema();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
