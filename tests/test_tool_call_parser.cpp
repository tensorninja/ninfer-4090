#include "serve/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::ordered_json;
using ninfer::serve::QwenToolCallStream;
using ninfer::serve::ToolCall;
using ninfer::serve::ToolCallEvents;
using ninfer::serve::ToolCallState;
using ninfer::serve::ToolCallStreamResult;
using ninfer::serve::ToolDefinition;
using ninfer::serve::ToolParameterKind;
using ninfer::serve::ToolParameterKinds;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

bool valid_utf8(std::string_view text) {
    std::size_t index = 0;
    while (index < text.size()) {
        const auto lead        = static_cast<unsigned char>(text[index]);
        const std::size_t size = lead < 0x80U            ? 1
                                 : (lead >> 5U) == 0x6U  ? 2
                                 : (lead >> 4U) == 0xEU  ? 3
                                 : (lead >> 3U) == 0x1EU ? 4
                                                         : 0;
        if (size == 0 || index + size > text.size()) { return false; }
        for (std::size_t offset = 1; offset < size; ++offset) {
            if ((static_cast<unsigned char>(text[index + offset]) & 0xC0U) != 0x80U) {
                return false;
            }
        }
        index += size;
    }
    return true;
}

ToolDefinition tool(std::string name, std::string_view properties) {
    ToolDefinition definition;
    definition.name            = std::move(name);
    definition.parameters_json = R"({"type":"object","properties":)" + std::string(properties) + "}";
    return definition;
}

ToolParameterKinds test_kinds() {
    return ToolParameterKinds::from_tools({
        tool("get_weather", R"({"city":{"type":"string"},"days":{"type":"integer"}})"),
        tool("write_file", R"({"path":{"type":"string"},"content":{"type":"string"}})"),
        tool("batch", R"({"tool_calls":{"type":"array"}})"),
        tool("edit", R"({"replace_all":{"type":"boolean"},"note":{"type":["string","null"]}})"),
    });
}

struct RecordedCall {
    std::string id;
    std::string name;
    std::string arguments;
    std::size_t argument_events = 0;
    bool ended                  = false;
};

// Records the event stream and checks its grammar as it arrives.
class Recorder final : public ToolCallEvents {
public:
    std::string text;
    std::vector<RecordedCall> calls;
    std::vector<std::string> violations;

    void content(std::string_view delta) override {
        if (!calls.empty()) { violations.emplace_back("content after a call"); }
        if (!valid_utf8(delta)) { violations.emplace_back("content delta is not UTF-8"); }
        text.append(delta);
    }

    void tool_call_begin(std::size_t index, const ToolCall& call) override {
        if (index != calls.size()) { violations.emplace_back("begin index is not the next call"); }
        if (!calls.empty() && !calls.back().ended) {
            violations.emplace_back("begin before the previous call ended");
        }
        if (!call.arguments_json.empty() || call.state != ToolCallState::Incomplete) {
            violations.emplace_back("begin of a call that already has arguments");
        }
        calls.push_back(RecordedCall{call.id, call.name, {}, 0, false});
    }

    void tool_call_arguments(std::size_t index, std::string_view delta) override {
        if (index + 1 != calls.size() || calls[index].ended) {
            violations.emplace_back("arguments outside the open call");
            return;
        }
        if (delta.empty()) { violations.emplace_back("empty argument delta"); }
        if (!valid_utf8(delta)) { violations.emplace_back("argument delta is not UTF-8"); }
        calls[index].arguments.append(delta);
        ++calls[index].argument_events;
    }

    void tool_call_end(std::size_t index, const ToolCall& call) override {
        if (index + 1 != calls.size() || calls[index].ended) {
            violations.emplace_back("end outside the open call");
            return;
        }
        if (call.arguments_json != calls[index].arguments) {
            violations.emplace_back("end arguments differ from the deltas");
        }
        if (call.state != ToolCallState::Complete) {
            violations.emplace_back("end of an incomplete call");
        }
        calls[index].ended = true;
    }
};

struct Parsed {
    ToolCallStreamResult result;
    Recorder events;
};

// Feeds `text` split at `cuts` (ascending character boundaries), as the engine publishes it.
Parsed parse(std::string_view text, const std::vector<std::size_t>& cuts = {},
             std::size_t max_name_length = 64) {
    Parsed parsed;
    QwenToolCallStream stream(test_kinds(), max_name_length, &parsed.events);
    std::size_t begin = 0;
    for (const std::size_t cut : cuts) {
        stream.feed(text.substr(begin, cut - begin));
        begin = cut;
    }
    stream.feed(text.substr(begin));
    parsed.result = stream.finish();
    return parsed;
}

std::vector<std::size_t> character_cuts(std::string_view text) {
    std::vector<std::size_t> cuts;
    for (std::size_t index = 1; index < text.size(); ++index) {
        if ((static_cast<unsigned char>(text[index]) & 0xC0U) != 0x80U) { cuts.push_back(index); }
    }
    return cuts;
}

std::vector<std::size_t> random_cuts(std::string_view text, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<std::size_t> cuts;
    for (const std::size_t cut : character_cuts(text)) {
        if (rng() % 5 == 0) { cuts.push_back(cut); }
    }
    return cuts;
}

std::vector<std::vector<std::size_t>> split_policies(std::string_view text) {
    std::vector<std::vector<std::size_t>> policies = {{}, character_cuts(text)};
    for (std::uint64_t seed = 1; seed <= 12; ++seed) {
        policies.push_back(random_cuts(text, seed));
    }
    return policies;
}

// The events must replay the result: same content, same calls, same arguments.
int check_events(const Parsed& parsed, const std::string& label) {
    int failures = 0;
    for (const std::string& violation : parsed.events.violations) {
        failures += fail(label + ": " + violation);
    }
    failures += check(parsed.events.text == parsed.result.content, label + ": content events");
    failures += check(parsed.events.calls.size() == parsed.result.tool_calls.size(),
                      label + ": call events");
    if (parsed.events.calls.size() != parsed.result.tool_calls.size()) { return failures; }
    for (std::size_t index = 0; index < parsed.events.calls.size(); ++index) {
        const RecordedCall& recorded = parsed.events.calls[index];
        const ToolCall& call         = parsed.result.tool_calls[index];
        failures += check(recorded.id == call.id && call.id.rfind("call_", 0) == 0 &&
                              recorded.name == call.name &&
                              recorded.arguments == call.arguments_json &&
                              recorded.ended == (call.state == ToolCallState::Complete),
                          label + ": call " + std::to_string(index) + " events");
    }
    return failures;
}

bool same_result(const ToolCallStreamResult& lhs, const ToolCallStreamResult& rhs) {
    if (lhs.content != rhs.content || lhs.discarded_bytes != rhs.discarded_bytes ||
        lhs.tool_calls.size() != rhs.tool_calls.size()) {
        return false;
    }
    for (std::size_t index = 0; index < lhs.tool_calls.size(); ++index) {
        const ToolCall& left  = lhs.tool_calls[index];
        const ToolCall& right = rhs.tool_calls[index];
        if (left.name != right.name || left.arguments_json != right.arguments_json ||
            left.state != right.state) {
            return false;
        }
    }
    return true;
}

// Parses `text` under every split policy, checks the events of each, and requires every split to
// produce the whole-feed result, which is returned.
ToolCallStreamResult parse_all(std::string_view text, int& failures,
                               std::size_t max_name_length = 64) {
    const std::string label = "\"" + std::string(text.substr(0, 60)) + "\"";
    Parsed whole            = parse(text, {}, max_name_length);
    failures += check_events(whole, label + " whole");
    for (const std::vector<std::size_t>& cuts : split_policies(text)) {
        const Parsed split = parse(text, cuts, max_name_length);
        failures += check_events(split, label + " split");
        failures += check(same_result(whole.result, split.result),
                          label + ": result depends on the split (" +
                              std::to_string(cuts.size()) + " cuts)");
    }
    return std::move(whole.result);
}

const std::string kWeatherCall = "<tool_call>\n"
                                 "<function=get_weather>\n"
                                 "<parameter=city>\nParis\n</parameter>\n"
                                 "<parameter=days>\n2\n</parameter>\n"
                                 "</function>\n"
                                 "</tool_call>";

int test_parameter_kinds() {
    const ToolParameterKinds kinds = ToolParameterKinds::from_tools({tool("t", R"({
        "s": {"type": "string"},
        "s_null": {"type": ["string", "null"]},
        "s_any": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "s_enum": {"enum": ["a", "b"]},
        "s_const": {"const": "x"},
        "i": {"type": "integer"},
        "b": {"type": "boolean"},
        "a": {"type": "array", "items": {"type": "string"}},
        "o": {"type": "object"},
        "ao_null": {"type": ["array", "object", "null"]},
        "a_one": {"oneOf": [{"type": "array"}, {"type": "null"}]},
        "mixed": {"type": ["string", "number"]},
        "untyped": {"description": "anything"},
        "any_untyped": {"anyOf": [{"type": "string"}, {"description": "anything"}]},
        "null_only": {"type": "null"}
    })")});

    int failures = 0;
    for (const char* name : {"s", "s_null", "s_any", "s_enum", "s_const"}) {
        failures += check(kinds.kind("t", name) == ToolParameterKind::String,
                          std::string("string kind: ") + name);
    }
    for (const char* name : {"a", "o", "ao_null", "a_one"}) {
        failures += check(kinds.kind("t", name) == ToolParameterKind::JsonContainer,
                          std::string("container kind: ") + name);
    }
    for (const char* name : {"i", "b", "mixed", "untyped", "any_untyped", "null_only", "absent"}) {
        failures += check(kinds.kind("t", name) == ToolParameterKind::Buffered,
                          std::string("buffered kind: ") + name);
    }
    failures += check(kinds.kind("undeclared", "s") == ToolParameterKind::Buffered,
                      "undeclared tool parameters are buffered");
    return failures;
}

int test_split_invariance() {
    const std::vector<std::string> corpus = {
        "Plain answer with <b>markup</b>, <tool_call spelled out, and trailing space  \n",
        "Calling weather.\n" + kWeatherCall,
        kWeatherCall + "\n" + kWeatherCall + "\n",
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"days\": 2, \"city\": "
        "\"Paris\"}}\n</tool_call>",
        "<tool_call>\n{\"name\":\"first\",\"arguments\":{\"value\":1}}\n</tool_call>\n" +
            kWeatherCall,
        "Use <tool_call> tags like <tool_call><function=bad name!> here.",
        kWeatherCall + "\nextra answer",
        "<tool_call>\n<function=write_file>\n<parameter=path>\nsrc/\xc3\xa9.txt\n</parameter>\n"
        "<parameter=content>\n  indented \"quote\" \\ back\n\ttab \xe4\xb8\x96\xe7\x95\x8c "
        "\xf0\x9f\x8e\x89\n</paramete\n\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=batch>\n<parameter=tool_calls>\n[{\"tool\": \"read\", "
        "\"parameters\": {\"path\": \"a</b>\"}}, 2]\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=write_file>\n<parameter=content>\nline one\nline tw",
        "<tool_call>\n<function=get_weather>\nnoise <x> \n<parameter=city>\nParis\n"
        "</parameter>\n</tool_call>",
        "Thinking done.\n\n<tool_call>\n<function=edit>\n<parameter=replace_all>\ntrue\n"
        "</parameter>\n<parameter=note>\n{\"looks\": \"like json\"}\n</parameter>\n</function>\n"
        "</tool_call>\n\n",
    };
    int failures = 0;
    for (const std::string& text : corpus) { static_cast<void>(parse_all(text, failures)); }
    return failures;
}

int test_announce_timing() {
    Recorder events;
    QwenToolCallStream stream(test_kinds(), 64, &events);
    int failures = 0;

    stream.feed("Calling.\n<tool_call>\n<function=get_weat");
    failures += check(events.calls.empty(), "call announced before its name closed");
    failures += check(events.text == "Calling.", "content before the call was held back");
    stream.feed("her>");
    failures += check(events.calls.size() == 1 && events.calls[0].name == "get_weather" &&
                          events.calls[0].arguments == "{",
                      "begin and { must arrive with the feed that closes <function=NAME>");
    if (events.calls.size() != 1) { return failures + 1; }
    const std::string& arguments = events.calls[0].arguments;

    stream.feed("\n<parameter=city>\nPar");
    failures += check(arguments == "{\"city\":\"Par", "string value did not stream: " + arguments);
    stream.feed("is\n");
    failures += check(arguments == "{\"city\":\"Paris", "a possible framing newline streamed");
    stream.feed("</parameter>\n<parameter=days>\n2");
    failures += check(arguments == "{\"city\":\"Paris\",\"days\":",
                      "buffered value streamed before </parameter>: " + arguments);
    stream.feed("\n</parameter>\n");
    failures += check(arguments == "{\"city\":\"Paris\",\"days\":2", "buffered value missing");
    failures += check(!events.calls[0].ended, "call ended before </function>");
    stream.feed("</function>");
    failures += check(events.calls[0].ended && arguments == "{\"city\":\"Paris\",\"days\":2}",
                      "call did not end with </function>");
    stream.feed("\n</tool_call>");
    const ToolCallStreamResult result = stream.finish();
    failures += check(result.content == "Calling." && result.tool_calls.size() == 1 &&
                          result.tool_calls[0].state == ToolCallState::Complete &&
                          result.discarded_bytes == 0,
                      "timed call result");
    failures += check(events.violations.empty(), "timed call event grammar");
    return failures;
}

// The value framing is the exact inverse of render_tool_call, so any value without a literal
// </parameter> survives a render/parse round trip, including one that looks like JSON.
int test_string_round_trip() {
    const std::vector<std::string> values = {
        "",
        " ",
        "\n",
        "\n\n",
        "  indented\n\ttab  ",
        "trailing newline\n",
        "\nleading newline",
        "quote \" and \\ backslash",
        std::string("control \x01\x08\x0c\x1f\x7f end"),
        "partial </param and </parameter",
        "ends with </paramete",
        "h\xc3\xa9llo \xe4\xb8\x96\xe7\x95\x8c \xf0\x9f\x8e\x89",
        "{\"a\": 1}",
        "[1, 2]",
        "42",
        "true",
        "null",
        "\"quoted\"",
    };
    int failures = 0;
    for (const std::string& value : values) {
        const std::string text = "<tool_call>\n<function=write_file>\n<parameter=content>\n" +
                                 value + "\n</parameter>\n</function>\n</tool_call>";
        const ToolCallStreamResult result = parse_all(text, failures);
        const std::string expected        = Json{{"content", value}}.dump();
        failures += check(result.tool_calls.size() == 1 &&
                              result.tool_calls[0].arguments_json == expected,
                          "string value round trip: " + expected);
    }
    return failures;
}

int test_raw_container() {
    int failures      = 0;
    const std::string raw =
        "[{\"tool\": \"read\", \"parameters\": {\"path\": \"a.txt\"}}, {\"tool\": \"grep\"}]";
    const std::string text = "<tool_call>\n<function=batch>\n<parameter=tool_calls>\n  " + raw +
                             "\n</parameter>\n</function>\n</tool_call>";
    const ToolCallStreamResult result = parse_all(text, failures);
    failures += check(result.tool_calls.size() == 1 &&
                          result.tool_calls[0].arguments_json == "{\"tool_calls\":" + raw + "}",
                      "raw array passes through as written");
    const Parsed streamed = parse(text, character_cuts(text));
    failures += check(streamed.events.calls.size() == 1 &&
                          streamed.events.calls[0].argument_events > raw.size() / 2,
                      "raw array did not stream incrementally");

    const ToolCallStreamResult invalid = parse_all(
        "<tool_call>\n<function=batch>\n<parameter=tool_calls>\n[1, 2\n</parameter>\n"
        "</function>\n</tool_call>",
        failures);
    failures += check(invalid.tool_calls.size() == 1 &&
                          invalid.tool_calls[0].arguments_json == "{\"tool_calls\":[1, 2}" &&
                          invalid.tool_calls[0].state == ToolCallState::Complete,
                      "invalid raw JSON is kept as written");

    const ToolCallStreamResult bare = parse_all(
        "<tool_call>\n<function=batch>\n<parameter=tool_calls>\nnone\n</parameter>\n"
        "</function>\n</tool_call>",
        failures);
    failures += check(bare.tool_calls.size() == 1 &&
                          bare.tool_calls[0].arguments_json == "{\"tool_calls\":\"none\"}",
                      "container value without a JSON opening is buffered");
    const ToolCallStreamResult null_value = parse_all(
        "<tool_call>\n<function=batch>\n<parameter=tool_calls>\nnull\n</parameter>\n"
        "</function>\n</tool_call>",
        failures);
    failures += check(null_value.tool_calls.size() == 1 &&
                          null_value.tool_calls[0].arguments_json == "{\"tool_calls\":null}",
                      "container null is buffered JSON");
    return failures;
}

int test_buffered_values() {
    int failures                      = 0;
    const ToolCallStreamResult result = parse_all(
        "<tool_call>\n<function=edit>\n<parameter=replace_all>\ntrue\n</parameter>\n"
        "<parameter=extra>\nhello world\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=unknown>\n<parameter=n>\n3.5\n</parameter>\n"
        "<parameter=payload>\n{\"ok\": true, \"items\": [1, 2]}\n</parameter>\n"
        "<parameter=days>\n 7 \n</parameter>\n</function>\n</tool_call>",
        failures);
    failures += check(result.tool_calls.size() == 2, "two buffered calls");
    if (result.tool_calls.size() != 2) { return failures; }
    failures += check(result.tool_calls[0].arguments_json ==
                          "{\"replace_all\":true,\"extra\":\"hello world\"}",
                      "boolean and undeclared parameters: " + result.tool_calls[0].arguments_json);
    failures += check(result.tool_calls[1].arguments_json ==
                          "{\"n\":3.5,\"payload\":{\"ok\":true,\"items\":[1,2]},\"days\":7}",
                      "undeclared tool values: " + result.tool_calls[1].arguments_json);
    return failures;
}

// Markup that never becomes a call before any call was announced is ordinary text.
int test_pre_announce_fallback() {
    int failures = 0;
    for (const char* text : {
             "Use <tool_call> tags.",
             "<tool_call>\n<function=bad name!>\n</function>\n</tool_call>",
             "prefix  \n<tool_call>\n<func",
             "<tool_call>\n<function=",
             "<tool_call>\n<function=get_weather",
             "<tool_call>\n",
             "ordinary text  ",
             "<tool_call>\n{\"name\":\"get_weather\",\"argum",
             "<tool_call>\n[\"get_weather\"]\n</tool_call>",
             "<tool_call>\n{\"arguments\":{}}\n</tool_call>",
             "<tool_call>\n{\"name\":\"bad name!\",\"arguments\":{}}\n</tool_call>",
             "<tool_call>\n{\"name\":\"ok\",\"arguments\":[1,2]}\n</tool_call>",
             "<tool_call>\n{\"name\":\"ok\",\"arguments\":{}} tail\n</tool_call>",
         }) {
        const ToolCallStreamResult result = parse_all(text, failures);
        failures += check(result.tool_calls.empty() && result.content == text &&
                              result.discarded_bytes == 0,
                          std::string("rejected markup preserved verbatim: ") + text);
    }
    return failures;
}

int test_text_after_call_is_dropped() {
    int failures                    = 0;
    const ToolCallStreamResult tail = parse_all(kWeatherCall + "\nextra answer\n", failures);
    failures += check(tail.tool_calls.size() == 1 &&
                          tail.tool_calls[0].state == ToolCallState::Complete &&
                          tail.content.empty() && tail.discarded_bytes == 11,
                      "suffix after a call is dropped and counted");

    const ToolCallStreamResult between =
        parse_all("Prefix.\n" + kWeatherCall + "\nmore text\n" + kWeatherCall, failures);
    failures += check(between.tool_calls.size() == 2 && between.content == "Prefix." &&
                          between.discarded_bytes == 8,
                      "text between calls is dropped and counted");

    const ToolCallStreamResult failed =
        parse_all(kWeatherCall + "\n<tool_call>\nnot a call\n</tool_call>", failures);
    failures += check(failed.tool_calls.size() == 1 && failed.content.empty() &&
                          failed.discarded_bytes > 0,
                      "a failed block after a call is dropped and counted");

    const ToolCallStreamResult stray = parse_all(
        "<tool_call>\n<function=get_weather>\nnoise\n<parameter=city>\nParis\n</parameter>\n"
        "</function>\n</tool_call>",
        failures);
    failures += check(stray.tool_calls.size() == 1 &&
                          stray.tool_calls[0].arguments_json == "{\"city\":\"Paris\"}" &&
                          stray.discarded_bytes == 5,
                      "stray text inside a call is dropped and counted");
    return failures;
}

int test_incomplete_calls() {
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n"
         "<parameter=content>\nline one\nline tw",
         "{\"path\":\"a.txt\",\"content\":\"line one\\nline tw"},
        {"<tool_call>\n<function=write_file>\n<parameter=content>\nabc\n</para",
         "{\"content\":\"abc"},
        {"<tool_call>\n<function=get_weather>", "{"},
        {"<tool_call>\n<function=x>", "{"},
        {"<tool_call>\n<function=get_weather>\n<parameter=days>\n2", "{\"days\":"},
        {"<tool_call>\n<function=get_weather>\n<parameter=ci", "{"},
        {"<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</func",
         "{\"city\":\"Paris\""},
        {"<tool_call>\n<function=batch>\n<parameter=tool_calls>\n[1, ", "{\"tool_calls\":[1, "},
    };
    int failures = 0;
    for (const auto& [text, arguments] : cases) {
        const ToolCallStreamResult result = parse_all(text, failures);
        failures += check(result.tool_calls.size() == 1 &&
                              result.tool_calls[0].state == ToolCallState::Incomplete &&
                              result.tool_calls[0].arguments_json == arguments &&
                              result.content.empty() && result.discarded_bytes == 0,
                          "incomplete call keeps exactly the streamed prefix: " + text);
    }
    const ToolCallStreamResult second =
        parse_all(kWeatherCall + "\n<tool_call>\n<function=get_weather>\n", failures);
    failures += check(second.tool_calls.size() == 2 &&
                          second.tool_calls[0].state == ToolCallState::Complete &&
                          second.tool_calls[1].state == ToolCallState::Incomplete,
                      "a later call can be incomplete after a complete one");
    return failures;
}

int test_bare_tool_close_completes() {
    int failures                      = 0;
    const ToolCallStreamResult result = parse_all(
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
        "</tool_call>\n",
        failures);
    failures += check(result.tool_calls.size() == 1 &&
                          result.tool_calls[0].state == ToolCallState::Complete &&
                          result.tool_calls[0].arguments_json == "{\"city\":\"Paris\"}" &&
                          result.discarded_bytes == 0,
                      "bare </tool_call> completes the call");
    return failures;
}

int test_hermes_calls() {
    int failures                      = 0;
    const ToolCallStreamResult single = parse_all(
        "Calling weather.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": "
        "{\"days\": 2, \"city\": \"Paris\"}}\n</tool_call>",
        failures);
    failures += check(single.content == "Calling weather." && single.tool_calls.size() == 1 &&
                          single.tool_calls[0].name == "get_weather" &&
                          single.tool_calls[0].arguments_json == "{\"days\":2,\"city\":\"Paris\"}",
                      "hermes call keeps the model's key order");

    const Parsed atomic = parse(
        "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>",
        character_cuts("<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}"
                       "\n</tool_call>"));
    failures += check(atomic.events.calls.size() == 1 &&
                          atomic.events.calls[0].argument_events == 1 &&
                          atomic.events.calls[0].ended,
                      "hermes call is emitted as begin, one delta, and end");

    const ToolCallStreamResult mixed = parse_all(
        "<tool_call>\n{\"name\":\"first\",\"arguments\":{\"value\":1}}\n</tool_call>\n"
        "<tool_call>\n<function=second>\n<parameter=value>\n2\n</parameter>\n</function>\n"
        "</tool_call>",
        failures);
    failures += check(mixed.tool_calls.size() == 2 &&
                          mixed.tool_calls[0].arguments_json == "{\"value\":1}" &&
                          mixed.tool_calls[1].arguments_json == "{\"value\":2}",
                      "hermes and xml calls mix in one turn");

    const ToolCallStreamResult encoded = parse_all(
        "<tool_call>\n{\"name\":\"call\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}\n"
        "</tool_call>",
        failures);
    const ToolCallStreamResult absent =
        parse_all("<tool_call>\n{\"name\":\"ping\"}\n</tool_call>", failures);
    const ToolCallStreamResult null_args =
        parse_all("<tool_call>\n{\"name\":\"ping\",\"arguments\":null}\n</tool_call>", failures);
    failures += check(encoded.tool_calls.size() == 1 &&
                          encoded.tool_calls[0].arguments_json == "{\"city\":\"Paris\"}",
                      "double-encoded arguments string decoded");
    failures += check(absent.tool_calls.size() == 1 && absent.tool_calls[0].arguments_json == "{}",
                      "absent arguments become an empty object");
    failures += check(null_args.tool_calls.size() == 1 &&
                          null_args.tool_calls[0].arguments_json == "{}",
                      "null arguments become an empty object");
    return failures;
}

int test_name_limits() {
    int failures             = 0;
    const std::string name   = std::string(128, 'a');
    const std::string xml    = "<tool_call>\n<function=" + name + ">\n</function>\n</tool_call>";
    const std::string hermes = "<tool_call>\n{\"name\":\"" + name + "\",\"arguments\":{}}\n</tool_call>";
    const std::string too_long =
        "<tool_call>\n<function=" + std::string(129, 'a') + ">\n</function>\n</tool_call>";

    const ToolCallStreamResult xml_anthropic = parse_all(xml, failures, 128);
    const ToolCallStreamResult xml_openai    = parse_all(xml, failures, 64);
    const ToolCallStreamResult xml_too_long  = parse_all(too_long, failures, 128);
    const ToolCallStreamResult hermes_anthropic = parse_all(hermes, failures, 128);
    const ToolCallStreamResult hermes_openai    = parse_all(hermes, failures, 64);

    failures += check(xml_anthropic.tool_calls.size() == 1 &&
                          xml_anthropic.tool_calls[0].name == name &&
                          xml_anthropic.tool_calls[0].arguments_json == "{}",
                      "128-character name accepted with the Anthropic limit");
    failures += check(xml_openai.tool_calls.empty() && xml_openai.content == xml,
                      "128-character name rejected with the OpenAI limit");
    failures += check(xml_too_long.tool_calls.empty() && xml_too_long.content == too_long,
                      "129-character name rejected with the Anthropic limit");
    failures += check(hermes_anthropic.tool_calls.size() == 1,
                      "hermes 128-character name accepted at 128");
    failures += check(hermes_openai.tool_calls.empty() && hermes_openai.content == hermes,
                      "hermes 128-character name rejected at 64");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_parameter_kinds();
    failures += test_split_invariance();
    failures += test_announce_timing();
    failures += test_string_round_trip();
    failures += test_raw_container();
    failures += test_buffered_values();
    failures += test_pre_announce_fallback();
    failures += test_text_after_call_is_dropped();
    failures += test_incomplete_calls();
    failures += test_bare_tool_close_completes();
    failures += test_hermes_calls();
    failures += test_name_limits();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
