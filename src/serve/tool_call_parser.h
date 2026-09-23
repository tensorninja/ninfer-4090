#pragma once

#include "serve/request.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// How the XML dialect's text value of one declared parameter becomes its JSON argument. The kind
// comes from the request's tool schema: the dialect itself writes every value as bare text.
enum class ToolParameterKind : std::uint8_t {
    // Type set, ignoring null, is exactly {string}: streamed as an escaped JSON string.
    String,
    // Type set, ignoring null, lies within {object, array}: once the value opens with `{` or
    // `[`, the model's JSON is streamed as written; any other opening is Buffered.
    JsonContainer,
    // Everything else, including undeclared tools and parameters: held until </parameter>, then
    // emitted as the value's JSON when it parses and as a JSON string otherwise.
    Buffered,
};

class ToolParameterKinds {
public:
    [[nodiscard]] static ToolParameterKinds from_tools(const std::vector<ToolDefinition>& tools);

    [[nodiscard]] ToolParameterKind kind(std::string_view tool, std::string_view parameter) const;

private:
    std::map<std::string, std::map<std::string, ToolParameterKind, std::less<>>, std::less<>>
        tools_;
};

// Receives the parser's output as soon as it is final. `content` is text outside tool calls. An
// announced call receives `tool_call_begin` (id and name known, no arguments yet), argument
// deltas whose concatenation is its `arguments_json`, and `tool_call_end` once it completes. A
// call the output cut short never receives `tool_call_end`.
class ToolCallEvents {
public:
    virtual ~ToolCallEvents() = default;

    virtual void content(std::string_view text)                                 = 0;
    virtual void tool_call_begin(std::size_t index, const ToolCall& call)       = 0;
    virtual void tool_call_arguments(std::size_t index, std::string_view delta) = 0;
    virtual void tool_call_end(std::size_t index, const ToolCall& call)         = 0;
};

struct ToolCallStreamResult {
    std::string content;
    std::vector<ToolCall> tool_calls;
    // Non-whitespace bytes dropped because they followed the first announced call outside a
    // well-formed call: the template allows reasoning before a call but no suffix.
    std::size_t discarded_bytes = 0;
};

// Incremental parser for Qwen tool calls in the content channel. An XML call is announced once
// `<tool_call>`, optional whitespace and `<function=NAME>` have arrived and NAME is valid; its
// arguments then stream as a JSON object whose closing `}` arrives only with `</function>` or a
// bare `</tool_call>`. A Hermes JSON block is validated whole at `</tool_call>` and then emitted
// as begin, one argument delta, and end. Before any call is announced, markup that turns out not
// to be a call is emitted as content with no byte lost; after one, everything outside a call is
// dropped and counted. Content and each call's argument concatenation are independent of how
// the output is split into feeds, and every delta splits the input only at ASCII bytes, so UTF-8
// input stays UTF-8.
class QwenToolCallStream {
public:
    QwenToolCallStream(ToolParameterKinds kinds, std::size_t max_name_length,
                       ToolCallEvents* events = nullptr);

    void feed(std::string_view text);
    // Resolves every pending decision as end of output and returns the accumulated result. A
    // call the output ended inside stays Incomplete with exactly the arguments already emitted.
    [[nodiscard]] ToolCallStreamResult finish();

private:
    enum class State : std::uint8_t {
        Text,            // before any call: ordinary content
        Header,          // after <tool_call>, before <function= or a Hermes `{`
        FunctionName,    // reading NAME of <function=NAME>
        Hermes,          // Hermes JSON block, held until </tool_call>
        Body,            // inside an announced XML call, between parameters
        ParameterName,   // reading NAME of <parameter=NAME>
        StringValue,     // streaming a String value as escaped JSON string content
        ContainerStart,  // JsonContainer value before its first non-space byte
        RawValue,        // streaming a JsonContainer value as the model's JSON
        BufferedValue,   // holding a Buffered value until </parameter>
        CallClose,       // after </function>, expecting </tool_call>
        AfterCall,       // after a call: only further <tool_call> blocks are accepted
    };

    bool step();
    bool step_text();
    bool step_header();
    bool step_function_name();
    bool step_hermes();
    bool step_body();
    bool step_parameter_name();
    bool step_container_start();
    bool step_streamed_value(bool escape);
    bool step_buffered_value();
    bool step_call_close();
    bool step_after_call();

    void set_state(State state) noexcept;
    void consume_markup(std::size_t size);
    void reject_markup();
    void begin_call(std::string name);
    void begin_parameter(const std::string& name);
    void complete_call();
    void emit_content(std::string_view text);
    void emit_arguments(std::string_view delta);
    void discard(std::string_view text) noexcept;
    [[nodiscard]] std::size_t find_unscanned(std::string_view marker);

    ToolParameterKinds kinds_;
    std::size_t max_name_length_ = 0;
    ToolCallEvents* events_      = nullptr;
    State state_                 = State::Text;
    std::string buffer_;  // input not yet decided
    std::string markup_;  // consumed bytes of a block that is not yet a call
    std::string name_;    // function name read so far
    std::size_t scanned_  = 0;
    bool announced_       = false;
    bool first_parameter_ = true;
    bool value_started_   = false;
    bool at_end_          = false;
    bool finished_        = false;
    ToolCallStreamResult result_;
};

} // namespace ninfer::serve
