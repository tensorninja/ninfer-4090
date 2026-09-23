#pragma once

#include "ninfer/types.h"
#include "product/media_acquire/source.h"

// Internal, wire-format-independent representation of a generation request.
//
// OpenAI and Anthropic schemas both map into this wire-independent value.
// translate.cpp then produces the public PromptInput and RequestOptions consumed
// by Engine; media sources remain unresolved until the product service acquires
// owning bytes.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// A structured API error mapped onto an error object + HTTP status. Wire-format
// independent: each protocol layer renders it into its own error body shape.
struct ApiError {
    int status       = 400;
    std::string type = "invalid_request_error";
    std::string message;
    std::string param; // optional
    std::string code;  // optional
};

class ApiException : public std::runtime_error {
public:
    explicit ApiException(ApiError error)
        : std::runtime_error(error.message), error_(std::move(error)) {}

    [[nodiscard]] const ApiError& error() const noexcept { return error_; }

private:
    ApiError error_;
};

// Server-side context needed while parsing/validating a request.
struct RequestLimits {
    int default_max_tokens = 8192;
};

struct CompletionUsage {
    int prompt_tokens     = 0;
    int completion_tokens = 0;

    // llama.cpp-compatible `timings` block data, emitted so proxies like
    // llama-swap can derive per-request Prefill/Decode rates and draft stats.
    // has_timings gates emission; seconds values are wall-clock phase times.
    bool has_timings          = false;
    double prefill_seconds    = 0.0;
    double decode_seconds     = 0.0;
    double ttft_seconds       = 0.0;
    std::int64_t cache_hit_tokens  = 0;
    // Prompt tokens this request captured for content-addressed reuse beyond what it restored.
    std::int64_t cache_write_tokens = 0;
    std::uint64_t draft_tokens     = 0;
    std::uint64_t accepted_tokens  = 0;

    // Slot identity extras emitted next to `timings`: the lane that served the request
    // (id_slot >= 0) and its retained session's digest, the handle for /slots save
    // preconditions. Both are omitted from payloads when absent.
    int id_slot = -1;
    std::string session_digest;
};

enum class ContentKind {
    Text,
    Image,
    Video,
    InputAudio,
    Unsupported,
};

struct ContentPart {
    ContentKind kind = ContentKind::Text;
    std::string text;     // populated for Text
    std::string type_raw; // original OpenAI "type" string (diagnostics / future use)
    ninfer::product::media_acquire::Source source;
    // Client prompt-cache breakpoint at the end of this part (OpenAI `prompt_cache_breakpoint`,
    // Anthropic `cache_control`).
    bool cache_breakpoint = false;
};

// Both wire contracts allow this many breakpoints per request.
constexpr std::size_t kMaxPromptCacheBreakpoints = 4;

struct ToolDefinition {
    std::string name;
    std::string description;
    std::string parameters_json;
    std::string definition_json; // normalized OpenAI function-tool object for Qwen prompt rendering
    bool strict = false;
};

// A generated call is Incomplete when the output ended before the model closed it. Its arguments
// are then exactly the prefix that was streamed, with no closing added, so they are not a JSON
// object and cannot be replayed as history.
enum class ToolCallState : std::uint8_t {
    Complete,
    Incomplete,
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
    ToolCallState state = ToolCallState::Complete;
};

// Whether generated calls hand the turn to the client: at least one call, and every one closed.
// Only then do the protocols report a tool-call finish instead of the engine's finish reason.
[[nodiscard]] inline bool tool_calls_completed(const std::vector<ToolCall>& calls) noexcept {
    for (const ToolCall& call : calls) {
        if (call.state != ToolCallState::Complete) { return false; }
    }
    return !calls.empty();
}

enum class ToolChoiceMode {
    Auto,
    None,
    Required,
    Named,
};

struct ToolChoice {
    ToolChoiceMode mode = ToolChoiceMode::Auto;
    std::string name;
};

struct ChatTurn {
    std::string role; // system | user | assistant | tool (validated in translate)
    std::vector<ContentPart>
        content; // one or more parts; assistant content may be empty with tool_calls
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;      // populated for role=tool
    std::string reasoning_content; // assistant thinking carried across turns (round-tripped to the
                                   // template)
};

struct SamplingParams {
    std::optional<double> temperature;
    std::optional<double> top_p;
    std::optional<int> top_k;
    std::optional<double> presence_penalty;
    std::optional<double> frequency_penalty;
    std::optional<std::uint64_t> seed;
    int n = 1;
};

// Protocol-level effort vocabulary. Each wire adapter accepts the values from
// its external contract; translation then resolves them against the capabilities
// advertised by the chat template embedded in the loaded artifact.
enum class RequestedReasoningEffort : std::uint8_t {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
};

[[nodiscard]] constexpr std::optional<RequestedReasoningEffort>
parse_requested_reasoning_effort(std::string_view value) noexcept {
    if (value == "none") { return RequestedReasoningEffort::None; }
    if (value == "minimal") { return RequestedReasoningEffort::Minimal; }
    if (value == "low") { return RequestedReasoningEffort::Low; }
    if (value == "medium") { return RequestedReasoningEffort::Medium; }
    if (value == "high") { return RequestedReasoningEffort::High; }
    if (value == "xhigh") { return RequestedReasoningEffort::XHigh; }
    if (value == "max") { return RequestedReasoningEffort::Max; }
    return std::nullopt;
}

[[nodiscard]] constexpr std::string_view
requested_reasoning_effort_name(RequestedReasoningEffort effort) noexcept {
    switch (effort) {
    case RequestedReasoningEffort::None:
        return "none";
    case RequestedReasoningEffort::Minimal:
        return "minimal";
    case RequestedReasoningEffort::Low:
        return "low";
    case RequestedReasoningEffort::Medium:
        return "medium";
    case RequestedReasoningEffort::High:
        return "high";
    case RequestedReasoningEffort::XHigh:
        return "xhigh";
    case RequestedReasoningEffort::Max:
        return "max";
    }
    return {};
}

struct GenerationRequest {
    std::string model;
    // Registered LoRA adapter the `model` string selected; empty selects the base weights.
    std::string adapter;
    std::vector<ChatTurn> messages;
    std::vector<ToolDefinition> tools;
    std::size_t tool_name_max_length = 64;
    ToolChoice tool_choice;
    std::vector<std::string> stop_strings;
    int max_tokens      = 0; // 0 => use server default
    bool max_tokens_set = false;
    bool stream         = false;
    bool include_usage  = false;
    std::optional<bool> enable_thinking; // non-standard extension; falls back to server default
    std::optional<RequestedReasoningEffort> reasoning_effort;
    std::string reasoning_effort_param = "reasoning_effort";
    std::optional<bool> preserve_thinking;
    bool preserve_thinking_semantic_change = false;
    std::optional<std::string> prompt_cache_routing_hint;
    ninfer::PromptCacheMode prompt_cache_mode = ninfer::PromptCacheMode::Implicit;
    SamplingParams sampling;

    [[nodiscard]] bool uses_tools() const noexcept {
        return !tools.empty() && tool_choice.mode != ToolChoiceMode::None;
    }

    [[nodiscard]] bool has_tool_history() const noexcept {
        for (const ChatTurn& message : messages) {
            if (!message.tool_calls.empty() || message.role == "tool") { return true; }
        }
        return false;
    }
};

} // namespace ninfer::serve
