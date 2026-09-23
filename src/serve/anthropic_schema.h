#pragma once

// Anthropic Messages API wire-format layer: parses /v1/messages request JSON into
// the internal GenerationRequest and serializes internal results back into
// Anthropic message bodies / SSE events. This is a sibling of openai_schema.h;
// both map to the same wire-agnostic GenerationRequest / GenerationOutcome, so the
// engine and generation service below know nothing about either protocol.

#include "serve/request.h"
#include "ninfer/types.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::serve {

// Parse an already-decoded Anthropic Messages body into a GenerationRequest.
// `system` becomes a leading system turn; user tool_result blocks become tool
// turns; assistant tool_use blocks become tool calls. Throws ApiException on
// malformed / unsupported requests. The `model` field is accepted verbatim (any
// Claude model name) and echoed back, never validated against the loaded model.
GenerationRequest parse_messages_request(const nlohmann::json& body, const RequestLimits& limits);

// Map an internal finish reason onto the Anthropic stop_reason wire value. `tool_use` requires
// that the output produced calls and completed every one; a call the output cut short leaves the
// engine's reason (`max_tokens`), as Anthropic reports a truncated tool_use block.
const char* messages_stop_reason(ninfer::FinishReason reason, bool tool_calls_completed);

// Non-streaming Messages response body (JSON string). Content blocks are emitted
// in order: an optional `thinking` block (from reasoning), an optional `text`
// block (from content), then a `tool_use` block per tool call. When nothing was
// produced an empty text block is emitted so `content` is never empty.
std::string make_messages_response(const std::string& id, const std::string& model,
                                   const std::string& content, const std::string& reasoning,
                                   const std::vector<ToolCall>& tool_calls, const char* stop_reason,
                                   const CompletionUsage& usage);

// Stateful encoder for one Messages SSE stream ("event: <type>\ndata: {...}\n\n"). Content
// blocks open and close in output order with contiguous indices: thinking, then text, then one
// tool_use block per call. A tool_use block starts with an empty `input`, streams
// `input_json_delta` fragments that concatenate to the call's arguments, and stops as soon as
// the model closes the call. finish() stops a block the output cut short, emits an empty text
// block when none was produced, and ends with `message_delta` and `message_stop`.
// `message_start` carries the prompt size before admission; the final `message_delta` reports
// the cumulative usage with the prompt split into uncached, cache-read and cache-written tokens.
class MessagesEventStream {
public:
    MessagesEventStream(std::string id, std::string model, int input_tokens);

    std::vector<std::string> start();
    std::vector<std::string> reasoning_delta(const std::string& text);
    std::vector<std::string> content_delta(const std::string& text);
    std::vector<std::string> tool_call_begin(std::size_t index, const ToolCall& call);
    std::vector<std::string> tool_call_arguments(std::size_t index, const std::string& delta);
    std::vector<std::string> tool_call_end(std::size_t index);
    std::vector<std::string> finish(ninfer::FinishReason reason, const CompletionUsage& usage);

private:
    enum class Block : std::uint8_t { None, Thinking, Text, ToolUse };

    void require_streaming() const;
    void require_open_call(std::size_t index) const;
    void open_block(Block block, std::vector<std::string>& events);
    void stop_open_block(std::vector<std::string>& events);

    std::string id_;
    std::string model_;
    int input_tokens_           = 0;
    int next_index_             = 0;
    int open_index_             = -1;
    Block open_                 = Block::None;
    std::size_t calls_begun_    = 0;
    std::size_t calls_complete_ = 0;
    bool started_               = false;
    bool finished_              = false;
};

std::string make_messages_ping();

// Error object body (Anthropic shape) and its SSE `event: error` form.
std::string make_messages_error_body(const ApiError& error);
std::string messages_sse_error_event(const ApiError& error);

// /v1/messages/count_tokens response body.
std::string make_count_tokens_response(int input_tokens);

// Message identifier ("msg_...").
std::string new_message_id();

} // namespace ninfer::serve
