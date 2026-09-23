#pragma once

// OpenAI wire-format layer: parses request JSON into the internal GenerationRequest
// and serializes internal results back into OpenAI Chat Completions bodies/chunks.
// This layer knows nothing about the engine; it only speaks the OpenAI schema.

#include "serve/request.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::serve {

// ApiError, ApiException, RequestLimits, and CompletionUsage are the wire-format
// independent request/error types; they live in request.h and are shared by the
// OpenAI and Anthropic schema layers.

// Parse an already-decoded JSON body into a GenerationRequest. Throws ApiException
// on malformed or unsupported requests (n>1, tools, non-text response_format, ...).
GenerationRequest parse_chat_completion_request(const nlohmann::json& body,
                                                const RequestLimits& limits);

std::optional<bool> parse_openai_preserve_thinking(const nlohmann::json& body);

// Non-streaming chat completion response body (JSON string). When `reasoning` is
// non-empty it is attached as `message.reasoning_content` (the DeepSeek/vLLM-style
// convention consumed by Chatbox, Open WebUI, etc.), leaving `content` = answer.
// When `usage.has_timings` is set, a llama.cpp-compatible top-level `timings`
// block is included so proxies (llama-swap) can derive Prefill/Decode rates.
std::string make_chat_completion_response(const std::string& id, const std::string& model,
                                          std::int64_t created, const std::string& content,
                                          const std::string& reasoning, const char* finish_reason,
                                          const CompletionUsage& usage);
// A response carrying tool calls. `finish_reason` is "tool_calls" only when every call completed;
// a call the output cut short keeps its partial arguments and the engine's reason ("length").
std::string make_chat_completion_tool_response(const std::string& id, const std::string& model,
                                               std::int64_t created, const std::string& content,
                                               const std::string& reasoning,
                                               const std::vector<ToolCall>& tool_calls,
                                               const char* finish_reason,
                                               const CompletionUsage& usage);

// Streaming SSE event strings ("data: {...}\n\n"). The first chunk carries the
// assistant role; reasoning chunks carry `reasoning_content` deltas (the <think>
// block), content chunks carry `content` deltas; the final chunk carries the
// finish_reason with an empty delta. Per the OpenAI stream_options contract, when
// usage reporting is enabled every content-bearing chunk carries `usage: null`
// and a single dedicated usage chunk (empty choices) is emitted before [DONE];
// pass include_usage accordingly.
std::string make_chat_chunk_role(const std::string& id, const std::string& model,
                                 std::int64_t created, bool include_usage);
std::string make_chat_chunk_reasoning(const std::string& id, const std::string& model,
                                      std::int64_t created, const std::string& delta_text,
                                      bool include_usage);
std::string make_chat_chunk_content(const std::string& id, const std::string& model,
                                    std::int64_t created, const std::string& delta_text,
                                    bool include_usage);
// A streamed tool call, in OpenAI's grammar: the first chunk for call `index` carries its id,
// type and name with empty arguments; later chunks carry only argument fragments. The fragments
// concatenate to the call's arguments, whose closing `}` arrives only when the call completes.
std::string make_chat_chunk_tool_call_begin(const std::string& id, const std::string& model,
                                            std::int64_t created, std::size_t index,
                                            const ToolCall& call, bool include_usage);
std::string make_chat_chunk_tool_call_arguments(const std::string& id, const std::string& model,
                                                std::int64_t created, std::size_t index,
                                                const std::string& delta, bool include_usage);
// Final chunk: `delta: {}` with finish_reason. When `usage.has_timings` is set a
// top-level `timings` block is included (matching llama.cpp's stream convention)
// so proxies can derive per-request rates from the final chunk alone.
std::string make_chat_chunk_final(const std::string& id, const std::string& model,
                                  std::int64_t created, const char* finish_reason,
                                  bool include_usage, const CompletionUsage& usage = {});
// Dedicated usage chunk: `choices: []` with the request's token usage. Emitted
// only when stream_options.include_usage is true.
std::string make_chat_chunk_usage(const std::string& id, const std::string& model,
                                  std::int64_t created, const CompletionUsage& usage);
std::string sse_done();

// /v1/models payloads. `context_window` is the serving max-context, reported
// so clients can size prompts without a llama.cpp /props or vLLM
// max_model_len to read. `modalities` mirrors the llama.cpp /props shape so
// clients can tell a vision-enabled server from a text-only one behind the
// same model id; a server without the field is read as text-only.
std::string make_models_list(const std::string& model_id,
                             const std::vector<std::string>& adapter_model_ids,
                             std::int64_t created, std::uint32_t context_window, bool vision);
std::string make_model_object(const std::string& model_id, std::int64_t created,
                              std::uint32_t context_window, bool vision);

// Error object body.
std::string make_error_body(const ApiError& error);

// Identifiers / timestamps.
std::string new_chat_completion_id();
std::int64_t unix_time_now();

} // namespace ninfer::serve
