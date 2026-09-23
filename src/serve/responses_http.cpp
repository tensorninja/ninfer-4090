#include "serve/http_server.h"

#include "serve/console_log.h"
#include "serve/openai_schema.h"
#include "serve/responses_schema.h"
#include "serve/responses_state.h"
#include "serve/responses_transport.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

class ClientDisconnected final : public std::exception {
public:
    [[nodiscard]] const char* what() const noexcept override { return "client disconnected"; }
};

struct StreamingResponse {
    PreparedRequest prepared;
    ResponsesRequest request;
    ResponseContext previous_context;
    RequestLogContext log_context;
    std::unique_ptr<ResponsesEventStream> encoder;
    std::atomic<bool> cancelled{false};
    bool started = false;
};

void write_error(httplib::Response& response, const ApiError& error) {
    ApiError rendered = error;
    if (rendered.status == 500) {
        rendered.type = "server_error";
        rendered.code.clear();
        rendered.param.clear();
        rendered.message = "Internal server error.";
    }
    response.status = rendered.status;
    if (rendered.status == 429 || rendered.status == 503) {
        response.set_header("Retry-After", "1");
    }
    response.set_content(make_error_body(rendered), "application/json");
}

ApiError responses_error(ApiError error) {
    if (error.param == "messages") { error.param = "input"; }
    return error;
}

ApiError internal_error() {
    ApiError error;
    error.status  = 500;
    error.type    = "server_error";
    error.message = "Internal server error.";
    return error;
}

ApiError request_cancelled_error() {
    ApiError error;
    error.status  = 499;
    error.type    = "request_cancelled";
    error.code    = "request_cancelled";
    error.message = "Request was cancelled.";
    return error;
}

ApiError response_not_found(const std::string& id) {
    ApiError error;
    error.status  = 404;
    error.type    = "invalid_request_error";
    error.param   = "response_id";
    error.code    = "response_not_found";
    error.message = "response '" + id + "' not found";
    return error;
}

Json parse_json_body(const httplib::Request& request) {
    try {
        return Json::parse(request.body);
    } catch (const std::exception&) {
        ApiError error;
        error.status  = 400;
        error.type    = "invalid_request_error";
        error.message = "request body is not valid JSON";
        throw ApiException(std::move(error));
    }
}

bool disconnected(const httplib::Request& request) {
    return request.is_connection_alive && !request.is_connection_alive();
}

bool disconnected(const httplib::DataSink& sink, const StreamingResponse& request) {
    return request.cancelled.load(std::memory_order_acquire) ||
           (sink.is_writable && !sink.is_writable());
}

void write_stream_item(httplib::DataSink& sink, StreamingResponse& request,
                       const std::string& item) {
    if (request.cancelled.load(std::memory_order_acquire) ||
        (sink.is_writable && !sink.is_writable()) || !sink.write(item.data(), item.size())) {
        request.cancelled.store(true, std::memory_order_release);
        throw ClientDisconnected();
    }
}

void write_stream_items(httplib::DataSink& sink, StreamingResponse& request,
                        std::vector<std::string> items) {
    for (const std::string& item : items) { write_stream_item(sink, request, item); }
}

void set_owned_content(httplib::Response& response, std::string body,
                       std::shared_ptr<RequestLifetime> lifetime) {
    response.set_content(std::move(body), "application/json");
    response.hold_resource(std::move(lifetime));
}

ResponseContext terminal_context(const ResponseContext& previous, const ResponsesRequest& request,
                                 const BuiltResponse& response) {
    ResponseContext input = append_response_context(previous, request.input_turns);
    return append_response_context(std::move(input), response.output_history);
}

ResponsesRuntimeValues runtime_values(const PreparedRequest& prepared,
                                      const GenerationOutcome* outcome = nullptr) {
    ResponsesRuntimeValues runtime;
    runtime.temperature = prepared.sampling.temperature;
    runtime.top_p       = prepared.sampling.top_p;
    if (outcome != nullptr) {
        runtime.cached_input_tokens = static_cast<int>(outcome->metrics.prefix_cache_hit_tokens);
    }
    return runtime;
}

std::string path_response_id(const httplib::Request& request) {
    return request.matches.size() > 1 ? request.matches[1].str() : std::string();
}

std::vector<std::pair<std::string, std::string>> query_parameters(const httplib::Request& request) {
    return {request.params.begin(), request.params.end()};
}

Json paginated_input_items(const RetrievalQuery& query, const std::vector<Json>& stored_items) {
    std::vector<Json> ordered = stored_items;
    if (query.order == "desc") { std::reverse(ordered.begin(), ordered.end()); }
    std::size_t begin = 0;
    if (query.after) {
        const auto found = std::find_if(ordered.begin(), ordered.end(), [&](const Json& item) {
            return item.contains("id") && item.at("id").is_string() &&
                   item.at("id").get<std::string>() == *query.after;
        });
        if (found == ordered.end()) {
            ApiError error;
            error.status  = 400;
            error.param   = "after";
            error.code    = "invalid_pagination";
            error.message = "after does not identify an input Item in this response";
            throw ApiException(std::move(error));
        }
        begin = static_cast<std::size_t>(std::distance(ordered.begin(), found)) + 1;
    }
    const std::size_t end = std::min(ordered.size(), begin + static_cast<std::size_t>(query.limit));
    Json data             = Json::array();
    for (std::size_t index = begin; index < end; ++index) { data.push_back(ordered[index]); }
    return Json{{"object", "list"},
                {"data", data},
                {"first_id", data.empty() ? Json(nullptr) : data.front().at("id")},
                {"last_id", data.empty() ? Json(nullptr) : data.back().at("id")},
                {"has_more", end < ordered.size()}};
}

} // namespace

void HttpServer::handle_responses(const httplib::Request& req, httplib::Response& res) {
    ResponsesRequest request;
    ResponseContext previous_context;
    PreparedRequest prepared;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        Json body                 = parse_json_body(req);
        resolve_response_item_references(body, response_store_);
        request                   = parse_responses_request(body, limits);
        request.generation.adapter = require_model(request.generation.model);
        if (request.previous_response_id) {
            const std::shared_ptr<const StoredResponse> previous =
                response_store_.get(*request.previous_response_id);
            if (!previous) {
                throw ApiException(response_not_found(*request.previous_response_id));
            }
            inherit_responses_preserve_thinking(request, previous->preserve_thinking);
            previous_context = previous->context;
        }
        compose_responses_generation_messages(request, flatten_response_context(previous_context));
        prepared = service_->prepare(request.generation, [&req] { return disconnected(req); });
    } catch (const ApiException& exception) {
        write_error(res, responses_error(exception.error()));
        return;
    } catch (const std::exception& exception) {
        write_console_log(ConsoleLogLevel::Error,
                          "responses preparation exception: " + std::string(exception.what()));
        write_error(res, internal_error());
        return;
    }

    const std::string id       = new_response_id();
    const std::int64_t created = unix_time_now();
    const std::uint64_t req_id = ++request_seq_;
    const RequestLogContext log_context =
        make_request_log_context(req_id, res.get_header_value("x-request-id"), "openai_responses",
                                 request.generation, prepared);
    log_request_start(log_context);

    if (!request.stream) {
        try {
            const GenerationOutcome outcome =
                service_->run(prepared, nullptr, [&req] { return disconnected(req); });
            const ResponseCommitAction commit =
                response_commit_action(disconnected(req), outcome.finish_reason, request.store);
            if (commit == ResponseCommitAction::FailCancelled) {
                throw ApiException(request_cancelled_error());
            }
            const ResponsesRuntimeValues runtime = runtime_values(prepared, &outcome);
            BuiltResponse response = make_response_object(id, created, request, runtime, outcome);
            // Storage is the success commit point. Re-check immediately before mutation; after
            // put succeeds, a later socket disconnect must not erase the durable record.
            if (commit == ResponseCommitAction::StoreThenSend) {
                if (response_commit_action(disconnected(req), outcome.finish_reason, true) ==
                    ResponseCommitAction::FailCancelled) {
                    throw ApiException(request_cancelled_error());
                }
                StoredResponse stored;
                stored.id                = id;
                stored.response          = response.body;
                stored.input_items       = request.input_items;
                stored.context           = terminal_context(previous_context, request, response);
                stored.preserve_thinking = prepared.preserve_thinking;
                response_store_.put(std::move(stored));
            }
            log_request_done(log_context, outcome);
            set_owned_content(res, response.body.dump(), prepared.lifetime);
        } catch (const ApiException& exception) {
            const ApiError error = responses_error(exception.error());
            log_request_error(log_context, error.message);
            write_error(res, error);
        } catch (const std::exception& exception) {
            log_request_error(log_context, exception.what());
            write_error(res, internal_error());
        }
        return;
    }

    auto stream              = std::make_shared<StreamingResponse>();
    stream->prepared         = std::move(prepared);
    stream->request          = std::move(request);
    stream->previous_context = std::move(previous_context);
    stream->log_context      = log_context;
    stream->encoder          = std::make_unique<ResponsesEventStream>(id, created, stream->request,
                                                                      runtime_values(stream->prepared));

    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");
    res.set_chunked_content_provider(
        "text/event-stream",
        [this, stream](std::size_t, httplib::DataSink& sink) -> bool {
            if (stream->started) {
                sink.done();
                return true;
            }
            stream->started = true;
            try {
                write_stream_items(sink, *stream, stream->encoder->start());
                StreamSink output;
                output.on_reasoning = [&](const std::string& text) {
                    write_stream_items(sink, *stream, stream->encoder->reasoning_delta(text));
                };
                output.on_content = [&](const std::string& text) {
                    write_stream_items(sink, *stream, stream->encoder->content_delta(text));
                };
                output.on_tool_call_begin = [&](std::size_t index, const ToolCall& call) {
                    write_stream_items(sink, *stream, stream->encoder->tool_call_begin(index, call));
                };
                output.on_tool_call_arguments = [&](std::size_t index, const std::string& delta) {
                    write_stream_items(sink, *stream,
                                       stream->encoder->tool_call_arguments(index, delta));
                };
                output.on_tool_call_end = [&](std::size_t index, const ToolCall& call) {
                    write_stream_items(sink, *stream, stream->encoder->tool_call_end(index, call));
                };
                output.on_prompt_progress = [&](const ninfer::PromptProgress& progress) {
                    write_stream_items(sink, *stream, stream->encoder->prompt_progress(progress));
                };
                output.is_cancelled = [&] {
                    return stream->cancelled.load(std::memory_order_acquire) ||
                           (sink.is_writable && !sink.is_writable());
                };

                const GenerationOutcome outcome = service_->run(stream->prepared, &output);
                if (response_commit_action(disconnected(sink, *stream), outcome.finish_reason,
                                           stream->request.store) ==
                    ResponseCommitAction::FailCancelled) {
                    throw ApiException(request_cancelled_error());
                }
                ResponsesStreamFinish finished    = stream->encoder->finish(outcome);
                const ResponseCommitAction commit = response_commit_action(
                    disconnected(sink, *stream), outcome.finish_reason, stream->request.store);
                if (commit == ResponseCommitAction::FailCancelled) {
                    throw ApiException(request_cancelled_error());
                }
                // Items went out live as the model wrote them; finish() only stages the events
                // that close what is still open. Those and the terminal reach the wire only after
                // put() succeeds, so a storage failure can emit exactly one response.failed
                // terminal.
                if (commit == ResponseCommitAction::StoreThenSend) {
                    StoredResponse stored;
                    stored.id          = finished.response.body.at("id").get<std::string>();
                    stored.response    = finished.response.body;
                    stored.input_items = stream->request.input_items;
                    stored.context     = terminal_context(stream->previous_context, stream->request,
                                                          finished.response);
                    stored.preserve_thinking = stream->prepared.preserve_thinking;
                    response_store_.put(std::move(stored));
                }
                // The record is committed now. Any disconnect while flushing staged events keeps
                // it retrievable, and ResponsesEventStream enforces at most one terminal event.
                write_stream_items(sink, *stream, std::move(finished.events_before_terminal));
                log_request_done(stream->log_context, outcome);
                write_stream_item(sink, *stream, stream->encoder->terminal(finished.response));
                sink.done();
                return true;
            } catch (const ClientDisconnected& exception) {
                log_request_error(stream->log_context, exception.what());
                return false;
            } catch (const ApiException& exception) {
                const ApiError reported = responses_error(exception.error());
                const ApiError error    = reported.status == 500 ? internal_error() : reported;
                log_request_error(stream->log_context, reported.message);
                try {
                    write_stream_item(sink, *stream, stream->encoder->failed(error));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            } catch (const std::exception& exception) {
                const ApiError error = internal_error();
                log_request_error(stream->log_context, exception.what());
                try {
                    write_stream_item(sink, *stream, stream->encoder->failed(error));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            }
        },
        [stream](bool) { stream->cancelled.store(true, std::memory_order_release); });
}

void HttpServer::handle_response_input_tokens(const httplib::Request& req, httplib::Response& res) {
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        Json body                 = parse_json_body(req);
        resolve_response_item_references(body, response_store_);
        ResponsesRequest request = parse_response_input_tokens_request(body, limits);
        // Token counting is adapter-independent, but an unknown `model` must still 404.
        static_cast<void>(require_model(request.generation.model));
        const int tokens =
            service_->count_prompt_tokens(request.generation, [&req] { return disconnected(req); });
        res.set_content(make_response_input_tokens_body(tokens), "application/json");
    } catch (const ApiException& exception) {
        write_error(res, responses_error(exception.error()));
    } catch (const std::exception& exception) {
        write_console_log(ConsoleLogLevel::Error,
                          "response input_tokens exception: " + std::string(exception.what()));
        write_error(res, internal_error());
    }
}

void HttpServer::handle_response_get(const httplib::Request& req, httplib::Response& res) {
    try {
        (void)parse_retrieval_query(RetrievalRoute::Response, query_parameters(req));
    } catch (const ApiException& exception) {
        write_error(res, exception.error());
        return;
    }
    const std::string id                               = path_response_id(req);
    const std::shared_ptr<const StoredResponse> stored = response_store_.get(id);
    if (!stored) {
        write_error(res, response_not_found(id));
        return;
    }
    res.set_content(stored->response.dump(), "application/json");
}

void HttpServer::handle_response_delete(const httplib::Request& req, httplib::Response& res) {
    try {
        (void)parse_retrieval_query(RetrievalRoute::DeleteResponse, query_parameters(req));
    } catch (const ApiException& exception) {
        write_error(res, exception.error());
        return;
    }
    const std::string id = path_response_id(req);
    if (!response_store_.erase(id)) {
        write_error(res, response_not_found(id));
        return;
    }
    res.set_content(Json{{"id", id}, {"object", "response.deleted"}, {"deleted", true}}.dump(),
                    "application/json");
}

void HttpServer::handle_response_input_items(const httplib::Request& req, httplib::Response& res) {
    RetrievalQuery query;
    try {
        query = parse_retrieval_query(RetrievalRoute::InputItems, query_parameters(req));
    } catch (const ApiException& exception) {
        write_error(res, exception.error());
        return;
    }
    const std::string id                               = path_response_id(req);
    const std::shared_ptr<const StoredResponse> stored = response_store_.get(id);
    if (!stored) {
        write_error(res, response_not_found(id));
        return;
    }
    try {
        res.set_content(paginated_input_items(query, stored->input_items).dump(),
                        "application/json");
    } catch (const ApiException& exception) { write_error(res, exception.error()); }
}

void HttpServer::handle_response_cancel(const httplib::Request& req, httplib::Response& res) {
    try {
        (void)parse_retrieval_query(RetrievalRoute::CancelResponse, query_parameters(req));
    } catch (const ApiException& exception) {
        write_error(res, exception.error());
        return;
    }
    const std::string id = path_response_id(req);
    if (!response_store_.get(id)) {
        write_error(res, response_not_found(id));
        return;
    }
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.code    = "background_not_supported";
    error.message = "only background responses can be cancelled; NInfer does not support "
                    "background execution";
    write_error(res, error);
}

void HttpServer::handle_response_compact(const httplib::Request&, httplib::Response& res) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.param   = "context_management";
    error.code    = "compaction_not_supported";
    error.message = "Responses compaction is not supported";
    write_error(res, error);
}

} // namespace ninfer::serve
