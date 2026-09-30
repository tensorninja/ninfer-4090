#include "serve/generation_service.h"

#include "product/media_acquire/acquire.h"
#include "serve/console_log.h"
#include "serve/tool_call_parser.h"
#include "serve/translate.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::serve {

struct RequestCapacity {
    explicit RequestCapacity(std::size_t limit) : maximum(limit) {}

    std::mutex mutex;
    std::size_t active = 0;
    const std::size_t maximum;
};

struct RequestLifetime {
    RequestLifetime(std::shared_ptr<RequestCapacity> owner,
                    std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point limit)
        : capacity(std::move(owner)), started(begin), deadline(limit) {}

    ~RequestLifetime() {
        std::lock_guard lock(capacity->mutex);
        --capacity->active;
    }

    std::shared_ptr<RequestCapacity> capacity;
    std::chrono::steady_clock::time_point started;
    std::chrono::steady_clock::time_point deadline;
};

struct MediaInputCapacity {
    std::mutex mutex;
    std::condition_variable cv;
    bool occupied = false;
};

struct MediaInputPermit {
    explicit MediaInputPermit(std::shared_ptr<MediaInputCapacity> owner)
        : capacity(std::move(owner)) {}

    ~MediaInputPermit() {
        {
            std::lock_guard lock(capacity->mutex);
            capacity->occupied = false;
        }
        capacity->cv.notify_one();
    }

    std::shared_ptr<MediaInputCapacity> capacity;
};

namespace {

using Clock                              = std::chrono::steady_clock;
constexpr std::size_t kMaximumMediaItems = 16;

[[noreturn]] void throw_preparation_cancelled();

[[noreturn]] void throw_media_error(const ninfer::product::media_acquire::Error& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninfer::product::media_acquire::ErrorKind::BudgetExceeded:
        error.status = 413;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteUnavailable:
        error.status = 502;
        error.type   = "server_error";
        error.code   = "media_fetch_failed";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteTimeout:
        error.status = 504;
        error.type   = "server_error";
        error.code   = "media_fetch_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::DeadlineExceeded:
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::Cancelled:
        throw_preparation_cancelled();
    }
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_invalid_input(const std::exception& exception,
                                      const char* code = "invalid_media") {
    ApiError error;
    error.status  = 400;
    error.param   = "messages";
    error.code    = code;
    error.message = exception.what();
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_preparation_cancelled() {
    ApiError error;
    error.status  = 499;
    error.type    = "request_cancelled";
    error.code    = "client_disconnected";
    error.message = "client disconnected during media preparation";
    throw ApiException(std::move(error));
}

std::size_t media_item_count(const GenerationRequest& request) {
    std::size_t count = 0;
    for (const ChatTurn& message : request.messages) {
        for (const ContentPart& part : message.content) {
            if (part.kind == ContentKind::Image || part.kind == ContentKind::Video) { ++count; }
        }
    }
    return count;
}

ninfer::OwnedMedia acquire_media(const ContentPart& part, Clock::time_point deadline,
                                 const std::function<bool()>& is_cancelled,
                                 std::size_t& remaining_bytes) {
    if (remaining_bytes == 0) {
        throw_media_error(ninfer::product::media_acquire::Error(
            ninfer::product::media_acquire::ErrorKind::BudgetExceeded,
            "request media exceeds aggregate byte limit"));
    }
    ninfer::product::media_acquire::Policy policy;
    policy.max_bytes    = std::min(policy.max_bytes, remaining_bytes);
    policy.deadline     = deadline;
    policy.is_cancelled = is_cancelled;
    std::vector<std::uint8_t> source_bytes;
    try {
        source_bytes = ninfer::product::media_acquire::acquire_bytes(part.source, policy);
    } catch (const ninfer::product::media_acquire::Error& exception) {
        throw_media_error(exception);
    } catch (const std::invalid_argument& exception) { throw_invalid_input(exception); }

    remaining_bytes -= source_bytes.size();
    ninfer::OwnedMedia media;
    media.kind =
        part.kind == ContentKind::Image ? ninfer::MediaKind::Image : ninfer::MediaKind::Video;
    media.media_type = part.source.media_type;
    switch (part.source.kind) {
    case ninfer::product::media_acquire::SourceKind::Path:
    case ninfer::product::media_acquire::SourceKind::Url:
        media.source_name = part.source.value;
        break;
    case ninfer::product::media_acquire::SourceKind::Data:
        media.source_name = "inline-data";
        break;
    case ninfer::product::media_acquire::SourceKind::Bytes:
        media.source_name = "inline-bytes";
        break;
    }
    media.bytes = std::move(source_bytes);
    return media;
}

[[noreturn]] void throw_request_error(const ninfer::RequestError& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninfer::RequestErrorKind::ContextLengthExceeded:
        error.status = 400;
        error.code   = "context_length_exceeded";
        break;
    case ninfer::RequestErrorKind::MediaBudgetExceeded:
        error.status = 413;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::RequestErrorKind::Overloaded:
        error.param.clear();
        error.status = 429;
        error.type   = "rate_limit_error";
        error.code   = "server_overloaded";
        break;
    case ninfer::RequestErrorKind::QueueTimeout:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::RequestErrorKind::Unavailable:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "service_unavailable";
        break;
    case ninfer::RequestErrorKind::UnknownAdapter:
        error.param  = "model";
        error.status = 404;
        error.code   = "model_not_found";
        break;
    case ninfer::RequestErrorKind::Cancelled:
        error.param.clear();
        error.status = 499;
        error.type   = "request_cancelled";
        error.code   = "client_disconnected";
        break;
    }
    throw ApiException(std::move(error));
}

void check_preparation_control(Clock::time_point deadline,
                               const std::function<bool()>& is_cancelled) {
    if (is_cancelled && is_cancelled()) { throw_preparation_cancelled(); }
    if (Clock::now() >= deadline) {
        throw_request_error(ninfer::RequestError(RequestErrorKind::QueueTimeout,
                                                 "inference request expired during preparation"));
    }
}

// Forwards engine output to a StreamSink. A tool-capable request's content runs through the
// incremental tool-call parser, so calls reach the client while the model is still writing them.
class ServiceOutputSink final : public ninfer::OutputSink, private ToolCallEvents {
public:
    ServiceOutputSink(const StreamSink& sink, const PreparedRequest& prepared) : sink_(&sink) {
        if (prepared.tool_capable) {
            tool_stream_.emplace(prepared.tool_parameters, prepared.tool_name_max_length,
                                 static_cast<ToolCallEvents*>(this));
        }
    }

    void publish(ninfer::OutputDelta delta) override {
        if (delta.text.empty()) { return; }
        if (delta.channel == ninfer::OutputChannel::Reasoning) {
            if (sink_->on_reasoning) { sink_->on_reasoning(delta.text); }
        } else if (tool_stream_) {
            tool_stream_->feed(delta.text);
        } else if (sink_->on_content) {
            sink_->on_content(delta.text);
        }
    }

    void publish_prompt_progress(ninfer::PromptProgress progress) override {
        if (sink_->on_prompt_progress) { sink_->on_prompt_progress(progress); }
    }

    // Ends the parse of a tool-capable request, publishing whatever the end of output decides.
    ToolCallStreamResult finish_tool_calls() { return tool_stream_->finish(); }

private:
    void content(std::string_view text) override {
        if (sink_->on_content) { sink_->on_content(std::string(text)); }
    }

    void tool_call_begin(std::size_t index, const ToolCall& call) override {
        if (sink_->on_tool_call_begin) { sink_->on_tool_call_begin(index, call); }
    }

    void tool_call_arguments(std::size_t index, std::string_view delta) override {
        if (sink_->on_tool_call_arguments) {
            sink_->on_tool_call_arguments(index, std::string(delta));
        }
    }

    void tool_call_end(std::size_t index, const ToolCall& call) override {
        if (sink_->on_tool_call_end) { sink_->on_tool_call_end(index, call); }
    }

    const StreamSink* sink_ = nullptr;
    std::optional<QwenToolCallStream> tool_stream_;
};

} // namespace

GenerationService::GenerationService(ServeOptions options, LoadProgress load_progress)
    : options_(std::move(options)) {
    ninfer::EngineOptions engine_options;
    engine_options.artifact_path        = options_.artifact_path;
    engine_options.device               = options_.device;
    engine_options.max_context          = options_.max_context;
    engine_options.kv_capacity          = options_.kv_capacity;
    engine_options.max_concurrency      = options_.max_concurrency;
    engine_options.max_pending_requests = options_.max_pending_requests;
    engine_options.pending_timeout_ms   = options_.pending_timeout_ms;
    engine_options.prefill_chunk        = options_.prefill_chunk;
    engine_options.prefill_decode_balance = options_.prefill_decode_balance;
    engine_options.turn_checkpoint_ring = options_.turn_checkpoint_ring;
    engine_options.auto_save_evicted    = options_.auto_save_evicted;
    if (options_.auto_save_evicted) {
        engine_options.auto_save_listener = [](const ninfer::SlotAutoSaveEvent& event) {
            if (event.error.empty()) {
                write_console_log(ConsoleLogLevel::Info,
                                  "slot auto-save file=" + event.path +
                                      " n_saved=" + std::to_string(event.tokens) +
                                      " bytes=" + std::to_string(event.bytes));
            } else {
                write_console_log(ConsoleLogLevel::Warning,
                                  "slot auto-save FAILED file=" + event.path + ": " + event.error);
            }
        };
    }
    engine_options.kv_cache             = options_.kv_cache;
    engine_options.prefix_checkpoint_policy = options_.prefix_checkpoint_policy;
    engine_options.continuation_cache   = options_.continuation_cache;
    engine_options.enable_vision        = options_.enable_vision;
    engine_options.vision_max_tokens    = options_.vision_max_tokens;
    engine_options.use_cuda_graph       = options_.use_cuda_graph;
    engine_options.lora                 = options_.lora;
    engine_options.speculative          = options_.speculative;
    engine_options.repetition_guard         = options_.repetition_guard;
    engine_options.load_progress        = std::move(load_progress);
    engine_              = std::make_unique<ninfer::Engine>(std::move(engine_options));
    prompt_capabilities_ = engine_->prompt_capabilities();
    request_capacity_    = std::make_shared<RequestCapacity>(
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests);
    media_input_capacity_ = std::make_shared<MediaInputCapacity>();
}

std::shared_ptr<RequestLifetime> GenerationService::acquire_request_lifetime() const {
    const auto started = Clock::now();
    {
        std::lock_guard lock(request_capacity_->mutex);
        if (request_capacity_->active >= request_capacity_->maximum) {
            throw ninfer::RequestError(RequestErrorKind::Overloaded,
                                       "inference request queue is full");
        }
        ++request_capacity_->active;
    }
    try {
        return std::make_shared<RequestLifetime>(
            request_capacity_, started,
            started + std::chrono::milliseconds(options_.pending_timeout_ms));
    } catch (...) {
        std::lock_guard lock(request_capacity_->mutex);
        --request_capacity_->active;
        throw;
    }
}

HostInputLease
GenerationService::acquire_media_input(Clock::time_point deadline,
                                       const std::function<bool()>& is_cancelled) const {
    std::unique_lock lock(media_input_capacity_->mutex);
    while (media_input_capacity_->occupied) {
        if (is_cancelled && is_cancelled()) { throw_preparation_cancelled(); }
        const Clock::time_point now = Clock::now();
        if (now >= deadline) {
            throw_request_error(ninfer::RequestError(
                RequestErrorKind::QueueTimeout,
                "inference request expired while waiting for media preparation"));
        }
        media_input_capacity_->cv.wait_until(
            lock, std::min(deadline, now + std::chrono::milliseconds(10)));
    }
    if (is_cancelled && is_cancelled()) { throw_preparation_cancelled(); }
    if (Clock::now() >= deadline) {
        throw_request_error(
            ninfer::RequestError(RequestErrorKind::QueueTimeout,
                                 "inference request expired while waiting for media preparation"));
    }

    media_input_capacity_->occupied = true;
    lock.unlock();
    try {
        auto permit = std::make_shared<MediaInputPermit>(media_input_capacity_);
        return HostInputLease(std::static_pointer_cast<void>(std::move(permit)));
    } catch (...) {
        {
            std::lock_guard capacity_lock(media_input_capacity_->mutex);
            media_input_capacity_->occupied = false;
        }
        media_input_capacity_->cv.notify_one();
        throw;
    }
}

PreparedRequest GenerationService::prepare(const GenerationRequest& request,
                                           std::function<bool()> is_cancelled) const {
    PreparedRequest prepared;
    ninfer::RequestOptions request_options = to_request_options(request, options_);
    prepared.include_usage                 = request.include_usage;
    prepared.tool_capable                  = request.uses_tools() || request.has_tool_history();
    prepared.tool_name_max_length          = request.tool_name_max_length;
    if (prepared.tool_capable) {
        prepared.tool_parameters = ToolParameterKinds::from_tools(request.tools);
    }
    const ResolvedPromptSemantics semantics =
        resolve_prompt_semantics(request, options_, prompt_capabilities_);
    prepared.enable_thinking                   = semantics.enable_thinking;
    prepared.preserve_thinking                 = semantics.preserve_thinking;
    prepared.preserve_thinking_semantic_change = request.preserve_thinking_semantic_change;
    prepared.prompt_cache_routing_hint         = request.prompt_cache_routing_hint;
    const std::size_t media_items              = media_item_count(request);
    const bool request_has_media               = media_items != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    if (media_items > kMaximumMediaItems) {
        throw_request_error(ninfer::RequestError(RequestErrorKind::MediaBudgetExceeded,
                                                 "request exceeds the 16-item media limit"));
    }
    HostInputLease host_input;
    try {
        prepared.lifetime = acquire_request_lifetime();
        if (request_has_media) {
            host_input = acquire_media_input(prepared.lifetime->deadline, is_cancelled);
        }
        std::size_t remaining_media_bytes = options_.max_request_bytes;
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, prepared.lifetime->deadline, is_cancelled,
                                     remaining_media_bytes);
            });
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        ninfer::PreparedPrompt prompt = engine_->prepare(std::move(input));
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        prepared.prompt_tokens = static_cast<int>(prompt.summary().prompt_tokens);
        prepared.prepare_seconds =
            std::chrono::duration<double>(Clock::now() - prepared.lifetime->started).count();
        prepared.generation = engine_->submit(std::move(prompt), std::move(request_options),
                                              prepared.lifetime->deadline, std::move(host_input));
        prepared.sampling   = prepared.generation.resolved_sampling();
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) { throw_invalid_input(exception); }
    return prepared;
}

int GenerationService::count_prompt_tokens(const GenerationRequest& request,
                                           std::function<bool()> is_cancelled) const {
    const std::size_t media_items = media_item_count(request);
    const bool request_has_media  = media_items != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    if (media_items > kMaximumMediaItems) {
        throw_request_error(ninfer::RequestError(RequestErrorKind::MediaBudgetExceeded,
                                                 "request exceeds the 16-item media limit"));
    }
    const Clock::time_point deadline =
        Clock::now() + std::chrono::milliseconds(options_.pending_timeout_ms);
    const ResolvedPromptSemantics semantics =
        resolve_prompt_semantics(request, options_, prompt_capabilities_);
    HostInputLease host_input;
    if (request_has_media) { host_input = acquire_media_input(deadline, is_cancelled); }
    try {
        std::size_t remaining_media_bytes = options_.max_request_bytes;
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, deadline, is_cancelled, remaining_media_bytes);
            });
        check_preparation_control(deadline, is_cancelled);
        const int prompt_tokens = static_cast<int>(engine_->count_tokens(std::move(input)));
        check_preparation_control(deadline, is_cancelled);
        return prompt_tokens;
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) { throw_invalid_input(exception); }
}

GenerationOutcome GenerationService::run(PreparedRequest& prepared, const StreamSink* sink,
                                         std::function<bool()> is_cancelled) {
    std::unique_ptr<ServiceOutputSink> output_sink;
    if (sink != nullptr) { output_sink = std::make_unique<ServiceOutputSink>(*sink, prepared); }
    ninfer::OutputSink* public_sink = output_sink.get();
    ninfer::CancellationView cancellation;
    if (is_cancelled || (sink != nullptr && sink->is_cancelled)) {
        cancellation = ninfer::CancellationView([external = std::move(is_cancelled), sink]() {
            return (external && external()) ||
                   (sink != nullptr && sink->is_cancelled && sink->is_cancelled());
        });
    }

    ninfer::GenerationResult result;
    try {
        result = prepared.generation.wait(public_sink, cancellation);
    } catch (const ninfer::RequestError& exception) { throw_request_error(exception); }
    GenerationOutcome outcome;
    outcome.text              = std::move(result.content);
    outcome.reasoning         = std::move(result.reasoning);
    outcome.prompt_tokens     = static_cast<int>(result.prompt.prompt_tokens);
    outcome.completion_tokens = static_cast<int>(result.generated_token_ids.size());
    outcome.reasoning_tokens  = static_cast<int>(result.reasoning_tokens);
    outcome.finish_reason     = result.finish_reason;
    outcome.id_slot           = result.slot;
    outcome.session_digest    = std::move(result.session_digest);

    outcome.metrics.prepare_seconds = prepared.prepare_seconds;
    outcome.metrics.ttft_seconds =
        prepared.prepare_seconds +
        std::max(0.0, result.timings.first_token_seconds - result.timings.prepare_seconds);
    outcome.metrics.vision_seconds  = result.timings.vision_seconds;
    outcome.metrics.queue_seconds   = result.timings.queue_seconds;
    outcome.metrics.restore_seconds = result.timings.restore_seconds;
    outcome.metrics.publish_seconds = result.timings.publish_seconds;
    outcome.metrics.prefill_seconds = result.timings.prefill_seconds;
    outcome.metrics.decode_seconds  = result.timings.decode_seconds;
    outcome.metrics.total_seconds =
        prepared.prepare_seconds +
        std::max(0.0, result.timings.total_seconds - result.timings.prepare_seconds);
    outcome.metrics.prefix_cache_hit_tokens     = result.reused_prompt_tokens;
    outcome.metrics.prefix_reuse_path           = result.prefix_reuse_path;
    outcome.metrics.continuation                = result.continuation;
    outcome.metrics.speculative_backend         = result.speculative.backend;
    outcome.metrics.speculative_draft_window    = result.speculative.draft_window;
    outcome.metrics.speculative_rounds          = result.speculative.rounds;
    outcome.metrics.speculative_draft_tokens    = result.speculative.drafted_tokens;
    outcome.metrics.speculative_accepted_tokens = result.speculative.accepted_tokens;
    outcome.metrics.speculative_fallback_steps  = result.speculative.fallback_steps;
    outcome.metrics.speculative_accepted_per_position =
        std::move(result.speculative.accepted_per_position);

    if (prepared.tool_capable) {
        // A streamed request was parsed as it was published; the engine's terminal content is
        // the concatenation of those deltas, so a buffered request parses it in one feed.
        ToolCallStreamResult parsed;
        if (output_sink) {
            parsed = output_sink->finish_tool_calls();
        } else {
            QwenToolCallStream parser(std::move(prepared.tool_parameters),
                                      prepared.tool_name_max_length);
            parser.feed(outcome.text);
            parsed = parser.finish();
        }
        outcome.text                      = std::move(parsed.content);
        outcome.tool_calls                = std::move(parsed.tool_calls);
        outcome.tool_call_discarded_bytes = parsed.discarded_bytes;
    }
    return outcome;
}

PreparedDecisionRequest GenerationService::prepare_decision(ninfer::DecisionInput input,
                                                           std::string adapter) const {
    PreparedDecisionRequest prepared;
    prepared.lifetime                  = acquire_request_lifetime();
    ninfer::PreparedDecision decision = engine_->prepare_decision(std::move(input));
    prepared.summary                   = decision.summary();
    prepared.prepare_seconds =
        std::chrono::duration<double>(Clock::now() - prepared.lifetime->started).count();
    prepared.decision = engine_->submit_decision(
        std::move(decision),
        ninfer::DecisionOptions{.adapter            = std::move(adapter),
                                .allow_prefix_reuse = options_.allow_prefix_reuse},
        prepared.lifetime->deadline);
    return prepared;
}

ninfer::DecisionResult GenerationService::decide(PreparedDecisionRequest& prepared,
                                                 std::function<bool()> is_cancelled) {
    ninfer::CancellationView cancellation;
    if (is_cancelled) { cancellation = ninfer::CancellationView(std::move(is_cancelled)); }
    return prepared.decision.wait(cancellation);
}

void GenerationService::warmup() {
    try {
        GenerationRequest request;
        ChatTurn turn;
        turn.role = "user";
        ContentPart content;
        content.kind     = ContentKind::Text;
        content.text     = "hi";
        content.type_raw = "text";
        turn.content.push_back(std::move(content));
        request.messages.push_back(std::move(turn));
        request.max_tokens       = 4;
        request.max_tokens_set   = true;
        PreparedRequest prepared = prepare(request);
        run(prepared, nullptr);
    } catch (const std::exception& exception) {
        write_console_log(ConsoleLogLevel::Warning,
                          std::string("warmup failed (continuing): ") + exception.what());
    }

    // A decision's branch passes reach prefill routes the short chat warmup never does: one pass
    // per GEMM width class (up to 128, 129 to 256, and from 257 columns) plus the segmented
    // mixers and the pointer head. One cold decision per class loads those modules here instead
    // of inside the first request. The kernels do not depend on the adapter, so one suffices.
    const ninfer::LoadSummary summary = load_summary();
    const auto& adapters              = summary.lora_adapters;
    const auto decision = std::find_if(adapters.begin(), adapters.end(), [](const auto& adapter) {
        return adapter.kind == ninfer::LoraAdapterKind::Decision;
    });
    if (decision == adapters.end()) { return; }
    try {
        for (const std::size_t words : {56U, 184U, 504U}) {
            ninfer::DecisionInput input;
            input.state = "warmup";
            ninfer::DecisionQuestion question;
            question.instructions.reserve(2 * words);
            for (std::size_t word = 0; word < words; ++word) { question.instructions += " a"; }
            question.options = {"yes", "no"};
            input.questions.push_back(std::move(question));
            (void)engine_->decide(engine_->prepare_decision(std::move(input)),
                                  ninfer::DecisionOptions{.adapter            = decision->name,
                                                          .allow_prefix_reuse = false});
        }
    } catch (const std::exception& exception) {
        write_console_log(ConsoleLogLevel::Warning,
                          std::string("decision warmup failed (continuing): ") + exception.what());
    }
}

} // namespace ninfer::serve
