#include "serve/http_server.h"

#include "serve/anthropic_schema.h"
#include "serve/console_log.h"
#include "serve/openai_schema.h"
#include "serve/request_log.h"
#include "serve/responses_schema.h"
#include "serve/slot_files.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

// Records retained for replay to a newly connected /events reader. At the default 5s reporting
// interval this is roughly the last forty minutes of throughput samples interleaved with the
// request records from the same window - enough for a dashboard opened mid-run to draw a
// populated chart immediately rather than starting blank.
constexpr std::size_t kEventReplayCapacity = 512;

// SSE keepalive period. Also bounds how long a writer blocks before re-testing that its socket is
// still writable, which is how a closed browser tab is noticed on an idle server.
constexpr std::chrono::milliseconds kEventKeepalive{15000};

std::string format_seconds(double seconds) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.2f", seconds);
    return text;
}

nlohmann::json arena_json(const ninfer::ArenaMemorySummary& arena) {
    return {{"capacity_bytes", arena.capacity_bytes},
            {"used_bytes", arena.used_bytes},
            {"peak_used_bytes", arena.peak_used_bytes}};
}

nlohmann::json gpu_json(const GpuTelemetry& gpu, const ServerEnergyTotals& energy) {
    if (!gpu.available) { return {{"available", false}, {"error", gpu.error}}; }
    // Null where the board implements no cumulative energy counter. Many GeForce parts do not, and
    // a zero there would be indistinguishable from a board that drew nothing.
    nlohmann::json energy_joules_total = nlohmann::json(nullptr);
    nlohmann::json server_energy       = nlohmann::json(nullptr);
    nlohmann::json idle_watts          = nlohmann::json(nullptr);
    if (gpu.energy_available) { energy_joules_total = gpu.energy_joules_total; }
    if (energy.available) {
        server_energy = energy.board_joules_total;
        idle_watts    = energy.idle_watts;
    }
    return {{"available", true},
            {"name", gpu.name},
            {"uuid", gpu.uuid},
            {"driver_version", gpu.driver_version},
            {"temperature_c", gpu.temperature_c},
            {"fan_percent", gpu.fan_percent},
            {"power_watts", gpu.power_watts},
            {"power_limit_watts", gpu.power_limit_watts},
            {"energy_available", gpu.energy_available},
            {"energy_joules_total", energy_joules_total},
            {"server_energy_joules", server_energy},
            {"idle_watts", idle_watts},
            {"utilization_gpu_percent", gpu.utilization_gpu_percent},
            {"utilization_memory_percent", gpu.utilization_memory_percent},
            {"sm_clock_mhz", gpu.sm_clock_mhz},
            {"sm_clock_max_mhz", gpu.sm_clock_max_mhz},
            {"memory_clock_mhz", gpu.memory_clock_mhz},
            {"memory_used_bytes", gpu.memory_used_bytes},
            {"memory_total_bytes", gpu.memory_total_bytes},
            {"pcie_rx_bytes_per_second", gpu.pcie_rx_bytes_per_second},
            {"pcie_tx_bytes_per_second", gpu.pcie_tx_bytes_per_second},
            {"throttle_reasons", gpu.throttle_reasons}};
}

// One SSE frame carrying a complete schema-19 record. The record's own `event` field names the
// frame so a browser can attach one listener per record type.
std::string event_frame(std::string_view event, std::string_view payload) {
    std::string frame;
    frame.reserve(payload.size() + event.size() + 16);
    frame += "event: ";
    frame += event;
    frame += "\ndata: ";
    frame += payload;
    frame += "\n\n";
    return frame;
}

// Paths owned by the API. Everything else, under --web-dir, belongs to the dashboard's own
// client-side router and resolves to the application shell.
bool is_api_path(const std::string& path) {
    static constexpr std::string_view kPrefixes[] = {"/v1/",     "/slots",     "/metrics",
                                                     "/health",  "/telemetry", "/events"};
    for (const std::string_view prefix : kPrefixes) {
        if (path.rfind(prefix, 0) == 0) { return true; }
    }
    return false;
}

std::string_view record_event_name(const std::string& record) {
    // The formatters emit sorted keys, so `"event":"<name>"` is a stable substring. Parsing the
    // whole record again only to recover its type would double the cost of every frame.
    const std::size_t key = record.find("\"event\":\"");
    if (key == std::string::npos) { return "message"; }
    const std::size_t begin = key + 9;
    const std::size_t end   = record.find('"', begin);
    if (end == std::string::npos) { return "message"; }
    return std::string_view(record).substr(begin, end - begin);
}

struct StreamingRequest {
    explicit StreamingRequest(PreparedRequest request) : prepared(std::move(request)) {}

    PreparedRequest prepared;
    std::atomic<bool> cancelled{false};
    bool started = false;
};

class ClientDisconnected final : public std::exception {
public:
    [[nodiscard]] const char* what() const noexcept override { return "client disconnected"; }
};

void write_stream_item(httplib::DataSink& sink, StreamingRequest& request,
                       const std::string& item) {
    if (request.cancelled.load(std::memory_order_acquire) ||
        (sink.is_writable && !sink.is_writable()) || !sink.write(item.data(), item.size())) {
        request.cancelled.store(true, std::memory_order_release);
        throw ClientDisconnected();
    }
}

void set_owned_content(httplib::Response& response, std::string body,
                       std::shared_ptr<RequestLifetime> lifetime) {
    response.set_content(std::move(body), "application/json");
    response.hold_resource(std::move(lifetime));
}

ApiError public_internal_error() {
    ApiError error;
    error.status  = 500;
    error.type    = "internal_error";
    error.message = "Internal server error.";
    return error;
}

ApiError public_error(ApiError error) {
    return error.status == 500 ? public_internal_error() : std::move(error);
}

// CompletionUsage carrying the engine's measured phase timings, so the OpenAI
// schema layer can emit the llama.cpp-compatible `timings` block (llama-swap
// derives Prefill/Decode rates and draft stats from it).
CompletionUsage usage_with_timings(const GenerationOutcome& outcome) {
    CompletionUsage usage;
    usage.prompt_tokens     = outcome.prompt_tokens;
    usage.completion_tokens = outcome.completion_tokens;
    usage.has_timings       = true;
    usage.prefill_seconds   = outcome.metrics.prefill_seconds;
    usage.decode_seconds    = outcome.metrics.decode_seconds;
    usage.ttft_seconds      = outcome.metrics.ttft_seconds;
    usage.cache_hit_tokens  = outcome.metrics.prefix_cache_hit_tokens;
    usage.cache_write_tokens = outcome.metrics.continuation.cache_write_tokens;
    usage.draft_tokens      = outcome.metrics.speculative_draft_tokens;
    usage.accepted_tokens   = outcome.metrics.speculative_accepted_tokens;
    usage.id_slot           = outcome.id_slot;
    usage.session_digest    = outcome.session_digest;
    return usage;
}

void write_error(httplib::Response& res, const ApiError& error) {
    const ApiError rendered = public_error(error);
    res.status              = rendered.status;
    if (rendered.status == 429 || rendered.status == 503) { res.set_header("Retry-After", "1"); }
    res.set_content(make_error_body(rendered), "application/json");
}

// Anthropic-shaped error body ({"type":"error","error":{...}}), used by the
// /v1/messages endpoints so Claude clients see the error format they expect.
void write_messages_error(httplib::Response& res, const ApiError& error) {
    const ApiError rendered = public_error(error);
    res.status              = rendered.status;
    if (rendered.status == 429 || rendered.status == 503) { res.set_header("Retry-After", "1"); }
    res.set_content(make_messages_error_body(rendered), "application/json");
}

void write_exception(httplib::Response& res) { write_error(res, public_internal_error()); }

void ensure_request_id(httplib::Response& response) {
    if (!response.has_header("x-request-id")) {
        static const std::string process_prefix = new_response_item_id("req");
        static std::atomic<std::uint64_t> sequence{0};
        response.set_header("x-request-id", process_prefix + "_" + std::to_string(++sequence));
    }
}

std::string sse_error_event(const ApiError& error) {
    return "data: " + make_error_body(error) + "\n\n";
}

// Reconcile the executor's per-unit power integration against the board's own energy counter.
//
// The counter is exact; the split is not, because the board refreshes power at roughly 50 Hz and a
// decode round is shorter than that. Time no execution unit claimed is priced at the measured idle
// baseline, and whatever remains is reported as a residual so a consumer can decide whether the
// split is trustworthy instead of being handed a number with its error hidden inside it.
IntervalEnergy reconcile_interval_energy(const ninfer::RuntimeStats& previous,
                                         const ninfer::RuntimeStats& current,
                                         double interval_seconds,
                                         const BoardEnergySample& board_energy) {
    IntervalEnergy energy;
    if (!board_energy.available || interval_seconds <= 0.0) { return energy; }
    const auto monotonic_double = [](double before, double after) {
        return after >= before ? after - before : after;
    };
    energy.available        = true;
    energy.board_joules     = board_energy.joules;
    energy.prefill_joules   = monotonic_double(previous.prefill_energy_joules,
                                               current.prefill_energy_joules);
    energy.decode_joules    = monotonic_double(previous.decode_energy_joules,
                                               current.decode_energy_joules);
    energy.accounted_seconds = monotonic_double(previous.energy_accounted_seconds,
                                                current.energy_accounted_seconds);
    energy.idle_watts        = board_energy.idle_watts;

    const double unaccounted_seconds =
        std::max(0.0, interval_seconds - energy.accounted_seconds);
    energy.idle_joules = energy.idle_watts * unaccounted_seconds;
    energy.residual_joules =
        energy.board_joules - energy.prefill_joules - energy.decode_joules - energy.idle_joules;
    energy.residual_fraction =
        energy.board_joules > 0.0 ? energy.residual_joules / energy.board_joules : 0.0;
    return energy;
}

ThroughputReport make_throughput_report_impl(const ninfer::RuntimeStats& previous,
                                              const ninfer::RuntimeStats& current,
                                              double interval_seconds,
                                              const BoardEnergySample& board_energy) {
    ninfer::RuntimeStats delta;
    const auto monotonic_delta = [](std::uint64_t before, std::uint64_t after) {
        return after >= before ? after - before : after;
    };
#define NINFER_DELTA(field) delta.field = monotonic_delta(previous.field, current.field)
    NINFER_DELTA(continuation_lookup_hits);
    NINFER_DELTA(continuation_lookup_misses);
    NINFER_DELTA(continuation_preflight_rejections);
    NINFER_DELTA(continuation_restore_failures);
    NINFER_DELTA(continuation_restore_deferrals);
    NINFER_DELTA(continuation_l1_restore_successes);
    NINFER_DELTA(continuation_l2_restore_successes);
    NINFER_DELTA(continuation_l3_restore_successes);
    NINFER_DELTA(continuation_l1_restored_tokens);
    NINFER_DELTA(continuation_l2_restored_tokens);
    NINFER_DELTA(continuation_l3_restored_tokens);
    NINFER_DELTA(continuation_l1_restored_bytes);
    NINFER_DELTA(continuation_l2_restored_bytes);
    NINFER_DELTA(continuation_l3_restored_bytes);
    NINFER_DELTA(continuation_l2_evictions);
    NINFER_DELTA(continuation_l2_evicted_bytes);
    NINFER_DELTA(continuation_l3_evictions);
    NINFER_DELTA(continuation_l3_evicted_bytes);
    NINFER_DELTA(continuation_session_restores);
    NINFER_DELTA(continuation_stable_prefix_restores);
    NINFER_DELTA(continuation_miss_disabled);
    NINFER_DELTA(continuation_miss_no_alias);
    NINFER_DELTA(continuation_miss_not_attempted);
    NINFER_DELTA(continuation_miss_entry_unavailable_or_corrupt);
    NINFER_DELTA(continuation_miss_not_deeper);
    NINFER_DELTA(continuation_miss_preflight_rejected);
    NINFER_DELTA(continuation_miss_rollback_conflict);
    NINFER_DELTA(continuation_miss_no_lane);
    NINFER_DELTA(continuation_miss_restore_failed);
    NINFER_DELTA(continuation_l2_lookup_microseconds);
    NINFER_DELTA(continuation_l2_lookup_operations);
    NINFER_DELTA(continuation_l3_lookup_microseconds);
    NINFER_DELTA(continuation_l3_lookup_operations);
    NINFER_DELTA(continuation_preflight_microseconds);
    NINFER_DELTA(continuation_preflight_operations);
    NINFER_DELTA(continuation_l2_restore_microseconds);
    NINFER_DELTA(continuation_l2_restore_operations);
    NINFER_DELTA(continuation_l3_restore_microseconds);
    NINFER_DELTA(continuation_l3_restore_operations);
    NINFER_DELTA(continuation_l2_admission_microseconds);
    NINFER_DELTA(continuation_l2_admission_operations);
    NINFER_DELTA(continuation_export_microseconds);
    NINFER_DELTA(continuation_export_operations);
    NINFER_DELTA(continuation_l3_persistence_microseconds);
    NINFER_DELTA(continuation_l3_persistence_operations);
    NINFER_DELTA(continuation_publication_successes);
    NINFER_DELTA(continuation_publication_failures);
    NINFER_DELTA(continuation_publication_superseded);
    NINFER_DELTA(continuation_publication_coalesced);
    NINFER_DELTA(continuation_publication_failed_capacity);
    NINFER_DELTA(continuation_publication_failed_evicted);
    NINFER_DELTA(continuation_publication_failed_alias_moved);
    NINFER_DELTA(continuation_publication_failed_lineage);
    NINFER_DELTA(continuation_publication_failed_error);
    NINFER_DELTA(continuation_persistence_queued);
    NINFER_DELTA(continuation_persistence_coalesced);
    NINFER_DELTA(continuation_persistence_successes);
    NINFER_DELTA(continuation_persistence_failures);
    NINFER_DELTA(l1_evictions);
    NINFER_DELTA(l1_demotions);
    NINFER_DELTA(kv_restore_reclaimed_lanes);
    NINFER_DELTA(kv_growth_attempts);
    NINFER_DELTA(kv_growth_forced_spills);
    NINFER_DELTA(kv_growth_curtailed);
    NINFER_DELTA(continuation_restore_failed_kv_reservation);
    NINFER_DELTA(continuation_restore_failed_verify_depth);
    NINFER_DELTA(continuation_restore_failed_inventory);
    NINFER_DELTA(continuation_restore_failed_metadata);
    NINFER_DELTA(continuation_restore_failed_decode);
    NINFER_DELTA(continuation_restore_failed_lane);
    NINFER_DELTA(admission_rejected_overloaded);
    NINFER_DELTA(admission_rejected_queue_timeout);
    NINFER_DELTA(admitted_requests);
#undef NINFER_DELTA
    delta.queue_seconds_total = current.queue_seconds_total >= previous.queue_seconds_total
                                    ? current.queue_seconds_total - previous.queue_seconds_total
                                    : current.queue_seconds_total;
    delta.continuation_restore_successes = delta.continuation_l1_restore_successes +
                                             delta.continuation_l2_restore_successes +
                                             delta.continuation_l3_restore_successes;
    delta.continuation_restored_tokens = delta.continuation_l1_restored_tokens +
                                          delta.continuation_l2_restored_tokens +
                                          delta.continuation_l3_restored_tokens;
    delta.continuation_restored_bytes = delta.continuation_l1_restored_bytes +
                                         delta.continuation_l2_restored_bytes +
                                         delta.continuation_l3_restored_bytes;
    return ThroughputReport{
        .interval_seconds = interval_seconds,
        .computed_prefill_tokens =
            monotonic_delta(previous.computed_prefill_tokens, current.computed_prefill_tokens),
        .committed_decode_tokens =
            monotonic_delta(previous.committed_decode_tokens, current.committed_decode_tokens),
        .decode_rounds = monotonic_delta(previous.decode_rounds, current.decode_rounds),
        .decode_row_rounds = monotonic_delta(previous.decode_row_rounds,
                                             current.decode_row_rounds),
        .energy = reconcile_interval_energy(previous, current, interval_seconds, board_energy),
        .scheduler          = current,
        .continuation_delta = delta,
    };
}

bool report_has_activity_impl(const ThroughputReport& report) noexcept {
    const auto& delta = report.continuation_delta;
    return report.computed_prefill_tokens != 0 || report.committed_decode_tokens != 0 ||
            report.decode_rounds != 0 || report.scheduler.running_requests != 0 ||
            report.scheduler.waiting_requests != 0 ||
            delta.continuation_l1_restore_successes != 0 ||
            delta.continuation_l2_restore_successes != 0 ||
            delta.continuation_l3_restore_successes != 0 ||
            delta.continuation_lookup_hits != 0 || delta.continuation_lookup_misses != 0 ||
            delta.continuation_preflight_rejections != 0 ||
            delta.continuation_restore_failures != 0 ||
            delta.continuation_miss_disabled != 0 || delta.continuation_miss_no_alias != 0 ||
            delta.continuation_miss_not_attempted != 0 || delta.admitted_requests != 0 ||
            delta.admission_rejected_overloaded != 0 ||
            delta.admission_rejected_queue_timeout != 0 ||
            delta.continuation_miss_entry_unavailable_or_corrupt != 0 ||
            delta.continuation_miss_not_deeper != 0 ||
            delta.continuation_miss_preflight_rejected != 0 ||
            delta.continuation_miss_rollback_conflict != 0 ||
            delta.continuation_miss_no_lane != 0 ||
            delta.continuation_miss_restore_failed != 0 ||
            delta.continuation_l2_lookup_operations != 0 ||
            delta.continuation_l3_lookup_operations != 0 ||
            delta.continuation_preflight_operations != 0 ||
            delta.continuation_l2_restore_operations != 0 ||
            delta.continuation_l3_restore_operations != 0 ||
            delta.continuation_l2_admission_operations != 0 ||
            delta.continuation_export_operations != 0 ||
            delta.continuation_l3_persistence_operations != 0 ||
            delta.continuation_publication_successes != 0 ||
            delta.continuation_publication_failures != 0 ||
            delta.continuation_publication_superseded != 0 ||
            delta.continuation_persistence_queued != 0 ||
            delta.continuation_persistence_coalesced != 0 ||
            delta.continuation_persistence_successes != 0 ||
            delta.continuation_persistence_failures != 0 || delta.l1_evictions != 0 ||
            delta.l1_demotions != 0 || delta.kv_growth_forced_spills != 0 ||
            delta.kv_growth_curtailed != 0;
}

std::string_view unstreamed_content(const GenerationOutcome& outcome) {
    if (outcome.streamed_content_bytes > outcome.text.size()) {
        throw std::logic_error("streamed content exceeds terminal content");
    }
    return std::string_view(outcome.text).substr(outcome.streamed_content_bytes);
}

} // namespace

ThroughputReport make_throughput_report(const ninfer::RuntimeStats& previous,
                                        const ninfer::RuntimeStats& current,
                                        double interval_seconds,
                                        const BoardEnergySample& board_energy) {
    return make_throughput_report_impl(previous, current, interval_seconds, board_energy);
}

bool throughput_report_has_activity(const ThroughputReport& report) noexcept {
    return report_has_activity_impl(report);
}

HttpServer::HttpServer(ServeOptions options)
    : options_(std::move(options)),
      response_store_(options_.response_store_max_records, options_.response_store_max_bytes),
      events_(options_.request_log_jsonl, options_.artifact_path, kEventReplayCapacity),
      gpu_(options_.device) {
    const std::size_t queued_requests =
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests;
    const std::size_t worker_count = queued_requests + 1;
    server_.new_task_queue         = [queued_requests, worker_count] {
        return new httplib::ThreadPool(worker_count, queued_requests);
    };
    server_.set_payload_max_length(options_.max_request_bytes);
    register_routes();
}

void HttpServer::log_line(const std::string& line) {
    write_console_log(ConsoleLogLevel::Info, line);
}

void HttpServer::log_request_start(const RequestLogContext& context) {
    log_line(format_request_start(context));
    events_.emit_request_start(context);
    metrics_.begin_request(context.id, context.prompt_tokens);
}

void HttpServer::log_request_done(const RequestLogContext& context,
                                  const GenerationOutcome& outcome) {
    log_line(format_request_done(context, outcome));
    events_.emit_request_done(context, outcome);
    metrics_.end_request(context.id);
    metrics_.record(outcome);
}

void HttpServer::log_request_error(const RequestLogContext& context, const std::string& message) {
    log_line(format_request_error(context, message));
    events_.emit_request_error(context, message);
    metrics_.end_request(context.id);
}

void HttpServer::log_throughput(const ThroughputReport& report) {
    log_line(format_throughput(report));
    events_.emit_throughput(report);
}

void HttpServer::run_stats_reporter() {
    using Clock                     = std::chrono::steady_clock;
    ninfer::RuntimeStats previous   = service_->runtime_stats();
    Clock::time_point previous_time = Clock::now();
    const auto interval             = std::chrono::milliseconds(options_.log_stats_interval_ms);

    // Energy is differenced across every tick, including the idle ticks a throughput report skips,
    // because an idle tick is the only place the idle baseline can be measured rather than assumed.
    // The board counter costs milliseconds to read and is quantized to about 100 ms, so this thread
    // is the right place for it: an execution thread could not afford the call, and a window
    // shorter than a second or two would not resolve the counter's own step size.
    std::optional<double> previous_energy   = read_board_energy_joules();
    ninfer::RuntimeStats energy_anchor      = previous;
    Clock::time_point energy_anchor_time    = previous_time;
    double carried_joules                   = 0.0;

    const auto tick = [&](const ninfer::RuntimeStats& current, Clock::time_point now) {
        BoardEnergySample sample;
        const std::optional<double> energy_now = read_board_energy_joules();
        if (previous_energy && energy_now && *energy_now >= *previous_energy) {
            const double consumed = *energy_now - *previous_energy;
            carried_joules += consumed;
            observe_idle_power(energy_anchor, current,
                               std::chrono::duration<double>(now - energy_anchor_time).count(),
                               consumed);
            sample.available = true;
            board_energy_joules_total_.store(
                board_energy_joules_total_.load(std::memory_order_relaxed) + consumed,
                std::memory_order_relaxed);
            published_idle_watts_.store(idle_watts_, std::memory_order_relaxed);
            energy_available_.store(true, std::memory_order_relaxed);
        }
        // A driver reload resets the counter. Dropping that one difference is correct; carrying a
        // negative or restarted value into an interval would fabricate energy that was not used.
        if (energy_now) { previous_energy = energy_now; }
        energy_anchor      = current;
        energy_anchor_time = now;
        sample.joules      = carried_joules;
        sample.idle_watts  = idle_watts_;
        return sample;
    };

    for (;;) {
        {
            std::unique_lock lock(stats_mutex_);
            if (stats_cv_.wait_for(lock, interval, [this] { return stats_stopping_; })) { break; }
        }

        const ninfer::RuntimeStats current  = service_->runtime_stats();
        const Clock::time_point now         = Clock::now();
        const BoardEnergySample board       = tick(current, now);
        const ThroughputReport report       = make_throughput_report(
            previous, current, std::chrono::duration<double>(now - previous_time).count(), board);
        if (throughput_report_has_activity(report)) {
            log_throughput(report);
            previous       = current;
            previous_time  = now;
            carried_joules = 0.0;
        }
    }

    const ninfer::RuntimeStats current = service_->runtime_stats();
    const Clock::time_point now        = Clock::now();
    const BoardEnergySample board      = tick(current, now);
    const ThroughputReport tail        = make_throughput_report(
        previous, current, std::chrono::duration<double>(now - previous_time).count(), board);
    if (throughput_report_has_activity(tail)) { log_throughput(tail); }
}

std::optional<double> HttpServer::read_board_energy_joules() const {
    return gpu_.power().energy_joules();
}

void HttpServer::observe_idle_power(const ninfer::RuntimeStats& before,
                                    const ninfer::RuntimeStats& after, double seconds,
                                    double joules) {
    // Only a tick during which no execution unit ran at all, with nothing running or queued,
    // measures the idle draw. Anything else mixes execution into the baseline and would inflate it,
    // which would then be subtracted from the phases and understate their cost.
    if (seconds <= 0.0 || joules <= 0.0) { return; }
    if (after.energy_accounted_seconds != before.energy_accounted_seconds) { return; }
    if (after.running_requests != 0 || after.waiting_requests != 0) { return; }
    const double observed = joules / seconds;
    idle_watts_ = idle_watts_ == 0.0 ? observed : 0.75 * idle_watts_ + 0.25 * observed;
}

ServerEnergyTotals HttpServer::energy_totals() const {
    ServerEnergyTotals totals;
    totals.available = energy_available_.load(std::memory_order_relaxed);
    totals.board_joules_total = board_energy_joules_total_.load(std::memory_order_relaxed);
    totals.idle_watts         = published_idle_watts_.load(std::memory_order_relaxed);
    return totals;
}

void HttpServer::stop_stats_reporter() {
    if (!stats_thread_.joinable()) { return; }
    {
        std::lock_guard lock(stats_mutex_);
        stats_stopping_ = true;
    }
    stats_cv_.notify_one();
    stats_thread_.join();
}

void HttpServer::register_routes() {
    server_.set_error_handler([this](const httplib::Request& req, httplib::Response& res) {
        ensure_request_id(res);
        // Single-page dashboard fallback. Registered API routes are matched before mount points,
        // so a 404 on a GET that is not an API path is a client-side route: serve the shell and
        // let the app resolve it. Reached only when --web-dir is configured.
        if (res.status == 404 && req.method == "GET" && !options_.web_dir.empty() &&
            !is_api_path(req.path)) {
            std::ifstream shell(std::filesystem::path(options_.web_dir) / "index.html",
                                std::ios::binary);
            if (shell) {
                std::ostringstream body;
                body << shell.rdbuf();
                res.status = 200;
                res.set_content(body.str(), "text/html");
                return;
            }
        }
        // httplib invokes this for EVERY status >= 400, including 413s that
        // route handlers already answered with a specific error body (e.g.
        // media_budget_exceeded). Only synthesize the generic payload-limit
        // error for the bare 413 httplib itself produces on oversized bodies.
        if (res.status != 413 || !res.body.empty()) { return; }
        ApiError error;
        error.status  = 413;
        error.type    = "invalid_request_error";
        error.code    = "request_too_large";
        error.message = "request body exceeds the configured payload limit";
        if (req.path.rfind("/v1/messages", 0) == 0) {
            write_messages_error(res, error);
        } else {
            write_error(res, error);
        }
    });
    if (options_.enable_cors) {
        server_.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Allow-Headers",
              "Authorization, Content-Type, OpenAI-Organization, OpenAI-Project, X-API-Key, "
              "Anthropic-Version, Anthropic-Beta, X-Stainless-Arch, X-Stainless-Lang, "
              "X-Stainless-OS, X-Stainless-Package-Version, X-Stainless-Retry-Count, "
              "X-Stainless-Runtime, X-Stainless-Runtime-Version"},
             {"Access-Control-Allow-Methods", "GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS"},
             {"Access-Control-Expose-Headers", "x-request-id, Retry-After"},
             {"Access-Control-Max-Age", "86400"}});
        // CORS preflight: browsers send OPTIONS with no credentials before the real
        // request; answer it without auth so the actual GET/POST can carry the key.
        server_.Options(R"(.*)",
                        [](const httplib::Request&, httplib::Response& res) { res.status = 204; });
    }

    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        ensure_request_id(res);
        if (options_.api_key.empty() || req.path == "/health" || req.method == "OPTIONS") {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        // Accept both the OpenAI-style bearer token and the Anthropic-style
        // x-api-key header so OpenAI clients and Claude Code (ANTHROPIC_API_KEY
        // -> x-api-key, ANTHROPIC_AUTH_TOKEN -> Authorization: Bearer) both work.
        const bool bearer_ok =
            req.get_header_value("Authorization") == ("Bearer " + options_.api_key);
        const bool x_api_key_ok = req.get_header_value("x-api-key") == options_.api_key;
        if (!bearer_ok && !x_api_key_ok) {
            ApiError error;
            error.status  = 401;
            error.type    = "invalid_request_error";
            error.code    = "invalid_api_key";
            error.message = "missing or invalid API key";
            // Render the 401 in the shape the target endpoint speaks.
            if (req.path.rfind("/v1/messages", 0) == 0) {
                write_messages_error(res, error);
            } else {
                write_error(res, error);
            }
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    server_.set_post_routing_handler(
        [](const httplib::Request&, httplib::Response& res) { ensure_request_id(res); });

    server_.set_exception_handler(
        [](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
            ensure_request_id(res);
            try {
                std::rethrow_exception(ep);
            } catch (const ApiException& e) {
                if (e.error().status == 500) {
                    write_console_log(ConsoleLogLevel::Error,
                                      "unhandled HTTP API exception: " + e.error().message);
                }
                write_error(res, e.error());
            } catch (const std::exception& e) {
                write_console_log(ConsoleLogLevel::Error,
                                  "unhandled HTTP exception: " + std::string(e.what()));
                write_exception(res);
            } catch (...) {
                write_console_log(ConsoleLogLevel::Error, "unhandled non-standard HTTP exception");
                write_exception(res);
            }
        });

    server_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(nlohmann::json{{"status", "ok"}}.dump(), "application/json");
    });
    // One complete live snapshot for the dashboard: board telemetry, the scheduler's own
    // occupancy view, the VRAM budget, and continuation-cache occupancy against its configured
    // capacity. Every field here is either absent from /metrics or only derivable there by
    // differencing counters; the capacity denominators exist nowhere else on the wire.
    server_.Get("/telemetry", [this](const httplib::Request& req, httplib::Response& res) {
        handle_telemetry(req, res);
    });
    // The schema-19 record stream, identical to what --request-log-jsonl appends, as named SSE
    // events. A new reader is replayed the retained server_start record and the recent ring
    // before live delivery begins.
    server_.Get("/events", [this](const httplib::Request& req, httplib::Response& res) {
        handle_events(req, res);
    });
    server_.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(metrics_.render(options_.max_concurrency,
                                        service_ != nullptr ? service_->runtime_stats()
                                                            : ninfer::RuntimeStats{},
                                        energy_totals()),
                        "text/plain; version=0.0.4");
    });
    // llama.cpp-shaped slot detail, read from the Engine's real lane table: a busy lane
    // reports its request's prompt and reused-prefix sizes; an idle retained lane reports
    // the resident session's depth (as both tokens and cache, matching llama.cpp's retained
    // slot) plus its identifying `session_digest`. Before the service attaches (model still
    // loading) every slot reads idle.
    server_.Get("/slots", [this](const httplib::Request&, httplib::Response& res) {
        const bool speculative =
            options_.speculative.backend != ninfer::SpeculativeBackend::None;
        std::vector<ninfer::SlotState> states;
        if (service_ != nullptr) { states = service_->slot_states(); }
        nlohmann::json slots = nlohmann::json::array();
        for (std::uint32_t i = 0; i < options_.max_concurrency; ++i) {
            const ninfer::SlotState state =
                i < states.size() ? states[i] : ninfer::SlotState{};
            nlohmann::json checkpoints = nlohmann::json::array();
            for (const ninfer::SlotCheckpoint& checkpoint : state.checkpoints) {
                checkpoints.push_back({{"frontier", checkpoint.frontier},
                                       {"session_digest", checkpoint.session_digest}});
            }
            slots.push_back({{"id", i},
                             {"is_processing", state.processing},
                             {"retained", state.retained},
                             {"session_digest", state.session_digest},
                             {"checkpoints", std::move(checkpoints)},
                             {"n_ctx", options_.max_context},
                             {"n_prompt_tokens", state.prompt_tokens},
                             {"n_prompt_tokens_cache", state.cached_tokens},
                             {"speculative", speculative}});
        }
        res.set_content(slots.dump(), "application/json");
    });
    // llama.cpp-shaped session persistence: POST /slots/{id}?action=save|restore|erase with
    // {"filename": NAME}. Enabled only by --slot-save-path.
    server_.Post(R"(/slots/(\d+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_slot_action(req, res);
    });
    server_.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res);
    });
    server_.Get(R"(/v1/models/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_model(req, res);
    });
    server_.Post("/v1/chat/completions",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_chat_completions(req, res);
                 });
    server_.Post("/v1/responses", [this](const httplib::Request& req, httplib::Response& res) {
        handle_responses(req, res);
    });
    server_.Post("/v1/responses/input_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_input_tokens(req, res);
                 });
    server_.Post("/v1/responses/compact",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_compact(req, res);
                 });
    server_.Post(R"(/v1/responses/([^/]+)/cancel)",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_cancel(req, res);
                 });
    server_.Get(R"(/v1/responses/([^/]+)/input_items)",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_input_items(req, res);
                });
    server_.Get(R"(/v1/responses/([^/]+))",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_get(req, res);
                });
    server_.Delete(R"(/v1/responses/([^/]+))",
                   [this](const httplib::Request& req, httplib::Response& res) {
                       handle_response_delete(req, res);
                   });
    server_.Post("/v1/messages/count_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_count_tokens(req, res);
                 });
    server_.Post("/v1/messages", [this](const httplib::Request& req, httplib::Response& res) {
        handle_messages(req, res);
    });

    // Mount points are consulted only after every route above misses, so serving the dashboard
    // cannot shadow an API path. Hashed asset filenames are immutable; the shell is not.
    if (!options_.web_dir.empty()) {
        server_.set_file_extension_and_mimetype_mapping("webmanifest", "application/manifest+json");
        server_.set_mount_point("/", options_.web_dir);
        server_.set_file_request_handler([](const httplib::Request& req, httplib::Response& res) {
            res.set_header("Cache-Control", req.path.rfind("/assets/", 0) == 0
                                                ? "public, max-age=31536000, immutable"
                                                : "no-cache");
        });
    }
}

std::optional<std::string> HttpServer::resolve_model(const std::string& model) const {
    if (model == public_model_id_) { return std::string(); }
    for (std::size_t index = 0; index < adapter_model_ids_.size(); ++index) {
        if (model == adapter_model_ids_[index]) { return adapter_names_[index]; }
    }
    return std::nullopt;
}

std::string HttpServer::require_model(const std::string& model) const {
    if (const std::optional<std::string> selected = resolve_model(model)) { return *selected; }
    ApiError error;
    error.status  = 404;
    error.type    = "invalid_request_error";
    error.param   = "model";
    error.code    = "model_not_found";
    error.message = "model '" + model + "' not found";
    throw ApiException(std::move(error));
}

void HttpServer::handle_telemetry(const httplib::Request&, httplib::Response& res) const {
    const ninfer::RuntimeStats stats =
        service_ != nullptr ? service_->runtime_stats() : ninfer::RuntimeStats{};
    const ninfer::MemorySummary memory =
        service_ != nullptr ? service_->memory_summary() : ninfer::MemorySummary{};

    nlohmann::json scheduler = {
        {"running", stats.running_requests},
        {"prefilling", stats.prefilling_requests},
        {"decode_ready", stats.decode_ready_requests},
        {"waiting", stats.waiting_requests},
        {"max_concurrency", options_.max_concurrency},
        {"max_pending_requests", options_.max_pending_requests},
        {"admitted_requests", stats.admitted_requests},
        {"queue_seconds_total", stats.queue_seconds_total},
        {"rejected_overloaded", stats.admission_rejected_overloaded},
        {"rejected_queue_timeout", stats.admission_rejected_queue_timeout},
        {"decode_rounds", stats.decode_rounds},
        {"decode_row_rounds", stats.decode_row_rounds},
        {"decode_rounds_abandoned", stats.decode_rounds_abandoned},
        {"worker_seconds",
         {{"decode", stats.worker_decode_seconds},
          {"prefill", stats.worker_prefill_seconds},
          {"admission", stats.worker_admission_seconds},
          {"publish", stats.worker_publish_seconds},
          {"upkeep", stats.worker_upkeep_seconds}}},
        // Admission is the one unit whose cost is not self-explanatory: it can be planning,
        // continuation import, or commit. Most scheduler iterations that enter admission admit
        // nothing, so `calls` is what turns these seconds into a per-call cost.
        {"worker_admission",
         {{"calls", stats.worker_admission_calls},
          {"plan_seconds", stats.worker_admission_plan_seconds},
          {"restore_seconds", stats.worker_admission_restore_seconds},
          {"commit_seconds", stats.worker_admission_commit_seconds}}},
        {"worker_decode_rounds", stats.worker_decode_rounds},
        {"worker_prefill_steps", stats.worker_prefill_steps}};

    // Occupancy paired with the capacity it is measured against. A tier's byte count alone cannot
    // say whether the cache is healthy or saturated, and the configured budget reaches the wire
    // nowhere else.
    const std::size_t mib = 1024ULL * 1024ULL;
    nlohmann::json cache  = {
        {"l1",
          {{"entries", stats.l1_resident_entries},
           {"bytes", stats.l1_resident_bytes},
           {"capacity_bytes", options_.continuation_cache.l1_capacity_mib * mib},
           {"evictions", stats.l1_evictions},
           {"demotions", stats.l1_demotions}}},
        {"l2",
          {{"entries", stats.continuation_l2_entries},
           {"bytes", stats.continuation_l2_bytes},
           {"capacity_bytes", options_.continuation_cache.l2_capacity_mib * mib},
           {"evictions", stats.continuation_l2_evictions},
           {"evicted_bytes", stats.continuation_l2_evicted_bytes}}},
        {"l3",
          {{"entries", stats.continuation_l3_entries},
           {"bytes", stats.continuation_l3_bytes},
           {"capacity_bytes", options_.continuation_cache.l3_capacity_mib * mib},
           {"evictions", stats.continuation_l3_evictions},
           {"evicted_bytes", stats.continuation_l3_evicted_bytes}}},
        {"kv_restore_reclaimed_lanes", stats.kv_restore_reclaimed_lanes},
        {"kv_growth",
          {{"attempts", stats.kv_growth_attempts},
           {"forced_spills", stats.kv_growth_forced_spills},
           {"curtailed", stats.kv_growth_curtailed}}},
        {"restore_successes", stats.continuation_restore_successes},
        {"restore_failures", stats.continuation_restore_failures},
        {"restore_deferrals", stats.continuation_restore_deferrals},
        {"lookup_hits", stats.continuation_lookup_hits},
        {"lookup_misses", stats.continuation_lookup_misses},
        {"publication_successes", stats.continuation_publication_successes},
        {"publication_failures", stats.continuation_publication_failures},
        {"persistence_successes", stats.continuation_persistence_successes},
        {"persistence_failures", stats.continuation_persistence_failures}};

    nlohmann::json slots = nlohmann::json::array();
    if (service_ != nullptr) {
        for (const ninfer::SlotState& state : service_->slot_states()) {
            slots.push_back({{"processing", state.processing},
                             {"retained", state.retained},
                             {"prompt_tokens", state.prompt_tokens},
                             {"cached_tokens", state.cached_tokens},
                             {"session_digest", state.session_digest},
                             {"checkpoints", state.checkpoints.size()}});
        }
    }

    // Adapter inventory. Names come from the load summary rather than from the served model ids,
    // so an adapter that has taken no traffic is still reported. `count` is the whole discovered
    // pool and every entry is selectable; `slots` is how many are device-resident at once, which
    // is a residency detail clients never see.
    nlohmann::json adapters = nlohmann::json::object();
    {
        nlohmann::json names = nlohmann::json::array();
        for (const std::string& name : adapter_names_) { names.push_back(name); }
        nlohmann::json ids = nlohmann::json::array();
        for (const std::string& id : adapter_model_ids_) { ids.push_back(id); }
        const ninfer::LoadSummary load =
            service_ != nullptr ? service_->load_summary() : ninfer::LoadSummary{};
        adapters = {{"count", adapter_names_.size()},
                    {"names", std::move(names)},
                    {"model_ids", std::move(ids)},
                    {"rank", load.lora_rank},
                    {"slots", load.lora_slots},
                    {"device_bytes", load.lora_device_bytes},
                    {"file_bytes", load.lora_file_bytes}};
    }

    const nlohmann::json payload = {
        {"timestamp_unix_ms", unix_time_ms()},
        {"server_instance_id", events_.server_instance_id()},
        {"uptime_seconds",
         std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at_).count()},
        {"attached", service_ != nullptr},
        {"model_id", public_model_id_},
        {"gpu", gpu_json(gpu_.read(), energy_totals())},
        {"scheduler", std::move(scheduler)},
        {"cache", std::move(cache)},
        {"slots", std::move(slots)},
        {"adapters", std::move(adapters)},
        {"memory",
         {{"device", memory.device},
          {"max_context", memory.max_context},
          {"kv_capacity", memory.kv_capacity},
          {"kv_capacity_page_groups", memory.kv_capacity_page_groups},
          {"kv_capacity_max_page_groups", memory.kv_capacity_max_page_groups},
          {"weights", arena_json(memory.weights)},
          {"sequence", arena_json(memory.sequence)},
          {"workspace", arena_json(memory.workspace)},
          {"request_transient", arena_json(memory.request_transient)},
          {"kv_payload_bytes", memory.kv_payload_bytes},
          {"text_kv_bytes", memory.text_kv_bytes},
          {"mtp_kv_bytes", memory.mtp_kv_bytes},
          {"gdn_state_bytes", memory.gdn_state_bytes},
          {"dflash_kv_bytes", memory.dflash_kv_bytes},
          {"replay_records_bytes", memory.replay_records_bytes},
          {"cuda_graph_observed_bytes", memory.cuda_graph_observed_bytes},
          {"cuda_graph_allowance_bytes", memory.cuda_graph_allowance_bytes},
          {"available_after_startup_bytes", memory.available_after_startup_bytes},
          {"available_after_weights_bytes", memory.available_after_weights_bytes},
          {"lora_bank_bytes", memory.lora_bank_bytes}}},
        {"events", {{"jsonl_enabled", events_.jsonl_enabled()},
                    {"subscribers", events_.subscriber_count()}}}};
    res.set_content(payload.dump(), "application/json");
}

void HttpServer::handle_events(const httplib::Request&, httplib::Response& res) {
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");

    std::vector<std::string> backlog;
    std::shared_ptr<EventSubscriber> subscriber = events_.subscribe(backlog);

    res.set_chunked_content_provider(
        "text/event-stream",
        [this, subscriber, backlog = std::move(backlog)](std::size_t offset,
                                                         httplib::DataSink& sink) mutable {
            if (offset == 0) {
                // SSE retry hint plus the retained opening state, so a reconnecting dashboard is
                // immediately consistent without a separate bootstrap request.
                std::string opening = "retry: 2000\n\n";
                for (const std::string& record : backlog) {
                    opening += event_frame(record_event_name(record), record);
                }
                backlog.clear();
                if (!sink.write(opening.data(), opening.size())) { return false; }
            }
            std::string record;
            if (subscriber->next(record, kEventKeepalive)) {
                const std::string frame = event_frame(record_event_name(record), record);
                return sink.write(frame.data(), frame.size());
            }
            static constexpr std::string_view kKeepalive = ": keepalive\n\n";
            return sink.write(kKeepalive.data(), kKeepalive.size());
        },
        [this, subscriber](bool) { events_.unsubscribe(subscriber); });
}

void HttpServer::handle_models(const httplib::Request&, httplib::Response& res) const {
    res.set_content(make_models_list(public_model_id_, adapter_model_ids_, unix_time_now(),
                                     options_.max_context, options_.enable_vision),
                    "application/json");
}

void HttpServer::handle_model(const httplib::Request& req, httplib::Response& res) const {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    if (!resolve_model(id).has_value()) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_not_found";
        error.message = "model '" + id + "' not found";
        write_error(res, error);
        return;
    }
    res.set_content(
        make_model_object(id, unix_time_now(), options_.max_context, options_.enable_vision),
        "application/json");
}

void HttpServer::handle_slot_action(const httplib::Request& req, httplib::Response& res) {
    const auto fail = [&res](int status, std::string code, std::string message) {
        ApiError error;
        error.status  = status;
        error.type    = status >= 500 ? "server_error" : "invalid_request_error";
        error.code    = std::move(code);
        error.message = std::move(message);
        write_error(res, error);
    };
    if (options_.slot_save_path.empty()) {
        fail(501, "slot_persistence_disabled",
             "this server was started without --slot-save-path; slot save/restore is disabled");
        return;
    }
    const std::string id_text = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    std::uint32_t slot        = 0;
    try {
        slot = static_cast<std::uint32_t>(std::stoul(id_text));
    } catch (const std::exception&) {
        fail(400, "invalid_slot", "slot id is not a number");
        return;
    }
    if (slot >= options_.max_concurrency) {
        fail(400, "invalid_slot",
             "slot " + id_text + " is outside this server's " +
                 std::to_string(options_.max_concurrency) + " slots");
        return;
    }
    const std::string action = req.get_param_value("action");

    // Body: {"filename": NAME} for save/restore, plus optional {"if_digest": DIGEST} on save
    // and erase - a precondition that the slot still holds the session the client means,
    // checked atomically with the operation (mismatch = 409 slot_session_mismatch).
    std::string filename;
    std::string if_digest;
    try {
        const nlohmann::json body =
            req.body.empty() ? nlohmann::json::object() : nlohmann::json::parse(req.body);
        filename  = body.value("filename", std::string());
        if_digest = body.value("if_digest", std::string());
    } catch (const std::exception&) {
        fail(400, "invalid_request", "request body is not valid JSON");
        return;
    }

    if (action == "erase") {
        try {
            const std::uint32_t erased = service_->slot_erase(slot, if_digest);
            log_line("slot erase id=" + id_text + " n_erased=" + std::to_string(erased));
            res.set_content(nlohmann::json{{"id_slot", slot}, {"n_erased", erased}}.dump(),
                            "application/json");
        } catch (const ninfer::RequestError& engine_error) {
            fail(409, "slot_busy", engine_error.what());
        } catch (const ninfer::SlotSessionMismatch& mismatch) {
            fail(409, "slot_session_mismatch", mismatch.what());
        }
        return;
    }
    if (action != "save" && action != "restore") {
        fail(400, "invalid_action", "action must be save, restore, or erase");
        return;
    }
    const std::optional<std::string> sanitized = sanitize_slot_filename(filename);
    if (!sanitized) {
        fail(400, "invalid_filename",
             "filename must be 1-" + std::to_string(kSlotFilenameMaxBytes) +
                 " chars of [A-Za-z0-9._-] and must not start with a dot");
        return;
    }
    const std::string path = options_.slot_save_path + "/" + *sanitized;

    try {
        if (action == "save") {
            const ninfer::SlotSaveResult saved = service_->slot_save(slot, path, if_digest);
            log_line("slot save id=" + id_text + " file=" + *sanitized +
                     " n_saved=" + std::to_string(saved.tokens) +
                     " n_written=" + std::to_string(saved.bytes) +
                     " session=" + saved.session_digest + " in " +
                     format_seconds(saved.seconds) + " s");
            res.set_content(
                nlohmann::json{{"id_slot", slot},
                               {"filename", *sanitized},
                               {"n_saved", saved.tokens},
                               {"n_written", saved.bytes},
                               {"session_digest", saved.session_digest},
                               {"timings", {{"save_ms", saved.seconds * 1000.0}}}}
                    .dump(),
                "application/json");
        } else {
            const ninfer::SlotRestoreResult restored = service_->slot_restore(slot, path);
            log_line("slot restore id=" + id_text + " file=" + *sanitized +
                     " n_restored=" + std::to_string(restored.tokens) +
                     " n_read=" + std::to_string(restored.bytes) +
                     " session=" + restored.session_digest + " in " +
                     format_seconds(restored.seconds) + " s");
            res.set_content(
                nlohmann::json{{"id_slot", slot},
                               {"filename", *sanitized},
                               {"n_restored", restored.tokens},
                               {"n_read", restored.bytes},
                               {"session_digest", restored.session_digest},
                               {"timings", {{"restore_ms", restored.seconds * 1000.0}}}}
                    .dump(),
                "application/json");
        }
    } catch (const ninfer::RequestError& engine_error) {
        fail(409, "slot_busy", engine_error.what());
    } catch (const ninfer::SlotSessionMismatch& mismatch) {
        fail(409, "slot_session_mismatch", mismatch.what());
    } catch (const std::invalid_argument& engine_error) {
        fail(400, "slot_" + action + "_failed", engine_error.what());
    }
}

void HttpServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req.body);
    } catch (const std::exception&) {
        ApiError error;
        error.status  = 400;
        error.message = "request body is not valid JSON";
        write_error(res, error);
        return;
    }

    GenerationRequest request;
    PreparedRequest prepared;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        request                   = parse_chat_completion_request(body, limits);
        request.adapter           = require_model(request.model);
        prepared                  = service_->prepare(
            request, [&req] { return req.is_connection_alive && !req.is_connection_alive(); });
    } catch (const ApiException& e) {
        write_error(res, e.error());
        return;
    }

    const std::string id       = new_chat_completion_id();
    const std::int64_t created = unix_time_now();
    const std::string model    = request.model;

    const std::uint64_t req_id = ++request_seq_;
    // Pre-routing creates the client-visible ID before any generation handler runs. Capturing that
    // exact response header joins operator logs to the client while req_id remains metrics
    // identity.
    const RequestLogContext log_context = make_request_log_context(
        req_id, res.get_header_value("x-request-id"), "openai_chat_completions", request, prepared);
    log_request_start(log_context);

    if (!request.stream) {
        try {
            const GenerationOutcome outcome = service_->run(prepared, nullptr, [&req] {
                return req.is_connection_alive && !req.is_connection_alive();
            });
            log_request_done(log_context, outcome);
            const CompletionUsage usage = usage_with_timings(outcome);
            std::string response_body;
            if (!outcome.tool_calls.empty()) {
                response_body = make_chat_completion_tool_response(
                    id, model, created, outcome.text, outcome.reasoning, outcome.tool_calls, usage);
            } else {
                response_body = make_chat_completion_response(
                    id, model, created, outcome.text, outcome.reasoning,
                    finish_reason_wire(outcome.finish_reason), usage);
            }
            set_owned_content(res, std::move(response_body), prepared.lifetime);
        } catch (const std::exception& e) {
            log_request_error(log_context, e.what());
            throw;
        }
        return;
    }

    auto stream              = std::make_shared<StreamingRequest>(std::move(prepared));
    const bool include_usage = stream->prepared.include_usage;
    const bool tool_capable  = stream->prepared.tool_capable;

    // SSE hints: disable client/proxy caching and reverse-proxy response buffering
    // so tokens flush immediately. Content-Type is set by the chunked provider.
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");

    res.set_chunked_content_provider(
        "text/event-stream",
        [this, stream, id, created, model, include_usage, tool_capable,
         log_context](std::size_t, httplib::DataSink& sink) -> bool {
            if (stream->started) {
                sink.done();
                return true;
            }
            stream->started = true;
            try {
                write_stream_item(sink, *stream,
                                  make_chat_chunk_role(id, model, created, include_usage));
                StreamSink output;
                output.on_content = [&](const std::string& text) {
                    write_stream_item(
                        sink, *stream,
                        make_chat_chunk_content(id, model, created, text, include_usage));
                };
                output.on_reasoning = [&](const std::string& text) {
                    write_stream_item(
                        sink, *stream,
                        make_chat_chunk_reasoning(id, model, created, text, include_usage));
                };
                output.is_cancelled = [&] {
                    return stream->cancelled.load(std::memory_order_acquire) ||
                           (sink.is_writable && !sink.is_writable());
                };

                const GenerationOutcome outcome = service_->run(stream->prepared, &output);
                log_request_done(log_context, outcome);
                const CompletionUsage usage = usage_with_timings(outcome);
                const std::string_view remaining = unstreamed_content(outcome);
                if (!outcome.tool_calls.empty()) {
                    if (!remaining.empty()) {
                        write_stream_item(sink, *stream,
                                          make_chat_chunk_content(id, model, created,
                                                                  std::string(remaining),
                                                                  include_usage));
                    }
                    write_stream_item(sink, *stream,
                                      make_chat_chunk_tool_calls(
                                          id, model, created, outcome.tool_calls, include_usage));
                    write_stream_item(sink, *stream,
                                      make_chat_chunk_final(id, model, created, "tool_calls",
                                                            include_usage, usage));
                } else {
                    if (tool_capable && !remaining.empty()) {
                        write_stream_item(sink, *stream,
                                          make_chat_chunk_content(id, model, created,
                                                                  std::string(remaining),
                                                                  include_usage));
                    }
                    write_stream_item(
                        sink, *stream,
                        make_chat_chunk_final(id, model, created,
                                              finish_reason_wire(outcome.finish_reason),
                                              include_usage, usage));
                }
                if (include_usage) {
                    write_stream_item(sink, *stream,
                                      make_chat_chunk_usage(id, model, created, usage));
                }
                write_stream_item(sink, *stream, sse_done());
                sink.done();
                return true;
            } catch (const ClientDisconnected& e) {
                log_request_error(log_context, e.what());
                return false;
            } catch (const ApiException& e) {
                log_request_error(log_context, e.error().message);
                try {
                    write_stream_item(sink, *stream, sse_error_event(public_error(e.error())));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            } catch (const std::exception& e) {
                log_request_error(log_context, e.what());
                try {
                    write_stream_item(sink, *stream, sse_error_event(public_internal_error()));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            }
        },
        [stream](bool) { stream->cancelled.store(true, std::memory_order_release); });
}

void HttpServer::handle_count_tokens(const httplib::Request& req, httplib::Response& res) {
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req.body);
    } catch (const std::exception&) {
        ApiError error;
        error.status  = 400;
        error.message = "request body is not valid JSON";
        write_messages_error(res, error);
        return;
    }
    try {
        RequestLimits limits;
        limits.default_max_tokens       = options_.default_max_tokens;
        const GenerationRequest request = parse_messages_request(body, limits);
        const int input_tokens          = service_->count_prompt_tokens(
            request, [&req] { return req.is_connection_alive && !req.is_connection_alive(); });
        res.set_content(make_count_tokens_response(input_tokens), "application/json");
    } catch (const ApiException& e) {
        write_messages_error(res, e.error());
    } catch (const std::exception& e) {
        write_console_log(ConsoleLogLevel::Error,
                          "count_tokens exception: " + std::string(e.what()));
        write_messages_error(res, public_internal_error());
    }
}

void HttpServer::handle_messages(const httplib::Request& req, httplib::Response& res) {
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req.body);
    } catch (const std::exception&) {
        ApiError error;
        error.status  = 400;
        error.message = "request body is not valid JSON";
        write_messages_error(res, error);
        return;
    }

    GenerationRequest request;
    PreparedRequest prepared;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        // The Anthropic endpoint accepts any `model` string (Claude Code sends real
        // Claude model names) and echoes it back; it never 404s on model id. A string that
        // does name a served model still selects it, so adapters are reachable here too.
        request         = parse_messages_request(body, limits);
        request.adapter = resolve_model(request.model).value_or(std::string());
        prepared        = service_->prepare(
            request, [&req] { return req.is_connection_alive && !req.is_connection_alive(); });
    } catch (const ApiException& e) {
        write_messages_error(res, e.error());
        return;
    } catch (const std::exception& e) {
        write_console_log(ConsoleLogLevel::Error,
                          "messages preparation exception: " + std::string(e.what()));
        write_messages_error(res, public_internal_error());
        return;
    }

    const std::string id    = new_message_id();
    const std::string model = request.model; // echo the requested model
    const int input_tokens  = prepared.prompt_tokens;

    const std::uint64_t req_id          = ++request_seq_;
    const RequestLogContext log_context = make_request_log_context(
        req_id, res.get_header_value("x-request-id"), "anthropic_messages", request, prepared);
    log_request_start(log_context);

    if (!request.stream) {
        try {
            const GenerationOutcome outcome = service_->run(prepared, nullptr, [&req] {
                return req.is_connection_alive && !req.is_connection_alive();
            });
            log_request_done(log_context, outcome);
            const CompletionUsage usage = usage_with_timings(outcome);
            const char* stop_reason =
                messages_stop_reason(outcome.finish_reason, !outcome.tool_calls.empty());
            set_owned_content(res,
                              make_messages_response(id, model, outcome.text, outcome.reasoning,
                                                     outcome.tool_calls, stop_reason, usage),
                              prepared.lifetime);
        } catch (const ApiException& e) {
            log_request_error(log_context, e.error().message);
            write_messages_error(res, e.error());
        } catch (const std::exception& e) {
            log_request_error(log_context, e.what());
            write_messages_error(res, public_internal_error());
        }
        return;
    }

    auto stream             = std::make_shared<StreamingRequest>(std::move(prepared));
    const bool tool_capable = stream->prepared.tool_capable;

    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");

    res.set_chunked_content_provider(
        "text/event-stream",
        [this, stream, id, model, input_tokens, tool_capable,
         log_context](std::size_t, httplib::DataSink& sink) -> bool {
            if (stream->started) {
                sink.done();
                return true;
            }
            stream->started = true;

            int next_index     = 0;
            bool thinking_open = false;
            int thinking_index = -1;
            bool text_open     = false;
            int text_index     = -1;
            try {
                write_stream_item(sink, *stream, make_message_start(id, model, input_tokens));

                StreamSink output;
                output.on_reasoning = [&](const std::string& text) {
                    if (!thinking_open) {
                        thinking_index = next_index++;
                        thinking_open  = true;
                        write_stream_item(sink, *stream,
                                          make_content_block_start_thinking(thinking_index));
                    }
                    write_stream_item(sink, *stream,
                                      make_content_block_delta_thinking(thinking_index, text));
                };
                output.on_content = [&](const std::string& text) {
                    if (thinking_open) {
                        write_stream_item(sink, *stream, make_content_block_stop(thinking_index));
                        thinking_open = false;
                    }
                    if (!text_open) {
                        text_index = next_index++;
                        text_open  = true;
                        write_stream_item(sink, *stream, make_content_block_start_text(text_index));
                    }
                    write_stream_item(sink, *stream,
                                      make_content_block_delta_text(text_index, text));
                };
                output.is_cancelled = [&] {
                    return stream->cancelled.load(std::memory_order_acquire) ||
                           (sink.is_writable && !sink.is_writable());
                };

                const GenerationOutcome outcome = service_->run(stream->prepared, &output);
                log_request_done(log_context, outcome);
                const std::string_view remaining = unstreamed_content(outcome);

                if (thinking_open) {
                    write_stream_item(sink, *stream, make_content_block_stop(thinking_index));
                    thinking_open = false;
                }
                if (text_open) {
                    write_stream_item(sink, *stream, make_content_block_stop(text_index));
                    text_open = false;
                }

                if (tool_capable) {
                    if (!remaining.empty()) {
                        const int idx = next_index++;
                        write_stream_item(sink, *stream, make_content_block_start_text(idx));
                        write_stream_item(
                            sink, *stream,
                            make_content_block_delta_text(idx, std::string(remaining)));
                        write_stream_item(sink, *stream, make_content_block_stop(idx));
                    }
                    for (const ToolCall& call : outcome.tool_calls) {
                        const int idx = next_index++;
                        write_stream_item(sink, *stream,
                                          make_content_block_start_tool_use(idx, call));
                        write_stream_item(
                            sink, *stream,
                            make_content_block_delta_tool_json(idx, call.arguments_json));
                        write_stream_item(sink, *stream, make_content_block_stop(idx));
                    }
                }

                if (next_index == 0) {
                    const int idx = next_index++;
                    write_stream_item(sink, *stream, make_content_block_start_text(idx));
                    write_stream_item(sink, *stream, make_content_block_stop(idx));
                }

                const char* stop_reason =
                    messages_stop_reason(outcome.finish_reason, !outcome.tool_calls.empty());
                write_stream_item(sink, *stream,
                                  make_message_delta(stop_reason, usage_with_timings(outcome)));
                write_stream_item(sink, *stream, make_message_stop());
                sink.done();
                return true;
            } catch (const ClientDisconnected& e) {
                log_request_error(log_context, e.what());
                return false;
            } catch (const ApiException& e) {
                log_request_error(log_context, e.error().message);
                try {
                    write_stream_item(sink, *stream,
                                      messages_sse_error_event(public_error(e.error())));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            } catch (const std::exception& e) {
                log_request_error(log_context, e.what());
                try {
                    write_stream_item(sink, *stream,
                                      messages_sse_error_event(public_internal_error()));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            }
        },
        [stream](bool) { stream->cancelled.store(true, std::memory_order_release); });
}

bool HttpServer::bind() { return server_.bind_to_port(options_.host, options_.port); }

void HttpServer::attach(GenerationService& service) {
    if (service_ != nullptr) {
        throw std::logic_error("HTTP generation service is already attached");
    }
    const ninfer::LoadSummary load = service.load_summary();
    public_model_id_               = resolve_public_model_id(options_, load.model_id);
    adapter_names_                 = load.lora_adapter_names;
    adapter_model_ids_.clear();
    adapter_model_ids_.reserve(adapter_names_.size());
    for (const std::string& name : adapter_names_) {
        adapter_model_ids_.push_back(public_model_id_ + "-" + name);
    }
    service_                       = &service;
    events_.emit_server_start(options_, service.sampling_defaults(), public_model_id_, load,
                              service.memory_summary());
}

bool HttpServer::listen() {
    if (service_ == nullptr) { throw std::logic_error("HTTP generation service is not attached"); }
    if (public_model_id_.empty()) {
        throw std::logic_error("HTTP public model id is not resolved");
    }
    if (options_.log_stats_interval_ms != 0) {
        stats_stopping_ = false;
        stats_thread_   = std::thread([this] { run_stats_reporter(); });
    }
    try {
        const bool result = server_.listen_after_bind();
        stop_stats_reporter();
        return result;
    } catch (...) {
        stop_stats_reporter();
        throw;
    }
}

void HttpServer::stop() {
    events_.close_all();
    server_.stop();
}

} // namespace ninfer::serve
