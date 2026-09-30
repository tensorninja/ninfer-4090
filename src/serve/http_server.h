#pragma once

#include "serve/event_stream.h"
#include "serve/generation_service.h"
#include "serve/gpu_telemetry.h"
#include "serve/response_store.h"
#include "serve/request_log.h"
#include "serve/serve_metrics.h"
#include "serve/serve_options.h"
#include "serve/systemone_service.h"

#include <httplib.h>

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace ninfer::serve {

class HttpServer {
public:
    explicit HttpServer(ServeOptions options);

    // Reserves the configured address before model loading. The service is attached only after its
    // Engine is ready, then listen() enters the blocking accept loop on the already-bound socket.
    bool bind();
    void attach(GenerationService& service);
    bool listen();
    void stop();

    [[nodiscard]] const std::string& public_model_id() const noexcept { return public_model_id_; }

    // The decision adapter the System One SDK-default model name answers with; empty when the
    // alias is unbound or no service is attached.
    [[nodiscard]] std::string systemone_alias_binding() const {
        return systemone_ ? systemone_->alias_binding() : std::string();
    }

    // Resolves an API `model` string to the adapter it selects. Returns nullopt when the string
    // names no served model; an empty string means the base weights.
    [[nodiscard]] std::optional<std::string> resolve_model(const std::string& model) const;

    // Same resolution for endpoints that must reject an unknown `model`, returning the selected
    // adapter name and throwing the OpenAI-shaped 404 otherwise.
    [[nodiscard]] std::string require_model(const std::string& model) const;

private:
    void register_routes();
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);
    void handle_messages(const httplib::Request& req, httplib::Response& res);
    void handle_count_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_responses(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_response_get(const httplib::Request& req, httplib::Response& res);
    void handle_response_delete(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_items(const httplib::Request& req, httplib::Response& res);
    void handle_response_cancel(const httplib::Request& req, httplib::Response& res);
    void handle_response_compact(const httplib::Request& req, httplib::Response& res);
    void handle_models(const httplib::Request& req, httplib::Response& res) const;
    void handle_model(const httplib::Request& req, httplib::Response& res) const;
    void handle_slot_action(const httplib::Request& req, httplib::Response& res);
    void handle_telemetry(const httplib::Request& req, httplib::Response& res) const;
    void handle_events(const httplib::Request& req, httplib::Response& res);
    // System One (TypeSafe) routes under the /typesafe base path.
    void handle_systemone(const httplib::Request& req, httplib::Response& res);
    void handle_systemone_models(const httplib::Request& req, httplib::Response& res) const;

    // The process-wide console logger serializes lines from request and reporter threads.
    void log_line(const std::string& line);
    void log_request_start(const RequestLogContext& context);
    void log_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void log_request_error(const RequestLogContext& context, const std::string& message);
    void log_decision_start(const DecisionLogContext& context);
    void log_decision_done(const DecisionLogContext& context, const ninfer::DecisionResult& result,
                           std::uint32_t output_tokens);
    void log_decision_error(const DecisionLogContext& context, int status,
                            const std::string& message);
    void log_throughput(const ThroughputReport& report);
    void run_stats_reporter();
    void stop_stats_reporter();

    // Written only by the reporter thread; the idle baseline needs no synchronization there.
    [[nodiscard]] std::optional<double> read_board_energy_joules() const;
    void observe_idle_power(const ninfer::RuntimeStats& before, const ninfer::RuntimeStats& after,
                            double seconds, double joules);

    // Published to /metrics and /telemetry, which are served from request threads.
    [[nodiscard]] ServerEnergyTotals energy_totals() const;

    GenerationService* service_ = nullptr;
    ServeOptions options_;
    std::string public_model_id_;
    // Served model id per registered adapter, in bank order: `<public model id>-<adapter name>`.
    std::vector<std::string> adapter_model_ids_;
    std::vector<std::string> adapter_names_;
    // Present once a service is attached.
    std::optional<SystemOneService> systemone_;
    ResponseStore response_store_;
    ServeMetrics metrics_;
    EventStream events_;
    GpuTelemetryReader gpu_;
    std::chrono::steady_clock::time_point started_at_ = std::chrono::steady_clock::now();
    httplib::Server server_;
    std::atomic<std::uint64_t> request_seq_{0};
    std::mutex stats_mutex_;
    std::condition_variable stats_cv_;
    std::thread stats_thread_;
    bool stats_stopping_ = false;
    // Measured board draw with nothing executing, used to price the part of an interval that no
    // execution unit claimed. Zero until an idle interval has actually been observed.
    double idle_watts_ = 0.0;
    // Board energy since this server started. Accumulated from reporter-thread differences so that
    // a driver-reload reset drops one sample instead of fabricating or losing a whole total.
    std::atomic<bool> energy_available_{false};
    std::atomic<double> board_energy_joules_total_{0.0};
    std::atomic<double> published_idle_watts_{0.0};
};

} // namespace ninfer::serve
