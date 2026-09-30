#pragma once

// Human-readable request summaries and the optional full-precision JSONL event log used for
// measurement. The HTTP layer owns request ids; this module owns one stable JSON schema and
// serializes concurrent writes from non-streaming handlers and streaming workers.

#include "serve/generation_service.h"
#include "serve/request.h"
#include "serve/serve_options.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>

namespace ninfer::serve {

// Schema 22 reports chat and System One side by side: server_start lists the adapter pool with each
// adapter's kind and served model id plus the System One surface, and the throughput record splits
// out decision prefill and counts LoRA bank swaps and slot waits.
inline constexpr int kRequestLogSchemaVersion        = 22;
inline constexpr const char* kRequestLogArtifactType = "ninfer_serve_request_log";

// The --kv-dtype spelling of a KV storage.
[[nodiscard]] const char* kv_cache_name(ninfer::KvCacheStorage storage) noexcept;

// A float widened to the double that prints as the float's shortest decimal: 1.1F becomes 1.1,
// not 1.100000023841858.
[[nodiscard]] double shortest_double(float value) noexcept;

// The LoRA pool as server_start and /telemetry both report it, so the two cannot drift: the bank
// (`count`, `rank`, `slots`, `device_bytes`, `file_bytes`) and `pool`, every discovered adapter in
// pool order with its kind and the model id that selects it. A generative adapter is the chat
// model `<public model id>-<name>` on /v1; a decision adapter is the System One model `<name>` on
// /systemone and carries its calibration and model-card fields.
[[nodiscard]] nlohmann::json adapter_inventory_json(const ninfer::LoadSummary& load,
                                                    const std::string& public_model_id);
// The System One surface: whether the target serves decisions, and the decision adapter the
// SDK-default model name answers with (empty when that name is unbound).
[[nodiscard]] nlohmann::json systemone_json(const ninfer::LoadSummary& load,
                                            const std::string& binding);

struct RequestLogContext {
    std::uint64_t id = 0;
    std::string x_request_id;
    std::string protocol;
    std::string model;
    bool stream                             = false;
    int prompt_tokens                       = 0;
    std::size_t message_count               = 0;
    int requested_output_tokens             = 0;
    bool requested_output_tokens_client_set = false;
    std::size_t tool_count                  = 0;
    ToolChoice tool_choice;
    bool has_tool_history                  = false;
    bool enable_thinking                   = true;
    bool preserve_thinking                 = false;
    bool preserve_thinking_semantic_change = false;
    // Adapter selected by the model string; empty means base weights. Continuation aliases are
    // namespaced by adapter, so cache behaviour cannot be read without it.
    std::string adapter;
    // Truncated digest of the client's prompt_cache_key, or empty when the client sent none.
    // The key itself can carry client-chosen text, and only its identity matters here: equal
    // digests are the same routed session, different digests are different sessions.
    std::string prompt_cache_key_digest;
    ninfer::ResolvedSamplingParameters sampling;
};

// A System One decision as submitted: the model it names, the decision adapter that answers it,
// and its token layout.
struct DecisionLogContext {
    std::uint64_t id = 0;
    // The x-typesafe-request-id the response carries.
    std::string x_request_id;
    std::string model;
    std::string adapter;
    ninfer::DecisionSummary summary;
    bool allow_prefix_reuse = true;
    // Receipt to submission: validation, rendering, the ingress reservation and the layout.
    double prepare_seconds = 0.0;
};

struct ServerLogEnvironment {
    int device = 0;
    std::string gpu_name;
    std::string gpu_uuid;
    std::uint64_t total_device_memory_bytes = 0;
    int compute_capability_major            = 0;
    int compute_capability_minor            = 0;
    std::string cuda_compile_version;
    std::string cuda_runtime_version;
    std::string cuda_driver_version;
};

// One interval's board energy, exact where the board implements the cumulative counter.
//
// `board_joules` is measured; the prefill and decode splits are the executor's per-unit integration
// of instantaneous power. `residual_joules` is what the splits and the idle baseline together fail
// to explain, and it is published rather than distributed, because silently folding it into a phase
// would make a wrong number look like a measured one.
struct IntervalEnergy {
    bool available            = false;
    double board_joules       = 0.0;
    double prefill_joules     = 0.0;
    double decode_joules      = 0.0;
    double accounted_seconds  = 0.0;
    double idle_watts         = 0.0;
    double idle_joules        = 0.0;
    double residual_joules    = 0.0;
    double residual_fraction  = 0.0;
};

// Exact board energy consumed over an interval, plus the idle baseline calibrated from intervals
// that ran no execution unit at all.
struct BoardEnergySample {
    bool available    = false;
    double joules     = 0.0;
    double idle_watts = 0.0;
};

struct ThroughputReport {
    double interval_seconds               = 0.0;
    std::uint64_t computed_prefill_tokens = 0;
    // The part of computed_prefill_tokens System One decisions evaluated.
    std::uint64_t decision_prefill_tokens = 0;
    std::uint64_t committed_decode_tokens = 0;
    std::uint64_t decode_rounds           = 0;
    std::uint64_t decode_row_rounds       = 0;
    IntervalEnergy energy;
    ninfer::RuntimeStats scheduler;
    ninfer::RuntimeStats continuation_delta;
};

ThroughputReport make_throughput_report(const ninfer::RuntimeStats& previous,
                                        const ninfer::RuntimeStats& current,
                                        double interval_seconds,
                                        const BoardEnergySample& board_energy = {});
[[nodiscard]] bool throughput_report_has_activity(const ThroughputReport& report) noexcept;

RequestLogContext make_request_log_context(std::uint64_t id, std::string x_request_id,
                                           std::string protocol, const GenerationRequest& request,
                                           const PreparedRequest& prepared);

// Compact console records retained for operator visibility.
std::string format_request_start(const RequestLogContext& context);
std::string format_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
std::string format_request_error(const RequestLogContext& context, const std::string& message);
std::string format_throughput(const ThroughputReport& report);
std::string format_decision_start(const DecisionLogContext& context);
std::string format_decision_done(const DecisionLogContext& context,
                                 const ninfer::DecisionResult& result, std::uint32_t output_tokens);
std::string format_decision_error(const DecisionLogContext& context, int status,
                                  const std::string& message);

// Pure JSON formatters are public to repository tests. Each return value is one complete JSON
// object without a trailing newline.
// `systemone_binding` is the resolved SDK-default binding, not the --systemone-default flag.
std::string format_server_start_json(
    const std::string& server_instance_id, std::uint64_t timestamp_unix_ms,
    const ServeOptions& options, const ninfer::ModelSamplingDefaults& sampling_defaults,
    const std::string& public_model_id, const std::string& systemone_binding,
    const ninfer::LoadSummary& load, const ninfer::MemorySummary& memory,
    const ServerLogEnvironment& environment, std::optional<std::uint64_t> artifact_size_bytes);
std::string format_request_start_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp_unix_ms,
                                      const RequestLogContext& context);
std::string format_request_done_json(const std::string& server_instance_id,
                                     std::uint64_t timestamp_unix_ms,
                                     const RequestLogContext& context,
                                     const GenerationOutcome& outcome);
std::string format_request_error_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp_unix_ms,
                                      const RequestLogContext& context, const std::string& message);
std::string format_throughput_json(const std::string& server_instance_id,
                                   std::uint64_t timestamp_unix_ms, const ThroughputReport& report);
std::string format_decision_start_json(const std::string& server_instance_id,
                                       std::uint64_t timestamp_unix_ms,
                                       const DecisionLogContext& context);
std::string format_decision_done_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp_unix_ms,
                                      const DecisionLogContext& context,
                                      const ninfer::DecisionResult& result,
                                      std::uint32_t output_tokens);
std::string format_decision_error_json(const std::string& server_instance_id,
                                       std::uint64_t timestamp_unix_ms,
                                       const DecisionLogContext& context, int status,
                                       const std::string& message);

ServerLogEnvironment query_server_log_environment(int device);

// Record timestamps and the per-process identity every record carries. Exposed because EventStream
// owns the funnel that stamps a record once and fans it out to its sinks.
std::uint64_t unix_time_ms();
std::string new_server_instance_id();

// Append-only file sink for already-formatted records. Opens in append mode so one campaign file
// can contain multiple independently started MTP/model blocks. It does not format or stamp
// anything: EventStream owns the schema instance and hands this class complete lines.
class JsonlRequestLog {
public:
    explicit JsonlRequestLog(const std::string& path,
                             const std::string& protected_artifact_path = {});

    JsonlRequestLog(const JsonlRequestLog&)            = delete;
    JsonlRequestLog& operator=(const JsonlRequestLog&) = delete;

    [[nodiscard]] bool enabled() const noexcept { return output_.is_open(); }

    void write_record(const std::string& record);

private:
    std::string path_;
    std::ofstream output_;
    std::mutex mutex_;
    bool failed_ = false;
};

} // namespace ninfer::serve
