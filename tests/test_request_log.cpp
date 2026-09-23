#include "serve/console_log.h"
#include "serve/event_stream.h"
#include "serve/request_log.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#    include <process.h>
#else
#    include <unistd.h>
#endif

namespace {

using namespace ninfer::serve;
using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    bool protected_artifact_rejected = false;
    try {
        EventStream unsafe("same-path.ninfer", "same-path.ninfer", 8);
    } catch (const std::invalid_argument&) { protected_artifact_rejected = true; }
    failures += check(protected_artifact_rejected,
                      "request log accepted the model artifact as its output path");

    ServeOptions options;
    options.artifact_path                = "/models/qwen3_8_27b.ninfer";
    options.host                         = "127.0.0.1";
    options.port                         = 8123;
    options.api_key                      = "must-not-appear";
    options.model_id_override            = "deployment-alias";
    options.request_log_jsonl            = "requests.jsonl";
    options.max_context                  = 262144;
    options.kv_capacity                  = ninfer::KvCapacityPolicy::explicit_capacity(524288);
    options.prefill_chunk                = 1024;
    options.log_stats_interval_ms        = 2500;
    options.kv_cache                     = ninfer::KvCacheStorage::Int8Group64;
    options.speculative.backend          = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens     = 3;
    options.speculative.proposal_head    = ninfer::ProposalHead::Optimized;
    options.enable_vision                = false;
    options.allow_prefix_reuse           = false;
    options.preserve_thinking            = true;
    options.continuation_cache.tiers     = ninfer::ContinuationCacheTiers::L1L2L3;
    options.continuation_cache.directory = "/var/cache/ninfer";
    options.continuation_cache.cache_namespace        = "tenant-a";
    options.continuation_cache.persist_min_tokens     = 512;
    options.continuation_cache.filesystem_reserve_mib = 4096;
    options.sampling_overrides.temperature            = 0.6F;
    options.startup_argv = {"ninfer-serve", options.artifact_path, "--api-key", "<redacted>"};

    const ninfer::ModelSamplingDefaults sampling_defaults{
        .thinking     = {.temperature = 1.0F, .top_k = 20, .top_p = 0.95F},
        .non_thinking = {.temperature = 0.7F, .top_k = 20, .top_p = 0.8F, .presence_penalty = 1.5F},
    };

    ninfer::LoadSummary load;
    load.target               = "qwen3_8_27b";
    load.model_id             = "qwen3.8-27b";
    load.weights_id           = "groupwise-int";
    load.load_seconds         = 1.234567890123;
    load.upload_seconds       = 0.345678901234;
    load.artifact_bytes_read  = 1000;
    load.host_to_device_bytes = 900;
    load.peak_staging_bytes   = 128;
    load.tensor_count         = 42;
    load.resource_count       = 6;
    load.lora_adapter_names   = {"caveman", "sudoku"};
    load.lora_rank            = 16;
    load.lora_device_bytes    = 176947200;
    load.lora_file_bytes      = 88473600;

    ninfer::MemorySummary memory;
    memory.max_context                       = 262144;
    memory.kv_capacity_mode                  = ninfer::KvCapacityMode::Explicit;
    memory.kv_capacity                       = 524288;
    memory.kv_capacity_page_groups           = 8192;
    memory.kv_capacity_max_page_groups       = 16384;
    memory.kv_cache                          = ninfer::KvCacheStorage::Int8Group64;
    memory.weights.capacity_bytes            = 100;
    memory.sequence.capacity_bytes           = 200;
    memory.workspace.capacity_bytes          = 300;
    memory.request_transient                 = {500, 0, 450};
    memory.minimum_runtime_reservation_bytes = 1300;
    memory.kv_capacity_increment_bytes       = 100;
    memory.runtime_reservation_bytes         = 1600;
    memory.available_after_weights_bytes     = 1700;
    memory.available_after_startup_bytes     = 180;
    memory.planned_slack_bytes               = 100;
    memory.cuda_graph_allowance_bytes        = 600;
    memory.cuda_graph_observed_bytes         = 550;
    memory.kv_payload_bytes                  = 400;
    memory.lora_bank_bytes                   = 176947200;

    ServerLogEnvironment environment;
    environment.device                    = 0;
    environment.gpu_name                  = "NVIDIA GeForce RTX 5090";
    environment.gpu_uuid                  = "GPU-00000000-0000-0000-0000-000000000000";
    environment.total_device_memory_bytes = 32000000000ULL;
    environment.compute_capability_major  = 12;
    environment.compute_capability_minor  = 0;
    environment.cuda_compile_version      = "13.1";
    environment.cuda_runtime_version      = "13.1";
    environment.cuda_driver_version       = "13.1";

    const Json server = Json::parse(
        format_server_start_json("serve-test", 1000, options, sampling_defaults, "deployment-alias",
                                 load, memory, environment, std::uint64_t{123456}));
    failures += check(server.at("artifact_type") == kRequestLogArtifactType,
                      "server record artifact type mismatch");
    failures += check(server.at("schema_version") == kRequestLogSchemaVersion,
                      "server record schema mismatch");
    failures += check(server.at("event") == "server_start", "server event mismatch");
    failures += check(server.at("server").at("public_model_id") == "deployment-alias",
                      "resolved public model id missing");
    failures += check(server.at("artifact").at("target") == "qwen3_8_27b", "server target missing");
    failures += check(server.at("artifact").at("weights_id") == "groupwise-int",
                      "server weights id missing");
    failures += check(server.at("artifact").at("size_bytes") == 123456, "artifact size missing");
    failures += check(server.at("adapters").at("count") == 2, "adapter count missing");
    failures += check(server.at("adapters").at("names").at(1) == "sudoku",
                      "adapter names missing or out of bank order");
    failures += check(server.at("adapters").at("rank") == 16, "adapter rank missing");
    failures += check(server.at("adapters").at("device_bytes") == 176947200,
                      "adapter bank device bytes missing");
    failures += check(server.at("memory").at("lora_bank_bytes") == 176947200,
                      "lora bank bytes missing from the memory division");
    failures += check(server.at("engine").at("max_context") == 262144, "max context missing");
    failures += check(server.at("engine").at("kv_capacity") == 524288, "KV capacity missing");
    failures += check(server.at("engine").at("kv_capacity_mode") == "explicit" &&
                          server.at("engine").at("kv_capacity_page_groups") == 8192 &&
                          server.at("engine").at("kv_capacity_max_page_groups") == 16384,
                      "KV capacity resolution metadata missing");
    failures +=
        check(server.at("engine").at("log_stats_interval_ms") == 2500, "stats interval missing");
    failures += check(server.at("server").at("request_log_jsonl") == "requests.jsonl",
                      "request log path missing");
    failures += check(server.at("engine").at("kv_cache") == "int8-group64", "KV type missing");
    failures += check(server.at("engine").at("vision") == false, "Vision state missing");
    failures += check(server.at("engine").at("speculative_backend") == "mtp",
                      "speculative backend missing");
    failures +=
        check(server.at("engine").at("proposal_head") == "optimized", "proposal head missing");
    failures +=
        check(server.at("engine").at("prefix_reuse") == false, "prefix-reuse state missing");
    failures += check(server.at("engine").at("prefix_checkpoint_policy") == "rolling-tool",
                      "prefix checkpoint policy missing");
    const Json& continuation_cache = server.at("engine").at("continuation_cache");
    failures += check(continuation_cache.at("tiers") == "l1-l2-l3" &&
                          continuation_cache.at("policy") == "adaptive" &&
                          continuation_cache.at("directory") == "/var/cache/ninfer" &&
                          continuation_cache.at("namespace") == "tenant-a" &&
                          continuation_cache.at("l1_capacity_mib") == 768 &&
                          continuation_cache.at("l2_capacity_mib") == 16384 &&
                          continuation_cache.at("l3_capacity_mib") == 49152 &&
                          continuation_cache.at("l1_idle_ttl_seconds") == 600 &&
                          continuation_cache.at("l2_idle_ttl_seconds") == 7200 &&
                          continuation_cache.at("l3_idle_ttl_seconds") == 86400 &&
                          continuation_cache.at("persist_interval_seconds") == 60 &&
                          continuation_cache.at("persist_min_tokens") == 512 &&
                          continuation_cache.at("filesystem_reserve_mib") == 4096 &&
                          continuation_cache.at("prefix_checkpoint_history") == 4,
                      "continuation cache configuration missing from server-start record");
    failures += check(server.at("server").at("default_preserve_thinking") == true,
                      "server preserve-thinking default missing");
    failures +=
        check(server.at("sampling_defaults").at("thinking").at("temperature") == 1.0 &&
                  server.at("sampling_defaults").at("non_thinking").at("presence_penalty") == 1.5,
              "registered mode-specific sampling defaults missing");
    failures += check(
        server.at("sampling_defaults").at("server_overrides").at("temperature").get<float>() ==
                0.6F &&
            server.at("sampling_defaults").at("server_overrides").at("top_p").is_null(),
        "server sampling overrides lost omission state");
    failures += check(server.at("environment").at("gpu_name") == "NVIDIA GeForce RTX 5090",
                      "GPU name missing");
    failures += check(server.at("memory").at("request_transient").at("capacity_bytes") == 500 &&
                          server.at("memory").at("request_transient").at("peak_used_bytes") == 450,
                      "request transient memory missing");
    failures += check(server.at("memory").at("cuda_graph_allowance_bytes") == 600,
                      "CUDA Graph allowance missing");
    failures += check(server.at("memory").at("runtime_reservation_bytes") == 1600 &&
                          server.at("memory").at("available_after_weights_bytes") == 1700 &&
                          server.at("memory").at("available_after_startup_bytes") == 180 &&
                          server.at("memory").at("kv_capacity_headroom_bytes") == 0 &&
                          server.at("memory").at("planned_slack_bytes") == 100 &&
                          server.at("memory").at("cuda_graph_observed_bytes") == 550,
                      "adaptive KV memory ledger missing");
    failures += check(server.dump().find("must-not-appear") == std::string::npos,
                      "server JSON leaked the API key");
    failures += check(server.at("argv").at(3) == "<redacted>",
                      "server argv did not retain the redaction marker");

    GenerationRequest request;
    request.model          = "qwen3.8-27b";
    request.stream         = false;
    request.max_tokens     = 4096;
    request.max_tokens_set = true;
    request.messages.resize(2);
    request.adapter        = "sudoku-lora";
    // A routed session is logged as a digest, never as the caller's key.
    request.prompt_cache_routing_hint = "opencode-session-7f31";

    PreparedRequest prepared;
    prepared.enable_thinking                   = false;
    prepared.preserve_thinking                 = true;
    prepared.preserve_thinking_semantic_change = true;
    prepared.sampling.temperature              = 0.6F;
    prepared.sampling.top_p                    = 0.95F;
    prepared.sampling.top_k                    = 20;
    prepared.sampling.min_p                    = 0.0F;
    prepared.sampling.presence_penalty         = 1.0F;
    prepared.sampling.frequency_penalty        = 0.0F;
    prepared.sampling.seed                     = 7632647173703958409ULL;

    const RequestLogContext context =
        make_request_log_context(7, "req_public_abc", "openai_chat_completions", request, prepared);
    const Json started = Json::parse(format_request_start_json("serve-test", 2000, context));
    failures +=
        check(started.at("request").at("request_id") == 7, "request id missing from start record");
    failures += check(started.at("request").at("x_request_id") == "req_public_abc",
                      "client-visible request id missing from start record");
    failures += check(started.at("request").at("requested_output_tokens") == 4096,
                      "request output budget missing");
    failures += check(started.at("request").at("enable_thinking") == false,
                      "resolved thinking mode missing");
    failures += check(started.at("request").at("preserve_thinking") == true &&
                          started.at("request").at("preserve_thinking_semantic_change") == true,
                      "resolved preserve-thinking metadata missing");
    failures += check(started.at("request").at("sampling").at("seed") == 7632647173703958409ULL,
                      "resolved seed missing");
    failures += check(started.at("request").at("adapter") == "sudoku-lora",
                      "adapter missing from start record");
    failures += check(started.at("request").at("prompt_cache_key_digest").is_string() &&
                          started.at("request").at("prompt_cache_key_digest") !=
                              "opencode-session-7f31",
                      "routing hint must be logged as a digest, not verbatim");

    GenerationOutcome outcome;
    outcome.prompt_tokens                       = 401;
    outcome.completion_tokens                   = 1024;
    outcome.finish_reason                       = ninfer::FinishReason::OutputLimit;
    outcome.metrics.prepare_seconds             = 0.1234567890123;
    outcome.metrics.ttft_seconds                = 0.3580246791357;
    outcome.metrics.vision_seconds              = 0.0;
    outcome.metrics.prefill_seconds             = 0.2345678901234;
    outcome.metrics.decode_seconds              = 5.3456789012345;
    outcome.metrics.total_seconds               = 5.7037035803702;
    outcome.metrics.prefix_cache_hit_tokens     = 101;
    outcome.metrics.prefix_reuse_path           = ninfer::PrefixReusePath::RestoreTurnCheckpoint;
    outcome.metrics.speculative_backend         = ninfer::SpeculativeBackend::Mtp;
    outcome.metrics.speculative_draft_window    = 3;
    outcome.metrics.speculative_rounds          = 300;
    outcome.metrics.speculative_draft_tokens    = 900;
    outcome.metrics.speculative_accepted_tokens = 720;
    outcome.metrics.speculative_fallback_steps  = 2;
    outcome.metrics.speculative_accepted_per_position = {290, 240, 190};
    outcome.metrics.continuation.source = ninfer::ContinuationSource::L3;
    outcome.metrics.continuation.alias_kind = ninfer::ContinuationAliasKind::Session;
    outcome.metrics.continuation.final_miss_reason = ninfer::ContinuationMissReason::None;
    outcome.metrics.continuation.lookup_microseconds = 1234;
    outcome.metrics.continuation.preflight_microseconds = 56;
    outcome.metrics.continuation.restore_microseconds = 7890;
    outcome.metrics.continuation.restored_tokens = 101;
    outcome.metrics.continuation.restored_bytes = 4096;
    outcome.metrics.continuation.destructive_rollback = true;
    outcome.metrics.continuation.completion_publication_queued = true;
    outcome.metrics.continuation.restore_failure =
        ninfer::ContinuationRestoreFailure::KvReservationExhausted;
    // Queueing is the dominant TTFT term under concurrency, so it is part of the logged contract.
    outcome.metrics.queue_seconds   = 12.3456789012345;
    outcome.metrics.restore_seconds = 0.7890123456789;

    const Json done = Json::parse(format_request_done_json("serve-test", 3000, context, outcome));
    failures += check(done.at("request").at("x_request_id") == "req_public_abc",
                      "client-visible request id missing from done record");
    failures +=
        check(done.at("result").at("finish_reason") == "output_limit", "finish reason missing");
    failures += check(done.at("result").at("prompt_tokens") == 401, "prompt tokens missing");
    failures += check(done.at("result").at("computed_prefill_tokens") == 300,
                      "computed prefill tokens missing");
    failures += check(done.at("result").at("prefix_reuse_path") == "restore_turn_checkpoint",
                      "prefix reuse path missing");
    failures += check(done.at("timings_seconds").at("decode").get<double>() ==
                          outcome.metrics.decode_seconds,
                      "decode time lost precision");
    failures +=
        check(done.at("timings_seconds").at("ttft").get<double>() == outcome.metrics.ttft_seconds,
              "TTFT missing or lost precision");
    failures += check(done.at("speculative").at("backend") == "mtp", "speculative backend missing");
    failures +=
        check(done.at("speculative").at("draft_window") == 3, "speculative draft window missing");
    failures += check(done.at("speculative").at("fallback_steps") == 2,
                      "speculative fallback count missing");
    failures +=
        check(done.at("speculative").at("accepted_per_position") == Json::array({290, 240, 190}),
               "speculative position counts missing");
    failures += check(done.at("schema_version") == kRequestLogSchemaVersion &&
                          done.at("continuation_cache").at("source") == "l3" &&
                           done.at("continuation_cache").at("alias_kind") == "routed_session" &&
                          done.at("continuation_cache").at("final_miss_reason") == "none" &&
                          done.at("continuation_cache").at("lookup_microseconds") == 1234 &&
                          done.at("continuation_cache").at("restore_microseconds") == 7890 &&
                          done.at("continuation_cache").at("destructive_rollback") == true &&
                          done.at("continuation_cache")
                                  .at("completion_publication_queued") == true,
                       "schema-v13 continuation diagnostics missing");
    failures += check(done.at("continuation_cache").at("restore_failure") ==
                          "kv_reservation_exhausted",
                      "restore failure must be attributed, not discarded");
    failures += check(done.at("timings_seconds").at("queue").get<double>() ==
                              outcome.metrics.queue_seconds &&
                          done.at("timings_seconds").at("restore").get<double>() ==
                              outcome.metrics.restore_seconds,
                      "TTFT decomposition into queue and restore missing or imprecise");

    const Json error =
        Json::parse(format_request_error_json("serve-test", 4000, context, "generation failed"));
    failures += check(error.at("event") == "request_error", "request error event mismatch");
    failures += check(error.at("request").at("x_request_id") == "req_public_abc",
                      "client-visible request id missing from error record");
    failures += check(error.at("error").at("message") == "generation failed",
                      "request error message missing");

    failures += check(format_request_start(context).find("thinking=off") != std::string::npos,
                      "human request log omits resolved thinking mode");
    failures +=
        check(format_request_start(context).find("preserve_thinking=on") != std::string::npos,
              "human request log omits preserve-thinking mode");
    failures += check(format_request_done(context, outcome).find("reuse=restore_turn_checkpoint") !=
                          std::string::npos,
                       "human request log omits prefix reuse path");
    const std::string human_done = format_request_done(context, outcome);
    failures += check(human_done.find("cache_source=l3") != std::string::npos &&
                           human_done.find("cache_alias=routed_session") != std::string::npos &&
                          human_done.find("cache_miss=none") != std::string::npos &&
                          human_done.find("cache_lookup=1.234ms") != std::string::npos &&
                           human_done.find("cache_preflight=0.056ms") != std::string::npos &&
                           human_done.find("cache_restore=7.890ms") != std::string::npos &&
                           human_done.find("cache_tokens=101") != std::string::npos &&
                           human_done.find("cache_bytes=4096") != std::string::npos &&
                           human_done.find("cache_rollback=yes") != std::string::npos &&
                           human_done.find("cache_publish_queued=yes") != std::string::npos,
                      "human request log omits deterministic cache diagnostics");
    failures += check(format_request_start(context).find("submitted") != std::string::npos,
                      "human request log mislabels a submitted request");
    failures += check(
        format_request_start(context).find("x_request_id=req_public_abc") != std::string::npos &&
            format_request_done(context, outcome).find("x_request_id=req_public_abc") !=
                std::string::npos &&
            format_request_error(context, "failed").find("x_request_id=req_public_abc") !=
                std::string::npos,
        "human generation logs omit the client-visible request id");

    // Only a turn whose calls all closed hands control to the client; a call the output cut short
    // keeps the engine's finish and is counted, as are completed calls with non-object arguments.
    GenerationOutcome tool_outcome = outcome;
    tool_outcome.tool_calls        = {ToolCall{"call_a", "read", R"({"path":"a"})"},
                                      ToolCall{"call_b", "read", R"(["a"])"},
                                      ToolCall{"call_c", "read", R"({"path":"b)",
                                               ToolCallState::Incomplete}};
    tool_outcome.tool_call_discarded_bytes = 7;
    const Json tool_done =
        Json::parse(format_request_done_json("serve-test", 3000, context, tool_outcome));
    failures += check(tool_done.at("result").at("finish_reason") == "output_limit" &&
                          tool_done.at("result").at("tool_call_count") == 3 &&
                          tool_done.at("result").at("tool_calls_incomplete") == 1 &&
                          tool_done.at("result").at("tool_calls_invalid_arguments") == 1 &&
                          tool_done.at("result").at("tool_call_discarded_bytes") == 7,
                      "request_done must count incomplete, invalid, and dropped tool output");
    const std::string tool_human = format_request_done(context, tool_outcome);
    failures += check(tool_human.find("finish=output_limit tool_calls=3 tool_calls_incomplete=1 "
                                      "tool_calls_invalid_arguments=1 "
                                      "tool_call_discarded_bytes=7") != std::string::npos,
                      "human request log must not report an incomplete call turn as tool_calls");
    tool_outcome.tool_calls.pop_back();
    failures += check(format_request_done(context, tool_outcome).find("finish=tool_calls ") !=
                          std::string::npos,
                      "human request log must report a completed call turn as tool_calls");

    ThroughputReport throughput;
    throughput.interval_seconds                            = 2.0;
    throughput.computed_prefill_tokens                     = 100;
    throughput.committed_decode_tokens                     = 40;
    throughput.decode_rounds                               = 10;
    throughput.decode_row_rounds                           = 18;
    throughput.scheduler.running_requests                  = 2;
    throughput.scheduler.prefilling_requests               = 1;
    throughput.scheduler.decode_ready_requests             = 1;
    throughput.scheduler.waiting_requests                  = 3;
    throughput.scheduler.continuation_lookup_hits          = 7;
    throughput.scheduler.continuation_lookup_misses        = 2;
    throughput.scheduler.continuation_restore_successes    = 4;
    throughput.scheduler.continuation_publication_failures = 1;
    throughput.scheduler.continuation_publication_superseded = 2;
    throughput.scheduler.continuation_l1_restore_successes = 3;
    throughput.scheduler.continuation_l2_restore_successes = 4;
    throughput.scheduler.continuation_l3_restore_successes = 5;
    throughput.scheduler.continuation_l2_lookup_microseconds = 600;
    throughput.scheduler.continuation_l3_restore_microseconds = 700;
    throughput.continuation_delta.continuation_l1_restore_successes = 1;
    throughput.continuation_delta.continuation_l2_restore_successes = 2;
    throughput.continuation_delta.continuation_l3_restore_successes = 3;
    throughput.continuation_delta.continuation_l2_lookup_microseconds = 100;
    throughput.continuation_delta.continuation_l3_restore_microseconds = 200;
    throughput.continuation_delta.continuation_l2_lookup_operations = 6;
    throughput.scheduler.continuation_l2_restored_bytes = 4096;
    throughput.continuation_delta.continuation_l2_restored_bytes = 2048;
    throughput.scheduler.continuation_miss_preflight_rejected = 4;
    throughput.continuation_delta.continuation_miss_preflight_rejected = 1;
    throughput.scheduler.l1_resident_entries = 2;
    const std::string human_throughput                     = format_throughput(throughput);
    failures += check(human_throughput.find("prefill=50.0tok/s") != std::string::npos &&
                           human_throughput.find("decode=20.0tok/s") != std::string::npos &&
                           human_throughput.find("avg_decode_batch=1.80") != std::string::npos &&
                           human_throughput.find("cache_tier_delta=1/2/3") != std::string::npos &&
                           human_throughput.find("cache_lookup_us_delta=100/0") !=
                               std::string::npos,
                      "human throughput report mismatch");
    const Json throughput_json =
        Json::parse(format_throughput_json("serve-test", 5000, throughput));
    failures += check(throughput_json.at("event") == "throughput", "throughput event mismatch");
    failures += check(throughput_json.at("tokens").at("computed_prefill") == 100 &&
                          throughput_json.at("tokens").at("committed_decode") == 40,
                      "throughput token deltas mismatch");
    failures += check(throughput_json.at("decode_batch").at("average_size") == 1.8,
                      "throughput batch average mismatch");
    failures += check(throughput_json.at("continuation_cache").at("lookup_hits") == 7 &&
                          throughput_json.at("continuation_cache").at("lookup_misses") == 2 &&
                           throughput_json.at("continuation_cache").at("restore_successes") == 4 &&
                            throughput_json.at("continuation_cache").at("publication_failures") == 1 &&
                            throughput_json.at("continuation_cache").at("publication_superseded") == 2 &&
                            throughput_json.at("continuation_cache").at("tiers").at(
                                "delta_l3_restore_successes") == 3 &&
                            throughput_json.at("continuation_cache")
                                    .at("latency_microseconds")
                                    .at("l2_lookup_delta") == 100 &&
                            throughput_json.at("continuation_cache")
                                    .at("latency_microseconds")
                                     .at("l2_lookup_operations_delta") == 6,
                        "throughput continuation counters mismatch");
    failures += check(
        throughput_json.at("continuation_cache").at("tiers").at("l2_restored_bytes") == 4096 &&
            throughput_json.at("continuation_cache")
                    .at("tiers")
                    .at("delta_l2_restored_bytes") == 2048 &&
            throughput_json.at("continuation_cache")
                    .at("miss_reasons")
                    .at("delta_preflight_rejected") == 1 &&
            throughput_json.at("continuation_cache").at("occupancy").at("l1_entries") == 2,
        "throughput tier bytes, miss reasons, and occupancy missing");

    // Energy is absent, not zero, when the board exposes no cumulative counter. A consumer must be
    // able to tell "this server could not measure energy" from "this server used none".
    failures += check(throughput_json.at("energy").is_null(),
                      "throughput energy must be null without a board energy sample");

    // The board counter is exact; the phase split is integrated from power samples. Reconciliation
    // prices the time no execution unit claimed at the measured idle baseline and publishes what
    // is left over, so a consumer can see the estimator's error instead of inheriting it silently.
    ninfer::RuntimeStats energy_before;
    ninfer::RuntimeStats energy_after;
    energy_after.prefill_energy_joules    = 200.0;
    energy_after.decode_energy_joules     = 400.0;
    energy_after.energy_accounted_seconds = 6.0;
    energy_after.computed_prefill_tokens  = 1000;
    energy_after.committed_decode_tokens  = 100;
    BoardEnergySample board;
    board.available = true;
    board.joules    = 1000.0; // 200 prefill + 400 decode + 4 idle seconds at 95 W + 20 unexplained
    board.idle_watts = 95.0;
    const ThroughputReport metered =
        make_throughput_report(energy_before, energy_after, 10.0, board);
    failures += check(metered.energy.available && metered.energy.board_joules == 1000.0 &&
                          metered.energy.prefill_joules == 200.0 &&
                          metered.energy.decode_joules == 400.0 &&
                          metered.energy.accounted_seconds == 6.0 &&
                          metered.energy.idle_joules == 380.0 &&
                          metered.energy.residual_joules == 20.0 &&
                          std::abs(metered.energy.residual_fraction - 0.02) < 1e-12,
                      "interval energy reconciliation mismatch");

    const Json metered_json = Json::parse(format_throughput_json("serve-test", 5000, metered));
    // Each phase divides by its own denominator. Dividing prefill energy by every token, or decode
    // energy by prompt tokens, would produce a plausible number that means nothing.
    failures += check(
        metered_json.at("energy").at("board_joules") == 1000.0 &&
            metered_json.at("energy").at("idle_watts") == 95.0 &&
            std::abs(static_cast<double>(
                         metered_json.at("energy").at("joules_per_token").at("prefill")) -
                     0.2) < 1e-12 &&
            std::abs(static_cast<double>(
                         metered_json.at("energy").at("joules_per_token").at("decode")) -
                     4.0) < 1e-12 &&
            std::abs(static_cast<double>(
                         metered_json.at("energy").at("joules_per_token").at("served")) -
                     1000.0 / 1100.0) < 1e-12 &&
            std::abs(static_cast<double>(
                         metered_json.at("energy").at("joules_per_token").at("active")) -
                     620.0 / 1100.0) < 1e-12,
        "throughput energy json mismatch");

    // A phase that produced no tokens has no denominator. Null distinguishes that from free.
    ninfer::RuntimeStats decode_only_after = energy_after;
    decode_only_after.computed_prefill_tokens = 0;
    decode_only_after.prefill_energy_joules   = 0.0;
    const Json decode_only_json = Json::parse(format_throughput_json(
        "serve-test", 5000, make_throughput_report(energy_before, decode_only_after, 10.0, board)));
    failures +=
        check(decode_only_json.at("energy").at("joules_per_token").at("prefill").is_null(),
              "energy per token must be null for a phase that produced no tokens");

    // A driver reload restarts the board counter. The reporter drops that difference rather than
    // passing a negative or restarted value through, so an unavailable sample must stay absent.
    const ThroughputReport unmetered =
        make_throughput_report(energy_before, energy_after, 10.0, BoardEnergySample{});
    failures += check(!unmetered.energy.available && unmetered.energy.board_joules == 0.0,
                      "energy must stay absent without a usable board sample");

    ninfer::RuntimeStats continuation_before;
    ninfer::RuntimeStats continuation_after;
    continuation_after.continuation_publication_successes = 1;
    continuation_after.continuation_persistence_queued = 2;
    continuation_after.continuation_persistence_successes = 1;
    continuation_after.continuation_l3_persistence_microseconds = 900;
    continuation_after.continuation_l3_persistence_operations = 1;
    const ThroughputReport continuation_only =
        make_throughput_report(continuation_before, continuation_after, 2.0);
    failures += check(
        throughput_report_has_activity(continuation_only) &&
            continuation_only.computed_prefill_tokens == 0 &&
            continuation_only.committed_decode_tokens == 0 &&
            continuation_only.continuation_delta.continuation_publication_successes == 1 &&
            continuation_only.continuation_delta.continuation_persistence_queued == 2 &&
            continuation_only.continuation_delta.continuation_l3_persistence_microseconds == 900 &&
            continuation_only.continuation_delta.continuation_l3_persistence_operations == 1,
        "continuation-only reporter interval was suppressed or lost async deltas");

    ninfer::RuntimeStats before_restart;
    before_restart.computed_prefill_tokens = 100;
    before_restart.continuation_l2_restore_successes = 9;
    before_restart.continuation_miss_no_alias = 7;
    ninfer::RuntimeStats after_restart;
    after_restart.computed_prefill_tokens = 3;
    after_restart.continuation_l2_restore_successes = 2;
    after_restart.continuation_miss_no_alias = 1;
    const ThroughputReport restarted =
        make_throughput_report(before_restart, after_restart, 1.0);
    failures += check(restarted.computed_prefill_tokens == 3 &&
                          restarted.continuation_delta.continuation_l2_restore_successes == 2 &&
                          restarted.continuation_delta.continuation_miss_no_alias == 1,
                      "counter restart deltas use the current snapshot without underflow");

    const std::string console_prefix =
        format_console_log_prefix(std::chrono::system_clock::time_point{}, ConsoleLogLevel::Info);
    failures += check(console_prefix.starts_with('[') &&
                          console_prefix.ends_with("] [info] ninfer-serve: "),
                      "console log prefix mismatch");

    const std::filesystem::path log_path = std::filesystem::temp_directory_path() /
                                           ("ninfer-request-log-test-" +
#ifdef _WIN32
                                            std::to_string(static_cast<long long>(::_getpid())) +
#else
                                            std::to_string(static_cast<long long>(::getpid())) +
#endif
                                            ".jsonl");
    std::filesystem::remove(log_path);
    // One formatted record must reach both sinks. A live /events reader and a post-hoc reader of
    // the JSONL file are required to see byte-identical lines, which is what lets the dashboard
    // replay a file through the same code path it uses for the stream.
    std::string streamed_start;
    std::string streamed_error;
    {
        EventStream stream(log_path.string(), {}, 8);
        std::vector<std::string> backlog;
        std::shared_ptr<EventSubscriber> reader = stream.subscribe(backlog);
        failures += check(backlog.empty(), "a fresh stream replayed records that were never sent");
        stream.emit_request_start(context);
        failures += check(reader->next(streamed_start, std::chrono::milliseconds(1000)),
                          "subscriber did not receive the emitted request_start");

        // A second reader must be replayed what it missed, so a dashboard opened mid-run renders
        // without waiting for the next record.
        std::vector<std::string> late_backlog;
        std::shared_ptr<EventSubscriber> late = stream.subscribe(late_backlog);
        failures += check(late_backlog.size() == 1 && late_backlog.front() == streamed_start,
                          "late subscriber was not replayed the retained record");
        stream.unsubscribe(late);
        stream.unsubscribe(reader);
    }
    {
        EventStream stream(log_path.string(), {}, 8);
        std::vector<std::string> backlog;
        std::shared_ptr<EventSubscriber> reader = stream.subscribe(backlog);
        stream.emit_request_error(context, "generation failed");
        failures += check(reader->next(streamed_error, std::chrono::milliseconds(1000)),
                          "subscriber did not receive the emitted request_error");
        stream.unsubscribe(reader);
    }
    std::ifstream input(log_path);
    std::string first_line;
    std::string second_line;
    std::string extra_line;
    std::getline(input, first_line);
    std::getline(input, second_line);
    std::getline(input, extra_line);
    failures += check(!first_line.empty() && !second_line.empty() && extra_line.empty(),
                      "JSONL writer did not append exactly one flushed line per event");
    if (!first_line.empty() && !second_line.empty()) {
        failures += check(Json::parse(first_line).at("event") == "request_start",
                          "first appended event mismatch");
        failures += check(Json::parse(second_line).at("event") == "request_error",
                          "second appended event mismatch");
        failures += check(first_line == streamed_start && second_line == streamed_error,
                          "streamed records differ from the appended JSONL lines");
    }
    input.close();
    std::filesystem::remove(log_path);

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
