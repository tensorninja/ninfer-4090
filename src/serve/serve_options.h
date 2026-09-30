#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// Protocol default when the client omits max_tokens. Engine independently
// clamps the request to its effective context capacity.
inline constexpr int kDefaultMaxTokens                    = 8192;
inline constexpr std::size_t kDefaultMaxRequestBytes      = 384ULL << 20;
inline constexpr std::size_t kDefaultResponseStoreRecords = 1024;
inline constexpr std::size_t kDefaultResponseStoreBytes   = 256ULL << 20;

struct ServeOptions {
    bool help_requested = false;
    std::string artifact_path;
    std::string host = "127.0.0.1";
    int port         = 8080;
    std::string api_key;                          // empty => no auth
    std::optional<std::string> model_id_override; // unset => artifact identity.model_id
    std::string request_log_jsonl;                // empty => structured request logging disabled
    std::string slot_save_path;                   // empty => /slots save/restore/erase disabled
    std::string web_dir; // empty => the built dashboard is not served from this process
    std::uint32_t max_context              = 8192;
    KvCapacityPolicy kv_capacity           = KvCapacityPolicy::explicit_capacity(8192);
    std::uint32_t max_concurrency          = 1;
    std::uint32_t max_pending_requests     = 16;
    std::uint32_t pending_timeout_ms       = 30000;
    std::uint32_t prefill_chunk            = 1024;
    double prefill_decode_balance          = 0.0;
    std::uint32_t turn_checkpoint_ring     = 0; // 0 => host turn-checkpoint ring disabled
    bool auto_save_evicted                 = false; // spill evicted sessions to their slot file
    std::uint32_t log_stats_interval_ms    = 5000; // 0 disables periodic Engine throughput logs
    std::size_t max_request_bytes          = kDefaultMaxRequestBytes;
    std::size_t response_store_max_records = kDefaultResponseStoreRecords;
    std::size_t response_store_max_bytes   = kDefaultResponseStoreBytes;
    int device                             = 0;
    KvCacheStorage kv_cache                = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    RepetitionGuardOptions repetition_guard;
    PrefixCheckpointPolicy prefix_checkpoint_policy = PrefixCheckpointPolicy::RollingTool;
    ContinuationCacheOptions continuation_cache;
    std::uint32_t vision_max_tokens = 8192;
    bool enable_vision      = false;
    bool use_cuda_graph     = true;
    // Startup-discovered LoRA pool and its bounded device residency. Each filename-derived name
    // is served as an additional model id `<public model id>-<name>`.
    LoraOptions lora;
    // The decision adapter the System One SDK-default model name (`jev-latest`) answers with.
    // Empty binds the pool's only decision adapter, or leaves the alias unbound when there are
    // several.
    std::string systemone_default;
    bool allow_prefix_reuse = true;
    bool enable_thinking =
        true; // default thinking mode for the generation prompt (--no-thinking opts out)
    bool preserve_thinking = false;
    int default_max_tokens = kDefaultMaxTokens;
    bool enable_cors       = false; // send permissive CORS headers for browser UIs
    // Process-level explicit overrides layered between registered model/mode defaults and request
    // fields. An omitted seed is replaced per request with a fresh random seed.
    SamplingOverrides sampling_overrides;
    bool greedy = false; // --greedy: force temperature 0 (exact argmax)

    // Exact process argv for the server-start record. Secret-bearing option values are redacted
    // while parsing; this is provenance only and never affects execution.
    std::vector<std::string> startup_argv;
};

ServeOptions parse_serve_options(int argc, char** argv);
std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_id);
std::string serve_usage_text(const char* argv0);

} // namespace ninfer::serve
