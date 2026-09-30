#include "serve/serve_options.h"
#include "product/prefix_checkpoint_options.h"
#include "product/speculative_options.h"

#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::serve {
namespace {

int parse_nonnegative_int(const char* text, const char* label) {
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<int>(value);
}

float parse_float_in(const char* text, const char* label, float lo, float hi) {
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !(value >= lo) || !(value <= hi)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<float>(value);
}

std::uint64_t parse_u64(const char* text, const char* label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

KvCacheStorage parse_kv_dtype(const char* text) {
    const std::string value(text);
    if (value == "bf16") { return KvCacheStorage::BFloat16; }
    if (value == "int8") { return KvCacheStorage::Int8Group64; }
    if (value == "rk8v4") { return KvCacheStorage::RotatedInt8KeyInt4ValueGroup64; }
    if (value == "rk4v4") { return KvCacheStorage::RotatedInt4KeyInt4ValueGroup64; }
    if (value == "rk4v4-e8") { return KvCacheStorage::RK4V4E8; }
    if (value == "rk2v4-e8") { return KvCacheStorage::RK2V4E8; }
    throw std::invalid_argument("invalid kv-dtype: " + value);
}

KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (std::string_view(text) == "auto") { return KvCapacityPolicy::automatic(); }
    const int value = parse_nonnegative_int(text, "kv-capacity");
    if (value == 0) { throw std::invalid_argument("--kv-capacity must be positive"); }
    return KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(value));
}

ContinuationCacheTiers parse_continuation_cache_tiers(std::string_view value) {
    if (value == "off") { return ContinuationCacheTiers::Off; }
    if (value == "l1") { return ContinuationCacheTiers::L1; }
    if (value == "l1-l2") { return ContinuationCacheTiers::L1L2; }
    if (value == "l1-l2-l3") { return ContinuationCacheTiers::L1L2L3; }
    throw std::invalid_argument("invalid continuation-cache: " + std::string(value));
}

ContinuationCachePolicy parse_continuation_cache_policy(std::string_view value) {
    if (value == "adaptive") { return ContinuationCachePolicy::Adaptive; }
    throw std::invalid_argument("invalid continuation-cache-policy: " + std::string(value));
}

std::size_t parse_cache_mib(const char* text, const char* label) {
    const std::uint64_t value = parse_u64(text, label);
    if (value == 0 || value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string("--") + label + " is out of range");
    }
    return static_cast<std::size_t>(value);
}

std::size_t parse_cache_reserve_mib(const char* text) {
    const std::uint64_t value = parse_u64(text, "continuation-cache-filesystem-reserve-mib");
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("--continuation-cache-filesystem-reserve-mib is out of range");
    }
    return static_cast<std::size_t>(value);
}

std::uint32_t parse_cache_u32(const char* text, const char* label) {
    const std::uint64_t value = parse_u64(text, label);
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(std::string("--") + label + " is out of range");
    }
    return static_cast<std::uint32_t>(value);
}

} // namespace

std::string serve_usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <model.ninfer> [--host H] [--port N] [--api-key KEY] "
           "[--model-id ID] [--max-context N] [--kv-capacity N|auto] [--max-concurrency N] "
           "[--max-pending-requests N] [--pending-timeout-ms N] "
           "[--prefill-chunk N] [--prefill-decode-balance F] [--turn-checkpoints N] "
           "[--log-stats-interval-ms N] [--device N] "
           "[--max-request-mib N] [--request-log-jsonl FILE] [--slot-save-path DIR] "
           "[--web-dir DIR] [--auto-save-evicted] "
           "[--response-store-max-records N] [--response-store-max-mib N] "
           "[--kv-dtype bf16|int8|rk8v4|rk4v4|rk4v4-e8|rk2v4-e8] [--spec mtp|dflash --draft-tokens "
           "N] "
           "[--default-max-tokens N] "
           "[--lora-dir DIR] [--lora-slots N] [--lora-rank N] [--systemone-default NAME] "
           "[--vision] [--vision-max-tokens N] [--no-cuda-graph] [--no-prefix-reuse] "
           "[--prefix-checkpoint-policy stable-turn|rolling-tool] "
           "[--continuation-cache off|l1|l1-l2|l1-l2-l3] "
           "[--continuation-cache-policy adaptive] [--continuation-cache-dir DIR] "
           "[--continuation-cache-namespace NAME] "
           "[--continuation-cache-l1-mib N] [--continuation-cache-l2-mib N] "
           "[--continuation-cache-l3-mib N] "
           "[--continuation-cache-l1-idle-seconds N] "
           "[--continuation-cache-l2-idle-seconds N] "
           "[--continuation-cache-l3-ttl-seconds N] "
           "[--continuation-cache-persist-interval-seconds N] "
           "[--continuation-cache-persist-min-tokens N] "
           "[--continuation-cache-filesystem-reserve-mib N] [--prefix-checkpoint-history N] "
           "[--lm-head-draft] [--no-thinking] [--preserve-thinking] [--cors] "
           "[--no-repetition-guard] [--repetition-guard-window N] "
           "[--repetition-guard-ngram N] [--repetition-guard-cycles N] "
           "[--repetition-guard-min-tokens N] "
           "[--temperature F] [--top-p F] [--top-k N] [--min-p F] [--presence-penalty F] "
           "[--frequency-penalty F] [--seed N] [--greedy]\n"
           "       serves OpenAI Responses/Chat Completions and Anthropic Messages endpoints\n"
           "       --default-max-tokens defaults to " +
           std::to_string(kDefaultMaxTokens) +
           " when omitted\n"
           "       --max-request-mib defaults to 384 and is enforced before JSON parsing\n"
           "       --request-log-jsonl appends full-precision server/request records; the same "
           "records stream live from GET /events regardless of this option\n"
           "       --web-dir serves the built dashboard (apps/web/dist) from / on this port, "
           "same-origin with /telemetry, /events, /metrics and /slots (disabled when omitted)\n"
           "       --slot-save-path enables llama.cpp-style session persistence: POST "
           "/slots/{id}?action=save|restore|erase with {\"filename\": NAME} moves one idle "
           "slot's resident session to or from DIR (disabled when omitted)\n"
           "       --turn-checkpoints retains N host turn checkpoints per slot so a prompt "
           "that diverges mid-history re-prefills from the nearest checkpoint instead of from "
           "zero (0 disables; each entry holds the full GDN state image in host memory)\n"
           "       --auto-save-evicted spills an involuntarily evicted session back to the "
           "slot file it was last saved to or restored from, before the eviction destroys it "
           "(requires --slot-save-path; explicit erase never auto-saves)\n"
           "       --model-id overrides the artifact identity.model_id reported by the server\n"
           "       Responses state is process-local and bounded to 1024 records / 256 MiB by "
           "default\n"
           "       --log-stats-interval-ms defaults to 5000; 0 disables periodic throughput logs\n"
           "       --vision enables media and loads the fixed Vision GPU allocations\n"
           "       --vision-max-tokens sets the Vision scratchpad token capacity (default 8192)\n"
           "       --lora-dir discovers every .ninfer adapter in DIR; each is served as model\n"
           "         id <model>-<name>, where name strips .ninfer and a trailing .lora from the\n"
           "         filename. --lora-slots sets how many stay device-resident\n"
           "         (default 2); the engine swaps the rest in on demand. A decision adapter is\n"
           "         instead a System One model under /systemone (POST /systemone/v1/systemone,\n"
           "         GET /systemone/v1/models), named by its filename\n"
           "       --systemone-default binds the System One SDK default model jev-latest to a\n"
           "         decision adapter; without it jev-latest answers with the only one\n"
           "       --kv-capacity auto leaves " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           " MiB of sizing headroom\n"
           "       --no-prefix-reuse disables compatible-prefix caching (enabled by default)\n"
           "       --prefix-checkpoint-policy defaults to rolling-tool\n"
           "       --continuation-cache defaults to l1-l2 without a directory and l1-l2-l3 "
           "with --continuation-cache-dir; l1-l2-l3 requires a nonempty directory\n"
           "       l1/l2/l3 capacities default to 768/16384/49152 MiB; policy defaults to "
           "adaptive; namespace defaults to local\n"
           "       L1/L2/L3 idle TTLs default to 600/7200/86400 seconds (0 means no expiry).\n"
           "       Persistence defaults to 60 seconds or 8192 tokens; interval 0 disables only "
           "the timer trigger; token growth and orderly shutdown still persist\n"
           "       Persist minimum 0 makes every publication due; filesystem reserve defaults to 0 "
           "MiB; prefix checkpoint history defaults "
           "to 4\n"
           "       --preserve-thinking retains closed-turn assistant reasoning in later prompts\n"
           "       sampler defaults come from the loaded model and resolved thinking mode; "
           "server flags and request fields override individual values.\n"
           "       --greedy forces temperature 0 (exact argmax).\n";
}

ServeOptions parse_serve_options(int argc, char** argv) {
    ServeOptions options;
    options.startup_argv.reserve(static_cast<std::size_t>(argc));
    bool redact_next = false;
    for (int i = 0; i < argc; ++i) {
        if (redact_next) {
            options.startup_argv.emplace_back("<redacted>");
            redact_next = false;
            continue;
        }
        options.startup_argv.emplace_back(argv[i] == nullptr ? "" : argv[i]);
        redact_next = options.startup_argv.back() == "--api-key";
    }
    bool default_max_tokens_explicit = false;
    bool kv_capacity_explicit        = false;
    bool continuation_tiers_explicit = false;
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument("artifact path is required"); }
    options.artifact_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg    = argv[i];
        const auto require_value = [&](const char* flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };
        if (arg == "--host") {
            options.host = require_value("--host");
        } else if (arg == "--port") {
            options.port = parse_nonnegative_int(require_value("--port"), "port");
        } else if (arg == "--api-key") {
            options.api_key = require_value("--api-key");
        } else if (arg == "--model-id") {
            options.model_id_override = require_value("--model-id");
            if (options.model_id_override->empty()) {
                throw std::invalid_argument("--model-id must not be empty");
            }
        } else if (arg == "--max-context") {
            options.max_context = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-context"), "max-context"));
        } else if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(require_value("--kv-capacity"));
            kv_capacity_explicit = true;
        } else if (arg == "--max-concurrency") {
            options.max_concurrency = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-concurrency"), "max-concurrency"));
        } else if (arg == "--max-pending-requests") {
            options.max_pending_requests = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--max-pending-requests"), "max-pending-requests"));
        } else if (arg == "--pending-timeout-ms") {
            options.pending_timeout_ms = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--pending-timeout-ms"), "pending-timeout-ms"));
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--prefill-chunk"), "prefill-chunk"));
        } else if (arg == "--prefill-decode-balance") {
            options.prefill_decode_balance = parse_float_in(
                require_value("--prefill-decode-balance"), "prefill-decode-balance", 0.0f, 8.0f);
        } else if (arg == "--turn-checkpoints") {
            options.turn_checkpoint_ring = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--turn-checkpoints"), "turn-checkpoints"));
        } else if (arg == "--auto-save-evicted") {
            options.auto_save_evicted = true;
        } else if (arg == "--log-stats-interval-ms") {
            options.log_stats_interval_ms = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--log-stats-interval-ms"), "log-stats-interval-ms"));
        } else if (arg == "--max-request-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--max-request-mib"), "max-request-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--max-request-mib is out of range");
            }
            options.max_request_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--request-log-jsonl") {
            options.request_log_jsonl = require_value("--request-log-jsonl");
            if (options.request_log_jsonl.empty()) {
                throw std::invalid_argument("--request-log-jsonl must not be empty");
            }
        } else if (arg == "--slot-save-path") {
            options.slot_save_path = require_value("--slot-save-path");
            if (options.slot_save_path.empty()) {
                throw std::invalid_argument("--slot-save-path must not be empty");
            }
        } else if (arg == "--web-dir") {
            options.web_dir = require_value("--web-dir");
            if (options.web_dir.empty()) {
                throw std::invalid_argument("--web-dir must not be empty");
            }
            if (!std::filesystem::is_directory(options.web_dir)) {
                throw std::invalid_argument("--web-dir is not a directory: " + options.web_dir);
            }
            if (!std::filesystem::is_regular_file(std::filesystem::path(options.web_dir) /
                                                  "index.html")) {
                throw std::invalid_argument("--web-dir has no index.html: " + options.web_dir);
            }
        } else if (arg == "--response-store-max-records") {
            const int records = parse_nonnegative_int(require_value("--response-store-max-records"),
                                                      "response-store-max-records");
            if (records == 0) {
                throw std::invalid_argument("--response-store-max-records must be positive");
            }
            options.response_store_max_records = static_cast<std::size_t>(records);
        } else if (arg == "--response-store-max-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--response-store-max-mib"), "response-store-max-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--response-store-max-mib is out of range");
            }
            options.response_store_max_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--device") {
            options.device = parse_nonnegative_int(require_value("--device"), "device");
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_dtype(require_value("--kv-dtype"));
        } else if (arg == "--spec") {
            options.speculative.backend =
                product::parse_speculative_backend(require_value("--spec"));
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--draft-tokens"), "draft-tokens"));
        } else if (arg == "--default-max-tokens") {
            options.default_max_tokens =
                parse_nonnegative_int(require_value("--default-max-tokens"), "default-max-tokens");
            default_max_tokens_explicit = true;
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--vision-max-tokens" || arg == "--vision-limit") {
            const int val = parse_nonnegative_int(require_value(arg.c_str()), "vision-max-tokens");
            if (val <= 0) {
                throw std::invalid_argument(std::string(arg) + " must be positive");
            }
            options.vision_max_tokens = static_cast<std::uint32_t>(val);
            options.enable_vision     = true;
        } else if (arg == "--lora-dir") {
            options.lora.directory = std::filesystem::path(require_value("--lora-dir"));
            if (options.lora.directory.empty()) {
                throw std::invalid_argument("--lora-dir must name a directory");
            }
        } else if (arg == "--systemone-default") {
            options.systemone_default = require_value("--systemone-default");
            if (options.systemone_default.empty()) {
                throw std::invalid_argument("--systemone-default must name a decision adapter");
            }
        } else if (arg == "--lora-slots") {
            const int val = parse_nonnegative_int(require_value("--lora-slots"), "lora-slots");
            if (val <= 0) { throw std::invalid_argument("--lora-slots must be positive"); }
            options.lora.slots = static_cast<std::uint32_t>(val);
        } else if (arg == "--lora-rank") {
            const int val = parse_nonnegative_int(require_value("--lora-rank"), "lora-rank");
            if (val <= 0) { throw std::invalid_argument("--lora-rank must be positive"); }
            options.lora.rank_ceiling = static_cast<std::int32_t>(val);
        } else if (arg == "--no-repetition-guard") {
            options.repetition_guard.enabled = false;
        } else if (arg == "--repetition-guard-window") {
            options.repetition_guard.window = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--repetition-guard-window"), "repetition-guard-window"));
        } else if (arg == "--repetition-guard-ngram") {
            options.repetition_guard.ngram = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--repetition-guard-ngram"), "repetition-guard-ngram"));
        } else if (arg == "--repetition-guard-cycles") {
            options.repetition_guard.cycles = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--repetition-guard-cycles"), "repetition-guard-cycles"));
        } else if (arg == "--repetition-guard-min-tokens") {
            options.repetition_guard.min_tokens = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--repetition-guard-min-tokens"), "repetition-guard-min-tokens"));
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--no-prefix-reuse") {
            options.allow_prefix_reuse = false;
        } else if (arg == "--prefix-checkpoint-policy") {
            options.prefix_checkpoint_policy = product::parse_prefix_checkpoint_policy(
                require_value("--prefix-checkpoint-policy"));
        } else if (arg == "--continuation-cache") {
            options.continuation_cache.tiers =
                parse_continuation_cache_tiers(require_value("--continuation-cache"));
            continuation_tiers_explicit = true;
        } else if (arg == "--continuation-cache-policy") {
            options.continuation_cache.policy =
                parse_continuation_cache_policy(require_value("--continuation-cache-policy"));
        } else if (arg == "--continuation-cache-dir") {
            options.continuation_cache.directory = require_value("--continuation-cache-dir");
            if (options.continuation_cache.directory.empty()) {
                throw std::invalid_argument("--continuation-cache-dir must not be empty");
            }
        } else if (arg == "--continuation-cache-namespace") {
            options.continuation_cache.cache_namespace =
                require_value("--continuation-cache-namespace");
            if (options.continuation_cache.cache_namespace.empty()) {
                throw std::invalid_argument("--continuation-cache-namespace must not be empty");
            }
        } else if (arg == "--continuation-cache-l1-mib") {
            options.continuation_cache.l1_capacity_mib = parse_cache_mib(
                require_value("--continuation-cache-l1-mib"), "continuation-cache-l1-mib");
        } else if (arg == "--continuation-cache-l2-mib") {
            options.continuation_cache.l2_capacity_mib = parse_cache_mib(
                require_value("--continuation-cache-l2-mib"), "continuation-cache-l2-mib");
        } else if (arg == "--continuation-cache-l3-mib") {
            options.continuation_cache.l3_capacity_mib = parse_cache_mib(
                require_value("--continuation-cache-l3-mib"), "continuation-cache-l3-mib");
        } else if (arg == "--continuation-cache-l1-idle-seconds") {
            options.continuation_cache.l1_idle_ttl_seconds = parse_cache_u32(
                require_value("--continuation-cache-l1-idle-seconds"),
                "continuation-cache-l1-idle-seconds");
        } else if (arg == "--continuation-cache-l2-idle-seconds") {
            options.continuation_cache.l2_idle_ttl_seconds = parse_cache_u32(
                require_value("--continuation-cache-l2-idle-seconds"),
                "continuation-cache-l2-idle-seconds");
        } else if (arg == "--continuation-cache-l3-ttl-seconds") {
            options.continuation_cache.l3_idle_ttl_seconds = parse_cache_u32(
                require_value("--continuation-cache-l3-ttl-seconds"),
                "continuation-cache-l3-ttl-seconds");
        } else if (arg == "--continuation-cache-persist-interval-seconds") {
            options.continuation_cache.persist_interval_seconds = parse_cache_u32(
                require_value("--continuation-cache-persist-interval-seconds"),
                "continuation-cache-persist-interval-seconds");
        } else if (arg == "--continuation-cache-persist-min-tokens") {
            options.continuation_cache.persist_min_tokens = parse_cache_u32(
                require_value("--continuation-cache-persist-min-tokens"),
                "continuation-cache-persist-min-tokens");
        } else if (arg == "--continuation-cache-filesystem-reserve-mib") {
            options.continuation_cache.filesystem_reserve_mib = parse_cache_reserve_mib(
                require_value("--continuation-cache-filesystem-reserve-mib"));
        } else if (arg == "--prefix-checkpoint-history") {
            options.continuation_cache.prefix_checkpoint_history = parse_cache_u32(
                require_value("--prefix-checkpoint-history"), "prefix-checkpoint-history");
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--preserve-thinking") {
            options.preserve_thinking = true;
        } else if (arg == "--cors") {
            options.enable_cors = true;
        } else if (arg == "--temperature") {
            options.sampling_overrides.temperature =
                parse_float_in(require_value("--temperature"), "temperature", 0.0f, 2.0f);
        } else if (arg == "--top-p") {
            options.sampling_overrides.top_p =
                parse_float_in(require_value("--top-p"), "top-p", 0.0f, 1.0f);
        } else if (arg == "--top-k") {
            options.sampling_overrides.top_k =
                parse_nonnegative_int(require_value("--top-k"), "top-k");
        } else if (arg == "--min-p") {
            options.sampling_overrides.min_p =
                parse_float_in(require_value("--min-p"), "min-p", 0.0f, 1.0f);
        } else if (arg == "--presence-penalty") {
            options.sampling_overrides.presence_penalty = parse_float_in(
                require_value("--presence-penalty"), "presence-penalty", -2.0f, 2.0f);
        } else if (arg == "--frequency-penalty") {
            options.sampling_overrides.frequency_penalty = parse_float_in(
                require_value("--frequency-penalty"), "frequency-penalty", -2.0f, 2.0f);
        } else if (arg == "--seed") {
            options.sampling_overrides.seed = parse_u64(require_value("--seed"), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (!kv_capacity_explicit) {
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(options.max_context);
    }
    if (!continuation_tiers_explicit && !options.continuation_cache.directory.empty()) {
        options.continuation_cache.tiers = ContinuationCacheTiers::L1L2L3;
    }
    // L1 and L2 are memory tiers; only L3 has a filesystem dependency.
    if (options.continuation_cache.tiers == ContinuationCacheTiers::L1L2L3 &&
        options.continuation_cache.directory.empty()) {
        throw std::invalid_argument("--continuation-cache-dir is required for l1-l2-l3");
    }
    if (options.auto_save_evicted && options.slot_save_path.empty()) {
        throw std::invalid_argument("--auto-save-evicted requires --slot-save-path");
    }
    if (options.port <= 0 || options.port > 65535) {
        throw std::invalid_argument("--port must be in [1,65535]");
    }
    if (options.max_context == 0) { throw std::invalid_argument("--max-context must be positive"); }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-context");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("--max-concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0) {
        throw std::invalid_argument("--max-pending-requests must be positive");
    }
    if (options.pending_timeout_ms == 0) {
        throw std::invalid_argument("--pending-timeout-ms must be positive");
    }
    if (options.max_request_bytes == 0) {
        throw std::invalid_argument("--max-request-mib must be positive");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a positive multiple of 128");
    }
    product::validate_speculative_cli_options(options.speculative);
    if (options.speculative.backend == SpeculativeBackend::DFlash && options.enable_vision) {
        throw std::invalid_argument("--spec dflash cannot be combined with --vision");
    }
    if (default_max_tokens_explicit) {
        if (options.default_max_tokens <= 0) {
            throw std::invalid_argument("--default-max-tokens must be positive");
        }
    }
    return options;
}

std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_id) {
    if (options.model_id_override.has_value()) { return *options.model_id_override; }
    if (artifact_model_id.empty()) {
        throw std::logic_error("loaded artifact model_id must not be empty");
    }
    return std::string(artifact_model_id);
}

} // namespace ninfer::serve
