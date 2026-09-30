#include "options.h"
#include "product/prefix_checkpoint_options.h"
#include "product/speculative_options.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace ninfer::cli {
namespace {

std::uint64_t parse_u64(const char* text, std::string_view label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument("invalid " + std::string(label) + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

std::uint32_t parse_u32(const char* text, std::string_view label, bool allow_zero = false) {
    const std::uint64_t value = parse_u64(text, label);
    if ((!allow_zero && value == 0) || value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<std::uint32_t>(value);
}

int parse_device(const char* text) {
    const std::uint64_t value = parse_u64(text, "device");
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid device: ") + text);
    }
    return static_cast<int>(value);
}

float parse_float(const char* text, std::string_view label, float minimum, float maximum) {
    errno              = 0;
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' || !std::isfinite(value) ||
        value < static_cast<double>(minimum) || value > static_cast<double>(maximum)) {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<float>(value);
}

KvCacheStorage parse_kv_cache(std::string_view text) {
    if (text == "bf16") { return KvCacheStorage::BFloat16; }
    if (text == "int8") { return KvCacheStorage::Int8Group64; }
    if (text == "rk8v4") { return KvCacheStorage::RotatedInt8KeyInt4ValueGroup64; }
    if (text == "rk4v4") { return KvCacheStorage::RotatedInt4KeyInt4ValueGroup64; }
    if (text == "rk4v4-e8") { return KvCacheStorage::RK4V4E8; }
    if (text == "rk2v4-e8") { return KvCacheStorage::RK2V4E8; }
    throw std::invalid_argument("invalid kv-dtype: " + std::string(text));
}

KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (std::string_view(text) == "auto") { return KvCapacityPolicy::automatic(); }
    return KvCapacityPolicy::explicit_capacity(parse_u32(text, "kv-capacity"));
}

ReasoningEffort parse_reasoning_effort(std::string_view text) {
    if (text == "low") { return ReasoningEffort::Low; }
    if (text == "medium") { return ReasoningEffort::Medium; }
    if (text == "xhigh") { return ReasoningEffort::XHigh; }
    throw std::invalid_argument("invalid reasoning-effort: " + std::string(text));
}

ContinuationCacheTiers parse_continuation_cache_tiers(std::string_view text) {
    if (text == "off") { return ContinuationCacheTiers::Off; }
    if (text == "l1") { return ContinuationCacheTiers::L1; }
    if (text == "l1-l2") { return ContinuationCacheTiers::L1L2; }
    if (text == "l1-l2-l3") { return ContinuationCacheTiers::L1L2L3; }
    throw std::invalid_argument("invalid continuation-cache: " + std::string(text));
}

ContinuationCachePolicy parse_continuation_cache_policy(std::string_view text) {
    if (text == "adaptive") { return ContinuationCachePolicy::Adaptive; }
    throw std::invalid_argument("invalid continuation-cache-policy: " + std::string(text));
}

std::size_t parse_cache_mib(const char* text, std::string_view label) {
    const std::uint64_t value = parse_u64(text, label);
    if (value == 0 || value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<std::size_t>(value);
}

std::size_t parse_cache_reserve_mib(const char* text) {
    const std::uint64_t value = parse_u64(text, "continuation-cache-filesystem-reserve-mib");
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("invalid continuation-cache-filesystem-reserve-mib: " +
                                    std::string(text));
    }
    return static_cast<std::size_t>(value);
}

} // namespace

std::string usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <model.ninfer> (--prompt <text>|--messages <messages.json>|--systemone <file|->)\n"
           "[--systemone-jsonl] [--systemone-default NAME] [--systemone-cold]\n"
           "[--systemone-probabilities] [--systemone-dump-prepared FILE]\n"
           "[--max-context N] [--kv-capacity N|auto] [--prefill-chunk N] [--max-new N]\n"
           "[--device N]\n"
           "[--kv-dtype bf16|int8|rk8v4|rk4v4|rk4v4-e8|rk2v4-e8] [--spec mtp|dflash --draft-tokens "
           "N]\n"
           "       [--lm-head-draft]\n"
           "       [--temperature F] [--top-p F] [--top-k N] [--min-p F]\n"
           "       [--presence-penalty F] [--frequency-penalty F] [--seed N] [--greedy]\n"
           "       [--stop-token-id N]... [--stop <text>]... [--reasoning-stop <text>]...\n"
           "       [--raw-output] [--print-token-ids] [--no-thinking]\n"
           "       [--reasoning-effort low|medium|xhigh] [--vision] [--vision-max-tokens N]\n"
           "       [--lora-dir DIR] [--lora-slots N] [--lora-rank N] [--adapter NAME]\n"
           "       [--no-cuda-graph] [--prefix-checkpoint-policy stable-turn|rolling-tool]\n"
           "       [--no-repetition-guard] [--repetition-guard-window N]\n"
           "       [--repetition-guard-ngram N] [--repetition-guard-cycles N]\n"
           "       [--repetition-guard-min-tokens N]\n"
           "       [--continuation-cache off|l1|l1-l2|l1-l2-l3] "
           "[--continuation-cache-policy adaptive]\n"
           "       [--continuation-cache-dir DIR] [--continuation-cache-namespace NAME]\n"
           "       [--continuation-cache-l1-mib N] [--continuation-cache-l2-mib N] "
           "[--continuation-cache-l3-mib N]\n"
           "       [--continuation-cache-l1-idle-seconds N] "
           "[--continuation-cache-l2-idle-seconds N] "
           "[--continuation-cache-l3-ttl-seconds N]\n"
           "       [--continuation-cache-persist-interval-seconds N] "
           "[--continuation-cache-persist-min-tokens N] "
           "[--continuation-cache-filesystem-reserve-mib N] [--prefix-checkpoint-history N]\n"
           "\n"
           "Streams answer content to stdout and reasoning plus diagnostics to stderr.\n"
           "--systemone answers TypeSafe System One request bodies (one per line with\n"
           "--systemone-jsonl) with the decision adapter each `model` names; jev-latest binds\n"
           "--systemone-default, else the pool's only decision adapter. Each response body is\n"
           "printed on its own stdout line, an error as {\"detail\": ...}.\n"
           "Structured message content accepts text, image/image_url, and video/video_url parts;\n"
           "media sources may be local paths, HTTP(S) URLs, or base64 data URIs.\n"
           "--vision enables image/video input and loads the fixed Vision GPU allocations.\n"
           "--vision-max-tokens sets the Vision scratchpad token capacity (default 8192).\n"
           "--kv-capacity auto leaves " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           " MiB of sizing headroom.\n"
           "--prefix-checkpoint-policy defaults to rolling-tool.\n"
           "Continuation cache defaults to off for one-shot inference. Explicit cache tuning "
           "enables l1-l2, or l1-l2-l3 when --continuation-cache-dir is set.\n"
           "L1/L2/L3 capacities default to 768/16384/49152 MiB; adaptive policy and namespace "
           "local are defaults.\n"
           "L1/L2/L3 idle TTLs default to 600/7200/86400 seconds (0 means no expiry).\n"
           "Persistence defaults to 60 seconds or 8192 tokens; interval 0 disables only the "
           "timer trigger; token growth and orderly shutdown still persist.\n"
           "Persist minimum 0 makes every publication due; filesystem reserve defaults to 0 "
           "MiB; checkpoint history defaults to 4.\n"
           "Sampling defaults come from the loaded model and thinking mode; flags override "
           "individual fields.\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    if (argc >= 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument(".ninfer model path is required"); }
    options.artifact_path     = argv[1];
    bool kv_capacity_explicit = false;
    bool continuation_tiers_explicit = false;
    bool continuation_config_explicit = false;

    for (int i = 2; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg.starts_with("--continuation-cache-") || arg == "--prefix-checkpoint-history") {
            continuation_config_explicit = true;
        }
        const auto value = [&](std::string_view flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };

        if (arg == "--prompt") {
            options.prompt = value(arg);
        } else if (arg == "--messages") {
            options.messages_path = value(arg);
        } else if (arg == "--systemone") {
            options.systemone_path = value(arg);
            if (options.systemone_path.empty()) {
                throw std::invalid_argument("--systemone must name a file or -");
            }
        } else if (arg == "--systemone-jsonl") {
            options.systemone_jsonl = true;
        } else if (arg == "--systemone-default") {
            options.systemone_default = value(arg);
            if (options.systemone_default.empty()) {
                throw std::invalid_argument("--systemone-default must name a decision adapter");
            }
        } else if (arg == "--systemone-cold") {
            options.systemone_cold = true;
        } else if (arg == "--systemone-probabilities") {
            options.systemone_probabilities = true;
        } else if (arg == "--systemone-dump-prepared") {
            options.systemone_dump_prepared = value(arg);
            if (options.systemone_dump_prepared.empty()) {
                throw std::invalid_argument("--systemone-dump-prepared must name a file");
            }
        } else if (arg == "--max-new") {
            options.max_new = parse_u32(value(arg), "max-new");
        } else if (arg == "--max-context") {
            options.max_context = parse_u32(value(arg), "max-context");
        } else if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(value(arg));
            kv_capacity_explicit = true;
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = parse_u32(value(arg), "prefill-chunk");
        } else if (arg == "--device") {
            options.device = parse_device(value(arg));
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_cache(value(arg));
        } else if (arg == "--spec") {
            options.speculative.backend = product::parse_speculative_backend(value(arg));
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = parse_u32(value(arg), "draft-tokens");
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--raw-output") {
            options.raw_output = true;
        } else if (arg == "--print-token-ids") {
            options.print_token_ids = true;
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--reasoning-effort") {
            options.reasoning_effort = parse_reasoning_effort(value(arg));
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--vision-max-tokens" || arg == "--vision-limit") {
            options.vision_max_tokens = parse_u32(value(arg), "vision-max-tokens", false);
            options.enable_vision     = true;
        } else if (arg == "--lora-dir") {
            options.lora.directory = std::filesystem::path(value(arg));
            if (options.lora.directory.empty()) {
                throw std::invalid_argument("--lora-dir must name a directory");
            }
        } else if (arg == "--lora-slots") {
            options.lora.slots = parse_u32(value(arg), "lora-slots", false);
        } else if (arg == "--lora-rank") {
            options.lora.rank_ceiling =
                static_cast<std::int32_t>(parse_u32(value(arg), "lora-rank", false));
        } else if (arg == "--adapter") {
            options.adapter = value(arg);
            if (options.adapter.empty()) {
                throw std::invalid_argument("--adapter must name a pool adapter");
            }
        } else if (arg == "--no-repetition-guard") {
            options.repetition_guard.enabled = false;
        } else if (arg == "--repetition-guard-window") {
            options.repetition_guard.window =
                parse_u32(value(arg), "repetition-guard-window", false);
        } else if (arg == "--repetition-guard-ngram") {
            options.repetition_guard.ngram = parse_u32(value(arg), "repetition-guard-ngram", false);
        } else if (arg == "--repetition-guard-cycles") {
            options.repetition_guard.cycles =
                parse_u32(value(arg), "repetition-guard-cycles", false);
        } else if (arg == "--repetition-guard-min-tokens") {
            options.repetition_guard.min_tokens =
                parse_u32(value(arg), "repetition-guard-min-tokens", true);
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--prefix-checkpoint-policy") {
            options.prefix_checkpoint_policy = product::parse_prefix_checkpoint_policy(value(arg));
        } else if (arg == "--continuation-cache") {
            options.continuation_cache.tiers = parse_continuation_cache_tiers(value(arg));
            continuation_tiers_explicit = true;
        } else if (arg == "--continuation-cache-policy") {
            options.continuation_cache.policy = parse_continuation_cache_policy(value(arg));
        } else if (arg == "--continuation-cache-dir") {
            options.continuation_cache.directory = value(arg);
            if (options.continuation_cache.directory.empty()) {
                throw std::invalid_argument("--continuation-cache-dir must not be empty");
            }
        } else if (arg == "--continuation-cache-namespace") {
            options.continuation_cache.cache_namespace = value(arg);
            if (options.continuation_cache.cache_namespace.empty()) {
                throw std::invalid_argument("--continuation-cache-namespace must not be empty");
            }
        } else if (arg == "--continuation-cache-l1-mib") {
            options.continuation_cache.l1_capacity_mib =
                parse_cache_mib(value(arg), "continuation-cache-l1-mib");
        } else if (arg == "--continuation-cache-l2-mib") {
            options.continuation_cache.l2_capacity_mib =
                parse_cache_mib(value(arg), "continuation-cache-l2-mib");
        } else if (arg == "--continuation-cache-l3-mib") {
            options.continuation_cache.l3_capacity_mib =
                parse_cache_mib(value(arg), "continuation-cache-l3-mib");
        } else if (arg == "--continuation-cache-l1-idle-seconds") {
            options.continuation_cache.l1_idle_ttl_seconds =
                parse_u32(value(arg), "continuation-cache-l1-idle-seconds", true);
        } else if (arg == "--continuation-cache-l2-idle-seconds") {
            options.continuation_cache.l2_idle_ttl_seconds =
                parse_u32(value(arg), "continuation-cache-l2-idle-seconds", true);
        } else if (arg == "--continuation-cache-l3-ttl-seconds") {
            options.continuation_cache.l3_idle_ttl_seconds =
                parse_u32(value(arg), "continuation-cache-l3-ttl-seconds", true);
        } else if (arg == "--continuation-cache-persist-interval-seconds") {
            options.continuation_cache.persist_interval_seconds =
                parse_u32(value(arg), "continuation-cache-persist-interval-seconds", true);
        } else if (arg == "--continuation-cache-persist-min-tokens") {
            options.continuation_cache.persist_min_tokens =
                parse_u32(value(arg), "continuation-cache-persist-min-tokens", true);
        } else if (arg == "--continuation-cache-filesystem-reserve-mib") {
            options.continuation_cache.filesystem_reserve_mib = parse_cache_reserve_mib(value(arg));
        } else if (arg == "--prefix-checkpoint-history") {
            options.continuation_cache.prefix_checkpoint_history =
                parse_u32(value(arg), "prefix-checkpoint-history", true);
        } else if (arg == "--stop-token-id") {
            const std::uint32_t token = parse_u32(value(arg), "stop-token-id", true);
            if (token > static_cast<std::uint32_t>(std::numeric_limits<TokenId>::max())) {
                throw std::invalid_argument("--stop-token-id exceeds the token domain");
            }
            options.stop_token_ids.push_back(static_cast<TokenId>(token));
        } else if (arg == "--stop" || arg == "--reasoning-stop") {
            std::string text = value(arg);
            if (text.empty()) {
                throw std::invalid_argument(std::string(arg) + " must not be empty");
            }
            options.stop_strings.push_back(StopString{
                .text    = std::move(text),
                .channel = arg == "--stop" ? OutputChannel::Content : OutputChannel::Reasoning,
            });
        } else if (arg == "--temperature") {
            options.sampling.temperature = parse_float(value(arg), "temperature", 0.0F, 2.0F);
        } else if (arg == "--top-p") {
            options.sampling.top_p = parse_float(value(arg), "top-p", 0.0F, 1.0F);
        } else if (arg == "--top-k") {
            const std::uint32_t top_k = parse_u32(value(arg), "top-k", true);
            if (top_k > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
                throw std::invalid_argument("--top-k exceeds INT32_MAX");
            }
            options.sampling.top_k = static_cast<std::int32_t>(top_k);
        } else if (arg == "--min-p") {
            options.sampling.min_p = parse_float(value(arg), "min-p", 0.0F, 1.0F);
        } else if (arg == "--presence-penalty") {
            options.sampling.presence_penalty =
                parse_float(value(arg), "presence-penalty", -2.0F, 2.0F);
        } else if (arg == "--frequency-penalty") {
            options.sampling.frequency_penalty =
                parse_float(value(arg), "frequency-penalty", -2.0F, 2.0F);
        } else if (arg == "--seed") {
            options.sampling.seed = parse_u64(value(arg), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }

    if (!kv_capacity_explicit) {
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(options.max_context);
    }
    if (!continuation_tiers_explicit) {
        if (!options.continuation_cache.directory.empty()) {
            options.continuation_cache.tiers = ContinuationCacheTiers::L1L2L3;
        } else if (continuation_config_explicit) {
            options.continuation_cache.tiers = ContinuationCacheTiers::L1L2;
        }
    }
    // L1 and L2 are memory tiers; only L3 has a filesystem dependency.
    if (options.continuation_cache.tiers == ContinuationCacheTiers::L1L2L3 &&
        options.continuation_cache.directory.empty()) {
        throw std::invalid_argument("--continuation-cache-dir is required for l1-l2-l3");
    }

    const bool has_prompt    = !options.prompt.empty();
    const bool has_messages  = !options.messages_path.empty();
    const bool has_systemone = !options.systemone_path.empty();
    if (static_cast<int>(has_prompt) + static_cast<int>(has_messages) +
            static_cast<int>(has_systemone) !=
        1) {
        throw std::invalid_argument("pass exactly one of --prompt, --messages or --systemone");
    }
    if (!has_systemone &&
        (options.systemone_jsonl || !options.systemone_default.empty() ||
         options.systemone_cold || options.systemone_probabilities ||
         !options.systemone_dump_prepared.empty())) {
        throw std::invalid_argument("--systemone-* options need --systemone");
    }
    if (options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a multiple of 128");
    }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-context");
    }
    product::validate_speculative_cli_options(options.speculative);
    if (options.speculative.backend == SpeculativeBackend::DFlash && options.enable_vision) {
        throw std::invalid_argument("--spec dflash cannot be combined with --vision");
    }
    if (!options.enable_thinking && options.reasoning_effort) {
        throw std::invalid_argument("--reasoning-effort cannot be combined with --no-thinking");
    }
    if (options.greedy) { options.sampling.temperature = 0.0F; }
    return options;
}

} // namespace ninfer::cli
