#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::cli {

struct Options {
    bool help_requested = false;

    std::filesystem::path artifact_path;
    std::string prompt;
    std::filesystem::path messages_path;

    // System One decision mode: request bodies from a file ("-" reads stdin), one body or one per
    // line, answered with the adapter their `model` names (the alias binds `systemone_default`).
    std::filesystem::path systemone_path;
    bool systemone_jsonl = false;
    std::string systemone_default;
    // Compute every state cold instead of reusing a retained one.
    bool systemone_cold = false;
    // Print {"response", "probabilities", ...} per request instead of the bare response body.
    bool systemone_probabilities = false;
    // Write each prepared layout as a `ninfer-decision-prepared` JSON line for parity tools.
    std::filesystem::path systemone_dump_prepared;

    std::uint32_t max_new        = 128;
    std::uint32_t max_context    = 2048;
    KvCapacityPolicy kv_capacity = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t prefill_chunk  = 1024;
    int device                   = 0;

    KvCacheStorage kv_cache = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    RepetitionGuardOptions repetition_guard;
    PrefixCheckpointPolicy prefix_checkpoint_policy = PrefixCheckpointPolicy::RollingTool;
    ContinuationCacheOptions continuation_cache{.tiers = ContinuationCacheTiers::Off};
    std::uint32_t vision_max_tokens = 8192;
    bool enable_vision  = false;
    bool use_cuda_graph = true;
    // LoRA adapter pool discovered from a directory, and how many are device-resident at once.
    LoraOptions lora;
    // Adapter selected for this run; empty selects the base weights.
    std::string adapter;

    bool raw_output      = false;
    bool print_token_ids = false;
    bool enable_thinking = true;
    std::optional<ReasoningEffort> reasoning_effort;

    std::vector<TokenId> stop_token_ids;
    std::vector<StopString> stop_strings;

    // Omitted fields are resolved from the loaded model and rendered prompt mode by Engine.
    SamplingOverrides sampling;
    bool greedy = false;
};

[[nodiscard]] Options parse_options(int argc, char** argv);
[[nodiscard]] std::string usage_text(const char* argv0);

} // namespace ninfer::cli
