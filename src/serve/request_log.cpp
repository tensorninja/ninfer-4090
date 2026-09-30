#include "serve/request_log.h"
#include "product/prefix_checkpoint_options.h"
#include "product/speculative_options.h"
#include "serve/console_log.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#ifdef _WIN32
#    include <process.h>
#else
#    include <unistd.h>
#endif

namespace ninfer::serve {

std::uint64_t unix_time_ms() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

std::string new_server_instance_id() {
    const auto now    = std::chrono::system_clock::now().time_since_epoch();
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
#ifdef _WIN32
    const auto process_id = ::_getpid();
#else
    const auto process_id = ::getpid();
#endif
    return "serve-" + std::to_string(static_cast<long long>(process_id)) + '-' +
           std::to_string(micros);
}

const char* kv_cache_name(ninfer::KvCacheStorage storage) noexcept {
    if (storage == ninfer::KvCacheStorage::BFloat16) { return "bf16"; }
    if (storage == ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64) { return "rk8v4"; }
    if (storage == ninfer::KvCacheStorage::RotatedInt4KeyInt4ValueGroup64) { return "rk4v4"; }
    if (storage == ninfer::KvCacheStorage::RK2V4E8) { return "rk2v4-e8"; }
    return "int8-group64";
}

namespace {

using Json = nlohmann::json;

// Stable identity of a client routing key without reproducing the key itself, which is
// client-authored text. FNV-1a 64 as 16 hex characters, the same encoding as a session digest.
std::string routing_hint_digest(std::string_view value) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const char byte : value) {
        hash ^= static_cast<std::uint8_t>(byte);
        hash *= 1099511628211ULL;
    }
    std::string out(16, '0');
    for (int index = 15; index >= 0; --index) {
        out[static_cast<std::size_t>(index)] = "0123456789abcdef"[hash & 0xFULL];
        hash >>= 4;
    }
    return out;
}

std::filesystem::path normalized_absolute_path(const std::string& value) {
    std::error_code error;
    std::filesystem::path path = std::filesystem::weakly_canonical(value, error);
    if (!error) { return path; }
    error.clear();
    path = std::filesystem::absolute(value, error);
    return error ? std::filesystem::path(value).lexically_normal() : path.lexically_normal();
}

std::string cuda_version_string(int version) {
    if (version <= 0) { return {}; }
    return std::to_string(version / 1000) + '.' + std::to_string((version % 1000) / 10);
}

std::string cuda_uuid_string(const cudaUUID_t& uuid) {
    std::ostringstream out;
    out << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) { out << '-'; }
        out << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(uuid.bytes[i]));
    }
    return out.str();
}

const char* finish_reason_name(ninfer::FinishReason reason) {
    switch (reason) {
    case ninfer::FinishReason::None:
        return "none";
    case ninfer::FinishReason::OutputLimit:
        return "output_limit";
    case ninfer::FinishReason::ContextCapacity:
        return "context_capacity";
    case ninfer::FinishReason::StopToken:
        return "stop_token";
    case ninfer::FinishReason::StopString:
        return "stop_string";
    case ninfer::FinishReason::Cancelled:
        return "cancelled";
    case ninfer::FinishReason::RepetitionCycle:
        return "repetition_cycle";
    }
    return "unknown";
}

// A completed call whose arguments are not a JSON object cannot be replayed as history, which
// requires an object; an incomplete call holds only the prefix the output reached.
struct ToolCallCounts {
    std::size_t incomplete        = 0;
    std::size_t invalid_arguments = 0;
};

ToolCallCounts count_tool_calls(const std::vector<ToolCall>& calls) {
    ToolCallCounts counts;
    for (const ToolCall& call : calls) {
        if (call.state == ToolCallState::Incomplete) {
            ++counts.incomplete;
            continue;
        }
        const Json arguments = Json::parse(call.arguments_json, nullptr, false);
        if (arguments.is_discarded() || !arguments.is_object()) { ++counts.invalid_arguments; }
    }
    return counts;
}

const char* continuation_cache_tiers_name(ContinuationCacheTiers tiers) {
    switch (tiers) {
    case ContinuationCacheTiers::Off:
        return "off";
    case ContinuationCacheTiers::L1:
        return "l1";
    case ContinuationCacheTiers::L1L2:
        return "l1-l2";
    case ContinuationCacheTiers::L1L2L3:
        return "l1-l2-l3";
    }
    return "unknown";
}

const char* continuation_cache_policy_name(ContinuationCachePolicy policy) {
    switch (policy) {
    case ContinuationCachePolicy::Adaptive:
        return "adaptive";
    }
    return "unknown";
}

std::string tool_choice_name(const ToolChoice& choice) {
    switch (choice.mode) {
    case ToolChoiceMode::Auto:
        return "auto";
    case ToolChoiceMode::None:
        return "none";
    case ToolChoiceMode::Required:
        return "required";
    case ToolChoiceMode::Named:
        return choice.name.empty() ? "named" : choice.name;
    }
    return "unknown";
}

const char* kv_capacity_mode_name(ninfer::KvCapacityMode mode) {
    return mode == ninfer::KvCapacityMode::Automatic ? "auto" : "explicit";
}

const char* proposal_head_name(ninfer::ProposalHead proposal) {
    return proposal == ninfer::ProposalHead::Optimized ? "optimized" : "full";
}

const char* prefix_reuse_path_name(ninfer::PrefixReusePath path) {
    switch (path) {
    case ninfer::PrefixReusePath::FullReset:
        return "full_reset";
    case ninfer::PrefixReusePath::AppendAtFrontier:
        return "append_frontier";
    case ninfer::PrefixReusePath::RestoreTurnCheckpoint:
        return "restore_turn_checkpoint";
    case ninfer::PrefixReusePath::RestoreUserTurnAnchor:
        return "restore_user_turn_anchor";
    }
    return "unknown";
}

Json event_base(const std::string& server_instance_id, std::uint64_t timestamp, const char* event) {
    return Json{{"artifact_type", kRequestLogArtifactType},
                {"schema_version", kRequestLogSchemaVersion},
                {"event", event},
                {"timestamp_unix_ms", timestamp},
                {"server_instance_id", server_instance_id}};
}

Json sampler_json(const ninfer::ResolvedSamplingParameters& sampling) {
    return Json{{"temperature", sampling.temperature},
                {"top_p", sampling.top_p},
                {"top_k", sampling.top_k},
                {"min_p", sampling.min_p},
                {"presence_penalty", sampling.presence_penalty},
                {"frequency_penalty", sampling.frequency_penalty},
                {"seed", sampling.seed}};
}

Json preset_json(const ninfer::SamplingPreset& preset) {
    return Json{{"temperature", preset.temperature},
                {"top_p", preset.top_p},
                {"top_k", preset.top_k},
                {"min_p", preset.min_p},
                {"presence_penalty", preset.presence_penalty},
                {"frequency_penalty", preset.frequency_penalty}};
}

Json overrides_json(const ninfer::SamplingOverrides& overrides) {
    Json result{{"temperature", nullptr},
                {"top_p", nullptr},
                {"top_k", nullptr},
                {"min_p", nullptr},
                {"presence_penalty", nullptr},
                {"frequency_penalty", nullptr},
                {"seed", nullptr}};
    if (overrides.temperature) { result["temperature"] = *overrides.temperature; }
    if (overrides.top_p) { result["top_p"] = *overrides.top_p; }
    if (overrides.top_k) { result["top_k"] = *overrides.top_k; }
    if (overrides.min_p) { result["min_p"] = *overrides.min_p; }
    if (overrides.presence_penalty) { result["presence_penalty"] = *overrides.presence_penalty; }
    if (overrides.frequency_penalty) { result["frequency_penalty"] = *overrides.frequency_penalty; }
    if (overrides.seed) { result["seed"] = *overrides.seed; }
    return result;
}

Json request_json(const RequestLogContext& context) {
    return Json{{"request_id", context.id},
                {"x_request_id", context.x_request_id},
                {"protocol", context.protocol},
                {"model", context.model},
                {"stream", context.stream},
                {"message_count", context.message_count},
                {"requested_output_tokens", context.requested_output_tokens},
                {"requested_output_tokens_source",
                 context.requested_output_tokens_client_set ? "client" : "server_default"},
                {"tool_count", context.tool_count},
                {"tool_choice", tool_choice_name(context.tool_choice)},
                {"has_tool_history", context.has_tool_history},
                {"enable_thinking", context.enable_thinking},
                {"preserve_thinking", context.preserve_thinking},
                {"preserve_thinking_semantic_change", context.preserve_thinking_semantic_change},
                {"adapter", context.adapter},
                {"prompt_cache_key_digest", context.prompt_cache_key_digest},
                {"sampling", sampler_json(context.sampling)}};
}

Json decision_request_json(const DecisionLogContext& context) {
    const ninfer::DecisionSummary& summary = context.summary;
    return Json{{"request_id", context.id},
                {"x_request_id", context.x_request_id},
                {"protocol", "systemone"},
                {"model", context.model},
                {"adapter", context.adapter},
                {"allow_prefix_reuse", context.allow_prefix_reuse},
                {"questions", summary.questions},
                {"options", summary.options},
                {"state_tokens", summary.state_tokens},
                {"state_truncated", summary.state_truncated},
                {"branch_tokens", summary.branch_tokens},
                {"longest_branch", summary.longest_branch},
                {"input_tokens", summary.input_tokens()}};
}

// Tokens per second, or null when either side is zero: a phase that ran no tokens, or none
// measurably, has no rate rather than a zero one.
Json tokens_per_second(std::uint64_t tokens, double seconds) {
    if (tokens == 0 || !(seconds > 0.0)) { return Json(nullptr); }
    return static_cast<double>(tokens) / seconds;
}

Json arena_json(const ninfer::ArenaMemorySummary& arena) {
    return Json{{"capacity_bytes", arena.capacity_bytes},
                {"used_bytes", arena.used_bytes},
                {"peak_used_bytes", arena.peak_used_bytes}};
}

Json speculative_json(const GenerationMetrics& metrics) {
    return Json{{"backend", product::speculative_backend_name(metrics.speculative_backend)},
                {"draft_window", metrics.speculative_draft_window},
                {"rounds", metrics.speculative_rounds},
                {"drafted_tokens", metrics.speculative_draft_tokens},
                {"accepted_tokens", metrics.speculative_accepted_tokens},
                {"fallback_steps", metrics.speculative_fallback_steps},
                {"accepted_per_position", metrics.speculative_accepted_per_position}};
}

Json continuation_json(const GenerationMetrics& metrics) {
    const auto& value = metrics.continuation;
    // Null rather than zero when no candidate was preflighted: "nothing to compare against" and
    // "diverged at token 0" are different findings and must not read the same.
    Json agreement = Json(nullptr);
    if (value.candidate_agreement_observed) { agreement = value.deepest_candidate_agreement; }
    return Json{{"deepest_candidate_agreement", std::move(agreement)},
                {"source", continuation_source_name(value.source)},
                {"alias_kind", continuation_alias_kind_name(value.alias_kind)},
                {"final_miss_reason", continuation_miss_reason_name(value.final_miss_reason)},
                {"restore_failure", continuation_restore_failure_name(value.restore_failure)},
                {"lookup_microseconds", value.lookup_microseconds},
                {"preflight_microseconds", value.preflight_microseconds},
                {"restore_microseconds", value.restore_microseconds},
                {"restored_tokens", value.restored_tokens},
                {"restored_bytes", value.restored_bytes},
                {"cache_write_tokens", value.cache_write_tokens},
                {"restore_gate_checks", value.restore_gate_checks},
                {"restore_gate_passes", value.restore_gate_passes},
                {"restore_target_lane", value.restore_target_lane},
                {"admitted_lane", value.admitted_lane},
                {"destructive_rollback", value.destructive_rollback},
                {"completion_publication_queued", value.completion_publication_queued}};
}

// Joules per token, or null where the denominator is zero. A rate over no tokens is not zero
// energy per token, it is no measurement, and the two must not render the same.
Json joules_per_token(double joules, std::uint64_t tokens) {
    if (tokens == 0) { return Json(nullptr); }
    return joules / static_cast<double>(tokens);
}

// Energy is reported per token rather than tokens per joule because energy composes additively
// across phases and rates do not: prefill and decode joules-per-token can be summed against their
// own token counts, whereas averaging tokens-per-joule would need a harmonic mean and silently
// gives the wrong answer when it is done arithmetically.
Json energy_json(const ThroughputReport& report) {
    const IntervalEnergy& energy = report.energy;
    if (!energy.available) { return Json(nullptr); }
    const std::uint64_t total_tokens =
        report.computed_prefill_tokens + report.committed_decode_tokens;
    // `served` prices every joule the board drew, including the idle draw between requests, which
    // is what an operator actually pays. `active` removes the measured idle baseline and is the
    // figure that tracks the schedule rather than the duty cycle. They differ by a large factor on
    // a mostly-idle server, so both are published instead of one standing in for the other.
    return Json{{"board_joules", energy.board_joules},
                {"prefill_joules", energy.prefill_joules},
                {"decode_joules", energy.decode_joules},
                {"idle_joules", energy.idle_joules},
                {"idle_watts", energy.idle_watts},
                {"accounted_seconds", energy.accounted_seconds},
                {"residual_joules", energy.residual_joules},
                {"residual_fraction", energy.residual_fraction},
                {"joules_per_token",
                 Json{{"served", joules_per_token(energy.board_joules, total_tokens)},
                      {"active", joules_per_token(energy.board_joules - energy.idle_joules,
                                                  total_tokens)},
                      {"prefill", joules_per_token(energy.prefill_joules,
                                                   report.computed_prefill_tokens)},
                      {"decode", joules_per_token(energy.decode_joules,
                                                  report.committed_decode_tokens)}}}};
}

// Tokens/second with fixed precision, or "n/a" when the interval is degenerate.
std::string rate(double tokens, double seconds) {
    std::ostringstream out;
    if (seconds > 0.0 && tokens > 0.0) {
        out << std::fixed << std::setprecision(1) << (tokens / seconds) << "tok/s";
    } else {
        out << "n/a";
    }
    return out.str();
}

std::string seconds_str(double seconds) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << seconds << 's';
    return out.str();
}

// Compact resolved-sampler summary. temperature <= 0 is the exact-argmax path.
std::string sampler_str(const ninfer::ResolvedSamplingParameters& sampling) {
    if (sampling.temperature <= 0.0f) { return "greedy"; }
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << "temp=" << sampling.temperature
        << " top_p=" << sampling.top_p << " top_k=" << sampling.top_k;
    if (sampling.min_p > 0.0f) { out << " min_p=" << sampling.min_p; }
    if (sampling.presence_penalty != 0.0f) { out << " pres=" << sampling.presence_penalty; }
    if (sampling.frequency_penalty != 0.0f) { out << " freq=" << sampling.frequency_penalty; }
    out << " seed=" << sampling.seed;
    return out.str();
}

std::string speculative_str(const GenerationMetrics& metrics) {
    if (metrics.speculative_backend == SpeculativeBackend::None) { return "off"; }
    std::ostringstream out;
    out << product::speculative_backend_name(metrics.speculative_backend) << ' ' << std::fixed
        << std::setprecision(2);
    if (metrics.speculative_rounds > 0) {
        const double per_round = 1.0 + static_cast<double>(metrics.speculative_accepted_tokens) /
                                           static_cast<double>(metrics.speculative_rounds);
        out << per_round << "tok/round";
    } else {
        out << "n/a";
    }
    if (metrics.speculative_draft_tokens > 0) {
        const double accept_pct = 100.0 * static_cast<double>(metrics.speculative_accepted_tokens) /
                                  static_cast<double>(metrics.speculative_draft_tokens);
        out << " (" << std::setprecision(1) << accept_pct << "%)";
    }
    return out.str();
}

} // namespace

RequestLogContext make_request_log_context(std::uint64_t id, std::string x_request_id,
                                           std::string protocol, const GenerationRequest& request,
                                           const PreparedRequest& prepared) {
    RequestLogContext context;
    context.id                                 = id;
    context.x_request_id                       = std::move(x_request_id);
    context.protocol                           = std::move(protocol);
    context.model                              = request.model;
    context.stream                             = request.stream;
    context.prompt_tokens                      = prepared.prompt_tokens;
    context.message_count                      = request.messages.size();
    context.requested_output_tokens            = request.max_tokens;
    context.requested_output_tokens_client_set = request.max_tokens_set;
    context.tool_count                         = request.tools.size();
    context.tool_choice                        = request.tool_choice;
    context.has_tool_history                   = request.has_tool_history();
    context.enable_thinking                    = prepared.enable_thinking;
    context.preserve_thinking                  = prepared.preserve_thinking;
    context.preserve_thinking_semantic_change  = prepared.preserve_thinking_semantic_change;
    context.adapter                            = request.adapter;
    context.prompt_cache_key_digest =
        request.prompt_cache_routing_hint ? routing_hint_digest(*request.prompt_cache_routing_hint)
                                          : std::string{};
    context.sampling                           = prepared.sampling;
    return context;
}

std::string format_request_start(const RequestLogContext& context) {
    std::ostringstream out;
    out << "[req " << context.id << " x_request_id=" << context.x_request_id << "] "
        << context.protocol << ' ' << (context.stream ? "stream" : "non-stream")
        << " msgs=" << context.message_count << " max_tokens=" << context.requested_output_tokens
        << ' ' << (context.requested_output_tokens_client_set ? "(client)" : "(server default)")
        << " tools=" << context.tool_count
        << " tool_choice=" << tool_choice_name(context.tool_choice)
        << " tool_history=" << (context.has_tool_history ? "yes" : "no")
        << " thinking=" << (context.enable_thinking ? "on" : "off")
        << " preserve_thinking=" << (context.preserve_thinking ? "on" : "off")
        << " preserve_change=" << (context.preserve_thinking_semantic_change ? "yes" : "no")
        << " sampler=[" << sampler_str(context.sampling) << "] \xE2\x86\x92 submitted";
    return out.str();
}

std::string format_request_done(const RequestLogContext& context,
                                const GenerationOutcome& outcome) {
    const GenerationMetrics& metrics = outcome.metrics;
    const double ttft_ms             = metrics.ttft_seconds * 1000.0;
    // Prefill emits the first token; the remaining (gen - 1) come from decode.
    const double decode_tokens =
        outcome.completion_tokens > 0 ? static_cast<double>(outcome.completion_tokens - 1) : 0.0;
    const double computed_prefill_tokens = static_cast<double>(
        std::max(0, outcome.prompt_tokens - static_cast<int>(metrics.prefix_cache_hit_tokens)));

    std::ostringstream out;
    out << "[req " << context.id << " x_request_id=" << context.x_request_id << "] done finish="
        << (tool_calls_completed(outcome.tool_calls) ? "tool_calls"
                                                      : finish_reason_name(outcome.finish_reason));
    if (!outcome.tool_calls.empty()) {
        const ToolCallCounts counts = count_tool_calls(outcome.tool_calls);
        out << " tool_calls=" << outcome.tool_calls.size();
        if (counts.incomplete != 0) { out << " tool_calls_incomplete=" << counts.incomplete; }
        if (counts.invalid_arguments != 0) {
            out << " tool_calls_invalid_arguments=" << counts.invalid_arguments;
        }
    }
    if (outcome.tool_call_discarded_bytes != 0) {
        out << " tool_call_discarded_bytes=" << outcome.tool_call_discarded_bytes;
    }
    out << " prompt=" << outcome.prompt_tokens << " gen=" << outcome.completion_tokens
        << " cache=" << metrics.prefix_cache_hit_tokens
        << " reuse=" << prefix_reuse_path_name(metrics.prefix_reuse_path) << " ttft=" << std::fixed
        << std::setprecision(0) << ttft_ms << "ms"
        << " queue=" << (metrics.queue_seconds * 1000.0) << "ms"
        << " restore=" << (metrics.restore_seconds * 1000.0) << "ms"
        << " publish=" << (metrics.publish_seconds * 1000.0) << "ms"
        << " prefill=" << rate(computed_prefill_tokens, metrics.prefill_seconds)
        << " decode=" << rate(decode_tokens, metrics.decode_seconds)
        << " wall=" << seconds_str(metrics.total_seconds)
        << " speculative=" << speculative_str(metrics)
        << " cache_source=" << continuation_source_name(metrics.continuation.source)
        << " cache_alias=" << continuation_alias_kind_name(metrics.continuation.alias_kind)
        << " cache_miss=" << continuation_miss_reason_name(metrics.continuation.final_miss_reason)
        << " cache_restore_failure="
        << continuation_restore_failure_name(metrics.continuation.restore_failure)
        << " cache_lookup=" << std::setprecision(3)
        << static_cast<double>(metrics.continuation.lookup_microseconds) / 1000.0 << "ms"
        << " cache_preflight="
        << static_cast<double>(metrics.continuation.preflight_microseconds) / 1000.0 << "ms"
        << " cache_restore="
        << static_cast<double>(metrics.continuation.restore_microseconds) / 1000.0 << "ms"
        << " cache_tokens=" << metrics.continuation.restored_tokens
        << " cache_bytes=" << metrics.continuation.restored_bytes
        << " cache_write_tokens=" << metrics.continuation.cache_write_tokens
        << " cache_rollback=" << (metrics.continuation.destructive_rollback ? "yes" : "no")
        << " cache_publish_queued="
        << (metrics.continuation.completion_publication_queued ? "yes" : "no");
    return out.str();
}

std::string format_request_error(const RequestLogContext& context, const std::string& message) {
    std::ostringstream out;
    out << "[req " << context.id << " x_request_id=" << context.x_request_id << "] error "
        << message;
    return out.str();
}

std::string format_decision_start(const DecisionLogContext& context) {
    const ninfer::DecisionSummary& summary = context.summary;
    std::ostringstream out;
    out << "[req " << context.id << " x_request_id=" << context.x_request_id
        << "] systemone model=" << context.model << " adapter=" << context.adapter
        << " questions=" << summary.questions << " options=" << summary.options
        << " state=" << summary.state_tokens << (summary.state_truncated ? " (truncated)" : "")
        << " branches=" << summary.branch_tokens << " longest_branch=" << summary.longest_branch
        << " reuse=" << (context.allow_prefix_reuse ? "on" : "off") << " \xE2\x86\x92 submitted";
    return out.str();
}

std::string format_decision_done(const DecisionLogContext& context,
                                 const ninfer::DecisionResult& result,
                                 std::uint32_t output_tokens) {
    const ninfer::DecisionTimings& timings = result.timings;
    const std::uint32_t computed_state =
        result.summary.state_tokens - result.reused_state_tokens;
    std::ostringstream out;
    out << "[req " << context.id << " x_request_id=" << context.x_request_id
        << "] done decision questions=" << result.summary.questions
        << " state=" << result.summary.state_tokens << " reused=" << result.reused_state_tokens
        << " source=" << continuation_source_name(result.state_source)
        << " branches=" << result.summary.branch_tokens << " passes=" << result.branch_passes
        << " long_chunks=" << result.long_branch_chunks << " slot=" << result.slot
        << " output_tokens=" << output_tokens << std::fixed << std::setprecision(1)
        << " queue=" << timings.queue_seconds * 1000.0 << "ms"
        << " state_prefill=" << rate(computed_state, timings.state_seconds)
        << " branch_prefill=" << rate(result.summary.branch_tokens, timings.branch_seconds)
        << " execution=" << timings.execution_seconds * 1000.0 << "ms"
        << " wall="
        << seconds_str(context.prepare_seconds + timings.total_seconds - timings.prepare_seconds);
    return out.str();
}

std::string format_decision_error(const DecisionLogContext& context, int status,
                                  const std::string& message) {
    std::ostringstream out;
    out << "[req " << context.id << " x_request_id=" << context.x_request_id << "] error "
        << status << ' ' << message;
    return out.str();
}

std::string format_throughput(const ThroughputReport& report) {
    const double prefill_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.computed_prefill_tokens) / report.interval_seconds
            : 0.0;
    const double decode_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.committed_decode_tokens) / report.interval_seconds
            : 0.0;
    std::ostringstream out;
    out << "throughput interval=" << std::fixed << std::setprecision(3) << report.interval_seconds
        << "s prefill=" << std::setprecision(1) << prefill_rate << "tok/s decode=" << decode_rate
        << "tok/s running=" << report.scheduler.running_requests
        << " prefilling=" << report.scheduler.prefilling_requests
        << " decode_ready=" << report.scheduler.decode_ready_requests
        << " waiting=" << report.scheduler.waiting_requests
        << " continuation_lookup=" << report.scheduler.continuation_lookup_hits << '/'
        << report.scheduler.continuation_lookup_misses
        << " continuation_restore=" << report.scheduler.continuation_restore_successes << '/'
        << report.scheduler.continuation_restore_failures
        << " continuation_publish=" << report.scheduler.continuation_publication_successes << '/'
        << report.scheduler.continuation_publication_failures << '/'
        << report.scheduler.continuation_publication_superseded << " avg_decode_batch=";
    if (report.decode_rounds == 0) {
        out << "n/a";
    } else {
        out << std::setprecision(2)
            << static_cast<double>(report.decode_row_rounds) /
                   static_cast<double>(report.decode_rounds);
    }
    out << " cache_tier_delta=" << report.continuation_delta.continuation_l1_restore_successes
        << '/' << report.continuation_delta.continuation_l2_restore_successes << '/'
        << report.continuation_delta.continuation_l3_restore_successes
        << " cache_tier_total=" << report.scheduler.continuation_l1_restore_successes << '/'
        << report.scheduler.continuation_l2_restore_successes << '/'
        << report.scheduler.continuation_l3_restore_successes
        << " cache_lookup_us_delta="
        << report.continuation_delta.continuation_l2_lookup_microseconds << '/'
        << report.continuation_delta.continuation_l3_lookup_microseconds
        << " cache_restore_us_delta="
        << report.continuation_delta.continuation_l2_restore_microseconds << '/'
        << report.continuation_delta.continuation_l3_restore_microseconds
        << " cache_publish_us_delta="
        << report.continuation_delta.continuation_l2_admission_microseconds << '/'
        << report.continuation_delta.continuation_l3_persistence_microseconds
        << " cache_ops_delta=" << report.continuation_delta.continuation_l2_lookup_operations << '/'
        << report.continuation_delta.continuation_l3_lookup_operations << '/'
        << report.continuation_delta.continuation_l2_restore_operations << '/'
        << report.continuation_delta.continuation_l3_restore_operations;
    out << " cache_tokens_delta="
        << report.continuation_delta.continuation_l1_restored_tokens << '/'
        << report.continuation_delta.continuation_l2_restored_tokens << '/'
        << report.continuation_delta.continuation_l3_restored_tokens
        << " cache_bytes_delta="
        << report.continuation_delta.continuation_l1_restored_bytes << '/'
        << report.continuation_delta.continuation_l2_restored_bytes << '/'
        << report.continuation_delta.continuation_l3_restored_bytes;
    if (report.energy.available) {
        out << " energy=" << std::setprecision(1) << report.energy.board_joules << "J idle="
            << report.energy.idle_watts << 'W';
        out << " prefill_energy=";
        if (report.computed_prefill_tokens != 0) {
            out << std::setprecision(3)
                << report.energy.prefill_joules /
                       static_cast<double>(report.computed_prefill_tokens)
                << "J/tok";
        } else {
            out << "n/a";
        }
        out << " decode_energy=";
        if (report.committed_decode_tokens != 0) {
            out << std::setprecision(3)
                << report.energy.decode_joules /
                       static_cast<double>(report.committed_decode_tokens)
                << "J/tok";
        } else {
            out << "n/a";
        }
        out << " energy_residual=" << std::setprecision(1)
            << (report.energy.residual_fraction * 100.0) << '%';
    }
    return out.str();
}

std::string format_server_start_json(
    const std::string& server_instance_id, std::uint64_t timestamp, const ServeOptions& options,
    const ninfer::ModelSamplingDefaults& sampling_defaults, const std::string& public_model_id,
    const ninfer::LoadSummary& load, const ninfer::MemorySummary& memory,
    const ServerLogEnvironment& environment, std::optional<std::uint64_t> artifact_size_bytes) {
    Json record = event_base(server_instance_id, timestamp, "server_start");

    Json artifact_size = nullptr;
    if (artifact_size_bytes.has_value()) { artifact_size = *artifact_size_bytes; }

    record["server"]   = Json{{"host", options.host},
                              {"port", options.port},
                              {"public_model_id", public_model_id},
                              {"api_key_configured", !options.api_key.empty()},
                              {"cors_enabled", options.enable_cors},
                              {"max_request_bytes", options.max_request_bytes},
                              {"request_log_jsonl", options.request_log_jsonl},
                              {"slot_save_path", options.slot_save_path},
                              {"default_output_tokens", options.default_max_tokens},
                              {"default_thinking", options.enable_thinking},
                              {"default_preserve_thinking", options.preserve_thinking}};
    record["artifact"] = Json{{"path", options.artifact_path},
                              {"size_bytes", std::move(artifact_size)},
                              {"target", load.target},
                              {"weights_id", load.weights_id},
                              {"bytes_read", load.artifact_bytes_read},
                              {"host_to_device_bytes", load.host_to_device_bytes},
                              {"peak_staging_bytes", load.peak_staging_bytes},
                              {"tensor_count", load.tensor_count},
                              {"resource_count", load.resource_count},
                              {"load_seconds", load.load_seconds},
                              {"upload_seconds", load.upload_seconds}};
    // The discovered adapter pool, so a replayed log can name every selectable adapter even
    // when no request used one. `count` is the pool; `slots` is how many are resident at once.
    // `decision_names` are the System One models; `systemone_default` is the configured binding of
    // the SDK-default model name (empty: the only decision adapter, if there is exactly one).
    Json adapter_names  = Json::array();
    Json decision_names = Json::array();
    for (const ninfer::LoraAdapterInfo& adapter : load.lora_adapters) {
        adapter_names.push_back(adapter.name);
        if (adapter.kind == ninfer::LoraAdapterKind::Decision) {
            decision_names.push_back(adapter.name);
        }
    }
    record["adapters"] = Json{{"count", load.lora_adapters.size()},
                              {"names", std::move(adapter_names)},
                              {"decision_names", std::move(decision_names)},
                              {"systemone_default", options.systemone_default},
                              {"rank", load.lora_rank},
                              {"slots", load.lora_slots},
                              {"device_bytes", load.lora_device_bytes},
                              {"file_bytes", load.lora_file_bytes}};
    record["engine"]   = Json{
          {"device", options.device},
          {"max_context", options.max_context},
          {"kv_capacity_mode", kv_capacity_mode_name(memory.kv_capacity_mode)},
          {"kv_capacity", memory.kv_capacity},
          {"kv_capacity_page_groups", memory.kv_capacity_page_groups},
          {"kv_capacity_max_page_groups", memory.kv_capacity_max_page_groups},
          {"max_concurrency", options.max_concurrency},
          {"max_pending_requests", options.max_pending_requests},
          {"pending_timeout_ms", options.pending_timeout_ms},
          {"prefill_chunk", options.prefill_chunk},
          {"log_stats_interval_ms", options.log_stats_interval_ms},
          {"kv_cache", kv_cache_name(options.kv_cache)},
          {"vision", options.enable_vision},
          {"cuda_graph", options.use_cuda_graph},
          {"prefix_reuse", options.allow_prefix_reuse},
          {"prefix_checkpoint_policy",
           product::prefix_checkpoint_policy_name(options.prefix_checkpoint_policy)},
          {"speculative_backend", product::speculative_backend_name(options.speculative.backend)},
          {"speculative_draft_window", options.speculative.draft_tokens},
          {"proposal_head", proposal_head_name(options.speculative.proposal_head)},
          {"continuation_cache",
           Json{
               {"tiers", continuation_cache_tiers_name(options.continuation_cache.tiers)},
               {"policy", continuation_cache_policy_name(options.continuation_cache.policy)},
               {"l1_capacity_mib", options.continuation_cache.l1_capacity_mib},
               {"l2_capacity_mib", options.continuation_cache.l2_capacity_mib},
               {"l3_capacity_mib", options.continuation_cache.l3_capacity_mib},
               {"directory", options.continuation_cache.directory.string()},
               {"namespace", options.continuation_cache.cache_namespace},
               {"l1_idle_ttl_seconds", options.continuation_cache.l1_idle_ttl_seconds},
               {"l2_idle_ttl_seconds", options.continuation_cache.l2_idle_ttl_seconds},
               {"l3_idle_ttl_seconds", options.continuation_cache.l3_idle_ttl_seconds},
               {"persist_interval_seconds", options.continuation_cache.persist_interval_seconds},
               {"persist_min_tokens", options.continuation_cache.persist_min_tokens},
               {"filesystem_reserve_mib", options.continuation_cache.filesystem_reserve_mib},
               {"prefix_checkpoint_history", options.continuation_cache.prefix_checkpoint_history}}}};
    record["sampling_defaults"] =
        Json{{"thinking", preset_json(sampling_defaults.thinking)},
             {"non_thinking", preset_json(sampling_defaults.non_thinking)},
             {"server_overrides", overrides_json(options.sampling_overrides)},
             {"omitted_seed", "random"},
             {"greedy", options.greedy}};
    record["memory"] =
        Json{{"weights", arena_json(memory.weights)},
             {"sequence", arena_json(memory.sequence)},
             {"workspace", arena_json(memory.workspace)},
             {"request_transient", arena_json(memory.request_transient)},
             {"minimum_runtime_reservation_bytes", memory.minimum_runtime_reservation_bytes},
             {"kv_capacity_increment_bytes", memory.kv_capacity_increment_bytes},
             {"runtime_reservation_bytes", memory.runtime_reservation_bytes},
             {"available_after_weights_bytes", memory.available_after_weights_bytes},
             {"available_after_startup_bytes", memory.available_after_startup_bytes},
             {"kv_capacity_headroom_bytes", memory.kv_capacity_headroom_bytes},
             {"planned_slack_bytes", memory.planned_slack_bytes},
             {"cuda_graph_allowance_bytes", memory.cuda_graph_allowance_bytes},
             {"cuda_graph_observed_bytes", memory.cuda_graph_observed_bytes},
             {"kv_payload_bytes", memory.kv_payload_bytes},
             {"lora_bank_bytes", memory.lora_bank_bytes}};
    record["environment"] =
        Json{{"device", environment.device},
             {"gpu_name", environment.gpu_name},
             {"gpu_uuid", environment.gpu_uuid},
             {"total_device_memory_bytes", environment.total_device_memory_bytes},
             {"compute_capability_major", environment.compute_capability_major},
             {"compute_capability_minor", environment.compute_capability_minor},
             {"cuda_compile_version", environment.cuda_compile_version},
             {"cuda_runtime_version", environment.cuda_runtime_version},
             {"cuda_driver_version", environment.cuda_driver_version}};
    record["argv"] = options.startup_argv;
    return record.dump();
}

std::string format_request_start_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp, const RequestLogContext& context) {
    Json record       = event_base(server_instance_id, timestamp, "request_start");
    record["request"] = request_json(context);
    return record.dump();
}

std::string format_request_done_json(const std::string& server_instance_id, std::uint64_t timestamp,
                                     const RequestLogContext& context,
                                     const GenerationOutcome& outcome) {
    Json record                 = event_base(server_instance_id, timestamp, "request_done");
    record["request"]           = request_json(context);
    const ToolCallCounts counts = count_tool_calls(outcome.tool_calls);
    record["result"] =
        Json{{"finish_reason", finish_reason_name(outcome.finish_reason)},
             {"prompt_tokens", outcome.prompt_tokens},
             {"completion_tokens", outcome.completion_tokens},
             {"computed_prefill_tokens",
              std::max(0, outcome.prompt_tokens -
                              static_cast<int>(outcome.metrics.prefix_cache_hit_tokens))},
             {"prefix_cache_hit_tokens", outcome.metrics.prefix_cache_hit_tokens},
             {"prefix_reuse_path", prefix_reuse_path_name(outcome.metrics.prefix_reuse_path)},
             {"tool_call_count", outcome.tool_calls.size()},
             {"tool_calls_incomplete", counts.incomplete},
             {"tool_calls_invalid_arguments", counts.invalid_arguments},
             {"tool_call_discarded_bytes", outcome.tool_call_discarded_bytes}};
    // queue + restore + prefill decompose the engine-side part of ttft. prepare and vision are
    // frontend work that precedes submission.
    record["timings_seconds"] = Json{{"prepare", outcome.metrics.prepare_seconds},
                                     {"ttft", outcome.metrics.ttft_seconds},
                                     {"vision", outcome.metrics.vision_seconds},
                                     {"queue", outcome.metrics.queue_seconds},
                                     {"restore", outcome.metrics.restore_seconds},
                                     {"publish", outcome.metrics.publish_seconds},
                                     {"prefill", outcome.metrics.prefill_seconds},
                                     {"decode", outcome.metrics.decode_seconds},
                                     {"total", outcome.metrics.total_seconds}};
    record["speculative"] = speculative_json(outcome.metrics);
    record["continuation_cache"] = continuation_json(outcome.metrics);
    return record.dump();
}

std::string format_request_error_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp, const RequestLogContext& context,
                                      const std::string& message) {
    Json record       = event_base(server_instance_id, timestamp, "request_error");
    record["request"] = request_json(context);
    record["error"]   = Json{{"message", message}};
    return record.dump();
}

std::string format_decision_start_json(const std::string& server_instance_id,
                                       std::uint64_t timestamp, const DecisionLogContext& context) {
    Json record       = event_base(server_instance_id, timestamp, "decision_start");
    record["request"] = decision_request_json(context);
    return record.dump();
}

std::string format_decision_done_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp, const DecisionLogContext& context,
                                      const ninfer::DecisionResult& result,
                                      std::uint32_t output_tokens) {
    const std::uint32_t computed_state =
        result.summary.state_tokens - result.reused_state_tokens;
    const ninfer::DecisionTimings& timings = result.timings;
    Json record       = event_base(server_instance_id, timestamp, "decision_done");
    record["request"] = decision_request_json(context);
    record["result"]  = Json{{"output_tokens", output_tokens},
                             {"reused_state_tokens", result.reused_state_tokens},
                             {"computed_state_tokens", computed_state},
                             {"state_source", continuation_source_name(result.state_source)},
                             {"branch_passes", result.branch_passes},
                             {"long_branch_chunks", result.long_branch_chunks},
                             {"slot", result.slot}};
    // prepare is the HTTP side (validation, rendering, layout); the rest is the Engine's, from
    // submission. state + branch decompose execution, which is what latency_ms reports.
    record["timings_seconds"] =
        Json{{"prepare", context.prepare_seconds},
             {"queue", timings.queue_seconds},
             {"restore", timings.restore_seconds},
             {"state", timings.state_seconds},
             {"branch", timings.branch_seconds},
             {"execution", timings.execution_seconds},
             {"total", context.prepare_seconds + timings.total_seconds - timings.prepare_seconds}};
    record["rates"] =
        Json{{"state_tok_s", tokens_per_second(computed_state, timings.state_seconds)},
             {"branch_tok_s", tokens_per_second(result.summary.branch_tokens,
                                                timings.branch_seconds)}};
    return record.dump();
}

std::string format_decision_error_json(const std::string& server_instance_id,
                                       std::uint64_t timestamp, const DecisionLogContext& context,
                                       int status, const std::string& message) {
    Json record       = event_base(server_instance_id, timestamp, "decision_error");
    record["request"] = decision_request_json(context);
    record["error"]   = Json{{"status", status}, {"message", message}};
    return record.dump();
}

std::string format_throughput_json(const std::string& server_instance_id, std::uint64_t timestamp,
                                   const ThroughputReport& report) {
    Json record = event_base(server_instance_id, timestamp, "throughput");
    const double prefill_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.computed_prefill_tokens) / report.interval_seconds
            : 0.0;
    const double decode_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.committed_decode_tokens) / report.interval_seconds
            : 0.0;
    Json average_batch = nullptr;
    if (report.decode_rounds != 0) {
        average_batch = static_cast<double>(report.decode_row_rounds) /
                        static_cast<double>(report.decode_rounds);
    }
    record["interval_seconds"] = report.interval_seconds;
    record["tokens"]           = Json{{"computed_prefill", report.computed_prefill_tokens},
                                      {"committed_decode", report.committed_decode_tokens}};
    record["throughput_tokens_per_second"] =
        Json{{"prefill", prefill_rate}, {"decode", decode_rate}};
    // Null on boards with no cumulative energy counter, so a replayed log distinguishes "this
    // server could not measure energy" from "this server used none".
    record["energy"] = energy_json(report);
    record["scheduler"] = Json{
        {"running", report.scheduler.running_requests},
        {"prefilling", report.scheduler.prefilling_requests},
        {"decode_ready", report.scheduler.decode_ready_requests},
        {"waiting", report.scheduler.waiting_requests},
        {"admitted_requests", report.scheduler.admitted_requests},
        {"queue_seconds_total", report.scheduler.queue_seconds_total},
        {"delta_admitted_requests", report.continuation_delta.admitted_requests},
        {"delta_queue_seconds", report.continuation_delta.queue_seconds_total},
        {"rejected_overloaded", report.scheduler.admission_rejected_overloaded},
        {"rejected_queue_timeout", report.scheduler.admission_rejected_queue_timeout}};
    record["continuation_cache"] =
        Json{{"lookup_hits", report.scheduler.continuation_lookup_hits},
              {"lookup_misses", report.scheduler.continuation_lookup_misses},
              {"preflight_candidate_rejections",
               report.scheduler.continuation_preflight_rejections},
              {"restore_successes", report.scheduler.continuation_restore_successes},
              {"restore_failures", report.scheduler.continuation_restore_failures},
              // A deferral is a restore refused for shared-KV capacity with the candidate left
              // live, so it is restore pressure rather than a lost restore. It is reported apart
              // from failures because conflating them hides which of the two actually happened.
              {"restore_deferrals", report.scheduler.continuation_restore_deferrals},
              {"delta_restore_deferrals",
               report.continuation_delta.continuation_restore_deferrals},
              {"delta_lookup_hits", report.continuation_delta.continuation_lookup_hits},
              {"delta_lookup_misses", report.continuation_delta.continuation_lookup_misses},
              {"delta_preflight_candidate_rejections",
               report.continuation_delta.continuation_preflight_rejections},
              {"delta_restore_failures",
               report.continuation_delta.continuation_restore_failures},
              {"delta_restore_successes",
               report.continuation_delta.continuation_restore_successes},
              {"delta_restored_tokens", report.continuation_delta.continuation_restored_tokens},
              {"delta_restored_bytes", report.continuation_delta.continuation_restored_bytes},
              {"publication_successes", report.scheduler.continuation_publication_successes},
              {"publication_failures", report.scheduler.continuation_publication_failures},
              {"publication_superseded", report.scheduler.continuation_publication_superseded},
              {"delta_publication_successes",
               report.continuation_delta.continuation_publication_successes},
              {"delta_publication_failures",
               report.continuation_delta.continuation_publication_failures},
              {"delta_publication_superseded",
               report.continuation_delta.continuation_publication_superseded},
             {"restored_tokens", report.scheduler.continuation_restored_tokens},
              {"restored_bytes", report.scheduler.continuation_restored_bytes}};
    record["continuation_cache"]["tiers"] =
        Json{{"l1_restore_successes", report.scheduler.continuation_l1_restore_successes},
             {"l2_restore_successes", report.scheduler.continuation_l2_restore_successes},
             {"l3_restore_successes", report.scheduler.continuation_l3_restore_successes},
             {"delta_l1_restore_successes",
              report.continuation_delta.continuation_l1_restore_successes},
             {"delta_l2_restore_successes",
              report.continuation_delta.continuation_l2_restore_successes},
              {"delta_l3_restore_successes",
               report.continuation_delta.continuation_l3_restore_successes},
              {"l1_restored_tokens", report.scheduler.continuation_l1_restored_tokens},
              {"l2_restored_tokens", report.scheduler.continuation_l2_restored_tokens},
              {"l3_restored_tokens", report.scheduler.continuation_l3_restored_tokens},
              {"delta_l1_restored_tokens",
               report.continuation_delta.continuation_l1_restored_tokens},
              {"delta_l2_restored_tokens",
               report.continuation_delta.continuation_l2_restored_tokens},
              {"delta_l3_restored_tokens",
               report.continuation_delta.continuation_l3_restored_tokens},
              {"l1_restored_bytes", report.scheduler.continuation_l1_restored_bytes},
              {"l2_restored_bytes", report.scheduler.continuation_l2_restored_bytes},
              {"l3_restored_bytes", report.scheduler.continuation_l3_restored_bytes},
              {"delta_l1_restored_bytes",
               report.continuation_delta.continuation_l1_restored_bytes},
              {"delta_l2_restored_bytes",
               report.continuation_delta.continuation_l2_restored_bytes},
              {"delta_l3_restored_bytes",
               report.continuation_delta.continuation_l3_restored_bytes}};
    record["continuation_cache"]["aliases"] =
        Json{{"routed_session_restores", report.scheduler.continuation_session_restores},
             {"stable_prefix_restores", report.scheduler.continuation_stable_prefix_restores},
             {"delta_routed_session_restores",
              report.continuation_delta.continuation_session_restores},
             {"delta_stable_prefix_restores",
              report.continuation_delta.continuation_stable_prefix_restores}};
    record["continuation_cache"]["miss_reasons"] =
        Json{{"disabled", report.scheduler.continuation_miss_disabled},
             {"no_alias", report.scheduler.continuation_miss_no_alias},
             {"entry_unavailable_or_corrupt",
              report.scheduler.continuation_miss_entry_unavailable_or_corrupt},
             {"not_deeper", report.scheduler.continuation_miss_not_deeper},
             {"preflight_rejected", report.scheduler.continuation_miss_preflight_rejected},
             {"rollback_conflict", report.scheduler.continuation_miss_rollback_conflict},
             {"no_lane", report.scheduler.continuation_miss_no_lane},
             {"restore_failed", report.scheduler.continuation_miss_restore_failed},
             {"delta_disabled", report.continuation_delta.continuation_miss_disabled},
             {"delta_no_alias", report.continuation_delta.continuation_miss_no_alias},
             {"delta_entry_unavailable_or_corrupt",
              report.continuation_delta.continuation_miss_entry_unavailable_or_corrupt},
             {"delta_not_deeper", report.continuation_delta.continuation_miss_not_deeper},
             {"delta_preflight_rejected",
              report.continuation_delta.continuation_miss_preflight_rejected},
             {"delta_rollback_conflict",
              report.continuation_delta.continuation_miss_rollback_conflict},
             {"delta_no_lane", report.continuation_delta.continuation_miss_no_lane},
             {"delta_restore_failed",
              report.continuation_delta.continuation_miss_restore_failed}};
    record["continuation_cache"]["latency_microseconds"] =
        Json{{"l2_lookup_total", report.scheduler.continuation_l2_lookup_microseconds},
             {"l3_lookup_total", report.scheduler.continuation_l3_lookup_microseconds},
              {"l2_restore_total", report.scheduler.continuation_l2_restore_microseconds},
              {"l3_restore_total", report.scheduler.continuation_l3_restore_microseconds},
              {"preflight_total", report.scheduler.continuation_preflight_microseconds},
             {"l2_lookup_delta", report.continuation_delta.continuation_l2_lookup_microseconds},
             {"l3_lookup_delta", report.continuation_delta.continuation_l3_lookup_microseconds},
             {"l2_restore_delta", report.continuation_delta.continuation_l2_restore_microseconds},
              {"l3_restore_delta", report.continuation_delta.continuation_l3_restore_microseconds},
              {"preflight_delta",
               report.continuation_delta.continuation_preflight_microseconds},
             {"l2_admission_total", report.scheduler.continuation_l2_admission_microseconds},
             {"l3_persistence_total",
              report.scheduler.continuation_l3_persistence_microseconds},
             {"l2_admission_delta",
              report.continuation_delta.continuation_l2_admission_microseconds},
             {"l3_persistence_delta",
              report.continuation_delta.continuation_l3_persistence_microseconds},
             {"export_total", report.scheduler.continuation_export_microseconds},
             {"export_delta", report.continuation_delta.continuation_export_microseconds},
             {"export_operations_total", report.scheduler.continuation_export_operations},
             {"export_operations_delta",
              report.continuation_delta.continuation_export_operations},
             {"l2_lookup_operations_total",
              report.scheduler.continuation_l2_lookup_operations},
             {"l3_lookup_operations_total",
              report.scheduler.continuation_l3_lookup_operations},
             {"l2_restore_operations_total",
              report.scheduler.continuation_l2_restore_operations},
              {"l3_restore_operations_total",
               report.scheduler.continuation_l3_restore_operations},
              {"preflight_operations_total",
               report.scheduler.continuation_preflight_operations},
              {"l2_admission_operations_total",
               report.scheduler.continuation_l2_admission_operations},
              {"l3_persistence_operations_total",
               report.scheduler.continuation_l3_persistence_operations},
             {"l2_lookup_operations_delta",
              report.continuation_delta.continuation_l2_lookup_operations},
             {"l3_lookup_operations_delta",
              report.continuation_delta.continuation_l3_lookup_operations},
              {"l2_restore_operations_delta",
               report.continuation_delta.continuation_l2_restore_operations},
              {"l3_restore_operations_delta",
               report.continuation_delta.continuation_l3_restore_operations},
              {"preflight_operations_delta",
               report.continuation_delta.continuation_preflight_operations},
              {"l2_admission_operations_delta",
               report.continuation_delta.continuation_l2_admission_operations},
               {"l3_persistence_operations_delta",
                report.continuation_delta.continuation_l3_persistence_operations}};
    record["continuation_cache"]["occupancy"] =
        Json{{"l1_entries", report.scheduler.l1_resident_entries},
             {"l1_bytes", report.scheduler.l1_resident_bytes},
             {"l2_entries", report.scheduler.continuation_l2_entries},
             {"l2_bytes", report.scheduler.continuation_l2_bytes},
             {"l3_entries", report.scheduler.continuation_l3_entries},
              {"l3_bytes", report.scheduler.continuation_l3_bytes},
              {"l1_evictions", report.scheduler.l1_evictions},
              {"l1_demotions", report.scheduler.l1_demotions},
              {"kv_restore_reclaimed_lanes", report.scheduler.kv_restore_reclaimed_lanes},
              {"kv_growth_attempts", report.scheduler.kv_growth_attempts},
              {"kv_growth_forced_spills", report.scheduler.kv_growth_forced_spills},
              {"kv_growth_curtailed", report.scheduler.kv_growth_curtailed},
              {"delta_l1_evictions", report.continuation_delta.l1_evictions},
              {"delta_l1_demotions", report.continuation_delta.l1_demotions},
              {"l2_evictions", report.scheduler.continuation_l2_evictions},
              {"l2_evicted_bytes", report.scheduler.continuation_l2_evicted_bytes},
              {"l3_evictions", report.scheduler.continuation_l3_evictions},
              {"l3_evicted_bytes", report.scheduler.continuation_l3_evicted_bytes},
              {"delta_l2_evictions", report.continuation_delta.continuation_l2_evictions},
              {"delta_l2_evicted_bytes", report.continuation_delta.continuation_l2_evicted_bytes},
              {"delta_l3_evictions", report.continuation_delta.continuation_l3_evictions},
              {"delta_l3_evicted_bytes", report.continuation_delta.continuation_l3_evicted_bytes}};
    record["continuation_cache"]["persistence_delta"] =
        Json{{"queued", report.continuation_delta.continuation_persistence_queued},
             {"coalesced", report.continuation_delta.continuation_persistence_coalesced},
              {"successes", report.continuation_delta.continuation_persistence_successes},
              {"failures", report.continuation_delta.continuation_persistence_failures}};
    record["continuation_cache"]["persistence_total"] =
        Json{{"queued", report.scheduler.continuation_persistence_queued},
             {"coalesced", report.scheduler.continuation_persistence_coalesced},
             {"successes", report.scheduler.continuation_persistence_successes},
             {"failures", report.scheduler.continuation_persistence_failures}};
    record["decode_batch"] = Json{{"rounds", report.decode_rounds},
                                  {"row_rounds", report.decode_row_rounds},
                                  {"average_size", std::move(average_batch)}};
    return record.dump();
}

ServerLogEnvironment query_server_log_environment(int device) {
    ServerLogEnvironment environment;
    environment.device               = device;
    environment.cuda_compile_version = cuda_version_string(CUDART_VERSION);

    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        environment.cuda_runtime_version = cuda_version_string(runtime_version);
    }
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
        environment.cuda_driver_version = cuda_version_string(driver_version);
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
        environment.gpu_name                  = properties.name;
        environment.gpu_uuid                  = cuda_uuid_string(properties.uuid);
        environment.total_device_memory_bytes = properties.totalGlobalMem;
        environment.compute_capability_major  = properties.major;
        environment.compute_capability_minor  = properties.minor;
    }
    return environment;
}

JsonlRequestLog::JsonlRequestLog(const std::string& path,
                                 const std::string& protected_artifact_path)
    : path_(path) {
    if (path_.empty()) { return; }
    if (!protected_artifact_path.empty() &&
        normalized_absolute_path(path_) == normalized_absolute_path(protected_artifact_path)) {
        throw std::invalid_argument("request JSONL log must not overwrite the model artifact");
    }
    output_.open(path_, std::ios::out | std::ios::app);
    if (!output_) {
        throw std::runtime_error("failed to open request JSONL log for append: " + path_);
    }
}

void JsonlRequestLog::write_record(const std::string& record) {
    if (!enabled()) { return; }
    std::lock_guard<std::mutex> lock(mutex_);
    if (failed_) { return; }
    output_ << record << '\n';
    output_.flush();
    if (!output_) {
        failed_ = true;
        write_console_log(ConsoleLogLevel::Error, "request JSONL logging failed for " + path_);
    }
}

} // namespace ninfer::serve
