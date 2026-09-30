#include "serve/serve_metrics.h"

#include <algorithm>
#include <cstdio>
#include <string_view>

namespace ninfer::serve {

namespace {

void append_counter(std::string& out, const char* name, std::uint64_t value) {
    const std::string_view metric(name);
    out += "# TYPE ";
    out += name;
    out += metric.ends_with("_total") ? " counter\n" : " gauge\n";
    char line[160];
    std::snprintf(line, sizeof(line), "%s %llu\n", name, static_cast<unsigned long long>(value));
    out += line;
}

void append_counter(std::string& out, const char* name, double value) {
    const std::string_view metric(name);
    out += "# TYPE ";
    out += name;
    out += metric.ends_with("_total") ? " counter\n" : " gauge\n";
    char line[160];
    std::snprintf(line, sizeof(line), "%s %.6f\n", name, value);
    out += line;
}

} // namespace

void ServeMetrics::begin_request(std::uint64_t id, int prompt_tokens) {
    const std::lock_guard<std::mutex> lock(mutex_);
    active_[id] = prompt_tokens > 0 ? prompt_tokens : 0;
}

void ServeMetrics::end_request(std::uint64_t id) {
    const std::lock_guard<std::mutex> lock(mutex_);
    active_.erase(id);
}

std::vector<std::pair<std::uint64_t, int>> ServeMetrics::active_snapshot() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return {active_.begin(), active_.end()};
}

void ServeMetrics::record(const GenerationOutcome& outcome) {
    const GenerationMetrics& m = outcome.metrics;
    const std::uint64_t cached = m.prefix_cache_hit_tokens;
    const std::uint64_t prompt = outcome.prompt_tokens > 0
                                     ? static_cast<std::uint64_t>(outcome.prompt_tokens)
                                     : 0;

    const std::lock_guard<std::mutex> lock(mutex_);
    requests_total_ += 1;
    prefix_cache_hit_tokens_total_ += cached;
    speculative_draft_tokens_total_ += m.speculative_draft_tokens;
    speculative_accepted_tokens_total_ += m.speculative_accepted_tokens;
    last_completed_.prompt_tokens = static_cast<int>(prompt);
    // A cache figure reported larger than the prompt must not advertise
    // more resident tokens than exist.
    last_completed_.cached_tokens = static_cast<int>(std::min(cached, prompt));
}

void ServeMetrics::record_decision(const ninfer::DecisionResult& result) {
    const std::lock_guard<std::mutex> lock(mutex_);
    decisions_total_ += 1;
    decision_questions_total_ += result.summary.questions;
    decision_state_tokens_total_ += result.summary.state_tokens;
    decision_reused_state_tokens_total_ += result.reused_state_tokens;
    decision_branch_tokens_total_ += result.summary.branch_tokens;
    decision_execution_seconds_total_ += result.timings.execution_seconds;
}

ServeMetrics::LastCompleted ServeMetrics::last_completed() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return last_completed_;
}

std::string ServeMetrics::render(std::uint32_t max_concurrency, const ninfer::RuntimeStats& live,
                                 const ServerEnergyTotals& energy) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    const std::uint64_t in_flight  = active_.size();
    const std::uint64_t processing = std::min<std::uint64_t>(in_flight, max_concurrency);
    std::string out;
    // Sized for the continuation-tier series this fork adds on top of the llamacpp set.
    out.reserve(16384);
    append_counter(out, "llamacpp:prompt_tokens_total", live.computed_prefill_tokens);
    append_counter(out, "llamacpp:prompt_seconds_total", live.prefill_seconds_total);
    append_counter(out, "llamacpp:tokens_predicted_total", live.committed_decode_tokens);
    append_counter(out, "llamacpp:tokens_predicted_seconds_total", live.decode_seconds_total);
    append_counter(out, "llamacpp:requests_processing", processing);
    append_counter(out, "llamacpp:requests_deferred", in_flight - processing);
    append_counter(out, "ninfer:requests_total", requests_total_);
    append_counter(out, "ninfer:prefix_cache_hit_tokens_total", prefix_cache_hit_tokens_total_);
    append_counter(out, "ninfer:draft_tokens_total", speculative_draft_tokens_total_);
    append_counter(out, "ninfer:draft_accepted_tokens_total", speculative_accepted_tokens_total_);
    // Answered System One decisions. State tokens include the reused ones; execution excludes the
    // queue wait, as the response's latency_ms does.
    append_counter(out, "ninfer:decisions_total", decisions_total_);
    append_counter(out, "ninfer:decision_questions_total", decision_questions_total_);
    append_counter(out, "ninfer:decision_state_tokens_total", decision_state_tokens_total_);
    append_counter(out, "ninfer:decision_reused_state_tokens_total",
                   decision_reused_state_tokens_total_);
    append_counter(out, "ninfer:decision_branch_tokens_total", decision_branch_tokens_total_);
    append_counter(out, "ninfer:decision_execution_seconds_total",
                   decision_execution_seconds_total_);
    append_counter(out, "ninfer:continuation_lookup_hits_total", live.continuation_lookup_hits);
    append_counter(out, "ninfer:continuation_lookup_misses_total",
                   live.continuation_lookup_misses);
    // Per candidate, not per request: one request preflights every candidate its session alias
    // offers, so this exceeds continuation_miss_preflight_rejected_total whenever a later
    // candidate succeeded.
    append_counter(out, "ninfer:continuation_preflight_candidate_rejections_total",
                   live.continuation_preflight_rejections);
    append_counter(out, "ninfer:continuation_preparation_hits_total",
                   live.continuation_preparation_hits);
    append_counter(out, "ninfer:continuation_preparation_decoded_total",
                   live.continuation_preparation_decoded);
    append_counter(out, "ninfer:continuation_preparation_inline_total",
                   live.continuation_preparation_inline);
    append_counter(out, "ninfer:continuation_restore_successes_total",
                   live.continuation_restore_successes);
    append_counter(out, "ninfer:continuation_restore_failures_total",
                   live.continuation_restore_failures);
    append_counter(out, "ninfer:continuation_restore_deferrals_total",
                   live.continuation_restore_deferrals);
    append_counter(out, "ninfer:continuation_publication_successes_total",
                   live.continuation_publication_successes);
    append_counter(out, "ninfer:continuation_publication_failures_total",
                   live.continuation_publication_failures);
    append_counter(out, "ninfer:continuation_publication_superseded_total",
                   live.continuation_publication_superseded);
    append_counter(out, "ninfer:continuation_publication_coalesced_total",
                   live.continuation_publication_coalesced);
    append_counter(out, "ninfer:continuation_publication_failed_capacity_total",
                   live.continuation_publication_failed_capacity);
    append_counter(out, "ninfer:continuation_publication_failed_evicted_total",
                   live.continuation_publication_failed_evicted);
    append_counter(out, "ninfer:continuation_publication_failed_alias_moved_total",
                   live.continuation_publication_failed_alias_moved);
    append_counter(out, "ninfer:continuation_publication_failed_lineage_total",
                   live.continuation_publication_failed_lineage);
    append_counter(out, "ninfer:continuation_publication_failed_error_total",
                   live.continuation_publication_failed_error);
    append_counter(out, "ninfer:continuation_restored_tokens_total",
                   live.continuation_restored_tokens);
    append_counter(out, "ninfer:continuation_restored_bytes_total",
                   live.continuation_restored_bytes);
    append_counter(out, "ninfer:continuation_l1_restore_successes_total",
                   live.continuation_l1_restore_successes);
    append_counter(out, "ninfer:continuation_l2_restore_successes_total",
                   live.continuation_l2_restore_successes);
    append_counter(out, "ninfer:continuation_l3_restore_successes_total",
                   live.continuation_l3_restore_successes);
    append_counter(out, "ninfer:continuation_l1_restored_tokens_total",
                   live.continuation_l1_restored_tokens);
    append_counter(out, "ninfer:continuation_l2_restored_tokens_total",
                   live.continuation_l2_restored_tokens);
    append_counter(out, "ninfer:continuation_l3_restored_tokens_total",
                   live.continuation_l3_restored_tokens);
    append_counter(out, "ninfer:continuation_l1_restored_bytes_total",
                   live.continuation_l1_restored_bytes);
    append_counter(out, "ninfer:continuation_l2_restored_bytes_total",
                   live.continuation_l2_restored_bytes);
    append_counter(out, "ninfer:continuation_l3_restored_bytes_total",
                   live.continuation_l3_restored_bytes);
    append_counter(out, "ninfer:continuation_session_restores_total",
                   live.continuation_session_restores);
    append_counter(out, "ninfer:continuation_stable_prefix_restores_total",
                   live.continuation_stable_prefix_restores);
    append_counter(out, "ninfer:continuation_miss_disabled_total",
                   live.continuation_miss_disabled);
    append_counter(out, "ninfer:continuation_miss_no_alias_total",
                   live.continuation_miss_no_alias);
    append_counter(out, "ninfer:continuation_miss_not_attempted_total",
                   live.continuation_miss_not_attempted);
    append_counter(out, "ninfer:continuation_miss_entry_unavailable_or_corrupt_total",
                   live.continuation_miss_entry_unavailable_or_corrupt);
    append_counter(out, "ninfer:continuation_miss_not_deeper_total",
                   live.continuation_miss_not_deeper);
    append_counter(out, "ninfer:continuation_miss_preflight_rejected_total",
                   live.continuation_miss_preflight_rejected);
    append_counter(out, "ninfer:continuation_miss_rollback_conflict_total",
                   live.continuation_miss_rollback_conflict);
    append_counter(out, "ninfer:continuation_miss_no_lane_total",
                   live.continuation_miss_no_lane);
    append_counter(out, "ninfer:continuation_miss_restore_failed_total",
                   live.continuation_miss_restore_failed);
    // Restore-failure attribution: these sum to continuation_restore_failures_total.
    append_counter(out, "ninfer:continuation_restore_failed_kv_reservation_total",
                   live.continuation_restore_failed_kv_reservation);
    append_counter(out, "ninfer:continuation_restore_failed_verify_depth_total",
                   live.continuation_restore_failed_verify_depth);
    append_counter(out, "ninfer:continuation_restore_failed_inventory_total",
                   live.continuation_restore_failed_inventory);
    append_counter(out, "ninfer:continuation_restore_failed_metadata_total",
                   live.continuation_restore_failed_metadata);
    append_counter(out, "ninfer:continuation_restore_failed_decode_total",
                   live.continuation_restore_failed_decode);
    append_counter(out, "ninfer:continuation_restore_failed_lane_total",
                   live.continuation_restore_failed_lane);
    // Ingress refusals and the admission delay that dominates TTFT under concurrency.
    append_counter(out, "ninfer:admission_rejected_overloaded_total",
                   live.admission_rejected_overloaded);
    append_counter(out, "ninfer:admission_rejected_queue_timeout_total",
                   live.admission_rejected_queue_timeout);
    append_counter(out, "ninfer:admitted_requests_total", live.admitted_requests);
    append_counter(out, "ninfer:queue_seconds_total", live.queue_seconds_total);
    // Execution-thread wall clock. These units are mutually exclusive: their ratio decides how
    // much of a decoding lane's time is spent waiting on another lane's prompt.
    append_counter(out, "ninfer:worker_decode_seconds_total", live.worker_decode_seconds);
    append_counter(out, "ninfer:worker_prefill_seconds_total", live.worker_prefill_seconds);
    append_counter(out, "ninfer:worker_admission_seconds_total", live.worker_admission_seconds);
    append_counter(out, "ninfer:worker_admission_calls_total", live.worker_admission_calls);
    append_counter(out, "ninfer:worker_admission_plan_seconds_total",
                   live.worker_admission_plan_seconds);
    append_counter(out, "ninfer:worker_admission_restore_seconds_total",
                   live.worker_admission_restore_seconds);
    append_counter(out, "ninfer:worker_admission_commit_seconds_total",
                   live.worker_admission_commit_seconds);
    append_counter(out, "ninfer:worker_publish_seconds_total", live.worker_publish_seconds);
    append_counter(out, "ninfer:worker_upkeep_seconds_total", live.worker_upkeep_seconds);
    append_counter(out, "ninfer:worker_decode_rounds_total", live.worker_decode_rounds);
    append_counter(out, "ninfer:worker_prefill_steps_total", live.worker_prefill_steps);
    append_counter(out, "ninfer:continuation_l2_lookup_microseconds_total",
                   live.continuation_l2_lookup_microseconds);
    append_counter(out, "ninfer:continuation_l2_lookup_operations_total",
                   live.continuation_l2_lookup_operations);
    append_counter(out, "ninfer:continuation_l3_lookup_microseconds_total",
                   live.continuation_l3_lookup_microseconds);
    append_counter(out, "ninfer:continuation_l3_lookup_operations_total",
                   live.continuation_l3_lookup_operations);
    append_counter(out, "ninfer:continuation_preflight_microseconds_total",
                   live.continuation_preflight_microseconds);
    append_counter(out, "ninfer:continuation_preflight_operations_total",
                   live.continuation_preflight_operations);
    append_counter(out, "ninfer:continuation_l2_restore_microseconds_total",
                   live.continuation_l2_restore_microseconds);
    append_counter(out, "ninfer:continuation_l2_restore_operations_total",
                   live.continuation_l2_restore_operations);
    append_counter(out, "ninfer:continuation_l3_restore_microseconds_total",
                   live.continuation_l3_restore_microseconds);
    append_counter(out, "ninfer:continuation_l3_restore_operations_total",
                   live.continuation_l3_restore_operations);
    append_counter(out, "ninfer:continuation_l2_admission_microseconds_total",
                   live.continuation_l2_admission_microseconds);
    append_counter(out, "ninfer:continuation_l2_admission_operations_total",
                   live.continuation_l2_admission_operations);
    append_counter(out, "ninfer:continuation_export_microseconds_total",
                   live.continuation_export_microseconds);
    append_counter(out, "ninfer:continuation_export_operations_total",
                   live.continuation_export_operations);
    append_counter(out, "ninfer:continuation_l3_persistence_microseconds_total",
                   live.continuation_l3_persistence_microseconds);
    append_counter(out, "ninfer:continuation_l3_persistence_operations_total",
                   live.continuation_l3_persistence_operations);
    append_counter(out, "ninfer:continuation_persistence_queued_total",
                   live.continuation_persistence_queued);
    append_counter(out, "ninfer:continuation_persistence_coalesced_total",
                   live.continuation_persistence_coalesced);
    append_counter(out, "ninfer:continuation_persistence_successes_total",
                   live.continuation_persistence_successes);
    append_counter(out, "ninfer:continuation_persistence_failures_total",
                   live.continuation_persistence_failures);
    append_counter(out, "ninfer:continuation_l2_entries",
                   static_cast<std::uint64_t>(live.continuation_l2_entries));
    append_counter(out, "ninfer:continuation_l2_bytes", live.continuation_l2_bytes);
    append_counter(out, "ninfer:continuation_l3_entries",
                   static_cast<std::uint64_t>(live.continuation_l3_entries));
    append_counter(out, "ninfer:continuation_l3_bytes", live.continuation_l3_bytes);
    append_counter(out, "ninfer:continuation_l2_evictions_total", live.continuation_l2_evictions);
    append_counter(out, "ninfer:continuation_l2_evicted_bytes_total",
                   live.continuation_l2_evicted_bytes);
    append_counter(out, "ninfer:continuation_l3_evictions_total", live.continuation_l3_evictions);
    append_counter(out, "ninfer:continuation_l3_evicted_bytes_total",
                   live.continuation_l3_evicted_bytes);
    append_counter(out, "ninfer:l1_evictions_total", live.l1_evictions);
    append_counter(out, "ninfer:l1_demotions_total", live.l1_demotions);
    append_counter(out, "ninfer:kv_restore_reclaimed_lanes_total", live.kv_restore_reclaimed_lanes);
    append_counter(out, "ninfer:kv_growth_attempts_total", live.kv_growth_attempts);
    append_counter(out, "ninfer:kv_growth_forced_spills_total", live.kv_growth_forced_spills);
    append_counter(out, "ninfer:kv_growth_curtailed_total", live.kv_growth_curtailed);
    append_counter(out, "ninfer:l1_resident_entries",
                   static_cast<std::uint64_t>(live.l1_resident_entries));
    append_counter(out, "ninfer:l1_resident_bytes", live.l1_resident_bytes);
    // Energy is exported as counters, never as a joules-per-token gauge. The denominators are
    // already here (llamacpp:prompt_tokens_total and llamacpp:tokens_predicted_total), so a scraper
    // divides two rates over a window it chose; a gauge computed here would fix that window and
    // would average incorrectly when aggregated.
    if (energy.available) {
        append_counter(out, "ninfer:board_energy_joules_total", energy.board_joules_total);
        append_counter(out, "ninfer:board_idle_watts", energy.idle_watts);
        append_counter(out, "ninfer:prefill_energy_joules_total", live.prefill_energy_joules);
        append_counter(out, "ninfer:decode_energy_joules_total", live.decode_energy_joules);
        append_counter(out, "ninfer:energy_accounted_seconds_total",
                       live.energy_accounted_seconds);
        append_counter(out, "ninfer:energy_samples_total", live.energy_samples);
    }
    return out;
}

} // namespace ninfer::serve
