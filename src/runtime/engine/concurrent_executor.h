#pragma once

// Small fixed-capacity request scheduling and batched decode execution for every backend.

#include "core/power_meter.h"
#include "ninfer/types.h"
#include "runtime/cache/continuation_cache.h"
#include "runtime/contract/types.h"
#include "runtime/engine/admission_policy.h"
#include "runtime/engine/continuation_candidate_selection.h"
#include "runtime/engine/l1_retention_policy.h"
#include "runtime/engine/request_memory.h"
#include "runtime/engine/stable_prefix_flights.h"
#include "runtime/generation/generation_budget.h"
#include "runtime/generation/repetition_guard.h"
#include "runtime/generation/row_commit.h"
#include "targets/qwen3_8/export/ninfer/targets/qwen3_8/frontend.h"
#include "targets/qwen3_8/export/ninfer/targets/qwen3_8/prepared_prompt.h"
#include "targets/qwen3_8/export/ninfer/targets/qwen3_8/runtime.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// Namespaces a continuation-cache alias by the adapter that produced it. Every alias is scoped,
// including the base weights, so no unscoped key exists and two adapters can never collide. The
// separator is ASCII unit-separator, which no session id or stable-prefix digest contains.
//
// `scope` is the adapter's content fingerprint rather than its pool position or its bank slot.
// Both of those are engine bookkeeping that changes as adapters are swapped and as the directory
// gains or loses files, while L3 aliases persist across restarts; keying by either would let one
// adapter inherit another's KV and GDN state.
inline std::string adapter_scoped_alias(std::string_view scope, std::string_view alias) {
    std::string scoped(scope);
    scoped += '\x1f';
    scoped += alias;
    return scoped;
}

template <class Instance>
class ConcurrentExecutor {
    struct Request;

public:
    using Package  = typename Instance::Package;
    using Program  = typename Package::Program;
    using BasePlan = typename Package::RequestBasePlan;
    using Plan     = typename Package::RequestPlan;
    using Clock    = std::chrono::steady_clock;

    // `adapters` is the LoRA pool in index order; telemetry names a lane's or a bank slot's adapter
    // through it.
    ConcurrentExecutor(Instance& instance, const EngineOptions& options,
                       const std::vector<LoraAdapterInfo>& adapters)
        : instance_(instance), max_concurrency_(options.max_concurrency),
          max_outstanding_(static_cast<std::size_t>(options.max_concurrency) +
                           options.max_pending_requests),
          pending_timeout_(std::chrono::milliseconds(options.pending_timeout_ms)),
          auto_save_evicted_(options.auto_save_evicted),
          repetition_guard_(options.repetition_guard),
          admission_capacity_(instance.program->admission_capacity()),
          adapter_names_(adapter_names(adapters)),
          prefill_decode_balance_(options.prefill_decode_balance),
          continuation_cache_(make_continuation_cache(options.continuation_cache)),
          l1_policy_active_(options.continuation_cache.tiers != ContinuationCacheTiers::Off),
          l1_byte_budget_(
              mib_to_bytes(options.continuation_cache.l1_capacity_mib, "continuation L1 capacity")),
          l1_idle_ttl_(std::chrono::seconds(options.continuation_cache.l1_idle_ttl_seconds)),
          publication_l2_ttl_(std::chrono::seconds(options.continuation_cache.l2_idle_ttl_seconds)),
          publication_l3_ttl_(std::chrono::seconds(options.continuation_cache.l3_idle_ttl_seconds)),
          power_meter_(options.device) {
        if (max_concurrency_ == 0 || max_concurrency_ > kMaximumConcurrency ||
            options.max_pending_requests == 0 || pending_timeout_.count() <= 0) {
            throw std::invalid_argument("concurrent executor bounds are invalid");
        }
        if (admission_capacity_.active_lanes != max_concurrency_ ||
            admission_capacity_.main_kv_pages == 0) {
            throw std::logic_error("target admission capacity does not match the Engine");
        }
        // Readers are served from the published snapshot, so it must describe the startup state
        // before the first unit runs.
        publish_runtime_stats();
        if (continuation_cache_) {
            publication_worker_ = std::thread([this] { publication_loop(); });
            try {
                preparation_worker_ = std::thread([this] { preparation_loop(); });
            } catch (...) {
                stop_publication_worker();
                throw;
            }
        }
        try {
            worker_ = std::thread([this] { worker_loop(); });
        } catch (...) {
            stop_preparation_worker();
            stop_publication_worker();
            throw;
        }
    }

    ~ConcurrentExecutor() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
        // Preparation is speculative and holds no cache state, so it is abandoned before the
        // publication worker drains the work that must survive shutdown.
        stop_preparation_worker();
        stop_publication_worker();
    }

    ConcurrentExecutor(const ConcurrentExecutor&)            = delete;
    ConcurrentExecutor& operator=(const ConcurrentExecutor&) = delete;

    class Submission {
    public:
        Submission() noexcept = default;

        ~Submission() { reset(); }

        Submission(Submission&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), request_(std::move(other.request_)) {}

        Submission& operator=(Submission&& other) noexcept {
            if (this != &other) {
                reset();
                owner_   = std::exchange(other.owner_, nullptr);
                request_ = std::move(other.request_);
            }
            return *this;
        }

        Submission(const Submission&)            = delete;
        Submission& operator=(const Submission&) = delete;

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("concurrent submission is empty");
            }
            ConcurrentExecutor* owner = std::exchange(owner_, nullptr);
            return owner->wait_for_request(std::exchange(request_, nullptr), sink, cancellation);
        }

    private:
        Submission(ConcurrentExecutor& owner, std::shared_ptr<Request> request) noexcept
            : owner_(&owner), request_(std::move(request)) {}

        void reset() noexcept {
            if (owner_ != nullptr && request_ != nullptr) {
                owner_->abandon_request(std::move(request_));
            }
            owner_ = nullptr;
        }

        ConcurrentExecutor* owner_ = nullptr;
        std::shared_ptr<Request> request_;

        friend class ConcurrentExecutor;
    };

    // A decision's handle. Destroying it unconsumed cancels the decision.
    class DecisionSubmission {
    public:
        DecisionSubmission() noexcept = default;

        ~DecisionSubmission() { reset(); }

        DecisionSubmission(DecisionSubmission&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), request_(std::move(other.request_)) {}

        DecisionSubmission& operator=(DecisionSubmission&& other) noexcept {
            if (this != &other) {
                reset();
                owner_   = std::exchange(other.owner_, nullptr);
                request_ = std::move(other.request_);
            }
            return *this;
        }

        DecisionSubmission(const DecisionSubmission&)            = delete;
        DecisionSubmission& operator=(const DecisionSubmission&) = delete;

        DecisionResult wait(const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("decision submission is empty");
            }
            ConcurrentExecutor* owner = std::exchange(owner_, nullptr);
            return owner->wait_for_decision(std::exchange(request_, nullptr), cancellation);
        }

    private:
        DecisionSubmission(ConcurrentExecutor& owner, std::shared_ptr<Request> request) noexcept
            : owner_(&owner), request_(std::move(request)) {}

        void reset() noexcept {
            if (owner_ != nullptr && request_ != nullptr) {
                owner_->abandon_request(std::move(request_));
            }
            owner_ = nullptr;
        }

        ConcurrentExecutor* owner_ = nullptr;
        std::shared_ptr<Request> request_;

        friend class ConcurrentExecutor;
    };

    // A System One decision joins the same bounded FIFO as a generation and is admitted under the
    // same lane, KV and adapter-slot rules. `adapter_name` is echoed in the result.
    DecisionSubmission submit_decision(targets::qwen3_8::DecisionPrompt prompt,
                                       ResolvedDecisionOptions options, std::string adapter_name,
                                       Clock::time_point pending_deadline = {}) {
        const Clock::time_point submitted = Clock::now();
        pending_deadline                  = resolve_pending_deadline(submitted, pending_deadline);
        const std::uint64_t request_id    = claim_outstanding();
        std::shared_ptr<Request> request;
        try {
            // The exact state is the decision's only cacheable unit. Its alias is namespaced by the
            // adapter like every chat alias, since the state encodes the adapter's weights, and the
            // lookup is descriptors only, as for a prompt boundary.
            std::string state_alias;
            CachedContinuation state_continuation;
            if (continuation_lookup_enabled(static_cast<bool>(continuation_cache_),
                                            options.allow_prefix_reuse)) {
                if (std::optional<std::string> alias =
                        instance_.program->decision_state_alias(prompt)) {
                    state_alias = adapter_scoped_alias(
                        instance_.program->adapter_scope(options.adapter), *alias);
                    state_continuation = lookup_continuation(
                        state_alias, ContinuationAliasKind::StablePrefix, false, true);
                }
            }
            // Published at completion only when the alias names no usable image, so a state reused
            // from L1 is republished only if its alias lapsed. An unavailable entry counts as
            // lapsed: a cold recomputation is bit-identical and so re-stores the same content.
            const bool publish_state =
                !state_alias.empty() && !state_continuation.image &&
                std::ranges::none_of(state_continuation.candidates, [](const auto& item) {
                    return item.status == cache::CacheLookupStatus::Hit;
                });
            request = std::make_shared<Request>(
                request_id,
                DecisionWork{.summary       = prompt.summary,
                             .prompt        = std::move(prompt),
                             .options       = options,
                             .adapter       = std::move(adapter_name),
                             .state_alias   = std::move(state_alias),
                             .publish_state = publish_state},
                pending_deadline, submitted, std::move(state_continuation));
        } catch (...) {
            release_reserved_capacity();
            throw;
        }
        enqueue_pending(request);
        return DecisionSubmission(*this, std::move(request));
    }

    Submission submit(targets::qwen3_8::PreparedPrompt prompt, PromptSummary prompt_summary,
                      double prepare_seconds, ResolvedRequestOptions options,
                      Clock::time_point pending_deadline = {}, HostInputLease host_input = {}) {
        const Clock::time_point submitted = Clock::now();
        pending_deadline                  = resolve_pending_deadline(submitted, pending_deadline);
        const std::uint64_t request_id    = claim_outstanding();

        // A continuation produced under one adapter encodes that adapter's weights in its KV and
        // GDN recurrent state, so every cache alias is namespaced by the selected adapter. This
        // is a correctness requirement, not an optimization.
        const std::string adapter_scope =
            instance_.program->adapter_scope(options.execution.adapter);
        if (options.routing_hint) {
            options.routing_hint = adapter_scoped_alias(adapter_scope, *options.routing_hint);
        }

        std::shared_ptr<Request> request;
        try {
            CachedContinuation routed_continuation;
            std::optional<PendingSessionPublication> pending_session_publication;
            std::vector<StableBoundary> boundaries;
            CachedContinuation stable_continuation;
            if (continuation_lookup_enabled(static_cast<bool>(continuation_cache_),
                                            options.execution.allow_prefix_reuse)) {
                if (options.routing_hint) {
                    // Bracket the non-waiting snapshot so a publication that completes or is
                    // queued during cache I/O is still reconciled by the scheduler.
                    const auto publication_before =
                        this->pending_session_publication(*options.routing_hint);
                    routed_continuation = lookup_continuation(
                        *options.routing_hint, ContinuationAliasKind::Session, false, true);
                    const auto publication_after =
                        this->pending_session_publication(*options.routing_hint);
                    if (publication_after &&
                        (!publication_before ||
                         publication_after->sequence != publication_before->sequence ||
                         !publication_before->completed)) {
                        pending_session_publication = publication_after;
                    }
                }
                for (targets::qwen3_8::PromptBoundaryAlias& boundary :
                     instance_.program->boundary_aliases(prompt)) {
                    // Descriptors only. Materialising here would put a full L3 image read plus
                    // verification on the caller's thread, ahead of the queue, for a prefix that
                    // may not even beat what a lane already holds.
                    StableBoundary entry{.depth   = boundary.depth,
                                         .publish = boundary.publish,
                                         .alias   = adapter_scoped_alias(adapter_scope,
                                                                         boundary.alias)};
                    CachedContinuation lookup = lookup_continuation(
                        entry.alias, ContinuationAliasKind::StablePrefix, false, true);
                    const bool unpublished =
                        entry.publish && !lookup.image && lookup.candidates.empty();
                    entry.completion_publish = unpublished && boundary.rewrite_frontier;
                    entry.flight_needed      = unpublished && !boundary.rewrite_frontier;
                    merge_stable_lookup(stable_continuation, std::move(lookup));
                    boundaries.push_back(std::move(entry));
                }
            }
            auto output = instance_.loaded->frontend.make_output_session(prompt, options.stop,
                                                                          options.output);
            if (Clock::now() >= pending_deadline) {
                continuation_stats_.rejected_queue_timeout.fetch_add(1,
                                                                      std::memory_order_relaxed);
                throw RequestError(RequestErrorKind::QueueTimeout,
                                   "inference request expired before submission");
            }
            request     = std::make_shared<Request>(
                request_id, std::move(prompt), std::move(output), prompt_summary, prepare_seconds,
                std::move(options), pending_deadline, submitted, std::move(host_input),
                std::move(routed_continuation), std::move(stable_continuation),
                std::move(boundaries), std::move(pending_session_publication));
            acquire_stable_flights(request, false);
        } catch (...) {
            release_reserved_capacity();
            throw;
        }
        enqueue_pending(request);
        return Submission(*this, std::move(request));
    }

    // Served from the snapshot the worker publishes at every unit boundary and before a completion
    // wakes its client, like slot_states(): a scraper waiting on the execution mutex would starve
    // for as long as the worker keeps it busy, because the worker reacquires it between units
    // before a blocked reader is scheduled.
    [[nodiscard]] MemorySummary memory_summary() const {
        std::lock_guard lock(stats_mutex_);
        return published_memory_;
    }

    [[nodiscard]] RuntimeStats runtime_stats() const {
        std::lock_guard lock(stats_mutex_);
        RuntimeStats snapshot = published_stats_;
        snapshot.continuation_lookup_hits =
            continuation_stats_.lookup_hits.load(std::memory_order_relaxed);
        snapshot.continuation_lookup_misses =
            continuation_stats_.lookup_misses.load(std::memory_order_relaxed);
        snapshot.continuation_preflight_rejections =
            continuation_stats_.preflight_rejections.load(std::memory_order_relaxed);
        snapshot.continuation_preparation_hits =
            continuation_preparation_hits_.load(std::memory_order_relaxed);
        snapshot.continuation_preparation_decoded =
            continuation_preparation_decoded_.load(std::memory_order_relaxed);
        snapshot.continuation_preparation_inline =
            continuation_preparation_inline_.load(std::memory_order_relaxed);
        reconcile_continuation_aggregate_totals(snapshot);
        snapshot.continuation_restore_failures =
            continuation_stats_.restore_failures.load(std::memory_order_relaxed);
        snapshot.continuation_restore_deferrals =
            continuation_stats_.restore_deferrals.load(std::memory_order_relaxed);
        snapshot.continuation_publication_successes =
            continuation_stats_.publication_successes.load(std::memory_order_relaxed);
        snapshot.continuation_publication_failures =
            continuation_stats_.publication_failures.load(std::memory_order_relaxed);
        snapshot.continuation_publication_coalesced =
            continuation_stats_.publication_coalesced.load(std::memory_order_relaxed);
        snapshot.continuation_publication_failed_capacity =
            continuation_stats_.publication_failed_capacity.load(std::memory_order_relaxed);
        snapshot.continuation_publication_failed_lineage =
            continuation_stats_.publication_failed_lineage.load(std::memory_order_relaxed);
        snapshot.continuation_publication_failed_error =
            continuation_stats_.publication_failed_error.load(std::memory_order_relaxed);
        snapshot.continuation_publication_failed_evicted =
            continuation_stats_.publication_failed_evicted.load(std::memory_order_relaxed);
        snapshot.continuation_publication_failed_alias_moved =
            continuation_stats_.publication_failed_alias_moved.load(std::memory_order_relaxed);
        snapshot.continuation_publication_superseded =
            continuation_stats_.publication_superseded.load(std::memory_order_relaxed);
        snapshot.continuation_l2_lookup_microseconds =
            continuation_stats_.l2_lookup_microseconds.load(std::memory_order_relaxed);
        snapshot.continuation_l2_lookup_operations =
            continuation_stats_.l2_lookup_operations.load(std::memory_order_relaxed);
        snapshot.continuation_export_microseconds =
            continuation_stats_.export_microseconds.load(std::memory_order_relaxed);
        snapshot.continuation_export_operations =
            continuation_stats_.export_operations.load(std::memory_order_relaxed);
        snapshot.continuation_l3_lookup_microseconds =
            continuation_stats_.l3_lookup_microseconds.load(std::memory_order_relaxed);
        snapshot.continuation_l3_lookup_operations =
            continuation_stats_.l3_lookup_operations.load(std::memory_order_relaxed);
        snapshot.continuation_restore_failed_kv_reservation =
            continuation_stats_.restore_failed_kv_reservation.load(std::memory_order_relaxed);
        snapshot.continuation_restore_failed_verify_depth =
            continuation_stats_.restore_failed_verify_depth.load(std::memory_order_relaxed);
        snapshot.continuation_restore_failed_inventory =
            continuation_stats_.restore_failed_inventory.load(std::memory_order_relaxed);
        snapshot.continuation_restore_failed_metadata =
            continuation_stats_.restore_failed_metadata.load(std::memory_order_relaxed);
        snapshot.continuation_restore_failed_decode =
            continuation_stats_.restore_failed_decode.load(std::memory_order_relaxed);
        snapshot.continuation_restore_failed_lane =
            continuation_stats_.restore_failed_lane.load(std::memory_order_relaxed);
        snapshot.admission_rejected_overloaded =
            continuation_stats_.rejected_overloaded.load(std::memory_order_relaxed);
        snapshot.admission_rejected_queue_timeout =
            continuation_stats_.rejected_queue_timeout.load(std::memory_order_relaxed);
        if (continuation_cache_) {
            const cache::CacheStats cache = continuation_cache_->stats();
            snapshot.continuation_persistence_queued = cache.persistence_queued;
            snapshot.continuation_persistence_coalesced = cache.persistence_coalesced;
            snapshot.continuation_persistence_successes = cache.persistence_successes;
            snapshot.continuation_persistence_failures = cache.persistence_failures;
            snapshot.continuation_l2_entries = static_cast<std::uint32_t>(std::min<std::size_t>(
                cache.l2_entries, std::numeric_limits<std::uint32_t>::max()));
            snapshot.continuation_l2_bytes = cache.l2_bytes;
            snapshot.continuation_l3_entries = static_cast<std::uint32_t>(std::min<std::size_t>(
                cache.l3_entries, std::numeric_limits<std::uint32_t>::max()));
            snapshot.continuation_l3_bytes = cache.l3_bytes;
            snapshot.continuation_l2_evictions = cache.l2_evictions;
            snapshot.continuation_l2_evicted_bytes = cache.l2_evicted_bytes;
            snapshot.continuation_l3_evictions = cache.l3_evictions;
            snapshot.continuation_l3_evicted_bytes = cache.l3_evicted_bytes;
            snapshot.continuation_l2_admission_microseconds = cache.l2_admission_microseconds;
            snapshot.continuation_l2_admission_operations = cache.l2_admission_operations;
            snapshot.continuation_l3_persistence_microseconds = cache.l3_persistence_microseconds;
            snapshot.continuation_l3_persistence_operations = cache.l3_persistence_operations;
        }
        return snapshot;
    }

    void reset_memory_peaks() noexcept {
        try {
            std::scoped_lock lock(execution_mutex_);
            instance_.program->reset_memory_peaks();
            instance_.request_memory.reset_peak();
            publish_runtime_stats();
        } catch (...) {}
    }

    // Session persistence entry points. Each claims the execution mutex, so GPU copies land at
    // a request boundary; the worker resumes as soon as the device round trip completes. A lane
    // with an active request is refused rather than drained. A non-empty expected_digest is a
    // precondition on the lane's resident session, checked atomically with the operation.
    // session_path is the file the operation targets; a successful save or restore binds the
    // lane to it so an involuntary eviction can spill the session back (see
    // spill_retained_lane).
    [[nodiscard]] targets::qwen3_8::RetainedSessionSnapshot
    save_retained_lane(std::uint32_t lane, std::string_view model_binding,
                       std::string_view expected_digest, std::string_view session_path = {}) {
        std::scoped_lock lock(execution_mutex_);
        require_idle_lane(lane);
        require_session_digest(lane, expected_digest);
        wait_lane_export(lane);
        auto snapshot = instance_.program->save_retained_lane(lane, model_binding);
        if (!session_path.empty()) { lane_session_path_[lane] = session_path; }
        return snapshot;
    }

    [[nodiscard]] std::pair<std::uint32_t, std::string>
    restore_retained_lane(std::uint32_t lane, std::span<const std::uint8_t> snapshot,
                          std::string_view model_binding, std::string_view session_path = {}) {
        std::scoped_lock lock(execution_mutex_);
        require_idle_lane(lane);
        wait_lane_export(lane);
        if (instance_.program->has_retained_lane(lane)) {
            // Involuntary for whatever session held the lane: the client asked for a restore,
            // not for that session's destruction.
            spill_retained_lane(lane);
            instance_.program->evict_retained_lane(lane);
            invalidate_lane_plans(lane);
        }
        lane_session_path_[lane].clear();
        const std::uint32_t tokens = instance_.program->restore_retained_lane(
            lane, snapshot, model_binding,
            [this](std::uint32_t victim) { evict_retained_lane(victim); });
        invalidate_lane_plans(lane);
        if (!session_path.empty()) { lane_session_path_[lane] = session_path; }
        retained_digest_cache_[lane] = instance_.program->retained_lane_digest(lane);
        retained_checkpoints_cache_[lane] = instance_.program->retained_lane_checkpoints(lane);
        publish_runtime_stats();
        return {tokens, retained_digest_cache_[lane]};
    }

    std::uint32_t erase_retained_lane(std::uint32_t lane, std::string_view expected_digest) {
        std::scoped_lock lock(execution_mutex_);
        require_idle_lane(lane);
        require_session_digest(lane, expected_digest);
        const std::uint32_t tokens = instance_.program->retained_lane_depth(lane);
        // Explicit erase is a deletion request: never auto-save, and drop the binding.
        lane_session_path_[lane].clear();
        if (instance_.program->has_retained_lane(lane)) {
            wait_lane_export(lane);
            instance_.program->evict_retained_lane(lane);
            invalidate_lane_plans(lane);
            publish_runtime_stats();
        }
        return tokens;
    }

    // Installs the auto-save sink: the model binding save_retained_lane needs, and a consumer
    // that receives (path, snapshot) for each spilled session and writes the file off-thread.
    void set_eviction_sink(
        std::string model_binding,
        std::function<void(std::string, targets::qwen3_8::RetainedSessionSnapshot&&)> sink) {
        std::scoped_lock lock(execution_mutex_);
        eviction_model_binding_ = std::move(model_binding);
        eviction_sink_          = std::move(sink);
    }

    // Truthful per-lane occupancy: an active request's prompt size, or the retained session's
    // depth and identifying digest. Served from the snapshot the worker publishes at every unit
    // boundary - the execution mutex is held nearly continuously while a request runs, so a
    // scraper that waited on it would starve for the length of a deep prefill.
    [[nodiscard]] std::vector<SlotState> slot_states() const {
        std::lock_guard lock(stats_mutex_);
        std::vector<SlotState> states = published_slots_;
        states.resize(max_concurrency_);
        return states;
    }

    // The LoRA bank's slots from the same published snapshot, so a reader sees residency and lane
    // occupancy of one boundary.
    [[nodiscard]] std::vector<AdapterSlotState> adapter_slot_states() const {
        std::lock_guard lock(stats_mutex_);
        return published_adapter_slots_;
    }

private:
    static std::vector<std::string> adapter_names(const std::vector<LoraAdapterInfo>& adapters) {
        std::vector<std::string> names;
        names.reserve(adapters.size());
        for (const LoraAdapterInfo& adapter : adapters) { names.push_back(adapter.name); }
        return names;
    }

    // Empty for the base weights (-1).
    [[nodiscard]] std::string adapter_name(std::int32_t adapter) const {
        return adapter < 0 ? std::string{} : adapter_names_.at(static_cast<std::size_t>(adapter));
    }

    static std::size_t mib_to_bytes(std::size_t mib, const char* field) {
        constexpr std::size_t mib_bytes = 1024U * 1024U;
        if (mib > std::numeric_limits<std::size_t>::max() / mib_bytes) {
            throw std::invalid_argument(std::string(field) + " exceeds the addressable byte range");
        }
        return mib * mib_bytes;
    }

    static void validate_cache_namespace(const std::string& value) {
        const std::filesystem::path path(value);
        const bool safe_characters = std::all_of(value.begin(), value.end(), [](unsigned char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                   c == '.' || c == '_' || c == '-';
        });
        if (value.empty() || value == "." || value == ".." || path.is_absolute() ||
            path.has_root_name() || path.has_root_directory() ||
            std::distance(path.begin(), path.end()) != 1 || path.filename().string() != value ||
            !safe_characters) {
            throw std::invalid_argument(
                "continuation cache namespace must be one safe relative path component");
        }
    }

    static std::unique_ptr<cache::ContinuationCache>
    make_continuation_cache(const ContinuationCacheOptions& options) {
        if (options.policy != ContinuationCachePolicy::Adaptive) {
            throw std::invalid_argument("unsupported continuation cache policy");
        }

        switch (options.tiers) {
        case ContinuationCacheTiers::Off:
        case ContinuationCacheTiers::L1:
            return nullptr;
        case ContinuationCacheTiers::L1L2: {
            if (options.prefix_checkpoint_history == 0) {
                throw std::invalid_argument("continuation cache history must be positive");
            }
            return std::make_unique<cache::ContinuationCache>(cache::CacheConfig{
                .l2_byte_budget = mib_to_bytes(options.l2_capacity_mib, "continuation L2 capacity"),
                .session_history_depth = options.prefix_checkpoint_history,
                .l2_idle_ttl           = std::chrono::seconds(options.l2_idle_ttl_seconds),
            });
        }
        case ContinuationCacheTiers::L1L2L3: {
            if (options.directory.empty()) {
                throw std::invalid_argument("continuation L3 requires a cache directory");
            }
            if (options.prefix_checkpoint_history == 0) {
                throw std::invalid_argument("continuation cache history must be positive");
            }
            validate_cache_namespace(options.cache_namespace);
            return std::make_unique<cache::ContinuationCache>(cache::CacheConfig{
                .enable_l3      = true,
                .root           = options.directory / options.cache_namespace,
                .l2_byte_budget = mib_to_bytes(options.l2_capacity_mib, "continuation L2 capacity"),
                .l3_byte_budget = mib_to_bytes(options.l3_capacity_mib, "continuation L3 capacity"),
                .filesystem_reserve_bytes =
                    mib_to_bytes(options.filesystem_reserve_mib, "continuation filesystem reserve"),
                .session_history_depth = options.prefix_checkpoint_history,
                .l2_idle_ttl           = std::chrono::seconds(options.l2_idle_ttl_seconds),
                .l3_idle_ttl           = std::chrono::seconds(options.l3_idle_ttl_seconds),
                .persist_interval = std::chrono::seconds(options.persist_interval_seconds),
                .persist_min_tokens = options.persist_min_tokens,
            });
        }
        }
        throw std::invalid_argument("unsupported continuation cache tier selection");
    }

    struct CachedContinuation {
        std::optional<cache::ContentId> id;
        std::shared_ptr<const cache::ContinuationImage> image;
        std::vector<cache::SessionCandidateDescriptor> candidates;
        std::optional<std::uint64_t> generation;
        ContinuationAliasKind alias_kind = ContinuationAliasKind::None;
        cache::CacheSource source = cache::CacheSource::None;
        cache::CacheLookupStatus status = cache::CacheLookupStatus::Absent;
        std::uint64_t lookup_microseconds = 0;
    };

    // One content-addressed boundary of a request's prompt. Every boundary was looked up at
    // submission; a `publish` boundary that missed needs a write-once flight, which this request
    // either builds (captures and publishes the image) or follows (waits for its builder).
    struct StableBoundary {
        std::uint32_t depth = 0;
        bool publish        = false;
        std::string alias; // adapter-scoped
        // The last lookup found nothing and the boundary publishes, so someone must build it.
        bool flight_needed = false;
        bool builder       = false;
        // A captured image is queued; the publication worker releases the flight when it lands.
        bool publication_pending = false;
        // The boundary is the lane's turn-rewrite frontier, where the lane holds its turn
        // checkpoint through decode. Its image is the completed lane's image, published at
        // completion like a session head; nothing is captured for it during prefill and no flight
        // guards it, because a follower would otherwise wait on this request's whole decode.
        bool completion_publish = false;
    };

    // Descriptors from every boundary lookup (history lookups carry descriptors, never an image)
    // pool into one candidate list, deepest first, so the restore ranks them as one set. Status
    // carries the worst outcome seen: an unavailable entry stays reportable even when a
    // shallower boundary hit.
    static void merge_stable_lookup(CachedContinuation& pool, CachedContinuation&& lookup) {
        pool.lookup_microseconds += lookup.lookup_microseconds;
        pool.alias_kind = ContinuationAliasKind::StablePrefix;
        if (lookup.status == cache::CacheLookupStatus::UnavailableOrCorrupt ||
            pool.status == cache::CacheLookupStatus::Absent) {
            pool.status = lookup.status;
        }
        // Boundaries arrive ascending by depth; the pool is kept deepest first.
        pool.candidates.insert(pool.candidates.begin(),
                               std::make_move_iterator(lookup.candidates.begin()),
                               std::make_move_iterator(lookup.candidates.end()));
    }

    enum class PublicationStatus : std::uint8_t { Pending, Success, Failed, Superseded };
    using PublicationTicket = std::shared_ptr<std::atomic<PublicationStatus>>;

    // One unit of publication work. A plain item carries a complete image for one alias:
    // `session` names it and `immutable` selects the write-once stable-prefix path. An export
    // job (`export_lane` set) carries no image yet: the worker exports the retained lane first,
    // then publishes the image under `boundary_alias` (stable prefix, when non-empty) and under
    // `session` (session alias, when non-empty). The session part of an export job can be
    // superseded by a newer snapshot of the same session while queued; the boundary part cannot.
    // A `decision_state` export job exports the lane's retained decision state instead, whose
    // alias names its whole state (`boundary_depth` is the image frontier), and has no session.
    struct Publication {
        cache::ContinuationImage image;
        std::string session;
        std::optional<cache::ContentId> expected_head;
        std::optional<std::uint64_t> expected_generation;
        PublicationTicket ticket;
        std::uint64_t sequence = 0;
        bool immutable         = false;
        std::optional<std::pair<std::string, std::uint64_t>> stable_flight;
        std::optional<std::uint32_t> export_lane;
        std::string boundary_alias;
        std::uint32_t boundary_depth = 0;
        bool decision_state          = false;
    };

    struct PendingSessionPublication {
        std::uint64_t sequence = 0;
        bool completed         = false;
    };

    struct LaneSession {
        std::string name;
        std::optional<cache::ContentId> expected_head;
        std::optional<std::uint64_t> expected_generation;
        PublicationTicket publication;
    };

    void refresh_lane_provenance(std::uint32_t lane) noexcept {
        if (!instance_.program->has_retained_lane(lane)) {
            lane_provenance_[lane] = {};
            return;
        }
        if (lane_sessions_[lane] && lane_sessions_[lane]->publication &&
            lane_sessions_[lane]->publication->load(std::memory_order_acquire) ==
                PublicationStatus::Success) {
            lane_provenance_[lane] = completion_publication_provenance(
                lane_provenance_[lane], ContinuationAliasKind::Session);
        }
    }

    struct AtomicContinuationStats {
        std::atomic<std::uint64_t> lookup_hits{0};
        std::atomic<std::uint64_t> lookup_misses{0};
        std::atomic<std::uint64_t> preflight_rejections{0};
        std::atomic<std::uint64_t> restore_failures{0};
        // Restores refused for shared-KV capacity alone. These are retried, so they are not
        // failures and deliberately do not feed restore_failures or its attribution counters.
        std::atomic<std::uint64_t> restore_deferrals{0};
        std::atomic<std::uint64_t> publication_successes{0};
        std::atomic<std::uint64_t> publication_failures{0};
        std::atomic<std::uint64_t> publication_superseded{0};
        // Older queued publications for a session that a newer snapshot replaced before they ran.
        std::atomic<std::uint64_t> publication_coalesced{0};
        // Attribution for publication_failures: these five sum to it.
        std::atomic<std::uint64_t> publication_failed_capacity{0};
        std::atomic<std::uint64_t> publication_failed_evicted{0};
        std::atomic<std::uint64_t> publication_failed_alias_moved{0};
        // The image was parented on a head the session had already advanced past.
        std::atomic<std::uint64_t> publication_failed_lineage{0};
        // A publication threw. Kept separate because an unclassified throw was previously
        // indistinguishable from a capacity rejection, which made the counter unreadable.
        std::atomic<std::uint64_t> publication_failed_error{0};
        std::atomic<std::uint64_t> l2_lookup_microseconds{0};
        std::atomic<std::uint64_t> l2_lookup_operations{0};
        std::atomic<std::uint64_t> l3_lookup_microseconds{0};
        std::atomic<std::uint64_t> l3_lookup_operations{0};
        std::atomic<std::uint64_t> export_microseconds{0};
        std::atomic<std::uint64_t> export_operations{0};
        // Restore-failure attribution; these sum to restore_failures.
        std::atomic<std::uint64_t> restore_failed_kv_reservation{0};
        std::atomic<std::uint64_t> restore_failed_verify_depth{0};
        std::atomic<std::uint64_t> restore_failed_inventory{0};
        std::atomic<std::uint64_t> restore_failed_metadata{0};
        std::atomic<std::uint64_t> restore_failed_decode{0};
        std::atomic<std::uint64_t> restore_failed_lane{0};
        // Requests refused at ingress, which never reach the request log.
        std::atomic<std::uint64_t> rejected_overloaded{0};
        std::atomic<std::uint64_t> rejected_queue_timeout{0};
    };

    void record_restore_failure(ContinuationRestoreFailure failure) noexcept {
        auto* counter = [&]() -> std::atomic<std::uint64_t>* {
            switch (failure) {
            case ContinuationRestoreFailure::KvReservationExhausted:
                return &continuation_stats_.restore_failed_kv_reservation;
            case ContinuationRestoreFailure::VerifyDepthMismatch:
                return &continuation_stats_.restore_failed_verify_depth;
            case ContinuationRestoreFailure::SegmentInventoryMismatch:
                return &continuation_stats_.restore_failed_inventory;
            case ContinuationRestoreFailure::MetadataMismatch:
                return &continuation_stats_.restore_failed_metadata;
            case ContinuationRestoreFailure::DecodeFailed:
                return &continuation_stats_.restore_failed_decode;
            case ContinuationRestoreFailure::LaneUnavailable:
                return &continuation_stats_.restore_failed_lane;
            case ContinuationRestoreFailure::None: return nullptr;
            }
            return nullptr;
        }();
        if (counter != nullptr) { counter->fetch_add(1, std::memory_order_relaxed); }
    }

    // Charges a lazily resolved candidate's I/O to the request and to the tier that served it.
    void account_candidate_lookup(const std::shared_ptr<Request>& request,
                                   const cache::CacheLookupResult& lookup) noexcept {
        request->continuation.lookup_microseconds += lookup.io_microseconds;
        const CacheLookupAccountingTier tier = cache_lookup_accounting_tier(lookup.source);
        if (tier == CacheLookupAccountingTier::L3) {
            continuation_stats_.l3_lookup_microseconds.fetch_add(lookup.io_microseconds,
                                                                  std::memory_order_relaxed);
            continuation_stats_.l3_lookup_operations.fetch_add(1, std::memory_order_relaxed);
        } else if (tier == CacheLookupAccountingTier::L2) {
            continuation_stats_.l2_lookup_microseconds.fetch_add(lookup.io_microseconds,
                                                                  std::memory_order_relaxed);
            continuation_stats_.l2_lookup_operations.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Pages a restore of this image needs. The request's planned entitlement already covers the
    // whole prompt plus its output budget, so it dominates the image's own frontier; taking the
    // maximum only guards a plan that is somehow shallower than the candidate it selected.
    [[nodiscard]] runtime::KvPageFootprint
    continuation_image_kv_footprint(const cache::ContinuationImage& image,
                                     const std::shared_ptr<Request>& request) const noexcept {
        constexpr std::uint64_t kPageTokens = 64;
        const std::uint32_t frontier_pages =
            image.frontier_tokens == 0
                ? 0U
                : static_cast<std::uint32_t>((image.frontier_tokens + kPageTokens - 1) /
                                             kPageTokens);
        if (!request->base_plan) { return {.text_pages = frontier_pages, .backend_pages = 0}; }
        const AdmissionResources& planned = request->base_plan->summary().admission;
        return {.text_pages    = std::max(frontier_pages, planned.main_kv_pages),
                .backend_pages = planned.backend_kv_pages};
    }

    CachedContinuation lookup_continuation(const std::string& session,
                                             ContinuationAliasKind alias_kind,
                                             bool wait_for_publication = true,
                                             bool include_history = false,
                                             Clock::time_point deadline = {}) {
        const auto lookup_started = Clock::now();
        CachedContinuation candidate;
        candidate.alias_kind = alias_kind;
        try {
            if (wait_for_publication) {
                std::unique_lock lock(publication_mutex_);
                const auto pending = session_publications_.find(session);
                if (pending != session_publications_.end()) {
                    const std::uint64_t pending_sequence = pending->second.sequence;
                    const auto completed = [&] {
                        return publication_completed_ >= pending_sequence || publication_stopping_;
                    };
                    if (deadline == Clock::time_point{}) {
                        publication_cv_.wait(lock, completed);
                    } else if (!publication_cv_.wait_until(lock, deadline, completed)) {
                        throw RequestError(RequestErrorKind::QueueTimeout,
                                           "inference request expired before submission");
                    }
                }
            }
            if (include_history) {
                auto snapshot        = continuation_cache_->session_candidates(session);
                candidate.candidates = std::move(snapshot.newest_to_oldest);
                candidate.generation = snapshot.generation;
                if (!candidate.candidates.empty()) {
                    candidate.id    = candidate.candidates.front().id;
                    candidate.source = candidate.candidates.front().source;
                    candidate.status = candidate.candidates.front().status;
                }
                for (const auto& item : candidate.candidates) {
                    if (item.status == cache::CacheLookupStatus::UnavailableOrCorrupt) {
                        candidate.status = item.status;
                    }
                }
            } else {
                candidate.id = continuation_cache_->session_current(session);
                if (candidate.id) {
                    auto lookup = continuation_cache_->lookup_shared(*candidate.id);
                    candidate.image = std::move(lookup.image);
                    candidate.source = lookup.source;
                    candidate.status = lookup.status;
                    candidate.lookup_microseconds = lookup.io_microseconds;
                }
            }
        } catch (const RequestError&) {
            throw;
        } catch (...) {
            candidate.image.reset();
            candidate.status = cache::CacheLookupStatus::UnavailableOrCorrupt;
        }
        const bool hit = candidate.image ||
                         std::ranges::any_of(candidate.candidates, [](const auto& item) {
                             return item.status == cache::CacheLookupStatus::Hit;
                         });
        (hit ? continuation_stats_.lookup_hits : continuation_stats_.lookup_misses)
            .fetch_add(1, std::memory_order_relaxed);
        const auto account_lookup = [&](cache::CacheSource source, std::uint64_t microseconds) {
            if (cache_lookup_accounting_tier(source) == CacheLookupAccountingTier::L3) {
                continuation_stats_.l3_lookup_microseconds.fetch_add(microseconds,
                                                                     std::memory_order_relaxed);
                continuation_stats_.l3_lookup_operations.fetch_add(1, std::memory_order_relaxed);
            } else if (cache_lookup_accounting_tier(source) == CacheLookupAccountingTier::L2) {
                continuation_stats_.l2_lookup_microseconds.fetch_add(microseconds,
                                                                     std::memory_order_relaxed);
                continuation_stats_.l2_lookup_operations.fetch_add(1, std::memory_order_relaxed);
            }
        };
        if (!include_history && candidate.id) {
            account_lookup(candidate.source, candidate.lookup_microseconds);
        }
        candidate.lookup_microseconds = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - lookup_started)
                .count());
        return candidate;
    }

    [[nodiscard]] std::optional<PendingSessionPublication>
    pending_session_publication(const std::string& session) {
        std::lock_guard lock(publication_mutex_);
        const auto pending = session_publications_.find(session);
        return pending == session_publications_.end()
                   ? std::nullopt
                   : std::optional<PendingSessionPublication>(PendingSessionPublication{
                         .sequence = pending->second.sequence,
                         .completed = publication_completed_ >= pending->second.sequence ||
                                      publication_stopping_});
    }

    [[nodiscard]] bool publication_completed(std::uint64_t sequence) {
        std::lock_guard lock(publication_mutex_);
        return publication_completed_ >= sequence || publication_stopping_;
    }

    PublicationTicket queue_publication(cache::ContinuationImage image, const std::string& session,
                                         std::optional<cache::ContentId> expected_head,
                                         std::optional<std::uint64_t> expected_generation =
                                             std::nullopt,
                                         bool immutable = false,
                                        std::optional<std::pair<std::string, std::uint64_t>>
                                            stable_flight = std::nullopt) noexcept {
        Publication item{.image               = std::move(image),
                         .session             = session,
                         .expected_head       = std::move(expected_head),
                         .expected_generation = expected_generation,
                         .immutable           = immutable,
                         .stable_flight       = std::move(stable_flight)};
        return enqueue_publication(std::move(item));
    }

    // Queues one publication item and returns the ticket its session part reports through (an
    // export job without a session part still returns a ticket for its boundary part). Empty
    // when the worker is stopping or the queue could not take the item.
    PublicationTicket enqueue_publication(Publication item) noexcept {
        try {
            auto ticket =
                std::make_shared<std::atomic<PublicationStatus>>(PublicationStatus::Pending);
            std::lock_guard lock(publication_mutex_);
            if (publication_stopping_) {
                continuation_stats_.publication_failures.fetch_add(1, std::memory_order_relaxed);
                return {};
            }
            const std::uint64_t sequence = ++publication_issued_;
            // A session publishes a complete self-contained snapshot, so an older queued
            // publication for the same session is already dead: its CAS expects a head the newer
            // one is about to replace. Executing it costs a full multi-hundred-megabyte admission
            // and then records a failure. Fold it into the newer one instead, which inherits the
            // head the dropped publication was chaining from so the alias still advances. An
            // export job loses only its session part: its boundary alias is write-once and still
            // wants the image, so the job stays queued unless nothing is left to publish.
            if (!item.immutable && !item.session.empty()) {
                for (auto queued = publications_.begin(); queued != publications_.end();) {
                    if (queued->session != item.session || queued->immutable) {
                        ++queued;
                        continue;
                    }
                    item.expected_head       = queued->expected_head;
                    item.expected_generation = queued->expected_generation;
                    item.image.parent_id     = item.expected_head;
                    queued->ticket->store(PublicationStatus::Superseded,
                                          std::memory_order_release);
                    if (queued->stable_flight) {
                        std::lock_guard flight_lock(stable_flight_mutex_);
                        (void)stable_flights_.release(queued->stable_flight->first,
                                                      queued->stable_flight->second);
                        queued->stable_flight.reset();
                    }
                    continuation_stats_.publication_coalesced.fetch_add(
                        1, std::memory_order_relaxed);
                    if (queued->export_lane && !queued->boundary_alias.empty()) {
                        queued->session.clear();
                        queued->expected_head.reset();
                        queued->expected_generation.reset();
                        ++queued;
                        continue;
                    }
                    if (queued->export_lane) {
                        lane_export_pending_[*queued->export_lane] = false;
                    }
                    queued = publications_.erase(queued);
                }
            }
            if (item.export_lane) { lane_export_pending_[*item.export_lane] = true; }
            item.ticket   = ticket;
            item.sequence = sequence;
            if (!item.session.empty()) { session_publications_[item.session].sequence = sequence; }
            if (!item.boundary_alias.empty()) {
                session_publications_[item.boundary_alias].sequence = sequence;
            }
            publications_.push_back(std::move(item));
            publication_cv_.notify_all();
            return ticket;
        } catch (...) {
            continuation_stats_.publication_failures.fetch_add(1, std::memory_order_relaxed);
            continuation_stats_.publication_failed_error.fetch_add(1, std::memory_order_relaxed);
            return {};
        }
    }

    // Runs one alias publication and returns its status. `outcome`/`threw` carry the
    // attribution the failure counters need.
    PublicationStatus run_publication(cache::ContinuationImage image, const std::string& alias,
                                      bool immutable,
                                      const std::optional<cache::ContentId>& expected_head,
                                      std::optional<std::uint64_t> expected_generation,
                                      std::optional<cache::SessionPublishOutcome>& outcome,
                                      bool& threw) noexcept {
        PublicationStatus status = PublicationStatus::Failed;
        outcome.reset();
        threw = false;
        try {
            const cache::StoreOptions options{
                .recompute_cost = static_cast<double>(image.frontier_tokens),
                .l2_idle_ttl    = publication_l2_ttl_,
                .l3_idle_ttl    = publication_l3_ttl_,
            };
            const auto frontier_tokens = image.frontier_tokens;
            if (immutable) {
                const auto result =
                    continuation_cache_->publish_stable_alias(std::move(image), alias, options);
                outcome = result.outcome;
                if (result.alias_advanced) {
                    status = PublicationStatus::Success;
                    (void)continuation_cache_->queue_persistence(alias, result.id, frontier_tokens,
                                                                 cache::AliasKind::StablePrefix);
                } else if (result.outcome == cache::SessionPublishOutcome::AliasAlreadyOwned) {
                    // Another lane already owns this write-once stable prefix. The alias it
                    // holds serves this prompt just as well, so the contest is not a failure.
                    status = PublicationStatus::Superseded;
                }
            } else {
                const auto result = continuation_cache_->publish_session_l2(
                    std::move(image), alias, expected_head, options, expected_generation);
                if (result.alias_advanced) {
                    (void)continuation_cache_->queue_persistence(alias, result.id, frontier_tokens,
                                                                 cache::AliasKind::Session);
                }
                status  = result.alias_advanced ? PublicationStatus::Success
                                                : (result.stored ? PublicationStatus::Superseded
                                                                 : PublicationStatus::Failed);
                outcome = result.outcome;
            }
        } catch (...) {
            // A publication must not take the worker down, but it must not disappear either.
            threw  = true;
            status = PublicationStatus::Failed;
        }
        return status;
    }

    void account_publication(PublicationStatus status,
                             const std::optional<cache::SessionPublishOutcome>& outcome,
                             bool threw) noexcept {
        if (status == PublicationStatus::Success) {
            continuation_stats_.publication_successes.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (status == PublicationStatus::Superseded) {
            continuation_stats_.publication_superseded.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        continuation_stats_.publication_failures.fetch_add(1, std::memory_order_relaxed);
        // No default outcome stands in for an unclassified failure: a throw is its own
        // attribution. Defaulting this to a real outcome once reported every swallowed exception
        // as a capacity rejection.
        if (threw || !outcome) {
            continuation_stats_.publication_failed_error.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        switch (*outcome) {
        case cache::SessionPublishOutcome::EvictedOnAdmission:
            continuation_stats_.publication_failed_evicted.fetch_add(1, std::memory_order_relaxed);
            break;
        case cache::SessionPublishOutcome::HeadMoved:
        case cache::SessionPublishOutcome::GenerationMoved:
        // Reaching the failure branch with this outcome would be a logic surprise: the
        // immutable path reports it as Superseded. Group it with the other lost races rather
        // than with a capacity limit it has nothing to do with.
        case cache::SessionPublishOutcome::AliasAlreadyOwned:
            continuation_stats_.publication_failed_alias_moved.fetch_add(
                1, std::memory_order_relaxed);
            break;
        case cache::SessionPublishOutcome::LineageMismatch:
            continuation_stats_.publication_failed_lineage.fetch_add(1, std::memory_order_relaxed);
            break;
        // Only a real budget rejection reaches the capacity counter. A malformed or mis-kinded
        // alias is a caller defect, and Advanced here is a logic surprise; counting either as
        // capacity once reported a structural publication bug as pressure on a tier that was
        // almost empty.
        case cache::SessionPublishOutcome::InvalidAlias:
        case cache::SessionPublishOutcome::Advanced:
            continuation_stats_.publication_failed_error.fetch_add(1, std::memory_order_relaxed);
            break;
        case cache::SessionPublishOutcome::RejectedTooLarge:
            continuation_stats_.publication_failed_capacity.fetch_add(
                1, std::memory_order_relaxed);
            break;
        }
    }

    void publication_loop() noexcept {
        for (;;) {
            Publication item;
            {
                std::unique_lock lock(publication_mutex_);
                publication_cv_.wait(
                    lock, [&] { return publication_stopping_ || !publications_.empty(); });
                if (publications_.empty()) {
                    if (publication_stopping_) { return; }
                    continue;
                }
                item = std::move(publications_.front());
                publications_.pop_front();
            }
            bool export_failed = false;
            if (item.export_lane) {
                // The lane is retained and untouched until this returns: the execution thread
                // waits on the pending flag before reusing or releasing it.
                const auto export_started = Clock::now();
                try {
                    item.image = item.decision_state
                                     ? instance_.program->export_decision_state_background(
                                           *item.export_lane)
                                     : instance_.program->export_continuation_lane_background(
                                           *item.export_lane);
                } catch (...) { export_failed = true; }
                continuation_stats_.export_operations.fetch_add(1, std::memory_order_relaxed);
                continuation_stats_.export_microseconds.fetch_add(
                    static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() -
                                                                              export_started)
                            .count()),
                    std::memory_order_relaxed);
                {
                    std::lock_guard lock(publication_mutex_);
                    lane_export_pending_[*item.export_lane] = false;
                }
                publication_cv_.notify_all();
                // The session part may have been superseded while the job was queued; read the
                // parent the job holds now, not the one it was created with.
                if (!item.session.empty()) { item.image.parent_id = item.expected_head; }
            }
            std::optional<cache::SessionPublishOutcome> outcome;
            bool threw                       = false;
            PublicationStatus boundary_status = PublicationStatus::Failed;
            if (!item.boundary_alias.empty()) {
                // The cache is content-addressed over the serialized image, parent included, so
                // the boundary copy carries the session's parent too: both aliases then resolve
                // to one L2 payload instead of storing the same state twice. A chat image serves
                // its boundary alias at its turn checkpoint, a decision state at its frontier.
                const std::uint64_t alias_depth = item.decision_state
                                                      ? item.image.frontier_tokens
                                                      : item.image.boundary_tokens;
                if (export_failed || alias_depth != item.boundary_depth) {
                    threw = true;
                } else {
                    boundary_status = run_publication(
                        item.session.empty() ? std::move(item.image) : item.image,
                        item.boundary_alias, true, std::nullopt, std::nullopt, outcome, threw);
                }
                account_publication(boundary_status, outcome, threw);
            }
            if (!item.session.empty()) {
                PublicationStatus status = PublicationStatus::Failed;
                if (export_failed) {
                    threw = true;
                    outcome.reset();
                } else {
                    status = run_publication(std::move(item.image), item.session, item.immutable,
                                             item.expected_head, item.expected_generation,
                                             outcome, threw);
                }
                account_publication(status, outcome, threw);
                item.ticket->store(status, std::memory_order_release);
            } else if (item.ticket->load(std::memory_order_acquire) == PublicationStatus::Pending) {
                // A boundary-only job reports its boundary outcome; a job whose session part was
                // superseded while queued keeps that verdict.
                item.ticket->store(boundary_status, std::memory_order_release);
            }
            {
                std::lock_guard lock(publication_mutex_);
                publication_completed_ = item.sequence;
            }
            publication_cv_.notify_all();
            queue_cv_.notify_all();
            if (item.stable_flight) {
                {
                    std::lock_guard lock(stable_flight_mutex_);
                    (void)stable_flights_.release(item.stable_flight->first,
                                                  item.stable_flight->second);
                }
                queue_cv_.notify_all();
            }
        }
    }

    // Decode ahead for the requests nearest admission. Runs on the executor thread but outside
    // execution_mutex_, alongside refresh_stable_flights(), so it also runs during a prefill.
    void refresh_restore_preparation() noexcept {
        if (!continuation_cache_) { return; }
        try {
            const std::vector<std::shared_ptr<Request>> queued = pending_snapshot();

            std::lock_guard lock(preparation_mutex_);
            // Payloads are large, so an unclaimed one is dropped rather than held indefinitely.
            const auto now = Clock::now();
            // A session publishes a fresh image every turn, so an unclaimed payload is stale
            // quickly; holding one only starves the next preparation against kPreparedLimit.
            std::erase_if(prepared_, [&](const auto& entry) {
                return now - entry.second.prepared_at > std::chrono::seconds(10);
            });

            std::size_t considered = 0;
            for (const auto& request : queued) {
                if (considered >= kPreparationDepth) { break; }
                ++considered;
                if (request->cancelled.load(std::memory_order_relaxed)) { continue; }
                if (!request->options.routing_hint) { continue; }
                const std::string& session = *request->options.routing_hint;
                if (preparation_in_flight_.contains(session)) { continue; }
                if (prepared_.size() + preparation_queue_.size() >= kPreparedLimit) { break; }
                preparation_in_flight_.insert(session);
                preparation_queue_.push_back(PreparationJob{session});
            }
            if (!preparation_queue_.empty()) { preparation_cv_.notify_one(); }
        } catch (...) {
            // Preparation is an optimisation; a failure here must not disturb admission.
        }
    }

    void preparation_loop() noexcept {
        for (;;) {
            PreparationJob job;
            {
                std::unique_lock lock(preparation_mutex_);
                preparation_cv_.wait(lock, [this] {
                    return preparation_stopping_ || !preparation_queue_.empty();
                });
                if (preparation_stopping_) { return; }
                job = std::move(preparation_queue_.front());
                preparation_queue_.pop_front();
            }
            std::shared_ptr<targets::qwen3_8::DecodedContinuation> payload;
            cache::ContentId prepared_id;
            try {
                // Newest first, which is the head admission reconciles to.
                auto snapshot = continuation_cache_->session_candidates(job.session);
                for (const auto& descriptor : snapshot.newest_to_oldest) {
                    if (descriptor.status != cache::CacheLookupStatus::Hit) { continue; }
                    {
                        std::lock_guard lock(preparation_mutex_);
                        if (prepared_.contains(descriptor.id.hex)) { break; }
                    }
                    if (auto image = continuation_cache_->resolve_candidate(descriptor).image) {
                        payload     = instance_.program->decode_continuation(*image);
                        prepared_id = descriptor.id;
                    }
                    break;
                }
            } catch (...) {
                // A malformed image fails identically when the executor decodes it inline.
                payload.reset();
            }
            {
                std::lock_guard lock(preparation_mutex_);
                preparation_in_flight_.erase(job.session);
                if (payload) {
                    continuation_preparation_decoded_.fetch_add(1, std::memory_order_relaxed);
                }
                if (payload && !preparation_stopping_) {
                    prepared_[prepared_id.hex] =
                        PreparedContinuationEntry{std::move(payload), Clock::now()};
                }
            }
            queue_cv_.notify_all();
        }
    }

    // Claim the payload decoded for exactly this image. A payload is valid solely for the content
    // it was decoded from, so content identity is the whole matching rule.
    [[nodiscard]] std::shared_ptr<targets::qwen3_8::DecodedContinuation>
    take_prepared_continuation(const std::optional<cache::ContentId>& chosen) noexcept {
        if (!chosen || !chosen->valid()) { return nullptr; }
        std::lock_guard lock(preparation_mutex_);
        const auto entry = prepared_.find(chosen->hex);
        if (entry == prepared_.end()) { return nullptr; }
        auto payload = std::move(entry->second.payload);
        prepared_.erase(entry);
        continuation_preparation_hits_.fetch_add(1, std::memory_order_relaxed);
        return payload;
    }

    void stop_preparation_worker() noexcept {
        {
            std::lock_guard lock(preparation_mutex_);
            preparation_stopping_ = true;
            preparation_queue_.clear();
        }
        preparation_cv_.notify_all();
        if (preparation_worker_.joinable()) { preparation_worker_.join(); }
        std::lock_guard lock(preparation_mutex_);
        prepared_.clear();
    }

    void stop_publication_worker() noexcept {
        {
            std::lock_guard lock(publication_mutex_);
            publication_stopping_ = true;
        }
        publication_cv_.notify_all();
        if (publication_worker_.joinable()) { publication_worker_.join(); }
    }

    void require_idle_lane(std::uint32_t lane) const {
        if (lane >= max_concurrency_) {
            throw std::invalid_argument("slot id is outside the Engine lane count");
        }
        if (slots_[lane] != nullptr) {
            throw RequestError(RequestErrorKind::Overloaded, "slot is processing a request");
        }
    }

    void require_session_digest(std::uint32_t lane, std::string_view expected_digest) const {
        if (expected_digest.empty()) { return; }
        if (instance_.program->retained_lane_digest(lane) != expected_digest) {
            throw SlotSessionMismatch("slot session does not match if_digest");
        }
    }

    // Best-effort spill of a retained session about to be destroyed involuntarily. The device
    // snapshot runs on the calling thread (it synchronizes the stream); the file write happens
    // on the Engine's writer thread through the sink. Only sessions bound to a slot file are
    // spilled, and a spill failure never blocks the eviction itself.
    void spill_retained_lane(std::uint32_t lane) noexcept {
        if (!auto_save_evicted_ || !eviction_sink_ || lane >= kMaximumConcurrency) { return; }
        if (lane_session_path_[lane].empty() || !instance_.program->has_retained_lane(lane)) {
            return;
        }
        try {
            auto snapshot = instance_.program->save_retained_lane(lane, eviction_model_binding_);
            eviction_sink_(lane_session_path_[lane], std::move(snapshot));
        } catch (...) {
            // The session was going to be destroyed either way; losing the spill costs the
            // client one cold prefill, exactly the pre-feature behavior.
        }
    }

    void publish_runtime_stats() {
        RuntimeStats snapshot = cumulative_stats_;
        {
            std::lock_guard lock(queue_mutex_);
            snapshot.waiting_requests = static_cast<std::uint32_t>(pending_.size());
        }
        snapshot.prefilling_requests = static_cast<std::uint32_t>(prefill_lanes_.size());
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (l1_policy_active_ && instance_.program->has_retained_lane(lane)) {
                ++snapshot.l1_resident_entries;
                const std::size_t bytes = instance_.program->retained_lane_resident_bytes(lane);
                if (bytes <= std::numeric_limits<std::uint64_t>::max() -
                                 snapshot.l1_resident_bytes) {
                    snapshot.l1_resident_bytes += bytes;
                } else {
                    snapshot.l1_resident_bytes = std::numeric_limits<std::uint64_t>::max();
                }
            }
            if (slots_[lane] == nullptr) { continue; }
            ++snapshot.running_requests;
            if (slots_[lane]->decode_ready) { ++snapshot.decode_ready_requests; }
        }

        snapshot.lora_stages        = instance_.program->lora_stage_count();
        snapshot.lora_stage_seconds = instance_.program->lora_stage_seconds();

        // Per-lane occupancy for /slots-style readers. Digests come from the cache the
        // completion and restore paths maintain, so publishing costs no ledger hashing.
        std::vector<SlotState> slot_snapshot(max_concurrency_);
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            SlotState& state    = slot_snapshot[lane];
            const auto& request = slots_[lane];
            if (request != nullptr) {
                state.processing    = true;
                state.prompt_tokens = request->prompt_summary.prompt_tokens;
                if (request->begin) { state.cached_tokens = request->begin->reused_prompt_tokens; }
                state.work    = request->decision ? SlotWork::Decision : SlotWork::Generation;
                state.adapter = adapter_name(request->options.execution.adapter);
            } else if (instance_.program->has_retained_lane(lane)) {
                state.retained       = true;
                state.prompt_tokens  = instance_.program->retained_lane_depth(lane);
                state.cached_tokens  = state.prompt_tokens;
                state.session_digest = retained_digest_cache_[lane];
                state.checkpoints    = retained_checkpoints_cache_[lane];
                state.work           = instance_.program->retained_lane_holds_decision(lane)
                                           ? SlotWork::Decision
                                           : SlotWork::Generation;
                state.adapter        = adapter_name(instance_.program->retained_lane_adapter(lane));
            }
        }

        std::vector<AdapterSlotState> adapter_snapshot;
        for (const targets::qwen3_8::LoraSlotState& slot : instance_.program->lora_slot_states()) {
            adapter_snapshot.push_back(
                AdapterSlotState{.adapter = adapter_name(slot.adapter), .pinned = slot.pinned});
        }

        MemorySummary memory                     = instance_.program->memory_summary();
        memory.request_transient                 = instance_.request_memory.summary();
        const KvCapacityResolution& resolution   = instance_.kv_capacity_resolution;
        memory.kv_capacity_mode                  = resolution.mode;
        memory.kv_capacity_page_groups           = resolution.main_page_groups;
        memory.kv_capacity_max_page_groups       = resolution.maximum_main_page_groups;
        memory.minimum_runtime_reservation_bytes = resolution.minimum_runtime_reservation_bytes;
        memory.kv_capacity_increment_bytes       = resolution.bytes_per_additional_main_page_group;
        memory.runtime_reservation_bytes         = resolution.runtime_reservation_bytes;
        memory.available_after_weights_bytes     = resolution.available_after_weights_bytes;
        memory.available_after_startup_bytes     = resolution.available_after_startup_bytes;
        memory.kv_capacity_headroom_bytes        = resolution.automatic_headroom_bytes;
        memory.planned_slack_bytes               = resolution.planned_slack_bytes;

        std::lock_guard lock(stats_mutex_);
        published_stats_         = snapshot;
        published_slots_         = std::move(slot_snapshot);
        published_adapter_slots_ = std::move(adapter_snapshot);
        published_memory_        = memory;
    }

    GenerationResult wait_for_request(std::shared_ptr<Request> request, OutputSink* sink,
                                      const CancellationView& cancellation) {
        struct ConsumerGuard {
            ConcurrentExecutor* owner;
            std::shared_ptr<Request> request;

            ~ConsumerGuard() { owner->release_consumer(request); }
        } guard{this, request};

        std::exception_ptr caller_error;
        std::vector<OutputDelta> events;
        // Prefill of a long prompt can run for minutes without producing a
        // token. Reporting each chunk boundary keeps the caller informed and
        // keeps a streaming transport from looking dead to its client.
        std::optional<std::uint32_t> reported_prefill;
        for (;;) {
            events.clear();
            bool done = false;
            {
                std::unique_lock lock(request->mutex);
                request->cv.wait_for(lock, std::chrono::milliseconds(10),
                                     [&] { return request->done || !request->events.empty(); });
                events.swap(request->events);
                done = request->done;
            }

            if (caller_error == nullptr && sink != nullptr) {
                try {
                    for (OutputDelta& event : events) { sink->publish(std::move(event)); }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    queue_cv_.notify_one();
                }
            }

            if (caller_error == nullptr && sink != nullptr) {
                const std::uint32_t processed =
                    request->prefill_processed.load(std::memory_order_relaxed);
                if (reported_prefill != processed) {
                    reported_prefill = processed;
                    try {
                        sink->publish_prompt_progress(PromptProgress{
                            .processed_prompt_tokens = processed,
                            .prompt_tokens           = request->prompt_summary.prompt_tokens,
                            .reused_prompt_tokens =
                                request->prefill_reused.load(std::memory_order_relaxed),
                        });
                    } catch (...) {
                        caller_error = std::current_exception();
                        request->cancelled.store(true, std::memory_order_release);
                        queue_cv_.notify_one();
                    }
                }
            }

            if (caller_error == nullptr) {
                try {
                    if (cancellation.requested()) {
                        request->cancelled.store(true, std::memory_order_release);
                        queue_cv_.notify_one();
                    }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    queue_cv_.notify_one();
                }
            }
            if (!done) { continue; }

            if (caller_error != nullptr) { std::rethrow_exception(caller_error); }
            std::lock_guard lock(request->mutex);
            if (request->error != nullptr) { std::rethrow_exception(request->error); }
            return std::move(request->result);
        }
    }

    DecisionResult wait_for_decision(std::shared_ptr<Request> request,
                                     const CancellationView& cancellation) {
        struct ConsumerGuard {
            ConcurrentExecutor* owner;
            std::shared_ptr<Request> request;

            ~ConsumerGuard() { owner->release_consumer(request); }
        } guard{this, request};

        std::exception_ptr caller_error;
        for (;;) {
            bool done = false;
            {
                std::unique_lock lock(request->mutex);
                request->cv.wait_for(lock, std::chrono::milliseconds(10),
                                     [&] { return request->done; });
                done = request->done;
            }
            if (caller_error == nullptr) {
                try {
                    if (cancellation.requested()) {
                        request->cancelled.store(true, std::memory_order_release);
                        queue_cv_.notify_one();
                    }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    queue_cv_.notify_one();
                }
            }
            if (!done) { continue; }

            if (caller_error != nullptr) { std::rethrow_exception(caller_error); }
            std::lock_guard lock(request->mutex);
            if (request->error != nullptr) { std::rethrow_exception(request->error); }
            return std::move(request->decision->result);
        }
    }

    // What a System One decision carries through the queue instead of a prompt, an output session
    // and a budget. `prompt` moves into the Program at admission; `summary` stays for the result.
    struct DecisionWork {
        DecisionSummary summary;
        targets::qwen3_8::DecisionPrompt prompt;
        ResolvedDecisionOptions options;
        std::string adapter;
        // Adapter-scoped alias of the exact state row, empty when the cache is not consulted. The
        // retained state is published under it at completion when submission found it unused.
        std::string state_alias;
        bool publish_state = false;
        DecisionResult result;
    };

    struct Request {
        // A System One decision. It has no routing hint, prompt boundaries or output session. Its
        // state candidates are a lane that retained the same adapter's decision state, planned per
        // lane, and the cached image of exactly its state, looked up under `state_alias` and
        // carried as the stable candidate pool.
        Request(std::uint64_t request_identity, DecisionWork work, Clock::time_point limit,
                Clock::time_point submit_time, CachedContinuation state_continuation)
            : id(request_identity),
              prompt_summary{.prompt_tokens = work.summary.input_tokens(), .has_media = false},
              prepare_seconds(work.prompt.prepare_seconds), deadline(limit),
              submitted(submit_time), stable_continuation(std::move(state_continuation)),
              decision(std::move(work)) {
            options.execution.adapter            = decision->options.adapter;
            options.execution.allow_prefix_reuse = decision->options.allow_prefix_reuse;
            continuation.lookup_microseconds     = stable_continuation.lookup_microseconds;
            if (!decision->options.allow_prefix_reuse) {
                continuation.final_miss_reason = ContinuationMissReason::Disabled;
            } else {
                if (!decision->state_alias.empty()) {
                    continuation.alias_kind = ContinuationAliasKind::StablePrefix;
                }
                continuation.final_miss_reason =
                    stable_continuation.status == cache::CacheLookupStatus::UnavailableOrCorrupt
                        ? ContinuationMissReason::EntryUnavailableOrCorrupt
                        : (!stable_continuation.image && stable_continuation.candidates.empty()
                               ? ContinuationMissReason::NoAlias
                               : ContinuationMissReason::NotAttempted);
            }
        }

        Request(std::uint64_t request_identity, targets::qwen3_8::PreparedPrompt input,
                targets::qwen3_8::OutputSession output_session, PromptSummary summary,
                double frontend_seconds, ResolvedRequestOptions request_options,
                Clock::time_point limit, Clock::time_point submit_time, HostInputLease input_lease,
                 CachedContinuation routed_cached_continuation,
                 CachedContinuation stable_cached_continuation,
                 std::vector<StableBoundary> prompt_boundaries,
                 std::optional<PendingSessionPublication> pending_publication)
            : id(request_identity), host_input(std::move(input_lease)), prompt(std::move(input)),
              output(std::move(output_session)), prompt_summary(summary),
              prepare_seconds(frontend_seconds), options(std::move(request_options)),
              deadline(limit), submitted(submit_time),
              routed_continuation(std::move(routed_cached_continuation)),
              stable_continuation(std::move(stable_cached_continuation)),
              boundaries(std::move(prompt_boundaries)),
              pending_session_publication(std::move(pending_publication)) {
            continuation.lookup_microseconds = routed_continuation.lookup_microseconds +
                                               stable_continuation.lookup_microseconds;
            if (!options.execution.allow_prefix_reuse) {
                continuation.final_miss_reason = ContinuationMissReason::Disabled;
            } else {
                continuation.alias_kind = options.routing_hint
                                              ? ContinuationAliasKind::Session
                                              : (!boundaries.empty()
                                                     ? ContinuationAliasKind::StablePrefix
                                                     : ContinuationAliasKind::None);
                const bool any_candidate =
                    routed_continuation.image || !routed_continuation.candidates.empty() ||
                    stable_continuation.image || !stable_continuation.candidates.empty();
                if (routed_continuation.status == cache::CacheLookupStatus::UnavailableOrCorrupt ||
                    stable_continuation.status == cache::CacheLookupStatus::UnavailableOrCorrupt) {
                    continuation.final_miss_reason =
                        ContinuationMissReason::EntryUnavailableOrCorrupt;
                } else if (!any_candidate) {
                    // No alias this prompt could use names anything, either because the prompt is
                    // unaliasable or because nothing was ever published under the aliases it has.
                    // Both are the same fact for a reader: there was nothing to restore from. A
                    // stable flight that resolves later overrides this with what it observes,
                    // since every concrete candidate outcome outranks NoAlias.
                    continuation.final_miss_reason = ContinuationMissReason::NoAlias;
                } else {
                    // A candidate exists. Until restoration actually evaluates it this is an
                    // absence of evidence, not evidence that nothing was cacheable.
                    continuation.final_miss_reason = ContinuationMissReason::NotAttempted;
                }
            }
        }

        const std::uint64_t id;
        HostInputLease host_input;
        targets::qwen3_8::PreparedPrompt prompt;
        targets::qwen3_8::OutputSession output;
        PromptSummary prompt_summary;
        double prepare_seconds = 0.0;
        ResolvedRequestOptions options;
        Clock::time_point deadline;
        Clock::time_point submitted;
        // Set exactly once, when admission gives this request a lane.
        std::optional<Clock::time_point> admitted;
        double publish_seconds = 0.0;
        std::optional<Clock::time_point> first_token;
        // Written by the worker at each prefill chunk boundary, read by the
        // waiting consumer thread. Relaxed: these only drive progress
        // reporting, and the completion handshake already orders the result.
        std::atomic<std::uint32_t> prefill_processed{0};
        std::atomic<std::uint32_t> prefill_reused{0};
        // Prompt length of the admitted plan; with the two counters above it gives the tokens a
        // prefilling lane still has to consume, which orders chunk issue across lanes.
        std::uint32_t prefill_total_tokens = 0;
        std::optional<GenerationBudget> budget;
        // Absent when the guard is disabled. Trips between rounds rather than inside one, so a
        // confirmed cycle costs at most one further round before the lane is terminated.
        std::optional<RepetitionGuard> repetition;
        std::optional<BeginSummary> begin;
        std::vector<TokenId> generated;
        std::string content;
        std::string reasoning;
        std::optional<std::uint32_t> lane;
        std::atomic<bool> cancelled{false};
        bool decode_ready = false;
        CachedContinuation routed_continuation;
        // Pooled descriptors from every boundary lookup; the restore ranks them as one set.
        CachedContinuation stable_continuation;
        // Ascending by depth.
        std::vector<StableBoundary> boundaries;
        // Deepest boundary this request captured and queued for publication.
        std::uint32_t deepest_published_depth = 0;
        bool continuation_restore_attempted = false;
        // Set when the only thing that stopped a restore was shared-KV capacity. That is a
        // transient condition owned by whichever requests currently hold pages, so the candidate
        // stays live and the restore is retried instead of decaying into a full cold prefill.
        bool continuation_restore_deferred_kv = false;
        std::optional<PendingSessionPublication> pending_session_publication;
        bool session_lookup_deferred = false;
        std::uint64_t continuation_preflight_operations = 0;
        std::uint64_t continuation_l2_restore_microseconds = 0;
        std::uint64_t continuation_l2_restore_operations = 0;
        std::uint64_t continuation_l3_restore_microseconds = 0;
        std::uint64_t continuation_l3_restore_operations = 0;
        ContinuationDiagnostics continuation;
        // Set for a System One decision, whose prompt/output/budget fields stay empty.
        std::optional<DecisionWork> decision;
        // Set the first time admission found every LoRA bank slot pinned by another adapter.
        bool adapter_slot_waited = false;

        std::optional<BasePlan> base_plan;
        std::array<std::optional<Plan>, kMaximumConcurrency> lane_plans{};
        std::array<std::uint64_t, kMaximumConcurrency> lane_plan_versions{};
        AdmissionResources admission_resources;
        std::uint64_t remaining_service_work = 0;
        std::uint64_t backfill_epoch         = 0;
        BackfillClass backfill_class         = BackfillClass::None;

        std::mutex mutex;
        std::condition_variable cv;
        std::vector<OutputDelta> events;
        GenerationResult result;
        std::exception_ptr error;
        bool done              = false;
        bool consumer_released = false;
        bool capacity_released = false;
    };

    struct RoundMembership {
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<RoundBudget, kMaximumConcurrency> budgets{};
        std::size_t size = 0;

        [[nodiscard]] bool empty() const noexcept { return size == 0; }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept {
            return {lanes.data(), size};
        }

        [[nodiscard]] std::span<const RoundBudget> budget_span() const noexcept {
            return {budgets.data(), size};
        }
    };

    struct ActiveAdmissionSet {
        std::array<ActiveAdmissionSnapshot, kMaximumConcurrency> requests{};
        std::size_t size = 0;

        [[nodiscard]] std::span<const ActiveAdmissionSnapshot> span() const noexcept {
            return {requests.data(), size};
        }
    };

    enum class AdmissionProgress : std::uint8_t {
        None,
        ControlProgress,
        RanGpuUnit,
    };

    struct LaneChoice {
        std::uint32_t lane  = 0;
        bool evict_retained = false;
    };

    void append_output(const std::shared_ptr<Request>& request,
                       targets::qwen3_8::PublishedOutput output) {
        if (output.empty()) { return; }
        {
            std::lock_guard lock(request->mutex);
            for (OutputDelta& delta : output) {
                std::string& full = delta.channel == OutputChannel::Reasoning ? request->reasoning
                                                                              : request->content;
                full += delta.text;
                request->events.push_back(std::move(delta));
            }
        }
        request->cv.notify_one();
    }

    // Defaults an unset pending deadline to the Engine timeout and refuses one already past.
    [[nodiscard]] Clock::time_point resolve_pending_deadline(Clock::time_point submitted,
                                                             Clock::time_point pending_deadline) {
        if (pending_deadline == Clock::time_point{}) {
            pending_deadline = submitted + pending_timeout_;
        }
        if (submitted >= pending_deadline) {
            // A refused request never reaches the request log, so ingress rejection is only
            // visible if it is counted here.
            continuation_stats_.rejected_queue_timeout.fetch_add(1, std::memory_order_relaxed);
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired before submission");
        }
        return pending_deadline;
    }

    // Claims one place of the bounded ingress and returns the new request's id.
    [[nodiscard]] std::uint64_t claim_outstanding() {
        std::lock_guard lock(queue_mutex_);
        if (stopping_ || failed_) {
            throw RequestError(RequestErrorKind::Unavailable, "inference engine is unavailable");
        }
        if (outstanding_ >= max_outstanding_) {
            continuation_stats_.rejected_overloaded.fetch_add(1, std::memory_order_relaxed);
            throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
        }
        ++outstanding_;
        return next_request_id_++;
    }

    // Queues a request holding an ingress place; a stopped Engine takes the place back.
    void enqueue_pending(const std::shared_ptr<Request>& request) {
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                --outstanding_;
                release_stable_builders(request);
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            pending_.push_back(request);
        }
        queue_cv_.notify_one();
    }

    void release_reserved_capacity() noexcept {
        std::lock_guard lock(queue_mutex_);
        if (outstanding_ != 0) { --outstanding_; }
    }

    void release_consumer(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            request->consumer_released = true;
            if (request->done && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        if (release) { release_reserved_capacity(); }
    }

    void abandon_request(std::shared_ptr<Request> request) noexcept {
        request->cancelled.store(true, std::memory_order_release);
        queue_cv_.notify_one();
        release_consumer(request);
    }

    bool mark_completed(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            if (request->consumer_released && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        return release;
    }

    void release_planning_state(const std::shared_ptr<Request>& request) noexcept {
        request->base_plan.reset();
        for (auto& plan : request->lane_plans) { plan.reset(); }
    }

    // Claims the write-once flight of every publish boundary that still needs one, shallowest
    // first, and stops at the first one another request already builds: the request follows
    // that boundary and re-tries once its builder finishes. Claiming in depth order is what
    // keeps two requests from each following a boundary the other holds - both order shared
    // boundaries identically, since an alias names one exact prefix.
    //
    // With `relookup`, a boundary claimed here is looked up again first: its previous builder
    // may have published since this request last looked, in which case the hit is adopted and
    // the flight released. The submission call skips that because it just looked.
    void acquire_stable_flights(const std::shared_ptr<Request>& request, bool relookup) {
        bool adopted = false;
        for (StableBoundary& boundary : request->boundaries) {
            if (!boundary.flight_needed || boundary.builder) { continue; }
            bool became_builder = false;
            {
                std::lock_guard lock(stable_flight_mutex_);
                became_builder = stable_flights_.acquire(boundary.alias, request->id) ==
                                 StablePrefixFlights::AcquireResult::Builder;
            }
            if (!became_builder) { break; }
            boundary.builder = true;
            if (!relookup) { continue; }
            CachedContinuation lookup = lookup_continuation(
                boundary.alias, ContinuationAliasKind::StablePrefix, false, true);
            request->continuation.lookup_microseconds += lookup.lookup_microseconds;
            if (lookup.candidates.empty()) { continue; }
            merge_stable_lookup(request->stable_continuation, std::move(lookup));
            boundary.flight_needed = false;
            release_stable_builder(request, boundary);
            adopted = true;
        }
        if (adopted) {
            request->continuation_restore_attempted   = false;
            request->continuation_restore_deferred_kv = false;
        }
        // Lane plans carry the capture depths, which follow the builder roles.
        for (auto& plan : request->lane_plans) { plan.reset(); }
    }

    void release_stable_builder(const std::shared_ptr<Request>& request,
                                StableBoundary& boundary) noexcept {
        if (!boundary.builder || boundary.publication_pending) { return; }
        {
            std::lock_guard lock(stable_flight_mutex_);
            (void)stable_flights_.release(boundary.alias, request->id);
        }
        boundary.builder = false;
        queue_cv_.notify_all();
    }

    // Releases every flight this request builds and has not handed to the publication worker.
    void release_stable_builders(const std::shared_ptr<Request>& request) noexcept {
        if (!request) { return; }
        for (StableBoundary& boundary : request->boundaries) {
            release_stable_builder(request, boundary);
        }
    }

    void refresh_stable_flights() {
        for (const auto& request : pending_snapshot()) {
            if (stable_flight_blocked(*request)) { acquire_stable_flights(request, true); }
        }
    }

    // A request waits while any publish boundary it needs is being built by another request.
    [[nodiscard]] static bool stable_flight_blocked(const Request& request) noexcept {
        return std::ranges::any_of(request.boundaries, [](const StableBoundary& boundary) {
            return boundary.flight_needed && !boundary.builder;
        });
    }

    void complete_error(const std::shared_ptr<Request>& request, std::exception_ptr error) {
        release_stable_builders(request);
        release_planning_state(request);
        request->prompt = {};
        request->host_input.reset();
        {
            std::lock_guard lock(request->mutex);
            if (request->done) { return; }
            request->error = std::move(error);
            request->done  = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_success(const std::shared_ptr<Request>& request, FinishReason reason) {
        release_stable_builders(request);
        release_planning_state(request);
        request->continuation.cache_write_tokens =
            request->deepest_published_depth > request->continuation.restored_tokens
                ? request->deepest_published_depth - request->continuation.restored_tokens
                : 0;
        request->prompt = {};
        request->host_input.reset();
        GenerationResult result;
        result.prompt                  = request->prompt_summary;
        result.generated_token_ids     = std::move(request->generated);
        result.content                 = std::move(request->content);
        result.reasoning               = std::move(request->reasoning);
        result.reasoning_tokens        = request->output.reasoning_tokens();
        result.finish_reason           = reason;
        // Time between the engine accepting the request and the admission that gave it a lane.
        // Under concurrent load this, not prefill, is what TTFT is mostly made of, so it is
        // reported even when the request never reached a lane. It is applied after the lane's
        // timings are copied in, because that copy replaces the whole struct.
        const double queue_seconds =
            request->admitted ? std::chrono::duration<double>(*request->admitted -
                                                              request->submitted)
                                    .count()
                              : std::chrono::duration<double>(Clock::now() - request->submitted)
                                    .count();
        if (request->begin) {
            result.reused_prompt_tokens = request->begin->reused_prompt_tokens;
            result.prefix_reuse_path    = request->begin->prefix_reuse_path;
        }
        if (request->lane) {
            result.timings = instance_.program->generation_timings_lane(*request->lane);
            result.speculative = instance_.program->speculative_stats_lane(*request->lane);
            result.timings.restore_seconds =
                static_cast<double>(request->continuation.restore_microseconds) / 1e6;
            result.slot        = static_cast<std::int32_t>(*request->lane);
            // Empty unless the lane retained the finished session (aborts and cancels clear it).
            result.session_digest = instance_.program->retained_lane_digest(*request->lane);
            // Completion and restore are the only paths that make a lane retained, so keeping
            // the cache here means publish_runtime_stats never has to hash a ledger.
            retained_digest_cache_[*request->lane] = result.session_digest;
            retained_checkpoints_cache_[*request->lane] =
                instance_.program->retained_lane_checkpoints(*request->lane);
        }
        account_continuation(*request);
        result.continuation = request->continuation;
        if (request->first_token) {
            result.timings.first_token_seconds =
                request->prepare_seconds +
                std::chrono::duration<double>(*request->first_token - request->submitted).count();
        }
        result.timings.prepare_seconds = request->prepare_seconds;
        result.timings.queue_seconds   = queue_seconds;
        result.timings.publish_seconds = request->publish_seconds;
        result.timings.total_seconds =
            request->prepare_seconds +
            std::chrono::duration<double>(Clock::now() - request->submitted).count();
        {
            std::lock_guard lock(request->mutex);
            if (request->done) { return; }
            request->result = std::move(result);
            request->done   = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    // Folds one completed request's continuation outcome into the cumulative metrics: the tier a
    // restored state came from, or why none was, and the request's preflight and restore costs.
    // A generation and a decision are accounted alike.
    void account_continuation(Request& request) {
        if (request.continuation.source != ContinuationSource::None) {
            request.continuation.final_miss_reason = ContinuationMissReason::None;
            auto add_tier = [&](std::uint64_t& successes, std::uint64_t& tokens,
                                std::uint64_t& bytes) {
                ++successes;
                tokens += request.continuation.restored_tokens;
                bytes += request.continuation.restored_bytes;
            };
            switch (request.continuation.source) {
            case ContinuationSource::L1:
                add_tier(cumulative_stats_.continuation_l1_restore_successes,
                         cumulative_stats_.continuation_l1_restored_tokens,
                         cumulative_stats_.continuation_l1_restored_bytes);
                break;
            case ContinuationSource::L2:
                add_tier(cumulative_stats_.continuation_l2_restore_successes,
                         cumulative_stats_.continuation_l2_restored_tokens,
                         cumulative_stats_.continuation_l2_restored_bytes);
                break;
            case ContinuationSource::L3:
                add_tier(cumulative_stats_.continuation_l3_restore_successes,
                         cumulative_stats_.continuation_l3_restored_tokens,
                         cumulative_stats_.continuation_l3_restored_bytes);
                break;
            case ContinuationSource::None: break;
            }
            if (request.continuation.alias_kind == ContinuationAliasKind::Session) {
                ++cumulative_stats_.continuation_session_restores;
            } else if (request.continuation.alias_kind == ContinuationAliasKind::StablePrefix) {
                ++cumulative_stats_.continuation_stable_prefix_restores;
            }
        } else {
            switch (request.continuation.final_miss_reason) {
            case ContinuationMissReason::Disabled:
                ++cumulative_stats_.continuation_miss_disabled;
                break;
            case ContinuationMissReason::NoAlias:
                ++cumulative_stats_.continuation_miss_no_alias;
                break;
            case ContinuationMissReason::NotAttempted:
                ++cumulative_stats_.continuation_miss_not_attempted;
                break;
            case ContinuationMissReason::EntryUnavailableOrCorrupt:
                ++cumulative_stats_.continuation_miss_entry_unavailable_or_corrupt;
                break;
            case ContinuationMissReason::NotDeeper:
                ++cumulative_stats_.continuation_miss_not_deeper;
                break;
            case ContinuationMissReason::PreflightRejected:
                ++cumulative_stats_.continuation_miss_preflight_rejected;
                break;
            case ContinuationMissReason::RollbackConflict:
                ++cumulative_stats_.continuation_miss_rollback_conflict;
                break;
            case ContinuationMissReason::NoLane:
                ++cumulative_stats_.continuation_miss_no_lane;
                break;
            case ContinuationMissReason::RestoreFailed:
                ++cumulative_stats_.continuation_miss_restore_failed;
                break;
            case ContinuationMissReason::None: break;
            }
        }
        if (request.continuation_preflight_operations != 0) {
            cumulative_stats_.continuation_preflight_operations +=
                request.continuation_preflight_operations;
            cumulative_stats_.continuation_preflight_microseconds +=
                request.continuation.preflight_microseconds;
        }
        cumulative_stats_.continuation_l2_restore_operations +=
            request.continuation_l2_restore_operations;
        cumulative_stats_.continuation_l2_restore_microseconds +=
            request.continuation_l2_restore_microseconds;
        cumulative_stats_.continuation_l3_restore_operations +=
            request.continuation_l3_restore_operations;
        cumulative_stats_.continuation_l3_restore_microseconds +=
            request.continuation_l3_restore_microseconds;
    }

    void complete_cancelled(const std::shared_ptr<Request>& request) {
        if (request->decision) {
            complete_error(request, std::make_exception_ptr(RequestError(
                                        RequestErrorKind::Cancelled,
                                        "decision was cancelled before it completed")));
            return;
        }
        (void)request->output.preview_terminal(FinishReason::Cancelled);
        append_output(request, request->output.commit_preview());
        complete_success(request, FinishReason::Cancelled);
    }

    void publish_retained_completion(const std::shared_ptr<Request>& request, std::uint32_t lane,
                                     FinishReason reason) noexcept {
        retained_last_used_[lane] = Clock::now();
        lane_provenance_[lane] = completion_publication_provenance(
            lane_provenance_[lane], request->options.routing_hint
                                        ? ContinuationAliasKind::Session
                                        : (!request->boundaries.empty()
                                               ? ContinuationAliasKind::StablePrefix
                                               : request->continuation.alias_kind));
        lane_sessions_[lane].reset();
        if (!continuation_cache_ || !request->options.execution.allow_prefix_reuse ||
            reason == FinishReason::Cancelled) {
            return;
        }
        // A deferred session lookup means the retained lane is newer than the alias that is still
        // being published. Do not block completion or enqueue a child with a stale CAS parent; a
        // later turn can publish the newest retained state after the predecessor becomes visible.
        const bool publish_session =
            request->options.routing_hint && !request->session_lookup_deferred;
        const auto frontier_boundary =
            std::ranges::find_if(request->boundaries, [](const StableBoundary& boundary) {
                return boundary.completion_publish;
            });
        // The boundary alias names the exact prefix through the rewrite frontier, which is the
        // lane's turn checkpoint: a request sharing only that prefix restores the image and
        // rewinds to the checkpoint, and a request sharing the generated turn as well appends at
        // the frontier. One export serves both aliases.
        const bool publish_boundary =
            frontier_boundary != request->boundaries.end() &&
            instance_.program->retained_lane_boundary_tokens(lane) == frontier_boundary->depth;
        if (!publish_session && !publish_boundary) { return; }
        const auto publish_started = Clock::now();
        try {
            // The export itself runs on the publication worker: the lane's device state is
            // final and stays untouched until the worker has copied it out (every path that
            // reuses or releases the lane waits on `wait_lane_export`), so the execution thread
            // only fences the lane and queues the job. The other lanes keep decoding and this
            // request completes without waiting for gigabytes to cross PCIe.
            Publication job;
            job.export_lane = lane;
            if (publish_boundary) {
                job.boundary_alias = frontier_boundary->alias;
                job.boundary_depth = frontier_boundary->depth;
            }
            if (publish_session) {
                lane_sessions_[lane] =
                    LaneSession{.name                = *request->options.routing_hint,
                                .expected_head       = request->routed_continuation.id,
                                .expected_generation = request->routed_continuation.generation};
                job.session             = lane_sessions_[lane]->name;
                job.expected_head       = lane_sessions_[lane]->expected_head;
                job.expected_generation = lane_sessions_[lane]->expected_generation;
            }
            instance_.program->fence_lane_for_export(lane);
            PublicationTicket ticket = enqueue_publication(std::move(job));
            if (ticket) {
                if (publish_session) { lane_sessions_[lane]->publication = ticket; }
                if (publish_boundary) {
                    request->deepest_published_depth =
                        std::max(request->deepest_published_depth, frontier_boundary->depth);
                }
                request->continuation.completion_publication_queued = true;
            } else if (publish_session) {
                lane_sessions_[lane].reset();
            }
        } catch (...) {
            // A continuation is an optimization; generation has already completed successfully.
            lane_sessions_[lane].reset();
        }
        const double publish_seconds =
            std::chrono::duration<double>(Clock::now() - publish_started).count();
        request->publish_seconds += publish_seconds;
        cumulative_stats_.worker_publish_seconds += publish_seconds;
    }

    // Blocks until no background export reads the lane. Every path that mutates or releases a
    // retained lane's device state goes through here first.
    void wait_lane_export(std::uint32_t lane) noexcept {
        if (lane >= kMaximumConcurrency) { return; }
        try {
            std::unique_lock lock(publication_mutex_);
            publication_cv_.wait(lock, [&] {
                return !lane_export_pending_[lane] || publication_stopping_;
            });
        } catch (...) {}
    }

    [[nodiscard]] bool lane_export_in_flight(std::uint32_t lane) noexcept {
        if (lane >= kMaximumConcurrency) { return false; }
        std::lock_guard lock(publication_mutex_);
        return lane_export_pending_[lane];
    }

    // Every involuntary loss of a retained session funnels through here: L1 retention pressure,
    // a continuation restore taking the lane, and a FullReset admission over it. The session is
    // offered to both retention tiers - the continuation cache (L2/L3) and, when the lane is
    // bound to a slot file, that file - and the lane always ends up unbound so a later eviction
    // can never write a new session over a previous session's file. Explicit erase is a deletion
    // request and does not come through here.
    void evict_retained_lane(std::uint32_t lane) noexcept {
        wait_lane_export(lane);
        spill_retained_lane(lane);
        if (lane < kMaximumConcurrency) { lane_session_path_[lane].clear(); }
        if (!instance_.program->has_retained_lane(lane)) {
            lane_sessions_[lane].reset();
            lane_provenance_[lane] = {};
            retained_last_used_[lane].reset();
            return;
        }
        bool demoted = false;
        if (continuation_cache_ && lane_sessions_[lane]) {
            const PublicationStatus status = lane_sessions_[lane]->publication
                                                 ? lane_sessions_[lane]->publication->load(
                                                       std::memory_order_acquire)
                                                 : PublicationStatus::Failed;
            if (status == PublicationStatus::Failed) {
                try {
                    auto image      = instance_.program->export_continuation_lane(lane);
                    image.parent_id = lane_sessions_[lane]->expected_head;
                    lane_sessions_[lane]->publication =
                        queue_publication(std::move(image), lane_sessions_[lane]->name,
                                          lane_sessions_[lane]->expected_head,
                                          lane_sessions_[lane]->expected_generation);
                } catch (...) {
                    // Demotion is best effort, but a failed export is state loss and is counted
                    // as such rather than being silently absorbed into the eviction.
                    continuation_stats_.publication_failures.fetch_add(1,
                                                                       std::memory_order_relaxed);
                    continuation_stats_.publication_failed_error.fetch_add(
                        1, std::memory_order_relaxed);
                }
            }
            demoted = static_cast<bool>(lane_sessions_[lane]->publication);
        }
        instance_.program->evict_retained_lane(lane);
        lane_sessions_[lane].reset();
        lane_provenance_[lane] = {};
        retained_last_used_[lane].reset();
        invalidate_lane_plans(lane);
        if (l1_policy_active_) {
            ++cumulative_stats_.l1_evictions;
            if (demoted) { ++cumulative_stats_.l1_demotions; }
        }
    }

    [[nodiscard]] std::array<L1RetentionEntry, kMaximumConcurrency>
    retention_entries(Clock::time_point now) noexcept {
        std::array<L1RetentionEntry, kMaximumConcurrency> entries{};
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const bool resident = instance_.program->has_retained_lane(lane);
            if (resident && !retained_last_used_[lane]) { retained_last_used_[lane] = now; }
            if (!resident && slots_[lane] == nullptr) { retained_last_used_[lane].reset(); }
            entries[lane] = L1RetentionEntry{
                .lane           = lane,
                .resident_bytes = resident
                                      ? instance_.program->retained_lane_resident_bytes(lane)
                                      : 0,
                .last_used      = retained_last_used_[lane].value_or(now),
                .resident       = resident,
                .active         = slots_[lane] != nullptr,
            };
        }
        return entries;
    }

    // Demote the least recently used retained lane that no request occupies, so its pages return
    // to the shared pool. Unlike the L1 sweep this ignores the byte budget and the idle TTL: the
    // caller has already established that a running lane cannot proceed without those pages.
    // The session is not lost - evict_retained_lane publishes it to L2/L3 on the way out.
    [[nodiscard]] bool spill_lru_retained_lane(std::uint32_t exclude_lane) noexcept {
        try {
            const Clock::time_point now = Clock::now();
            auto entries                = retention_entries(now);
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                if (lane == exclude_lane) { entries[lane].resident = false; }
            }
            const auto victim = select_l1_retention_victim(
                std::span<const L1RetentionEntry>(entries.data(), max_concurrency_), 0,
                Clock::duration::zero(), now);
            if (!victim) { return false; }
            evict_retained_lane(*victim);
            return true;
        } catch (...) { return false; }
    }

    struct Reclaimable {
        std::uint32_t lane = 0;
        Clock::time_point last_used;
    };

    // Whether some idle lane could host a restore of this request right now. Asks the same
    // predicates admission asks, so the answer cannot drift from what admission would decide.
    // Side-effect free apart from lane planning, which admission performs anyway: it gates the
    // deferred retry so a request waiting on capacity does not re-pay lookup and preflight on
    // every admission pass.
    [[nodiscard]] bool restore_could_be_hosted(const std::shared_ptr<Request>& request) {
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) { continue; }
            ensure_lane_plan(request, lane);
            const Plan& plan = *request->lane_plans[lane];
            if (instance_.program->can_admit_lane(lane, plan) ||
                instance_.program->can_admit_lane_after_retained_eviction(lane, plan)) {
                return true;
            }
        }
        return false;
    }

    // Makes `target` able to host the restore, demoting idle retained sessions least-recently used
    // first. Feasibility and sufficiency are both decided by the admission predicates rather than
    // by a second capacity rule: a restore reserves exactly the entitlement a cold admission of the
    // same request would, so anything admission can place, a restore can place. Nothing is evicted
    // unless eviction would actually be enough, because a restore that is going to be refused
    // anyway must not also destroy a healthy session.
    [[nodiscard]] bool prepare_lane_for_restore(std::uint32_t target, const Plan& plan) noexcept {
        const auto fits = [&] { return instance_.program->can_admit_lane(target, plan); };
        if (!fits()) {
            if (!instance_.program->can_admit_lane_after_retained_eviction(target, plan)) {
                return false;
            }
            std::array<Reclaimable, kMaximumConcurrency> victims{};
            std::size_t count = 0;
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                if (lane == target || slots_[lane] != nullptr ||
                    !instance_.program->has_retained_lane(lane)) {
                    continue;
                }
                victims[count++] = Reclaimable{
                    .lane      = lane,
                    .last_used = retained_last_used_[lane].value_or(Clock::time_point{})};
            }
            // Coldest first: the session least likely to be resumed is the cheapest to demote.
            std::sort(victims.begin(), victims.begin() + static_cast<std::ptrdiff_t>(count),
                      [](const Reclaimable& left, const Reclaimable& right) {
                          return left.last_used < right.last_used;
                      });
            for (std::size_t i = 0; i < count && !fits(); ++i) {
                evict_retained_lane(victims[i].lane);
                ++cumulative_stats_.kv_restore_reclaimed_lanes;
            }
            if (!fits()) { return false; }
        }
        // The target's own retained session is what this restore overwrites; release it only now
        // that the lane is known to be usable.
        if (instance_.program->has_retained_lane(target)) { evict_retained_lane(target); }
        return true;
    }

    [[nodiscard]] bool enforce_l1_retention() noexcept {
        if (!l1_policy_active_) { return false; }
        bool changed = false;
        try {
            for (;;) {
                const Clock::time_point now = Clock::now();
                const auto entries          = retention_entries(now);
                const auto victim = select_l1_retention_victim(
                    std::span<const L1RetentionEntry>(entries.data(), max_concurrency_),
                    l1_byte_budget_, l1_idle_ttl_, now);
                if (!victim) { break; }
                evict_retained_lane(*victim);
                changed = true;
            }
        } catch (...) {
            // Retention is an optimization and must never fail active execution or admission.
        }
        return changed;
    }

    // Resolves the single token a prefill licenses. Returns the finish reason when the request
    // ended here; the caller releases the lane and republishes before waking the waiter, so a
    // client never observes its own completion ahead of the slot state that produced it.
    std::optional<FinishReason> resolve_round(const std::shared_ptr<Request>& request,
                                              TokenId token, bool cancel_at_boundary) {
        const std::uint32_t lane = *request->lane;
        if (cancel_at_boundary) {
            (void)request->output.preview_terminal(FinishReason::Cancelled);
            instance_.program->abort_lane(lane);
            lane_sessions_[lane].reset();
            append_output(request, request->output.commit_preview());
            return FinishReason::Cancelled;
        }

        const std::span<const TokenId> tokens(&token, 1);
        const OutputDecision decision = request->output.preview(
            tokens, request->budget->remaining(), request->budget->limit_reason());
        if (decision.accepted_tokens != 1) {
            throw std::logic_error("prefill output policy did not accept its licensed token");
        }
        request->generated.push_back(token);
        instance_.program->resolve_prefill_lane(lane, decision.finished());
        request->budget->commit(1);
        if (request->repetition) { (void)request->repetition->observe(std::span(&token, 1)); }
        auto published = request->output.commit_preview();
        if (!request->first_token) { request->first_token = Clock::now(); }
        append_output(request, std::move(published));
        if (decision.finished()) {
            publish_retained_completion(request, lane, decision.finish_reason);
            return decision.finish_reason;
        }
        return std::nullopt;
    }

    void invalidate_lane_plans(std::uint32_t lane) noexcept { ++lane_plan_versions_[lane]; }

    void remove_completed_slot(std::uint32_t lane) {
        slots_[lane].reset();
        invalidate_lane_plans(lane);
    }

    void consume_service_work(const std::shared_ptr<Request>& request, std::uint64_t work) {
        if (work == 0 || work > request->remaining_service_work) {
            throw std::logic_error("request service projection consumed " + std::to_string(work) +
                                   " quanta with " +
                                   std::to_string(request->remaining_service_work) + " remaining");
        }
        request->remaining_service_work -= work;
    }

    [[nodiscard]] std::array<bool, kMaximumConcurrency> snapshot_cancellations() const noexcept {
        std::array<bool, kMaximumConcurrency> cancelled{};
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                cancelled[lane] = slots_[lane]->cancelled.load(std::memory_order_acquire);
            }
        }
        return cancelled;
    }

    void
    cancel_active_requests(const std::array<bool, kMaximumConcurrency>& cancelled_at_boundary) {
        bool changed = false;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || !cancelled_at_boundary[lane]) { continue; }
            instance_.program->abort_lane(lane);
            lane_sessions_[lane].reset();
            leave_prefill(lane);
            complete_cancelled(request);
            remove_completed_slot(lane);
            changed = true;
        }
        if (changed) { publish_runtime_stats(); }
    }

    [[nodiscard]] bool expire_pending_requests() {
        std::vector<std::shared_ptr<Request>> cancelled;
        std::vector<std::shared_ptr<Request>> expired;
        bool have_pending = false;
        {
            std::lock_guard lock(queue_mutex_);
            const auto now = Clock::now();
            for (auto it = pending_.begin(); it != pending_.end();) {
                if ((*it)->cancelled.load(std::memory_order_acquire)) {
                    cancelled.push_back(*it);
                    it = pending_.erase(it);
                } else if (now >= (*it)->deadline) {
                    expired.push_back(*it);
                    it = pending_.erase(it);
                } else {
                    ++it;
                }
            }
            have_pending = !pending_.empty();
        }
        if (protection_) {
            const auto removed_protected = [&](const std::shared_ptr<Request>& request) {
                return request->id == protection_->head_request_id;
            };
            if (std::any_of(cancelled.begin(), cancelled.end(), removed_protected) ||
                std::any_of(expired.begin(), expired.end(), removed_protected)) {
                protection_.reset();
            }
        }
        for (const auto& request : cancelled) { complete_cancelled(request); }
        for (const auto& request : expired) {
            complete_error(request, std::make_exception_ptr(RequestError(
                                        RequestErrorKind::QueueTimeout,
                                        "inference request expired while waiting for admission")));
        }
        if (!cancelled.empty() || !expired.empty()) { publish_runtime_stats(); }
        return have_pending;
    }

    // A lane is admitted holding a bounded decode window, so the pages for a long generation are
    // acquired here, one step at a time, as the lane actually reaches them. Three rungs, cheapest
    // first: take free pages; reclaim pages from a retained session no request is using; and
    // failing both, end this request at its current length rather than fail it mid-stream.
    [[nodiscard]] RoundMembership build_round_membership() {
        RoundMembership membership;
        std::array<std::shared_ptr<Request>, kMaximumConcurrency> curtailed{};
        std::size_t curtailed_size = 0;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || !request->decode_ready) { continue; }
            if (!request->budget) {
                throw std::logic_error("decode-ready request has no generation budget");
            }
            if (!grow_decode_headroom(lane)) {
                // Copied before the slot is released: curtail_request resets slots_[lane].
                std::shared_ptr<Request> ending = request;
                curtail_request(ending, lane);
                curtailed[curtailed_size++] = std::move(ending);
                continue;
            }
            membership.lanes[membership.size]   = lane;
            membership.budgets[membership.size] = request->budget->round_budget();
            ++membership.size;
        }
        if (curtailed_size != 0) {
            // The lane table already reflects every curtailed completion before its client is
            // woken, matching the ordering a terminal decode round establishes.
            publish_runtime_stats();
            for (std::size_t i = 0; i < curtailed_size; ++i) {
                complete_success(curtailed[i], FinishReason::ContextCapacity);
            }
        }
        return membership;
    }

    // Acquire the pages this lane needs for one more round. Three rungs, cheapest first: take
    // free pages; reclaim pages from a retained session that no request is using; and failing
    // both, report that the lane cannot continue.
    [[nodiscard]] bool grow_decode_headroom(std::uint32_t lane) {
        ++cumulative_stats_.kv_growth_attempts;
        while (!instance_.program->try_grow_decode_headroom(lane)) {
            if (!spill_lru_retained_lane(lane)) { return false; }
            ++cumulative_stats_.kv_growth_forced_spills;
        }
        return true;
    }

    // End a request at its current length because the pool cannot hold another round. The request
    // finishes successfully with `length`, and its session is retained and published, so the next
    // turn of the same conversation still restores instead of prefilling from scratch.
    void curtail_request(const std::shared_ptr<Request>& request, std::uint32_t lane) {
        ++cumulative_stats_.kv_growth_curtailed;
        (void)request->output.preview_terminal(FinishReason::ContextCapacity);
        instance_.program->retire_lane(lane);
        append_output(request, request->output.commit_preview());
        publish_retained_completion(request, lane, FinishReason::ContextCapacity);
        remove_completed_slot(lane);
    }

    [[nodiscard]] ActiveAdmissionSet active_admission_set() const {
        ActiveAdmissionSet active;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr) { continue; }
            if (request->admission_resources.active_lanes == 0 ||
                request->remaining_service_work == 0) {
                throw std::logic_error("active request has no admission accounting");
            }
            active.requests[active.size++] = ActiveAdmissionSnapshot{
                .request_id            = request->id,
                .resources             = request->admission_resources,
                .remaining_work_quanta = request->remaining_service_work,
                .backfill_epoch        = request->backfill_epoch,
                .backfill_class        = request->backfill_class,
            };
        }
        return active;
    }

    // One decision unit ran. A decision ends in its own completion rather than a licensed token:
    // the lane hands back the probabilities and keeps the state [0, Ls) as a retained decision
    // state for the next decision of the same adapter.
    void resolve_decision_step(const std::shared_ptr<Request>& request,
                               const PrefillStepResult& step, bool cancel_at_boundary) {
        if (!request->lane) { throw std::logic_error("decision step has no request lane"); }
        const std::uint32_t lane = *request->lane;
        cumulative_stats_.computed_prefill_tokens += step.processed_prompt_tokens;
        cumulative_stats_.decision_prefill_tokens += step.processed_prompt_tokens;
        request->prefill_processed.fetch_add(step.processed_prompt_tokens,
                                             std::memory_order_relaxed);
        request->prefill_reused.store(step.summary.reused_prompt_tokens,
                                      std::memory_order_relaxed);
        consume_service_work(request, 1);
        if (!step.complete) {
            if (cancel_at_boundary) {
                leave_prefill(lane);
                instance_.program->abort_lane(lane);
                lane_sessions_[lane].reset();
                complete_cancelled(request);
                remove_completed_slot(lane);
            }
            return;
        }

        leave_prefill(lane);
        request->begin = step.summary;
        targets::qwen3_8::DecisionOutcome outcome = instance_.program->take_decision_lane(lane);
        const Clock::time_point completed         = Clock::now();
        DecisionWork& work                        = *request->decision;
        DecisionResult& result                    = work.result;
        result.probabilities                      = std::move(outcome.probabilities);
        result.summary                            = work.summary;
        result.reused_state_tokens                = outcome.reused_state_tokens;
        // Admission classified a resident state as L1 unless a restore had already named the tier
        // it imported the state from.
        result.state_source = outcome.reused_state_tokens == 0 ? ContinuationSource::None
                              : request->continuation.source != ContinuationSource::None
                                  ? request->continuation.source
                                  : ContinuationSource::L1;
        result.branch_passes     = outcome.branch_passes;
        result.long_branch_chunks = outcome.long_branch_chunks;
        result.adapter           = work.adapter;
        result.slot              = static_cast<std::int32_t>(lane);
        const Clock::time_point admitted = request->admitted.value_or(request->submitted);
        result.timings = DecisionTimings{
            .prepare_seconds = request->prepare_seconds,
            .queue_seconds   = std::chrono::duration<double>(admitted - request->submitted).count(),
            .restore_seconds =
                static_cast<double>(request->continuation.restore_microseconds) / 1e6,
            .state_seconds   = outcome.state_seconds,
            .branch_seconds  = outcome.branch_seconds,
            .execution_seconds = std::chrono::duration<double>(completed - admitted).count(),
            .total_seconds =
                request->prepare_seconds +
                std::chrono::duration<double>(completed - request->submitted).count(),
        };
        lane_sessions_[lane].reset();
        if (instance_.program->has_retained_lane(lane)) {
            retained_last_used_[lane]     = completed;
            retained_digest_cache_[lane]  = instance_.program->retained_lane_digest(lane);
            retained_checkpoints_cache_[lane].clear();
            publish_decision_state(request, lane);
        } else {
            retained_digest_cache_[lane].clear();
            retained_checkpoints_cache_[lane].clear();
        }
        account_continuation(*request);
        remove_completed_slot(lane);
        (void)enforce_l1_retention();
        publish_runtime_stats();
        complete_decision(request);
    }

    // The decision counterpart of publish_retained_completion's boundary part. The retained state
    // is published under its exact-state alias when submission found that alias unused, so a
    // state reused from L1 or restored from L2/L3 is republished only if its alias had lapsed.
    // The execution thread fences the lane and queues the export; the publication worker copies
    // it out, and every path that reuses or releases the lane first waits on wait_lane_export.
    void publish_decision_state(const std::shared_ptr<Request>& request,
                                std::uint32_t lane) noexcept {
        const DecisionWork& work = *request->decision;
        lane_provenance_[lane]   = completion_publication_provenance(
            lane_provenance_[lane], work.state_alias.empty() ? ContinuationAliasKind::None
                                                             : ContinuationAliasKind::StablePrefix);
        if (!continuation_cache_ || !request->options.execution.allow_prefix_reuse ||
            !work.publish_state || !instance_.program->retained_lane_holds_decision(lane)) {
            return;
        }
        const auto publish_started = Clock::now();
        try {
            Publication job;
            job.export_lane    = lane;
            job.decision_state = true;
            job.boundary_alias = work.state_alias;
            job.boundary_depth = work.summary.state_tokens;
            instance_.program->fence_lane_for_export(lane);
            if (enqueue_publication(std::move(job))) {
                request->deepest_published_depth = work.summary.state_tokens;
                request->continuation.completion_publication_queued = true;
            }
        } catch (...) {
            // The cached state is an optimization; the decision has already completed.
        }
        const double publish_seconds =
            std::chrono::duration<double>(Clock::now() - publish_started).count();
        request->publish_seconds += publish_seconds;
        cumulative_stats_.worker_publish_seconds += publish_seconds;
    }

    void complete_decision(const std::shared_ptr<Request>& request) {
        release_planning_state(request);
        {
            std::lock_guard lock(request->mutex);
            if (request->done) { return; }
            request->done = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void resolve_prefill_step(const std::shared_ptr<Request>& request,
                              const PrefillStepResult& step, bool cancel_at_boundary) {
        if (request->decision) {
            resolve_decision_step(request, step, cancel_at_boundary);
            return;
        }
        cumulative_stats_.computed_prefill_tokens += step.processed_prompt_tokens;
        request->prefill_processed.fetch_add(step.processed_prompt_tokens,
                                             std::memory_order_relaxed);
        request->prefill_reused.store(step.summary.reused_prompt_tokens,
                                      std::memory_order_relaxed);
        consume_service_work(request, 1);
        if (step.host_input_consumed || step.complete) { request->host_input.reset(); }
        if (continuation_cache_ && request->options.execution.allow_prefix_reuse && request->lane &&
            !request->boundaries.empty()) {
            try {
                for (targets::qwen3_8::CapturedContinuation& captured :
                     instance_.program->take_captured_continuations_lane(*request->lane)) {
                    const auto boundary = std::ranges::find_if(
                        request->boundaries, [&](const StableBoundary& candidate) {
                            return candidate.depth == captured.depth && candidate.builder &&
                                   !candidate.publication_pending;
                        });
                    // The plan captures only at depths this request builds, so an image without
                    // a builder role is a flight released underneath it; dropping it is correct.
                    if (boundary == request->boundaries.end()) { continue; }
                    auto ticket = queue_publication(
                        std::move(captured.image), boundary->alias, std::nullopt, std::nullopt,
                        true, std::pair<std::string, std::uint64_t>{boundary->alias, request->id});
                    if (ticket) {
                        boundary->publication_pending = true;
                        request->deepest_published_depth =
                            std::max(request->deepest_published_depth, captured.depth);
                        request->continuation.completion_publication_queued = true;
                    } else {
                        release_stable_builder(request, *boundary);
                    }
                }
            } catch (...) { release_stable_builders(request); }
        }
        if (step.complete) { release_stable_builders(request); }
        if (cancel_at_boundary) {
            if (!request->lane) { throw std::logic_error("cancelled prefill has no request lane"); }
            const std::uint32_t lane = *request->lane;
            leave_prefill(lane);
            instance_.program->abort_lane(lane);
            lane_sessions_[lane].reset();
            complete_cancelled(request);
            remove_completed_slot(lane);
            return;
        }
        if (!step.complete) { return; }
        if (!request->lane) { throw std::logic_error("completed prefill has no request lane"); }
        leave_prefill(*request->lane);
        request->begin = step.summary;
        if (step.round.tokens.size() != 1) {
            throw std::logic_error("prefill did not license exactly one token");
        }
        if (const std::optional<FinishReason> reason =
                resolve_round(request, step.round.tokens.front(), false)) {
            remove_completed_slot(*request->lane);
            (void)enforce_l1_retention();
            publish_runtime_stats();
            complete_success(request, *reason);
        } else {
            request->decode_ready = true;
        }
    }

    // One instantaneous board-power read, or nullopt where the board exposes none. Measured at
    // ~0.5 us mean / 0.8 us p99 on an RTX 4090, against execution units that cost tens of
    // milliseconds, so bracketing every unit with two reads is free at this scale.
    [[nodiscard]] std::optional<double> sample_board_watts() const {
        return power_meter_.instant_watts();
    }

    // Charge [started, ended] to a phase accumulator. Both endpoint samples are boundary reads and
    // units run back to back, so the trapezoid rule integrates continuously across a busy period:
    // the closing sample of one unit is taken at essentially the same moment as the opening sample
    // of the next. Time outside a unit bracket is deliberately not charged to any phase.
    void accumulate_phase_energy(double& accumulator, const std::optional<double>& opening_watts,
                                 const std::optional<double>& closing_watts,
                                 Clock::time_point started, Clock::time_point ended) {
        if (!opening_watts || !closing_watts) { return; }
        const double seconds = std::chrono::duration<double>(ended - started).count();
        if (seconds <= 0.0) { return; }
        accumulator += 0.5 * (*opening_watts + *closing_watts) * seconds;
        cumulative_stats_.energy_accounted_seconds += seconds;
        ++cumulative_stats_.energy_samples;
    }

    void timed_decode_round(const RoundMembership& membership) {
        const auto started      = Clock::now();
        const auto opening_watts = sample_board_watts();
        run_decode_round(membership);
        const auto ended         = Clock::now();
        const auto closing_watts = sample_board_watts();
        cumulative_stats_.worker_decode_seconds +=
            std::chrono::duration<double>(ended - started).count();
        ++cumulative_stats_.worker_decode_rounds;
        accumulate_phase_energy(cumulative_stats_.decode_energy_joules, opening_watts,
                                closing_watts, started, ended);
    }

    void timed_prefill_step() {
        const auto started       = Clock::now();
        const auto opening_watts = sample_board_watts();
        run_prefill_step();
        const auto ended         = Clock::now();
        const auto closing_watts = sample_board_watts();
        cumulative_stats_.worker_prefill_seconds +=
            std::chrono::duration<double>(ended - started).count();
        ++cumulative_stats_.worker_prefill_steps;
        accumulate_phase_energy(cumulative_stats_.prefill_energy_joules, opening_watts,
                                closing_watts, started, ended);
    }

    [[nodiscard]] bool lane_is_prefilling(std::uint32_t lane) const noexcept {
        return std::ranges::find(prefill_lanes_, lane) != prefill_lanes_.end();
    }

    void enter_prefill(std::uint32_t lane, bool holds_transient) {
        if (!lane_is_prefilling(lane)) { prefill_lanes_.push_back(lane); }
        if (holds_transient) { transient_owner_ = lane; }
    }

    void leave_prefill(std::uint32_t lane) noexcept {
        std::erase(prefill_lanes_, lane);
        if (transient_owner_ && *transient_owner_ == lane) {
            instance_.request_memory.deactivate();
            transient_owner_.reset();
        }
    }

    void clear_prefill_lanes() noexcept {
        prefill_lanes_.clear();
        if (transient_owner_) {
            instance_.request_memory.deactivate();
            transient_owner_.reset();
        }
    }

    [[nodiscard]] static std::uint32_t remaining_prefill_tokens(const Request& request) noexcept {
        const std::uint32_t consumed = request.prefill_reused.load(std::memory_order_relaxed) +
                                       request.prefill_processed.load(std::memory_order_relaxed);
        return request.prefill_total_tokens > consumed ? request.prefill_total_tokens - consumed
                                                       : 0;
    }

    // Shortest remaining prompt first, admission order on ties. Chunks are the unit, so this is
    // preemption at chunk granularity: a lane with a few hundred tokens left finishes within a
    // couple of units even while another lane still has a hundred thousand to consume, which is
    // the order that minimises mean time to first token for the set.
    [[nodiscard]] std::uint32_t select_prefill_lane() const {
        if (prefill_lanes_.empty()) { throw std::logic_error("no request owns staged prefill"); }
        std::uint32_t chosen        = prefill_lanes_.front();
        std::uint32_t chosen_remain = std::numeric_limits<std::uint32_t>::max();
        for (const std::uint32_t lane : prefill_lanes_) {
            const auto& request = slots_[lane];
            if (request == nullptr || request->decode_ready) {
                throw std::logic_error("staged prefill lane has invalid request state");
            }
            const std::uint32_t remain = remaining_prefill_tokens(*request);
            if (remain < chosen_remain) {
                chosen        = lane;
                chosen_remain = remain;
            }
        }
        return chosen;
    }

    void run_prefill_step() {
        const std::uint32_t lane = select_prefill_lane();
        const auto request       = slots_[lane];
        if (request == nullptr || request->decode_ready) {
            throw std::logic_error("staged prefill lane has invalid request state");
        }
        const auto unit_started       = Clock::now();
        const PrefillStepResult step  = instance_.program->advance_prefill_lane(lane);
        cumulative_stats_.prefill_seconds_total +=
            std::chrono::duration<double>(Clock::now() - unit_started).count();
        const bool cancel_at_boundary = request->cancelled.load(std::memory_order_acquire);
        resolve_prefill_step(request, step, cancel_at_boundary);
        publish_runtime_stats();
    }

    [[nodiscard]] std::vector<std::shared_ptr<Request>> pending_snapshot() const {
        std::lock_guard lock(queue_mutex_);
        return {pending_.begin(), pending_.end()};
    }

    [[nodiscard]] bool erase_pending(const std::shared_ptr<Request>& request) {
        std::lock_guard lock(queue_mutex_);
        const auto it = std::find(pending_.begin(), pending_.end(), request);
        if (it == pending_.end()) { return false; }
        pending_.erase(it);
        return true;
    }

    void clear_protection_if_head(const std::shared_ptr<Request>& request) noexcept {
        if (protection_ && protection_->head_request_id == request->id) { protection_.reset(); }
    }

    // Depths the request captures during prefill: the publish boundaries it builds.
    [[nodiscard]] static std::vector<std::uint32_t> capture_depths(const Request& request) {
        std::vector<std::uint32_t> depths;
        for (const StableBoundary& boundary : request.boundaries) {
            if (boundary.builder && !boundary.publication_pending) {
                depths.push_back(boundary.depth);
            }
        }
        return depths;
    }

    void ensure_base_plan(const std::shared_ptr<Request>& request) {
        if (!request->base_plan) {
            request->base_plan.emplace(
                request->decision
                    ? instance_.program->plan_decision_base(request->decision->prompt,
                                                            request->decision->options)
                    : instance_.program->plan_request_base(request->prompt,
                                                           request->options.execution));
        }
        const RequestPlanSummary& summary = request->base_plan->summary();
        if (summary.admission.active_lanes != 1 || summary.service_work_quanta == 0) {
            throw std::logic_error("target request plan has invalid admission accounting");
        }
    }

    void ensure_lane_plan(const std::shared_ptr<Request>& request, std::uint32_t lane) {
        if (slots_[lane] != nullptr) { return; }
        if (request->lane_plan_versions[lane] == lane_plan_versions_[lane] &&
            request->lane_plans[lane]) {
            return;
        }
        request->lane_plans[lane].reset();
        request->lane_plans[lane].emplace(
            request->decision
                ? instance_.program->plan_decision_for_lane(lane, request->decision->prompt,
                                                            *request->base_plan)
                : instance_.program->plan_request_for_lane(lane, request->prompt,
                                                           *request->base_plan,
                                                           capture_depths(*request)));
        request->lane_plan_versions[lane] = lane_plan_versions_[lane];
    }

    // Settles the request's adapter into a device slot after capacity and lane preflight but before
    // continuation restore. Restore writes the adapter identity onto its lane, while residency can
    // evict soft-retained lanes, so both sides of that ordering matter.
    //
    // Staging is a bounded host pass over one slab plus one contiguous upload, on the worker
    // thread between rounds with no graph replay in flight. The bank's device addresses are fixed
    // at load, so only its bytes change and the captured graph is untouched.
    //
    // False means every slot is held by a generating lane using another adapter. That is a
    // temporal block like a full KV pool: the request stays queued until a lane frees, subject to
    // its own deadline. It cannot deadlock, because an idle engine holds no slot.
    [[nodiscard]] bool ensure_adapter_resident(const std::shared_ptr<Request>& request) {
        PhaseTimer timer(cumulative_stats_.worker_admission_plan_seconds);
        const bool resident = instance_.program->ensure_adapter_resident(
            request->options.execution.adapter,
            [this](std::uint32_t lane) { evict_retained_lane(lane); });
        if (!resident && !request->adapter_slot_waited) {
            request->adapter_slot_waited = true;
            ++cumulative_stats_.lora_slot_waits;
        }
        return resident;
    }

    void try_restore_continuation(const std::shared_ptr<Request>& request) noexcept {
        if (request->continuation_restore_attempted && !request->continuation_restore_deferred_kv) {
            return;
        }
        // A decision runs the stable-candidate path below with its cached exact state as the only
        // candidate: it has no routed session, and lane planning ranks its retained L1 states, so
        // an image is restored only when no idle lane already holds the whole state. The target
        // calls it makes are the decision-state kind's.
        if (request->continuation_restore_deferred_kv) {
            // Re-resolving a candidate costs a lookup and a preflight, so only pay it once the
            // reservation could actually be satisfied. The restore reserves exactly the pages a
            // cold admission would, which makes this probe independent of which candidate wins.
            if (!request->base_plan) { return; }
            ++request->continuation.restore_gate_checks;
            if (!restore_could_be_hosted(request)) { return; }
            ++request->continuation.restore_gate_passes;
            // Deliberately not cleared here. `continuation_restore_attempted` is already set, so
            // clearing on the way in would make any later non-restoring outcome terminal. Only a
            // restore or an exhausted candidate set ends the deferral.
        }

        try {
            std::optional<std::uint32_t> target_lane;
            std::uint32_t target_reuse        = std::numeric_limits<std::uint32_t>::max();
            int target_rank                   = 0;
            std::uint32_t best_resident_reuse = 0;
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                if (slots_[lane] != nullptr) { continue; }
                ensure_lane_plan(request, lane);
                const std::uint32_t reuse =
                    request->lane_plans[lane]->summary().reusable_prompt_tokens;
                best_resident_reuse = std::max(best_resident_reuse, reuse);
                // Prefer an empty lane, then a settled retained lane, then one whose session is
                // still being exported; within a class replace the retained lane least useful to
                // this request. This is what makes two sessions switch through a single active
                // lane.
                const auto rank = [&](std::uint32_t candidate) -> int {
                    if (!instance_.program->has_retained_lane(candidate)) { return 0; }
                    return lane_export_in_flight(candidate) ? 2 : 1;
                };
                const int lane_rank = rank(lane);
                if (!target_lane || lane_rank < target_rank ||
                    (lane_rank == target_rank && reuse < target_reuse)) {
                    target_lane  = lane;
                    target_reuse = reuse;
                    target_rank  = lane_rank;
                }
            }
            if (!target_lane) {
                // Lane pressure is transient. Leave the candidate and preflight state untouched so
                // restoration is retried when admission next observes an idle lane. It is still
                // recorded, because a request that never got a lane is not the same miss as one
                // whose alias held nothing.
                observe_continuation_miss(request->continuation, ContinuationMissReason::NoLane);
                return;
            }

            if (request->pending_session_publication) {
                if (publication_completed(request->pending_session_publication->sequence)) {
                    if (request->options.routing_hint) {
                        CachedContinuation refreshed = lookup_continuation(
                            *request->options.routing_hint, ContinuationAliasKind::Session, false,
                            true);
                        request->continuation.lookup_microseconds +=
                            refreshed.lookup_microseconds;
                        request->routed_continuation = std::move(refreshed);
                    }
                    request->pending_session_publication.reset();
                } else {
                    // Use the deepest already-visible L1/L2/L3 checkpoint instead of waiting for
                    // publication. Completion skips its own publication so the pending predecessor
                    // remains the only writer that can advance this session generation.
                    request->session_lookup_deferred = true;
                    request->pending_session_publication.reset();
                }
            }

            if (!request->routed_continuation.image &&
                request->routed_continuation.candidates.empty() &&
                !request->stable_continuation.image &&
                request->stable_continuation.candidates.empty()) {
                request->continuation_restore_attempted   = true;
                request->continuation_restore_deferred_kv = false;
                return;
            }

            const std::uint32_t lane = *target_lane;
            request->continuation.restore_target_lane = static_cast<std::int32_t>(lane);
            const auto preflight_depth = [&](const cache::ContinuationImage& image,
                                             ContinuationAliasKind alias_kind) {
                const auto started = Clock::now();
                ++request->continuation_preflight_operations;
                std::uint32_t divergence = 0;
                const std::uint32_t depth =
                    request->decision
                        ? instance_.program->preflight_decision_state(
                              image, request->decision->prompt, request->options.execution.adapter,
                              &divergence)
                        : instance_.program->preflight_continuation(image, request->prompt,
                                                                    &divergence);
                request->continuation.preflight_microseconds += static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - started)
                        .count());
                request->continuation.candidate_agreement_observed = true;
                request->continuation.deepest_candidate_agreement =
                    std::max(request->continuation.deepest_candidate_agreement,
                             static_cast<std::uint64_t>(divergence));
                if (depth == 0) {
                    continuation_stats_.preflight_rejections.fetch_add(1,
                                                                        std::memory_order_relaxed);
                    observe_continuation_miss(request->continuation,
                                              ContinuationMissReason::PreflightRejected,
                                              alias_kind);
                }
                return depth;
            };
            // Metadata preflight compares 32-byte prefix digests carried by the manifest. It is
            // a negative filter and an upper bound on reusable depth; it never authorizes an
            // import, and it costs no chunk I/O.
            const auto metadata_preflight_depth = [&](
                const cache::SessionCandidateDescriptor& item,
                ContinuationAliasKind alias_kind) {
                const auto started = Clock::now();
                ++request->continuation_preflight_operations;
                const std::uint32_t depth =
                    request->decision
                        ? instance_.program->preflight_decision_state_metadata(
                              item, request->decision->prompt)
                        : instance_.program->preflight_continuation_metadata(item,
                                                                             request->prompt);
                request->continuation.preflight_microseconds += static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - started)
                        .count());
                if (depth == 0 && item.status == cache::CacheLookupStatus::Hit) {
                    continuation_stats_.preflight_rejections.fetch_add(1,
                                                                        std::memory_order_relaxed);
                    observe_continuation_miss(request->continuation,
                                              ContinuationMissReason::PreflightRejected,
                                              alias_kind);
                }
                return depth;
            };
            const auto try_candidate = [&](CachedContinuation& candidate,
                                           std::uint32_t reusable_depth = 0) {
                if (!candidate.image) {
                    if (candidate.status == cache::CacheLookupStatus::UnavailableOrCorrupt) {
                        observe_continuation_miss(
                            request->continuation,
                            ContinuationMissReason::EntryUnavailableOrCorrupt,
                            candidate.alias_kind);
                    }
                    return false;
                }
                if (candidate.image->frontier_tokens <= best_resident_reuse) {
                    observe_continuation_miss(request->continuation,
                                              ContinuationMissReason::NotDeeper,
                                              candidate.alias_kind);
                    candidate.image.reset();
                    return false;
                }
                if (reusable_depth == 0) {
                    reusable_depth = preflight_depth(*candidate.image, candidate.alias_kind);
                    if (reusable_depth == 0) {
                        candidate.image.reset();
                        return false;
                    }
                }
                if (reusable_depth <= best_resident_reuse) {
                    observe_continuation_miss(request->continuation,
                                              ContinuationMissReason::NotDeeper,
                                              candidate.alias_kind);
                    candidate.image.reset();
                    return false;
                }
                // The restore takes the lane plan's entitlement, which already covers the whole
                // prompt plus its decode window and so dominates the restored frontier. That is
                // exactly what a cold admission of this request would reserve, which is why the
                // admission predicates are the right authority here: refusing a restore does not
                // avoid the pages, it only trades a cheap import for a full cold prefill of the
                // same size. The lane's retained session is released only once the lane is known
                // to be usable, so a refused restore never also destroys a healthy session.
                const runtime::KvPageFootprint required =
                    continuation_image_kv_footprint(*candidate.image, request);
                if (!prepare_lane_for_restore(lane, *request->lane_plans[lane])) {
                    // Capacity, not content: this candidate still matches. Keeping it live for a
                    // later retry costs nothing, because admission cannot place the request either
                    // while the pool is this full.
                    continuation_stats_.restore_deferrals.fetch_add(1, std::memory_order_relaxed);
                    request->continuation.restore_failure =
                        ContinuationRestoreFailure::KvReservationExhausted;
                    request->continuation_restore_deferred_kv = true;
                    observe_continuation_miss(request->continuation,
                                              ContinuationMissReason::RestoreFailed,
                                              candidate.alias_kind);
                    return false;
                }
                const std::uint64_t restored_bytes =
                    cache::continuation_image_bytes(*candidate.image);
                const auto restore_started = Clock::now();
                // Decoding is the largest CPU term in a restore. The preparation thread decodes
                // ahead for the queue head, so the common path here is a move; decoding inline is
                // the same work in the same order, just charged to the executor.
                std::shared_ptr<targets::qwen3_8::DecodedContinuation> decoded =
                    take_prepared_continuation(candidate.id);
                if (!decoded) {
                    continuation_preparation_inline_.fetch_add(1, std::memory_order_relaxed);
                    decoded = request->decision
                                  ? instance_.program->decode_decision_state(*candidate.image)
                                  : instance_.program->decode_continuation(*candidate.image);
                }
                const ContinuationRestoreFailure failure =
                    request->decision
                        ? instance_.program->import_decision_state_lane(
                              lane, *candidate.image, *decoded, request->decision->prompt,
                              request->options.execution.adapter, required)
                        : instance_.program->import_continuation_lane(
                              lane, *candidate.image, *decoded, request->prompt,
                              request->options.execution.adapter, required);
                const bool restored = failure == ContinuationRestoreFailure::None;
                const std::uint64_t restore_microseconds = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() -
                                                                          restore_started)
                        .count());
                request->continuation.restore_microseconds += restore_microseconds;
                if (candidate.source == cache::CacheSource::L3) {
                    ++request->continuation_l3_restore_operations;
                    request->continuation_l3_restore_microseconds += restore_microseconds;
                } else {
                    ++request->continuation_l2_restore_operations;
                    request->continuation_l2_restore_microseconds += restore_microseconds;
                }
                if (restored) {
                    retained_last_used_[lane] = Clock::now();
                    lane_provenance_[lane] =
                        imported_lane_provenance(candidate.alias_kind, candidate.source);
                    request->continuation.source = candidate.source == cache::CacheSource::L3
                                                       ? ContinuationSource::L3
                                                       : ContinuationSource::L2;
                    request->continuation.alias_kind = candidate.alias_kind;
                    request->continuation.restored_tokens = reusable_depth;
                    request->continuation.restored_bytes = restored_bytes;
                    request->continuation.final_miss_reason = ContinuationMissReason::None;
                    request->continuation.restore_failure = ContinuationRestoreFailure::None;
                    request->continuation_restore_deferred_kv = false;
                } else {
                    continuation_stats_.restore_failures.fetch_add(1, std::memory_order_relaxed);
                    record_restore_failure(failure);
                    request->continuation.restore_failure = failure;
                    observe_continuation_miss(request->continuation,
                                              ContinuationMissReason::RestoreFailed,
                                              candidate.alias_kind);
                }
                candidate.image.reset();
                invalidate_lane_plans(lane);
                return restored;
            };
            request->continuation_restore_attempted = true;
            bool routed_ready = false;
            std::uint32_t routed_reusable_depth = 0;

            // Boundary candidates are evaluated in two steps. Their descriptors give a
            // metadata-only upper bound on reusable depth at no I/O cost; an image is
            // materialized only when its bound could still win, deepest first, and the first
            // one that verifies is the stable candidate. A prefix that lives in L3 costs a full
            // disk read plus verification, and paying that to lose to a routed candidate was the
            // single largest avoidable component of TTFT.
            std::uint32_t stable_bound = 0;
            std::vector<SelectedContinuationCandidate> stable_viable;
            if (!request->stable_continuation.candidates.empty()) {
                stable_viable = rank_reusable_candidate_descriptors(
                    std::span<const cache::SessionCandidateDescriptor>(
                        request->stable_continuation.candidates),
                    best_resident_reuse, 0, [&](const auto& item) {
                        return metadata_preflight_depth(item,
                                                        ContinuationAliasKind::StablePrefix);
                    });
                if (!stable_viable.empty()) {
                    stable_bound =
                        static_cast<std::uint32_t>(stable_viable.front().reusable_depth);
                }
            } else if (request->stable_continuation.image) {
                stable_bound = request->stable_continuation.image->frontier_tokens >
                                       best_resident_reuse
                                   ? static_cast<std::uint32_t>(
                                         request->stable_continuation.image->frontier_tokens)
                                   : 0;
            }
            std::optional<std::uint32_t> stable_exact_depth;
            const auto stable_reusable_depth = [&]() -> std::uint32_t {
                if (stable_exact_depth) { return *stable_exact_depth; }
                stable_exact_depth = 0;
                if (stable_bound == 0) { return 0; }
                if (request->stable_continuation.image) {
                    if (request->stable_continuation.image->frontier_tokens >
                        best_resident_reuse) {
                        stable_exact_depth = preflight_depth(*request->stable_continuation.image,
                                                              ContinuationAliasKind::StablePrefix);
                    }
                    return *stable_exact_depth;
                }
                for (const auto& viable_item : stable_viable) {
                    const cache::SessionCandidateDescriptor& descriptor =
                        request->stable_continuation.candidates[viable_item.index];
                    auto lookup = continuation_cache_->resolve_candidate(descriptor);
                    account_candidate_lookup(request, lookup);
                    if (!lookup.image) {
                        if (lookup.status == cache::CacheLookupStatus::UnavailableOrCorrupt) {
                            observe_continuation_miss(
                                request->continuation,
                                ContinuationMissReason::EntryUnavailableOrCorrupt,
                                ContinuationAliasKind::StablePrefix);
                        }
                        continue;
                    }
                    const std::uint32_t exact =
                        preflight_depth(*lookup.image, ContinuationAliasKind::StablePrefix);
                    if (exact <= best_resident_reuse) { continue; }
                    // Carry the content identity with the image: it is what binds a resolved
                    // image to work done for it elsewhere, such as a prepared decode.
                    request->stable_continuation.id     = descriptor.id;
                    request->stable_continuation.image  = std::move(lookup.image);
                    request->stable_continuation.source = lookup.source;
                    request->stable_continuation.status = lookup.status;
                    stable_exact_depth                  = exact;
                    break;
                }
                request->stable_continuation.candidates.clear();
                return *stable_exact_depth;
            };
            // A routed depth that already reaches the stable prefix's upper bound wins without
            // the stable image ever being materialized.
            const auto routed_wins = [&](std::uint32_t routed_depth) {
                if (routed_depth == 0) { return false; }
                if (routed_depth >= stable_bound) { return true; }
                return prefer_routed_candidate(routed_depth, stable_reusable_depth());
            };

            if (!request->routed_continuation.candidates.empty()) {
                bool unavailable = false;
                bool available = false;
                for (const auto& item : request->routed_continuation.candidates) {
                    unavailable |=
                        item.status == cache::CacheLookupStatus::UnavailableOrCorrupt;
                    available |= item.status == cache::CacheLookupStatus::Hit;
                }
                // No stable floor is applied here: ranking uses only what the lane already
                // holds, and the stable comparison happens once against the winner, so a routed
                // candidate is never discarded against an upper bound.
                const auto viable = rank_reusable_candidate_descriptors(
                    std::span<const cache::SessionCandidateDescriptor>(
                        request->routed_continuation.candidates),
                    best_resident_reuse, 0,
                    [&](const auto& item) {
                        return metadata_preflight_depth(item, ContinuationAliasKind::Session);
                    });

                std::optional<std::size_t> selected_index;
                std::uint32_t selected_depth = 0;
                bool resolved_image = false;
                for (const auto& viable_item : viable) {
                    if (selected_index && viable_item.reusable_depth <= selected_depth) break;
                    const auto& descriptor =
                        request->routed_continuation.candidates[viable_item.index];
                    auto lookup = continuation_cache_->resolve_candidate(descriptor);
                    account_candidate_lookup(request, lookup);
                    if (!lookup.image) {
                        unavailable |=
                            lookup.status == cache::CacheLookupStatus::UnavailableOrCorrupt;
                        continue;
                    }
                    resolved_image = true;
                    const std::uint32_t exact =
                        preflight_depth(*lookup.image, ContinuationAliasKind::Session);
                    if (exact <= best_resident_reuse ||
                        (selected_index && exact <= selected_depth)) {
                        continue;
                    }
                    selected_index = viable_item.index;
                    selected_depth = exact;
                    request->routed_continuation.id     = descriptor.id;
                    request->routed_continuation.image  = std::move(lookup.image);
                    request->routed_continuation.source = lookup.source;
                    request->routed_continuation.status = lookup.status;
                }
                if (selected_index && routed_wins(selected_depth)) {
                    const auto& chosen =
                        request->routed_continuation.candidates[*selected_index];
                    if (*selected_index == 0) {
                        routed_ready = true;
                    } else if (request->options.routing_hint &&
                               request->routed_continuation.generation) {
                        const auto rollback = continuation_cache_->rollback_session_to(
                            *request->options.routing_hint, chosen.id,
                            *request->routed_continuation.generation);
                        if (rollback.rolled_back) {
                            request->routed_continuation.generation = rollback.generation;
                            request->continuation.destructive_rollback = true;
                            routed_ready = true;
                        } else {
                            observe_continuation_miss(request->continuation,
                                                      ContinuationMissReason::RollbackConflict,
                                                      ContinuationAliasKind::Session);
                        }
                    }
                    if (routed_ready) {
                        routed_reusable_depth = selected_depth;
                        request->routed_continuation.id    = chosen.id;
                    }
                } else {
                    if (unavailable && (!available || (!viable.empty() && !resolved_image))) {
                        observe_continuation_miss(
                            request->continuation,
                            ContinuationMissReason::EntryUnavailableOrCorrupt,
                            ContinuationAliasKind::Session);
                    } else if (available) {
                        observe_continuation_miss(request->continuation,
                                                  ContinuationMissReason::NotDeeper,
                                                  ContinuationAliasKind::Session);
                    }
                }
                request->routed_continuation.candidates.clear();
            } else {
                routed_ready = static_cast<bool>(request->routed_continuation.image);
                if (routed_ready) {
                    routed_reusable_depth = preflight_depth(
                        *request->routed_continuation.image, ContinuationAliasKind::Session);
                    routed_ready = routed_reusable_depth > best_resident_reuse &&
                                   routed_wins(routed_reusable_depth);
                }
            }
            if (!routed_ready ||
                !try_candidate(request->routed_continuation, routed_reusable_depth)) {
                const std::uint32_t stable_depth = stable_reusable_depth();
                if (stable_depth > best_resident_reuse) {
                    (void)try_candidate(request->stable_continuation, stable_depth);
                }
            }
            request->routed_continuation.image.reset();
            request->routed_continuation.candidates.clear();
            request->stable_continuation.image.reset();
            request->stable_continuation.candidates.clear();
            return;
        } catch (...) {
            request->continuation_restore_attempted = true;
            continuation_stats_.restore_failures.fetch_add(1, std::memory_order_relaxed);
            record_restore_failure(ContinuationRestoreFailure::DecodeFailed);
            request->continuation.restore_failure = ContinuationRestoreFailure::DecodeFailed;
            observe_continuation_miss(request->continuation,
                                      ContinuationMissReason::RestoreFailed);
            request->routed_continuation.image.reset();
            request->routed_continuation.candidates.clear();
            request->stable_continuation.image.reset();
            request->stable_continuation.candidates.clear();
            return;
        }
    }

    // Lane choice maximizes reusable prefix; ties break toward the lane whose occupation costs
    // least to replace - an empty lane before any retained session, then the shallowest
    // retained session - so a fresh request never clobbers a deep resident session while a
    // cheaper lane is available. A lane whose session is still being exported costs more than
    // any settled lane: taking it would stall the execution thread on the export.
    [[nodiscard]] std::optional<LaneChoice>
    find_admission_lane(const std::shared_ptr<Request>& request) {
        std::optional<LaneChoice> selected;
        std::uint32_t selected_reuse = 0;
        std::uint64_t selected_cost  = 0;
        const auto prefer            = [&](std::uint32_t reuse, std::uint64_t cost) {
            return !selected || reuse > selected_reuse ||
                   (reuse == selected_reuse && cost < selected_cost);
        };
        const auto replacement_cost = [&](std::uint32_t lane) {
            const std::uint64_t depth = instance_.program->retained_lane_depth(lane);
            return lane_export_in_flight(lane)
                       ? depth + (std::uint64_t{1} << 32)
                       : depth;
        };
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) { continue; }
            ensure_lane_plan(request, lane);
            const Plan& plan          = *request->lane_plans[lane];
            const std::uint32_t reuse = plan.summary().reusable_prompt_tokens;
            const std::uint64_t cost  = replacement_cost(lane);
            if (instance_.program->can_admit_lane(lane, plan) && prefer(reuse, cost)) {
                selected       = LaneChoice{.lane = lane};
                selected_reuse = reuse;
                selected_cost  = cost;
            }
        }
        if (selected) { return selected; }

        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) { continue; }
            ensure_lane_plan(request, lane);
            const Plan& plan          = *request->lane_plans[lane];
            const std::uint32_t reuse = plan.summary().reusable_prompt_tokens;
            const std::uint64_t cost  = replacement_cost(lane);
            if (instance_.program->can_admit_lane_after_retained_eviction(lane, plan) &&
                prefer(reuse, cost)) {
                selected = LaneChoice{
                    .lane           = lane,
                    .evict_retained = true,
                };
                selected_reuse = reuse;
                selected_cost  = cost;
            }
        }
        return selected;
    }

    [[nodiscard]] AdmissionProgress remove_pending_error(const std::shared_ptr<Request>& request,
                                                         std::exception_ptr error) {
        if (!erase_pending(request)) { return AdmissionProgress::None; }
        clear_protection_if_head(request);
        complete_error(request, std::move(error));
        publish_runtime_stats();
        return AdmissionProgress::ControlProgress;
    }

    [[nodiscard]] AdmissionProgress admit_planned_request(const std::shared_ptr<Request>& request,
                                                          LaneChoice choice,
                                                          BackfillClass backfill_class,
                                                          std::uint64_t backfill_epoch) {
        if (Clock::now() >= request->deadline) {
            return remove_pending_error(
                request, std::make_exception_ptr(RequestError(
                             RequestErrorKind::QueueTimeout,
                             "inference request expired while waiting for admission")));
        }
        if (request->cancelled.load(std::memory_order_acquire)) {
            if (!erase_pending(request)) { return AdmissionProgress::None; }
            clear_protection_if_head(request);
            complete_cancelled(request);
            publish_runtime_stats();
            return AdmissionProgress::ControlProgress;
        }

        const std::uint32_t lane = choice.lane;
        if (!request->lane_plans[lane]) {
            throw std::logic_error("selected admission lane has no request plan");
        }
        // The request transient region is a single startup-frozen allocation that a Vision
        // prefill holds across all of its chunks. A second one waits until the holder finishes;
        // text prefills need no region and admit freely beside it.
        if (transient_owner_ && request->lane_plans[lane]->summary().transient_bytes != 0) {
            return AdmissionProgress::None;
        }
        // Compared against restore_target_lane: a restore that refused the lane admission then
        // used is a lane-selection divergence, not a genuine capacity shortfall.
        request->continuation.admitted_lane = static_cast<std::int32_t>(lane);
        if (choice.evict_retained) {
            for (std::uint32_t retained_lane = 0;
                 retained_lane < max_concurrency_ &&
                 !instance_.program->can_admit_lane(lane, *request->lane_plans[lane]);
                 ++retained_lane) {
                if (retained_lane != lane && slots_[retained_lane] == nullptr &&
                    instance_.program->has_retained_lane(retained_lane)) {
                    evict_retained_lane(retained_lane);
                }
            }
            if (!instance_.program->can_admit_lane(lane, *request->lane_plans[lane])) {
                throw std::logic_error("retained eviction did not make admission feasible");
            }
        }

        Plan selected_plan = std::move(*request->lane_plans[lane]);
        request->lane_plans[lane].reset();
        if (!erase_pending(request)) { return AdmissionProgress::None; }
        release_planning_state(request);

        const RequestPlanSummary summary = selected_plan.summary();
        if (backfill_class == BackfillClass::Temporal) {
            if (!protection_ || protection_->epoch_id != backfill_epoch ||
                summary.service_work_quanta > protection_->temporal_credit) {
                throw std::logic_error("temporal backfill lost its protected credit");
            }
            protection_->temporal_credit -= summary.service_work_quanta;
        }
        clear_protection_if_head(request);

        // The lane's device state is rewritten from here on; a background export of the session
        // it retained must have finished reading it first.
        wait_lane_export(lane);
        refresh_lane_provenance(lane);
        classify_resident_continuation(
            request->continuation, summary.reusable_prompt_tokens,
            instance_.program->retained_lane_reused_bytes(lane, selected_plan),
            lane_provenance_[lane]);
        // Zero reuse means the target takes the FullReset path and destroys whatever session
        // the lane retained. Spill it first, and start the new session unbound either way so a
        // later eviction can never write it over the previous session's file.
        if (summary.reusable_prompt_tokens == 0) {
            spill_retained_lane(lane);
            lane_session_path_[lane].clear();
        }

        // A decision always has branch units left after its reusable state.
        const bool needs_prefill = summary.reusable_prompt_tokens < summary.prompt_tokens;
        bool target_started      = false;
        try {
            if (!request->decision) {
                request->budget.emplace(summary.effective_output_tokens,
                                        summary.effective_limit_reason);
                if (repetition_guard_.enabled) { request->repetition.emplace(repetition_guard_); }
                request->generated.reserve(summary.effective_output_tokens);
            }
            const auto admitted_at          = Clock::now();
            request->lane                   = lane;
            request->admitted               = admitted_at;
            request->admission_resources    = summary.admission;
            request->remaining_service_work = summary.service_work_quanta;
            request->backfill_epoch         = backfill_epoch;
            request->backfill_class         = backfill_class;
            slots_[lane]                    = request;
            retained_last_used_[lane]       = admitted_at;
            cumulative_stats_.queue_seconds_total +=
                std::chrono::duration<double>(admitted_at - request->submitted).count();
            ++cumulative_stats_.admitted_requests;
            invalidate_lane_plans(lane);

            request->prefill_total_tokens = summary.prompt_tokens;
            TransientRegion transient;
            if (needs_prefill) {
                const bool holds_transient = summary.transient_bytes != 0;
                if (holds_transient) {
                    instance_.request_memory.activate(summary.transient_bytes,
                                                      summary.transient_alignment);
                    transient = instance_.request_memory.region();
                }
                enter_prefill(lane, holds_transient);
            }
            publish_runtime_stats();
            target_started                = true;
            const auto unit_started       = Clock::now();
            const auto opening_watts      = sample_board_watts();
            const PrefillStepResult first =
                request->decision
                    ? instance_.program->start_decision_lane(
                          lane, std::move(request->decision->prompt), std::move(selected_plan))
                    : instance_.program->start_prefill_lane(lane, std::move(request->prompt),
                                                            std::move(selected_plan), transient);
            const auto unit_ended    = Clock::now();
            const auto closing_watts = sample_board_watts();
            cumulative_stats_.prefill_seconds_total +=
                std::chrono::duration<double>(unit_ended - unit_started).count();
            // The first chunk of every request runs here rather than in the worker loop's prefill
            // branch. Charging it to prefill keeps admission from silently dropping out of the
            // energy split, which for short prompts is the whole of their prefill.
            accumulate_phase_energy(cumulative_stats_.prefill_energy_joules, opening_watts,
                                    closing_watts, unit_started, unit_ended);
            if (!first.complete && !lane_is_prefilling(lane)) {
                throw std::logic_error("partial prefill did not retain its execution owner");
            }
            const bool cancel_at_boundary = request->cancelled.load(std::memory_order_acquire);
            resolve_prefill_step(request, first, cancel_at_boundary);
            publish_runtime_stats();
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            if (target_started) { instance_.program->abort_lane(lane); }
            lane_sessions_[lane].reset();
            leave_prefill(lane);
            slots_[lane].reset();
            invalidate_lane_plans(lane);
            complete_error(request, error);
            throw;
        }
        return AdmissionProgress::RanGpuUnit;
    }

    // Accumulates elapsed time into a RuntimeStats field for the enclosing scope.
    class PhaseTimer {
    public:
        PhaseTimer(double& sink) noexcept : sink_(sink), started_(Clock::now()) {}
        ~PhaseTimer() {
            sink_ += std::chrono::duration<double>(Clock::now() - started_).count();
        }
        PhaseTimer(const PhaseTimer&)            = delete;
        PhaseTimer& operator=(const PhaseTimer&) = delete;

    private:
        double& sink_;
        Clock::time_point started_;
    };

    AdmissionProgress try_admit_one() {
        ++cumulative_stats_.worker_admission_calls;
        bool control_progress = false;
        for (;;) {
            std::vector<std::shared_ptr<Request>> queued = pending_snapshot();
            std::erase_if(queued, [](const std::shared_ptr<Request>& request) {
                return stable_flight_blocked(*request);
            });
            if (queued.empty()) {
                protection_.reset();
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }
            const std::shared_ptr<Request>& head = queued.front();
            if (protection_ && protection_->head_request_id != head->id) { protection_.reset(); }
            if (head->cancelled.load(std::memory_order_acquire)) {
                if (erase_pending(head)) {
                    clear_protection_if_head(head);
                    complete_cancelled(head);
                    publish_runtime_stats();
                    control_progress = true;
                }
                continue;
            }
            if (Clock::now() >= head->deadline) {
                continuation_stats_.rejected_queue_timeout.fetch_add(1,
                                                                      std::memory_order_relaxed);
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::QueueTimeout,
                              "inference request expired while waiting for admission")));
                control_progress = true;
                continue;
            }

            try {
                PhaseTimer timer(cumulative_stats_.worker_admission_plan_seconds);
                ensure_base_plan(head);
            } catch (...) {
                (void)remove_pending_error(head, std::current_exception());
                control_progress = true;
                continue;
            }
            const RequestPlanSummary& head_base = head->base_plan->summary();
            if (!admission_resources_fit(head_base.admission, admission_capacity_)) {
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::ContextLengthExceeded,
                              "request reservation exceeds Engine shared KV capacity")));
                control_progress = true;
                continue;
            }

            std::optional<LaneChoice> head_lane;
            try {
                PhaseTimer timer(cumulative_stats_.worker_admission_plan_seconds);
                head_lane = find_admission_lane(head);
            } catch (...) {
                (void)remove_pending_error(head, std::current_exception());
                control_progress = true;
                continue;
            }
            if (head_lane) {
                bool adapter_resident = false;
                try {
                    adapter_resident = ensure_adapter_resident(head);
                } catch (...) {
                    (void)remove_pending_error(head, std::current_exception());
                    control_progress = true;
                    continue;
                }
                if (!adapter_resident) {
                    // This is adapter-only contention, not a resource frontier. Feeding it to the
                    // KV/lane protection policy would violate that policy's blocked-by-resources
                    // precondition. Drain until a pinned slot is released; the head is checked
                    // before backfill on every pass.
                    return control_progress ? AdmissionProgress::ControlProgress
                                            : AdmissionProgress::None;
                }
                {
                    PhaseTimer timer(cumulative_stats_.worker_admission_restore_seconds);
                    try_restore_continuation(head);
                }
                // Residency may have evicted a soft-retained lane, and restore may have imported
                // a deeper candidate, so the preflight lane choice is no longer authoritative.
                try {
                    PhaseTimer timer(cumulative_stats_.worker_admission_plan_seconds);
                    head_lane = find_admission_lane(head);
                } catch (...) {
                    (void)remove_pending_error(head, std::current_exception());
                    control_progress = true;
                    continue;
                }
            }
            if (head_lane) {
                PhaseTimer timer(cumulative_stats_.worker_admission_commit_seconds);
                return admit_planned_request(head, *head_lane, BackfillClass::None, 0);
            }

            const ActiveAdmissionSet active = active_admission_set();
            if (active.size == 0) {
                throw std::logic_error("exclusive-feasible request cannot enter an idle Engine");
            }
            if (!protection_) {
                protection_.emplace(make_admission_protection(next_protection_epoch_++, head->id,
                                                              head_base.admission, active.span(),
                                                              admission_capacity_));
            }
            if (protected_head_safe_without_temporal(*protection_, active.span(),
                                                     admission_capacity_)) {
                protection_->phase = ProtectionPhase::Drain;
            }
            if (protection_->phase == ProtectionPhase::Drain) {
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }

            const std::uint64_t frontier_distance =
                protection_frontier_distance(*protection_, active.span());
            for (std::size_t i = 1; i < queued.size(); ++i) {
                const std::shared_ptr<Request>& candidate = queued[i];
                if (candidate->cancelled.load(std::memory_order_acquire)) {
                    if (erase_pending(candidate)) {
                        complete_cancelled(candidate);
                        publish_runtime_stats();
                        control_progress = true;
                    }
                    continue;
                }
                if (Clock::now() >= candidate->deadline) {
                    continuation_stats_.rejected_queue_timeout.fetch_add(
                        1, std::memory_order_relaxed);
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::QueueTimeout,
                                       "inference request expired while waiting for admission")));
                    control_progress = true;
                    continue;
                }

                try {
                    PhaseTimer timer(cumulative_stats_.worker_admission_plan_seconds);
                    ensure_base_plan(candidate);
                } catch (...) {
                    (void)remove_pending_error(candidate, std::current_exception());
                    control_progress = true;
                    continue;
                }
                const RequestPlanSummary& candidate_base = candidate->base_plan->summary();
                if (!admission_resources_fit(candidate_base.admission, admission_capacity_)) {
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::ContextLengthExceeded,
                                       "request reservation exceeds Engine shared KV capacity")));
                    control_progress = true;
                    continue;
                }

                std::optional<LaneChoice> candidate_lane;
                try {
                    candidate_lane = find_admission_lane(candidate);
                } catch (...) {
                    (void)remove_pending_error(candidate, std::current_exception());
                    control_progress = true;
                    continue;
                }
                if (!candidate_lane) { continue; }
                const RequestPlanSummary& candidate_plan =
                    candidate->lane_plans[candidate_lane->lane]->summary();

                BackfillClass backfill = BackfillClass::None;
                if (persistent_backfill_is_safe(*protection_, active.span(),
                                                candidate_plan.admission, admission_capacity_)) {
                    backfill = BackfillClass::Persistent;
                } else if (candidate_plan.service_work_quanta <= frontier_distance &&
                           candidate_plan.service_work_quanta <= protection_->temporal_credit) {
                    backfill = BackfillClass::Temporal;
                }
                if (backfill != BackfillClass::None) {
                    // Only mutate slot residency after this candidate has a lane and is safe to
                    // backfill. A candidate rejected above must not evict unrelated retained state
                    // merely because the scheduler examined it.
                    try {
                        if (!ensure_adapter_resident(candidate)) { continue; }
                    } catch (...) {
                        (void)remove_pending_error(candidate, std::current_exception());
                        control_progress = true;
                        continue;
                    }
                    {
                        PhaseTimer timer(cumulative_stats_.worker_admission_restore_seconds);
                        try_restore_continuation(candidate);
                    }
                    try {
                        candidate_lane = find_admission_lane(candidate);
                    } catch (...) {
                        (void)remove_pending_error(candidate, std::current_exception());
                        control_progress = true;
                        continue;
                    }
                    if (!candidate_lane) { continue; }
                    const RequestPlanSummary& restored_plan =
                        candidate->lane_plans[candidate_lane->lane]->summary();
                    backfill = BackfillClass::None;
                    if (persistent_backfill_is_safe(*protection_, active.span(),
                                                    restored_plan.admission,
                                                    admission_capacity_)) {
                        backfill = BackfillClass::Persistent;
                    } else if (restored_plan.service_work_quanta <= frontier_distance &&
                               restored_plan.service_work_quanta <=
                                   protection_->temporal_credit) {
                        backfill = BackfillClass::Temporal;
                    }
                    if (backfill == BackfillClass::None) { continue; }
                    return admit_planned_request(candidate, *candidate_lane, backfill,
                                                 protection_->epoch_id);
                }
            }
            return control_progress ? AdmissionProgress::ControlProgress : AdmissionProgress::None;
        }
    }

    void run_decode_round(const RoundMembership& membership) {
        const std::span<const std::uint32_t> lanes = membership.lane_span();
        const auto unit_started                    = Clock::now();
        const BatchedGeneratedRound round =
            instance_.program->decode_batch(lanes, membership.budget_span());
        cumulative_stats_.decode_seconds_total +=
            std::chrono::duration<double>(Clock::now() - unit_started).count();

        std::array<std::uint8_t, kMaximumConcurrency> cancelled{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            cancelled[row] =
                slots_[lanes[row]]->cancelled.load(std::memory_order_acquire) ? 1U : 0U;
        }

        if (round.row_stride == 0 ||
            (!round.row_counts.empty() && round.row_counts.size() != lanes.size()) ||
            round.tokens.size() < static_cast<std::size_t>(round.row_stride) * lanes.size()) {
            throw RoundFault("decode batch returned an invalid ragged layout: stride " +
                             std::to_string(round.row_stride) + ", " +
                             std::to_string(round.row_counts.size()) + " counts, " +
                             std::to_string(round.tokens.size()) + " tokens across " +
                             std::to_string(lanes.size()) + " lanes");
        }

        std::array<std::uint32_t, kMaximumConcurrency> accepted{};
        std::array<std::uint8_t, kMaximumConcurrency> terminal{};
        std::array<FinishReason, kMaximumConcurrency> finish_reasons{};
        // Terminal requests of this round, kept alive past their slot release so they can be
        // woken after the published lane table already reflects their completion.
        std::array<std::shared_ptr<Request>, kMaximumConcurrency> finished{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            const auto& request      = slots_[lane];
            const std::uint32_t count =
                round.row_counts.empty() ? 1U : static_cast<std::uint32_t>(round.row_counts[row]);
            if (count == 0 || count > round.row_stride) {
                throw RoundFault("decode batch returned an invalid licensed row extent: lane " +
                                 std::to_string(lane) + " produced " + std::to_string(count) +
                                 " of stride " + std::to_string(round.row_stride));
            }
            const auto row_tokens =
                round.tokens.subspan(row * round.row_stride, static_cast<std::size_t>(count));
            if (cancelled[row]) {
                (void)request->output.preview_terminal(FinishReason::Cancelled);
                accepted[row]       = 0;
                terminal[row]       = 1;
                finish_reasons[row] = FinishReason::Cancelled;
            } else {
                // A tripped guard ends the request, but the row still has to commit a licensed
                // prefix: only a cancelled row may commit nothing. Licensing the round against
                // its own extent makes the decoder terminalize here exactly as an exhausted
                // budget does, and a stop token inside the round still wins and reports itself.
                const bool repeating = request->repetition && request->repetition->tripped();
                const OutputDecision decision = request->output.preview(
                    row_tokens, repeating ? count : request->budget->remaining(),
                    repeating ? FinishReason::RepetitionCycle : request->budget->limit_reason());
                accepted[row]       = decision.accepted_tokens;
                terminal[row]       = decision.finished() ? 1 : 0;
                finish_reasons[row] = decision.finish_reason;
            }
            // Checked here as well as in the target so a branch that resolves a row without
            // consulting the output policy cannot hand the fold an unlicensed prefix.
            if (!runtime::row_commit_is_licensed(cancelled[row] != 0, accepted[row], count,
                                                 terminal[row] != 0)) {
                throw RoundFault("resolved decode row is not a licensed commit: lane " +
                                 std::to_string(lane) + " committed " +
                                 std::to_string(accepted[row]) + " of " + std::to_string(count) +
                                 " produced, terminal " + std::to_string(terminal[row]) +
                                 ", cancelled " + std::to_string(cancelled[row]));
            }
        }

        instance_.program->resolve_pending_batch(
            lanes, std::span<const std::uint32_t>(accepted.data(), lanes.size()),
            std::span<const std::uint8_t>(terminal.data(), lanes.size()),
            std::span<const std::uint8_t>(cancelled.data(), lanes.size()));

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            const auto& request      = slots_[lane];
            if (!cancelled[row]) {
                const auto row_tokens = round.tokens.subspan(
                    row * round.row_stride, static_cast<std::size_t>(accepted[row]));
                request->generated.insert(request->generated.end(), row_tokens.begin(),
                                          row_tokens.end());
                request->budget->commit(accepted[row]);
                if (request->repetition) { (void)request->repetition->observe(row_tokens); }
                consume_service_work(request, accepted[row]);
            }
            auto published = request->output.commit_preview();
            if (!request->first_token && accepted[row] != 0) {
                request->first_token = Clock::now();
            }
            append_output(request, std::move(published));
            if (terminal[row]) {
                // The waiter is not woken here: a client that observes its own completion must
                // also be able to observe the slot state that completion produced, so every
                // terminal lane in this round is released and republished first.
                publish_retained_completion(request, lane, finish_reasons[row]);
                finished[row] = request;
                remove_completed_slot(lane);
            }
        }
        (void)enforce_l1_retention();
        ++cumulative_stats_.decode_rounds;
        cumulative_stats_.decode_row_rounds += lanes.size();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            if (!cancelled[row]) { cumulative_stats_.committed_decode_tokens += accepted[row]; }
        }
        publish_runtime_stats();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            if (finished[row]) { complete_success(finished[row], finish_reasons[row]); }
        }
    }

    // Abandon the round in flight without retiring the executor. Every lane of a faulted round
    // is left with its pending candidate unresolved, so recovery is round-scoped rather than
    // per-lane: the requests it served fail, their lanes are aborted, and queued work still runs.
    void fail_round(std::exception_ptr error) noexcept {
        ++cumulative_stats_.decode_rounds_abandoned;
        clear_prefill_lanes();
        protection_.reset();
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] == nullptr) { continue; }
            const std::shared_ptr<Request> request = slots_[lane];
            instance_.program->abort_lane(lane);
            lane_sessions_[lane].reset();
            remove_completed_slot(lane);
            complete_error(request, error);
        }
        publish_runtime_stats();
    }

    void fail_all(std::exception_ptr error) noexcept {
        std::vector<std::shared_ptr<Request>> pending;
        {
            std::lock_guard lock(queue_mutex_);
            failed_ = true;
            pending.assign(pending_.begin(), pending_.end());
            pending_.clear();
        }
        clear_prefill_lanes();
        protection_.reset();
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                instance_.program->abort_lane(lane);
                lane_sessions_[lane].reset();
                complete_error(slots_[lane], error);
                slots_[lane].reset();
            }
        }
        for (const auto& request : pending) { complete_error(request, error); }
        publish_runtime_stats();
    }

    void worker_loop() noexcept {
        bool previous_unit_was_decode = false;
        for (;;) {
            {
                std::unique_lock lock(queue_mutex_);
                if (!stopping_ && pending_.empty()) {
                    bool active = false;
                    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                        active = active || slots_[lane] != nullptr;
                    }
                    if (!active) {
                        if (l1_policy_active_ &&
                            l1_idle_ttl_ > Clock::duration::zero()) {
                            const auto now = Clock::now();
                            std::optional<Clock::duration> until_expiry;
                            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                                if (!retained_last_used_[lane]) { continue; }
                                const auto elapsed = now >= *retained_last_used_[lane]
                                                         ? now - *retained_last_used_[lane]
                                                         : Clock::duration::zero();
                                const auto remaining = elapsed >= l1_idle_ttl_
                                                           ? Clock::duration::zero()
                                                           : l1_idle_ttl_ - elapsed;
                                if (!until_expiry || remaining < *until_expiry) {
                                    until_expiry = remaining;
                                }
                            }
                            if (until_expiry) {
                                queue_cv_.wait_for(lock, *until_expiry,
                                                   [&] { return stopping_ || !pending_.empty(); });
                            } else {
                                queue_cv_.wait(lock,
                                               [&] { return stopping_ || !pending_.empty(); });
                            }
                        } else {
                            queue_cv_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
                        }
                    }
                }
                if (stopping_) {
                    lock.unlock();
                    fail_all(std::make_exception_ptr(RequestError(
                        RequestErrorKind::Unavailable, "inference engine is shutting down")));
                    return;
                }
            }

            try {
                refresh_stable_flights();
                refresh_restore_preparation();
                std::unique_lock execution_lock(execution_mutex_);
                const auto upkeep_started = Clock::now();
                if (enforce_l1_retention()) { publish_runtime_stats(); }
                const bool have_pending          = expire_pending_requests();
                const auto cancelled_at_boundary = snapshot_cancellations();
                cancel_active_requests(cancelled_at_boundary);
                const RoundMembership membership = build_round_membership();

                cumulative_stats_.worker_upkeep_seconds +=
                    std::chrono::duration<double>(Clock::now() - upkeep_started).count();

                if (!prefill_lanes_.empty()) {
                    // Decode rounds and prefill chunks are not comparable units of work: a chunk
                    // costs several rounds, so alternating them one for one hands the execution
                    // thread almost entirely to whichever lane is still consuming its prompt and
                    // starves every lane that is already generating. Balance them by elapsed time
                    // instead of by count, which keeps the share independent of the chunk size and
                    // self-tunes to the measured cost of each unit on this model and device.
                    // A balance of zero keeps the historical one-for-one alternation: every
                    // chunk is still followed by exactly one decode round. Larger values buy the
                    // generating lanes a proportional share of the thread.
                    const bool decode_is_owed =
                        !membership.empty() &&
                        (!previous_unit_was_decode ||
                         decode_seconds_since_prefill_ <
                             prefill_decode_balance_ * prefill_step_seconds_);
                    if (decode_is_owed) {
                        const auto started = Clock::now();
                        timed_decode_round(membership);
                        decode_seconds_since_prefill_ +=
                            std::chrono::duration<double>(Clock::now() - started).count();
                        previous_unit_was_decode = true;
                        continue;
                    }
                    // A prefill unit. A waiting request that admission can place takes it for its
                    // first chunk; its lane then competes for chunks shortest-remaining-first, so
                    // a short continuation suffix is served in a couple of units instead of behind
                    // the whole of another lane's cold prompt. When nothing can be placed the unit
                    // goes to the shortest resident prefill as before.
                    const auto started = Clock::now();
                    bool ran_unit      = false;
                    if (have_pending && prefill_lanes_.size() < max_concurrency_) {
                        const auto admission_started = Clock::now();
                        const AdmissionProgress progress = try_admit_one();
                        cumulative_stats_.worker_admission_seconds +=
                            std::chrono::duration<double>(Clock::now() - admission_started)
                                .count();
                        ran_unit = progress == AdmissionProgress::RanGpuUnit;
                    }
                    if (!ran_unit) {
                        if (prefill_lanes_.empty()) { continue; }
                        timed_prefill_step();
                    }
                    const double step =
                        std::chrono::duration<double>(Clock::now() - started).count();
                    // Track the recent cost of a chunk so the balance follows prompt length
                    // and batch composition rather than a compiled-in constant.
                    prefill_step_seconds_ = prefill_step_seconds_ == 0.0
                                                ? step
                                                : 0.75 * prefill_step_seconds_ + 0.25 * step;
                    decode_seconds_since_prefill_ = 0.0;
                    previous_unit_was_decode      = false;
                    continue;
                }
                decode_seconds_since_prefill_ = 0.0;

                if (have_pending && (membership.empty() || previous_unit_was_decode)) {
                    const auto admission_started = Clock::now();
                    const AdmissionProgress progress = try_admit_one();
                    cumulative_stats_.worker_admission_seconds +=
                        std::chrono::duration<double>(Clock::now() - admission_started).count();
                    if (progress == AdmissionProgress::RanGpuUnit) {
                        previous_unit_was_decode = false;
                        continue;
                    }
                    if (progress == AdmissionProgress::ControlProgress && membership.empty()) {
                        continue;
                    }
                }

                if (!membership.empty()) {
                    timed_decode_round(membership);
                    previous_unit_was_decode = true;
                    continue;
                }
                execution_lock.unlock();
                std::unique_lock queue_lock(queue_mutex_);
                queue_cv_.wait_for(queue_lock, std::chrono::milliseconds(10));
            } catch (const RoundFault&) { fail_round(std::current_exception()); } catch (...) {
                fail_all(std::current_exception());
                return;
            }
        }
    }

    Instance& instance_;
    const std::uint32_t max_concurrency_;
    const std::size_t max_outstanding_;
    const std::chrono::milliseconds pending_timeout_;
    const bool auto_save_evicted_;
    const RepetitionGuardOptions repetition_guard_;
    const AdmissionResources admission_capacity_;
    // Pool adapter names by pool index.
    const std::vector<std::string> adapter_names_;
    std::unique_ptr<cache::ContinuationCache> continuation_cache_;
    // Off preserves historical unmanaged same-lane prefix reuse. Every active tier applies this
    // policy; zero capacity or zero TTL therefore retains no inactive GPU continuation.
    const bool l1_policy_active_;
    const std::size_t l1_byte_budget_;
    const Clock::duration l1_idle_ttl_;
    const std::chrono::seconds publication_l2_ttl_;
    const std::chrono::seconds publication_l3_ttl_;

    mutable std::mutex execution_mutex_;
    mutable std::mutex queue_mutex_;
    mutable std::mutex stats_mutex_;
    mutable std::mutex stable_flight_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<Request>> pending_;
    std::size_t outstanding_       = 0;
    std::uint64_t next_request_id_ = 1;
    std::array<std::shared_ptr<Request>, kMaximumConcurrency> slots_{};
    // Lanes still consuming their prompt, in admission order. Prefill chunks are issued
    // shortest-remaining-first across this set and admission stays open while it is non-empty,
    // so a short suffix never waits behind another lane's long cold prompt.
    std::vector<std::uint32_t> prefill_lanes_;
    // The prefilling lane holding the frozen request transient region (Vision encode output).
    // The region exists once, so at most one such lane prefills at a time.
    std::optional<std::uint32_t> transient_owner_;
    std::array<std::uint64_t, kMaximumConcurrency> lane_plan_versions_{};
    std::array<std::optional<LaneSession>, kMaximumConcurrency> lane_sessions_{};
    std::array<LaneContinuationProvenance, kMaximumConcurrency> lane_provenance_{};
    std::array<std::optional<Clock::time_point>, kMaximumConcurrency> retained_last_used_{};
    std::optional<AdmissionProtection> protection_;
    std::uint64_t next_protection_epoch_ = 1;
    RuntimeStats cumulative_stats_;
    RuntimeStats published_stats_;
    // Read only from the execution thread, at execution-unit boundaries. The expensive cumulative
    // energy counter is deliberately never read here: it costs milliseconds and would land on the
    // thread whose latency this engine exists to protect. An interval observer samples that.
    core::PowerMeter power_meter_;
    AtomicContinuationStats continuation_stats_;
    StablePrefixFlights stable_flights_;
    std::vector<SlotState> published_slots_;
    std::vector<AdapterSlotState> published_adapter_slots_;
    MemorySummary published_memory_;
    // Digest of each lane's retained session, maintained by the completion and restore paths
    // (the only ones that set `retained`) so publishing needs no ledger hashing.
    std::array<std::string, kMaximumConcurrency> retained_digest_cache_{};
    std::array<std::vector<SlotCheckpoint>, kMaximumConcurrency> retained_checkpoints_cache_{};
    // Slot file each lane's resident session was last saved to or restored from; empty means
    // unbound. Guarded by execution_mutex_ like the lane state it describes.
    std::array<std::string, kMaximumConcurrency> lane_session_path_{};
    std::string eviction_model_binding_;
    std::function<void(std::string, targets::qwen3_8::RetainedSessionSnapshot&&)> eviction_sink_;
    bool stopping_ = false;
    bool failed_   = false;
    std::thread worker_;

    std::mutex publication_mutex_;
    std::condition_variable publication_cv_;
    // Thread-time balance between decode rounds and prefill chunks while a prefill is in flight.
    // 1.0 splits the execution thread evenly between the two.
    const double prefill_decode_balance_ = 0.0;
    double prefill_step_seconds_        = 0.0;
    double decode_seconds_since_prefill_ = 0.0;
    std::deque<Publication> publications_;
    std::unordered_map<std::string, PendingSessionPublication> session_publications_;
    std::uint64_t publication_issued_    = 0;
    std::uint64_t publication_completed_ = 0;
    bool publication_stopping_           = false;
    // Set while a queued or running export job still reads the lane's device state. The
    // execution thread waits on it (wait_lane_export) before touching a retained lane.
    std::array<bool, kMaximumConcurrency> lane_export_pending_{};
    std::thread publication_worker_;

    // Continuation decoding, run ahead of admission.
    //
    // Decoding a continuation image is pure host work over startup-fixed geometry: it touches no
    // lane state, no device memory and no CUDA API, so §2.6 permits it off the execution thread.
    // Leaving it inline made every restore spend its decode between GPU units, because
    // admission was, at the time, skipped entirely while a prefill was in flight.
    //
    // Preparation is speculative and therefore strictly bounded: only the first
    // kPreparationDepth queued requests are decoded ahead, at most kPreparedLimit payloads are
    // resident, and entries whose request has left the queue are pruned every iteration. A miss is
    // free - the executor decodes inline exactly as before.
    // A job names a session, never a fixed image. The candidates captured when a request was
    // submitted are stale by the time it is admitted: it waits in the queue while the previous turn
    // of the same session completes and publishes a newer image, and that newer image is the one
    // admission reconciles to. Resolving the session's current head at decode time is what makes
    // the prepared payload the one the executor actually asks for.
    struct PreparationJob {
        std::string session;
    };
    struct PreparedContinuationEntry {
        std::shared_ptr<targets::qwen3_8::DecodedContinuation> payload;
        Clock::time_point prepared_at{};
    };
    static constexpr std::size_t kPreparationDepth = 2;
    static constexpr std::size_t kPreparedLimit    = 2;
    // A decoded payload belongs to the image, not to the request that asked for it: sessions share
    // images, the queue is reordered by stable-flight filtering, and the request that triggers a
    // decode need not be the one admitted next. Keying by content identity makes the payload
    // usable by whichever request the executor actually admits.
    std::mutex preparation_mutex_;
    std::condition_variable preparation_cv_;
    std::deque<PreparationJob> preparation_queue_;
    std::unordered_map<std::string, PreparedContinuationEntry> prepared_;
    std::unordered_set<std::string> preparation_in_flight_;
    bool preparation_stopping_ = false;
    std::thread preparation_worker_;
    std::atomic<std::uint64_t> continuation_preparation_hits_{0};
    std::atomic<std::uint64_t> continuation_preparation_decoded_{0};
    std::atomic<std::uint64_t> continuation_preparation_inline_{0};
};

} // namespace ninfer::runtime
