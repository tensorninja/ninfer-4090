#pragma once

#include "ninfer/types.h"
#include "runtime/contract/transient_region.h"
#include "runtime/contract/types.h"
#include "runtime/cache/continuation_cache.h"
#include <ninfer/targets/qwen3_8/decision.h>
#include <ninfer/targets/qwen3_8/prepared_prompt.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::targets::qwen3_8 {

// Owned host payload of a decoded continuation image. Opaque here: the engine only moves it from
// the thread that decoded it to the executor thread that imports it, and never inspects it.
struct DecodedContinuation;

// The pass kind a target leaf runs in. It selects the activation-compute profile a leaf may admit:
// INT8 activations are admitted in Prefill only. Decision is a System One pass (state chunk,
// packed branch pass or long-branch chunk): prefill-shaped, but held to the A16 profile so that
// decision probabilities stay within the serving tolerance of the BF16 evaluation path.
enum class TextPhase {
    Prefill,
    Verify,
    Decision,
};

struct GraphExecutionProfile {
    std::uint32_t min            = 0;
    std::uint32_t max            = 0;
    std::uint32_t topology_class = 0;
};

// One retained lane's complete session image (host bytes) for save/restore persistence. The
// byte layout is a target-private format; callers treat it as opaque and durable only across
// processes serving the identical model and KV configuration.
struct RetainedSessionSnapshot {
    std::vector<std::uint8_t> bytes;
    std::uint32_t tokens = 0;
    std::string session_digest;
};

// One prompt boundary with its content-addressed continuation alias. Every entry is a lookup
// candidate. A `publish` entry below the rewrite frontier is captured during prefill by the
// request that builds it; the entry at the rewrite frontier is where the lane keeps its turn
// checkpoint through decode, so its image is the completed lane's image (`export_continuation_lane`)
// and nothing is captured for it during prefill.
struct PromptBoundaryAlias {
    std::uint32_t depth     = 0;
    PromptBoundaryKind kind = PromptBoundaryKind::SystemTools;
    bool publish            = false;
    bool rewrite_frontier   = false;
    std::string alias;
};

// A complete continuation image captured at one prompt boundary during prefill.
struct CapturedContinuation {
    std::uint32_t depth = 0;
    cache::ContinuationImage image;
};

// One device slot of the LoRA bank: the pool index staged into it (-1 when never staged), and
// whether a lane executing against it pins it against a swap.
struct LoraSlotState {
    std::int32_t adapter = -1;
    bool pinned          = false;
};

namespace detail {
template <class Variant>
struct SequencePlanImpl;
template <class Variant>
struct SequencePlannerImpl;
template <class Variant>
struct RequestPlanImpl;
template <class Variant>
struct RequestBasePlanImpl;
template <class Variant>
class ProgramImpl;
} // namespace detail

template <class Variant>
class SequencePlanner;

// These are the complete family execution types. Exact packages bind them to a private Variant;
// target selection remains outside this layer and happens once in the closed Engine registry.
template <class Variant>
class SequencePlan {
public:
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t kv_capacity() const noexcept;
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept;
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t request_transient_capacity_bytes() const noexcept;

public:
    // Family-private construction/storage seam; exact packages expose only the completed alias.
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::SequencePlanImpl<Variant>> impl_;

    template <class V>
    friend class SequencePlanner;
    template <class V>
    friend class detail::ProgramImpl;
};

template <class Variant>
class SequencePlanner {
public:
    SequencePlanner(SequencePlanner&&) noexcept;
    SequencePlanner& operator=(SequencePlanner&&) noexcept;
    ~SequencePlanner();

    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const noexcept;
    [[nodiscard]] SequencePlan<Variant> finalize(std::uint32_t main_page_groups) &&;

public:
    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl_;

    template <class V>
    friend SequencePlanner<V> make_sequence_planner(DeviceContext&, const EngineOptions&,
                                                    typename V::WeightsProfile);
};

template <class Variant>
class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

public:
    explicit RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl_;
};

template <class Variant>
class RequestPlan {
public:
    RequestPlan(RequestPlan&&) noexcept;
    RequestPlan& operator=(RequestPlan&&) noexcept;
    ~RequestPlan();

    RequestPlan(const RequestPlan&)            = delete;
    RequestPlan& operator=(const RequestPlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

public:
    // Family-private construction/storage seam. This header is repository-internal; exact
    // packages expose only the completed alias and never inspect this pointer.
    explicit RequestPlan(std::unique_ptr<detail::RequestPlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::RequestPlanImpl<Variant>> impl_;
};

template <class Variant>
class Program {
public:
    ~Program() noexcept;

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    // Engine-internal fixed-lane execution surface. The public Engine owns scheduling; Program
    // owns target state images and executes one immutable decode batch membership.
    [[nodiscard]] RequestBasePlan<Variant>
    plan_request_base(const PreparedPrompt& prompt,
                      const runtime::ResolvedExecutionOptions& options);
    // `capture_depths` are the boundary depths this request builds (ascending); the plan captures
    // those that lie beyond the lane's reused prefix.
    [[nodiscard]] RequestPlan<Variant>
    plan_request_for_lane(std::uint32_t lane, const PreparedPrompt& prompt,
                          const RequestBasePlan<Variant>& base,
                          std::span<const std::uint32_t> capture_depths);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane,
                                      const RequestPlan<Variant>& plan) const noexcept;
    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                           const RequestPlan<Variant>& plan) const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    // Make a pool adapter device-resident, staging it over the least recently used slot that no
    // generating lane depends on. Retained lanes holding the displaced adapter are handed to
    // `release_retained` so the caller publishes their sessions instead of losing them. Returns
    // false, changing nothing, when every slot is held by a generating lane using another
    // adapter; the caller defers the request until a lane frees. The base weights are always
    // resident. This must succeed before a request selecting the adapter is admitted, and before
    // a saved session naming it is restored.
    [[nodiscard]] bool
    ensure_adapter_resident(std::int32_t adapter,
                            const std::function<void(std::uint32_t)>& release_retained);
    // Stable cache scope for a pool adapter: "base", or the adapter's artifact content
    // fingerprint. Continuation state produced under one adapter is invalid under any other, and
    // this is what keeps their aliases disjoint across swaps, pool reordering and restarts.
    [[nodiscard]] std::string adapter_scope(std::int32_t adapter) const;
    // Bank residency for telemetry: adapters staged into a slot so far, the execution-thread time
    // those stages took (slab preparation, demotion of displaced retained lanes, upload), and each
    // slot's occupant in bank order. The slot list is empty for a Program without a LoRA bank.
    [[nodiscard]] std::uint64_t lora_stage_count() const noexcept;
    [[nodiscard]] double lora_stage_seconds() const noexcept;
    [[nodiscard]] std::vector<LoraSlotState> lora_slot_states() const;
    // Pool index of the adapter whose state a retained lane holds; -1 for the base weights and
    // for a lane that retains nothing.
    [[nodiscard]] std::int32_t retained_lane_adapter(std::uint32_t lane) const noexcept;
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                                PreparedPrompt&& prompt,
                                                                RequestPlan<Variant>&& plan,
                                                                runtime::TransientRegion transient);
    // Advances a staged generation prefill or a decision by one unit.
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane);
    // System One decisions. A decision plans into the same base and lane plans as a generation,
    // is admitted under the same lane/KV/adapter rules, and runs as a prefill lane whose units are
    // its state chunks, branch passes and long-branch chunks; it never joins a decode batch. Once
    // advance_prefill_lane reports it complete, take_decision_lane returns its probabilities and
    // leaves the state [0, state_tokens) retained when the plan allows reuse. A retained decision
    // state is continued only by a later decision of the same adapter.
    [[nodiscard]] RequestBasePlan<Variant>
    plan_decision_base(const DecisionPrompt& prompt,
                       const runtime::ResolvedDecisionOptions& options);
    [[nodiscard]] RequestPlan<Variant> plan_decision_for_lane(std::uint32_t lane,
                                                              const DecisionPrompt& prompt,
                                                              const RequestBasePlan<Variant>& base);
    [[nodiscard]] runtime::PrefillStepResult start_decision_lane(std::uint32_t lane,
                                                                 DecisionPrompt&& prompt,
                                                                 RequestPlan<Variant>&& plan);
    [[nodiscard]] DecisionOutcome take_decision_lane(std::uint32_t lane);
    [[nodiscard]] bool retained_lane_holds_decision(std::uint32_t lane) const noexcept;
    // Continuation image kind `decision_state`: a retained decision state, its text KV and GDN
    // slot only. It is named by the alias of its exact state row (empty for a state that cannot be
    // named; the caller scopes it by adapter like every alias) and restored only for a decision of
    // exactly that state and adapter; continuing a shorter retained state stays L1 lane planning.
    // These calls mirror the chat continuation calls below. A decision call never accepts a chat
    // image and a chat call never accepts a decision image. The export follows
    // `fence_lane_for_export` under the same contract as `export_continuation_lane_background`.
    [[nodiscard]] std::optional<std::string> decision_state_alias(const DecisionPrompt& prompt) const;
    [[nodiscard]] cache::ContinuationImage
    export_decision_state_background(std::uint32_t lane) const;
    [[nodiscard]] std::uint32_t
    preflight_decision_state_metadata(const cache::SessionCandidateDescriptor& candidate,
                                      const DecisionPrompt& prompt) const noexcept;
    // The state extent when the image is exactly the decision's state under `adapter`, else zero.
    [[nodiscard]] std::uint32_t
    preflight_decision_state(const cache::ContinuationImage& image, const DecisionPrompt& prompt,
                             std::int32_t adapter,
                             std::uint32_t* divergence_tokens = nullptr) const noexcept;
    [[nodiscard]] std::shared_ptr<DecodedContinuation>
    decode_decision_state(const cache::ContinuationImage& image) const;
    // Leaves the lane holding the retained decision state that completing this decision cold
    // would have left, reserved for the decision's planned `entitlement`.
    [[nodiscard]] ContinuationRestoreFailure
    import_decision_state_lane(std::uint32_t lane, const cache::ContinuationImage& image,
                               const DecodedContinuation& decoded, const DecisionPrompt& prompt,
                               std::int32_t adapter, runtime::KvPageFootprint entitlement) noexcept;
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets);
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    // Complete a generating lane between rounds, retaining its session. Used when the KV pool
    // cannot be grown to execute one more round, so the request ends at its current length
    // instead of failing mid-stream.
    void retire_lane(std::uint32_t lane);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled);
    void abort_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] std::size_t retained_lane_resident_bytes(std::uint32_t lane) const noexcept;
    [[nodiscard]] std::size_t retained_lane_reused_bytes(
        std::uint32_t lane, const RequestPlan<Variant>& plan) const noexcept;
    void evict_retained_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] cache::ContinuationImage export_continuation_lane(std::uint32_t lane) const;
    // Background export of a retained lane. `fence_lane_for_export` runs on the execution thread
    // once the lane's state is final and orders the export after the lane's last engine-stream
    // work; `export_continuation_lane_background` then runs on another thread over its own
    // stream and pinned ring while the engine stream keeps executing other lanes. The caller
    // must keep the lane retained and untouched until the background export returns.
    void fence_lane_for_export(std::uint32_t lane);
    [[nodiscard]] cache::ContinuationImage
    export_continuation_lane_background(std::uint32_t lane) const;
    // The prompt's content-addressed boundaries, ascending by depth, each with its alias. Empty
    // for a prompt that is not reusable.
    [[nodiscard]] std::vector<PromptBoundaryAlias>
    boundary_aliases(const PreparedPrompt& prompt) const;
    // Drains the images the lane captured so far during its prefill, ascending by depth.
    [[nodiscard]] std::vector<CapturedContinuation>
    take_captured_continuations_lane(std::uint32_t lane);
    // Metadata preflight is a negative filter and upper bound only. A nonzero result never
    // authorizes import without resolving and exactly preflighting the complete image.
    [[nodiscard]] std::uint32_t preflight_continuation_metadata(
        const cache::SessionCandidateDescriptor& candidate,
        const PreparedPrompt& prompt) const noexcept;
    // Returns the deepest exactly matching reusable frontier: the execution frontier, a saved
    // turn-checkpoint boundary, or zero when the image cannot be reused safely.
    // divergence_tokens, when given, receives the leading tokens the prompt and the image
    // agree on. It is written whether or not the image is reusable, so a rejection reports where
    // the streams parted instead of only that they did.
    [[nodiscard]] std::uint32_t
    preflight_continuation(const cache::ContinuationImage& image, const PreparedPrompt& prompt,
                           std::uint32_t* divergence_tokens = nullptr) const noexcept;
    // Materialises the image as an owned host payload. This is the largest CPU term in a restore
    // and needs no lane, no device and no CUDA call, so the engine may run it on a preparation
    // thread concurrently with GPU execution. Throws on a malformed or incompatible image.
    [[nodiscard]] std::shared_ptr<DecodedContinuation>
    decode_continuation(const cache::ContinuationImage& image) const;
    // Imports an already-decoded image into an empty lane. `decoded` must be the payload produced
    // by decode_continuation() for this exact image; the caller binds the two by content identity.
    // `entitlement` is the caller's planned KV allocation: the restored lane is reserved for it
    // rather than for the bare frontier, so it does not have to grow on its first decode round.
    // Returns ContinuationRestoreFailure::None on success; any other value names the step that
    // refused and leaves the lane empty and the shared KV pool untouched.
    [[nodiscard]] ContinuationRestoreFailure
    import_continuation_lane(std::uint32_t lane, const cache::ContinuationImage& image,
                             const DecodedContinuation& decoded, const PreparedPrompt& prompt,
                             std::int32_t adapter, runtime::KvPageFootprint entitlement) noexcept;
    // Whether the shared paged-KV pools can currently satisfy a reservation of this size. The
    // engine uses this to choose a restore target before it disturbs any retained lane.
    [[nodiscard]] bool kv_reservation_fits(std::uint32_t text_pages,
                                           std::uint32_t backend_pages) const noexcept;
    // Extend a running lane's KV entitlement so it can execute one more decode round, bounded by
    // the ceiling its request was planned with. A request reserves a decode window at admission
    // and acquires the rest of `max_tokens` through this call as it generates. False means the
    // pool is full or the request has reached its ceiling; the lane is left untouched.
    [[nodiscard]] bool try_grow_decode_headroom(std::uint32_t lane);
    // Pages the pools would release if this retained lane were evicted.
    [[nodiscard]] runtime::KvPageFootprint
    retained_lane_kv_footprint(std::uint32_t lane) const noexcept;
    [[nodiscard]] std::uint32_t retained_lane_depth(std::uint32_t lane) const noexcept;
    // Frontier of the retained lane's turn checkpoint, the depth an export reports as
    // `boundary_tokens`; zero without a retained session or a valid checkpoint.
    [[nodiscard]] std::uint32_t retained_lane_boundary_tokens(std::uint32_t lane) const noexcept;
    // Stable identifier (FNV-1a 64 hex) of the lane's resident token ledger; empty unless the
    // lane holds a retained session.
    [[nodiscard]] std::string retained_lane_digest(std::uint32_t lane) const;
    // Retained turn checkpoints of the lane's resident session, oldest first: the frontiers a
    // diverging prompt can restore from, each with the digest of the ledger prefix it covers.
    [[nodiscard]] std::vector<SlotCheckpoint>
    retained_lane_checkpoints(std::uint32_t lane) const;
    // Session persistence for one idle retained lane. `model_binding` pins the snapshot to the
    // serving weights identity; restore rejects a mismatched binding or configuration. Both
    // synchronize the device before returning and require the lane to hold no active request.
    [[nodiscard]] RetainedSessionSnapshot save_retained_lane(std::uint32_t lane,
                                                             std::string_view model_binding);
    // `release_retained` is used only when the snapshot names a LoRA adapter that is not
    // resident and a slot must be freed for it; see ensure_adapter_resident.
    [[nodiscard]] std::uint32_t
    restore_retained_lane(std::uint32_t lane, std::span<const std::uint8_t> snapshot,
                          std::string_view model_binding,
                          const std::function<void(std::uint32_t)>& release_retained);
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;

private:
    explicit Program(std::unique_ptr<detail::ProgramImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::ProgramImpl<Variant>> impl_;

    template <class V>
    friend std::unique_ptr<Program<V>> create_program(const typename V::ModelView&,
                                                       typename V::WeightsProfile, SequencePlan<V>&&,
                                                       DeviceContext&, std::string_view,
                                                       std::string_view,
                                                       std::span<const std::uint8_t>);
};

template <class Variant>
[[nodiscard]] SequencePlanner<Variant>
make_sequence_planner(DeviceContext& device, const EngineOptions& options,
                      typename Variant::WeightsProfile weights_profile);

template <class Variant>
[[nodiscard]] std::unique_ptr<Program<Variant>>
create_program(const typename Variant::ModelView& model,
                 typename Variant::WeightsProfile weights_profile, SequencePlan<Variant>&& plan,
                 DeviceContext& device, std::string_view model_id, std::string_view weights_id,
                 std::span<const std::uint8_t> artifact_fingerprint);

} // namespace ninfer::targets::qwen3_8
