#pragma once
#include "targets/qwen3_8/impl/runtime/instance.h"
// Qwen3.8 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/gdn_replay_records.h"
#include "core/pinned_transfer.h"
#include "core/linear_attention_state.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"
#include "runtime/cache/continuation_cache.h"
#include <ninfer/targets/qwen3_8/decision.h>
#include <ninfer/targets/qwen3_8/prepared_prompt.h>

#include "targets/qwen3_8/impl/runtime/layouts.h"
#include "targets/qwen3_8/impl/runtime/dflash_context.h"
#include "targets/qwen3_8/impl/runtime/linear_state_slots.h"
#include "targets/qwen3_8/impl/runtime/prefix_identity.h"
#include "targets/qwen3_8/impl/runtime/schedule.h"
#include "targets/qwen3_8/impl/runtime/text_context.h"
#include "targets/qwen3_8/impl/runtime/vision_context.h"
#include "targets/qwen3_8/impl/runtime/vision_prefill.h"

#include <cstdint>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_8::detail::NINFER_QWEN38_RUNTIME_NS {

using PreparedPromptData = qwen3_8::PreparedPromptData;

using ReusePath = ninfer::PrefixReusePath;

enum class TurnCheckpointAction : std::uint8_t {
    Drop,
    KeepExisting,
    CaptureNew,
};

enum class MtpBridgeMode : std::uint8_t {
    None,
    BeforeSuffix,
    AfterExactHit,
};

// The unit sequence of one System One decision (decision.h), fixed when the request is planned.
// The state prefills in `prefill_chunk` chunks. Short branches are packed first-fit in request
// order into passes of at most `pass_columns`; each pass is one weight stream in which every
// branch continues the state independently and attends its own columns from the pass, so a pass
// writes no KV page. A branch longer than a pass runs alone as a long branch, chunk by chunk,
// over a copy of the state's GDN slot, with its KV in pages above the state.
struct DecisionPlan {
    struct Pass {
        // Branch indices in pass column order, and the columns they occupy together.
        std::vector<std::uint32_t> branches;
        std::uint32_t columns = 0;
    };

    std::vector<Pass> passes;
    // Branch indices in request order.
    std::vector<std::uint32_t> long_branches;
    std::uint32_t state_tokens   = 0;
    std::uint32_t input_tokens   = 0;
    std::uint32_t pass_columns   = 0;
    // KV tokens a lane holds above the state for its longest long branch (zero without one);
    // never interpreted afterwards.
    std::uint32_t scratch_extent = 0;
    // Passes plus long-branch chunks: every unit after the state.
    std::uint64_t branch_units = 0;
    // 1 / (sqrt(pointer_dim) * T): the pointer logit scale of the adapter's calibrated head.
    float logit_scale = 0.0F;
    // Keep the state [0, state_tokens) resident as a retained lane after the decision.
    bool retain_state = false;
};

} // namespace ninfer::targets::qwen3_8::detail::NINFER_QWEN38_RUNTIME_NS

namespace ninfer::targets::qwen3_8::detail {

// Output tokens reserved when a request is admitted. The rest of `max_tokens` is acquired per
// round as it is actually generated.
//
// 4096 is chosen from the measured workload: agent turns generate a mean of 263 and a p90 of 331
// tokens, so this covers all but the rarest turn outright, and it is the largest window at which
// four p90-sized sessions still fit the 262144-token pool together
// (4 x (58938 + 4096) = 252136). A larger window buys a stronger guarantee only by giving up a
// concurrent lane at the sizes this engine actually sees.
inline constexpr std::uint32_t kDecodeReservationWindow = 4096;

// Tokens acquired per growth step once a lane passes its reserved window. Raising the entitlement
// one round at a time would ask the pool for a new page every 64 tokens; 512 amortizes that
// without holding a window a short turn would never reach.
inline constexpr std::uint32_t kDecodeGrowthChunkTokens = 512;

template <>
struct RequestBasePlanImpl<NINFER_QWEN38_VARIANT> {
    runtime::RequestPlanSummary summary;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    // The pages the lane may grow to, which is the prompt plus the whole effective output. The
    // entitlement above is the bounded window actually reserved at admission.
    std::uint32_t text_kv_page_ceiling        = 0;
    std::uint32_t backend_kv_page_ceiling     = 0;
    std::shared_ptr<const qwen3_8::VisionControl> vision_control;
    std::size_t vision_transient_bytes = 0;
    std::optional<std::uint32_t> turn_rewrite_boundary;
    std::optional<std::uint32_t> user_turn_boundary;
    // Content-addressed boundaries copied from the prompt: strictly ascending, all inside the
    // prompt and none past the turn-rewrite frontier.
    std::vector<PromptBoundary> boundaries;
    // Requested LoRA bank index, or -1 for the base weights. Resident KV and GDN state are only
    // reusable by the adapter that produced them.
    std::int32_t adapter    = -1;
    bool allow_prefix_reuse = false;
    // Present for a System One decision, which prefills its state and branches and reads its
    // pointer head instead of generating.
    std::shared_ptr<const NINFER_QWEN38_RUNTIME_NS::DecisionPlan> decision;
};

template <>
struct RequestPlanImpl<NINFER_QWEN38_VARIANT> {
    runtime::RequestPlanSummary summary;
    NINFER_QWEN38_RUNTIME_NS::ReusePath reuse = NINFER_QWEN38_RUNTIME_NS::ReusePath::FullReset;
    std::uint32_t reuse_base                  = 0;
    NINFER_QWEN38_RUNTIME_NS::MtpBridgeMode mtp_bridge =
        NINFER_QWEN38_RUNTIME_NS::MtpBridgeMode::None;
    bool prepare_mtp = false;
    std::optional<NINFER_QWEN38_RUNTIME_NS::VisionPrefillPlan> vision;
    NINFER_QWEN38_RUNTIME_NS::TurnCheckpointAction turn_checkpoint_action =
        NINFER_QWEN38_RUNTIME_NS::TurnCheckpointAction::Drop;
    std::optional<std::uint32_t> turn_checkpoint_capture_frontier;
    std::optional<std::uint32_t> user_turn_capture_frontier;
    bool keep_user_turn_anchor = false;
    // Boundary depths this request captures a continuation image at, ascending, all past
    // `reuse_base`.
    std::vector<std::uint32_t> capture_frontiers;
    ops::SamplingConfig sampling;
    std::int32_t adapter                      = -1;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    std::uint32_t text_kv_page_ceiling        = 0;
    std::uint32_t backend_kv_page_ceiling     = 0;
    std::shared_ptr<const NINFER_QWEN38_RUNTIME_NS::DecisionPlan> decision;
};

} // namespace ninfer::targets::qwen3_8::detail

namespace ninfer::targets::qwen3_8::detail::NINFER_QWEN38_RUNTIME_NS {

using RequestPlanImpl     = qwen3_8::detail::RequestPlanImpl<Variant>;
using RequestBasePlanImpl = qwen3_8::detail::RequestBasePlanImpl<Variant>;

enum class PendingKind : std::uint8_t {
    None,
    Begin,
    Ordinary,
    Speculative,
};

struct PendingCandidate {
    PendingKind kind            = PendingKind::None;
    std::uint32_t base_E        = 0;
    std::uint32_t base_S        = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t produced      = 0;
};

enum class Lifecycle : std::uint8_t {
    Empty,
    Prefilling,
    Active,
    Pending,
    Complete,
};

struct TurnCheckpoint {
    bool valid             = false;
    std::uint32_t frontier = 0;
};

// Second, host-resident anchor pinned at the opener of the last real user query. The device
// TurnCheckpoint tracks the newest generation opener and therefore dies whenever a client
// rewrites the tail of an earlier user message; this one sits upstream of that edit. It is held
// on the host because it is read at most once per user turn, so a device slot (147 MiB per lane)
// would buy microseconds that nothing waits for.
struct UserTurnAnchor {
    bool valid             = false;
    std::uint32_t frontier = 0;
    LinearAttentionStateImage linear_state;
    std::vector<std::uint8_t> tail_hidden;
};

// Minimum ledger distance between two retained ring checkpoints. Denser entries are folded
// into their deeper neighbour when the ring is compacted (llama.cpp uses 8192 for the same
// policy; pi turns are shorter, so half that keeps recent turns individually restorable).
inline constexpr std::uint32_t kTurnCheckpointMinStep = 4096;

// One host-retained turn checkpoint: the full linear-attention state image (per-layer conv +
// recurrent) and the boundary hidden vector at `frontier`. The attention KV needs no copy -
// entries stay valid only while the resident ledger prefix below `frontier` is untouched, so
// the lane's paged KV still holds those positions. session_digest hashes the ledger prefix.
//
// The ring and the UserTurnAnchor are independent rewind points and are not redundant: ring
// entries land wherever the device checkpoint policy placed a checkpoint, which is always after
// the last user message's content, while the anchor is pinned at that message's opener.
struct HostTurnCheckpoint {
    std::uint32_t frontier = 0;
    std::string session_digest;
    std::vector<std::uint8_t> hidden;
    std::vector<std::uint8_t> conv;
    std::vector<std::uint8_t> recurrent;
};

// In-flight device-to-host copy of the newest captured checkpoint. The copy is enqueued on the
// engine stream right after prefill captures the device checkpoint slot; the next request for
// the lane drains it into the ring (event-synchronized) or discards it when the divergence
// point invalidated it.
struct CheckpointStaging {
    bool pending           = false;
    std::uint32_t frontier = 0;
    std::string session_digest;
};

struct SequenceKVBundle {
    PagedKVAllocation text;
    std::optional<PagedKVAllocation> backend;
};

struct DecodeGraphProfile {
    std::uint32_t batch_size             = 1;
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    std::uint32_t topology_class         = 0;
    DecodeGraphDefinition definition;
};

struct DecodeGraphTopology {
    std::uint32_t topology_class = 0;
    DecodeGraphExecutable executable;
    std::optional<std::size_t> installed_profile;
};

struct DecodeGraphFamily {
    std::vector<DecodeGraphProfile> profiles;
    std::vector<DecodeGraphTopology> topologies;
};

// Target model continuation for one logical sequence. This state remains meaningful after the
// request which produced it has finished, so it is deliberately separate from request lifecycle,
// output, sampling, and round-control state.
struct SequenceState {
    std::optional<SequenceKVBundle> kv;
    Tensor tail_hidden;
    Tensor turn_checkpoint_hidden;
    std::uint32_t lane = 0;
    // Pool index of the adapter that produced this continuation, or -1 for the base weights. KV
    // and GDN recurrent state are only valid for the adapter that produced them, so this is part
    // of the sequence identity rather than request state. It is deliberately the pool index and
    // not the bank slot: a slot is reassigned as adapters are swapped, and comparing slots would
    // let a lane splice one adapter's state into another's request.
    std::int32_t adapter = -1;

    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier    = 0;
    std::vector<TokenId> ledger;
    qwen3_8::detail::ResidentPrefixIdentity prefix_identity;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::array<TokenId, qwen3_8::kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count = 0;
    bool tail_hidden_valid        = false;
    bool retained                 = false;
    // The retained state is a System One decision state [0, execution_frontier): no sampled
    // token, tail hidden, backend KV or checkpoint. Only a later decision of the same adapter
    // continues it; it is never a generation prefix.
    bool decision_state = false;
    TurnCheckpoint turn_checkpoint;
    UserTurnAnchor user_turn_anchor;
    // Host ring of past turn checkpoints, ascending by frontier. Populated only when the
    // Program was planned with a non-zero turn checkpoint ring.
    std::vector<HostTurnCheckpoint> checkpoint_ring;
    // Images captured at boundary frontiers during the current prefill, ascending by depth,
    // awaiting the engine to drain them for publication.
    std::vector<CapturedContinuation> captured_continuations;
};

// Request/round control is not retained with a reusable SequenceState. A later concurrent Engine
// gives every occupied request slot its own instance of this state.
struct RequestControl {
    Lifecycle lifecycle = Lifecycle::Empty;
    PendingCandidate pending;
    ops::SamplingConfig sampling_host;
    GenerationTimings timings;
    SpeculativeStats speculative_stats;
    // Upper bound for on-demand KV growth: the pages this request would have reserved outright
    // before the decode window was bounded. Growth never goes past it.
    std::uint32_t text_kv_page_ceiling    = 0;
    std::uint32_t backend_kv_page_ceiling = 0;

    struct Prefill {
        PreparedPromptData prompt;
        std::optional<VisionPrefillPlan> vision_plan;
        std::unique_ptr<schedule::VisionPrefillSession> vision;
        runtime::TransientRegion transient;
        std::optional<std::uint32_t> turn_checkpoint_capture_frontier;
        std::optional<std::uint32_t> user_turn_capture_frontier;
        // Remaining boundary capture frontiers, ascending; the front is popped when crossed.
        std::vector<std::uint32_t> capture_frontiers;
        std::uint32_t base               = 0;
        std::uint32_t cursor             = 0;
        std::uint32_t prompt_tokens      = 0;
        std::uint32_t initial_mtp_extent = 0;
        double elapsed_seconds           = 0.0;
        bool host_input_consumed_pending = false;
        bool prepare_mtp                 = false;
        ReusePath reuse                  = ReusePath::FullReset;
        MtpBridgeMode mtp_bridge         = MtpBridgeMode::None;
    };

    std::optional<Prefill> prefill;

    // A System One decision in progress. Its lane stays Prefilling until every unit ran; it never
    // licenses a token, samples or drafts.
    struct Decision {
        DecisionPrompt prompt;
        std::shared_ptr<const DecisionPlan> plan;
        std::optional<VisionPrefillPlan> vision_plan;
        std::unique_ptr<schedule::VisionPrefillSession> vision;
        bool host_input_consumed_pending  = false;
        std::uint32_t reused_state_tokens = 0;
        std::uint32_t state_cursor        = 0;
        std::size_t next_pass             = 0;
        std::size_t next_long             = 0;
        // Branch tokens of the current long branch already computed, its row (state then
        // branch, the addressing prefill chunks use) and the option keys projected so far.
        std::uint32_t long_cursor = 0;
        std::vector<TokenId> long_row;
        std::uint32_t long_keys = 0;
        DecisionOutcome outcome;
        bool complete = false;
    };

    std::optional<Decision> decision;
};

class ProgramImplCore {
public:
    ProgramImplCore(const LoadedModelData& model, const SequencePlanImpl& plan,
                    DeviceContext& device, std::string_view model_id, std::string_view weights_id,
                    std::span<const std::uint8_t> artifact_fingerprint);
    ~ProgramImplCore() noexcept;

    [[nodiscard]] RequestBasePlan
    plan_request_base(const PreparedPromptData& prompt,
                      const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan plan_request_for_lane(std::uint32_t lane,
                                                    const PreparedPromptData& prompt,
                                                    const RequestBasePlan& base,
                                                    std::span<const std::uint32_t> capture_depths);
    // System One decisions (decision_impl.h). A decision plans into the same base and lane plans
    // as a generation; its lane runs through advance_prefill_lane and completes through
    // take_decision_lane.
    [[nodiscard]] RequestBasePlan plan_decision_base(const DecisionPrompt& prompt,
                                                     const runtime::ResolvedDecisionOptions& options);
    [[nodiscard]] RequestPlan plan_decision_for_lane(std::uint32_t lane, const DecisionPrompt& prompt,
                                                     const RequestBasePlan& base);
    [[nodiscard]] runtime::PrefillStepResult
    start_decision_lane(std::uint32_t lane, DecisionPrompt&& prompt, RequestPlan&& plan,
                        runtime::TransientRegion transient);
    [[nodiscard]] DecisionOutcome take_decision_lane(std::uint32_t lane);
    [[nodiscard]] bool retained_lane_holds_decision(std::uint32_t lane) const noexcept {
        return lane < max_concurrency && sequences[lane].retained &&
               sequences[lane].decision_state;
    }
    // Image kind `decision_state` (decision_impl.h): a retained decision state published to the
    // continuation cache under the alias of its exact state row, and restored only by a decision
    // of exactly that state and adapter.
    [[nodiscard]] std::optional<std::string> decision_state_alias(const DecisionPrompt& prompt) const;
    [[nodiscard]] cache::ContinuationImage export_decision_state_background(std::uint32_t lane) const;
    [[nodiscard]] std::uint32_t
    preflight_decision_state_metadata(const cache::SessionCandidateDescriptor& candidate,
                                      const DecisionPrompt& prompt) const noexcept;
    [[nodiscard]] std::uint32_t
    preflight_decision_state(const cache::ContinuationImage& image, const DecisionPrompt& prompt,
                             std::int32_t adapter,
                             std::uint32_t* divergence_tokens = nullptr) const noexcept;
    [[nodiscard]] std::shared_ptr<DecodedContinuation>
    decode_decision_state(const cache::ContinuationImage& image) const;
    [[nodiscard]] ContinuationRestoreFailure
    import_decision_state_lane(std::uint32_t lane, const cache::ContinuationImage& image,
                               const DecodedContinuation& decoded, const DecisionPrompt& prompt,
                               std::int32_t adapter, runtime::KvPageFootprint entitlement) noexcept;
    // LoRA residency. `SequenceState::adapter` is a pool index - the adapter's identity, stable
    // for the process - while the device bank holds a bounded number of slots. This resolves one
    // to the other, staging the adapter over the least recently used slot that no generating
    // lane depends on. Returns true when the adapter is resident afterwards.
    //
    // Retained lanes holding the displaced adapter are released through `release_retained`
    // rather than dropped here, so the caller publishes their sessions and keeps its own L1
    // accounting straight.
    //
    // Returns false, changing nothing, when every slot is held by a generating lane using
    // another adapter; the caller defers the request. This cannot deadlock: an idle engine has
    // no generating lane, so a request reaching the head of an empty engine always fits. The
    // base weights (-1) are always resident.
    [[nodiscard]] bool
    ensure_adapter_resident(std::int32_t adapter,
                            const std::function<void(std::uint32_t)>& release_retained);
    // Stable cache scope for a pool adapter: "base", or its artifact content fingerprint as hex.
    [[nodiscard]] std::string adapter_scope(std::int32_t adapter) const;
    [[nodiscard]] std::uint64_t lora_stage_count() const noexcept { return lora_stage_count_; }

    [[nodiscard]] double lora_stage_seconds() const noexcept { return lora_stage_seconds_; }

    [[nodiscard]] std::vector<LoraSlotState> lora_slot_states() const;

    [[nodiscard]] std::int32_t retained_lane_adapter(std::uint32_t lane) const noexcept {
        return lane < max_concurrency && sequences[lane].retained ? sequences[lane].adapter : -1;
    }

    [[nodiscard]] bool can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept;
    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                           const RequestPlan& plan) const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                                PreparedPromptData&& prompt,
                                                                RequestPlan&& plan,
                                                                runtime::TransientRegion transient);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets);
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    // Complete a generating lane between rounds, retaining its session. Used when the KV pool
    // cannot be grown to execute one more round.
    void retire_lane(std::uint32_t lane);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled);
    void abort_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] std::size_t retained_lane_resident_bytes(std::uint32_t lane) const noexcept;
    [[nodiscard]] std::size_t retained_lane_reused_bytes(
        std::uint32_t lane, const RequestPlanImpl& plan) const noexcept;
    void evict_retained_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] cache::ContinuationImage export_continuation_lane(std::uint32_t lane) const;
    void fence_lane_for_export(std::uint32_t lane);
    [[nodiscard]] cache::ContinuationImage
    export_continuation_lane_background(std::uint32_t lane) const;
    [[nodiscard]] cache::ContinuationImage export_lane_image(std::uint32_t lane,
                                                            PinnedTransferBuffer& transfer,
                                                            cudaStream_t stream) const;
    [[nodiscard]] std::vector<PromptBoundaryAlias>
    boundary_aliases(const PreparedPromptData& prompt) const;
    [[nodiscard]] std::vector<CapturedContinuation>
    take_captured_continuations_lane(std::uint32_t lane);
    [[nodiscard]] std::uint32_t preflight_continuation_metadata(
        const cache::SessionCandidateDescriptor& candidate,
        const PreparedPromptData& prompt) const noexcept;
    [[nodiscard]] std::uint32_t
    preflight_continuation(const cache::ContinuationImage& image, const PreparedPromptData& prompt,
                           std::uint32_t* divergence_tokens = nullptr) const noexcept;
    [[nodiscard]] std::set<std::string> continuation_segment_inventory(
        bool boundary_valid, bool mtp_backend, bool dflash_backend) const;
    [[nodiscard]] std::shared_ptr<DecodedContinuation>
    decode_continuation(const cache::ContinuationImage& image) const;
    [[nodiscard]] ContinuationRestoreFailure
    import_continuation_lane(std::uint32_t lane, const cache::ContinuationImage& image,
                             const DecodedContinuation& decoded, const PreparedPromptData& prompt,
                             std::int32_t adapter, runtime::KvPageFootprint entitlement) noexcept;
    [[nodiscard]] bool kv_reservation_fits(std::uint32_t text_pages,
                                           std::uint32_t backend_pages) const noexcept;
    // Raise the running lane's KV entitlement so it can execute one more decode round, up to
    // the ceiling its request was planned with. Returns false, changing nothing, when the pool has
    // no room or the request has reached its ceiling. Growth is idempotent and never shrinks.
    [[nodiscard]] bool try_grow_decode_headroom(std::uint32_t lane);
    [[nodiscard]] runtime::KvPageFootprint
    retained_lane_kv_footprint(std::uint32_t lane) const noexcept;
    [[nodiscard]] std::uint32_t retained_lane_depth(std::uint32_t lane) const noexcept;
    [[nodiscard]] std::uint32_t retained_lane_boundary_tokens(std::uint32_t lane) const noexcept;
    [[nodiscard]] std::string retained_lane_digest(std::uint32_t lane) const;
    [[nodiscard]] std::vector<SlotCheckpoint>
    retained_lane_checkpoints(std::uint32_t lane) const;
    [[nodiscard]] qwen3_8::RetainedSessionSnapshot
    save_retained_lane(std::uint32_t lane, std::string_view model_binding);
    [[nodiscard]] std::uint32_t
    restore_retained_lane(std::uint32_t lane, std::span<const std::uint8_t> snapshot,
                          std::string_view model_binding,
                          const std::function<void(std::uint32_t)>& release_retained);
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;

    void reset_memory_peaks() noexcept;

    const LoadedModelData& model;
    DeviceContext& device;
    const std::uint32_t capacity;
    const std::uint32_t kv_capacity;
    const std::uint32_t max_concurrency;
    const std::uint32_t prefill_chunk;
    const std::uint32_t draft_window;
    const SpeculativeBackend speculative_backend;
    const DType kv_dtype;
    const std::int32_t kv_quant_group;
    const bool kv_packed_v;
    const bool kv_rotate_k;
    const bool kv_rotate_v;
    const bool kv_packed_k;
    const bool kv_e8_root;
    const ProposalHead proposal_head;
    const bool vision_enabled;
    const bool use_cuda_graph;
    const std::uint32_t checkpoint_ring_capacity;
    const std::size_t kv_payload_bytes;
    const std::size_t text_kv_bytes;
    const std::size_t mtp_kv_bytes;
    const std::size_t gdn_state_bytes;
    const std::size_t dflash_kv_bytes;
    const std::size_t replay_records_bytes;
    const std::size_t graph_allowance_bytes;
    std::size_t graph_observed_bytes = 0;
    const WorkspacePlan workspace_plan;
    const cache::Bytes continuation_compatibility_key;
    const cache::Bytes decision_state_compatibility_key;

    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<qwen3_8::DecoderState> decoder;
    std::optional<GdnReplayRecords> replay_records;
    std::optional<DFlashPersistentState> dflash;
    qwen3_8::RoundState io;
    Tensor prefill_hidden;
    Tensor sampling_config;
    Tensor token_counts;
    Tensor tail_hidden_store;
    Tensor turn_checkpoint_hidden_store;
    // Per lane, the option keys of the long decision branch in flight (FP32
    // [pointer_dim, kMaximumDecisionOptions, lanes]); empty for a target without decisions.
    Tensor decision_keys;
    // Pinned landing buffer of one decision unit's probabilities.
    std::optional<PinnedHostBuffer> decision_host;

    std::array<SequenceState, kMaximumConcurrency> sequences;
    std::array<RequestControl, kMaximumConcurrency> requests;

    DecodeGraphFamily ordinary_graphs;
    DecodeGraphFamily mtp_graphs;
    DecodeGraphFamily dflash_graphs;

    PinnedHostBuffer round_host;
    // One bounded startup allocation for all continuation payload movement (8 MiB per ring slot).
    mutable PinnedTransferBuffer continuation_transfer;
    // Background continuation export: its own stream and pinned ring so the publication worker
    // can copy a completed lane out while the engine stream keeps decoding the other lanes, plus
    // one fence event per lane ordering the export after the lane's last engine-stream work.
    cudaStream_t export_stream = nullptr;
    mutable PinnedTransferBuffer export_transfer;
    std::array<cudaEvent_t, kMaximumConcurrency> export_fence_events{};
    TokenId* host_tokens = nullptr;
    std::optional<PinnedHostBuffer> ordinary_host;
    qwen3_8::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    qwen3_8::OrdinaryDecodeEgress* ordinary_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> mtp_host;
    qwen3_8::MtpDecodeIngress* mtp_host_ingress = nullptr;
    qwen3_8::MtpDecodeEgress* mtp_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> dflash_host;
    qwen3_8::DFlashDecodeIngress* dflash_host_ingress = nullptr;
    qwen3_8::DFlashDecodeEgress* dflash_host_egress   = nullptr;

    // One pinned staging entry per lane (hidden + per-layer conv + per-layer recurrent) for
    // asynchronous checkpoint capture, plus the per-lane copy-completion events.
    std::optional<PinnedHostBuffer> checkpoint_staging_store;
    std::array<CheckpointStaging, kMaximumConcurrency> checkpoint_staging{};
    std::array<cudaEvent_t, kMaximumConcurrency> checkpoint_staging_events{};

    std::size_t workspace_logical_peak_bytes = 0;

private:
    // Translates a pool adapter into the bank slot it occupies. -1 for the base weights, which
    // the kernels read as "leave this column alone". A live sequence whose adapter is not
    // resident is a scheduling fault, not a recoverable condition, because admission is the only
    // place residency is decided.
    [[nodiscard]] std::int32_t lora_slot(std::int32_t adapter) const;
    // A slot is pinned while a lane that is prefilling, decoding or deciding executes against its
    // occupant: swapping the bytes underneath would change that lane's arithmetic mid-request. A
    // lane that merely retains state its occupant produced does not pin it; ensure_adapter_resident
    // demotes such a lane through the caller before the swap.
    [[nodiscard]] bool lora_slot_pinned(std::size_t slot) const noexcept;

    // Pool index resident in each slot, or -1 when the slot has never been staged. Parallel LRU
    // stamps; the clock advances on every resolution, not only on a stage, so a repeatedly
    // selected adapter keeps its slot.
    std::vector<std::int32_t> lora_slot_pool_;
    std::vector<std::uint64_t> lora_slot_used_;
    std::uint64_t lora_clock_       = 0;
    std::uint64_t lora_stage_count_ = 0;
    // Execution-thread time of the stages counted above, from preparing the slab through the
    // demotion of retained lanes to the completed upload.
    double lora_stage_seconds_ = 0.0;

    [[nodiscard]] cache::ContinuationImage
    export_stable_continuation(const SequenceState& sequence, const PreparedPromptData& prompt,
                               std::uint32_t frontier) const;
    void clear_lane(SequenceState& sequence, RequestControl& request) noexcept;
    void ordered_reset(SequenceState& sequence);
    [[nodiscard]] std::size_t checkpoint_hidden_bytes() const noexcept;
    [[nodiscard]] std::size_t checkpoint_conv_bytes() const noexcept;
    [[nodiscard]] std::size_t checkpoint_recurrent_bytes() const noexcept;
    [[nodiscard]] std::size_t checkpoint_entry_bytes() const noexcept;
    [[nodiscard]] std::uint8_t* checkpoint_staging_base(std::uint32_t lane) const noexcept;
    void stage_turn_checkpoint(SequenceState& sequence);
    void drain_checkpoint_staging(SequenceState& sequence);
    void discard_checkpoint_staging(SequenceState& sequence) noexcept;
    void invalidate_checkpoint_ring(SequenceState& sequence,
                                    std::uint32_t keep_through) noexcept;
    [[nodiscard]] bool upload_ring_checkpoint(SequenceState& sequence, std::uint32_t frontier);
    void append_ring_checkpoint(SequenceState& sequence, HostTurnCheckpoint&& entry);
    void prepare_graphs();
    void install_sampling(SequenceState& sequence, RequestControl& request,
                          const ops::SamplingConfig& config);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    void copy_tail(SequenceState& sequence, const Tensor& source);
    void copy_round_token();
    void resolve_non_speculative_pending(SequenceState& sequence, RequestControl& request,
                                         std::uint32_t accepted_tokens, bool terminal);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill(SequenceState& sequence,
                                                             RequestControl& request);
    // Several lanes prefill interleaved while every prefill chunk addresses its KV through the
    // shared single-sequence row selectors, so each step re-selects its own lane's rows.
    void select_prefill_kv_rows(const SequenceState& sequence);
    [[nodiscard]] runtime::PrefillStepResult advance_decision(SequenceState& sequence,
                                                              RequestControl& request);
    [[nodiscard]] schedule::PrefillContext decision_prefill_context(SequenceState& sequence,
                                                                    std::uint32_t base,
                                                                    std::int32_t current_slot);
    void decision_pass_unit(SequenceState& sequence, RequestControl::Decision& decision,
                            const DecisionPlan::Pass& pass);
    // One chunk of the current long branch; returns the branch tokens it computed.
    [[nodiscard]] std::uint32_t decision_long_unit(SequenceState& sequence,
                                                   RequestControl::Decision& decision);
    void enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                       std::span<const std::uint32_t> starts,
                                       std::span<const std::uint32_t> counts);
    void validate_licensed_tokens(std::span<const TokenId> tokens) const;
    void mark_workspace_usage(std::size_t phase_bytes) noexcept;
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                          std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_mtp_batch(std::span<const std::uint32_t> lanes,
                     std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash_batch(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets);
    void reserve_sequence_kv(SequenceState& sequence, std::uint32_t text_pages,
                             std::uint32_t backend_pages);
    void resize_sequence_kv_entitlement(SequenceState& sequence, std::uint32_t text_pages,
                                        std::uint32_t backend_pages);
    void bind_sequence_kv(SequenceState& sequence);
    void unbind_sequence_kv(SequenceState& sequence) noexcept;
    void materialize_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                 std::uint32_t backend_tokens = 0);
    void trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                          std::uint32_t backend_tokens = 0);
    void release_sequence_growth_entitlement(SequenceState& sequence) noexcept;
    [[nodiscard]] qwen3_8::PagedKVCache* backend_kv_cache() noexcept;
    [[nodiscard]] const qwen3_8::PagedKVCache* backend_kv_cache() const noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid(const SequenceState& sequence) const noexcept;
    [[nodiscard]] qwen3_8::PagedKVCacheView text_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_8::PagedKVCacheView mtp_kv_view(const SequenceState& sequence) const;
};

} // namespace ninfer::targets::qwen3_8::detail::NINFER_QWEN38_RUNTIME_NS

namespace ninfer::targets::qwen3_8::detail {

template <>
class ProgramImpl<NINFER_QWEN38_VARIANT> final : public NINFER_QWEN38_RUNTIME_NS::ProgramImplCore {
public:
    using NINFER_QWEN38_RUNTIME_NS::ProgramImplCore::ProgramImplCore;
};

} // namespace ninfer::targets::qwen3_8::detail
