#include "targets/qwen3_8/impl/runtime/instance.h"
#include "targets/qwen3_8/impl/runtime/program.h"

#include "targets/qwen3_8/impl/runtime/schedule.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_8::detail::NINFER_QWEN38_RUNTIME_NS {
namespace {

void validate_sampling(const ResolvedSamplingParameters& sampling) {
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in [0,1]");
    }
    if (sampling.min_p < 0.0F || sampling.min_p > 1.0F) {
        throw std::invalid_argument("min_p must be in [0,1]");
    }
}

ops::SamplingConfig translate_sampling(const ResolvedSamplingParameters& source) {
    ops::SamplingConfig out;
    out.temperature       = source.temperature;
    out.top_k             = source.top_k;
    out.top_p             = source.top_p;
    out.min_p             = source.min_p;
    out.presence_penalty  = source.presence_penalty;
    out.frequency_penalty = source.frequency_penalty;
    out.seed              = source.seed;
    out.token_counts      = nullptr;
    return out;
}

std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
    return 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::uint64_t projected_service_work(const runtime::RequestPlanSummary& summary,
                                     std::uint32_t reuse_base, std::uint32_t prefill_chunk,
                                     std::size_t prefill_splits) noexcept {
    const std::uint32_t suffix = summary.prompt_tokens - reuse_base;
    const std::uint64_t prefill_units =
        suffix == 0
            ? 1ULL
            : 1ULL + (static_cast<std::uint64_t>(suffix) - 1ULL) / prefill_chunk + prefill_splits;
    const std::uint64_t decode_units =
        summary.effective_output_tokens == 0 ? 0ULL : summary.effective_output_tokens - 1ULL;
    return prefill_units + decode_units;
}

} // namespace

RequestBasePlan
ProgramImplCore::plan_request_base(const PreparedPromptData& prompt,
                                   const runtime::ResolvedExecutionOptions& options) {
    if (prompt.token_ids.empty()) { throw std::invalid_argument("prompt must contain tokens"); }
    if (prompt.token_ids.size() > capacity) {
        throw std::invalid_argument("prompt exceeds configured context capacity");
    }
    if (prompt.token_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("prompt token count exceeds uint32");
    }
    for (const TokenId id : prompt.token_ids) {
        if (id < 0 || id >= TextConfig::token_domain) {
            throw std::invalid_argument("prompt contains token outside the 248077-token domain");
        }
    }
    if (prompt.token_types.size() != prompt.token_ids.size() ||
        prompt.positions.size() != 3ULL * prompt.token_ids.size()) {
        throw std::invalid_argument("prepared prompt token metadata has an invalid shape");
    }
    if (prompt.has_media() != !prompt.patches.empty()) {
        throw std::invalid_argument("prepared prompt media payload is incomplete");
    }
    if (prompt.has_media() && !vision_enabled) {
        throw std::invalid_argument("Vision is disabled for this Engine");
    }
    validate_sampling(options.sampling);

    auto base                             = std::make_unique<RequestBasePlanImpl>();
    base->summary.prompt_tokens           = static_cast<std::uint32_t>(prompt.token_ids.size());
    base->summary.requested_output_tokens = options.requested_output_tokens;
    const std::uint32_t capacity_output =
        capacity - base->summary.prompt_tokens + static_cast<std::uint32_t>(1);
    base->summary.effective_output_tokens =
        std::min(options.requested_output_tokens, capacity_output);
    base->summary.effective_limit_reason = options.requested_output_tokens <= capacity_output
                                               ? FinishReason::OutputLimit
                                               : FinishReason::ContextCapacity;
    base->summary.transient_alignment    = 1;
    base->summary.transient_bytes        = 0;
    base->sampling                       = translate_sampling(options.sampling);
    base->adapter                        = options.adapter;
    base->allow_prefix_reuse             = options.allow_prefix_reuse;
    // A lane reserves its prompt plus a bounded decode window, not the client's whole output
    // request. `max_tokens` is a limit; reserving it was a promise to hold pages for tokens that
    // do not exist yet, and a client asking for 32k on a 59k prompt made four lanes need 364k
    // pages against a 262k pool. Tokens beyond the window are acquired per round by
    // try_grow_decode_headroom, whose ceiling is this same full extent, so the entitlement can
    // never exceed what the unconditional reservation would have taken.
    const std::uint32_t full_output_tokens = base->summary.effective_output_tokens == 0
                                                 ? 0U
                                                 : base->summary.effective_output_tokens - 1U;
    const std::uint32_t reserved_context_tokens =
        base->summary.prompt_tokens + std::min(full_output_tokens, kDecodeReservationWindow);
    const std::uint32_t ceiling_context_tokens = base->summary.prompt_tokens + full_output_tokens;
    base->text_kv_page_entitlement = pages_for_tokens(reserved_context_tokens);
    base->text_kv_page_ceiling     = pages_for_tokens(ceiling_context_tokens);
    const auto mtp_extent          = [&](std::uint32_t context_tokens) {
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(
            capacity, static_cast<std::uint64_t>(context_tokens) + draft_window - 1ULL));
    };
    if (speculative_backend == SpeculativeBackend::Mtp) {
        base->backend_kv_page_entitlement = pages_for_tokens(mtp_extent(reserved_context_tokens));
        base->backend_kv_page_ceiling     = pages_for_tokens(mtp_extent(ceiling_context_tokens));
    } else if (speculative_backend == SpeculativeBackend::DFlash) {
        base->backend_kv_page_entitlement = pages_for_tokens(reserved_context_tokens);
        base->backend_kv_page_ceiling     = pages_for_tokens(ceiling_context_tokens);
    }
    base->summary.admission = runtime::AdmissionResources{
        .active_lanes     = 1,
        .main_kv_pages    = base->text_kv_page_entitlement,
        .backend_kv_pages = base->backend_kv_page_entitlement,
    };
    if (prompt.has_media()) {
        auto control =
            std::make_shared<qwen3_8::VisionControl>(qwen3_8::build_vision_control(prompt));
        std::size_t max_merged     = 0;
        std::uint32_t previous_end = 0;
        for (const qwen3_8::VisionItemControl& item : control->items) {
            if (item.scatter_indices.empty()) {
                throw std::invalid_argument("vision item has no Text consumer columns");
            }
            const auto first = static_cast<std::uint32_t>(item.scatter_indices.front());
            const auto last  = static_cast<std::uint32_t>(item.scatter_indices.back());
            const std::uint32_t begin =
                speculative_backend == SpeculativeBackend::Mtp && first != 0 ? first - 1 : first;
            const std::uint32_t end = last + 1;
            if (begin < previous_end) {
                throw std::invalid_argument("vision item consumer spans overlap");
            }
            if (end > base->summary.prompt_tokens) {
                throw std::invalid_argument("vision item consumer span exceeds prompt");
            }
            if (schedule::VisionContext::workspace_bytes(item) > work.capacity()) {
                throw std::invalid_argument("vision item exceeds the Program workspace envelope");
            }
            previous_end = end;
            max_merged   = std::max(max_merged, item.merged_count);
        }
        base->vision_transient_bytes = schedule::VisionContext::output_transient_bytes(max_merged);
        base->vision_control         = std::move(control);
    }

    if (prompt.identity.turn_rewrite_boundary) {
        const std::uint32_t candidate = *prompt.identity.turn_rewrite_boundary;
        if (candidate == 0 || candidate >= base->summary.prompt_tokens) {
            throw std::invalid_argument("turn rewrite boundary must lie inside the prompt");
        }
        base->turn_rewrite_boundary = candidate;
    }
    std::uint32_t previous_boundary = 0;
    std::optional<std::uint32_t> system_tools_boundary;
    for (const PromptBoundary& boundary : prompt.identity.boundaries) {
        if (boundary.depth <= previous_boundary || boundary.depth > base->summary.prompt_tokens) {
            throw std::invalid_argument("prompt boundaries must ascend inside the prompt");
        }
        if (base->turn_rewrite_boundary && boundary.depth > *base->turn_rewrite_boundary) {
            throw std::invalid_argument("prompt boundary must not follow the rewrite boundary");
        }
        if (boundary.kind == PromptBoundaryKind::SystemTools) {
            system_tools_boundary = boundary.depth;
        }
        previous_boundary = boundary.depth;
    }
    base->boundaries = prompt.identity.boundaries;
    if (prompt.identity.user_turn_boundary) {
        const std::uint32_t candidate = *prompt.identity.user_turn_boundary;
        if (candidate == 0 || candidate >= base->summary.prompt_tokens) {
            throw std::invalid_argument("user turn boundary must lie inside the prompt");
        }
        base->user_turn_boundary = candidate;
    }
    // The user-turn anchor is only useful strictly between the system/tools prefix and the
    // rewrite frontier: outside that window it duplicates an anchor the lane already holds.
    if (base->user_turn_boundary &&
        ((system_tools_boundary && *base->user_turn_boundary <= *system_tools_boundary) ||
         (base->turn_rewrite_boundary &&
          *base->user_turn_boundary >= *base->turn_rewrite_boundary))) {
        base->user_turn_boundary.reset();
    }
    // Every publish boundary is a potential chunk split on a cold prefill; one that coincides
    // with the rewrite frontier is already counted.
    std::size_t publish_splits = 0;
    for (const PromptBoundary& boundary : base->boundaries) {
        if (boundary.publish && boundary.depth != base->turn_rewrite_boundary) { ++publish_splits; }
    }
    const std::size_t cold_prefill_splits =
        (base->vision_control != nullptr ? base->vision_control->items.size() : 0ULL) +
        (base->turn_rewrite_boundary ? 1ULL : 0ULL) + (base->user_turn_boundary ? 1ULL : 0ULL) +
        publish_splits;
    base->summary.service_work_quanta =
        projected_service_work(base->summary, 0, prefill_chunk, cold_prefill_splits);
    return RequestBasePlan(std::move(base));
}

RequestPlan ProgramImplCore::plan_request_for_lane(std::uint32_t lane,
                                                   const PreparedPromptData& prompt,
                                                   const RequestBasePlan& base_plan,
                                                   std::span<const std::uint32_t> capture_depths) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    const RequestControl& request = requests[lane];
    const SequenceState& sequence = sequences[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("cannot plan a request while Program is active or pending");
    }
    if (base_plan.impl_ == nullptr) { throw std::logic_error("request base plan is empty"); }
    const RequestBasePlanImpl& base = *base_plan.impl_;

    auto plan                         = std::make_unique<RequestPlanImpl>();
    plan->summary                     = base.summary;
    plan->sampling                    = base.sampling;
    plan->text_kv_page_entitlement    = base.text_kv_page_entitlement;
    plan->backend_kv_page_entitlement = base.backend_kv_page_entitlement;
    plan->text_kv_page_ceiling        = base.text_kv_page_ceiling;
    plan->backend_kv_page_ceiling     = base.backend_kv_page_ceiling;
    plan->adapter                     = base.adapter;

    // A resident prefix produced under a different adapter is not reusable: its KV and GDN
    // recurrent state encode that adapter's weights.
    if (base.allow_prefix_reuse && prompt.identity.reusable && sequence.retained &&
        !sequence.decision_state && sequence.adapter == base.adapter) {
        const bool dflash_append_ready =
            speculative_backend != SpeculativeBackend::DFlash ||
            sequence.dflash_context_frontier == sequence.execution_frontier;
        if (sequence.execution_frontier != 0 && dflash_append_ready &&
            qwen3_8::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                            sequence.execution_frontier)) {
            plan->reuse      = ReusePath::AppendAtFrontier;
            plan->reuse_base = sequence.execution_frontier;
        } else if (sequence.turn_checkpoint.valid && sequence.turn_checkpoint.frontier != 0 &&
                   sequence.turn_checkpoint.frontier < prompt.token_ids.size() &&
                   qwen3_8::detail::prefix_matches(prompt, sequence.ledger,
                                                   sequence.prefix_identity,
                                                   sequence.turn_checkpoint.frontier)) {
            plan->reuse      = ReusePath::RestoreTurnCheckpoint;
            plan->reuse_base = sequence.turn_checkpoint.frontier;
        } else {
            // Both device anchors sit after the last user message's content, so a client that
            // rewrites that message's tail invalidates them. Two host-resident rewind points
            // survive that edit, and they are not redundant: the user-turn anchor is pinned at
            // the opener of the last user query, while ring entries sit at frontiers the device
            // checkpoint policy visited earlier in the session. Take the deepest candidate that
            // still prefix-matches.
            const auto matches = [&](std::uint32_t frontier) {
                return frontier != 0 && frontier < prompt.token_ids.size() &&
                       qwen3_8::detail::prefix_matches(prompt, sequence.ledger,
                                                       sequence.prefix_identity, frontier);
            };
            // DFlash keeps a single turn checkpoint with no second rewind point. Its ring
            // capacity is already forced to zero at plan time, so only the anchor needs an
            // explicit exclusion here.
            std::uint32_t anchor = speculative_backend != SpeculativeBackend::DFlash &&
                                           sequence.user_turn_anchor.valid
                                       ? sequence.user_turn_anchor.frontier
                                       : 0;
            // The ring is ascending by frontier, so a reverse walk with the anchor spliced in
            // at its own depth visits every candidate deepest-first.
            for (auto entry = sequence.checkpoint_ring.rbegin();
                 entry != sequence.checkpoint_ring.rend(); ++entry) {
                if (anchor > entry->frontier) {
                    if (matches(anchor)) {
                        plan->reuse      = ReusePath::RestoreUserTurnAnchor;
                        plan->reuse_base = anchor;
                        break;
                    }
                    anchor = 0;
                }
                if (matches(entry->frontier)) {
                    // Execution re-lands the matched entry in the device checkpoint slot, so
                    // the ordinary RestoreTurnCheckpoint path proceeds unchanged.
                    plan->reuse      = ReusePath::RestoreTurnCheckpoint;
                    plan->reuse_base = entry->frontier;
                    break;
                }
            }
            if (plan->reuse == ReusePath::FullReset && matches(anchor)) {
                plan->reuse      = ReusePath::RestoreUserTurnAnchor;
                plan->reuse_base = anchor;
            }
        }
    }

    if (speculative_backend == SpeculativeBackend::Mtp) {
        const bool append_ready =
            plan->reuse == ReusePath::AppendAtFrontier && sequence.tail_hidden_valid &&
            decoder->mtp_cache() != nullptr &&
            (plan->reuse_base == 0 || sequence.mtp_kv_valid >= plan->reuse_base - 1);
        const bool checkpoint_ready = (plan->reuse == ReusePath::RestoreTurnCheckpoint ||
                                       plan->reuse == ReusePath::RestoreUserTurnAnchor) &&
                                      decoder->mtp_cache() != nullptr &&
                                      sequence.mtp_kv_valid >= plan->reuse_base - 1;
        if (plan->reuse != ReusePath::FullReset && !append_ready && !checkpoint_ready) {
            plan->reuse      = ReusePath::FullReset;
            plan->reuse_base = 0;
        }
    }

    if (plan->reuse == ReusePath::RestoreTurnCheckpoint &&
        speculative_backend == SpeculativeBackend::DFlash &&
        (!dflash || !sequence.kv || !sequence.kv->backend ||
         sequence.dflash_context_frontier < plan->reuse_base)) {
        plan->reuse      = ReusePath::FullReset;
        plan->reuse_base = 0;
    }

    const std::optional<std::uint32_t> desired = base.turn_rewrite_boundary;
    const bool can_keep                        = desired && plan->reuse != ReusePath::FullReset &&
                          sequence.turn_checkpoint.valid &&
                          sequence.turn_checkpoint.frontier == *desired &&
                          qwen3_8::detail::prefix_matches(prompt, sequence.ledger,
                                                          sequence.prefix_identity, *desired);
    if (!desired) {
        plan->turn_checkpoint_action = TurnCheckpointAction::Drop;
    } else if (can_keep) {
        plan->turn_checkpoint_action = TurnCheckpointAction::KeepExisting;
    } else {
        if (*desired <= plan->reuse_base) {
            plan->reuse      = ReusePath::FullReset;
            plan->reuse_base = 0;
        }
        plan->turn_checkpoint_action           = TurnCheckpointAction::CaptureNew;
        plan->turn_checkpoint_capture_frontier = desired;
    }

    // The user-turn anchor is stationary for the whole turn, so within a tool loop it is kept
    // rather than recaptured; it only moves when a new user query arrives.
    const std::optional<std::uint32_t> user_turn = base.user_turn_boundary;
    plan->keep_user_turn_anchor =
        user_turn && plan->reuse != ReusePath::FullReset && sequence.user_turn_anchor.valid &&
        sequence.user_turn_anchor.frontier == *user_turn &&
        qwen3_8::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                        *user_turn);
    if (user_turn && !plan->keep_user_turn_anchor && *user_turn > plan->reuse_base &&
        speculative_backend != SpeculativeBackend::DFlash) {
        plan->user_turn_capture_frontier = user_turn;
    }

    // Capture only at boundaries the caller builds and that this lane will actually prefill
    // through; a depth already covered by the reused prefix has nothing to capture. This follows
    // the reuse decision above so a reset to a cold prefill captures everything it builds.
    std::uint32_t previous_capture = 0;
    for (const std::uint32_t depth : capture_depths) {
        if (depth <= previous_capture) {
            throw std::invalid_argument("capture depths must be strictly ascending");
        }
        previous_capture = depth;
        const bool known = std::any_of(base.boundaries.begin(), base.boundaries.end(),
                                       [&](const PromptBoundary& boundary) {
                                           return boundary.depth == depth && boundary.publish;
                                       });
        if (!known) {
            throw std::invalid_argument("capture depth is not a publish boundary of the prompt");
        }
        if (depth > plan->reuse_base) { plan->capture_frontiers.push_back(depth); }
    }

    plan->summary.reusable_prompt_tokens = plan->reuse_base;
    if (speculative_backend == SpeculativeBackend::Mtp) {
        if (plan->reuse == ReusePath::FullReset) {
            plan->prepare_mtp = true;
        } else if (plan->reuse == ReusePath::AppendAtFrontier) {
            plan->prepare_mtp = true;
            plan->mtp_bridge  = plan->reuse_base < plan->summary.prompt_tokens
                                    ? MtpBridgeMode::BeforeSuffix
                                    : MtpBridgeMode::AfterExactHit;
        } else if (plan->reuse == ReusePath::RestoreTurnCheckpoint ||
                   plan->reuse == ReusePath::RestoreUserTurnAnchor) {
            plan->prepare_mtp = true;
            plan->mtp_bridge  = MtpBridgeMode::BeforeSuffix;
        }
    }

    if (base.vision_control != nullptr) {
        VisionPrefillPlan vision;
        vision.control = base.vision_control;
        vision.uses.reserve(base.vision_control->items.size());
        for (std::size_t index = 0; index < base.vision_control->items.size(); ++index) {
            const qwen3_8::VisionItemControl& item = base.vision_control->items[index];
            const auto first          = static_cast<std::uint32_t>(item.scatter_indices.front());
            const auto last           = static_cast<std::uint32_t>(item.scatter_indices.back());
            const std::uint32_t begin = plan->prepare_mtp && first != 0 ? first - 1 : first;
            const std::uint32_t end   = last + 1;
            if (end <= plan->reuse_base) { continue; }
            vision.uses.push_back(VisionUseSpan{begin, end, static_cast<std::uint32_t>(index)});
        }
        if (!vision.uses.empty()) {
            plan->summary.transient_alignment = 256;
            plan->summary.transient_bytes     = base.vision_transient_bytes;
            plan->vision                      = std::move(vision);
        }
    }

    std::size_t capture_splits = 0;
    for (const std::uint32_t depth : plan->capture_frontiers) {
        if (depth != plan->turn_checkpoint_capture_frontier) { ++capture_splits; }
    }
    const std::size_t prefill_splits =
        (plan->vision ? plan->vision->uses.size() : 0ULL) +
        (plan->turn_checkpoint_capture_frontier ? 1ULL : 0ULL) +
        (plan->user_turn_capture_frontier ? 1ULL : 0ULL) + capture_splits;
    plan->summary.service_work_quanta =
        projected_service_work(plan->summary, plan->reuse_base, prefill_chunk, prefill_splits);
    return RequestPlan(std::move(plan));
}

} // namespace ninfer::targets::qwen3_8::detail::NINFER_QWEN38_RUNTIME_NS
