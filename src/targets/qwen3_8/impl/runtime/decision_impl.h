#include "targets/qwen3_8/impl/runtime/instance.h"
#include "targets/qwen3_8/impl/runtime/program.h"

#include "targets/qwen3_8/impl/runtime/schedule.h"

#include "ninfer/ops/pointer_head.h"
#include "ninfer/ops/scatter.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// System One decisions (decision.h) on the family Program. A decision is a prefill-only lane: its
// state prefills like a prompt, its short branches run as packed passes over the resident state,
// its long branches run chunk by chunk over a copy of the state's GDN slot, and the pointer head
// scores every question from final-norm readouts. It never samples, drafts or checkpoints a turn.
// Included once from instantiate.h after program_impl.h and request_plan_impl.h, whose private
// helpers (Clock, pages_for_tokens) it shares.

namespace ninfer::targets::qwen3_8::detail::NINFER_QWEN38_RUNTIME_NS {
namespace {

std::uint64_t decision_state_units(std::uint32_t state, std::uint32_t reuse_base,
                                   std::uint32_t chunk) noexcept {
    const std::uint32_t suffix = state - reuse_base;
    return suffix == 0 ? 0ULL : 1ULL + (static_cast<std::uint64_t>(suffix) - 1ULL) / chunk;
}

// A decision state is text-only with ordinal positions on every MRoPE axis, exactly the identity
// a text prompt of the same tokens has.
bool ordinal_text_prefix(const ResidentPrefixIdentity& identity, std::uint32_t tokens) {
    if (identity.size() < tokens || !identity.vision_items().empty()) { return false; }
    const std::vector<std::uint8_t>& types = identity.token_types();
    if (std::any_of(types.begin(), types.begin() + tokens,
                    [](std::uint8_t type) { return type != 0; })) {
        return false;
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const std::vector<std::int32_t>& positions = identity.position_axis(axis);
        for (std::uint32_t index = 0; index < tokens; ++index) {
            if (positions[index] != static_cast<std::int32_t>(index)) { return false; }
        }
    }
    return true;
}

bool decision_state_matches(const SequenceState& sequence, const DecisionPrompt& prompt,
                            std::uint32_t frontier) {
    return frontier != 0 && frontier <= prompt.state_tokens() &&
           sequence.ledger.size() >= frontier &&
           std::equal(prompt.tokens.begin(),
                      prompt.tokens.begin() + static_cast<std::ptrdiff_t>(frontier),
                      sequence.ledger.begin()) &&
           ordinal_text_prefix(sequence.prefix_identity, frontier);
}

void assign_ordinal_identity(ResidentPrefixIdentity& identity, std::uint32_t tokens) {
    std::array<std::vector<std::int32_t>, 3> positions;
    for (std::vector<std::int32_t>& axis : positions) {
        axis.resize(tokens);
        std::iota(axis.begin(), axis.end(), 0);
    }
    identity.restore(std::vector<std::uint8_t>(tokens, 0), std::move(positions), {});
}

void upload_i32(const std::vector<std::int32_t>& host, const Tensor& device, cudaStream_t stream) {
    if (device.dtype != DType::I32 || device.data == nullptr ||
        device.bytes() != host.size() * sizeof(std::int32_t)) {
        throw std::logic_error("decision index upload does not match its device view");
    }
    CUDA_CHECK(cudaMemcpyAsync(device.data, host.data(), device.bytes(), cudaMemcpyHostToDevice,
                               stream));
}

// The pointer head planes of one bank slot.
struct DecisionHeadPlanes {
    Tensor query_weight;
    Tensor query_bias;
    Tensor key_weight;
    Tensor key_bias;
};

template <class Lora>
DecisionHeadPlanes decision_head_planes(const Lora& lora, std::int32_t slot) {
    const DecisionHeadWeights& head = lora.decision_head;
    if (!head.present() || slot < 0 || static_cast<std::uint32_t>(slot) >= lora.slots) {
        throw std::logic_error("decision adapter slot holds no pointer head");
    }
    const auto at = [&](const Tensor& plane) {
        Tensor out = plane;
        out.data   = static_cast<std::uint8_t*>(plane.data) +
                   head.slot_stride * static_cast<std::size_t>(slot);
        return out;
    };
    return DecisionHeadPlanes{at(head.query_weight), at(head.query_bias), at(head.key_weight),
                              at(head.key_bias)};
}

// One decision unit's head scratch, in the allocation order build_workspace_plan reserves as
// WorkspacePlan::decision_head.
struct DecisionHeadScratch {
    Tensor query_columns;
    Tensor key_columns;
    Tensor questions;
    Tensor query_hidden;
    Tensor key_hidden;
    Tensor queries;
    Tensor keys;
    Tensor probabilities;
};

DecisionHeadScratch allocate_decision_head(WorkspaceArena& work, std::int32_t chunk) {
    constexpr std::int32_t pointer = Variant::DecisionConfig::pointer_dim;
    const std::int32_t keys = std::max(chunk, static_cast<std::int32_t>(kMaximumDecisionOptions));
    DecisionHeadScratch out;
    out.query_columns = work.alloc(DType::I32, {1, chunk});
    out.key_columns   = work.alloc(DType::I32, {1, keys});
    out.questions     = work.alloc(DType::I32, {3, chunk});
    out.query_hidden  = work.alloc(DType::BF16, {TextConfig::hidden, chunk});
    out.key_hidden    = work.alloc(DType::BF16, {TextConfig::hidden, keys});
    out.queries       = work.alloc(DType::FP32, {pointer, chunk});
    out.keys          = work.alloc(DType::FP32, {pointer, keys});
    out.probabilities = work.alloc(DType::FP32, {1, keys});
    return out;
}

// Gathers `columns` of prefill_hidden and projects them through one head projection.
void project_readouts(const Tensor& prefill_hidden, const std::vector<std::int32_t>& columns,
                      const Tensor& index_storage, const Tensor& hidden_storage,
                      const Tensor& weight, const Tensor& bias, Tensor& out, cudaStream_t stream) {
    const auto count = static_cast<std::int32_t>(columns.size());
    const Tensor indices = index_storage.slice(1, 0, count).view({count});
    upload_i32(columns, indices, stream);
    Tensor hidden = hidden_storage.slice(1, 0, count);
    ops::gather_bf16_columns(prefill_hidden, indices, hidden, stream);
    ops::pointer_head_project(hidden, weight, bias, out, stream);
}

} // namespace

schedule::PrefillContext ProgramImplCore::decision_prefill_context(SequenceState& sequence,
                                                                   std::uint32_t base,
                                                                   std::int32_t current_slot) {
    // A decision lane holds no MTP or DFlash state, samples nothing and never captures a turn
    // checkpoint, so the card's checkpoint slot is only a second slot of the lane that differs
    // from the one it runs in. A long branch runs in the checkpoint slot and names the resident
    // state there, which a chunk without a capture frontier never writes.
    const std::int32_t resident = LinearStateSlots::current_state_slot(sequence.lane,
                                                                       max_concurrency);
    const std::int32_t checkpoint =
        LinearStateSlots::turn_checkpoint_state_slot(sequence.lane, max_concurrency);
    return schedule::PrefillContext{
        {device, model, work, decoder->linear_attention, nullptr, io, prefill_hidden,
         prefill_chunk, proposal_head},
        text_kv_view(sequence),
        qwen3_8::PagedKVCacheView(),
        decoder->text_kv,
        nullptr,
        nullptr,
        base,
        nullptr,
        &sequence.turn_checkpoint_hidden,
        current_slot,
        current_slot == checkpoint ? resident : checkpoint,
        0,
        nullptr,
        lora_slot(sequence.adapter),
        qwen3_8::TextPhase::Decision};
}

RequestBasePlan ProgramImplCore::plan_decision_base(const DecisionPrompt& prompt,
                                                    const runtime::ResolvedDecisionOptions& options) {
    if (!Variant::supports_decisions) {
        throw std::invalid_argument("this target does not serve System One decisions");
    }
    if (!model.lora || decision_keys.data == nullptr || !decision_host) {
        throw std::invalid_argument("System One decisions need the LoRA adapter pool");
    }
    const auto& lora = *model.lora;
    const auto adapter = static_cast<std::size_t>(options.adapter);
    if (options.adapter < 0 || adapter >= lora.kinds.size() ||
        lora.kinds[adapter] != LoraAdapterKind::Decision) {
        throw std::invalid_argument("the selected adapter is not a decision adapter of the pool");
    }
    if (!lora.decision_head.present() || lora.decision_temperatures.size() != lora.kinds.size()) {
        throw std::logic_error("the decision adapter pool carries no pointer head");
    }
    const double temperature = lora.decision_temperatures[adapter];
    if (!std::isfinite(temperature) || temperature <= 0.0) {
        throw std::logic_error("decision adapter temperature must be positive and finite");
    }

    const std::uint32_t state = prompt.state_tokens();
    if (state == 0 || state > prompt.tokens.size() || prompt.branches.empty()) {
        throw std::invalid_argument("decision prompt holds no state or no question");
    }
    std::uint64_t cursor = state;
    for (const DecisionBranch& branch : prompt.branches) {
        if (branch.begin != cursor || branch.length < 2 ||
            branch.length > prompt.tokens.size() - cursor) {
            throw std::invalid_argument("decision branches must follow the state back to back");
        }
        if (branch.option_readouts.empty() ||
            branch.option_readouts.size() > kMaximumDecisionOptions) {
            throw std::invalid_argument("decision question option count is outside [1, 255]");
        }
        const std::vector<std::uint32_t>& readouts = branch.option_readouts;
        for (std::size_t option = 0; option < readouts.size(); ++option) {
            if (readouts[option] >= branch.length - 1 ||
                (option != 0 && readouts[option] <= readouts[option - 1])) {
                throw std::invalid_argument("decision option readouts must ascend inside the branch");
            }
        }
        cursor += branch.length;
    }
    if (cursor != prompt.tokens.size()) {
        throw std::invalid_argument("decision branches must end the prompt");
    }
    for (const TokenId id : prompt.tokens) {
        if (id < 0 || id >= TextConfig::token_domain) {
            throw std::invalid_argument("decision prompt contains a token outside the token domain");
        }
    }

    const std::uint32_t pass_columns = std::min(prefill_chunk, capacity);
    auto plan                        = std::make_shared<DecisionPlan>();
    plan->state_tokens               = state;
    plan->input_tokens               = static_cast<std::uint32_t>(prompt.tokens.size());
    plan->pass_columns               = pass_columns;
    for (std::uint32_t index = 0; index < prompt.branches.size(); ++index) {
        const std::uint32_t length = prompt.branches[index].length;
        if (length > pass_columns) {
            plan->long_branches.push_back(index);
            plan->branch_units += 1ULL + (static_cast<std::uint64_t>(length) - 1ULL) / pass_columns;
            plan->scratch_extent = std::max(plan->scratch_extent, length);
            continue;
        }
        auto pass = std::find_if(plan->passes.begin(), plan->passes.end(),
                                 [&](const DecisionPlan::Pass& candidate) {
                                     return candidate.columns + length <= pass_columns;
                                 });
        if (pass == plan->passes.end()) {
            plan->passes.emplace_back();
            pass = std::prev(plan->passes.end());
            ++plan->branch_units;
        }
        pass->branches.push_back(index);
        pass->columns += length;
    }
    if (decision_context_tokens(prompt) > capacity) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           "decision state and its longest branch exceed Engine context capacity");
    }
    plan->logit_scale = static_cast<float>(
        1.0 / (std::sqrt(static_cast<double>(Variant::DecisionConfig::pointer_dim)) * temperature));
    plan->retain_state = options.allow_prefix_reuse;

    auto base                            = std::make_unique<RequestBasePlanImpl>();
    base->summary.prompt_tokens          = plan->input_tokens;
    base->summary.effective_limit_reason = FinishReason::None;
    base->summary.transient_alignment    = 1;
    base->summary.transient_bytes        = 0;
    base->text_kv_page_entitlement       = pages_for_tokens(state + plan->scratch_extent);
    base->text_kv_page_ceiling           = base->text_kv_page_entitlement;
    base->summary.admission              = runtime::AdmissionResources{
                     .active_lanes     = 1,
                     .main_kv_pages    = base->text_kv_page_entitlement,
                     .backend_kv_pages = 0,
    };
    base->summary.service_work_quanta =
        decision_state_units(state, 0, pass_columns) + plan->branch_units;
    base->adapter            = options.adapter;
    base->allow_prefix_reuse = options.allow_prefix_reuse;
    base->decision           = std::move(plan);
    return RequestBasePlan(std::move(base));
}

RequestPlan ProgramImplCore::plan_decision_for_lane(std::uint32_t lane, const DecisionPrompt& prompt,
                                                    const RequestBasePlan& base_plan) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    const RequestControl& request = requests[lane];
    const SequenceState& sequence = sequences[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("cannot plan a request while Program is active or pending");
    }
    if (base_plan.impl_ == nullptr || !base_plan.impl_->decision) {
        throw std::logic_error("request base plan is not a decision");
    }
    const RequestBasePlanImpl& base = *base_plan.impl_;
    const DecisionPlan& decision    = *base.decision;
    if (prompt.tokens.size() != decision.input_tokens ||
        prompt.state_tokens() != decision.state_tokens) {
        throw std::invalid_argument("decision base plan does not describe the prompt");
    }

    auto plan                         = std::make_unique<RequestPlanImpl>();
    plan->summary                     = base.summary;
    plan->sampling                    = base.sampling;
    plan->text_kv_page_entitlement    = base.text_kv_page_entitlement;
    plan->backend_kv_page_entitlement = 0;
    plan->text_kv_page_ceiling        = base.text_kv_page_ceiling;
    plan->backend_kv_page_ceiling     = 0;
    plan->adapter                     = base.adapter;
    plan->decision                    = base.decision;
    // Only a retained decision state of the same adapter continues: its GDN state exists only at
    // its own frontier, so that frontier must be a prefix of this state.
    if (base.allow_prefix_reuse && sequence.retained && sequence.decision_state &&
        sequence.adapter == base.adapter && sequence.text_kv_valid == sequence.execution_frontier &&
        decision_state_matches(sequence, prompt, sequence.execution_frontier)) {
        plan->reuse      = ReusePath::AppendAtFrontier;
        plan->reuse_base = sequence.execution_frontier;
    }
    plan->summary.reusable_prompt_tokens = plan->reuse_base;
    plan->summary.service_work_quanta =
        decision_state_units(decision.state_tokens, plan->reuse_base, decision.pass_columns) +
        decision.branch_units;
    return RequestPlan(std::move(plan));
}

runtime::PrefillStepResult ProgramImplCore::start_decision_lane(std::uint32_t lane,
                                                                DecisionPrompt&& prompt,
                                                                RequestPlan&& plan) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    SequenceState& sequence = sequences[lane];
    RequestControl& request = requests[lane];
    if (plan.impl_ == nullptr || !plan.impl_->decision) {
        throw std::invalid_argument("request plan is not a decision");
    }
    const RequestPlanImpl& request_plan = *plan.impl_;
    const std::shared_ptr<const DecisionPlan> decision_plan = request_plan.decision;
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("a decision requires a free request lane");
    }
    if (prompt.tokens.size() != decision_plan->input_tokens ||
        prompt.state_tokens() != decision_plan->state_tokens) {
        throw std::invalid_argument("request plan does not describe the decision");
    }
    const std::uint32_t state = decision_plan->state_tokens;
    const std::uint32_t base  = request_plan.reuse_base;
    if (request_plan.reuse != ReusePath::FullReset &&
        (request_plan.reuse != ReusePath::AppendAtFrontier || !sequence.retained ||
         !sequence.decision_state || sequence.adapter != request_plan.adapter ||
         sequence.execution_frontier != base || !sequence.kv || sequence.kv->backend ||
         sequence.text_kv_valid != base || !decision_state_matches(sequence, prompt, base))) {
        throw std::logic_error("planned resident decision state is no longer reusable");
    }

    request.lifecycle       = Lifecycle::Empty;
    sequence.adapter        = request_plan.adapter;
    sequence.retained       = false;
    sequence.decision_state = false;
    sequence.captured_continuations.clear();
    try {
        if (checkpoint_ring_capacity != 0) {
            discard_checkpoint_staging(sequence);
            sequence.checkpoint_ring.clear();
        }
        if (request_plan.reuse == ReusePath::FullReset) {
            sequence.kv.reset();
            ordered_reset(sequence);
            sequence.ledger.clear();
            reserve_sequence_kv(sequence, request_plan.text_kv_page_entitlement, 0);
        } else {
            trim_sequence_kv(sequence, base, 0);
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement, 0);
        }
        sequence.text_kv_valid = base;
        bind_sequence_kv(sequence);
        materialize_sequence_kv(sequence, state + decision_plan->scratch_extent, 0);
        sequence.rope_delta = 0;
        set_device_i32(io.rope_delta, 0);

        sequence.execution_frontier      = base;
        sequence.ledger_frontier         = base;
        sequence.mtp_kv_valid            = 0;
        sequence.dflash_context_frontier = 0;
        sequence.mtp_draft_count         = 0;
        sequence.tail_hidden_valid       = false;
        sequence.turn_checkpoint         = {};
        sequence.user_turn_anchor        = {};
        sequence.ledger.assign(prompt.tokens.begin(),
                               prompt.tokens.begin() + static_cast<std::ptrdiff_t>(state));
        assign_ordinal_identity(sequence.prefix_identity, state);
        request.timings                 = {};
        request.pending                 = {};
        request.text_kv_page_ceiling    = request_plan.text_kv_page_ceiling;
        request.backend_kv_page_ceiling = 0;

        const std::size_t questions = prompt.branches.size();
        request.decision.emplace(RequestControl::Decision{
            .prompt              = std::move(prompt),
            .plan                = decision_plan,
            .reused_state_tokens = base,
            .state_cursor        = base,
        });
        request.decision->outcome.probabilities.resize(questions);
        request.decision->outcome.reused_state_tokens = base;
        request.lifecycle                             = Lifecycle::Prefilling;
        return advance_decision(sequence, request);
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        clear_lane(sequence, request);
        throw;
    }
}

runtime::PrefillStepResult ProgramImplCore::advance_decision(SequenceState& sequence,
                                                            RequestControl& request) {
    if (request.lifecycle != Lifecycle::Prefilling || !request.decision ||
        request.decision->complete) {
        throw std::logic_error("decision step requires a decision in progress");
    }
    RequestControl::Decision& decision = *request.decision;
    const DecisionPlan& plan           = *decision.plan;
    const runtime::BeginSummary summary{
        .prompt_tokens        = plan.input_tokens,
        .reused_prompt_tokens = decision.reused_state_tokens,
        .prefix_reuse_path    = decision.reused_state_tokens != 0 ? ReusePath::AppendAtFrontier
                                                                  : ReusePath::FullReset,
    };
    const auto started = Clock::now();
    try {
        select_prefill_kv_rows(sequence);
        std::uint32_t processed = 0;
        if (decision.state_cursor < plan.state_tokens) {
            // Chunks of pass_columns, the width the service projection counted.
            const std::uint32_t nominal =
                std::min(plan.pass_columns, plan.state_tokens - decision.state_cursor);
            schedule::PrefillContext context = decision_prefill_context(
                sequence, decision.state_cursor,
                LinearStateSlots::current_state_slot(sequence.lane, max_concurrency));
            mark_workspace_usage(workspace_plan.decision_pass);
            const schedule::PrefillChunkResult result = schedule::prefill_text_chunk(
                context, std::span<const TokenId>(decision.prompt.tokens.data(), plan.state_tokens),
                nominal, std::nullopt, false);
            if (result.processed_tokens != nominal) {
                throw std::logic_error("decision state chunk did not consume its planned width");
            }
            processed = result.processed_tokens;
            decision.state_cursor += processed;
            sequence.text_kv_valid = decision.state_cursor;
            decision.outcome.state_seconds +=
                std::chrono::duration<double>(Clock::now() - started).count();
        } else if (decision.next_pass < plan.passes.size()) {
            const DecisionPlan::Pass& pass = plan.passes[decision.next_pass];
            decision_pass_unit(sequence, decision, pass);
            processed = pass.columns;
            ++decision.next_pass;
            ++decision.outcome.branch_passes;
            decision.outcome.branch_seconds +=
                std::chrono::duration<double>(Clock::now() - started).count();
        } else if (decision.next_long < plan.long_branches.size()) {
            processed = decision_long_unit(sequence, decision);
            ++decision.outcome.long_branch_chunks;
            decision.outcome.branch_seconds +=
                std::chrono::duration<double>(Clock::now() - started).count();
        }
        decision.complete = decision.state_cursor == plan.state_tokens &&
                            decision.next_pass == plan.passes.size() &&
                            decision.next_long == plan.long_branches.size();
        return runtime::PrefillStepResult{.summary                 = summary,
                                          .processed_prompt_tokens = processed,
                                          .complete                = decision.complete};
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        clear_lane(sequence, request);
        throw;
    }
}

void ProgramImplCore::decision_pass_unit(SequenceState& sequence,
                                         RequestControl::Decision& decision,
                                         const DecisionPlan::Pass& pass) {
    const DecisionPrompt& prompt = decision.prompt;
    const DecisionPlan& plan     = *decision.plan;
    std::vector<TokenId> ids;
    ids.reserve(pass.columns);
    std::vector<schedule::DecisionSegment> segments;
    segments.reserve(pass.branches.size());
    // Per question: its decide column in `queries`, its first option column in `keys` and its
    // option count (the pointer_head_score table).
    std::vector<std::int32_t> query_columns;
    std::vector<std::int32_t> key_columns;
    std::vector<std::int32_t> questions;
    query_columns.reserve(pass.branches.size());
    questions.reserve(3 * pass.branches.size());
    for (const std::uint32_t index : pass.branches) {
        const DecisionBranch& branch = prompt.branches[index];
        const auto column            = static_cast<std::int32_t>(ids.size());
        const auto length            = static_cast<std::int32_t>(branch.length);
        segments.push_back(schedule::DecisionSegment{.column = column, .length = length});
        const auto first = prompt.tokens.begin() + static_cast<std::ptrdiff_t>(branch.begin);
        ids.insert(ids.end(), first, first + length);
        questions.push_back(static_cast<std::int32_t>(query_columns.size()));
        questions.push_back(static_cast<std::int32_t>(key_columns.size()));
        questions.push_back(static_cast<std::int32_t>(branch.option_readouts.size()));
        query_columns.push_back(column + length - 1);
        for (const std::uint32_t readout : branch.option_readouts) {
            key_columns.push_back(column + static_cast<std::int32_t>(readout));
        }
    }
    if (ids.size() != pass.columns) {
        throw std::logic_error("decision pass does not hold its planned columns");
    }

    schedule::PrefillContext context = decision_prefill_context(
        sequence, plan.state_tokens,
        LinearStateSlots::current_state_slot(sequence.lane, max_concurrency));
    mark_workspace_usage(workspace_plan.decision_pass);
    schedule::decision_pass(context, ids, segments);

    const DecisionHeadPlanes head = decision_head_planes(*model.lora, lora_slot(sequence.adapter));
    const auto chunk = static_cast<std::int32_t>(std::min(prefill_chunk, capacity));
    const auto query_count = static_cast<std::int32_t>(query_columns.size());
    const auto key_count   = static_cast<std::int32_t>(key_columns.size());
    work.reset();
    mark_workspace_usage(workspace_plan.decision_head);
    const DecisionHeadScratch scratch = allocate_decision_head(work, chunk);
    Tensor queries                    = scratch.queries.slice(1, 0, query_count);
    project_readouts(prefill_hidden, query_columns, scratch.query_columns, scratch.query_hidden,
                     head.query_weight, head.query_bias, queries, device.stream);
    Tensor keys = scratch.keys.slice(1, 0, key_count);
    project_readouts(prefill_hidden, key_columns, scratch.key_columns, scratch.key_hidden,
                     head.key_weight, head.key_bias, keys, device.stream);
    const Tensor table = scratch.questions.slice(1, 0, query_count);
    upload_i32(questions, table, device.stream);
    Tensor probabilities = scratch.probabilities.slice(1, 0, key_count).view({key_count});
    ops::pointer_head_score(queries, keys, table, plan.logit_scale, probabilities, device.stream);
    CUDA_CHECK(cudaMemcpyAsync(decision_host->data(), probabilities.data, probabilities.bytes(),
                               cudaMemcpyDeviceToHost, device.stream));
    device.synchronize();
    work.reset();

    const auto* host = static_cast<const float*>(decision_host->data());
    for (std::size_t question = 0; question < pass.branches.size(); ++question) {
        const auto first = static_cast<std::size_t>(questions[3 * question + 1]);
        const auto count = static_cast<std::size_t>(questions[3 * question + 2]);
        decision.outcome.probabilities[pass.branches[question]].assign(host + first,
                                                                        host + first + count);
    }
}

std::uint32_t ProgramImplCore::decision_long_unit(SequenceState& sequence,
                                                  RequestControl::Decision& decision) {
    const DecisionPlan& plan           = *decision.plan;
    const std::uint32_t branch_index   = plan.long_branches[decision.next_long];
    const DecisionBranch& branch       = decision.prompt.branches[branch_index];
    const std::int32_t current_slot    = LinearStateSlots::current_state_slot(sequence.lane,
                                                                              max_concurrency);
    const std::int32_t branch_slot     = LinearStateSlots::turn_checkpoint_state_slot(sequence.lane,
                                                                                      max_concurrency);
    if (decision.long_cursor == 0) {
        // The branch continues a copy of the state's GDN slot in the lane's otherwise unused
        // turn-checkpoint slot; its KV lands above the state like a prompt suffix.
        decoder->linear_attention.copy_slot(current_slot, branch_slot, device.stream);
        decision.long_row.assign(decision.prompt.tokens.begin(),
                                 decision.prompt.tokens.begin() +
                                     static_cast<std::ptrdiff_t>(plan.state_tokens));
        const auto first =
            decision.prompt.tokens.begin() + static_cast<std::ptrdiff_t>(branch.begin);
        decision.long_row.insert(decision.long_row.end(), first,
                                 first + static_cast<std::ptrdiff_t>(branch.length));
        decision.long_keys = 0;
    }

    const std::uint32_t begin   = decision.long_cursor;
    const std::uint32_t nominal = std::min(plan.pass_columns, branch.length - begin);
    schedule::PrefillContext context =
        decision_prefill_context(sequence, plan.state_tokens + begin, branch_slot);
    mark_workspace_usage(workspace_plan.decision_pass);
    const schedule::PrefillChunkResult result = schedule::prefill_text_chunk(
        context, decision.long_row, nominal, std::nullopt, false);
    if (result.processed_tokens != nominal) {
        throw std::logic_error("decision long-branch chunk did not consume its planned width");
    }
    const std::uint32_t end = begin + result.processed_tokens;
    const bool final        = end == branch.length;

    std::vector<std::int32_t> key_columns;
    for (const std::uint32_t readout : branch.option_readouts) {
        if (readout >= begin && readout < end) {
            key_columns.push_back(static_cast<std::int32_t>(readout - begin));
        }
    }
    if (!key_columns.empty() || final) {
        const DecisionHeadPlanes head =
            decision_head_planes(*model.lora, lora_slot(sequence.adapter));
        const auto chunk = static_cast<std::int32_t>(std::min(prefill_chunk, capacity));
        work.reset();
        mark_workspace_usage(workspace_plan.decision_head);
        const DecisionHeadScratch scratch = allocate_decision_head(work, chunk);
        const Tensor lane_keys =
            decision_keys.slice(2, static_cast<std::int32_t>(sequence.lane), 1)
                .view({Variant::DecisionConfig::pointer_dim,
                       static_cast<std::int32_t>(kMaximumDecisionOptions)});
        if (!key_columns.empty()) {
            const auto count = static_cast<std::int32_t>(key_columns.size());
            Tensor keys      = lane_keys.slice(1, static_cast<std::int32_t>(decision.long_keys),
                                               count);
            project_readouts(prefill_hidden, key_columns, scratch.key_columns, scratch.key_hidden,
                             head.key_weight, head.key_bias, keys, device.stream);
            decision.long_keys += static_cast<std::uint32_t>(count);
        }
        if (final) {
            const auto option_count = static_cast<std::int32_t>(branch.option_readouts.size());
            if (decision.long_keys != branch.option_readouts.size()) {
                throw std::logic_error("decision long branch did not project every option key");
            }
            Tensor queries = scratch.queries.slice(1, 0, 1);
            project_readouts(prefill_hidden,
                             {static_cast<std::int32_t>(branch.length - 1U - begin)},
                             scratch.query_columns, scratch.query_hidden, head.query_weight,
                             head.query_bias, queries, device.stream);
            const Tensor table = scratch.questions.slice(1, 0, 1);
            upload_i32({0, 0, option_count}, table, device.stream);
            Tensor probabilities =
                scratch.probabilities.slice(1, 0, option_count).view({option_count});
            ops::pointer_head_score(queries, lane_keys.slice(1, 0, option_count), table,
                                    plan.logit_scale, probabilities, device.stream);
            CUDA_CHECK(cudaMemcpyAsync(decision_host->data(), probabilities.data,
                                       probabilities.bytes(), cudaMemcpyDeviceToHost,
                                       device.stream));
            device.synchronize();
            const auto* host = static_cast<const float*>(decision_host->data());
            decision.outcome.probabilities[branch_index].assign(host, host + option_count);
        }
        work.reset();
    }

    decision.long_cursor = end;
    if (final) {
        decision.long_cursor = 0;
        decision.long_keys   = 0;
        decision.long_row.clear();
        ++decision.next_long;
    }
    return result.processed_tokens;
}

DecisionOutcome ProgramImplCore::take_decision_lane(std::uint32_t lane) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    SequenceState& sequence = sequences[lane];
    RequestControl& request = requests[lane];
    if (request.lifecycle != Lifecycle::Prefilling || !request.decision ||
        !request.decision->complete) {
        throw std::logic_error("lane holds no completed decision");
    }
    const std::uint32_t state = request.decision->plan->state_tokens;
    const bool retain         = request.decision->plan->retain_state;
    DecisionOutcome outcome   = std::move(request.decision->outcome);
    if (!retain) {
        clear_lane(sequence, request);
        return outcome;
    }
    try {
        // Branch scratch above the state is never interpreted again: dropping it is a host page
        // map update, and the lane keeps [0, state) as a retained decision state.
        trim_sequence_kv(sequence, state, 0);
        release_sequence_growth_entitlement(sequence);
        unbind_sequence_kv(sequence);
    } catch (...) {
        clear_lane(sequence, request);
        throw;
    }
    sequence.text_kv_valid      = state;
    sequence.execution_frontier = state;
    sequence.ledger_frontier    = state;
    sequence.retained           = true;
    sequence.decision_state     = true;
    request.decision.reset();
    request.lifecycle = Lifecycle::Complete;
    return outcome;
}

// Image kind `decision_state`. A retained decision state is exactly the main-text KV [0, Ls) and
// the lane's current GDN slot: take_decision_lane leaves nothing else, and a decision continuing
// it reads nothing else. The image is published under the alias of its exact state row and serves
// only a decision of exactly that state and adapter; continuing a shorter retained state is L1
// lane planning alone. Alias, digest and metadata encodings are in continuation_image.h.

std::optional<std::string>
ProgramImplCore::decision_state_alias(const DecisionPrompt& prompt) const {
    const std::uint32_t state = prompt.state_tokens();
    if (state == 0 || state > capacity || state > prompt.tokens.size()) { return std::nullopt; }
    try {
        return image::decision_state_alias(decision_state_compatibility_key,
                                           std::span<const TokenId>(prompt.tokens.data(), state));
    } catch (...) { return std::nullopt; }
}

cache::ContinuationImage
ProgramImplCore::export_decision_state_background(std::uint32_t lane) const {
    if (lane >= max_concurrency) { throw std::out_of_range("continuation lane is out of range"); }
    CUDA_CHECK(cudaStreamWaitEvent(export_stream, export_fence_events[lane], 0));
    const SequenceState& sequence = sequences[lane];
    const RequestControl& request = requests[lane];
    const std::uint32_t state     = sequence.execution_frontier;
    if (!sequence.retained || !sequence.decision_state ||
        request.lifecycle != Lifecycle::Complete || request.prefill || request.decision ||
        request.pending.kind != PendingKind::None || !sequence.kv || sequence.kv->backend ||
        sequence.kv->text.bound_row() >= 0 || state == 0 || state > capacity ||
        sequence.text_kv_valid != state || sequence.ledger_frontier != state ||
        sequence.ledger.size() != state || sequence.prefix_identity.size() != state ||
        !ordinal_text_prefix(sequence.prefix_identity, state) || !model.lora ||
        sequence.adapter < 0 ||
        static_cast<std::size_t>(sequence.adapter) >= model.lora->fingerprints.size()) {
        throw std::logic_error("lane holds no complete exportable decision state");
    }

    cache::ContinuationImage out;
    out.format_version    = image::kTargetImageVersion;
    out.compatibility_key = decision_state_compatibility_key;
    out.frontier_tokens   = state;
    out.prefix_identity   = image::decision_state_prefix(std::span<const TokenId>(sequence.ledger));
    out.frontier_prefix_digest = image::decision_state_digest(out.prefix_identity);
    out.frontier_metadata      = image::encode_decision_state(image::DecisionStateMetadata{
             .state_tokens = state,
             .adapter      = model.lora->fingerprints[static_cast<std::size_t>(sequence.adapter)]});
    image::emit_paged(
        out.segments, "main.text_kv",
        export_paged_kv_logical(sequence.kv->text, state, export_transfer, export_stream));
    out.segments.emplace("main.gdn",
                         image::export_linear_segment(
                             decoder->linear_attention,
                             LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
                             export_transfer, export_stream));
    return out;
}

std::uint32_t ProgramImplCore::preflight_decision_state_metadata(
    const cache::SessionCandidateDescriptor& candidate,
    const DecisionPrompt& prompt) const noexcept {
    try {
        const std::uint32_t state = prompt.state_tokens();
        if (candidate.status != cache::CacheLookupStatus::Hit ||
            candidate.image_format_version != image::kTargetImageVersion ||
            candidate.compatibility_key != decision_state_compatibility_key || state == 0 ||
            state > capacity || state > prompt.tokens.size() ||
            candidate.frontier_tokens != state || candidate.boundary_tokens != 0 ||
            !candidate.boundary_prefix_digest.empty()) {
            return 0;
        }
        return image::decision_state_digest(image::decision_state_prefix(std::span<const TokenId>(
                   prompt.tokens.data(), state))) == candidate.frontier_prefix_digest
                   ? state
                   : 0;
    } catch (...) { return 0; }
}

std::uint32_t
ProgramImplCore::preflight_decision_state(const cache::ContinuationImage& candidate,
                                          const DecisionPrompt& prompt, std::int32_t adapter,
                                          std::uint32_t* divergence_tokens) const noexcept {
    if (divergence_tokens != nullptr) { *divergence_tokens = 0; }
    try {
        const std::uint32_t state = prompt.state_tokens();
        if (candidate.format_version != image::kTargetImageVersion ||
            candidate.compatibility_key != decision_state_compatibility_key || state == 0 ||
            state > capacity || state > prompt.tokens.size() ||
            candidate.frontier_tokens != state || candidate.boundary_tokens != 0 ||
            !candidate.boundary_prefix_digest.empty() || !candidate.boundary_metadata.empty() ||
            !model.lora || adapter < 0 ||
            static_cast<std::size_t>(adapter) >= model.lora->fingerprints.size()) {
            return 0;
        }
        // The exact state row with its ordinal text identity - one comparison of canonical prefix
        // bytes - and the adapter whose weights the state encodes.
        const std::span<const TokenId> row(prompt.tokens.data(), state);
        const cache::Bytes expected = image::decision_state_prefix(row);
        const bool exact            = candidate.prefix_identity == expected;
        if (divergence_tokens != nullptr) {
            if (exact) {
                *divergence_tokens = state;
            } else {
                const image::PrefixData prefix =
                    image::decode_prefix(candidate.prefix_identity, state);
                const auto agreed = std::mismatch(prefix.ledger.begin(), prefix.ledger.end(),
                                                  row.begin(), row.end());
                *divergence_tokens =
                    static_cast<std::uint32_t>(agreed.first - prefix.ledger.begin());
            }
        }
        const image::DecisionStateMetadata metadata =
            image::decode_decision_state(candidate.frontier_metadata);
        if (!exact || metadata.state_tokens != state ||
            metadata.adapter != model.lora->fingerprints[static_cast<std::size_t>(adapter)] ||
            candidate.frontier_prefix_digest != image::decision_state_digest(expected)) {
            return 0;
        }
        std::set<std::string> inventory{"main.gdn"};
        image::paged_segment_names("main.text_kv", decoder->text_kv.pool().plane_count(),
                                   inventory);
        if (candidate.segments.size() != inventory.size() ||
            !std::all_of(candidate.segments.begin(), candidate.segments.end(),
                         [&](const auto& segment) { return inventory.contains(segment.first); })) {
            return 0;
        }
        return state;
    } catch (...) { return 0; }
}

std::shared_ptr<DecodedContinuation>
ProgramImplCore::decode_decision_state(const cache::ContinuationImage& candidate) const {
    // Startup-fixed geometry only, like decode_continuation(): no lane state and no CUDA call.
    if (candidate.compatibility_key != decision_state_compatibility_key) {
        throw std::invalid_argument("continuation image is not a decision state");
    }
    const image::DecisionStateMetadata metadata =
        image::decode_decision_state(candidate.frontier_metadata);
    auto out     = std::make_shared<DecodedContinuation>();
    out->text_kv = image::decode_paged(candidate.segments, "main.text_kv", decoder->text_kv.pool(),
                                       metadata.state_tokens);
    out->current_gdn =
        image::decode_linear(candidate.segments.at("main.gdn"), decoder->linear_attention);
    return out;
}

ContinuationRestoreFailure ProgramImplCore::import_decision_state_lane(
    std::uint32_t lane, const cache::ContinuationImage& candidate,
    const DecodedContinuation& decoded, const DecisionPrompt& prompt, std::int32_t adapter,
    runtime::KvPageFootprint entitlement) noexcept {
    using Failure = ContinuationRestoreFailure;
    if (lane >= max_concurrency) { return Failure::LaneUnavailable; }
    SequenceState& sequence = sequences[lane];
    RequestControl& request = requests[lane];
    if (request.lifecycle != Lifecycle::Empty || request.prefill || request.decision ||
        request.pending.kind != PendingKind::None || sequence.retained || sequence.kv) {
        return Failure::LaneUnavailable;
    }
    const std::uint32_t state = preflight_decision_state(candidate, prompt, adapter);
    if (state == 0) { return Failure::VerifyDepthMismatch; }
    // The payload was decoded by decode_decision_state() for this exact image, bound to it by
    // content identity at the caller. Confirm its shape before a page is reserved.
    if (decoded.text_kv.valid_tokens != state || !decoded.tail_hidden.empty() ||
        decoded.checkpoint_gdn || decoded.checkpoint_hidden || decoded.backend_kv ||
        decoded.dflash_local || decoded.dflash_checkpoint_local) {
        return Failure::MetadataMismatch;
    }
    // The decision's planned entitlement, long-branch scratch included, as a cold admission
    // reserves.
    const std::uint32_t text_pages = std::max(pages_for_tokens(state), entitlement.text_pages);
    if (!kv_reservation_fits(text_pages, 0)) { return Failure::KvReservationExhausted; }
    try {
        reserve_sequence_kv(sequence, text_pages, 0);
    } catch (...) { return Failure::KvReservationExhausted; }

    try {
        import_paged_kv_logical(sequence.kv->text, decoded.text_kv, continuation_transfer,
                                device.stream);
        import_linear_attention_state(
            decoder->linear_attention,
            LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
            decoded.current_gdn, continuation_transfer, device.stream);
        device.synchronize();

        // Exactly the lane take_decision_lane leaves for this state.
        sequence.ledger.assign(prompt.tokens.begin(),
                               prompt.tokens.begin() + static_cast<std::ptrdiff_t>(state));
        assign_ordinal_identity(sequence.prefix_identity, state);
        sequence.execution_frontier      = state;
        sequence.ledger_frontier         = state;
        sequence.text_kv_valid           = state;
        sequence.rope_delta              = 0;
        sequence.mtp_kv_valid            = 0;
        sequence.dflash_context_frontier = 0;
        sequence.mtp_draft_count         = 0;
        sequence.tail_hidden_valid       = false;
        sequence.turn_checkpoint         = {};
        sequence.user_turn_anchor        = {};
        sequence.adapter                 = adapter;
        sequence.retained                = true;
        sequence.decision_state          = true;
        request.lifecycle                = Lifecycle::Complete;
        return Failure::None;
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        clear_lane(sequence, request);
        return Failure::DecodeFailed;
    }
}

} // namespace ninfer::targets::qwen3_8::detail::NINFER_QWEN38_RUNTIME_NS
