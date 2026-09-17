#include "targets/qwen3_8/impl/runtime/instance.h"

#include <ninfer/targets/qwen3_8/prepared_prompt.h>

#include "targets/qwen3_8/impl/runtime/layouts.h"
#include "targets/qwen3_8/impl/runtime/program.h"

#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_8 {

using detail::NINFER_QWEN38_RUNTIME_NS::Variant;

template <>
SequencePlan<Variant>::SequencePlan(
    std::unique_ptr<detail::SequencePlanImpl<Variant>> impl) noexcept
    : impl_(std::move(impl)) {}

template <>
SequencePlan<Variant>::SequencePlan(SequencePlan&& other) noexcept
    : impl_(std::move(other.impl_)) {}
template <>
SequencePlan<Variant>& SequencePlan<Variant>::operator=(SequencePlan&& other) noexcept {
    impl_ = std::move(other.impl_);
    return *this;
}
template <>
SequencePlan<Variant>::~SequencePlan() = default;

template <>
std::uint32_t SequencePlan<Variant>::capacity() const noexcept {
    return impl_ != nullptr ? impl_->capacity : 0;
}

template <>
std::uint32_t SequencePlan<Variant>::kv_capacity() const noexcept {
    return impl_ != nullptr ? impl_->kv_capacity : 0;
}

template <>
std::uint32_t SequencePlan<Variant>::max_concurrency() const noexcept {
    return impl_ != nullptr ? impl_->max_concurrency : 0;
}

template <>
std::size_t SequencePlan<Variant>::device_reservation_bytes() const noexcept {
    return impl_ != nullptr ? impl_->device_reservation_bytes : 0;
}

template <>
std::size_t SequencePlan<Variant>::workspace_capacity_bytes() const noexcept {
    return impl_ != nullptr ? impl_->workspace.capacity : 0;
}

template <>
std::size_t SequencePlan<Variant>::request_transient_capacity_bytes() const noexcept {
    return impl_ != nullptr ? impl_->request_transient_capacity_bytes : 0;
}

template <>
SequencePlanner<Variant>::SequencePlanner(
    std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl) noexcept
    : impl_(std::move(impl)) {}

template <>
SequencePlanner<Variant>::SequencePlanner(SequencePlanner&&) noexcept = default;
template <>
SequencePlanner<Variant>& SequencePlanner<Variant>::operator=(SequencePlanner&&) noexcept = default;
template <>
SequencePlanner<Variant>::~SequencePlanner() = default;

template <>
const runtime::SequenceCapacityCurve& SequencePlanner<Variant>::capacity_curve() const noexcept {
    static const runtime::SequenceCapacityCurve empty;
    return impl_ != nullptr ? impl_->curve : empty;
}

template <>
SequencePlan<Variant> SequencePlanner<Variant>::finalize(std::uint32_t main_page_groups) && {
    if (impl_ == nullptr) { throw std::logic_error("sequence planner is empty"); }
    return SequencePlan<Variant>(detail::NINFER_QWEN38_RUNTIME_NS::finalize_sequence_plan_impl(
        std::move(impl_), main_page_groups));
}

template <>
RequestBasePlan<Variant>::RequestBasePlan(
    std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl) noexcept
    : impl_(std::move(impl)) {}

template <>
RequestBasePlan<Variant>::RequestBasePlan(RequestBasePlan&& other) noexcept
    : impl_(std::move(other.impl_)) {}
template <>
RequestBasePlan<Variant>& RequestBasePlan<Variant>::operator=(RequestBasePlan&& other) noexcept {
    impl_ = std::move(other.impl_);
    return *this;
}
template <>
RequestBasePlan<Variant>::~RequestBasePlan() = default;

template <>
const runtime::RequestPlanSummary& RequestBasePlan<Variant>::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

template <>
RequestPlan<Variant>::RequestPlan(std::unique_ptr<detail::RequestPlanImpl<Variant>> impl) noexcept
    : impl_(std::move(impl)) {}

template <>
RequestPlan<Variant>::RequestPlan(RequestPlan&& other) noexcept
    : impl_(std::move(other.impl_)) {}
template <>
RequestPlan<Variant>& RequestPlan<Variant>::operator=(RequestPlan&& other) noexcept {
    impl_ = std::move(other.impl_);
    return *this;
}
template <>
RequestPlan<Variant>::~RequestPlan() = default;

template <>
const runtime::RequestPlanSummary& RequestPlan<Variant>::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

template <>
Program<Variant>::Program(std::unique_ptr<detail::ProgramImpl<Variant>> impl) noexcept
    : impl_(std::move(impl)) {}

template <>
Program<Variant>::~Program() noexcept = default;

template <>
RequestBasePlan<Variant>
Program<Variant>::plan_request_base(const PreparedPrompt& prompt,
                                    const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request_base(PreparedPromptAccess::view(prompt), options);
}

template <>
RequestPlan<Variant> Program<Variant>::plan_request_for_lane(
    std::uint32_t lane, const PreparedPrompt& prompt, const RequestBasePlan<Variant>& base,
    std::span<const std::uint32_t> capture_depths) {
    return impl_->plan_request_for_lane(lane, PreparedPromptAccess::view(prompt), base,
                                        capture_depths);
}

template <>
bool Program<Variant>::can_admit_lane(std::uint32_t lane,
                                      const RequestPlan<Variant>& plan) const noexcept {
    return impl_->can_admit_lane(lane, plan);
}

template <>
bool Program<Variant>::ensure_adapter_resident(
    std::int32_t adapter, const std::function<void(std::uint32_t)>& release_retained) {
    return impl_->ensure_adapter_resident(adapter, release_retained);
}

template <>
std::string Program<Variant>::adapter_scope(std::int32_t adapter) const {
    return impl_->adapter_scope(adapter);
}

template <>
std::uint64_t Program<Variant>::lora_stage_count() const noexcept {
    return impl_->lora_stage_count();
}

template <>
bool Program<Variant>::can_admit_lane_after_retained_eviction(
    std::uint32_t lane, const RequestPlan<Variant>& plan) const noexcept {
    return impl_->can_admit_lane_after_retained_eviction(lane, plan);
}

template <>
runtime::AdmissionResources Program<Variant>::admission_capacity() const noexcept {
    return impl_->admission_capacity();
}

template <>
runtime::PrefillStepResult
Program<Variant>::start_prefill_lane(std::uint32_t lane, PreparedPrompt&& prompt,
                                     RequestPlan<Variant>&& plan,
                                     runtime::TransientRegion transient) {
    return impl_->start_prefill_lane(lane, PreparedPromptAccess::take(std::move(prompt)),
                                     std::move(plan), transient);
}

template <>
runtime::PrefillStepResult Program<Variant>::advance_prefill_lane(std::uint32_t lane) {
    return impl_->advance_prefill_lane(lane);
}

template <>
runtime::BatchedGeneratedRound
Program<Variant>::decode_batch(std::span<const std::uint32_t> lanes,
                               std::span<const runtime::RoundBudget> budgets) {
    return impl_->decode_batch(lanes, budgets);
}

template <>
void Program<Variant>::resolve_pending_batch(std::span<const std::uint32_t> lanes,
                                             std::span<const std::uint32_t> accepted_tokens,
                                             std::span<const std::uint8_t> terminal,
                                             std::span<const std::uint8_t> cancelled) {
    impl_->resolve_pending_batch(lanes, accepted_tokens, terminal, cancelled);
}

template <>
void Program<Variant>::resolve_prefill_lane(std::uint32_t lane, bool terminal) {
    impl_->resolve_prefill_lane(lane, terminal);
}

template <>
void Program<Variant>::retire_lane(std::uint32_t lane) {
    impl_->retire_lane(lane);
}

template <>
void Program<Variant>::abort_lane(std::uint32_t lane) noexcept {
    impl_->abort_lane(lane);
}

template <>
bool Program<Variant>::has_retained_lane(std::uint32_t lane) const noexcept {
    return impl_->has_retained_lane(lane);
}

template <>
std::size_t Program<Variant>::retained_lane_resident_bytes(std::uint32_t lane) const noexcept {
    return impl_->retained_lane_resident_bytes(lane);
}

template <>
std::size_t Program<Variant>::retained_lane_reused_bytes(
    std::uint32_t lane, const RequestPlan<Variant>& plan) const noexcept {
    return plan.impl_ == nullptr ? 0 : impl_->retained_lane_reused_bytes(lane, *plan.impl_);
}

template <>
void Program<Variant>::evict_retained_lane(std::uint32_t lane) noexcept {
    impl_->evict_retained_lane(lane);
}

template <>
cache::ContinuationImage Program<Variant>::export_continuation_lane(std::uint32_t lane) const {
    return impl_->export_continuation_lane(lane);
}

template <>
void Program<Variant>::fence_lane_for_export(std::uint32_t lane) {
    impl_->fence_lane_for_export(lane);
}

template <>
cache::ContinuationImage
Program<Variant>::export_continuation_lane_background(std::uint32_t lane) const {
    return impl_->export_continuation_lane_background(lane);
}

template <>
std::vector<PromptBoundaryAlias>
Program<Variant>::boundary_aliases(const PreparedPrompt& prompt) const {
    return impl_->boundary_aliases(PreparedPromptAccess::view(prompt));
}

template <>
std::vector<CapturedContinuation>
Program<Variant>::take_captured_continuations_lane(std::uint32_t lane) {
    return impl_->take_captured_continuations_lane(lane);
}

template <>
std::uint32_t
Program<Variant>::preflight_continuation_metadata(
    const cache::SessionCandidateDescriptor& candidate,
    const PreparedPrompt& prompt) const noexcept {
    try {
        return impl_->preflight_continuation_metadata(candidate,
                                                      PreparedPromptAccess::view(prompt));
    } catch (...) { return 0; }
}

template <>
std::uint32_t
Program<Variant>::preflight_continuation(const cache::ContinuationImage& image,
                                         const PreparedPrompt& prompt,
                                         std::uint32_t* divergence_tokens) const noexcept {
    try {
        return impl_->preflight_continuation(image, PreparedPromptAccess::view(prompt),
                                             divergence_tokens);
    } catch (...) { return 0; }
}

template <>
std::shared_ptr<DecodedContinuation>
Program<Variant>::decode_continuation(const cache::ContinuationImage& image) const {
    return impl_->decode_continuation(image);
}

template <>
ContinuationRestoreFailure Program<Variant>::import_continuation_lane(
    std::uint32_t lane, const cache::ContinuationImage& image, const DecodedContinuation& decoded,
    const PreparedPrompt& prompt, std::int32_t adapter,
    runtime::KvPageFootprint entitlement) noexcept {
    try {
        return impl_->import_continuation_lane(lane, image, decoded,
                                               PreparedPromptAccess::view(prompt), adapter,
                                               entitlement);
    } catch (...) { return ContinuationRestoreFailure::DecodeFailed; }
}

template <>
bool Program<Variant>::kv_reservation_fits(std::uint32_t text_pages,
                                            std::uint32_t backend_pages) const noexcept {
    return impl_->kv_reservation_fits(text_pages, backend_pages);
}

template <>
bool Program<Variant>::try_grow_decode_headroom(std::uint32_t lane) {
    return impl_->try_grow_decode_headroom(lane);
}

template <>
runtime::KvPageFootprint
Program<Variant>::retained_lane_kv_footprint(std::uint32_t lane) const noexcept {
    return impl_->retained_lane_kv_footprint(lane);
}

template <>
std::uint32_t Program<Variant>::retained_lane_depth(std::uint32_t lane) const noexcept {
    return impl_->retained_lane_depth(lane);
}

template <>
std::uint32_t Program<Variant>::retained_lane_boundary_tokens(std::uint32_t lane) const noexcept {
    return impl_->retained_lane_boundary_tokens(lane);
}

template <>
std::string Program<Variant>::retained_lane_digest(std::uint32_t lane) const {
    return impl_->retained_lane_digest(lane);
}

template <>
std::vector<SlotCheckpoint>
Program<Variant>::retained_lane_checkpoints(std::uint32_t lane) const {
    return impl_->retained_lane_checkpoints(lane);
}

template <>
RetainedSessionSnapshot Program<Variant>::save_retained_lane(std::uint32_t lane,
                                                             std::string_view model_binding) {
    return impl_->save_retained_lane(lane, model_binding);
}

template <>
std::uint32_t Program<Variant>::restore_retained_lane(
    std::uint32_t lane, std::span<const std::uint8_t> snapshot, std::string_view model_binding,
    const std::function<void(std::uint32_t)>& release_retained) {
    return impl_->restore_retained_lane(lane, snapshot, model_binding, release_retained);
}

template <>
GenerationTimings Program<Variant>::generation_timings_lane(std::uint32_t lane) const noexcept {
    return impl_->generation_timings_lane(lane);
}

template <>
SpeculativeStats Program<Variant>::speculative_stats_lane(std::uint32_t lane) const noexcept {
    return impl_->speculative_stats_lane(lane);
}

template <>
MemorySummary Program<Variant>::memory_summary() const noexcept {
    return impl_->memory_summary();
}

template <>
void Program<Variant>::reset_memory_peaks() noexcept {
    impl_->reset_memory_peaks();
}

template <>
SequencePlanner<Variant> make_sequence_planner<Variant>(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        Variant::WeightsProfile weights_profile) {
    return SequencePlanner<Variant>(detail::NINFER_QWEN38_RUNTIME_NS::make_sequence_planner_impl(
        device, options, weights_profile));
}

template <>
std::unique_ptr<Program<Variant>>
create_program<Variant>(const Variant::ModelView& model, Variant::WeightsProfile weights_profile,
                          SequencePlan<Variant>&& plan, DeviceContext& device,
                          std::string_view model_id, std::string_view weights_id,
                          std::span<const std::uint8_t> artifact_fingerprint) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("sequence plan is empty"); }
    if (plan.impl_->weights_profile != weights_profile) {
        throw std::invalid_argument(
            "loaded model weights profile does not match the sequence plan");
    }
    auto impl = std::make_unique<detail::ProgramImpl<Variant>>(model, *plan.impl_, device, model_id,
                                                                weights_id, artifact_fingerprint);
    plan.impl_.reset();
    return std::unique_ptr<Program<Variant>>(new Program<Variant>(std::move(impl)));
}

} // namespace ninfer::targets::qwen3_8
