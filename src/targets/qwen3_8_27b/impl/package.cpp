#include <ninfer/targets/qwen3_8_27b/package.h>
#include <ninfer/targets/qwen3_8/frontend_resources.h>
#include <ninfer/targets/qwen3_8/prepared_prompt.h>

#include "artifact/reader.h"
#include "targets/qwen3_8_27b/impl/load/bindings.h"
#include "targets/qwen3_8_27b/impl/load/lora_bindings.h"
#include "targets/qwen3_8_27b/impl/variant.h"

#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_8_27b::detail {

static_assert(Package::supports_decisions == Variant::supports_decisions);

class LoadPlan::Impl {
public:
    Impl(WeightsProfile weights_profile_in, ArtifactLoadPlan target_plan)
        : weights_profile(weights_profile_in), plan(std::move(target_plan)) {}

    WeightsProfile weights_profile;
    ArtifactLoadPlan plan;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->plan.materialization;
}

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadedModel::~LoadedModel() = default;

} // namespace ninfer::targets::qwen3_8_27b::detail

namespace ninfer::targets::qwen3_8_27b {
namespace {

constexpr ModelSamplingDefaults kQwen3_8Defaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
};

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kQwen3_8Defaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id == model_id && identity.weights_id == "groupwise-int") {
        return WeightsProfile::GroupwiseIntW8Endpoints;
    }
    if (identity.model_id == model_id && identity.weights_id == "nvfp4") {
        return WeightsProfile::Nvfp4;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile,
        detail::bind_artifact(binder, weights_profile, qwen3_8::startup_features(options))));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }
    auto impl = std::make_unique<LoadedModel::Impl>(
        plan.impl_->weights_profile, std::move(plan.impl_->plan.bindings), std::move(materialized));
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

LoraAttachment Package::attach_lora(LoadedModel& model, const EngineOptions& options,
                                    DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    if (options.lora.directory.empty()) { return {}; }
    if (options.lora.slots == 0) {
        throw std::invalid_argument("a LoRA adapter directory needs at least one resident slot");
    }

    // Adapter fingerprints share the base artifact's fingerprint cache, so the pool's SHA-256
    // identities are computed once per file rather than once per process.
    detail::LoraDiscovery discovery = detail::discover_lora_pool(
        options.lora, artifact::FingerprintCacheOptions{
                          .directory       = options.continuation_cache.directory,
                          .cache_namespace = options.continuation_cache.cache_namespace});

    detail::LoadedModelData& data = model.impl_->data;
    data.lora_bank =
        std::make_unique<detail::LoraBank>(std::move(discovery), options.lora.slots, device);
    const detail::LoraBank& bank = *data.lora_bank;

    // `StartupFeatures` froze the workspace layout and the schedule before the pool was scanned
    // and must not be rewritten here: the plan the Program is constructed against carries the
    // requested slot count, and rewriting it would fail the frozen-features check. A pool smaller
    // than the request commits fewer slabs, and the bank's executed slot count and rank live on
    // the model view, which is what bounds every selection.
    data.runtime.lora = bank.view();

    LoraAttachment attachment;
    attachment.adapters.reserve(bank.pool().size());
    for (const detail::LoraPoolEntry& entry : bank.pool()) {
        LoraAdapterInfo info{.name = entry.name, .kind = entry.kind, .rank = entry.rank};
        if (entry.kind == LoraAdapterKind::Decision) {
            info.temperature  = entry.decision.temperature;
            info.pointer_dim  = static_cast<std::uint32_t>(detail::DecisionConfig::pointer_dim);
            info.description  = entry.decision.description;
            info.release_date = entry.decision.release_date;
        }
        attachment.adapters.push_back(std::move(info));
    }
    attachment.rank         = bank.profile().rank;
    attachment.slots        = bank.slots();
    attachment.device_bytes = bank.device_bytes();
    attachment.file_bytes   = bank.pool_file_bytes();
    return attachment;
}

Package::Frontend Package::make_frontend(const LoadedModel& model,
                                         PrefixCheckpointPolicy prefix_checkpoint_policy) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_8::make_frontend(model.impl_->data.frontend,
                                  model.impl_->data.runtime.features.vision,
                                  model.impl_->data.runtime.features.vision_max_tokens,
                                  prefix_checkpoint_policy);
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile) {
    return qwen3_8::make_sequence_planner<detail::Variant>(device, options, weights_profile);
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    (void)model;
    (void)plan;
    (void)device;
    throw std::invalid_argument("create_program requires artifact model_id and weights_id");
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device,
                         std::string_view model_id, std::string_view weights_id,
                         std::span<const std::uint8_t> artifact_fingerprint) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_8::create_program<detail::Variant>(
        model.impl_->data.runtime, model.impl_->weights_profile, std::move(plan), device, model_id,
        weights_id, artifact_fingerprint);
}

} // namespace ninfer::targets::qwen3_8_27b
