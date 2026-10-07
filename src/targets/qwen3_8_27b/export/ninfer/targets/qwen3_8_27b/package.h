#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "runtime/contract/transient_region.h"
#include <ninfer/targets/qwen3_8/decision.h>
#include <ninfer/targets/qwen3_8/frontend.h>
#include <ninfer/targets/qwen3_8/runtime.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer {

struct DeviceContext;

namespace artifact {
class Binder;
class MaterializedArtifact;
struct ArtifactIdentity;
struct MaterializationPlan;
} // namespace artifact

namespace targets::qwen3_8_27b {

struct Package;

namespace detail {

struct Variant;

enum class WeightsProfile : std::uint8_t {
    GroupwiseInt,
    GroupwiseIntW8Endpoints,
    Nvfp4,
};

using Frontend       = qwen3_8::Frontend;
using PreparedPrompt = qwen3_8::PreparedPrompt;
using OutputSession  = qwen3_8::OutputSession;

class LoadPlan {
public:
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    ~LoadPlan();

    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;

private:
    class Impl;
    explicit LoadPlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct qwen3_8_27b::Package;
};

class LoadedModel {
public:
    ~LoadedModel();

    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&)                 = delete;
    LoadedModel& operator=(LoadedModel&&)      = delete;

private:
    class Impl;
    explicit LoadedModel(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct qwen3_8_27b::Package;
};

} // namespace detail

struct Package {
    static constexpr std::string_view model_id   = "qwen3.8-27b";
    static constexpr std::string_view target_key = "qwen3_8_27b";
    // Serves System One decisions through decision adapters of its LoRA pool. Mirrors the
    // package's `DecisionConfig::supported`, which the family schedule reads.
    static constexpr bool supports_decisions = true;
    static constexpr DecisionLimits decision_limits{qwen3_8::kDecisionMaxStateTokens,
                                                    qwen3_8::kDecisionMaxRowTokens};

    using WeightsProfile  = detail::WeightsProfile;
    using LoadPlan        = detail::LoadPlan;
    using LoadedModel     = detail::LoadedModel;
    using Frontend        = detail::Frontend;
    using PreparedPrompt  = detail::PreparedPrompt;
    using OutputSession   = detail::OutputSession;
    using SequencePlanner = qwen3_8::SequencePlanner<detail::Variant>;
    using SequencePlan    = qwen3_8::SequencePlan<detail::Variant>;
    using RequestBasePlan = qwen3_8::RequestBasePlan<detail::Variant>;
    using RequestPlan     = qwen3_8::RequestPlan<detail::Variant>;
    using Program         = qwen3_8::Program<detail::Variant>;

    [[nodiscard]] static ModelSamplingDefaults sampling_defaults(std::string_view model);
    [[nodiscard]] static WeightsProfile resolve_weights(const artifact::ArtifactIdentity& identity);
    [[nodiscard]] static LoadPlan plan_load(artifact::Binder& binder, const EngineOptions& options,
                                            WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<LoadedModel>
    construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized);
    // Materializes the startup-registered LoRA adapters into a bank owned by the loaded model
    // and publishes the resident bank's inventory. Called after the base artifact is resident
    // and before KV capacity is resolved, so the bank's bytes are already committed when the
    // resolver reads free device memory.
    [[nodiscard]] static LoraAttachment attach_lora(LoadedModel& model, const EngineOptions& options,
                                                    DeviceContext& device);
    [[nodiscard]] static Frontend make_frontend(
        const LoadedModel& model,
        PrefixCheckpointPolicy prefix_checkpoint_policy = PrefixCheckpointPolicy::RollingTool);
    [[nodiscard]] static SequencePlanner make_sequence_planner(DeviceContext& device,
                                                               const EngineOptions& options,
                                                               WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device,
                   std::string_view model_id, std::string_view weights_id,
                   std::span<const std::uint8_t> artifact_fingerprint);
};

} // namespace targets::qwen3_8_27b
} // namespace ninfer
