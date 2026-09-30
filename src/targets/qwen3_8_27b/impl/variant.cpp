#include "targets/qwen3_8_27b/impl/variant.h"

#include "targets/qwen3_8_27b/impl/load/lora_bindings.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/lora.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"

#include <algorithm>
#include <array>
#include <stdexcept>

#define NINFER_QWEN38_VARIANT    ::ninfer::targets::qwen3_8_27b::detail::Variant
#define NINFER_QWEN38_RUNTIME_NS qwen3_8_27b_runtime
#include "targets/qwen3_8/impl/runtime/instantiate.h"

namespace ninfer::targets::qwen3_8_27b::detail {
namespace {

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

void validate_token_interval(std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid target leaf token interval");
    }
}

#if defined(NINFER_SM86)
constexpr ops::LinearPolicy kNvfp4TextPolicy = ops::LinearPolicy::A16Only;
#else
constexpr ops::LinearPolicy kNvfp4TextPolicy = ops::LinearPolicy::AllowA4;
#endif

ops::LinearPolicy text_policy(const Weight& weight) {
    return weight.qtype == QType::NVFP4 ? kNvfp4TextPolicy : ops::LinearPolicy::A16Only;
}

// The permitted activation-compute profile is per-Op and per phase: an Op admits AllowA8 only
// where it has a qualified INT8 route (docs/maintainer/op-development.md section 2.1), and only
// in Prefill. Verify carries the committed decode path, whose numerics stay bit-identical to the
// A16 catalog, and is the phase that CUDA Graphs capture; its widths (at most 8 x 6 columns) never
// reach an INT8-only route either. Every unit of a System One decision, state chunks and branch
// passes alike, runs in Decision: its probabilities are held to kev's serving tolerance against the
// BF16 evaluation path, which group-64 activation quantization alone exceeds.
//
// The projection pairs feed the attention scores and the FP32 GDN recurrence, so A8 there does
// move greedy output. That is an accepted property of the Prefill profile, not a defect: prefix
// reuse reproduces the input semantics of full prefill, not its arithmetic.
bool admits_a8(qwen3_8::TextPhase phase) {
    return phase == qwen3_8::TextPhase::Prefill;
}

// The capacity a phase's routes need. Verify sizes against the A8 catalog, which coincides with
// the A16 one over every Verify width and so covers it.
ops::LinearPolicy capacity_policy(qwen3_8::TextPhase phase) {
    return phase == qwen3_8::TextPhase::Decision ? ops::LinearPolicy::A16Only
                                                 : ops::LinearPolicy::AllowA8;
}

ops::LinearPolicy text_swiglu_policy(const Weight& weight, qwen3_8::TextPhase phase) {
    if (weight.qtype == QType::NVFP4) { return kNvfp4TextPolicy; }
    return weight.qtype == QType::Q4G64_F16S && admits_a8(phase) ? ops::LinearPolicy::AllowA8
                                                                 : ops::LinearPolicy::A16Only;
}

// The split Q4/Q5 projection pairs are named by their Q4 parent; both halves take the same profile.
ops::LinearPolicy text_proj_pair_policy(const Weight& weight, qwen3_8::TextPhase phase) {
    if (weight.qtype == QType::NVFP4) { return kNvfp4TextPolicy; }
    return weight.qtype == QType::Q4G64_F16S && admits_a8(phase) ? ops::LinearPolicy::AllowA8
                                                                 : ops::LinearPolicy::A16Only;
}

ops::LinearPolicy text_linear_add_policy(const Weight& weight, qwen3_8::TextPhase phase) {
    if (weight.qtype == QType::NVFP4) { return kNvfp4TextPolicy; }
    return weight.qtype == QType::Q5G64_F16S && admits_a8(phase) ? ops::LinearPolicy::AllowA8
                                                                 : ops::LinearPolicy::A16Only;
}

constexpr std::size_t kMinimumLeafWorkspaceBytes = 1;

std::size_t gdn_record_workspace_bytes(const Tensor& hidden,
                                       const Variant::GdnProjectionWeights& weights) {
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t width = hidden.ne[1];
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch,
                            width, width));
    }
    return std::max(
        kMinimumLeafWorkspaceBytes,
        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            QType::NVFP4, 16384, TextConfig::hidden, kNvfp4TextPolicy, batch, width, width));
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window. Early ranges limit empty producer CTAs; later ranges
    // follow measured split-policy transitions until the producer grid reaches its fixed cap.
    return graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t capacity,
                                                               std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    // Bound the final AR window E+2K at split-policy transitions until the grid reaches its cap.
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    // Target verify and MTP batch both have T=K+1 and W=E+K+1. Preserve one concrete INT8
    // implementation per range at the T=4/5/6 launch boundaries.
    if (draft_window == 3) {
        add_shifted(1029, draft_window + 1);
    } else if (draft_window == 4) {
        for (const std::uint32_t visible_end : {128U, 512U, 1029U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    } else if (draft_window == 5) {
        for (const std::uint32_t visible_end : {128U, 160U, 2054U, 8198U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(capacity - 1, ends);
}

std::vector<GraphExecutionProfile> Variant::dflash_graph_profiles(std::uint32_t, std::uint32_t,
                                                                  std::uint32_t) {
    return {};
}

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value,
                                   qwen3_8::TextPhase phase, WorkspaceArena& workspace,
                                   cudaStream_t stream) {
    if (const auto* split = std::get_if<SplitAttentionProjectionPayload>(&weights)) {
        ops::attn_input_proj(hidden, split->query_key, split->gate_value, query, gate, key, value,
                             text_proj_pair_policy(split->query_key, phase), workspace, stream);
        return;
    }
    const Weight& fused = std::get<FusedAttentionProjectionPayload>(weights).query_key_gate_value;
    ops::attn_input_proj(hidden, fused, query, gate, key, value, text_policy(fused), workspace,
                         stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, qwen3_8::TextPhase phase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear_add(attention, weight, residual, text_linear_add_policy(weight, phase), workspace,
                    stream);
}

void Variant::mtp_attention_projection(const Tensor& hidden,
                                       const MtpAttentionProjectionWeights& weights, Tensor& query,
                                       Tensor& gate, Tensor& key, Tensor& value,
                                       WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor packed  = workspace.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, cols});
    ops::linear(hidden, weights.packed, packed, stream);
    Tensor query_heads = query.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor key_heads   = key.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    Tensor gate_heads  = gate.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor value_heads = value.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    ops::mtp_split_attn_in(packed, query_heads, key_heads, gate_heads, value_heads, stream);
}

void Variant::mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                Tensor& key, Tensor& value, WorkspaceArena&, cudaStream_t stream) {
    ops::linear_pair(hidden, weights.key, weights.value, key, value, stream);
}

void Variant::mtp_q_gate_projection(const Tensor& hidden,
                                    const MtpAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, WorkspaceArena&, cudaStream_t stream) {
    ops::linear(hidden, weights.query, query, stream);
    ops::linear(hidden, weights.output_gate, gate, stream);
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, qwen3_8::TextPhase phase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor output_gate_flat =
        output_gate.view({TextConfig::value_dim, static_cast<int>(hidden.ne[1])});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj(hidden, split->query_key, split->value_z, qkv, output_gate_flat,
                            text_proj_pair_policy(split->query_key, phase), workspace, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj(hidden, fused, qkv, output_gate_flat, text_policy(fused), workspace,
                        stream);
}

void Variant::gdn_input_projection_snapshot(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, qwen3_8::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_snapshot(hidden, split->query_key, split->value_z, conv_weight,
                                          conv_states, valid_columns, initial_slot,
                                          snapshot_base_slot, query, key, value, output_gate_view,
                                          workspace, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_snapshot(hidden, fused, conv_weight, conv_states, valid_columns,
                                      initial_slot, snapshot_base_slot, query, key, value,
                                      output_gate_view, text_policy(fused), workspace, stream);
}

void Variant::gdn_input_projection_record(const Tensor& hidden, const GdnProjectionWeights& weights,
                                          const Tensor& conv_weight, const Tensor& conv_states,
                                          const Tensor& valid_columns, const Tensor& initial_slots,
                                          Tensor& conv_record, Tensor& query, Tensor& key,
                                          Tensor& value, Tensor& output_gate, qwen3_8::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_record_workspace_bytes(hidden, weights));
    WorkspaceArena leaf_workspace(storage);
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_record(hidden, split->query_key, split->value_z, conv_weight,
                                        conv_states, valid_columns, initial_slots, conv_record,
                                        query, key, value, output_gate_view, leaf_workspace,
                                        stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_record(hidden, fused, conv_weight, conv_states, valid_columns,
                                    initial_slots, conv_record, query, key, value, output_gate_view,
                                    text_policy(fused), leaf_workspace, stream);
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    qwen3_8::TextPhase phase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    ops::linear_add(hidden, weight, residual, text_linear_add_policy(weight, phase), workspace,
                    stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const GdnProjectionWeights& weights,
                                          Tensor& hidden, Tensor& g, Tensor& beta,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    ops::gdn_norm_gating_proj(residual, norm_weight, eps, weights.a_projection,
                              weights.b_projection, weights.a_log, weights.dt_bias, workspace,
                              hidden, g, beta, stream);
}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         qwen3_8::TextPhase phase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope        = workspace.scope();
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
    ops::linear_swiglu(hidden, weights.gate_up, activation,
                       text_swiglu_policy(weights.gate_up, phase), workspace, stream);
    ops::linear_add(activation, weights.down, residual, text_linear_add_policy(weights.down, phase),
                    workspace, stream);
}

void Variant::post_mixer_lora(const Tensor& hidden, const PostMixerWeights& weights,
                              Tensor& residual, const qwen3_8::LoraApplication& lora,
                              qwen3_8::TextPhase phase, WorkspaceArena& workspace,
                              cudaStream_t stream) {
    if (!lora.active()) {
        post_mixer(hidden, weights, residual, phase, workspace, stream);
        return;
    }
    auto scope        = workspace.scope();
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
    ops::linear_swiglu(hidden, weights.gate_up, activation,
                       text_swiglu_policy(weights.gate_up, phase), workspace, stream);

    // The delta must be accumulated into the residual, so it is applied to the down projection's
    // output rather than folded into the fused linear_add. The fused op writes `down(activation)
    // + residual` first, then the correction adds `B * (A * activation)` to the same tensor.
    ops::linear_add(activation, weights.down, residual, text_linear_add_policy(weights.down, phase),
                    workspace, stream);

    ops::LoraGroup group;
    group.rank          = lora.rank;
    group.adapter_count = lora.slot_count;
    group.site_count    = 1;
    group.sites[0]      = ops::LoraSite{
        .a                = lora.site->a.data,
        .b                = lora.site->b.data,
        .n                = residual.ne[0],
        .k                = TextConfig::intermediate,
        .a_adapter_stride = lora.site->a_adapter_stride,
        .b_adapter_stride = lora.site->b_adapter_stride,
    };
    std::array<Tensor*, 1> destinations{&residual};
    ops::lora_delta_add(activation, group, *lora.adapter_index, destinations, workspace, stream);
}

void Variant::lora_prepare_slot(const ModelView& model, std::size_t index) {
    if (!model.lora || model.lora->pool == nullptr) {
        throw std::logic_error("no LoRA bank is attached to this model");
    }
    model.lora->pool->prepare(index);
}

void Variant::lora_commit_slot(const ModelView& model, std::uint32_t slot, DeviceContext& device) {
    if (!model.lora || model.lora->pool == nullptr) {
        throw std::logic_error("no LoRA bank is attached to this model");
    }
    model.lora->pool->commit(slot, device);
}

std::size_t Variant::post_mixer_lora_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                              qwen3_8::TextPhase phase,
                                                              std::int32_t rank,
                                                              std::int32_t first,
                                                              std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    // The activation stays live across the correction, so its bytes are additive rather than
    // shared with the projection scratch.
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(
            post_mixer_workspace_capacity_bytes(weights_profile, phase, first, last));
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(
            ops::lora_delta_add_workspace_capacity_bytes(rank, 1, first, last));
    }
    return layout.peak_bytes(1);
}

void Variant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                             Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor gate_up = workspace.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, cols});
    ops::linear(hidden, weights.gate_up, gate_up, stream);
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, cols});
    ops::silu_mul(gate_up.slice(0, 0, TextConfig::intermediate),
                  gate_up.slice(0, TextConfig::intermediate, TextConfig::intermediate), activation,
                  stream);
    Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, cols});
    ops::linear(activation, weights.down, delta, stream);
    ops::residual_add(delta, residual, stream);
}

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(std::int32_t first,
                                                                       std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, last});
    return layout.peak_bytes(1);
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(std::int32_t first,
                                                                std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   qwen3_8::TextPhase phase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints:
        return ops::attn_input_proj_workspace_capacity_bytes(
            TextConfig::hidden, capacity_policy(phase), first, last);
    case WeightsProfile::Nvfp4:
        return ops::attn_input_proj_workspace_capacity_bytes(
            QType::NVFP4, 14336, TextConfig::hidden, kNvfp4TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_8::TextPhase phase, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints:
        return ops::linear_add_workspace_capacity_bytes(QType::Q5G64_F16S, TextConfig::hidden,
                                                        TextConfig::query_size,
                                                        capacity_policy(phase), first, last);
    case WeightsProfile::Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(QType::NVFP4, TextConfig::hidden,
                                                        TextConfig::query_size, kNvfp4TextPolicy,
                                                        first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   qwen3_8::TextPhase phase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints:
        return ops::gdn_input_proj_workspace_capacity_bytes(
            TextConfig::hidden, capacity_policy(phase), first, last);
    case WeightsProfile::Nvfp4:
        return ops::gdn_input_proj_workspace_capacity_bytes(QType::NVFP4, 16384, TextConfig::hidden,
                                                            kNvfp4TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_8::TextPhase, std::int32_t batch_size, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints:
        return ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch_size, first,
            last);
    case WeightsProfile::Nvfp4:
        return ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            QType::NVFP4, 16384, TextConfig::hidden, kNvfp4TextPolicy, batch_size, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_8::TextPhase, std::int32_t batch_size, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                            batch_size, first, last));
    case WeightsProfile::Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            QType::NVFP4, 16384, TextConfig::hidden, kNvfp4TextPolicy, batch_size,
                            first, last));
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                    qwen3_8::TextPhase phase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints:
        return ops::linear_add_workspace_capacity_bytes(QType::Q5G64_F16S, TextConfig::hidden,
                                                        TextConfig::value_dim,
                                                        capacity_policy(phase), first, last);
    case WeightsProfile::Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(
            QType::NVFP4, TextConfig::hidden, TextConfig::value_dim, kNvfp4TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t first,
                                                                          std::int32_t last) {
    return ops::gdn_norm_gating_proj_workspace_capacity_bytes(TextConfig::gdn_value_heads,
                                                              TextConfig::hidden, first, last);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                         qwen3_8::TextPhase phase, std::int32_t first,
                                                         std::int32_t last) {
    validate_token_interval(first, last);
    QType gate_up_qtype;
    QType down_qtype;
    ops::LinearPolicy gate_up_policy;
    ops::LinearPolicy down_policy;
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints:
        gate_up_qtype  = QType::Q4G64_F16S;
        down_qtype     = QType::Q5G64_F16S;
        gate_up_policy = capacity_policy(phase);
        down_policy    = capacity_policy(phase);
        break;
    case WeightsProfile::Nvfp4:
        gate_up_qtype  = QType::NVFP4;
        down_qtype     = QType::NVFP4;
        gate_up_policy = kNvfp4TextPolicy;
        down_policy    = kNvfp4TextPolicy;
        break;
    default:
        throw std::invalid_argument("qwen3_8_27b: invalid weights profile");
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
            gate_up_qtype, 2 * TextConfig::intermediate, TextConfig::hidden, gate_up_policy, first,
            last));
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
            down_qtype, TextConfig::hidden, TextConfig::intermediate, down_policy, first, last));
    }
    return layout.peak_bytes(1);
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                             std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, last});
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, last});
    return layout.peak_bytes(1);
}

} // namespace ninfer::targets::qwen3_8_27b::detail
