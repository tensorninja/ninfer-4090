#pragma once

#include <ninfer/targets/qwen3_8/startup_features.h>
#include <ninfer/targets/qwen3_8/vision.h>

#include "core/tensor.h"
#include "ninfer/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer {

class DeviceArena;

namespace targets::qwen3_8 {

template <class ProjectionPayload, class PostMixerPayload>
struct FullAttentionWeights {
    Tensor input_norm;
    ProjectionPayload projection;
    Tensor query_norm;
    Tensor key_norm;
    Weight output;
    Tensor post_attention_norm;
    PostMixerPayload post_mixer;
};

template <class ProjectionPayload, class PostMixerPayload>
struct GdnWeights {
    Tensor input_norm;
    ProjectionPayload projection;
    Tensor convolution;
    Tensor norm;
    Weight output;
    Tensor post_attention_norm;
    PostMixerPayload post_mixer;
};

template <class AttentionPayload, class PostMixerPayload>
struct MtpWeights {
    Weight input_projection;
    Tensor embedding_norm;
    Tensor hidden_norm;
    Tensor input_norm;
    AttentionPayload attention;
    Tensor query_norm;
    Tensor key_norm;
    Weight output;
    Tensor post_attention_norm;
    PostMixerPayload post_mixer;
    Tensor final_norm;
};

struct OptimizedProposalWeights {
    Weight head;
    Tensor token_ids;
};

// One registered low-rank correction site, banked over every resident slot.
//
// `a` holds the resident slots' [rank, K] factors back to back and `b` their [N, rank] factors,
// so a slot index selects a plane by a fixed byte stride. The alpha/r scale is folded into `b`
// at conversion, so no runtime scale exists. A site whose `a` is null is outside the bank
// profile - no pool adapter trained it - and is never scheduled. A site inside the profile that
// the slot's current occupant did not train holds zeros, which contribute nothing.
struct LoraSiteWeights {
    Tensor a;
    Tensor b;
    std::size_t a_adapter_stride = 0;
    std::size_t b_adapter_stride = 0;

    [[nodiscard]] bool present() const noexcept { return a.data != nullptr; }
};

// Everything a package execution leaf needs to correct one site whose input it owns privately.
// The family passes this only when the bank is resident and the site was trained.
struct LoraApplication {
    const LoraSiteWeights* site   = nullptr;
    const Tensor* adapter_index   = nullptr;
    std::int32_t rank             = 0;
    std::int32_t slot_count       = 0;

    [[nodiscard]] bool active() const noexcept {
        return site != nullptr && site->present() && adapter_index != nullptr;
    }
};

// The registered sites of one full-attention layer. `query` and `gate` are two row selections of
// the same source module, so they share one `a` plane and differ only in `b`.
struct LoraFullLayerWeights {
    LoraSiteWeights query;
    LoraSiteWeights gate;
    LoraSiteWeights key;
    LoraSiteWeights value;
    LoraSiteWeights output;
    LoraSiteWeights down;
};

struct LoraGdnLayerWeights {
    LoraSiteWeights output;
    LoraSiteWeights down;
};

// The System One pointer head (decision.h) of every resident slot: BF16 [hidden, pointer_dim]
// weights (pointer_dim rows of hidden) and [pointer_dim] biases of the query and key projections.
// The planes are slot 0's; slot s is `slot_stride` bytes further. Absent unless the pool holds a
// decision adapter. A slot whose occupant is generative holds zeros here that no decision reads.
struct DecisionHeadWeights {
    Tensor query_weight;
    Tensor query_bias;
    Tensor key_weight;
    Tensor key_bias;
    std::size_t slot_stride = 0;

    [[nodiscard]] bool present() const noexcept { return query_weight.data != nullptr; }
};

// A startup-fixed bank of resident slots drawn from an unbounded adapter pool. The rank, the
// site geometry and the per-site slot stride are frozen here and enter the captured graph as
// constants; which slot a row selects is a device-resident value the graph re-reads on every
// replay. Restaging a slot therefore changes only its bytes, never the capture.
template <class LoraPoolPayload, std::size_t FullAttentionLayers, std::size_t GdnLayers>
struct LoraWeights {
    // Device-resident slots, and the pool they are drawn from. A request names a pool adapter;
    // the engine owns which slot currently holds it.
    std::uint32_t slots     = 0;
    std::uint32_t pool_size = 0;
    std::int32_t rank       = 0;
    // The package-owned pool and staging mechanism the family's residency policy drives through
    // the Variant's prepare/commit entry points. The family never interprets it. Held by pointer so a
    // `const ModelView&` still reaches a mutable bank: which adapter occupies a slot changes
    // while the model itself does not.
    LoraPoolPayload* pool = nullptr;
    // Content fingerprint of every pool adapter, in pool order. This is an adapter's identity in
    // every persisted or cached form. A pool position is a load-order detail and a slot is a
    // residency detail; neither survives a restart or a swap, so neither may key state that
    // outlives the request that produced it.
    std::vector<std::array<std::uint8_t, 32>> fingerprints;
    // Kind of every pool adapter, in pool order, and for a decision adapter the calibrated
    // temperature its pointer logits are divided by (1 for a generative adapter).
    std::vector<LoraAdapterKind> kinds;
    std::vector<float> decision_temperatures;
    // Device bytes the package committed for the whole bank. The bank is its own arena outside
    // the weights arena, so this is the only route by which the family's memory summary can
    // account for it instead of leaving it as unexplained missing free memory.
    std::uint64_t device_bytes = 0;
    std::array<LoraFullLayerWeights, FullAttentionLayers> full_layers;
    std::array<LoraGdnLayerWeights, GdnLayers> gdn_layers;
    DecisionHeadWeights decision_head;
};

struct DFlashLayerWeights {
    Tensor input_norm;
    Weight query_key_value;
    Weight context_key;
    Weight context_value;
    Tensor query_norm;
    Tensor key_norm;
    Weight attention_output;
    Tensor post_attention_norm;
    Weight gate_up;
    Weight down;
};

template <std::size_t Layers>
struct DFlashWeights {
    Weight feature_projection;
    Tensor context_norm;
    std::array<DFlashLayerWeights, Layers> layers;
    Tensor final_norm;
};

template <class FullProjectionPayload, class GdnProjectionPayload, class MainPostMixerPayload,
          class MtpAttentionPayload, class MtpPostMixerPayload, class DFlashPayload,
          class LoraPoolPayload, std::size_t FullAttentionLayers, std::size_t GdnLayers>
struct ModelView {
    using FullLayer = FullAttentionWeights<FullProjectionPayload, MainPostMixerPayload>;
    using GdnLayer  = GdnWeights<GdnProjectionPayload, MainPostMixerPayload>;
    using MtpLayer  = MtpWeights<MtpAttentionPayload, MtpPostMixerPayload>;
    using DFlash    = DFlashPayload;
    using Lora      = LoraWeights<LoraPoolPayload, FullAttentionLayers, GdnLayers>;

    DeviceArena* weights_arena = nullptr;
    Weight token_embedding;
    std::array<FullLayer, FullAttentionLayers> full_layers;
    std::array<GdnLayer, GdnLayers> gdn_layers;
    Tensor final_norm;
    Weight output_head;
    StartupFeatures features;
    std::optional<OptimizedProposalWeights> optimized_proposal;
    std::optional<MtpLayer> mtp;
    std::optional<DFlashPayload> dflash;
    std::optional<VisionWeights> vision;
    std::optional<Lora> lora;
};

} // namespace targets::qwen3_8
} // namespace ninfer
