#pragma once
#include "targets/qwen3_8/impl/runtime/instance.h"
// Qwen3.8 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/ops/kv_cache_append_prefix.h"
#include "ninfer/ops/swa.h"
#include "core/decode_graph.h"
#include "runtime/contract/transient_region.h"
#include <ninfer/targets/qwen3_8/prepared_prompt.h>
#include <ninfer/targets/qwen3_8/decoder_state.h>
#include "targets/qwen3_8/impl/runtime/text_context.h"
#include "targets/qwen3_8/impl/runtime/dflash_context.h"
#include "targets/qwen3_8/impl/runtime/vision_context.h"
#include "targets/qwen3_8/impl/runtime/vision_prefill.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <functional>
#include <optional>
#include <span>

namespace ninfer::targets::qwen3_8::detail::NINFER_QWEN38_RUNTIME_NS::schedule {

using qwen3_8::PreparedPromptData;
using qwen3_8::PromptModality;

struct ExecutionCore {
    DeviceContext& device;
    const LoadedModelData& model;
    WorkspaceArena& work;
    LinearAttentionStatePool& linear_attention;
    const GdnReplayRecords* replay_records;
    qwen3_8::RoundState& io;
    Tensor& prefill_hidden;
    std::uint32_t prefill_chunk;
    ProposalHead proposal_head;
};

struct PrefillContext {
    ExecutionCore execution;
    qwen3_8::PagedKVCacheView text_kv;
    qwen3_8::PagedKVCacheView mtp_kv;
    const qwen3_8::PagedKVCache& text_cache;
    const qwen3_8::PagedKVCache* mtp_cache;
    DFlashPersistentState* dflash;
    std::uint32_t text_kv_base;
    const ops::SamplingConfig* sampling;
    Tensor* turn_checkpoint_hidden;
    std::int32_t current_state_slot                         = 0;
    std::int32_t turn_checkpoint_state_slot                 = 0;
    std::uint32_t mtp_proposal_extent                       = 0;
    const qwen3_8::DFlashDecodeIngress* dflash_host_ingress = nullptr;
    // A prefill chunk carries exactly one sequence, so its LoRA selection is a scalar rather than
    // the per-row vector the batched decode paths bind. -1 selects the base weights.
    std::int32_t adapter = -1;
    // Decision for every unit of a System One decision, which admits no INT8 activations.
    qwen3_8::TextPhase phase = qwen3_8::TextPhase::Prefill;
};

struct OrdinaryBatchContext {
    ExecutionCore execution;
    const qwen3_8::PagedKVCache& text_cache;
    qwen3_8::OrdinaryDecodeState& frame;
    const qwen3_8::OrdinaryDecodeIngress& host_ingress;
    qwen3_8::OrdinaryDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
};

struct MtpBatchContext {
    ExecutionCore execution;
    const qwen3_8::PagedKVCache& text_cache;
    const qwen3_8::PagedKVCache& mtp_cache;
    qwen3_8::MtpDecodeState& frame;
    const qwen3_8::MtpDecodeIngress& host_ingress;
    qwen3_8::MtpDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
};

struct DFlashBatchContext {
    ExecutionCore execution;
    const qwen3_8::PagedKVCache& text_cache;
    DFlashPersistentState& dflash;
    qwen3_8::DFlashDecodeState& frame;
    const qwen3_8::DFlashDecodeIngress& host_ingress;
    qwen3_8::DFlashDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
};

struct DFlashAppendContext {
    ExecutionCore execution;
    DFlashPersistentState& dflash;
};

struct MtpGqaEnvelopes {
    ops::GqaExecutionEnvelope target_verify;
    ops::GqaExecutionEnvelope batch;
    std::array<ops::GqaExecutionEnvelope, kMaximumMtpDraftTokens - 1> ar;
};

struct DFlashEnvelopes {
    ops::SwaContextExecutionEnvelope local;
    ops::GqaContextExecutionEnvelope full;
    ops::KVCacheAppendPrefixExecutionEnvelope append;
};

struct TargetVerifyFrameView {
    Tensor ids;
    Tensor cache_positions;
    Tensor rope_positions;
    Tensor valid_columns;
    Tensor kv_table_rows;
    Tensor lanes;
    // Per-sequence LoRA bank selector; one entry covers the whole [width, batch] tile column run
    // of its sequence. Null data leaves the target pass on the base weights.
    Tensor adapters;
    Tensor target_hidden;
    Tensor target_logits;
    Tensor target_tokens;
    Tensor drafts;
    Tensor current_extents;
    Tensor frontiers;
    Tensor anchors;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor selected_hidden;
    const GdnReplayRecords* replay_records = nullptr;
    const ops::SamplingConfig* sampling    = nullptr;
    DFlashFeatureSink* feature_sink        = nullptr;
};

void configure_text_card(TextContext& card, const ExecutionCore& execution,
                         const ops::SamplingConfig* sampling, std::int32_t current_state_slot,
                         std::int32_t turn_checkpoint_state_slot,
                         std::uint32_t mtp_proposal_extent, std::int32_t adapter);
void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::GqaExecutionEnvelope envelope);

[[nodiscard]] PrefillChunkResult prefill_text_chunk(
    PrefillContext& state, std::span<const TokenId> ids, std::uint32_t nominal_length,
    std::optional<std::uint32_t> turn_checkpoint_capture_frontier, bool finalize_at_end);

[[nodiscard]] PrefillChunkResult
prefill_multimodal_chunk(PrefillContext& state, const PreparedPromptData& prompt,
                         VisionPrefillSession& vision, std::uint32_t nominal_length,
                         std::optional<std::uint32_t> turn_checkpoint_capture_frontier,
                         bool finalize_at_end);

// One System One branch pass over the resident prefix [0, state.text_kv_base); see
// TextContext::decision_pass. `state` carries no MTP or DFlash backend.
void decision_pass(PrefillContext& state, std::span<const TokenId> ids,
                   std::span<const DecisionSegment> segments);

struct MtpBridgeInput {
    const Tensor* previous_hidden = nullptr;
    std::int32_t position         = 0;
    std::array<std::int32_t, 3> rope_position{};
};

void sample_from_hidden(PrefillContext& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose);
void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, std::int32_t position,
                            std::span<const std::int32_t> rope_position, bool build_proposal,
                            const Tensor* next_embedding = nullptr);
void mtp_bridge_multimodal(PrefillContext& state, const PreparedPromptData& prompt,
                           VisionPrefillSession& vision, const MtpBridgeInput& bridge);

// Executes one exact-B ordinary decode traversal. All request rows enter through the stable
// ordinary ingress, share one model schedule, publish continuation hidden by selector, and leave
// through one compact egress transfer.
void capture_ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   ops::GqaExecutionEnvelope envelope,
                                   DecodeGraphDefinition& definition);
void ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                           ops::GqaExecutionEnvelope envelope, DecodeGraphExecutable* executable);

// Executes one exact-B MTP verification/alignment/proposal transaction. Each row may carry a
// different current and next proposal extent while the model traversal remains batched.
void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              MtpGqaEnvelopes envelopes, DecodeGraphDefinition& definition);
void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      MtpGqaEnvelopes envelopes, DecodeGraphExecutable* executable);

[[nodiscard]] DFlashFeatureSink
dflash_feature_sink(PrefillContext& state, DFlashFeatureSink::PrefillConsumer consume_prefill = {});
void dflash_append_context(DFlashAppendContext& state, const Tensor& features,
                           const Tensor& positions, const Tensor& commit_counts,
                           const Tensor& lanes, const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope);
void dflash_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_counts, const Tensor& lanes,
                           const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope);
void capture_dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                                 std::uint32_t k, DFlashEnvelopes envelopes,
                                 ops::GqaExecutionEnvelope target_envelope,
                                 DecodeGraphDefinition& definition);
void dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                         DFlashEnvelopes envelopes, ops::GqaExecutionEnvelope target_envelope,
                         DecodeGraphExecutable* executable);

} // namespace ninfer::targets::qwen3_8::detail::NINFER_QWEN38_RUNTIME_NS::schedule
