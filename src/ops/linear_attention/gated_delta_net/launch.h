#pragma once

#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "ops/linear_attention/gated_delta_net/common.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {

struct alignas(8) GdnReplayFoldKernelRow {
    std::int32_t linear_state_slot;
    std::int32_t commit_columns;
};

struct alignas(16) GdnReplayFoldKernelRows {
    GdnReplayFoldKernelRow row[8];
};

void launch_recurrent_fp32(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                           const Tensor& beta, float scale, Tensor& ssm_state, Tensor& out,
                           cudaStream_t stream);

void launch_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, bool normalize_qk, Tensor& ssm_state,
                      Tensor& out, cudaStream_t stream);

void launch_recurrent_inout(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, bool normalize_qk,
                            const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                            cudaStream_t stream);

void launch_recurrent_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                               const Tensor& beta, float scale, bool normalize_qk,
                               Tensor& ssm_states, const Tensor& valid_columns,
                               const Tensor& initial_state_slots, const Tensor& snapshot_base_slots,
                               Tensor& out, cudaStream_t stream);

void launch_recurrent_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                             const Tensor& beta, float scale, const Tensor& ssm_states,
                             const Tensor& valid_columns, const Tensor& initial_state_slots,
                             Tensor& key_record, Tensor& value_record, Tensor& gate_record,
                             Tensor& out, cudaStream_t stream);

void launch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                        const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                        cudaStream_t stream);

std::size_t chunked_workspace_bytes(std::int32_t value_heads, std::int32_t tokens);

void launch_chunked(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                    const Tensor& beta, float scale, const Tensor& ssm_state_in,
                    Tensor& ssm_state_out, Tensor& out, void* workspace,
                    std::size_t workspace_bytes, cudaStream_t stream);

// Segmented route (see segment_table); q/k/v/g/beta/out are packed over N columns. The chunked
// stage runs every segment's full chunks from ssm_state and writes their output columns;
// chunk_states, FP32 [128,128,Hv,segmented_tail_state_slots(N)], receives the end state of every
// chunk run that has a tail. The workspace is chunked_workspace_bytes(Hv,N).
void launch_chunked_segmented(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                              const Tensor& beta, float scale, const Tensor& ssm_state,
                              segment_table segments, float* chunk_states, Tensor& out,
                              void* workspace, std::size_t workspace_bytes, cudaStream_t stream);

// Writes the L2-normalized BF16 q/k rows of every column whose segment has a full chunk, with the
// l2norm Op's row arithmetic. Rows of shorter segments are not written.
void launch_segmented_qk_l2norm(const Tensor& q, const Tensor& k, segment_table segments,
                                Tensor& q_normalized, Tensor& k_normalized, cudaStream_t stream);

// Runs the recurrent remainder of every segment and writes its output columns. A segment with a
// full chunk continues from its chunk-end state with staged q/k consumed as supplied; a shorter
// segment starts from ssm_state with raw q/k, normalized in-kernel when normalize_qk is true.
void launch_recurrent_segmented_tails(const Tensor& q, const Tensor& k, const Tensor& q_staged,
                                      const Tensor& k_staged, const Tensor& v, const Tensor& g,
                                      const Tensor& beta, float scale, bool normalize_qk,
                                      const Tensor& ssm_state, const float* chunk_states,
                                      segment_table segments, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail::gated_delta_net
