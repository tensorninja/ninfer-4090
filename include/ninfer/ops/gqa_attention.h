#pragma once

#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr std::uint32_t kGqaAttentionMaximumVisibleKeys = 1048576;

struct GqaExecutionEnvelope {
    std::uint32_t min_visible_keys = 0;
    std::uint32_t max_visible_keys = 0;
};

/**
 * Shared numerical contract for A1/A2/A3/A4.
 *
 * Public q/k/v inputs and BF16 cache values are interpreted after their BF16 storage boundary.
 * INT8-G64 cache rows use one FP16 scale for each contiguous 64-element group. For BF16 source
 * values x, their exact observable encoding is:
 *
 *   a          = max_i abs(FP32(x[i]))
 *   scale_bits = FP16_RNE(a / 127)
 *   s          = FP32(scale_bits)
 *   inv        = s == 0 ? 0 : FP32(1 / s)
 *   code[i]    = s == 0 ? 0 : I8(clamp(RNE_even(FP32(x[i]) * inv), -127, 127))
 *   decode[i]  = FP32(code[i]) * s
 *
 * The rotated caches (rk8v4, rk4v4, rk2v4-e8) encode each 64-element group of K and of V after
 * the orthonormal Walsh-Hadamard transform y = H64 x (natural order, scaled by 1/8, evaluated in
 * FP32). rk8v4 K applies INT8-G64 above to y; rk2v4-e8 K stores E8-root codes
 * (ops/kernel/e8_root_codec.cuh). Every packed 4-bit plane (rk8v4 V, rk4v4 K and V, rk2v4-e8 V)
 * holds the midrise codec, whose sixteen levels are the odd multiples of a half step h:
 *
 *   a          = max_i abs(y[i])
 *   scale_bits = FP16_RNE(a / 15)
 *   h          = FP32(scale_bits)
 *   inv        = h == 0 ? 0 : FP32(1 / h)
 *   code[i]    = clamp(floor(FP32(y[i] * inv) / 2), -8, 7)
 *   decode[i]  = (2 * code[i] + 1) * h
 *
 * The logical cache value of a rotated group is H64 applied to its decoded values.
 *
 * A1 and A2 produce identical code and scale bits. The common ideal attention oracle uses BF16 Q
 * and logical cache values (BF16 values for a BF16 cache, the FP32 decode above for INT8-G64 and
 * the rotated caches), then evaluates score dot products, stable softmax, and value reduction in
 * FP64. The BF16 Op output is promoted to FP64 for comparison with that result.
 *
 * The registered INT8 implementation defines Q8-G64, paired with INT8-G64 K, as its native query
 * compute profile. Its profile-defined query quantization and any narrower staging do not replace
 * BF16 Q in the ideal oracle. BF16-cache and INT8-cache compute profiles therefore have separate
 * named numerical criteria owned by the GQA conformance test. Those envelopes apply to the
 * registered geometries, tested token extents, conformance matrix, and target-representative
 * activation range; they are not a universal error bound for arbitrary adversarial BF16 tensors.
 * A1, A3, and A4 are each qualified directly against the ideal oracle. A1-versus-A3 and
 * A4-versus-per-segment-A1 parity are only additional consistency checks.
 */

/**
 * Returns the transient arena capacity required for every W in the inclusive interval at one
 * exact logical batch size. Head geometry, cache dtype, and execution envelope are the fixed
 * implementation profile. Invalid profiles or intervals throw; a legal B=1 prompt route may
 * return zero.
 */
[[nodiscard]] std::size_t
gqa_attention_workspace_capacity_bytes(std::int32_t q_heads, DType cache_dtype,
                                       GqaExecutionEnvelope envelope, std::int32_t batch_size,
                                       std::int32_t min_width, std::int32_t max_width);

/**
 * A1: append K/V for B independent sequences and compute causal grouped-query attention. Let
 * Vb=W when valid_columns is empty and Vb=valid_columns[b] otherwise. For row b, query head h,
 * kvh=floor(h/group), 0<=j<Vb, p=positions[j,b], and that row's populated cache history [0,p]:
 *
 *   score[x]      = scale * dot(q[:,h,j,b], K_cache[b][:,x,kvh]), 0 <= x <= p
 *   probability   = softmax_x(score)
 *   ideal[:,h,j,b] = sum_x probability[x] * V_cache[b][:,x,kvh].
 *
 * The registered geometries are `[256,24|4,W,B]` group 6 and `[256,16|2,W,B]` group 8.
 * q/k/v/out are contiguous BF16 in request-major order, positions is contiguous I32 [W,B], and
 * kv_table_rows is contiguous I32 [B]. valid_columns is either contiguous I32 [B], or an empty
 * Tensor meaning every row has exactly W valid columns. This dense/masked choice is part of the
 * call topology; it is not inferred by copying device metadata to the host. B=1 accepts every
 * positive W in the current prefill/decode domain; B=2..8 accepts W=1..16. Cache storage is BF16,
 * INT8-G64, or a rotated cache under the shared numerical contract above. PagedKVBatchLayerView
 * supplies shared planes and the complete block-table matrix; kv_table_rows[b] selects one row
 * for sequence b.
 *
 * In masked form, every row's valid columns are the prefix [0,valid_columns[b]); positions in that
 * prefix are sequential and address populated causal histories. Each nonempty row repeats its
 * final valid position through the invalid tail; an empty row uses zero positions. Other
 * invalid-tail inputs contain safe dummy values. A1 does not modify cache for invalid columns and
 * writes exact BF16 zero to their output. The caller guarantees that the maximum final valid
 * position plus one over nonempty rows lies in the declared execution envelope. The envelope is a
 * host launch-resource promise over that batch maximum; it does not alter any row's causal mask.
 *
 * q/k/v/positions/valid_columns/kv_table_rows/out, every cache plane/table, and live workspace
 * suballocations are pairwise non-overlapping. The Op overwrites every addressed cache row but
 * owns no persistent frontier, allocation, request identity, or commit authority.
 */
void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   const Tensor& valid_columns, const Tensor& kv_table_rows, float scale,
                   PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

/**
 * A2: perform only the cache-write part of A1. k/v are contiguous BF16 `[256,4|2,T]`, positions is
 * contiguous sequential I32 [T], and every addressed code and INT8 scale is overwritten. It reads
 * no unrelated cache row, receives no execution envelope, and owns no persistent frontier.
 */
void gqa_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                   PagedKVLayerView cache, cudaStream_t stream);

/**
 * A3: compute causal attention from an already populated cache without accepting new K/V or
 * mutating any cache plane. q/out are contiguous BF16 `[256,24|16,T]`, positions is contiguous
 * sequential I32 [T], and the mathematical formula and execution-envelope contract are identical
 * to A1. Caller workspace is reported by gqa_attention_workspace_capacity_bytes().
 */
void gqa_attention_cached(const Tensor& q, const Tensor& positions, float scale,
                          const PagedKVLayerView& cache, GqaExecutionEnvelope envelope,
                          WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

/**
 * Returns the transient arena capacity required by gqa_attention_segmented() for every N in
 * [1, max_columns], every prefix, and every segment table. It does not depend on the prefix.
 * Invalid intervals throw.
 */
[[nodiscard]] std::size_t gqa_attention_segmented_workspace_capacity_bytes(std::int32_t max_columns);

/**
 * A4: compute segment-causal grouped-query attention for N columns of one sequence behind a
 * populated shared prefix, without writing the cache. The only registered geometry is group 6:
 * q/out are contiguous BF16 `[256,24,N]` and k/v are contiguous BF16 `[256,4,N]`, N >= 1.
 * segments is contiguous I32 [2,S] holding (c_s, T_s) pairs with c_0 = 0, T_s >= 1, and
 * c_{s+1} = c_s + T_s, so the S segments tile [0,N) in order. kv_table_rows is contiguous I32 [1]
 * selecting one row of the PagedKVBatchLayerView block-table matrix. prefix is a host value with
 * 0 <= prefix no larger than the row capacity and prefix + N no larger than
 * kGqaAttentionMaximumVisibleKeys.
 *
 * Keys and values below the prefix are the logical cache values; a segment's own keys and values
 * are its columns of k/v at their BF16 storage boundary. For column i of segment s
 * (c_s <= i < c_s + T_s), query head h, and kvh = floor(h/6):
 *
 *   score[x]     = scale * dot(q[:,h,i], K_cache[:,x,kvh]),  0 <= x < prefix
 *   score[j]     = scale * dot(q[:,h,i], k[:,kvh,j]),        c_s <= j <= i
 *   probability  = softmax over both score sets
 *   ideal[:,h,i] = sum_x probability[x] * V_cache[:,x,kvh] + sum_j probability[j] * v[:,kvh,j].
 *
 * Every segment sees the whole shared prefix and none of another segment's columns. The prefix
 * follows the numerical contract and the BF16-cache/INT8-cache compute profiles of A1 above; the
 * own columns always take the BF16 profile. The segment table and the population of [0, prefix)
 * are caller promises that the host does not read; it checks shapes and ranges only. No cache
 * plane is modified, and a repeated call on identical inputs is bit-identical.
 * q/k/v/segments/kv_table_rows/out, every cache plane/table, and live workspace suballocations are
 * pairwise non-overlapping. The Op owns no persistent frontier, allocation, request identity, or
 * commit authority.
 */
void gqa_attention_segmented(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& segments, const Tensor& kv_table_rows,
                             std::int32_t prefix, float scale, PagedKVBatchLayerView cache,
                             WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
