#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: pointer_head_project
 *
 * Math / indexing:
 *   out[p,r] = bias[p] + sum_{d<D} weight[d,p] * hidden[d,r],  0<=p<P, 0<=r<R.
 *
 *   Column p of weight holds the D contiguous values of row p of the mathematical projection
 *   matrix (the row-major [P,D] weight of a linear layer), so each output column is the affine
 *   projection of one hidden column.
 *
 * Logical shapes:
 *   hidden is contiguous BF16 [D,R], weight is contiguous BF16 [D,P], bias is contiguous BF16 [P],
 *   and out is contiguous FP32 [P,R]. Dimension zero is stored fastest. A column range
 *   `buffer.slice(1,c0,R)` of a contiguous FP32 [P,C] buffer is itself a contiguous [P,R] view and
 *   is a valid out; the call then writes exactly columns [c0,c0+R) of that buffer.
 *
 * Supported domain:
 *   D is a positive multiple of 32, 1<=P<=65536, R>=1, and P*D and R*D are below 2^31. hidden and
 *   weight start 16-byte aligned.
 *
 * Numeric:
 *   BF16 inputs decode exactly. The oracle evaluates the formula naively in FP64 from the
 *   represented values; out is an FP32 approximation with the error of FP32 accumulation over the
 *   D products and the bias. Accumulation order, partial-sum decomposition, operand staging, and
 *   instructions are private. The reduction order is fixed for given extents, so repeated calls
 *   with the same inputs and R are bit-identical.
 *
 * Effects:
 *   Writes every element of out and nothing else. out must not overlap hidden, weight, or bias.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueued on `stream` without host synchronization or device allocation; CUDA Graph capturable.
 */
void pointer_head_project(const Tensor& hidden, const Tensor& weight, const Tensor& bias,
                          Tensor& out, cudaStream_t stream);

/**
 * Op: pointer_head_score
 *
 * Math / indexing:
 *   For every question q<Qn, with c=questions[0,q], b=questions[1,q], K=questions[2,q]:
 *
 *     z_i = scale * sum_{p<P} keys[p,b+i] * queries[p,c],        0<=i<K
 *     probabilities[b+i] = exp(z_i - max_j z_j) / sum_{j<K} exp(z_j - max_j z_j).
 *
 * Logical shapes:
 *   queries is contiguous FP32 [P,Nq], keys is contiguous FP32 [P,Nk], questions is contiguous
 *   I32 [3,Qn] (dimension zero holds query column, first key column, key count), and
 *   probabilities is contiguous FP32 [Nk]. The caller guarantees for every question 0<=c<Nq,
 *   1<=K<=255, 0<=b, b+K<=Nk, and that the key ranges of distinct questions are disjoint; they
 *   may appear in any order and need not cover all Nk key columns.
 *
 * Supported domain:
 *   P is a positive multiple of 4, queries and keys start 16-byte aligned, and scale is a finite
 *   host value.
 *
 * Numeric:
 *   The oracle evaluates the formula naively in FP64 from the represented FP32 queries and keys.
 *   Logits and probabilities are private FP32 approximations; dot-product order, exponent
 *   evaluation, and reduction trees are private and fixed for given extents, so repeated calls with
 *   the same inputs and extents are bit-identical.
 *
 * Effects:
 *   Writes probabilities[b,b+K) of every question; every other element keeps its old value.
 *   queries and keys may alias each other; probabilities must not overlap queries, keys, or
 *   questions.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueued on `stream` without host synchronization or device allocation; CUDA Graph capturable.
 *   The question table is read on the device.
 */
void pointer_head_score(const Tensor& queries, const Tensor& keys, const Tensor& questions,
                        float scale, Tensor& probabilities, cudaStream_t stream);

} // namespace ninfer::ops
