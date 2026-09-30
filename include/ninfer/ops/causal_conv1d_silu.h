#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Depthwise causal width-4 convolution followed by SiLU. Let u[c,-3..-1] be the three values in
 * the input state and u[c,t]=x[c,t] for t>=0. Then
 *
 *   ideal[c,t] = SiLU(sum_{j=0..3} weight[c,j] * u[c,t-3+j]).
 *
 * `x` and `out` are contiguous BF16 [C,T], `weight` is contiguous BF16 [C,4], and a state is
 * contiguous BF16 [C,3] ordered oldest to newest. The oracle evaluates `ideal` naively in FP64 from
 * the represented inputs. The BF16 output is promoted and compared directly with that result;
 * output storage rounding belongs to the Op's numerical criterion, not the oracle. Kernel
 * accumulator and staging precision are implementation choices. Input, weight, output, and state
 * storage do not overlap except for the explicitly allowed exact alias between state input and
 * state output. No caller workspace is used. T may be any positive value.
 */

// Reads conv_state as the initial window and replaces it with the final three values after x.
void causal_conv1d_silu(const Tensor& x, const Tensor& weight, Tensor& conv_state, Tensor& out,
                        cudaStream_t stream);

// Distinct-state form. conv_state_in and conv_state_out may be disjoint or exactly the same
// storage; conv_state_out receives the trailing width-3 window of concat(conv_state_in,x).
void causal_conv1d_silu(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                        Tensor& conv_state_out, Tensor& out, cudaStream_t stream);

/**
 * Segmented form for S independent continuations of one read-only window. `x` and `out` are
 * contiguous BF16 [C,N], `conv_state` is contiguous BF16 [C,3], and `segments` is a contiguous
 * device I32 [2,S] table whose column s is (c_s,T_s), the first column and length of segment s.
 * The caller promises that the table tiles [0,N) in order: c_0=0, c_{s+1}=c_s+T_s, T_s>=1, and
 * c_{S-1}+T_{S-1}=N. The host checks only the shapes and 1<=S<=N.
 *
 * Output columns [c_s,c_s+T_s) receive `ideal` evaluated over x[:,c_s..c_s+T_s) with
 * u[c,-3..-1]=conv_state, so every segment restarts from the same window and no column reads
 * another segment. The oracle and numerical criterion are those of the single-sequence form. No
 * state is written: x, weight, conv_state, and segments are read-only, and out overlaps none of
 * them. No caller workspace is used.
 */
void causal_conv1d_silu_segmented(const Tensor& x, const Tensor& weight, const Tensor& conv_state,
                                  const Tensor& segments, Tensor& out, cudaStream_t stream);

/**
 * Snapshot form for B independent sequences. `x` and `out` are contiguous BF16 [C,W,B],
 * `conv_states` is contiguous BF16 [C,3,Slots], and `initial_state_slots` and
 * `snapshot_base_slots` are contiguous I32 [B]. `valid_columns` is either contiguous I32 [B],
 * with every value in [1,W], or an empty Tensor meaning every row has W valid columns. B=1
 * accepts every positive W; B=2..8 accepts W=1..16.
 *
 * Row b starts from the window selected by initial_state_slots[b]. After valid column j, the new
 * window is written to snapshot_base_slots[b]+j. Invalid-tail output columns are exact BF16 zero
 * and do not mutate state. The caller reserves the complete [base,base+W) interval for every row;
 * all row reservations are disjoint and no row overwrites another row's initial slot. A row's own
 * initial slot may lie in its destination reservation. `conv_states` is the only persistent state
 * mutated and must not overlap x, weight, metadata, or out.
 */
void causal_conv1d_silu_snapshot(const Tensor& x, const Tensor& weight, Tensor& conv_states,
                                 const Tensor& valid_columns, const Tensor& initial_state_slots,
                                 const Tensor& snapshot_base_slots, Tensor& out,
                                 cudaStream_t stream);

} // namespace ninfer::ops
