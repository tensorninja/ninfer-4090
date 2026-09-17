#pragma once

#include "core/tensor.h"

#include "ops/common/int8_proj_launch.h"
#include "ops/common/small_t_rowsplit_mma.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q4_q5_attn_input_int8_launch(const Tensor& x, const Weight& query_key_weight,
                                  const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, const Int8ProjWorkspace& scratch,
                                  cudaStream_t stream);

// K = 5120 is 80 groups: 224 tiles of 40 stages fill one wave at two CTAs per SM, so the
// route streams each parent once without split-K (two splits measured 39 vs 33 us at T=2).
constexpr SmallTMmaSplitK q4_q5_attn_input_small_t_split_k() noexcept {
    return small_t_mma_split_k(5120, 1);
}

void q4_q5_attn_input_small_t_mma_launch(const Tensor& x, const Weight& query_key_weight,
                                         const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                         Tensor& k, Tensor& v, const SmallTMmaWorkspace& scratch,
                                         cudaStream_t stream);

void q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(const Tensor& x, const Weight& query_key_weight,
                                                    const Weight& gate_value_weight, Tensor& q,
                                                    Tensor& gate, Tensor& k, Tensor& v,
                                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
