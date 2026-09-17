#pragma once

#include "core/tensor.h"

#include "ops/common/int8_proj_launch.h"
#include "ops/common/small_t_rowsplit_mma.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q4_q5_gdn_input_int8_launch(const Tensor& x, const Weight& qk_weight,
                                 const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                 const Int8ProjWorkspace& scratch, cudaStream_t stream);

// K = 5120 is 80 groups: 256 tiles of 40 stages fill one wave at two CTAs per SM, so the
// route streams each parent once without split-K.
constexpr SmallTMmaSplitK q4_q5_gdn_input_small_t_split_k() noexcept {
    return small_t_mma_split_k(5120, 1);
}

void q4_q5_gdn_input_small_t_mma_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                        Tensor& z, const SmallTMmaWorkspace& scratch,
                                        cudaStream_t stream);

void q4_q5_gdn_input_grouped_mma_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                        cudaStream_t stream);

} // namespace ninfer::ops::detail
