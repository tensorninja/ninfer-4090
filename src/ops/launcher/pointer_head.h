#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void pointer_head_project_launch(const Tensor& hidden, const Tensor& weight, const Tensor& bias,
                                 Tensor& out, cudaStream_t stream);
void pointer_head_score_launch(const Tensor& queries, const Tensor& keys, const Tensor& questions,
                               float scale, Tensor& probabilities, cudaStream_t stream);

} // namespace ninfer::ops::detail
