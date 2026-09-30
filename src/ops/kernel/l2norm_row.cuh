#pragma once

// ninfer::ops - L2Norm fast-domain row body. It is separate from the l2norm kernels so that other
// Ops that stage L2-normalized rows can share this exact arithmetic without defining those kernels.

#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops {

// D=2*pairs in {64, 128, 192, 256}. The calling warp owns one row and keeps its input in registers.
__device__ __forceinline__ void l2norm_warp_row_bf16x2(const __nv_bfloat162* x_row,
                                                       __nv_bfloat162* out_row, int pairs,
                                                       float eps, int lane) {
    constexpr int kMaxPairsPerLane = 4;
    __nv_bfloat162 values[kMaxPairsPerLane];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < kMaxPairsPerLane; ++k) {
        const int pair = lane + k * kWarpSize;
        if (pair < pairs) {
            values[k]       = x_row[pair];
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    sum       = warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum + eps) : 0.0f;
    inv       = __shfl_sync(kFullWarpMask, inv, 0);

#pragma unroll
    for (int k = 0; k < kMaxPairsPerLane; ++k) {
        const int pair = lane + k * kWarpSize;
        if (pair < pairs) {
            const float2 xf = __bfloat1622float2(values[k]);
            out_row[pair]   = __floats2bfloat162_rn(xf.x * inv, xf.y * inv);
        }
    }
}

} // namespace ninfer::ops
