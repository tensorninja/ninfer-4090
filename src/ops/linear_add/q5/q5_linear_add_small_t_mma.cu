#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/common/small_t_rowsplit_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <int TileCols>
void launch_tile(const Tensor& x, const Weight& w, Tensor& residual_out,
                 const SmallTMmaWorkspace& scratch, const SmallTMmaSplitK& split_k,
                 cudaStream_t stream) {
    SmallTMmaParams p;
    p.x                = static_cast<const __nv_bfloat16*>(x.data);
    p.k                = x.ne[0];
    p.cols             = x.ne[1];
    p.groups_per_split = split_k.groups_per_split;
    p.job0.codes       = static_cast<const std::uint8_t*>(w.qdata);
    p.job0.high        = static_cast<const std::uint8_t*>(w.qhigh);
    p.job0.scales      = static_cast<const std::uint8_t*>(w.scales);
    p.job0.out0        = static_cast<__nv_bfloat16*>(residual_out.data);
    p.job0.n           = residual_out.ne[0];
    p.job0.split_row   = residual_out.ne[0];
    p.job0.ld0         = residual_out.ne[0];
    p.job0.q5          = true;
    p.tiles0           = residual_out.ne[0] / SmallTMmaShape::kRows;
    p.partials         = scratch.partials;
    p.counters         = scratch.counters;

    const dim3 grid(static_cast<unsigned>(p.tiles0), static_cast<unsigned>(split_k.splits), 1u);
    if (split_k.splits > 1) {
        CUDA_CHECK(cudaMemsetAsync(scratch.counters, 0, sizeof(int) * p.tiles0, stream));
    }
    small_t_rowsplit_mma_kernel<TileCols, SmallTMmaEpilogue::Residual, SmallTMmaCodec::Q5>
        <<<grid, SmallTMmaShape::kThreads, 0, stream>>>(p);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q5_linear_add_small_t_mma_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      const SmallTMmaWorkspace& scratch, cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (cols < 1 || cols > SmallTMmaShape::kMaxCols || (w.n % SmallTMmaShape::kRows) != 0 ||
        w.k != w.padded_shape[1] || (w.k % (2 * SmallTMmaShape::kGroupK)) != 0) {
        throw std::invalid_argument("q5 linear_add small-T mma: shape outside the route");
    }
    const SmallTMmaSplitK split_k = q5_linear_add_small_t_split_k(w.k);
    if (split_k.splits > 1 && (scratch.partials == nullptr || scratch.counters == nullptr)) {
        throw std::invalid_argument("q5 linear_add small-T mma: split-K workspace missing");
    }
    switch (small_t_mma_tile_cols(cols)) {
    case 8:
        launch_tile<8>(x, w, residual_out, scratch, split_k, stream);
        return;
    case 16:
        launch_tile<16>(x, w, residual_out, scratch, split_k, stream);
        return;
    case 24:
        launch_tile<24>(x, w, residual_out, scratch, split_k, stream);
        return;
    default:
        launch_tile<32>(x, w, residual_out, scratch, split_k, stream);
        return;
    }
}

} // namespace ninfer::ops::detail
