#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/common/small_t_rowsplit_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// A tile is 32 gate rows and their 32 up rows; the tile epilogue finishes silu(gate) * up,
// so the route never materializes the gate/up projections.
template <int TileCols>
void launch_tile(const Tensor& x, const Weight& w, Tensor& out, const SmallTMmaWorkspace& scratch,
                 const SmallTMmaSplitK& split_k, cudaStream_t stream) {
    SmallTMmaParams p;
    p.x              = static_cast<const __nv_bfloat16*>(x.data);
    p.k              = x.ne[0];
    p.cols           = x.ne[1];
    p.job0.codes     = static_cast<const std::uint8_t*>(w.qdata);
    p.job0.scales    = static_cast<const std::uint8_t*>(w.scales);
    p.job0.out0      = static_cast<__nv_bfloat16*>(out.data);
    p.job0.n         = w.n;
    p.job0.split_row = w.n;
    p.job0.ld0       = out.ne[0];
    p.tiles0         = out.ne[0] / (SmallTMmaShape::kRows / 2);
    p.groups_per_split = split_k.groups_per_split;
    p.partials       = scratch.partials;
    p.counters       = scratch.counters;

    const dim3 grid(static_cast<unsigned>(p.tiles0), static_cast<unsigned>(split_k.splits), 1u);
    if (split_k.splits > 1) {
        CUDA_CHECK(cudaMemsetAsync(scratch.counters, 0, sizeof(int) * p.tiles0, stream));
    }
    // 544 tiles of 40 stages: a fourth stage covers the per-tile pipeline ramp, and at eight
    // columns a third resident CTA measured faster still (150.7 vs 155.6 us at T=4).
    constexpr int kStages    = TileCols <= 24 ? 4 : 3;
    constexpr int kMinBlocks = TileCols == 8 ? 3 : 2;
    small_t_rowsplit_mma_kernel<TileCols, kStages, kMinBlocks, SmallTMmaEpilogue::SwiGlu,
                                SmallTMmaCodec::Q4><<<grid, SmallTMmaShape::kThreads, 0, stream>>>(p);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4_linear_swiglu_small_t_mma_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         const SmallTMmaWorkspace& scratch,
                                         cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (cols < 1 || cols > SmallTMmaShape::kMaxCols || w.n != 2 * out.ne[0] ||
        (out.ne[0] % (SmallTMmaShape::kRows / 2)) != 0 || w.k != w.padded_shape[1] ||
        w.k != x.ne[0] || (w.k % (2 * SmallTMmaShape::kGroupK)) != 0) {
        throw std::invalid_argument("q4 linear_swiglu small-T mma: shape outside the route");
    }
    const SmallTMmaSplitK split_k = q4_linear_swiglu_small_t_split_k(w.k);
    if (split_k.splits > 1 && (scratch.partials == nullptr || scratch.counters == nullptr)) {
        throw std::invalid_argument("q4 linear_swiglu small-T mma: split-K workspace missing");
    }
    switch (small_t_mma_tile_cols(cols)) {
    case 8:
        launch_tile<8>(x, w, out, scratch, split_k, stream);
        return;
    case 16:
        launch_tile<16>(x, w, out, scratch, split_k, stream);
        return;
    case 24:
        launch_tile<24>(x, w, out, scratch, split_k, stream);
        return;
    default:
        launch_tile<32>(x, w, out, scratch, split_k, stream);
        return;
    }
}

} // namespace ninfer::ops::detail
