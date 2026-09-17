#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/common/small_t_rowsplit_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kQkRows     = 4096;
constexpr std::int32_t kValueRows  = 6144;
constexpr std::int32_t kZRows      = 6144;
constexpr std::int32_t kValueZRows = kValueRows + kZRows;
constexpr std::int32_t kHidden     = 5120;

std::int32_t leading_dim(const Tensor& t) {
    return static_cast<std::int32_t>(t.nb[1] / sizeof(__nv_bfloat16));
}

// One launch covers both parents: tiles [0, 64) stream the Q4 query/key rows into the head of
// qkv, tiles [64, 256) the Q5 value/z rows into the qkv tail and z. The 6144-row value/z split
// falls on a tile boundary.
template <int TileCols>
void launch_tile(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                 Tensor& qk, Tensor& value, Tensor& z, const SmallTMmaWorkspace& scratch,
                 const SmallTMmaSplitK& split_k, cudaStream_t stream) {
    SmallTMmaParams p;
    p.x                = static_cast<const __nv_bfloat16*>(x.data);
    p.k                = kHidden;
    p.cols             = x.ne[1];
    p.groups_per_split = split_k.groups_per_split;
    p.partials         = scratch.partials;
    p.counters         = scratch.counters;

    p.job0.codes     = static_cast<const std::uint8_t*>(qk_weight.qdata);
    p.job0.scales    = static_cast<const std::uint8_t*>(qk_weight.scales);
    p.job0.out0      = static_cast<__nv_bfloat16*>(qk.data);
    p.job0.n         = kQkRows;
    p.job0.split_row = kQkRows;
    p.job0.ld0       = leading_dim(qk);

    p.job1.codes     = static_cast<const std::uint8_t*>(value_z_weight.qdata);
    p.job1.high      = static_cast<const std::uint8_t*>(value_z_weight.qhigh);
    p.job1.scales    = static_cast<const std::uint8_t*>(value_z_weight.scales);
    p.job1.out0      = static_cast<__nv_bfloat16*>(value.data);
    p.job1.out1      = static_cast<__nv_bfloat16*>(z.data);
    p.job1.n         = kValueZRows;
    p.job1.split_row = kValueRows;
    p.job1.ld0       = leading_dim(value);
    p.job1.ld1       = leading_dim(z);
    p.job1.q5        = true;
    p.tiles0         = kQkRows / SmallTMmaShape::kRows;

    const std::int32_t tiles = p.tiles0 + kValueZRows / SmallTMmaShape::kRows;
    const dim3 grid(static_cast<unsigned>(tiles), static_cast<unsigned>(split_k.splits), 1u);
    if (split_k.splits > 1) {
        CUDA_CHECK(cudaMemsetAsync(scratch.counters, 0, sizeof(int) * tiles, stream));
    }
    constexpr int kStages    = 3;
    constexpr int kMinBlocks = 2;
    small_t_rowsplit_mma_kernel<TileCols, kStages, kMinBlocks, SmallTMmaEpilogue::Store,
                                SmallTMmaCodec::Mixed>
        <<<grid, SmallTMmaShape::kThreads, 0, stream>>>(p);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4_q5_gdn_input_small_t_mma_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                        Tensor& z, const SmallTMmaWorkspace& scratch,
                                        cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (cols < 1 || cols > SmallTMmaShape::kMaxCols || x.ne[0] != kHidden ||
        qk_weight.n != kQkRows || value_z_weight.n != kValueZRows ||
        qk_weight.padded_shape[1] != kHidden || value_z_weight.padded_shape[1] != kHidden ||
        qk.ne[0] != kQkRows || value.ne[0] != kValueRows || z.ne[0] != kZRows) {
        throw std::invalid_argument("q4/q5 GDN input small-T mma: shape outside the route");
    }
    const SmallTMmaSplitK split_k = q4_q5_gdn_input_small_t_split_k();
    if (split_k.splits > 1 && (scratch.partials == nullptr || scratch.counters == nullptr)) {
        throw std::invalid_argument("q4/q5 GDN input small-T mma: split-K workspace missing");
    }
    switch (small_t_mma_tile_cols(cols)) {
    case 8:
        launch_tile<8>(x, qk_weight, value_z_weight, qk, value, z, scratch, split_k, stream);
        return;
    case 16:
        launch_tile<16>(x, qk_weight, value_z_weight, qk, value, z, scratch, split_k, stream);
        return;
    case 24:
        launch_tile<24>(x, qk_weight, value_z_weight, qk, value, z, scratch, split_k, stream);
        return;
    default:
        launch_tile<32>(x, qk_weight, value_z_weight, qk, value, z, scratch, split_k, stream);
        return;
    }
}

} // namespace ninfer::ops::detail
