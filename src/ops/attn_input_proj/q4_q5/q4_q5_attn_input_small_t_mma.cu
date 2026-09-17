#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/small_t_rowsplit_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kParentRows = 7168;
constexpr std::int32_t kSplitRow   = 6144;
constexpr std::int32_t kHidden     = 5120;

std::int32_t leading_dim(const Tensor& t) {
    return static_cast<std::int32_t>(t.nb[1] / sizeof(__nv_bfloat16));
}

// One launch covers both parents: tiles [0, 112) stream the Q4 query/key rows, tiles
// [112, 224) the Q5 gate/value rows. Each tile stores its 64 rows into the head or tail output
// of its parent; the 6144-row split falls on a tile boundary.
template <int TileCols>
void launch_tile(const Tensor& x, const Weight& query_key, const Weight& gate_value, Tensor& q,
                 Tensor& gate, Tensor& k, Tensor& v, const SmallTMmaWorkspace& scratch,
                 const SmallTMmaSplitK& split_k, cudaStream_t stream) {
    SmallTMmaParams p;
    p.x                = static_cast<const __nv_bfloat16*>(x.data);
    p.k                = kHidden;
    p.cols             = x.ne[1];
    p.groups_per_split = split_k.groups_per_split;
    p.partials         = scratch.partials;
    p.counters         = scratch.counters;

    p.job0.codes     = static_cast<const std::uint8_t*>(query_key.qdata);
    p.job0.scales    = static_cast<const std::uint8_t*>(query_key.scales);
    p.job0.out0      = static_cast<__nv_bfloat16*>(q.data);
    p.job0.out1      = static_cast<__nv_bfloat16*>(k.data);
    p.job0.n         = kParentRows;
    p.job0.split_row = kSplitRow;
    p.job0.ld0       = leading_dim(q);
    p.job0.ld1       = leading_dim(k);

    p.job1.codes     = static_cast<const std::uint8_t*>(gate_value.qdata);
    p.job1.high      = static_cast<const std::uint8_t*>(gate_value.qhigh);
    p.job1.scales    = static_cast<const std::uint8_t*>(gate_value.scales);
    p.job1.out0      = static_cast<__nv_bfloat16*>(gate.data);
    p.job1.out1      = static_cast<__nv_bfloat16*>(v.data);
    p.job1.n         = kParentRows;
    p.job1.split_row = kSplitRow;
    p.job1.ld0       = leading_dim(gate);
    p.job1.ld1       = leading_dim(v);
    p.job1.q5        = true;
    p.tiles0         = kParentRows / SmallTMmaShape::kRows;

    const std::int32_t tiles = 2 * p.tiles0;
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

void q4_q5_attn_input_small_t_mma_launch(const Tensor& x, const Weight& query_key_weight,
                                         const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                         Tensor& k, Tensor& v, const SmallTMmaWorkspace& scratch,
                                         cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (cols < 1 || cols > SmallTMmaShape::kMaxCols || x.ne[0] != kHidden ||
        query_key_weight.n != kParentRows || gate_value_weight.n != kParentRows ||
        query_key_weight.padded_shape[1] != kHidden ||
        gate_value_weight.padded_shape[1] != kHidden || q.ne[0] != kSplitRow ||
        gate.ne[0] != kSplitRow || k.ne[0] != kParentRows - kSplitRow ||
        v.ne[0] != kParentRows - kSplitRow) {
        throw std::invalid_argument("q4/q5 attention input small-T mma: shape outside the route");
    }
    const SmallTMmaSplitK split_k = q4_q5_attn_input_small_t_split_k();
    if (split_k.splits > 1 && (scratch.partials == nullptr || scratch.counters == nullptr)) {
        throw std::invalid_argument(
            "q4/q5 attention input small-T mma: split-K workspace missing");
    }
    switch (small_t_mma_tile_cols(cols)) {
    case 8:
        launch_tile<8>(x, query_key_weight, gate_value_weight, q, gate, k, v, scratch, split_k,
                       stream);
        return;
    case 16:
        launch_tile<16>(x, query_key_weight, gate_value_weight, q, gate, k, v, scratch, split_k,
                        stream);
        return;
    case 24:
        launch_tile<24>(x, query_key_weight, gate_value_weight, q, gate, k, v, scratch, split_k,
                        stream);
        return;
    default:
        launch_tile<32>(x, query_key_weight, gate_value_weight, q, gate, k, v, scratch, split_k,
                        stream);
        return;
    }
}

} // namespace ninfer::ops::detail
