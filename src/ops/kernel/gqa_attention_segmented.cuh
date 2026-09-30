#pragma once

// A4 shared-prefix segmented GQA attention: the split key-tile plan, per-row visibility, and
// the fixed-order split merge shared by the BF16 and INT8 split kernels
// (gqa_attention_segmented_{bf16,i8}.cuh).
//
// One CTA owns 64 GQA-packed query rows of one KV head (row = column * group + local head) and
// one split of the shared prefix. Every column sees the whole prefix, so the prefix splits are
// unmasked dense tiles; the last split also walks the tiles of the appended suffix under the
// per-row segment-causal mask. Tiles are 64 keys, which is exactly one KV page.

#include "ops/common/math.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaSegmentedHeadDim = 256;
inline constexpr int kGqaSegmentedRows    = 64;
inline constexpr int kGqaSegmentedKeys    = kPagedKVPageSize;
inline constexpr int kGqaSegmentedEmptyLo = 0x7fffffff;

static_assert(kGqaSegmentedKeys == 64);

struct GqaSegmentedParams {
    const std::int32_t* segments;   // contiguous I32 [2, segment_count] (start, length) pairs
    const std::int32_t* tables;     // block tables [logical_pages, table_rows]
    const std::int32_t* table_rows; // I32 [1]
    std::int32_t segment_count;
    std::int32_t table_stride;
    std::int32_t prefix;
    std::int32_t columns;
    std::int32_t prefix_tiles;
    std::int32_t splits;
    float scale_log2;
    float* partial_acc; // FP32 [256, q_heads, columns, splits], unnormalized
    float* partial_m;   // FP32 [q_heads, columns, splits], running max in the log2 domain
    float* partial_l;   // FP32 [q_heads, columns, splits]
};

// Start of the segment that owns `column`. The segment table tiles [0, columns) in order.
__device__ __forceinline__ std::int32_t gqa_segmented_segment_start(const std::int32_t* segments,
                                                                    std::int32_t count,
                                                                    std::int32_t column) {
    std::int32_t lo = 0;
    std::int32_t hi = count - 1;
    while (lo < hi) {
        const std::int32_t mid = (lo + hi + 1) >> 1;
        if (__ldg(&segments[2 * mid]) <= column) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return __ldg(&segments[2 * lo]);
}

// Visible absolute-key interval [lo, hi] of one packed row inside the appended suffix. Every
// row additionally sees the whole prefix [0, prefix). Rows past the call are empty.
struct GqaSegmentedRowRange {
    std::int32_t lo;
    std::int32_t hi;
};

template <int GroupSize>
__device__ __forceinline__ GqaSegmentedRowRange
gqa_segmented_row_range(const GqaSegmentedParams& params, std::int32_t packed_row, bool needed) {
    const std::int32_t column = packed_row / GroupSize;
    if (!needed || column >= params.columns) { return {kGqaSegmentedEmptyLo, -1}; }
    return {params.prefix +
                gqa_segmented_segment_start(params.segments, params.segment_count, column),
            params.prefix + column};
}

__device__ __forceinline__ bool gqa_segmented_visible(std::int32_t key, std::int32_t prefix,
                                                      GqaSegmentedRowRange range) {
    return key < prefix || (key >= range.lo && key <= range.hi);
}

// Key tiles walked by one CTA: its balanced share of the prefix tiles, followed (last split
// only) by the suffix tiles that intersect its rows' segment windows. A suffix tile that is
// also the partial last prefix tile is walked once, as a prefix tile.
struct GqaSegmentedTilePlan {
    std::int32_t prefix_begin;
    std::int32_t prefix_count;
    std::int32_t suffix_begin;
    std::int32_t suffix_count;

    __device__ __forceinline__ std::int32_t count() const { return prefix_count + suffix_count; }

    __device__ __forceinline__ std::int32_t tile(std::int32_t index) const {
        return index < prefix_count ? prefix_begin + index
                                    : suffix_begin + (index - prefix_count);
    }
};

template <int GroupSize>
__device__ __forceinline__ GqaSegmentedTilePlan
gqa_segmented_tile_plan(const GqaSegmentedParams& params, std::int32_t split,
                        std::int32_t first_row, std::int32_t last_row) {
    GqaSegmentedTilePlan plan{};
    plan.prefix_begin             = split * params.prefix_tiles / params.splits;
    const std::int32_t prefix_end = (split + 1) * params.prefix_tiles / params.splits;
    plan.prefix_count             = prefix_end - plan.prefix_begin;
    plan.suffix_begin             = prefix_end;
    plan.suffix_count             = 0;
    if (split == params.splits - 1) {
        const std::int32_t first_column = first_row / GroupSize;
        const std::int32_t last_column  = last_row / GroupSize;
        const std::int32_t first_key =
            params.prefix +
            gqa_segmented_segment_start(params.segments, params.segment_count, first_column);
        const std::int32_t last_key = params.prefix + last_column;
        plan.suffix_begin           = max(first_key / kGqaSegmentedKeys, prefix_end);
        plan.suffix_count           = last_key / kGqaSegmentedKeys + 1 - plan.suffix_begin;
    }
    return plan;
}

__device__ __forceinline__ std::int64_t gqa_segmented_stat_index(int q_heads, int q_head,
                                                                 int column, int split,
                                                                 int columns) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(q_heads) *
               (static_cast<std::int64_t>(column) + static_cast<std::int64_t>(columns) * split);
}

// Fixed split order: the merged value is a pure function of the partials, so a repeated call is
// bit-identical. Rotated V leaves the Hadamard domain here, before the single BF16 rounding.
template <int QHeads, bool RotateV>
__launch_bounds__(128) __global__
    void gqa_attention_segmented_merge_kernel(const float* __restrict__ partial_acc,
                                              const float* __restrict__ partial_m,
                                              const float* __restrict__ partial_l,
                                              std::int32_t columns, std::int32_t splits,
                                              __nv_bfloat16* __restrict__ out) {
    const int column = static_cast<int>(blockIdx.x);
    const int q_head = static_cast<int>(blockIdx.y);
    const int group  = static_cast<int>(threadIdx.x) >> 5;
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int d0     = group * kGqaKvQuantGroup + lane;
    const int d1     = d0 + 32;

    const std::int64_t split_stride = static_cast<std::int64_t>(QHeads) * columns;
    const std::int64_t stat0 = gqa_segmented_stat_index(QHeads, q_head, column, 0, columns);
    float maximum            = -CUDART_INF_F;
    for (int split = 0; split < splits; ++split) {
        maximum = fmaxf(maximum, partial_m[stat0 + split * split_stride]);
    }
    float sum = 0.0f;
    float a0  = 0.0f;
    float a1  = 0.0f;
    for (int split = 0; split < splits; ++split) {
        const std::int64_t stat = stat0 + split * split_stride;
        const float weight      = exp2f(partial_m[stat] - maximum);
        const float* acc        = partial_acc + kGqaSegmentedHeadDim * stat;
        sum                     = fmaf(weight, partial_l[stat], sum);
        a0                      = fmaf(weight, acc[d0], a0);
        a1                      = fmaf(weight, acc[d1], a1);
    }
    const float inverse = 1.0f / sum;
    float x0            = a0 * inverse;
    float x1            = a1 * inverse;
    if constexpr (RotateV) { gqa_kv_hadamard64(x0, x1); }
    __nv_bfloat16* row = out + kGqaSegmentedHeadDim * stat0;
    row[d0]            = __float2bfloat16(x0);
    row[d1]            = __float2bfloat16(x1);
}

} // namespace ninfer::ops
