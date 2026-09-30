#pragma once

// A4 shared-prefix segmented GQA attention: the prefix split plan, the own-column row windows,
// and the partial layout shared by the prefix split kernels (gqa_attention_segmented_{bf16,i8}.cuh)
// and the own-column kernel that folds them (gqa_attention_segmented_bf16.cuh).
//
// A prefix split CTA owns 64 GQA-packed query rows (row = column * group + local head) of one KV
// head and one balanced share of the cached prefix tiles. Every column sees the whole prefix, so
// prefix tiles are dense; only the partial last page masks keys at or past the prefix. Tiles are
// 64 keys, which is exactly one KV page. The own-column kernel reads the call's BF16 K/V instead
// of the cache, walks each row's segment window, and merges the prefix partials in fixed order.

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
    std::int32_t splits; // prefix splits; zero when the prefix is empty
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

// Balanced share [begin, end) of the prefix tiles for one split. The launcher never runs more
// splits than prefix tiles, so every share is nonempty.
struct GqaSegmentedPrefixShare {
    std::int32_t begin;
    std::int32_t end;
};

__device__ __forceinline__ GqaSegmentedPrefixShare
gqa_segmented_prefix_share(const GqaSegmentedParams& params, std::int32_t split) {
    return {split * params.prefix_tiles / params.splits,
            (split + 1) * params.prefix_tiles / params.splits};
}

// Own-column window [lo, hi] of one packed row: its segment start through its own column. Rows
// past the call are empty.
struct GqaSegmentedRowRange {
    std::int32_t lo;
    std::int32_t hi;
};

template <int GroupSize>
__device__ __forceinline__ GqaSegmentedRowRange
gqa_segmented_row_range(const GqaSegmentedParams& params, std::int32_t packed_row) {
    const std::int32_t column = packed_row / GroupSize;
    if (column >= params.columns) { return {kGqaSegmentedEmptyLo, -1}; }
    return {gqa_segmented_segment_start(params.segments, params.segment_count, column), column};
}

__device__ __forceinline__ bool gqa_segmented_in_range(std::int32_t column,
                                                       GqaSegmentedRowRange range) {
    return column >= range.lo && column <= range.hi;
}

__device__ __forceinline__ std::int64_t gqa_segmented_stat_index(int q_heads, int q_head,
                                                                 int column, int split,
                                                                 int columns) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(q_heads) *
               (static_cast<std::int64_t>(column) + static_cast<std::int64_t>(columns) * split);
}

} // namespace ninfer::ops
