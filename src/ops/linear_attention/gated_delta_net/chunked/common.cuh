#pragma once

#include "ops/common/segment_table.cuh"
#include "ops/linear_attention/gated_delta_net/common.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net::chunked {

inline constexpr int BT    = kChunkSize;
inline constexpr int BC    = 16;
inline constexpr int MMA_M = 16;
inline constexpr int MMA_N = 8;
inline constexpr int MMA_K = 8;

static_assert(BT % BC == 0, "BT must be a multiple of BC");
static_assert(BT % MMA_M == 0, "BT must be a multiple of MMA_M");

// Segmented calls (see segment_table). Returns the start column of the full chunk in chunk slot
// `slot`, or -1 when the slot is empty. The chunk in slot b, if any, contains column b*BT+BT-1:
// that column is < N for every slot below floor(N/BT), and it lies in a full chunk of its segment
// exactly when slot b is occupied.
__device__ __forceinline__ std::int32_t segmented_chunk_start(segment_table segments,
                                                              std::int32_t slot) {
    const std::int32_t last   = slot * BT + (BT - 1);
    const std::int32_t s      = segment_containing(segments.columns, segments.count, last);
    const std::int32_t first  = __ldg(segments.columns + 2 * s);
    const std::int32_t length = __ldg(segments.columns + 2 * s + 1);
    const std::int32_t local  = last - first;
    if (local >= length - length % BT) { return -1; }
    return last - local % BT;
}

struct segmented_chunk_run {
    std::int32_t first_column;
    std::int32_t chunks;
    bool has_tail;
};

// Resolves the segment owning run slot `slot`: the segment containing column slot*BT+BT-1 when it
// starts at or after slot*BT and has at least one full chunk.
__device__ __forceinline__ bool segmented_chunk_run_at(segment_table segments, std::int32_t slot,
                                                       segmented_chunk_run& run) {
    const std::int32_t begin = slot * BT;
    const std::int32_t s     = segment_containing(segments.columns, segments.count, begin + BT - 1);
    const std::int32_t first = __ldg(segments.columns + 2 * s);
    const std::int32_t length = __ldg(segments.columns + 2 * s + 1);
    if (first < begin || length < BT) { return false; }
    run = {first, length / BT, (length % BT) != 0};
    return true;
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked
