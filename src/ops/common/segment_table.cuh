#pragma once

// Device lookups over a packed segment table: a contiguous I32 [2,S] tensor whose column s is
// (c_s,T_s), the first column and length of segment s. Segmented Ops receive the caller's promise
// that the table tiles [0,N) in order, so the starts c_s are strictly increasing from c_0=0. Every
// lookup is a binary search over the starts; a whole warp or CTA querying one column reads the
// same addresses, which the read-only cache broadcasts.

#include <cstdint>

namespace ninfer::ops {

// Index of the segment containing `column`: the largest s with c_s <= column.
__device__ __forceinline__ std::int32_t
segment_containing(const std::int32_t* __restrict__ segments, std::int32_t count,
                   std::int32_t column) {
    std::int32_t lo = 0;
    std::int32_t hi = count - 1;
    while (lo < hi) {
        const std::int32_t mid = (lo + hi + 1) >> 1;
        if (__ldg(segments + 2 * mid) <= column) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

// Index of the first segment with c_s >= column, or `count` when every segment starts earlier.
__device__ __forceinline__ std::int32_t
first_segment_starting_at(const std::int32_t* __restrict__ segments, std::int32_t count,
                          std::int32_t column) {
    std::int32_t lo = 0;
    std::int32_t hi = count;
    while (lo < hi) {
        const std::int32_t mid = (lo + hi) >> 1;
        if (__ldg(segments + 2 * mid) < column) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

} // namespace ninfer::ops
