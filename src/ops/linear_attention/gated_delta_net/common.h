#pragma once

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {

inline constexpr std::int32_t kStateDim  = 128;
inline constexpr std::int32_t kChunkSize = 64;

[[nodiscard]] constexpr bool are_head_counts_valid(std::int64_t qk_heads,
                                                   std::int64_t value_heads) noexcept {
    return qk_heads > 0 && value_heads >= qk_heads && (value_heads % qk_heads) == 0;
}

// Device segment table of a segmented call: I32 [2,count] columns (c_s,T_s) that the caller
// promises tile [0,N) in order. Each segment counts its full kChunkSize-column chunks from its own
// start. A full chunk starting at column p occupies chunk slot floor(p/kChunkSize): distinct full
// chunks are disjoint kChunkSize-column intervals, so slots are unique and there are at most
// floor(N/kChunkSize) of them. A segment with at least one full chunk owns run slot
// floor(c_s/kChunkSize), the chunk slot of its first chunk, for its end-of-chunks state.
struct segment_table {
    const std::int32_t* columns = nullptr;
    std::int32_t count          = 0;
};

// Chunk-end states are needed only by segments that have full chunks and a tail. Such a segment
// starts at or before column N-kChunkSize-1, so its run slot is below floor((N-1)/kChunkSize).
[[nodiscard]] constexpr std::int32_t segmented_tail_state_slots(std::int32_t columns) noexcept {
    return (columns - 1) / kChunkSize;
}

} // namespace ninfer::ops::detail::gated_delta_net
