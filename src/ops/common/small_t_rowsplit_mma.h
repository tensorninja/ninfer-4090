#pragma once

// Host-visible facts of the Q4/Q5 RowSplit small-T tensor-core mechanism
// (small_t_rowsplit_mma.cuh): tile geometry, the split-K decomposition and the
// split-K workspace. Plans size workspaces and choose splits from this header
// without compiling device code.

#include "core/dtype.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#    define NINFER_SMALL_T_MMA_HD __host__ __device__
#else
#    define NINFER_SMALL_T_MMA_HD
#endif

namespace ninfer::ops::detail {

enum class SmallTMmaEpilogue : std::uint8_t {
    // rows < split_row -> out0[col * ld0 + row]; otherwise out1[col * ld1 + row - split_row].
    Store,
    // out0[col * ld0 + row] += projection, added in FP32 and rounded once.
    Residual,
    // The weight holds gate rows [0, n/2) and up rows [n/2, n). A CTA takes 32 gate rows and
    // their 32 up rows and stores silu(gate) * up for its 32 output rows into out0 (ld0).
    SwiGlu,
};

enum class SmallTMmaCodec : std::uint8_t { Q4, Q5, Mixed };

struct SmallTMmaShape {
    static constexpr int kThreads     = 256;
    static constexpr int kWarps       = 8;
    static constexpr int kRowGroups   = 4;
    static constexpr int kPhases      = 2;
    static constexpr int kRows        = 64;
    static constexpr int kGroupK      = 64;
    static constexpr int kMaxCols     = 32;
    static constexpr int kCodeBytes   = kPhases * kRows * 32; // 4096
    static constexpr int kHighBytes   = kPhases * kRows * 8;  // 1024
    static constexpr int kScaleBytes  = kRows * kPhases * 2;  // 256
    static constexpr int kCodeOffset  = 0;
    static constexpr int kHighOffset  = kCodeOffset + kCodeBytes;
    static constexpr int kScaleOffset = kHighOffset + kHighBytes;
    static constexpr int kActOffset   = kScaleOffset + kScaleBytes; // 5376

    NINFER_SMALL_T_MMA_HD static constexpr int stage_bytes(int tile_cols) noexcept {
        return kActOffset + kPhases * tile_cols * kGroupK * 2;
    }
    // Pipeline depth and residency are chosen per Op: long split-K ranges saturate the
    // weight stream with three stages at two CTAs per SM, short tiles want a fourth stage
    // and, at eight columns, a third resident CTA.
    NINFER_SMALL_T_MMA_HD static constexpr int smem_bytes(int tile_cols, int stages) noexcept {
        return stages * stage_bytes(tile_cols);
    }
};

// Tile columns for a token count: the smallest registered n-tile multiple that holds it.
constexpr int small_t_mma_tile_cols(std::int32_t cols) noexcept {
    return cols <= 8 ? 8 : (cols <= 16 ? 16 : (cols <= 24 ? 24 : 32));
}

// Split-K decomposition of k / 64 groups into at most `splits` even-sized group ranges. The
// last split takes the remainder; a split count that would leave it empty is reduced.
struct SmallTMmaSplitK {
    std::int32_t splits;
    std::int32_t groups_per_split;
};

constexpr SmallTMmaSplitK small_t_mma_split_k(std::int32_t k, std::int32_t splits) noexcept {
    // Split boundaries fall on 256-byte code-plane blocks (8 groups) so every split's row
    // runs start on the L2 prefetch block the copies request.
    constexpr std::int32_t kAlignGroups = 8;
    const std::int32_t groups           = k / SmallTMmaShape::kGroupK;
    const std::int32_t blocks           = (groups + kAlignGroups - 1) / kAlignGroups;
    const std::int32_t blocks_per_split = (blocks + splits - 1) / splits;
    const std::int32_t used             = (blocks + blocks_per_split - 1) / blocks_per_split;
    return {used, blocks_per_split * kAlignGroups};
}

// FP32 split partials plus one arrival counter per tile. The counters must be zero when the
// kernel starts; the launcher clears them on the stream before every split-K launch.
struct SmallTMmaWorkspace {
    float* partials = nullptr;
    int* counters   = nullptr;
};

template <class Allocator>
SmallTMmaWorkspace allocate_small_t_mma_workspace(Allocator& allocator, int tile_cols,
                                                  std::int32_t tiles, std::int32_t splits) {
    if (splits <= 1) { return {}; }
    Tensor partials = allocator.alloc(DType::FP32, {splits * tiles, SmallTMmaShape::kRows,
                                                    tile_cols});
    Tensor counters = allocator.alloc(DType::I32, {tiles});
    return {static_cast<float*>(partials.data), static_cast<int*>(counters.data)};
}

} // namespace ninfer::ops::detail

#undef NINFER_SMALL_T_MMA_HD
