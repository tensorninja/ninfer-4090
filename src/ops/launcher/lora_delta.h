#pragma once

// ninfer::ops::detail - private launch profile and prototype for lora_delta_add.
// Included by the wrapper (host) and defined by the launcher (.cu). The split
// policy lives here because the capacity query and the launch must derive the
// same partial-buffer geometry from the same facts.
// See docs/maintainer/op-development.md §5.2.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/lora.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kLoraTileTokensHost = 8;
inline constexpr std::int32_t kLoraMaximumSplits  = 64;
inline constexpr std::int32_t kLoraTargetBlocks   = 512;

[[nodiscard]] inline std::int32_t lora_token_tiles(std::int32_t tokens) {
    return (tokens + kLoraTileTokensHost - 1) / kLoraTileTokensHost;
}

/// K-splits chosen so small token counts still fill the device. Large token
/// counts already provide tile parallelism and take a single split.
[[nodiscard]] inline std::int32_t lora_k_splits(std::int32_t tokens, std::int32_t site_count) {
    const std::int32_t tiles = lora_token_tiles(tokens);
    const std::int32_t wide  = tiles * (site_count > 0 ? site_count : 1);
    if (wide <= 0) { return 1; }
    std::int32_t splits = kLoraTargetBlocks / wide;
    if (splits < 1) { splits = 1; }
    if (splits > kLoraMaximumSplits) { splits = kLoraMaximumSplits; }
    return splits;
}

[[nodiscard]] inline std::size_t lora_partial_bytes(std::int32_t rank, std::int32_t site_count,
                                                    std::int32_t tokens) {
    const std::int64_t splits = lora_k_splits(tokens, site_count);
    const std::int64_t count =
        splits * site_count * static_cast<std::int64_t>(rank) * tokens;
    return static_cast<std::size_t>(count) * sizeof(float);
}

void lora_delta_add_launch(const Tensor& x, const LoraGroup& group, const Tensor& adapter_index,
                           std::span<Tensor* const> destinations, WorkspaceArena& workspace,
                           cudaStream_t stream);

} // namespace ninfer::ops::detail
