// ninfer::ops - A4 shared-prefix segmented attention launcher: A2 append at prefix + i, one
// prefix-split attention launch, and one fixed-order merge.
#include "ops/launcher/gqa_attention.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_segmented_bf16.cuh"
#include "ops/kernel/gqa_attention_segmented_i8.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ninfer::ops::detail {
namespace {

using Geometry = Gqa27Geometry;

// Every split kernel holds one CTA per SM (shared memory), so a wave is one CTA per SM.
constexpr std::int32_t kRtx4090SmCount = 128;
// Split capacity targets four waves; the workspace is sized from this budget, not the prefix.
constexpr std::int32_t kSplitCtaBudget = 4 * kRtx4090SmCount;
constexpr std::int32_t kMaximumSplits  = 64;
// Per-CTA fixed cost (Q staging/quantization, pipeline fill, partial store) in key-tile units.
constexpr std::int64_t kSplitOverheadTiles = 2;
constexpr float kLog2E                     = 1.4426950408889634074f;

std::int32_t row_tiles(std::int32_t columns) {
    return div_up<std::int32_t>(columns * Geometry::GroupSize, kGqaSegmentedRows);
}

// Smallest modeled makespan (waves x tiles per CTA) inside the capacity; ties keep fewer splits.
std::int32_t choose_splits(std::int32_t columns, std::int32_t prefix_tiles) {
    const std::int64_t ctas_per_split = static_cast<std::int64_t>(row_tiles(columns)) *
                                        Geometry::KVHeads;
    const std::int32_t limit = std::min(gqa_attention_segmented_split_capacity(columns),
                                        std::max<std::int32_t>(1, prefix_tiles));
    std::int32_t best      = 1;
    std::int64_t best_cost = std::numeric_limits<std::int64_t>::max();
    for (std::int32_t splits = 1; splits <= limit; ++splits) {
        const std::int64_t waves = div_up<std::int64_t>(ctas_per_split * splits, kRtx4090SmCount);
        const std::int64_t cost =
            waves * (div_up<std::int64_t>(prefix_tiles, splits) + kSplitOverheadTiles);
        if (cost < best_cost) {
            best      = splits;
            best_cost = cost;
        }
    }
    return best;
}

template <GqaSegmentedKCodec KCodec, bool PackedV, bool Rotate>
void launch_i8_split(const Tensor& q, const PagedKVBatchLayerView& cache,
                     const GqaSegmentedParams& params, dim3 grid, cudaStream_t stream) {
    constexpr auto kernel = gqa_attention_segmented_i8_kernel<Geometry, KCodec, PackedV, Rotate>;
    constexpr int bytes   = GqaSegmentedI8Layout<KCodec, PackedV>::Bytes;
    static const cudaError_t attribute =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    CUDA_CHECK(attribute);
    kernel<<<grid, kGqaSegmentedI8Threads, bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const std::uint8_t*>(cache.k_pages.data),
        static_cast<const std::uint8_t*>(cache.v_pages.data),
        static_cast<const __half*>(cache.k_scale_pages.data),
        static_cast<const __half*>(cache.v_scale_pages.data), params);
}

} // namespace

std::int32_t gqa_attention_segmented_split_capacity(std::int32_t columns) {
    const std::int32_t ctas_per_split = row_tiles(columns) * Geometry::KVHeads;
    return std::clamp(div_up(kSplitCtaBudget, ctas_per_split), 1, kMaximumSplits);
}

void gqa_attention_segmented_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& segments, const Tensor& table_rows,
                                    std::int32_t prefix, float scale, PagedKVBatchLayerView cache,
                                    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                    Tensor& out, cudaStream_t stream) {
    const std::int32_t columns = q.ne[2];
    gqa_kv_append_at_launch(k, v, prefix, table_rows, cache, stream);

    const std::int32_t prefix_tiles = div_up<std::int32_t>(prefix, kGqaSegmentedKeys);
    const std::int32_t splits       = choose_splits(columns, prefix_tiles);
    const GqaSegmentedParams params{
        .segments      = static_cast<const std::int32_t*>(segments.data),
        .tables        = static_cast<const std::int32_t*>(cache.block_tables.data),
        .table_rows    = static_cast<const std::int32_t*>(table_rows.data),
        .segment_count = segments.ne[1],
        .table_stride  = cache.block_tables.ne[0],
        .prefix        = prefix,
        .columns       = columns,
        .prefix_tiles  = prefix_tiles,
        .splits        = splits,
        .scale_log2    = scale * kLog2E,
        .partial_acc   = static_cast<float*>(partial_acc.data),
        .partial_m     = static_cast<float*>(partial_m.data),
        .partial_l     = static_cast<float*>(partial_l.data),
    };
    const dim3 grid(static_cast<unsigned>(row_tiles(columns)), static_cast<unsigned>(splits),
                    static_cast<unsigned>(Geometry::KVHeads));

    if (cache.dtype == DType::I8) {
        if (cache.e8_root) {
            launch_i8_split<GqaSegmentedKCodec::E8Root, true, true>(q, cache, params, grid, stream);
        } else if (cache.packed_k) {
            launch_i8_split<GqaSegmentedKCodec::Packed4, true, true>(q, cache, params, grid,
                                                                     stream);
        } else if (cache.packed_v) {
            launch_i8_split<GqaSegmentedKCodec::Int8, true, true>(q, cache, params, grid, stream);
        } else {
            launch_i8_split<GqaSegmentedKCodec::Int8, false, false>(q, cache, params, grid,
                                                                    stream);
        }
    } else {
        constexpr auto kernel = gqa_attention_segmented_bf16_kernel<Geometry>;
        static const cudaError_t attribute = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kGqaSegmentedBf16SmemBytes);
        CUDA_CHECK(attribute);
        kernel<<<grid, kGqaSegmentedBf16Threads, kGqaSegmentedBf16SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const __nv_bfloat16*>(cache.k_pages.data),
            static_cast<const __nv_bfloat16*>(cache.v_pages.data), params);
    }
    CUDA_CHECK(cudaGetLastError());

    const dim3 merge_grid(static_cast<unsigned>(columns), static_cast<unsigned>(Geometry::QHeads));
    const auto merge = [&](auto kernel) {
        kernel<<<merge_grid, 128, 0, stream>>>(
            static_cast<const float*>(partial_acc.data), static_cast<const float*>(partial_m.data),
            static_cast<const float*>(partial_l.data), columns, splits,
            static_cast<__nv_bfloat16*>(out.data));
    };
    if (cache.rotate_v) {
        merge(gqa_attention_segmented_merge_kernel<Geometry::QHeads, true>);
    } else {
        merge(gqa_attention_segmented_merge_kernel<Geometry::QHeads, false>);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
