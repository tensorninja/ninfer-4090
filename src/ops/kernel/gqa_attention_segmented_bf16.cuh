#pragma once

// A4 BF16 kernels. One CTA runs FlashAttention-2 over a run of 64-key tiles for 64 GQA-packed
// rows of one KV head: 4 warps own 16 rows each, Q/K/V are staged in 96 KiB of swizzled shared
// memory, and m16n8k16 BF16 MMA with FP32 accumulation serves both QK and PV while P stays in
// registers. The next K tile loads under PV and the current V tile loads under QK.
//
// gqa_attention_segmented_bf16_kernel is the prefix split kernel of a BF16 cache: it walks one
// share of the cached prefix pages and writes unnormalized FP32 partials.
// gqa_attention_segmented_own_kernel serves every cache: it walks the call's own columns from the
// BF16 k/v inputs under each row's segment window, then folds the prefix partials of either split
// kernel in fixed split order, leaves the rotated V domain of the prefix, adds its own columns,
// and writes the normalized BF16 output.

#include "ops/kernel/gqa_attention_prefill_common.cuh"
#include "ops/kernel/gqa_attention_segmented.cuh"

#include <math_constants.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops {

inline constexpr int kGqaSegmentedBf16Threads = 128;
inline constexpr int kGqaSegmentedBf16SmemBytes =
    3 * kGqaSegmentedRows * kGqaSegmentedHeadDim * static_cast<int>(sizeof(__nv_bfloat16));

static_assert(kGqaSegmentedBf16SmemBytes == 98304);

// Stage one [64, 256] BF16 page tile into the swizzled layout. Keys at or past key_limit belong to
// no cached row, so they are zero-filled instead of read.
template <typename Geometry>
__device__ __forceinline__ void gqa_segmented_stage_bf16(__nv_bfloat16* dst,
                                                         const __nv_bfloat16* cache, int kv_head,
                                                         int physical_page, int k0, int key_limit,
                                                         int tid) {
    constexpr int D         = kGqaSegmentedHeadDim;
    constexpr int VecPerRow = D / 8;
    const __nv_bfloat16* block =
        cache + paged_kv_page_head_offset<D, Geometry::KVHeads>(physical_page, kv_head);
    const bool full = k0 + kGqaSegmentedKeys <= key_limit;
#pragma unroll 4
    for (int chunk = tid; chunk < kGqaSegmentedKeys * VecPerRow;
         chunk += kGqaSegmentedBf16Threads) {
        const int key_l  = chunk / VecPerRow;
        const int d      = (chunk % VecPerRow) * 8;
        __nv_bfloat16* p = &dst[key_l * D + gqa_prefill_swz(key_l, d)];
        if (full || k0 + key_l < key_limit) {
            cp_async<16, Cache::cg>(p, &block[key_l * D + d]);
        } else {
            store_vec(p, make_int4(0, 0, 0, 0));
        }
    }
}

// Stage the own columns [c0, c0 + 64) of one KV head from contiguous BF16 [256, KVHeads, N] into
// the swizzled layout; columns past the call are zero-filled.
template <typename Geometry>
__device__ __forceinline__ void gqa_segmented_stage_own_bf16(__nv_bfloat16* dst,
                                                             const __nv_bfloat16* source,
                                                             int kv_head, int c0, int columns,
                                                             int tid) {
    constexpr int D         = kGqaSegmentedHeadDim;
    constexpr int VecPerRow = D / 8;
#pragma unroll 4
    for (int chunk = tid; chunk < kGqaSegmentedKeys * VecPerRow;
         chunk += kGqaSegmentedBf16Threads) {
        const int key_l  = chunk / VecPerRow;
        const int d      = (chunk % VecPerRow) * 8;
        const int column = c0 + key_l;
        __nv_bfloat16* p = &dst[key_l * D + gqa_prefill_swz(key_l, d)];
        if (column < columns) {
            cp_async<16, Cache::cg>(
                p, &source[(static_cast<std::int64_t>(column) * Geometry::KVHeads + kv_head) * D + d]);
        } else {
            store_vec(p, make_int4(0, 0, 0, 0));
        }
    }
}

// Merge-phase layout of the own kernel inside the dynamic shared memory once Q/K/V are dead:
// own FP32 accumulators with a padded row stride (conflict-free fragment stores), own log2-domain
// maxima and sums, and per warp the split weights of the (at most sixteen) rows it folds (a padded
// stride keeps the per-row lanes of the statistics fold on distinct banks) followed by their own
// weights and normalizers.
inline constexpr int kGqaSegmentedOwnAccStride  = kGqaSegmentedHeadDim + 8;
inline constexpr int kGqaSegmentedMaxSplits     = 64;
inline constexpr int kGqaSegmentedWeightStride  = kGqaSegmentedMaxSplits + 2;
inline constexpr int kGqaSegmentedWarpMergeSize = 16 * kGqaSegmentedWeightStride + 2 * 16;
// Rows whose split accumulations run interleaved in one pass of the merge.
inline constexpr int kGqaSegmentedMergeRows = 4;
static_assert(static_cast<int>(sizeof(float)) *
                  (kGqaSegmentedRows * kGqaSegmentedOwnAccStride + 2 * kGqaSegmentedRows +
                   (kGqaSegmentedBf16Threads / 32) * kGqaSegmentedWarpMergeSize) <=
              kGqaSegmentedBf16SmemBytes);

template <typename Geometry, bool Own, bool RotateV>
__device__ __forceinline__ void gqa_segmented_bf16_attention(const __nv_bfloat16* __restrict__ q,
                                                             const __nv_bfloat16* __restrict__ keys,
                                                             const __nv_bfloat16* __restrict__ values,
                                                             const GqaSegmentedParams& params,
                                                             __nv_bfloat16* __restrict__ out) {
    static_assert(Own || !RotateV);
    constexpr int D             = kGqaSegmentedHeadDim;
    constexpr int Br            = kGqaSegmentedRows;
    constexpr int Bc            = kGqaSegmentedKeys;
    constexpr int Threads       = kGqaSegmentedBf16Threads;
    constexpr int Group         = Geometry::GroupSize;
    constexpr int QKNt          = Bc / 8;
    constexpr int QKKs          = D / 16;
    constexpr int PVNt          = D / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr unsigned FullMask = 0xffffffffu;

    extern __shared__ __align__(16) __nv_bfloat16 gqa_segmented_bf16_smem[];
    __nv_bfloat16* q_s = gqa_segmented_bf16_smem;
    __nv_bfloat16* k_s = q_s + Br * D;
    __nv_bfloat16* v_s = k_s + Bc * D;

    const int row_tile = static_cast<int>(blockIdx.x);
    const int kv_head  = static_cast<int>(blockIdx.z);
    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int gid      = lane >> 2;
    const int lid      = lane & 3;

    const int first_row  = row_tile * Br;
    const int tile_rows  = min(Br, Group * params.columns - first_row);
    const int prefix     = params.prefix;
    const float scale_l2 = params.scale_log2;
    // An own fold share that starts past the tile's last row has nothing to fold.
    if (Own && static_cast<int>(blockIdx.y) * (Br / static_cast<int>(gridDim.y)) >= tile_rows) {
        return;
    }

    // Own tiles are column tiles of the call from the first row's segment start through the last
    // row's column; prefix tiles are one split's share of the cached pages.
    int tile_begin = 0;
    int tiles      = 0;
    const std::int32_t* block_table = nullptr;
    if constexpr (Own) {
        const int first_column = first_row / Group;
        const int last_column  = (first_row + tile_rows - 1) / Group;
        tile_begin =
            gqa_segmented_segment_start(params.segments, params.segment_count, first_column) / Bc;
        tiles = last_column / Bc + 1 - tile_begin;
    } else {
        const GqaSegmentedPrefixShare share =
            gqa_segmented_prefix_share(params, static_cast<int>(blockIdx.y));
        tile_begin  = share.begin;
        tiles       = share.end - share.begin;
        block_table = params.tables +
                      static_cast<std::int64_t>(params.table_rows[0]) * params.table_stride;
    }
    const auto stage = [&](__nv_bfloat16* dst, const __nv_bfloat16* source, int tile) {
        if constexpr (Own) {
            gqa_segmented_stage_own_bf16<Geometry>(dst, source, kv_head, tile * Bc, params.columns,
                                                   tid);
        } else {
            gqa_segmented_stage_bf16<Geometry>(dst, source, kv_head, block_table[tile], tile * Bc,
                                               prefix, tid);
        }
    };

    // Q rows are gathered per (column, local head); padding rows are zero.
#pragma unroll 4
    for (int chunk = tid; chunk < Br * (D / 8); chunk += Threads) {
        const int row    = chunk / (D / 8);
        const int d      = (chunk % (D / 8)) * 8;
        __nv_bfloat16* p = &q_s[row * D + gqa_prefill_swz(row, d)];
        if (row < tile_rows) {
            const int packed = first_row + row;
            const int column = packed / Group;
            const int q_head = kv_head * Group + (packed - column * Group);
            cp_async<16, Cache::cg>(p, &q[gqa_prefill_q_index<Geometry>(q_head, d, column)]);
        } else {
            store_vec(p, make_int4(0, 0, 0, 0));
        }
    }
    cp_commit();
    stage(k_s, keys, tile_begin);
    cp_commit();

    const int warp_row0 = warp * 16;
    GqaSegmentedRowRange range0{kGqaSegmentedEmptyLo, -1};
    GqaSegmentedRowRange range1{kGqaSegmentedEmptyLo, -1};
    if constexpr (Own) {
        range0 = gqa_segmented_row_range<Group>(params, first_row + warp_row0 + gid);
        range1 = gqa_segmented_row_range<Group>(params, first_row + warp_row0 + gid + 8);
    }

    const int a_mat = lane >> 3;
    const int a_rin = lane & 7;
    const int b_rin = lane & 7;
    const int b_koff = ((lane >> 3) & 1) << 3;
    const unsigned q_lane_base =
        smem_addr(q_s) + static_cast<unsigned>((warp_row0 + a_rin + ((a_mat & 1) << 3)) * 512);
    const unsigned q_as = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned q_r  = static_cast<unsigned>(a_rin << 4);
    const unsigned k_lane_base =
        smem_addr(k_s) + static_cast<unsigned>(b_rin * 512) + (static_cast<unsigned>(lane >> 4) << 12);
    const unsigned k_as = static_cast<unsigned>((b_koff >> 3) << 4);
    const unsigned k_r  = static_cast<unsigned>(b_rin << 4);
    const unsigned v_lane_base = smem_addr(v_s) +
                                 static_cast<unsigned>(((lane >> 3) & 1) * 4096) +
                                 static_cast<unsigned>(b_rin * 512);
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r  = static_cast<unsigned>(b_rin << 4);

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F;
    float m1 = -CUDART_INF_F;
    float l0 = 0.0f;
    float l1 = 0.0f;

    auto process_tile = [&](int index, int tile, auto masked_tag) {
        constexpr bool Masked = decltype(masked_tag)::value;
        const int k0          = tile * Bc;
        cp_wait<0>(); // K(tile) landed; every warp finished the previous PV.
        __syncthreads();
        stage(v_s, values, tile);
        cp_commit();

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
        unsigned af[2][4];
        unsigned bf[2][QKNt][2];
        ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                    gqa_prefill_swz_addr(q_lane_base, 0u, q_as, q_r));
#pragma unroll
        for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
            ldmatrix_x4(bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                        gqa_prefill_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 4096), 0u,
                                             k_as, k_r));
        }
#pragma unroll
        for (int k = 0; k < QKKs; ++k) {
            const int cur = k & 1;
            const int nxt = cur ^ 1;
            if (k + 1 < QKKs) {
                const unsigned ck = static_cast<unsigned>((k + 1) << 5);
                ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                            gqa_prefill_swz_addr(q_lane_base, ck, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                    ldmatrix_x4(bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0],
                                bf[nxt][nt2 + 1][1],
                                gqa_prefill_swz_addr(
                                    k_lane_base + static_cast<unsigned>(nt2 * 4096), ck, k_as,
                                    k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                         af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
            }
        }

        if constexpr (Masked) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                if constexpr (Own) {
                    if (!gqa_segmented_in_range(key0, range0)) { score[nt][0] = -CUDART_INF_F; }
                    if (!gqa_segmented_in_range(key1, range0)) { score[nt][1] = -CUDART_INF_F; }
                    if (!gqa_segmented_in_range(key0, range1)) { score[nt][2] = -CUDART_INF_F; }
                    if (!gqa_segmented_in_range(key1, range1)) { score[nt][3] = -CUDART_INF_F; }
                } else {
                    if (key0 >= prefix) { score[nt][0] = score[nt][2] = -CUDART_INF_F; }
                    if (key1 >= prefix) { score[nt][1] = score[nt][3] = -CUDART_INF_F; }
                }
            }
        }
        float bm0 = -CUDART_INF_F;
        float bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0 = fmaxf(m0, bm0);
        const float nm1 = fmaxf(m1, bm1);
        // A masked row may still have no visible key; a finite exponent base sends every
        // -inf score and the -inf running max through ex2 to exact zero.
        const float nm0_scaled = (Masked && nm0 == -CUDART_INF_F) ? 0.0f : nm0 * scale_l2;
        const float nm1_scaled = (Masked && nm1 == -CUDART_INF_F) ? 0.0f : nm1 * scale_l2;
        const float alpha0     = exp2_approx(__fmaf_rn(m0, scale_l2, -nm0_scaled));
        const float alpha1     = exp2_approx(__fmaf_rn(m1, scale_l2, -nm1_scaled));

        float bl0 = 0.0f;
        float bl1 = 0.0f;
        unsigned p_frag[PVKs][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const float p00 = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled));
            const float p01 = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled));
            const float p10 = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled));
            const float p11 = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled));
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            const int pk = nt >> 1;
            if ((nt & 1) == 0) {
                p_frag[pk][0] = pack_bf16x2(p00, p01);
                p_frag[pk][1] = pack_bf16x2(p10, p11);
            } else {
                p_frag[pk][2] = pack_bf16x2(p00, p01);
                p_frag[pk][3] = pack_bf16x2(p10, p11);
            }
        }
        l0 = __fmaf_rn(l0, alpha0, bl0);
        l1 = __fmaf_rn(l1, alpha1, bl1);
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        cp_wait<0>(); // V(tile) landed; every warp finished reading K(tile).
        __syncthreads();
        if (index + 1 < tiles) {
            stage(k_s, keys, tile + 1);
            cp_commit();
        }

        constexpr int PVHalf  = PVNt / 2;
        constexpr int PVLoads = PVKs * PVHalf;
        unsigned vf[2][4];
        ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                      gqa_prefill_swz_addr(v_lane_base, 0u, v_as, v_r));
#pragma unroll
        for (int li = 0; li < PVLoads; ++li) {
            const int k   = li / PVHalf;
            const int n2  = (li % PVHalf) * 2;
            const int cur = li & 1;
            const int nxt = cur ^ 1;
            if (li + 1 < PVLoads) {
                const int k2       = (li + 1) / PVHalf;
                const int n2b      = ((li + 1) % PVHalf) * 2;
                const unsigned ckv = static_cast<unsigned>(n2b << 4);
                ldmatrix_x4_t(vf[nxt][0], vf[nxt][1], vf[nxt][2], vf[nxt][3],
                              gqa_prefill_swz_addr(v_lane_base + static_cast<unsigned>(k2 * 8192),
                                                   ckv, v_as, v_r));
            }
            mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_frag[k][0], p_frag[k][1],
                     p_frag[k][2], p_frag[k][3], vf[cur][0], vf[cur][1]);
            mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3], p_frag[k][0],
                     p_frag[k][1], p_frag[k][2], p_frag[k][3], vf[cur][2], vf[cur][3]);
        }
    };

    for (int index = 0; index < tiles; ++index) {
        const int tile = tile_begin + index;
        if (!Own && (tile + 1) * Bc <= prefix) {
            process_tile(index, tile, std::false_type{});
        } else {
            process_tile(index, tile, std::true_type{});
        }
    }

    l0 = warp_sum<4>(l0, FullMask);
    l1 = warp_sum<4>(l1, FullMask);
    if constexpr (!Own) {
        const int split = static_cast<int>(blockIdx.y);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = warp_row0 + gid + 8 * half;
            if (row >= tile_rows) { continue; }
            const int packed        = first_row + row;
            const int column        = packed / Group;
            const int q_head        = kv_head * Group + (packed - column * Group);
            const std::int64_t stat = gqa_segmented_stat_index(Geometry::QHeads, q_head, column,
                                                               split, params.columns);
            if (lid == 0) {
                params.partial_m[stat] = (half == 0 ? m0 : m1) * scale_l2;
                params.partial_l[stat] = half == 0 ? l0 : l1;
            }
            float* acc_row = params.partial_acc + kGqaSegmentedHeadDim * stat;
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
                store_vec(&acc_row[n * 8 + 2 * lid],
                          make_float2(acc[n][2 * half], acc[n][2 * half + 1]));
            }
        }
        return;
    } else {
        // Q/K/V are dead once every warp has left its last PV; the arena becomes the merge buffer.
        __syncthreads();
        float* own_acc     = reinterpret_cast<float*>(gqa_segmented_bf16_smem);
        float* own_m       = own_acc + Br * kGqaSegmentedOwnAccStride;
        float* own_l       = own_m + Br;
        float* weights     = own_l + Br + warp * kGqaSegmentedWarpMergeSize;
        float* own_weights = weights + 16 * kGqaSegmentedWeightStride;
        float* inverses    = own_weights + 16;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = warp_row0 + gid + 8 * half;
            float* dst    = own_acc + row * kGqaSegmentedOwnAccStride;
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
                store_vec(&dst[n * 8 + 2 * lid], make_float2(acc[n][2 * half], acc[n][2 * half + 1]));
            }
            if (lid == 0) {
                own_m[row] = (half == 0 ? m0 : m1) * scale_l2;
                own_l[row] = half == 0 ? l0 : l1;
            }
        }
        __syncthreads();

        // The gridDim.y CTAs of a row tile (one, two, or four) repeat its own attention and fold
        // one contiguous share of its rows, split evenly over their warps, so a small grid spreads
        // the partial reads over more SMs. Every valid row sees its own column, so its own maximum
        // is finite; every prefix split holds at least one visible key.
        const int shares    = static_cast<int>(gridDim.y);
        const int warp_rows = 16 / shares;
        const int row0      = static_cast<int>(blockIdx.y) * (Br / shares) + warp * warp_rows;
        const int rows      = min(warp_rows, tile_rows - row0);
        if (rows <= 0) { return; }
        const int splits                = params.splits;
        const std::int64_t split_stride = static_cast<std::int64_t>(Geometry::QHeads) * params.columns;
        const auto row_stat             = [&](int r) {
            const int packed = first_row + row0 + r;
            const int column = packed / Group;
            const int q_head = kv_head * Group + (packed - column * Group);
            return gqa_segmented_stat_index(Geometry::QHeads, q_head, column, 0, params.columns);
        };

        // Statistics of all the warp's rows at once: lanes 2r and 2r + 1 share the split maxima
        // and weights of row r, and lane 2r then accumulates its denominator in split order.
        {
            const int r              = lane >> 1;
            const int parity         = lane & 1;
            const bool valid         = r < rows;
            const std::int64_t stat0 = valid ? row_stat(r) : 0;
            float* row_weights       = weights + r * kGqaSegmentedWeightStride;
            float maximum            = -CUDART_INF_F;
            if (valid) {
                maximum = own_m[row0 + r];
#pragma unroll 4
                for (int split = parity; split < splits; split += 2) {
                    maximum = fmaxf(maximum, params.partial_m[stat0 + split * split_stride]);
                }
            }
            maximum = fmaxf(maximum, __shfl_xor_sync(FullMask, maximum, 1));
            if (valid) {
#pragma unroll 4
                for (int split = parity; split < splits; split += 2) {
                    row_weights[split] =
                        exp2f(params.partial_m[stat0 + split * split_stride] - maximum);
                }
            }
            __syncwarp();
            if (valid && parity == 0) {
                const float own_weight = exp2f(own_m[row0 + r] - maximum);
                float sum              = 0.0f;
#pragma unroll 8
                for (int split = 0; split < splits; ++split) {
                    sum = fmaf(row_weights[split], params.partial_l[stat0 + split * split_stride],
                               sum);
                }
                own_weights[r] = own_weight;
                inverses[r]    = 1.0f / fmaf(own_weight, own_l[row0 + r], sum);
            }
            __syncwarp();
        }

        // Split accumulation of kGqaSegmentedMergeRows rows per pass keeps their partial loads in
        // flight together; a lane holds dimensions lane and lane + 32 of every quantization group.
        // A pass past the last row repeats that row and stores nothing for it.
        constexpr int MergeRows = kGqaSegmentedMergeRows;
        for (int r0 = 0; r0 < rows; r0 += MergeRows) {
            std::int64_t stats[MergeRows];
            const float* row_weights[MergeRows];
#pragma unroll
            for (int i = 0; i < MergeRows; ++i) {
                const int r    = min(r0 + i, rows - 1);
                stats[i]       = row_stat(r);
                row_weights[i] = weights + r * kGqaSegmentedWeightStride;
            }
            float a[MergeRows][2 * kGqaKvQuantGroups] = {};
#pragma unroll 2
            for (int split = 0; split < splits; ++split) {
#pragma unroll
                for (int i = 0; i < MergeRows; ++i) {
                    const float* partial = params.partial_acc +
                                           kGqaSegmentedHeadDim * (stats[i] + split * split_stride);
                    const float weight = row_weights[i][split];
#pragma unroll
                    for (int group = 0; group < kGqaKvQuantGroups; ++group) {
                        const int d0        = group * kGqaKvQuantGroup + lane;
                        a[i][2 * group]     = fmaf(weight, partial[d0], a[i][2 * group]);
                        a[i][2 * group + 1] = fmaf(weight, partial[d0 + 32], a[i][2 * group + 1]);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < MergeRows; ++i) {
                const int r = r0 + i;
                if (r >= rows) { break; }
                const float own_weight = own_weights[r];
                const float inverse    = inverses[r];
                const float* own       = own_acc + (row0 + r) * kGqaSegmentedOwnAccStride;
                __nv_bfloat16* dst     = out + kGqaSegmentedHeadDim * stats[i];
#pragma unroll
                for (int group = 0; group < kGqaKvQuantGroups; ++group) {
                    const int d0 = group * kGqaKvQuantGroup + lane;
                    const int d1 = d0 + 32;
                    float a0     = a[i][2 * group];
                    float a1     = a[i][2 * group + 1];
                    // The prefix partials of a rotated-V cache accumulate H * v.
                    if constexpr (RotateV) { gqa_kv_hadamard64(a0, a1, FullMask); }
                    dst[d0] = __float2bfloat16(fmaf(own_weight, own[d0], a0) * inverse);
                    dst[d1] = __float2bfloat16(fmaf(own_weight, own[d1], a1) * inverse);
                }
            }
        }
    }
}

template <typename Geometry>
__launch_bounds__(kGqaSegmentedBf16Threads, 1) __global__
    void gqa_attention_segmented_bf16_kernel(const __nv_bfloat16* __restrict__ q,
                                             const __nv_bfloat16* __restrict__ cache_k,
                                             const __nv_bfloat16* __restrict__ cache_v,
                                             GqaSegmentedParams params) {
    gqa_segmented_bf16_attention<Geometry, false, false>(q, cache_k, cache_v, params, nullptr);
}

template <typename Geometry, bool RotateV>
__launch_bounds__(kGqaSegmentedBf16Threads, 1) __global__
    void gqa_attention_segmented_own_kernel(const __nv_bfloat16* __restrict__ q,
                                            const __nv_bfloat16* __restrict__ k,
                                            const __nv_bfloat16* __restrict__ v,
                                            GqaSegmentedParams params,
                                            __nv_bfloat16* __restrict__ out) {
    gqa_segmented_bf16_attention<Geometry, true, RotateV>(q, k, v, params, out);
}

} // namespace ninfer::ops
