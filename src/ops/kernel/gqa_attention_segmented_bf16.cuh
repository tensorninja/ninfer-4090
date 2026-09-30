#pragma once

// A4 BF16 split kernel. One CTA runs FlashAttention-2 over one key split for 64 GQA-packed rows
// of one KV head: 4 warps own 16 rows each, Q/K/V are staged in 96 KiB of swizzled shared
// memory, and m16n8k16 BF16 MMA with FP32 accumulation serves both QK and PV while P stays in
// registers. The next K tile loads under PV and the current V tile loads under QK. The CTA writes
// unnormalized FP32 partials; gqa_attention_segmented_merge_kernel folds the splits.

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

// Stage one [64, 256] BF16 page tile into the swizzled layout. Keys at or past key_limit have
// never been written, so they are zero-filled instead of read.
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

template <typename Geometry>
__launch_bounds__(kGqaSegmentedBf16Threads, 1) __global__
    void gqa_attention_segmented_bf16_kernel(const __nv_bfloat16* __restrict__ q,
                                             const __nv_bfloat16* __restrict__ cache_k,
                                             const __nv_bfloat16* __restrict__ cache_v,
                                             GqaSegmentedParams params) {
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
    const int split    = static_cast<int>(blockIdx.y);
    const int kv_head  = static_cast<int>(blockIdx.z);
    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int gid      = lane >> 2;
    const int lid      = lane & 3;

    const int first_row  = row_tile * Br;
    const int tile_rows  = min(Br, Group * params.columns - first_row);
    const bool last      = split == params.splits - 1;
    const int prefix     = params.prefix;
    const int key_limit  = prefix + params.columns;
    const float scale_l2 = params.scale_log2;
    const std::int32_t* block_table =
        params.tables + static_cast<std::int64_t>(params.table_rows[0]) * params.table_stride;
    const GqaSegmentedTilePlan plan =
        gqa_segmented_tile_plan<Group>(params, split, first_row, first_row + tile_rows - 1);

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
    {
        const int tile = plan.tile(0);
        gqa_segmented_stage_bf16<Geometry>(k_s, cache_k, kv_head, block_table[tile], tile * Bc,
                                           key_limit, tid);
        cp_commit();
    }

    const int warp_row0 = warp * 16;
    const GqaSegmentedRowRange range0 =
        gqa_segmented_row_range<Group>(params, first_row + warp_row0 + gid, last);
    const GqaSegmentedRowRange range1 =
        gqa_segmented_row_range<Group>(params, first_row + warp_row0 + gid + 8, last);

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

    const int tiles = plan.count();
    auto process_tile = [&](int index, int tile, auto masked_tag) {
        constexpr bool Masked = decltype(masked_tag)::value;
        const int k0          = tile * Bc;
        cp_wait<0>(); // K(tile) landed; every warp finished the previous PV.
        __syncthreads();
        gqa_segmented_stage_bf16<Geometry>(v_s, cache_v, kv_head, block_table[tile], k0,
                                           key_limit, tid);
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
                if (!gqa_segmented_visible(key0, prefix, range0)) { score[nt][0] = -CUDART_INF_F; }
                if (!gqa_segmented_visible(key1, prefix, range0)) { score[nt][1] = -CUDART_INF_F; }
                if (!gqa_segmented_visible(key0, prefix, range1)) { score[nt][2] = -CUDART_INF_F; }
                if (!gqa_segmented_visible(key1, prefix, range1)) { score[nt][3] = -CUDART_INF_F; }
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
            const int next = plan.tile(index + 1);
            gqa_segmented_stage_bf16<Geometry>(k_s, cache_k, kv_head, block_table[next],
                                               next * Bc, key_limit, tid);
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
        const int tile = plan.tile(index);
        if ((tile + 1) * Bc <= prefix) {
            process_tile(index, tile, std::false_type{});
        } else {
            process_tile(index, tile, std::true_type{});
        }
    }

    l0 = warp_sum<4>(l0, FullMask);
    l1 = warp_sum<4>(l1, FullMask);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int row = warp_row0 + gid + 8 * half;
        if (row >= tile_rows) { continue; }
        const int packed         = first_row + row;
        const int column         = packed / Group;
        const int q_head         = kv_head * Group + (packed - column * Group);
        const std::int64_t stat  = gqa_segmented_stat_index(Geometry::QHeads, q_head, column,
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
}

} // namespace ninfer::ops
