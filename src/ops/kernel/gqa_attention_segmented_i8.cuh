#pragma once

// A4 INT8-family prefix split kernel for the INT8-G64, rk8v4, rk4v4, and rk2v4-e8 caches.
// One CTA owns 64 GQA-packed rows of one KV head and one prefix split. QK runs the registered
// Q8-G64 x INT8-K profile on m16n8k32.s8 Tensor Cores; V is dequantized to FP16 and PV runs
// FP16 m16n8k16 with a per-tile FP16 accumulator folded into FP32 (the sm_89 prompt schedule).
//
// Sixteen warps: eight paired producers (4 row tiles x 2 key halves) compute QK and the online
// softmax while eight workers dequantize V; all sixteen then split PV by 64-dimension slice.
// Every cache code reaches shared memory through cp.async. Packed and E8-root K codes stream
// one tile ahead under QK and are expanded to INT8 under PV; V codes and all scales stream
// under PV. The CTA writes unnormalized FP32 partials for gqa_attention_segmented_own_kernel.

#include "ops/kernel/e8_root_codec.cuh"
#include "ops/kernel/gqa_attention_prefill_i8.cuh"
#include "ops/kernel/gqa_attention_segmented.cuh"

#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops {

enum class GqaSegmentedKCodec { Int8, Packed4, E8Root };

template <GqaSegmentedKCodec KCodec, bool PackedV>
struct GqaSegmentedI8Layout {
    static constexpr int KRowBytes = KCodec == GqaSegmentedKCodec::Packed4  ? 128
                                     : KCodec == GqaSegmentedKCodec::E8Root ? 64
                                                                            : 256;
    static constexpr int VRowBytes = PackedV ? 128 : 256;
    static constexpr bool RawK     = KCodec != GqaSegmentedKCodec::Int8;

    static constexpr int QBytes      = kGqaSegmentedRows * kGqaSegmentedHeadDim;
    static constexpr int QScaleBytes = kGqaSegmentedRows * kGqaKvQuantGroups * 4;
    static constexpr int KBytes      = kGqaSegmentedKeys * kGqaSegmentedHeadDim;
    static constexpr int KRawBytes   = RawK ? kGqaSegmentedKeys * KRowBytes : 0;
    static constexpr int VRawBytes   = kGqaSegmentedKeys * VRowBytes;
    static constexpr int VHalfBytes  = kGqaSegmentedKeys * kGqaSegmentedHeadDim * 2;
    static constexpr int PBytes      = kGqaSegmentedRows * kGqaSegmentedKeys * 2;
    static constexpr int ScaleBytes  = kGqaSegmentedKeys * kGqaKvQuantGroups * 2;
    static constexpr int StatBytes   = 5 * kGqaSegmentedRows * 4;

    static constexpr int QScale = QBytes;
    static constexpr int K      = QScale + QScaleBytes;
    static constexpr int KRaw   = K + KBytes;
    static constexpr int VRaw   = KRaw + KRawBytes;
    static constexpr int VHalf  = VRaw + VRawBytes;
    static constexpr int P      = VHalf + VHalfBytes;
    static constexpr int KScale = P + PBytes;
    static constexpr int VScale = KScale + ScaleBytes;
    static constexpr int Stats  = VScale + ScaleBytes;
    static constexpr int Bytes  = Stats + StatBytes;
};

inline constexpr int kGqaSegmentedI8Warps   = 16;
inline constexpr int kGqaSegmentedI8Threads = kGqaSegmentedI8Warps * 32;

template <typename Geometry, GqaSegmentedKCodec KCodec, bool PackedV, bool RotateK>
__global__ __maxnreg__(128) void gqa_attention_segmented_i8_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const __half* __restrict__ cache_k_scale,
    const __half* __restrict__ cache_v_scale, GqaSegmentedParams params) {
    using Layout                = GqaSegmentedI8Layout<KCodec, PackedV>;
    constexpr bool RawK         = Layout::RawK;
    constexpr int D             = kGqaSegmentedHeadDim;
    constexpr int Br            = kGqaSegmentedRows;
    constexpr int Bc            = kGqaSegmentedKeys;
    constexpr int DB16          = D / 2;
    constexpr int Groups        = kGqaKvQuantGroups;
    constexpr int GroupKc       = kGqaKvQuantGroup / 32;
    constexpr int Warps         = kGqaSegmentedI8Warps;
    constexpr int Threads       = kGqaSegmentedI8Threads;
    constexpr int RowTiles      = Br / 16;
    constexpr int ColSplit      = 2;
    constexpr int ProducerWarps = RowTiles * ColSplit;
    constexpr int WorkerThreads = (Warps - ProducerWarps) * 32;
    constexpr int QKNtL         = Bc / 8 / ColSplit;
    constexpr int DSlices       = Warps / RowTiles;
    constexpr int PVNtPerWarp   = D / (DSlices * 8);
    constexpr int PVKs          = Bc / 16;
    constexpr int Group         = Geometry::GroupSize;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(GroupKc == 2 && PVNtPerWarp == 8 && QKNtL == 4);
    static_assert(Layout::Bytes <= 101376);

    extern __shared__ __align__(16) unsigned char gqa_segmented_i8_smem[];
    unsigned char* smem  = gqa_segmented_i8_smem;
    std::int8_t* q_i8    = reinterpret_cast<std::int8_t*>(smem);
    float* q_scale       = reinterpret_cast<float*>(smem + Layout::QScale);
    std::int8_t* k_i8    = reinterpret_cast<std::int8_t*>(smem + Layout::K);
    std::uint8_t* k_raw  = smem + Layout::KRaw;
    std::uint8_t* v_raw  = smem + Layout::VRaw;
    __half* v_f16        = reinterpret_cast<__half*>(smem + Layout::VHalf);
    __half* p_s          = reinterpret_cast<__half*>(smem + Layout::P);
    __half* k_scale_s    = reinterpret_cast<__half*>(smem + Layout::KScale);
    __half* v_scale_s    = reinterpret_cast<__half*>(smem + Layout::VScale);
    float* alpha_s       = reinterpret_cast<float*>(smem + Layout::Stats);
    float* pair_m_s      = alpha_s + Br;
    float* pair_l_s      = pair_m_s + 2 * Br;
    const __nv_bfloat16* q_b16 = reinterpret_cast<const __nv_bfloat16*>(q_i8);
    const __nv_bfloat16* k_b16 = reinterpret_cast<const __nv_bfloat16*>(k_i8);

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
    const int prefix     = params.prefix;
    const float scale_l2 = params.scale_log2;
    const std::int32_t* block_table =
        params.tables + static_cast<std::int64_t>(params.table_rows[0]) * params.table_stride;
    const GqaSegmentedPrefixShare share = gqa_segmented_prefix_share(params, split);
    const int tiles                     = share.end - share.begin;

    // Cache staging. A tile is one physical page; keys at or past the prefix are zero-filled.
    auto live_bytes = [&](int k0, int key_l, int bytes) {
        return k0 + key_l < prefix ? bytes : 0;
    };
    auto stage_k = [&](int tile) {
        const int k0   = tile * Bc;
        const int page = block_table[tile];
        const std::uint8_t* block =
            cache_k + paged_kv_page_head_offset<Layout::KRowBytes, Geometry::KVHeads>(page, kv_head);
        if constexpr (RawK) {
            constexpr int Chunks = Bc * Layout::KRowBytes / 16;
            for (int chunk = tid; chunk < Chunks; chunk += Threads) {
                const int key_l = chunk / (Layout::KRowBytes / 16);
                cp_async_zfill<16, Cache::cg>(k_raw + chunk * 16, block + chunk * 16,
                                              live_bytes(k0, key_l, 16));
            }
        } else {
#pragma unroll
            for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
                const int key_l = chunk / (D / 16);
                const int dc    = chunk % (D / 16);
                std::int8_t* dst = &k_i8[(key_l * DB16 + gqa_prefill_swz(key_l, dc * 8)) * 2];
                cp_async_zfill<16, Cache::cg>(dst, block + key_l * D + dc * 16,
                                              live_bytes(k0, key_l, 16));
            }
        }
    };
    auto stage_v_and_scales = [&](int tile) {
        const int k0   = tile * Bc;
        const int page = block_table[tile];
        const std::uint8_t* block =
            cache_v + paged_kv_page_head_offset<Layout::VRowBytes, Geometry::KVHeads>(page, kv_head);
        constexpr int Chunks = Bc * Layout::VRowBytes / 16;
#pragma unroll
        for (int chunk = tid; chunk < Chunks; chunk += Threads) {
            const int key_l = chunk / (Layout::VRowBytes / 16);
            cp_async_zfill<16, Cache::cg>(v_raw + chunk * 16, block + chunk * 16,
                                          live_bytes(k0, key_l, 16));
        }
        // 64 keys x 4 FP16 scales = 512 bytes per plane; one 16-byte chunk spans two keys.
        if (tid < 64) {
            const int chunk  = tid & 31;
            const int key_l  = 2 * chunk;
            const int bytes  = min(16, max(0, (prefix - k0 - key_l) * 8));
            const std::int64_t offset =
                paged_kv_page_head_offset<Groups, Geometry::KVHeads>(page, kv_head) + key_l * Groups;
            if (tid < 32) {
                cp_async_zfill<16, Cache::cg>(k_scale_s + key_l * Groups, cache_k_scale + offset,
                                              bytes);
            } else {
                cp_async_zfill<16, Cache::cg>(v_scale_s + key_l * Groups, cache_v_scale + offset,
                                              bytes);
            }
        }
    };
    // Expand the staged raw K codes of one tile into the swizzled INT8 QK operand.
    auto expand_k = [&]() {
        if constexpr (KCodec == GqaSegmentedKCodec::Packed4) {
            const int key_l  = tid >> 3;
            const int chunk  = tid & 7;
            const uint4 raw  = load_vec<uint4>(k_raw + key_l * 128 + chunk * 16);
            const uint2 d0   = gqa_kv_unpack_mr4x8(raw.x);
            const uint2 d8   = gqa_kv_unpack_mr4x8(raw.y);
            const uint2 d16  = gqa_kv_unpack_mr4x8(raw.z);
            const uint2 d24  = gqa_kv_unpack_mr4x8(raw.w);
            std::int8_t* lo  = &k_i8[(key_l * DB16 + gqa_prefill_swz(key_l, chunk * 16)) * 2];
            std::int8_t* hi  = &k_i8[(key_l * DB16 + gqa_prefill_swz(key_l, chunk * 16 + 8)) * 2];
            store_vec(lo, make_uint4(d0.x, d0.y, d8.x, d8.y));
            store_vec(hi, make_uint4(d16.x, d16.y, d24.x, d24.y));
        } else if constexpr (KCodec == GqaSegmentedKCodec::E8Root) {
#pragma unroll
            for (int unit = tid; unit < Bc * (D / 16); unit += Threads) {
                const int key_l = unit / (D / 16);
                const int dc    = unit % (D / 16);
                const unsigned codes = load_vec<unsigned>(k_raw + key_l * 64 + dc * 4);
                alignas(16) std::int8_t decoded[16];
                e8_root_decode_8d_int8(static_cast<std::uint8_t>(codes),
                                       static_cast<std::uint8_t>(codes >> 8), decoded);
                e8_root_decode_8d_int8(static_cast<std::uint8_t>(codes >> 16),
                                       static_cast<std::uint8_t>(codes >> 24), decoded + 8);
                std::int8_t* dst = &k_i8[(key_l * DB16 + gqa_prefill_swz(key_l, dc * 8)) * 2];
                store_vec(dst, load_vec<uint4>(decoded));
            }
        }
    };

    stage_k(share.begin);
    stage_v_and_scales(share.begin);
    cp_commit();

    // Q8-G64: one warp quantizes one (row, 64-dimension group) unit at a time. Rotated-K caches
    // take the same orthonormal Hadamard on Q so the dot product is unchanged.
    for (int unit = warp; unit < Br * Groups; unit += Warps) {
        const int row = unit / Groups;
        const int grp = unit - row * Groups;
        const int d0  = grp * kGqaKvQuantGroup + lane;
        const int d1  = d0 + 32;
        float x0      = 0.0f;
        float x1      = 0.0f;
        if (row < tile_rows) {
            const int packed = first_row + row;
            const int column = packed / Group;
            const int q_head = kv_head * Group + (packed - column * Group);
            x0               = __bfloat162float(q[gqa_prefill_q_index<Geometry>(q_head, d0, column)]);
            x1               = __bfloat162float(q[gqa_prefill_q_index<Geometry>(q_head, d1, column)]);
        }
        if constexpr (RotateK) { gqa_kv_hadamard64(x0, x1, FullMask); }
        const float absmax = warp_max(fmaxf(fabsf(x0), fabsf(x1)), FullMask);
        const float qs     = absmax > 0.0f ? absmax / 127.0f : 0.0f;
        const float inv    = qs > 0.0f ? 1.0f / qs : 0.0f;
        gqa_prefill_i8_store_swz(q_i8, row, d0, gqa_kv_quant_code(x0, inv));
        gqa_prefill_i8_store_swz(q_i8, row, d1, gqa_kv_quant_code(x1, inv));
        if (lane == 0) { q_scale[row * Groups + grp] = qs; }
    }

    const bool producer = warp < ProducerWarps;
    const int row_base  = (warp / ColSplit) * 16;
    const int col_half  = warp % ColSplit;

    cp_wait<0>();
    __syncthreads();
    if constexpr (RawK) {
        expand_k();
        __syncthreads();
    }

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    // Two group scales stay in registers; groups 2/3 reload per tile (the spill-free point).
    float q_scale_r0[Groups - 2];
    float q_scale_r1[Groups - 2];
    if (producer) {
#pragma unroll
        for (int grp = 0; grp < Groups - 2; ++grp) {
            q_scale_r0[grp] = q_scale[(row_base + gid) * Groups + grp];
            q_scale_r1[grp] = q_scale[(row_base + gid + 8) * Groups + grp];
        }
    }

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float running_m0 = -CUDART_INF_F;
    float running_m1 = -CUDART_INF_F;
    float running_l0 = 0.0f;
    float running_l1 = 0.0f;

    auto process_tile = [&](int index, int tile, auto masked_tag) {
        constexpr bool Masked = decltype(masked_tag)::value;
        const int k0          = tile * Bc;
        const bool has_next   = index + 1 < tiles;
        const int next        = has_next ? tile + 1 : tile;
        if constexpr (RawK) {
            if (has_next) {
                stage_k(next);
                cp_commit();
            }
        }

        if (producer) {
            float score[QKNtL][4];
#pragma unroll
            for (int ntl = 0; ntl < QKNtL; ++ntl) {
                score[ntl][0] = score[ntl][1] = score[ntl][2] = score[ntl][3] = 0.0f;
            }
#pragma unroll 2
            for (int grp = 0; grp < Groups; ++grp) {
                float qs0;
                float qs1;
                if (grp < Groups - 2) {
                    qs0 = q_scale_r0[grp];
                    qs1 = q_scale_r1[grp];
                } else {
                    qs0 = q_scale[(row_base + gid) * Groups + grp];
                    qs1 = q_scale[(row_base + gid + 8) * Groups + grp];
                }
                unsigned af[GroupKc][4];
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int acol = (grp * GroupKc + kk) * 16 + a_coloff;
                    ldmatrix_x4(af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                                smem_addr(&q_b16[(row_base + a_rowoff) * DB16 +
                                                 gqa_prefill_swz(row_base + a_rowoff, acol)]));
                }
#pragma unroll
                for (int ntl = 0; ntl < QKNtL; ++ntl) {
                    const int nt = col_half * QKNtL + ntl;
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                    for (int kk = 0; kk < GroupKc; ++kk) {
                        const int brow = nt * 8 + b_rin;
                        const int bcol = (grp * GroupKc + kk) * 16 + b_koff;
                        unsigned bf[2];
                        ldmatrix_x2(bf[0], bf[1],
                                    smem_addr(&k_b16[brow * DB16 + gqa_prefill_swz(brow, bcol)]));
                        mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                               bf[1]);
                    }
                    const int keya  = nt * 8 + 2 * lid;
                    const float ks0 = __half2float(k_scale_s[keya * Groups + grp]);
                    const float ks1 = __half2float(k_scale_s[(keya + 1) * Groups + grp]);
                    score[ntl][0]   = __fmaf_rn(qs0 * ks0, static_cast<float>(c0), score[ntl][0]);
                    score[ntl][1]   = __fmaf_rn(qs0 * ks1, static_cast<float>(c1), score[ntl][1]);
                    score[ntl][2]   = __fmaf_rn(qs1 * ks0, static_cast<float>(c2), score[ntl][2]);
                    score[ntl][3]   = __fmaf_rn(qs1 * ks1, static_cast<float>(c3), score[ntl][3]);
                }
            }

            if constexpr (Masked) {
#pragma unroll
                for (int ntl = 0; ntl < QKNtL; ++ntl) {
                    const int key0 = k0 + (col_half * QKNtL + ntl) * 8 + 2 * lid;
                    if (key0 >= prefix) { score[ntl][0] = score[ntl][2] = -CUDART_INF_F; }
                    if (key0 + 1 >= prefix) { score[ntl][1] = score[ntl][3] = -CUDART_INF_F; }
                }
            }
            const int row0 = row_base + gid;
            const int row1 = row0 + 8;
            float bm0      = -CUDART_INF_F;
            float bm1      = -CUDART_INF_F;
#pragma unroll
            for (int ntl = 0; ntl < QKNtL; ++ntl) {
                bm0 = fmaxf(bm0, fmaxf(score[ntl][0], score[ntl][1]));
                bm1 = fmaxf(bm1, fmaxf(score[ntl][2], score[ntl][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);
            if (lid == 0) {
                pair_m_s[col_half * Br + row0] = bm0;
                pair_m_s[col_half * Br + row1] = bm1;
            }
            asm volatile("bar.sync 1, %0;" ::"r"(ProducerWarps * 32) : "memory");
            bm0 = fmaxf(pair_m_s[row0], pair_m_s[Br + row0]);
            bm1 = fmaxf(pair_m_s[row1], pair_m_s[Br + row1]);

            const float nm0 = fmaxf(running_m0, bm0);
            const float nm1 = fmaxf(running_m1, bm1);
            // A finite exponent base sends every -inf score and running max to exact zero.
            const float nm0_scaled = (Masked && nm0 == -CUDART_INF_F) ? 0.0f : nm0 * scale_l2;
            const float nm1_scaled = (Masked && nm1 == -CUDART_INF_F) ? 0.0f : nm1 * scale_l2;
            const float alpha0     = exp2_approx(__fmaf_rn(running_m0, scale_l2, -nm0_scaled));
            const float alpha1     = exp2_approx(__fmaf_rn(running_m1, scale_l2, -nm1_scaled));
            float bl0              = 0.0f;
            float bl1              = 0.0f;
#pragma unroll
            for (int ntl = 0; ntl < QKNtL; ++ntl) {
                const int col0  = (col_half * QKNtL + ntl) * 8 + 2 * lid;
                const float p00 = exp2_approx(__fmaf_rn(score[ntl][0], scale_l2, -nm0_scaled));
                const float p01 = exp2_approx(__fmaf_rn(score[ntl][1], scale_l2, -nm0_scaled));
                const float p10 = exp2_approx(__fmaf_rn(score[ntl][2], scale_l2, -nm1_scaled));
                const float p11 = exp2_approx(__fmaf_rn(score[ntl][3], scale_l2, -nm1_scaled));
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                *reinterpret_cast<__half2*>(&p_s[row0 * Bc + gqa_prefill_i8_p_swz(row0, col0)]) =
                    __floats2half2_rn(p00, p01);
                *reinterpret_cast<__half2*>(&p_s[row1 * Bc + gqa_prefill_i8_p_swz(row1, col0)]) =
                    __floats2half2_rn(p10, p11);
            }
            bl0 = warp_sum<4>(bl0, FullMask);
            bl1 = warp_sum<4>(bl1, FullMask);
            // Both halves share the block max, so their partial row sums add after the loop.
            running_l0 = __fmaf_rn(running_l0, alpha0, bl0);
            running_l1 = __fmaf_rn(running_l1, alpha1, bl1);
            running_m0 = nm0;
            running_m1 = nm1;
            if (col_half == 0 && lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        } else {
            const int worker_tid = tid - ProducerWarps * 32;
#pragma unroll 2
            for (int chunk = worker_tid; chunk < Bc * (D / 8); chunk += WorkerThreads) {
                const int key_l    = chunk / (D / 8);
                const int d        = (chunk % (D / 8)) * 8;
                const __half scale = v_scale_s[key_l * Groups + (d >> 6)];
                __half* dst        = &v_f16[key_l * D + gqa_prefill_swz(key_l, d)];
                if constexpr (PackedV) {
                    store_vec(dst, gqa_kv_dequant_mr4x8_f16(
                                       load_vec<unsigned>(v_raw + key_l * 128 + d / 2), scale));
                } else {
                    store_vec(dst, gqa_prefill_i8_dequant_f16x8(
                                       reinterpret_cast<const std::int8_t*>(v_raw) + key_l * D + d,
                                       scale));
                }
            }
        }
        cp_wait<0>();
        __syncthreads();

        if (has_next) {
            stage_v_and_scales(next);
            if constexpr (!RawK) { stage_k(next); }
            cp_commit();
            if constexpr (RawK) { expand_k(); }
        }

        const int pv_row_base = (warp % RowTiles) * 16;
        const int d_slice     = warp / RowTiles;
        const float alpha0    = alpha_s[pv_row_base + gid];
        const float alpha1    = alpha_s[pv_row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }
        // Ada runs f32-accumulate HMMA at half rate: accumulate the 64-key tile in packed FP16
        // and fold it into the FP32 accumulator once per tile.
        unsigned tacc[PVNtPerWarp][2];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) { tacc[n][0] = tacc[n][1] = 0u; }
#pragma unroll
        for (int k = 0; k < PVKs; ++k) {
            unsigned pf[4];
            const int pcol = k * 16 + a_coloff;
            ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                        smem_addr(&p_s[(pv_row_base + a_rowoff) * Bc +
                                       gqa_prefill_i8_p_swz(pv_row_base + a_rowoff, pcol)]));
#pragma unroll
            for (int n = 0; n < PVNtPerWarp; ++n) {
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = (d_slice * PVNtPerWarp + n) * 8;
                unsigned vf[2];
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_f16[vrow * D + gqa_prefill_swz(vrow, vcol)]));
                mma_f16_f16acc(tacc[n][0], tacc[n][1], pf[0], pf[1], pf[2], pf[3], vf[0], vf[1]);
            }
        }
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            const float2 lo = __half22float2(half2_from_bits(tacc[n][0]));
            const float2 hi = __half22float2(half2_from_bits(tacc[n][1]));
            acc[n][0] += lo.x;
            acc[n][1] += lo.y;
            acc[n][2] += hi.x;
            acc[n][3] += hi.y;
        }
        cp_wait<0>();
        __syncthreads();
    };

    for (int index = 0; index < tiles; ++index) {
        const int tile = share.begin + index;
        if ((tile + 1) * Bc <= prefix) {
            process_tile(index, tile, std::false_type{});
        } else {
            process_tile(index, tile, std::true_type{});
        }
    }

    if (producer && lid == 0) {
        pair_l_s[col_half * Br + row_base + gid]     = running_l0;
        pair_l_s[col_half * Br + row_base + gid + 8] = running_l1;
    }
    __syncthreads();
    if (producer && col_half == 0 && lid == 0) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = row_base + gid + 8 * half;
            if (row >= tile_rows) { continue; }
            const int packed        = first_row + row;
            const int column        = packed / Group;
            const int q_head        = kv_head * Group + (packed - column * Group);
            const std::int64_t stat = gqa_segmented_stat_index(Geometry::QHeads, q_head, column,
                                                               split, params.columns);
            params.partial_m[stat]  = (half == 0 ? running_m0 : running_m1) * scale_l2;
            params.partial_l[stat]  = pair_l_s[row] + pair_l_s[Br + row];
        }
    }

    const int pv_row_base = (warp % RowTiles) * 16;
    const int d_slice     = warp / RowTiles;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int row = pv_row_base + gid + 8 * half;
        if (row >= tile_rows) { continue; }
        const int packed        = first_row + row;
        const int column        = packed / Group;
        const int q_head        = kv_head * Group + (packed - column * Group);
        const std::int64_t stat = gqa_segmented_stat_index(Geometry::QHeads, q_head, column, split,
                                                           params.columns);
        float* acc_row          = params.partial_acc + kGqaSegmentedHeadDim * stat;
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            store_vec(&acc_row[(d_slice * PVNtPerWarp + n) * 8 + 2 * lid],
                      make_float2(acc[n][2 * half], acc[n][2 * half + 1]));
        }
    }
}

} // namespace ninfer::ops
