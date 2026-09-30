#pragma once

// ninfer::ops - the KV cache encoder of every append site: the A2 fill kernels
// (gqa_attention_prefill_i8.cuh) and the fused A1 decode append (gqa_attention_decode_i8.cuh).
// One full warp encodes one (token, KV head, 64-dimension group) unit; lane l holds dimensions
// group * 64 + l and group * 64 + l + 32 of the unrotated K and V values.

#include "ops/kernel/e8_root_codec.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"

#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

// physical_page_lane0 is read from lane 0 only, late, so the caller's page-table load overlaps
// the rotation and the scale reduction.
template <typename Geometry, bool PackedV, bool RotateK, bool RotateV, bool PackedK, bool E8Root>
__device__ __forceinline__ void gqa_kv_encode_group(float k0, float k1, float v0, float v1,
                                                    int physical_page_lane0, int kv_head, int group,
                                                    int page_offset, int lane,
                                                    std::int8_t* __restrict__ cache_k,
                                                    std::uint8_t* __restrict__ cache_v,
                                                    __half* __restrict__ scale_k,
                                                    __half* __restrict__ scale_v) {
    static_assert(!(PackedK && E8Root));
    constexpr unsigned FullMask = 0xffffffffu;
    constexpr float KDivisor    = PackedK  ? kGqaKvMr4ScaleDivisor
                                  : E8Root ? kGqaKvE8RootScaleDivisor
                                           : 127.0f;
    constexpr float VDivisor    = PackedV ? kGqaKvMr4ScaleDivisor : 127.0f;

    if constexpr (RotateK) { gqa_kv_hadamard64(k0, k1, FullMask); }
    if constexpr (RotateV) { gqa_kv_hadamard64(v0, v1, FullMask); }
    const float k_abs = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
    const float v_abs = warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMask);
    const __half ksh  = __float2half_rn(k_abs > 0.0f ? k_abs / KDivisor : 0.0f);
    const __half vsh  = __float2half_rn(v_abs > 0.0f ? v_abs / VDivisor : 0.0f);
    const float ks    = __half2float(ksh);
    const float vs    = __half2float(vsh);
    const float kinv  = ks > 0.0f ? 1.0f / ks : 0.0f;
    const float vinv  = vs > 0.0f ? 1.0f / vs : 0.0f;
    const int page    = __shfl_sync(FullMask, physical_page_lane0, 0);
    const int d0      = group * kGqaKvQuantGroup + lane;
    const int d1      = d0 + 32;

    if constexpr (E8Root) {
        // Each 8-lane subgroup encodes one 8-dimension block into a (root, radius/axis) pair.
        std::uint8_t root0 = 0, radius0 = 0, root1 = 0, radius1 = 0;
        e8_encode_cylinder_8d_warp(k0, ks, root0, radius0, lane);
        e8_encode_cylinder_8d_warp(k1, ks, root1, radius1, lane);
        if ((lane & 7) == 0) {
            std::uint8_t* row = reinterpret_cast<std::uint8_t*>(cache_k) +
                                paged_kv_page_head_offset<64, Geometry::KVHeads>(page, kv_head) +
                                static_cast<std::int64_t>(page_offset) * 64 + group * 16;
            const int block0    = lane / 8;
            const int block1    = 4 + lane / 8;
            row[2 * block0]     = root0;
            row[2 * block0 + 1] = radius0;
            row[2 * block1]     = root1;
            row[2 * block1 + 1] = radius1;
        }
    } else if constexpr (PackedK) {
        const std::int8_t c0    = gqa_kv_quant_mr4_code(k0, kinv);
        const std::int8_t c1    = gqa_kv_quant_mr4_code(k1, kinv);
        const std::int8_t c0_hi = static_cast<std::int8_t>(__shfl_down_sync(FullMask, static_cast<int>(c0), 1));
        const std::int8_t c1_hi = static_cast<std::int8_t>(__shfl_down_sync(FullMask, static_cast<int>(c1), 1));
        if ((lane & 1) == 0) {
            auto* codes = reinterpret_cast<std::uint8_t*>(cache_k);
            codes[gqa_kv_i4_code_index<Geometry>(page, kv_head, d0 / 2, page_offset)] =
                gqa_kv_pack_i4(c0, c0_hi);
            codes[gqa_kv_i4_code_index<Geometry>(page, kv_head, d1 / 2, page_offset)] =
                gqa_kv_pack_i4(c1, c1_hi);
        }
    } else {
        cache_k[gqa_kv_quant_code_index<Geometry>(page, kv_head, d0, page_offset)] =
            gqa_kv_quant_code(k0, kinv);
        cache_k[gqa_kv_quant_code_index<Geometry>(page, kv_head, d1, page_offset)] =
            gqa_kv_quant_code(k1, kinv);
    }

    if constexpr (PackedV) {
        const float v0_hi = __shfl_down_sync(FullMask, v0, 1);
        const float v1_hi = __shfl_down_sync(FullMask, v1, 1);
        if ((lane & 1) == 0) {
            cache_v[gqa_kv_i4_code_index<Geometry>(page, kv_head, d0 / 2, page_offset)] =
                gqa_kv_pack_i4(gqa_kv_quant_mr4_code(v0, vinv), gqa_kv_quant_mr4_code(v0_hi, vinv));
            cache_v[gqa_kv_i4_code_index<Geometry>(page, kv_head, d1 / 2, page_offset)] =
                gqa_kv_pack_i4(gqa_kv_quant_mr4_code(v1, vinv), gqa_kv_quant_mr4_code(v1_hi, vinv));
        }
    } else {
        auto* codes = reinterpret_cast<std::int8_t*>(cache_v);
        codes[gqa_kv_quant_code_index<Geometry>(page, kv_head, d0, page_offset)] =
            gqa_kv_quant_code(v0, vinv);
        codes[gqa_kv_quant_code_index<Geometry>(page, kv_head, d1, page_offset)] =
            gqa_kv_quant_code(v1, vinv);
    }

    if (lane == 0) {
        const std::int64_t offset = gqa_kv_quant_scale_index<Geometry>(page, kv_head, group, page_offset);
        scale_k[offset] = ksh;
        scale_v[offset] = vsh;
    }
}

} // namespace ninfer::ops
