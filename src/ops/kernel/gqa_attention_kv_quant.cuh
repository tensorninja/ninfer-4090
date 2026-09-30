#pragma once

// ninfer::ops - signed int8, per-token group-wise KV cache codec (shared device
// helpers). Quantization (append) and dequantization (stage) are FUSED into the
// GQA attention kernels themselves (decode partial kernel, prefill fill/attention);
// this header only provides the index math, the vectorized dequant, and the scalar
// quantize helper they share. There is deliberately no standalone quant/dequant
// kernel: that would defeat the halved-bandwidth goal.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaKvQuantHeadDim = 256;
inline constexpr int kGqaKvQuantGroup   = 64;
inline constexpr int kGqaKvQuantGroups  = kGqaKvQuantHeadDim / kGqaKvQuantGroup;

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_code_index(int physical_page, int kv_head,
                                                                int d, int page_offset) {
    return paged_kv_element_offset<kGqaKvQuantHeadDim, Geometry::KVHeads>(physical_page, kv_head,
                                                                          page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_scale_index(int physical_page, int kv_head,
                                                                 int group, int page_offset) {
    return paged_kv_element_offset<kGqaKvQuantGroups, Geometry::KVHeads>(physical_page, kv_head,
                                                                         page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_src_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaKvQuantHeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

// Quantize one bf16 value with a precomputed 1/scale (scale is the FP16-rounded
// per-group absmax/127). Round-to-nearest-even + symmetric clamp to keep codes
// bit-identical to the CPU oracle and to bf16 parity.
__device__ __forceinline__ std::int8_t gqa_kv_quant_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-127, min(127, q));
    return static_cast<std::int8_t>(q);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_i4_code_index(int physical_page, int kv_head,
                                                             int packed_d, int page_offset) {
    return paged_kv_element_offset<kGqaKvQuantHeadDim / 2, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, packed_d);
}

// Every packed 4-bit plane (rk8v4 and rk4v4 values, rk4v4 keys, rk2v4-e8 values) holds the
// midrise codec of one rotated 64-element group. Its FP16 group scale h is half a step:
//
//   a     = max_i abs(x[i])
//   h     = FP32(FP16_RNE(a / 15))
//   inv   = h == 0 ? 0 : FP32(1 / h)
//   c[i]  = clamp(floor(FP32(x[i]) * inv / 2), -8, 7)    signed nibble, even dimension low
//   value = (2 * c[i] + 1) * h
//
// The sixteen levels are the odd multiples of h in [-15h, 15h], so both group extremes decode to
// +-a up to the scale rounding. Decoders expand a nibble to its odd INT8 code 2c + 1 and apply h
// exactly as they apply an INT8-G64 scale.
inline constexpr float kGqaKvMr4ScaleDivisor = 15.0f;
// The E8-root key codec (rk2v4-e8) keeps its own scale convention.
inline constexpr float kGqaKvE8RootScaleDivisor = 7.0f;

__device__ __forceinline__ std::int8_t gqa_kv_quant_mr4_code(float x, float inv_scale) {
    const int code = __float2int_rd(0.5f * (x * inv_scale));
    return static_cast<std::int8_t>(max(-8, min(7, code)));
}

__device__ __forceinline__ std::uint8_t gqa_kv_pack_i4(std::int8_t lo, std::int8_t hi) {
    return static_cast<std::uint8_t>((static_cast<unsigned>(lo) & 0x0fu) |
                                     ((static_cast<unsigned>(hi) & 0x0fu) << 4));
}

// Eight midrise nibbles (dimension order, low nibble first) to eight odd INT8 codes 2c + 1. Each
// nibble lands in bits 1..4 of its byte above a set bit 0, which is 2c + 1 in five-bit two's
// complement; the multiply copies bit 4 into bits 5..7 without carrying across bytes.
__device__ __forceinline__ uint2 gqa_kv_unpack_mr4x8(unsigned word) {
    const unsigned shifted = word >> 4;
    unsigned lo = ((__byte_perm(word, shifted, 0x5140u) << 1) & 0x1e1e1e1eu) | 0x01010101u;
    unsigned hi = ((__byte_perm(word, shifted, 0x7362u) << 1) & 0x1e1e1e1eu) | 0x01010101u;
    lo |= (lo & 0x10101010u) * 0x0eu;
    hi |= (hi & 0x10101010u) * 0x0eu;
    return make_uint2(lo, hi);
}

// Eight midrise nibbles to eight FP16 values (2c + 1) * h. Biased nibbles c + 8 are spliced
// under the FP16 exponents of 2048 (low nibble) and 128 (high nibble at bits 4..7), where one
// nibble unit weighs 2; subtracting 2064 and 144 recovers 2c exactly, and one fused multiply-add
// forms 2c * h + h with the single rounding of the product.
__device__ __forceinline__ int4 gqa_kv_dequant_mr4x8_f16(unsigned word, __half scale) {
    const unsigned biased = word ^ 0x88888888u;
    const __half2 magic   = __halves2half2(__ushort_as_half(0x6808), __ushort_as_half(0x5880));
    const __half2 scale2  = __halves2half2(scale, scale);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned selector = 0x7050u | (static_cast<unsigned>(i) << 8) | static_cast<unsigned>(i);
        const unsigned pair     = __byte_perm(biased, 0x58006800u, selector) & 0xfff0ff0fu;
        const __half2 twice     = __hsub2(half2_from_bits(pair), magic);
        const __half2 value     = __hfma2(twice, scale2, scale2);
        packed[i]               = *reinterpret_cast<const unsigned*>(&value);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

__device__ __forceinline__ void gqa_kv_hadamard64(float& x0, float& x1,
                                                  unsigned mask = 0xffffffffu) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const float y0 = __shfl_xor_sync(mask, x0, offset);
        const float y1 = __shfl_xor_sync(mask, x1, offset);
        const bool hi  = (static_cast<int>(threadIdx.x) & offset) != 0;
        x0             = hi ? y0 - x0 : x0 + y0;
        x1             = hi ? y1 - x1 : x1 + y1;
    }
    const float a = x0;
    const float b = x1;
    x0            = (a + b) * 0.125f;
    x1            = (a - b) * 0.125f;
}

template <int QHeads>
__global__ void gqa_kv_inverse_rotate_output_kernel(__nv_bfloat16* output, int width,
                                                     int full_width, int column_begin,
                                                     const std::int32_t* valid_columns) {
    const int unit       = static_cast<int>(blockIdx.x);
    const int lane       = static_cast<int>(threadIdx.x);
    if (lane >= 32) { return; }
    const int group  = unit % kGqaKvQuantGroups;
    const int tmp    = unit / kGqaKvQuantGroups;
    const int q_head = tmp % QHeads;
    const int row    = tmp / QHeads;
    const int batch  = row / width;
    const int token  = row - batch * width;
    const int column = column_begin + token;
    if (token >= width || (valid_columns != nullptr && column >= valid_columns[batch])) { return; }
    const int d0 = group * kGqaKvQuantGroup + lane;
    const int d1 = d0 + 32;
    const std::int64_t base = static_cast<std::int64_t>(kGqaKvQuantHeadDim) *
                              (q_head + static_cast<std::int64_t>(QHeads) *
                                            (column + static_cast<std::int64_t>(full_width) * batch));
    float x0 = __bfloat162float(output[base + d0]);
    float x1 = __bfloat162float(output[base + d1]);
    gqa_kv_hadamard64(x0, x1);
    output[base + d0] = __float2bfloat16(x0);
    output[base + d1] = __float2bfloat16(x1);
}

// Sixteen midrise nibbles (eight bytes) to sixteen odd INT8 codes.
__device__ __forceinline__ void gqa_kv_unpack_mr4x16(const std::uint8_t* src8,
                                                     std::int8_t* dst16) {
    const uint2 raw = load_vec<uint2>(src8);
    const uint2 lo  = gqa_kv_unpack_mr4x8(raw.x);
    const uint2 hi  = gqa_kv_unpack_mr4x8(raw.y);
    store_vec(dst16, make_uint4(lo.x, lo.y, hi.x, hi.y));
}

// Dequantize 8 consecutive int8 codes (dims [d, d+8), aligned to a multiple of 8
// so they lie inside one 64-group) into 8 bf16 packed as an int4, given a pointer
// to the 8 codes and the group's dequant scale. The codes are read with ONE 64-bit
// (int2) load; the pointer may be in global or shared memory. This keeps the dequant
// ALU identical whether the codes were streamed via cp.async into smem (decode) or
// read directly from the cache (prefill).
__device__ __forceinline__ int4 gqa_kv_dequant_i8x8_from(const std::int8_t* codes8, float s) {
    const int2 raw       = load_vec<int2>(codes8);
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(c[2 * i]) * s;
        const float x1 = static_cast<float>(c[2 * i + 1]) * s;
        packed[i]      = pack_bf16x2(x0, x1);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

} // namespace ninfer::ops
