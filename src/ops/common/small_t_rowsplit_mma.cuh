#pragma once

// Closed Q4/Q5 RowSplit small-T tensor-core mechanism for the decode token
// domain (T <= 32). Semantic Ops own the exact job set, route plan, workspace
// and fixed instantiations.
//
// A CTA owns 64 weight rows and a contiguous range of 64-wide K groups. Eight
// warps form four 16-row groups times two K phases: the warps of phase p
// contract the odd/even groups of every pipeline stage, and the two phases are
// summed once at the end. Weight codes go straight from shared memory into
// BF16 A fragments as exact signed integers; the tensor core accumulates one
// quant group in FP32 and the FP16 group scale is applied in FP32 afterwards,
// so the arithmetic matches the SIMT routes' exact code*activation products
// with a different summation order only. Activations are staged once per CTA
// and shared by the four row groups, which is what keeps L2 traffic at or
// below the weight stream for T <= 32 - the 16-row draft-head kernel re-reads
// them four times as often and is L2-bound past T = 8.
//
// Weights whose row count cannot fill the machine at 64 rows per CTA split K
// across gridDim.y CTAs. Every split writes its FP32 tile to the caller's
// workspace; the last split to arrive (per-tile counter, zeroed by the caller
// before the launch) sums the splits in a fixed order and runs the epilogue,
// so the result is deterministic.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/small_t_rowsplit_mma.h"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

struct SmallTMmaJob {
    const std::uint8_t* codes  = nullptr;
    const std::uint8_t* high   = nullptr; // Q5 only
    const std::uint8_t* scales = nullptr;
    __nv_bfloat16* out0        = nullptr;
    __nv_bfloat16* out1        = nullptr;
    std::int32_t n             = 0; // weight rows, multiple of 64
    std::int32_t split_row     = 0; // Store: first row of out1; multiple of 64 or n
    std::int32_t ld0           = 0;
    std::int32_t ld1           = 0;
    bool q5                    = false;
};

struct SmallTMmaParams {
    const __nv_bfloat16* x = nullptr; // [k, cols], k contiguous per column
    std::int32_t k         = 0;       // multiple of 64
    std::int32_t cols      = 0;       // 1 <= cols <= TileCols
    // Groups per split; gridDim.y splits cover k / 64 groups. Even, so a stage's two groups
    // share one 4-byte scale word.
    std::int32_t groups_per_split = 0;
    SmallTMmaJob job0;
    SmallTMmaJob job1;
    std::int32_t tiles0 = 0; // blockIdx.x < tiles0 -> job0, else job1
    // Split-K (gridDim.y > 1): partials hold gridDim.y * tiles * 64 * TileCols floats,
    // counters one int per tile, zero at launch.
    float* partials = nullptr;
    int* counters   = nullptr;
};

namespace small_t_mma {

// Activation chunk swizzle: the eight 16-byte chunks of a token's 64 K values are XOR-rotated
// by the token so ldmatrix over eight tokens hits eight bank groups.
__device__ __forceinline__ int act_offset(int col, int k) {
    return col * SmallTMmaShape::kGroupK + ((((k >> 3) ^ (col & 7)) << 3) | (k & 7));
}

// Code chunk swizzle: rows r and r+4 share bank groups at the same chunk, so the two 16-byte
// chunks of a row swap places every four rows.
__device__ __forceinline__ int code_offset(int row, int chunk) {
    return row * 32 + ((chunk ^ ((row >> 2) & 1)) << 4);
}

// d = a - c in BF16x2, exact for the small integers this kernel feeds it.
__device__ __forceinline__ unsigned bf16x2_sub(unsigned a, unsigned c_neg) {
    unsigned d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(0x3F803F80u), "r"(c_neg));
    return d;
}

// Byte `lid` of `word` holds the codes of k = 2*lid (low nibble) and 2*lid + 1 (high nibble).
// Returns the BF16x2 pair of their signed values. `word` is pre-XORed with 0x88888888 so the
// nibbles are offset-binary, and 128 + nibble is exact in BF16 (0x4300 | nibble).
__device__ __forceinline__ unsigned q4_pair(unsigned word, int lid) {
    const unsigned v = __byte_perm(word, word >> 4, 0x0400u + static_cast<unsigned>(lid) * 0x101u);
    return bf16x2_sub((v & 0x000F000Fu) | 0x43004300u, 0xC308C308u); // - 136
}

// Same for Q5: `hi2` carries the fifth bits of k = 2*lid (bit 0) and 2*lid + 1 (bit 1); the
// five-bit code is sign-flipped (^ 0x10) into offset-binary and 128 + code is exact.
__device__ __forceinline__ unsigned q5_pair(unsigned word, unsigned hi2, int lid) {
    const unsigned v = __byte_perm(word, word >> 4, 0x0400u + static_cast<unsigned>(lid) * 0x101u);
    const unsigned h = hi2 ^ 3u;
    const unsigned t = ((h & 1u) << 4) | ((h & 2u) << 19);
    return bf16x2_sub((v & 0x000F000Fu) | t | 0x43004300u, 0xC310C310u); // - 144
}

} // namespace small_t_mma

template <int TileCols, int kStages, SmallTMmaEpilogue Epi, bool Q5>
__device__ __forceinline__ void small_t_mma_run(const SmallTMmaParams& p, const SmallTMmaJob& job,
                                                int tile, int tile_global, std::uint8_t* smem) {
    using Shape          = SmallTMmaShape;
    constexpr int kNt    = TileCols / 8;
    constexpr int kStageBytes = Shape::stage_bytes(TileCols);
    constexpr int kGroupK = Shape::kGroupK;
    static_assert(TileCols % 8 == 0 && TileCols >= 8 && TileCols <= 32);

    const int tid   = static_cast<int>(threadIdx.x);
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int gid   = lane >> 2;
    const int lid   = lane & 3;
    const int rg    = warp & 3;
    const int phase = warp >> 2;
    const int split = static_cast<int>(blockIdx.y);
    const int splits = static_cast<int>(gridDim.y);

    const std::int32_t groups_per_row = p.k / kGroupK;
    const std::int32_t group_begin    = split * p.groups_per_split;
    // The last split takes the (even) remainder; the launcher guarantees it is non-empty.
    const std::int32_t groups_here = min(p.groups_per_split, groups_per_row - group_begin);
    const int n_stages             = groups_here / Shape::kPhases;

    // Weight row of local row `r` (0..63).
    const auto weight_row = [&](int r) -> std::int32_t {
        if constexpr (Epi == SmallTMmaEpilogue::SwiGlu) {
            const std::int32_t half = job.n / 2;
            return (r < 32 ? 0 : half) + tile * 32 + (r & 31);
        } else {
            return tile * Shape::kRows + r;
        }
    };

    const auto stage_codes  = [&](int s) { return smem + s * kStageBytes + Shape::kCodeOffset; };
    const auto stage_high   = [&](int s) { return smem + s * kStageBytes + Shape::kHighOffset; };
    const auto stage_scales = [&](int s) {
        return reinterpret_cast<std::uint16_t*>(smem + s * kStageBytes + Shape::kScaleOffset);
    };
    const auto stage_act = [&](int s) {
        return reinterpret_cast<__nv_bfloat16*>(smem + s * kStageBytes + Shape::kActOffset);
    };

    // One stage = two consecutive groups; group g of the stage is contracted by phase g.
    const auto issue = [&](int stage_index, int buffer) {
        const std::int32_t g0 = group_begin + stage_index * Shape::kPhases;
        {
            // Codes: 2 groups x 64 rows x 2 chunks of 16 bytes, one per thread.
            const int g     = tid >> 7;
            const int row   = (tid >> 1) & 63;
            const int chunk = tid & 1;
            const std::uint8_t* src =
                job.codes + (static_cast<std::int64_t>(weight_row(row)) * groups_per_row + g0 + g) *
                                Q4RowSplitStorage::kCodeBytesPerGroup +
                chunk * 16;
            cp_async<16, Cache::cg, L2Prefetch::B256>(
                stage_codes(buffer) + g * (Shape::kRows * 32) + small_t_mma::code_offset(row, chunk),
                src);
        }
        if constexpr (Q5) {
            if (tid < Shape::kPhases * Shape::kRows) {
                const int g   = tid >> 6;
                const int row = tid & 63;
                const std::uint8_t* src =
                    job.high +
                    (static_cast<std::int64_t>(weight_row(row)) * groups_per_row + g0 + g) *
                        Q5RowSplitStorage::kHighBytesPerGroup;
                cp_async<8, Cache::ca, L2Prefetch::B256>(
                    stage_high(buffer) + (g * Shape::kRows + row) * 8, src);
            }
        }
        if (tid < Shape::kRows) {
            const int row = tid;
            const std::uint8_t* src =
                job.scales +
                (static_cast<std::int64_t>(weight_row(row)) * groups_per_row + g0) * 2;
            cp_async<4, Cache::ca, L2Prefetch::B256>(stage_scales(buffer) + row * 2, src);
        }
        {
            constexpr int kChunks = Shape::kPhases * TileCols * (kGroupK / 8);
            for (int item = tid; item < kChunks; item += Shape::kThreads) {
                const int g   = item / (TileCols * 8);
                const int rem = item - g * (TileCols * 8);
                const int col = rem >> 3;
                const int k8  = rem & 7;
                const bool live = col < p.cols;
                const __nv_bfloat16* src =
                    live ? p.x + static_cast<std::int64_t>(col) * p.k + (g0 + g) * kGroupK + k8 * 8
                         : p.x;
                cp_async_zfill<16>(stage_act(buffer) + g * (TileCols * kGroupK) +
                                       small_t_mma::act_offset(col, k8 * 8),
                                   src, live ? 16 : 0);
            }
        }
    };

    float acc[kNt][4];
#pragma unroll
    for (int nt = 0; nt < kNt; ++nt) {
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
    }

    const int row_a = rg * 16 + gid;
    const int row_b = row_a + 8;
    const int b_row = lane & 7;
    const int b_koff = ((lane >> 3) & 1) << 3;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < n_stages) { issue(s, s); }
        cp_commit();
    }

    for (int i = 0; i < n_stages; ++i) {
        cp_wait<kStages - 2>();
        __syncthreads();
        if (i + kStages - 1 < n_stages) { issue(i + kStages - 1, (i + kStages - 1) % kStages); }
        cp_commit();

        const int buffer               = i % kStages;
        const std::uint8_t* codes_s    = stage_codes(buffer) + phase * (Shape::kRows * 32);
        const std::uint16_t* scales_s  = stage_scales(buffer);
        const __nv_bfloat16* act_s     = stage_act(buffer) + phase * (TileCols * kGroupK);

        float gacc[kNt][4];
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            gacc[nt][0] = gacc[nt][1] = gacc[nt][2] = gacc[nt][3] = 0.0f;
        }

        [[maybe_unused]] unsigned long long high_a = 0, high_b = 0;
        if constexpr (Q5) {
            const std::uint8_t* high_s = stage_high(buffer) + phase * (Shape::kRows * 8);
            high_a = load_vec<unsigned long long>(high_s + row_a * 8);
            high_b = load_vec<unsigned long long>(high_s + row_b * 8);
        }

#pragma unroll
        for (int half = 0; half < 2; ++half) {
            uint4 ca = load_vec<uint4>(codes_s + small_t_mma::code_offset(row_a, half));
            uint4 cb = load_vec<uint4>(codes_s + small_t_mma::code_offset(row_b, half));
            if constexpr (!Q5) {
                ca.x ^= 0x88888888u; ca.y ^= 0x88888888u; ca.z ^= 0x88888888u; ca.w ^= 0x88888888u;
                cb.x ^= 0x88888888u; cb.y ^= 0x88888888u; cb.z ^= 0x88888888u; cb.w ^= 0x88888888u;
            }
#pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                const int ks       = half * 2 + sub;
                const unsigned wa0 = sub == 0 ? ca.x : ca.z;
                const unsigned wa1 = sub == 0 ? ca.y : ca.w;
                const unsigned wb0 = sub == 0 ? cb.x : cb.z;
                const unsigned wb1 = sub == 0 ? cb.y : cb.w;
                unsigned af0, af1, af2, af3;
                if constexpr (Q5) {
                    const unsigned ha = static_cast<unsigned>(high_a >> (16 * ks)) & 0xFFFFu;
                    const unsigned hb = static_cast<unsigned>(high_b >> (16 * ks)) & 0xFFFFu;
                    af0 = small_t_mma::q5_pair(wa0, (ha >> (2 * lid)) & 3u, lid);
                    af1 = small_t_mma::q5_pair(wb0, (hb >> (2 * lid)) & 3u, lid);
                    af2 = small_t_mma::q5_pair(wa1, (ha >> (8 + 2 * lid)) & 3u, lid);
                    af3 = small_t_mma::q5_pair(wb1, (hb >> (8 + 2 * lid)) & 3u, lid);
                } else {
                    af0 = small_t_mma::q4_pair(wa0, lid);
                    af1 = small_t_mma::q4_pair(wb0, lid);
                    af2 = small_t_mma::q4_pair(wa1, lid);
                    af3 = small_t_mma::q4_pair(wb1, lid);
                }
#pragma unroll
                for (int nt = 0; nt < kNt; ++nt) {
                    unsigned bf0, bf1;
                    ldmatrix_x2(bf0, bf1,
                                smem_addr(act_s + small_t_mma::act_offset(nt * 8 + b_row,
                                                                          ks * 16 + b_koff)));
                    mma_bf16(gacc[nt][0], gacc[nt][1], gacc[nt][2], gacc[nt][3], af0, af1, af2,
                             af3, bf0, bf1);
                }
            }
        }

        const float scale_a = __half2float(__ushort_as_half(scales_s[row_a * 2 + phase]));
        const float scale_b = __half2float(__ushort_as_half(scales_s[row_b * 2 + phase]));
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            acc[nt][0] = fmaf(gacc[nt][0], scale_a, acc[nt][0]);
            acc[nt][1] = fmaf(gacc[nt][1], scale_a, acc[nt][1]);
            acc[nt][2] = fmaf(gacc[nt][2], scale_b, acc[nt][2]);
            acc[nt][3] = fmaf(gacc[nt][3], scale_b, acc[nt][3]);
        }
    }

    cp_wait<0>();
    __syncthreads();

    // Phase reduction through shared memory (the pipeline buffers are free now).
    float4* red = reinterpret_cast<float4*>(smem);
    const auto red_at = [&](int w, int nt) { return red + (w * kNt + nt) * 32 + lane; };
#pragma unroll
    for (int nt = 0; nt < kNt; ++nt) {
        *red_at(warp, nt) = make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]);
    }
    __syncthreads();
    if (phase != 0) { return; }

    float4 sum[kNt];
#pragma unroll
    for (int nt = 0; nt < kNt; ++nt) {
        const float4 a = *red_at(rg, nt);
        const float4 b = *red_at(rg + 4, nt);
        sum[nt] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }

    if (splits > 1) {
        const std::int32_t tiles = static_cast<std::int32_t>(gridDim.x);
        const auto partial_at    = [&](int s, int nt) {
            return reinterpret_cast<float4*>(p.partials) +
                   ((static_cast<std::int64_t>(s) * tiles + tile_global) * 4 + rg) * (kNt * 32) +
                   nt * 32 + lane;
        };
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) { __stcg(partial_at(split, nt), sum[nt]); }
        __threadfence();
        // Warps 0-3 are the only live warps here; they agree on the arrival through smem.
        int* flag = reinterpret_cast<int*>(smem + Shape::kWarps * kNt * 32 * sizeof(float4));
        asm volatile("bar.sync 1, 128;");
        if (tid == 0) {
            const int prev = atomicAdd(p.counters + tile_global, 1);
            *flag          = prev == splits - 1;
        }
        asm volatile("bar.sync 1, 128;");
        if (*flag == 0) { return; }
        __threadfence();
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            float4 total = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int s = 0; s < splits; ++s) {
                const float4 v = __ldcg(partial_at(s, nt));
                total.x += v.x;
                total.y += v.y;
                total.z += v.z;
                total.w += v.w;
            }
            sum[nt] = total;
        }
    }

    if constexpr (Epi == SmallTMmaEpilogue::SwiGlu) {
        // Warps 0/1 hold gate rows, 2/3 the matching up rows. The folded up sums travel
        // through the phase-1 scratch of warps 2/3 (free since the reduction); only warps 0/1
        // store.
        if (rg >= 2) {
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) { *red_at(rg + 4, nt) = sum[nt]; }
        }
        asm volatile("bar.sync 1, 128;");
        if (rg >= 2) { return; }
        const std::int32_t out_row_a = tile * 32 + rg * 16 + gid;
        const std::int32_t out_row_b = out_row_a + 8;
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            const float4 up = *red_at(rg + 6, nt);
            const int col0  = nt * 8 + 2 * lid;
            if (col0 < p.cols) {
                job.out0[static_cast<std::int64_t>(col0) * job.ld0 + out_row_a] =
                    __float2bfloat16_rn(silu(sum[nt].x) * up.x);
                job.out0[static_cast<std::int64_t>(col0) * job.ld0 + out_row_b] =
                    __float2bfloat16_rn(silu(sum[nt].z) * up.z);
            }
            if (col0 + 1 < p.cols) {
                job.out0[static_cast<std::int64_t>(col0 + 1) * job.ld0 + out_row_a] =
                    __float2bfloat16_rn(silu(sum[nt].y) * up.y);
                job.out0[static_cast<std::int64_t>(col0 + 1) * job.ld0 + out_row_b] =
                    __float2bfloat16_rn(silu(sum[nt].w) * up.w);
            }
        }
        return;
    } else {
        const std::int32_t row_a_global = weight_row(row_a);
        const std::int32_t row_b_global = row_a_global + 8;
        __nv_bfloat16* out;
        std::int64_t ld;
        std::int32_t base;
        if constexpr (Epi == SmallTMmaEpilogue::Store) {
            const bool tail = row_a_global >= job.split_row;
            out             = tail ? job.out1 : job.out0;
            ld              = tail ? job.ld1 : job.ld0;
            base            = tail ? job.split_row : 0;
        } else {
            out  = job.out0;
            ld   = job.ld0;
            base = 0;
        }
        const auto emit = [&](std::int32_t col, std::int32_t row, float value) {
            __nv_bfloat16* dst = out + static_cast<std::int64_t>(col) * ld + (row - base);
            if constexpr (Epi == SmallTMmaEpilogue::Residual) {
                value += __bfloat162float(*dst);
            }
            *dst = __float2bfloat16_rn(value);
        };
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            const int col0 = nt * 8 + 2 * lid;
            if (col0 < p.cols) {
                emit(col0, row_a_global, sum[nt].x);
                emit(col0, row_b_global, sum[nt].z);
            }
            if (col0 + 1 < p.cols) {
                emit(col0 + 1, row_a_global, sum[nt].y);
                emit(col0 + 1, row_b_global, sum[nt].w);
            }
        }
    }
}

template <int TileCols, int Stages, int MinBlocks, SmallTMmaEpilogue Epi, SmallTMmaCodec Codec>
__global__ void __launch_bounds__(SmallTMmaShape::kThreads, MinBlocks)
    small_t_rowsplit_mma_kernel(SmallTMmaParams p) {
    __shared__ __align__(128) std::uint8_t smem[SmallTMmaShape::smem_bytes(TileCols, Stages)];
    const int tile_global = static_cast<int>(blockIdx.x);
    if constexpr (Codec == SmallTMmaCodec::Q4) {
        small_t_mma_run<TileCols, Stages, Epi, false>(p, p.job0, tile_global, tile_global, smem);
    } else if constexpr (Codec == SmallTMmaCodec::Q5) {
        small_t_mma_run<TileCols, Stages, Epi, true>(p, p.job0, tile_global, tile_global, smem);
    } else {
        if (tile_global < p.tiles0) {
            small_t_mma_run<TileCols, Stages, Epi, false>(p, p.job0, tile_global, tile_global,
                                                          smem);
        } else {
            small_t_mma_run<TileCols, Stages, Epi, true>(p, p.job1, tile_global - p.tiles0,
                                                         tile_global, smem);
        }
    }
}

} // namespace ninfer::ops::detail
