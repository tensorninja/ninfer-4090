#pragma once

// ninfer::ops - pointer-head kernels: bias-added BF16 projection to FP32 and per-question softmax
// scoring of FP32 key columns against one FP32 query column.

#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

// D values one lane group consumes per fragment load: two m16n8k16 k-steps.
inline constexpr int kPointerHeadChunk = 32;
// Largest key count of one question; the scoring kernel keeps every logit in shared memory.
inline constexpr int kPointerHeadMaxKeys = 255;

// Projection geometry. A tile of MTiles*16 A-operand rows and NTiles*8 B-operand rows is owned
// by one CTA whose KWarps warps split D into interleaved 32-value chunks. When WeightIsA, the
// A operand is the weight (M over P) and B the hidden columns (N over R); otherwise the roles
// swap so that eight weight rows form the smallest CTA tile. Grid x walks R tiles and grid y
// walks P tiles, which the P<=65536 domain keeps within the 65535 limit of grid y.
template <bool WeightIsA, int MTiles, int NTiles, int KWarps, int Depth>
struct PointerHeadProjectConfig {
    static constexpr bool kWeightIsA = WeightIsA;
    static constexpr int kMTiles     = MTiles;
    static constexpr int kNTiles     = NTiles;
    static constexpr int kKWarps     = KWarps;
    static constexpr int kDepth      = Depth;
    static constexpr int kThreads    = KWarps * kWarpSize;
    static constexpr int kTileM      = MTiles * 16;
    static constexpr int kTileN      = NTiles * 8;
    static constexpr int kTileP      = WeightIsA ? kTileM : kTileN;
    static constexpr int kTileR      = WeightIsA ? kTileN : kTileM;
    // Padding keeps the fragment scatter into the reduction tile free of 4-way bank conflicts.
    static constexpr int kReducePitch  = kTileP + 4;
    static constexpr int kReduceFloats = KWarps * kTileR * kReducePitch;
};

template <int MTiles, int NTiles>
struct PointerHeadFragments {
    uint4 a[MTiles][2];
    uint4 b[NTiles];
};

// Lane (g,t) owns D offsets [8t,8t+8) of a chunk for A rows g and g+8 and B row g. Within the
// chunk, k-step 0 consumes offsets {8t..8t+3} and k-step 1 offsets {8t+4..8t+7}; the same
// permutation of D inside a chunk is applied to both operands, so every product pairs equal D
// indices and the contraction is unchanged.
template <int MTiles, int NTiles>
__device__ __forceinline__ void pointer_head_load_chunk(
    PointerHeadFragments<MTiles, NTiles>& fragments, const __nv_bfloat16* __restrict__ a_source,
    const __nv_bfloat16* __restrict__ b_source, const std::int32_t (&a_offsets)[MTiles][2],
    const std::int32_t (&b_offsets)[NTiles], std::int32_t chunk_offset) {
#pragma unroll
    for (int mt = 0; mt < MTiles; ++mt) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            fragments.a[mt][half] = __ldg(
                reinterpret_cast<const uint4*>(a_source + a_offsets[mt][half] + chunk_offset));
        }
    }
#pragma unroll
    for (int nt = 0; nt < NTiles; ++nt) {
        fragments.b[nt] =
            __ldg(reinterpret_cast<const uint4*>(b_source + b_offsets[nt] + chunk_offset));
    }
}

template <int MTiles, int NTiles>
__device__ __forceinline__ void
pointer_head_mma_chunk(const PointerHeadFragments<MTiles, NTiles>& fragments,
                       float (&accumulators)[MTiles][NTiles][4]) {
#pragma unroll
    for (int step = 0; step < 2; ++step) {
#pragma unroll
        for (int mt = 0; mt < MTiles; ++mt) {
            const uint4& low  = fragments.a[mt][0];
            const uint4& high = fragments.a[mt][1];
            const unsigned a0 = step == 0 ? low.x : low.z;
            const unsigned a1 = step == 0 ? high.x : high.z;
            const unsigned a2 = step == 0 ? low.y : low.w;
            const unsigned a3 = step == 0 ? high.y : high.w;
#pragma unroll
            for (int nt = 0; nt < NTiles; ++nt) {
                const uint4& column = fragments.b[nt];
                mma_bf16(accumulators[mt][nt][0], accumulators[mt][nt][1], accumulators[mt][nt][2],
                         accumulators[mt][nt][3], a0, a1, a2, a3, step == 0 ? column.x : column.z,
                         step == 0 ? column.y : column.w);
            }
        }
    }
}

// out[r*p + row] = bias[row] + sum_d weight[row*d + i] * hidden[r*d + i]. Each warp accumulates
// its chunks in increasing order; the CTA then sums the KWarps partial tiles in warp order, so the
// reduction order is fixed for a given configuration and extents.
template <class Config>
__global__ __launch_bounds__(Config::kThreads) void pointer_head_project_kernel(
    const __nv_bfloat16* __restrict__ hidden, const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias, float* __restrict__ out, std::int32_t d, std::int32_t p,
    std::int32_t r) {
    constexpr int MTiles = Config::kMTiles;
    constexpr int NTiles = Config::kNTiles;
    constexpr int KWarps = Config::kKWarps;
    constexpr int Depth  = Config::kDepth;
    __shared__ float reduce[Config::kReduceFloats];

    const int lane  = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp  = static_cast<int>(threadIdx.x) / kWarpSize;
    const int group = lane >> 2;
    const int quad  = lane & 3;

    const __nv_bfloat16* a_source = Config::kWeightIsA ? weight : hidden;
    const __nv_bfloat16* b_source = Config::kWeightIsA ? hidden : weight;
    const std::int32_t m_extent   = Config::kWeightIsA ? p : r;
    const std::int32_t n_extent   = Config::kWeightIsA ? r : p;
    const std::int32_t r0         = static_cast<std::int32_t>(blockIdx.x) * Config::kTileR;
    const std::int32_t p0         = static_cast<std::int32_t>(blockIdx.y) * Config::kTileP;
    const std::int32_t m0         = Config::kWeightIsA ? p0 : r0;
    const std::int32_t n0         = Config::kWeightIsA ? r0 : p0;

    // Rows past an extent reread the last valid row; their results are never stored.
    std::int32_t a_offsets[MTiles][2];
    std::int32_t b_offsets[NTiles];
#pragma unroll
    for (int mt = 0; mt < MTiles; ++mt) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const std::int32_t row = min(m0 + mt * 16 + half * 8 + group, m_extent - 1);
            a_offsets[mt][half]    = row * d + quad * 8;
        }
    }
#pragma unroll
    for (int nt = 0; nt < NTiles; ++nt) {
        const std::int32_t row = min(n0 + nt * 8 + group, n_extent - 1);
        b_offsets[nt]          = row * d + quad * 8;
    }

    float accumulators[MTiles][NTiles][4] = {};
    const std::int32_t chunks             = d / kPointerHeadChunk;
    const std::int32_t count = warp < chunks ? (chunks - warp + KWarps - 1) / KWarps : 0;
    const auto chunk_offset  = [warp](std::int32_t index) {
        return (warp + index * KWarps) * kPointerHeadChunk;
    };

    PointerHeadFragments<MTiles, NTiles> ring[Depth];
#pragma unroll
    for (int stage = 0; stage + 1 < Depth; ++stage) {
        if (stage < count) {
            pointer_head_load_chunk(ring[stage], a_source, b_source, a_offsets, b_offsets,
                                    chunk_offset(stage));
        }
    }
    for (std::int32_t base = 0; base < count; base += Depth) {
#pragma unroll
        for (int stage = 0; stage < Depth; ++stage) {
            const std::int32_t ahead = base + stage + Depth - 1;
            if (ahead < count) {
                pointer_head_load_chunk(ring[(stage + Depth - 1) % Depth], a_source, b_source,
                                        a_offsets, b_offsets, chunk_offset(ahead));
            }
            if (base + stage < count) { pointer_head_mma_chunk(ring[stage], accumulators); }
        }
    }

    // Scatter this warp's partial tile as [r_local][p_local] rows of the reduction buffer.
    float* partial = reduce + warp * (Config::kTileR * Config::kReducePitch);
#pragma unroll
    for (int mt = 0; mt < MTiles; ++mt) {
#pragma unroll
        for (int nt = 0; nt < NTiles; ++nt) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                const int m_local = mt * 16 + group + (element >> 1) * 8;
                const int n_local = nt * 8 + quad * 2 + (element & 1);
                const int p_local = Config::kWeightIsA ? m_local : n_local;
                const int r_local = Config::kWeightIsA ? n_local : m_local;
                partial[r_local * Config::kReducePitch + p_local] = accumulators[mt][nt][element];
            }
        }
    }
    __syncthreads();

    for (int index = static_cast<int>(threadIdx.x); index < Config::kTileP * Config::kTileR;
         index += Config::kThreads) {
        const int p_local         = index % Config::kTileP;
        const int r_local         = index / Config::kTileP;
        const std::int32_t row    = p0 + p_local;
        const std::int32_t column = r0 + r_local;
        if (row >= p || column >= r) { continue; }
        const int slot = r_local * Config::kReducePitch + p_local;
        float total    = reduce[slot];
#pragma unroll
        for (int source = 1; source < KWarps; ++source) {
            total += reduce[source * (Config::kTileR * Config::kReducePitch) + slot];
        }
        out[static_cast<std::int64_t>(column) * p + row] = total + __bfloat162float(bias[row]);
    }
}

// One CTA per question. questions[3q..3q+2] = {query column, first key column, key count}.
// Warp w scores the Batch keys starting at w*Batch, then every Warps*Batch keys, keeping the loads
// of all Batch keys in flight together; each key's dot is a fixed lane-strided FP32 accumulation
// and butterfly reduction. The softmax max, exponent sum, and normalization run over the
// question's logits in shared memory with fixed block reductions.
template <int Threads, int Batch>
__global__ __launch_bounds__(Threads) void pointer_head_score_kernel(
    const float* __restrict__ queries, const float* __restrict__ keys,
    const std::int32_t* __restrict__ questions, float* __restrict__ probabilities, std::int32_t p,
    float scale) {
    static_assert(Threads % kWarpSize == 0 && Threads <= 1024);
    constexpr int kWarps = Threads / kWarpSize;
    __shared__ float logits[kPointerHeadMaxKeys];
    __shared__ float partials[kWarps];

    const int lane                  = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp                  = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int32_t* question    = questions + 3 * static_cast<std::int64_t>(blockIdx.x);
    const std::int32_t query_column = question[0];
    const std::int32_t key_begin    = question[1];
    const std::int32_t key_count    = question[2];

    const std::int32_t vectors = p / 4;
    const float4* query =
        reinterpret_cast<const float4*>(queries + static_cast<std::int64_t>(query_column) * p);
    const float4* question_keys =
        reinterpret_cast<const float4*>(keys + static_cast<std::int64_t>(key_begin) * p);
    for (std::int32_t first = warp * Batch; first < key_count; first += kWarps * Batch) {
        // Slots past the question reread its last key; their dots are never stored.
        const float4* columns[Batch];
#pragma unroll
        for (int slot = 0; slot < Batch; ++slot) {
            columns[slot] = question_keys +
                            static_cast<std::int64_t>(min(first + slot, key_count - 1)) * vectors;
        }
        float dots[Batch] = {};
#pragma unroll 2
        for (std::int32_t vector = lane; vector < vectors; vector += kWarpSize) {
            const float4 q = __ldg(query + vector);
            float4 k[Batch];
#pragma unroll
            for (int slot = 0; slot < Batch; ++slot) { k[slot] = __ldg(columns[slot] + vector); }
#pragma unroll
            for (int slot = 0; slot < Batch; ++slot) {
                dots[slot] = fmaf(k[slot].x, q.x, dots[slot]);
                dots[slot] = fmaf(k[slot].y, q.y, dots[slot]);
                dots[slot] = fmaf(k[slot].z, q.z, dots[slot]);
                dots[slot] = fmaf(k[slot].w, q.w, dots[slot]);
            }
        }
#pragma unroll
        for (int slot = 0; slot < Batch; ++slot) {
            const float dot = warp_sum(dots[slot]);
            if (lane == 0 && first + slot < key_count) { logits[first + slot] = dot * scale; }
        }
    }
    __syncthreads();

    float maximum = -CUDART_INF_F;
    for (std::int32_t key = static_cast<std::int32_t>(threadIdx.x); key < key_count;
         key += Threads) {
        maximum = fmaxf(maximum, logits[key]);
    }
    maximum = warp_max(maximum);
    if (lane == 0) { partials[warp] = maximum; }
    __syncthreads();
    maximum = partials[0];
#pragma unroll
    for (int source = 1; source < kWarps; ++source) { maximum = fmaxf(maximum, partials[source]); }
    __syncthreads();

    float sum = 0.0F;
    for (std::int32_t key = static_cast<std::int32_t>(threadIdx.x); key < key_count;
         key += Threads) {
        const float exponent = expf(logits[key] - maximum);
        logits[key]          = exponent;
        sum += exponent;
    }
    sum = warp_sum(sum);
    if (lane == 0) { partials[warp] = sum; }
    __syncthreads();
    sum = partials[0];
#pragma unroll
    for (int source = 1; source < kWarps; ++source) { sum += partials[source]; }

    for (std::int32_t key = static_cast<std::int32_t>(threadIdx.x); key < key_count;
         key += Threads) {
        probabilities[key_begin + key] = logits[key] / sum;
    }
}

} // namespace ninfer::ops
