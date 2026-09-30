#include "ops/launcher/pointer_head.h"

#include "core/device.h"
#include "ops/kernel/pointer_head.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <class Config>
void launch_project(const Tensor& hidden, const Tensor& weight, const Tensor& bias, Tensor& out,
                    cudaStream_t stream) {
    const std::int32_t d = hidden.ne[0];
    const std::int32_t r = hidden.ne[1];
    const std::int32_t p = weight.ne[1];
    const dim3 grid(static_cast<unsigned int>((r + Config::kTileR - 1) / Config::kTileR),
                    static_cast<unsigned int>((p + Config::kTileP - 1) / Config::kTileP));
    pointer_head_project_kernel<Config><<<grid, Config::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data),
        static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<const __nv_bfloat16*>(bias.data), static_cast<float*>(out.data), d, p, r);
}

} // namespace

// Routes by column count, measured on RTX 4090 at D=5120, P=256 with the weight both L2-resident
// and evicted. Narrow calls split D across up to 32 warps per CTA to expose enough loads in
// flight; wider calls trade that split for larger register tiles that reuse each fragment.
void pointer_head_project_launch(const Tensor& hidden, const Tensor& weight, const Tensor& bias,
                                 Tensor& out, cudaStream_t stream) {
    const std::int32_t r = hidden.ne[1];
    if (r <= 32) {
        launch_project<PointerHeadProjectConfig<false, 1, 1, 32, 2>>(hidden, weight, bias, out,
                                                                     stream);
    } else if (r <= 96) {
        launch_project<PointerHeadProjectConfig<false, 1, 2, 16, 2>>(hidden, weight, bias, out,
                                                                     stream);
    } else if (r <= 256) {
        launch_project<PointerHeadProjectConfig<true, 1, 4, 16, 2>>(hidden, weight, bias, out,
                                                                    stream);
    } else if (r <= 416) {
        launch_project<PointerHeadProjectConfig<true, 2, 4, 8, 2>>(hidden, weight, bias, out,
                                                                   stream);
    } else {
        launch_project<PointerHeadProjectConfig<true, 2, 5, 8, 2>>(hidden, weight, bias, out,
                                                                   stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

// One CTA scores one question, so a wide question is a serial latency chain on one SM. When the key
// table holds at least 32 columns per question, 512-thread CTAs keep eight keys in flight per warp;
// otherwise questions are narrow and the cheaper 256-thread, one-key-per-warp CTA is faster.
void pointer_head_score_launch(const Tensor& queries, const Tensor& keys, const Tensor& questions,
                               float scale, Tensor& probabilities, cudaStream_t stream) {
    const auto* query_data    = static_cast<const float*>(queries.data);
    const auto* key_data      = static_cast<const float*>(keys.data);
    const auto* question_data = static_cast<const std::int32_t*>(questions.data);
    auto* probability_data    = static_cast<float*>(probabilities.data);
    const std::int32_t p      = queries.ne[0];
    const auto grid           = static_cast<unsigned int>(questions.ne[1]);
    if (static_cast<std::int64_t>(keys.ne[1]) >= 32 * static_cast<std::int64_t>(questions.ne[1])) {
        pointer_head_score_kernel<512, 8><<<grid, 512, 0, stream>>>(
            query_data, key_data, question_data, probability_data, p, scale);
    } else {
        pointer_head_score_kernel<256, 1><<<grid, 256, 0, stream>>>(
            query_data, key_data, question_data, probability_data, p, scale);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
