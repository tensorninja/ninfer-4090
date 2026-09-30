#include "ops/linear_attention/gated_delta_net/chunked/launch.h"
#include "ops/linear_attention/gated_delta_net/chunked/state_passing.cuh"

#include <algorithm>

namespace ninfer::ops::detail::gated_delta_net::chunked {
namespace {

namespace kernel = state_passing;

template <int NStrip, bool Segmented>
cudaError_t launch_fixed(const state_passing_config& cfg, head_map qk_map, int NT, int rows) {
    using D = kernel::kernel_dims<NStrip>;
    constexpr int smem_bytes =
        kernel::smem_layout<NStrip>::SMEM_FLOATS * static_cast<int>(sizeof(float));
    const auto kernel_fn = kernel::state_passing_kernel<NStrip, Segmented>;

    cudaError_t err =
        cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (err != cudaSuccess) { return err; }

    const dim3 grid(static_cast<unsigned>(static_cast<std::int64_t>(cfg.H_v) * D::D_STRIPS),
                    static_cast<unsigned>(rows), 1);
    const dim3 block(D::THREADS, 1, 1);

    kernel_fn<<<grid, block, smem_bytes, cfg.stream>>>(cfg.W, cfg.U, cfg.k, cfg.g_cumsum,
                                                       cfg.state_in, cfg.v_new, cfg.h_chunk,
                                                       cfg.state_out, qk_map, NT, cfg.segments);
    return cudaGetLastError();
}

template <int NStrip>
cudaError_t launch_route(const state_passing_config& cfg, head_map qk_map, int NT, int rows) {
    if (cfg.segments.columns != nullptr) {
        return launch_fixed<NStrip, true>(cfg, qk_map, NT, rows);
    }
    return launch_fixed<NStrip, false>(cfg, qk_map, NT, rows);
}

} // namespace

cudaError_t launch_state_passing(const state_passing_config& cfg) {
    stage_validator v{"launch_state_passing", cfg.H_qk, cfg.H_v, cfg.L};
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_shape());
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_chunk_slots(cfg.segments));
    // A segmented call needs chunk-end state storage only when a run can have a tail.
    const bool needs_state_out =
        cfg.segments.columns == nullptr || segmented_tail_state_slots(cfg.L) > 0;
    if (cfg.W == nullptr || cfg.U == nullptr || cfg.k == nullptr || cfg.g_cumsum == nullptr ||
        cfg.state_in == nullptr || cfg.v_new == nullptr || cfg.h_chunk == nullptr ||
        (needs_state_out && cfg.state_out == nullptr)) {
        return cudaErrorInvalidValue;
    }

    const auto qk_map     = head_map::of((int)cfg.H_qk, (int)cfg.H_v);
    const std::int64_t NT = cfg.L / BT;
    // A segmented call has at most min(S, NT) chunk runs; see segmented_row_run.
    const std::int64_t rows =
        cfg.segments.columns != nullptr ? std::min<std::int64_t>(cfg.segments.count, NT) : 1;
    if (cfg.H_v >= 48) {
        NINFER_GATED_DELTA_NET_PROPAGATE(v.check_grid(
            static_cast<std::int64_t>(cfg.H_v) * kernel::kernel_dims<16>::D_STRIPS, rows));
        return launch_route<16>(cfg, qk_map, static_cast<int>(NT), static_cast<int>(rows));
    }
    NINFER_GATED_DELTA_NET_PROPAGATE(
        v.check_grid(static_cast<std::int64_t>(cfg.H_v) * kernel::kernel_dims<32>::D_STRIPS, rows));
    return launch_route<32>(cfg, qk_map, static_cast<int>(NT), static_cast<int>(rows));
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked
