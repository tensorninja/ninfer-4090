#include "ops/linear_attention/gated_delta_net/launch.h"

#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/chunked/launch.h"

#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include <new>

namespace ninfer::ops::detail::gated_delta_net {
namespace {

void launch_stages(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                   const Tensor& beta, float scale, const float* state_in, float* state_out,
                   segment_table segments, Tensor& out, void* workspace,
                   std::size_t workspace_bytes, cudaStream_t stream) {
    const auto layout = chunked::compute_workspace_layout(v.ne[1], q.ne[2]);
    if (workspace == nullptr || workspace_bytes < layout.total_bytes) { throw std::bad_alloc(); }

    const DeviceSpan backing{workspace, workspace_bytes};
    const Tensor g_cumsum = layout.g_cumsum.bind(backing);
    const Tensor W        = layout.W.bind(backing);
    const Tensor U        = layout.U.bind(backing);
    const Tensor v_new    = layout.v_new.bind(backing);
    const Tensor h_chunk  = layout.h_chunk.bind(backing);

    chunked::prepare_wy_wu_config prepare{};
    prepare.H_qk         = q.ne[1];
    prepare.H_v          = v.ne[1];
    prepare.L            = q.ne[2];
    prepare.segments     = segments;
    prepare.k            = static_cast<const __nv_bfloat16*>(k.data);
    prepare.v            = static_cast<const __nv_bfloat16*>(v.data);
    prepare.g_in         = static_cast<const float*>(g.data);
    prepare.beta         = static_cast<const float*>(beta.data);
    prepare.W            = static_cast<__nv_bfloat16*>(W.data);
    prepare.U            = static_cast<__nv_bfloat16*>(U.data);
    prepare.g_cumsum_out = static_cast<float*>(g_cumsum.data);
    prepare.stream       = stream;
    CUDA_CHECK(chunked::launch_prepare_wy_wu(prepare));

    chunked::state_passing_config state{};
    state.H_qk      = q.ne[1];
    state.H_v       = v.ne[1];
    state.L         = q.ne[2];
    state.segments  = segments;
    state.W         = static_cast<const __nv_bfloat16*>(W.data);
    state.U         = static_cast<const __nv_bfloat16*>(U.data);
    state.k         = static_cast<const __nv_bfloat16*>(k.data);
    state.g_cumsum  = static_cast<const float*>(g_cumsum.data);
    state.state_in  = state_in;
    state.v_new     = static_cast<__nv_bfloat16*>(v_new.data);
    state.h_chunk   = static_cast<__nv_bfloat16*>(h_chunk.data);
    state.state_out = state_out;
    state.stream    = stream;
    CUDA_CHECK(chunked::launch_state_passing(state));

    chunked::chunk_output_config output{};
    output.H_qk     = q.ne[1];
    output.H_v      = v.ne[1];
    output.L        = q.ne[2];
    output.segments = segments;
    output.q        = static_cast<const __nv_bfloat16*>(q.data);
    output.k        = static_cast<const __nv_bfloat16*>(k.data);
    output.v_new    = static_cast<const __nv_bfloat16*>(v_new.data);
    output.g_cumsum = static_cast<const float*>(g_cumsum.data);
    output.h_chunk  = static_cast<const __nv_bfloat16*>(h_chunk.data);
    output.attn_out = static_cast<__nv_bfloat16*>(out.data);
    output.scale    = scale;
    output.stream   = stream;
    CUDA_CHECK(chunked::launch_output(output));
}

} // namespace

std::size_t chunked_workspace_bytes(std::int32_t value_heads, std::int32_t tokens) {
    if (tokens <= 0) { return 0; }
    return chunked::workspace_bytes(value_heads, tokens);
}

void launch_chunked(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                    const Tensor& beta, float scale, const Tensor& ssm_state_in,
                    Tensor& ssm_state_out, Tensor& out, void* workspace,
                    std::size_t workspace_bytes, cudaStream_t stream) {
    launch_stages(q, k, v, g, beta, scale, static_cast<const float*>(ssm_state_in.data),
                  static_cast<float*>(ssm_state_out.data), segment_table{}, out, workspace,
                  workspace_bytes, stream);
}

void launch_chunked_segmented(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                              const Tensor& beta, float scale, const Tensor& ssm_state,
                              segment_table segments, float* chunk_states, Tensor& out,
                              void* workspace, std::size_t workspace_bytes, cudaStream_t stream) {
    launch_stages(q, k, v, g, beta, scale, static_cast<const float*>(ssm_state.data), chunk_states,
                  segments, out, workspace, workspace_bytes, stream);
}

} // namespace ninfer::ops::detail::gated_delta_net
