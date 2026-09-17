#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"

#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/layout.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct RouteSpec {
    ColsSet cols;
    Q4Q5GdnInputScheduleId schedule;
};

// The small-T tensor-core route covers every decode width, T=1 included: it beat the
// independent gemv pair there (33 vs 42 us warm, 87 vs 88 us cold).
constexpr std::array<RouteSpec, 2> kA16Routes{{
    {{1, 32}, Q4Q5GdnInputScheduleId::SmallTMma},
    {{33, kAnyCols}, Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128},
}};

// AllowA8 replaces the prefill interval with the INT8 route. This is the largest
// single dense consumer in the measured 100K prefill profile - 11.9% of kernel
// time across 48 layers - and the BF16 grouped route reaches 116.8 TFLOP/s
// against INT8 jobs at 235.0/212.5. Group-64 activation quantization is a
// declared semantic boundary and is reachable only through AllowA8. See
// docs/maintainer/op-development.md section 2.1.
// One route over every T; see the attention catalog for why A8 must not be
// split by token count.
constexpr std::array<RouteSpec, 1> kA8Routes{{
    {{1, kAnyCols}, Q4Q5GdnInputScheduleId::Int8Jobs},
}};

template <std::size_t N>
constexpr bool catalog_is_closed(const std::array<RouteSpec, N>& routes) noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return routes.back().cols.last == kAnyCols &&
           expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(kA16Routes) && catalog_is_closed(kA8Routes),
              "GDN input routes must be exact and closed");

template <class Visit>
auto visit_routes(LinearPolicy policy, Visit&& visit) {
    if (policy == LinearPolicy::AllowA8) { return visit(kA8Routes); }
    return visit(kA16Routes);
}

template <class Allocator>
Int8ProjWorkspace allocate_int8_workspace(Allocator& allocator, std::int32_t k, std::int32_t cols) {
    const std::int32_t padded_k = int8_proj_padded_k(k);
    const std::int32_t groups   = padded_k / 64;
    const std::int32_t tile     = int8_proj_token_tile(cols);
    Tensor codes                = allocator.alloc(DType::I8, {padded_k, tile});
    Tensor scales               = allocator.alloc(DType::FP32, {groups, tile});
    return {static_cast<std::int8_t*>(codes.data), static_cast<float*>(scales.data)};
}

std::size_t int8_workspace_bytes(std::int32_t k, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_int8_workspace(layout, k, cols);
    return layout.peak_bytes(1);
}

template <class Allocator>
SmallTMmaWorkspace allocate_small_t_workspace(Allocator& allocator, std::int32_t cols) {
    constexpr std::int32_t kTiles = (4096 + 12288) / SmallTMmaShape::kRows;
    return allocate_small_t_mma_workspace(allocator, small_t_mma_tile_cols(cols), kTiles,
                                          q4_q5_gdn_input_small_t_split_k().splits);
}

std::size_t small_t_workspace_bytes(std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_small_t_workspace(layout, cols);
    return layout.peak_bytes(1);
}

bool supported_shape(const Q4Q5GdnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.qk_rows == 4096 && problem.value_z_rows == 12288 &&
           problem.qkv_rows == 10240 && problem.z_rows == 6144 && problem.padded_k == 5120;
}

} // namespace

const char* q4_q5_gdn_input_schedule_name(Q4Q5GdnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5GdnInputScheduleId::SmallTMma:
        return "gdn_input_proj.q4_q5.small_t.mma.r64.store";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r64.c128";
    case Q4Q5GdnInputScheduleId::Int8Jobs:
        return "gdn_input_proj.q4_q5.int8.jobs";
    }
    return "gdn_input_proj.q4_q5.unknown";
}

bool q4_q5_gdn_input_admits(const Q4Q5GdnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4Q5GdnInputPlan q4_q5_gdn_input_resolve_plan(const Q4Q5GdnInputProblem& problem,
                                              LinearPolicy policy) {
    if (!q4_q5_gdn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 GDN input: exact problem or column count is not admitted");
    }

    return visit_routes(policy, [&](const auto& routes) -> Q4Q5GdnInputPlan {
        for (const RouteSpec& route : routes) {
            if (!route.cols.contains(problem.cols)) { continue; }
            if (route.schedule == Q4Q5GdnInputScheduleId::Int8Jobs) {
                return {route.schedule, int8_workspace_bytes(problem.input_rows, problem.cols)};
            }
            if (route.schedule == Q4Q5GdnInputScheduleId::SmallTMma) {
                return {route.schedule, small_t_workspace_bytes(problem.cols)};
            }
            return {route.schedule, 0};
        }
        throw std::logic_error("Q4/Q5 GDN input: admitted problem has no covering route");
    });
}

std::size_t q4_q5_gdn_input_capacity_workspace_bytes(std::int32_t min_tokens,
                                                     std::int32_t max_tokens, LinearPolicy policy) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("Q4/Q5 GDN input: invalid token interval");
    }
    // Every route's workspace grows with the token count inside its interval, so the
    // interval's high-water mark is at the last column of each intersecting route.
    std::size_t bytes = 0;
    visit_routes(policy, [&](const auto& routes) {
        for (const RouteSpec& route : routes) {
            if (route.cols.last < min_tokens || route.cols.first > max_tokens) { continue; }
            const std::int32_t cols = std::min(route.cols.last, max_tokens);
            const Q4Q5GdnInputProblem problem{5120, 4096, 12288, 10240, 6144, 5120, cols};
            bytes = std::max(bytes, q4_q5_gdn_input_resolve_plan(problem, policy).workspace_bytes);
        }
    });
    return bytes;
}

void q4_q5_gdn_input_execute_plan(const Q4Q5GdnInputPlan& plan, const Tensor& x,
                                  const Weight& qk_weight, const Weight& value_z_weight,
                                  Tensor& qkv, Tensor& z, WorkspaceArena* ws, LinearPolicy policy,
                                  cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1]};
    const Q4Q5GdnInputPlan resolved = q4_q5_gdn_input_resolve_plan(problem, policy);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("Q4/Q5 GDN input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5GdnInputScheduleId::Int8Jobs: {
        if (ws == nullptr) {
            throw std::invalid_argument("Q4/Q5 GDN input: INT8 route requires a workspace");
        }
        auto scratch_scope              = ws->scope();
        const Int8ProjWorkspace scratch = allocate_int8_workspace(*ws, problem.input_rows,
                                                                  problem.cols);
        q4_q5_gdn_input_int8_launch(x, qk_weight, value_z_weight, qkv, z, scratch, stream);
        return;
    }
    case Q4Q5GdnInputScheduleId::SmallTMma: {
        Tensor qk    = qkv.slice(0, 0, problem.qk_rows);
        Tensor value = qkv.slice(0, problem.qk_rows, problem.z_rows);
        if (plan.workspace_bytes == 0) {
            q4_q5_gdn_input_small_t_mma_launch(x, qk_weight, value_z_weight, qk, value, z, {},
                                               stream);
            return;
        }
        if (ws == nullptr) {
            throw std::invalid_argument("Q4/Q5 GDN input: small-T route requires a workspace");
        }
        auto scratch_scope               = ws->scope();
        const SmallTMmaWorkspace scratch = allocate_small_t_workspace(*ws, problem.cols);
        q4_q5_gdn_input_small_t_mma_launch(x, qk_weight, value_z_weight, qk, value, z, scratch,
                                           stream);
        return;
    }
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
        q4_q5_gdn_input_grouped_mma_launch(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    }
    throw std::logic_error("Q4/Q5 GDN input: unknown schedule");
}

void q4_q5_gdn_input_dispatch(const Tensor& x, const Weight& qk_weight,
                              const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                              WorkspaceArena* ws, LinearPolicy policy, cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1]};
    const Q4Q5GdnInputPlan plan = q4_q5_gdn_input_resolve_plan(problem, policy);
    q4_q5_gdn_input_execute_plan(plan, x, qk_weight, value_z_weight, qkv, z, ws, policy, stream);
}

} // namespace ninfer::ops::detail
