#pragma once

#include "ninfer/ops/linear.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Q4Q5GdnInputScheduleId {
    SmallTMma,
    GroupedMixedMmaR64C128,
    Int8Jobs,
};

struct Q4Q5GdnInputProblem {
    std::int32_t input_rows;
    std::int32_t qk_rows;
    std::int32_t value_z_rows;
    std::int32_t qkv_rows;
    std::int32_t z_rows;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q4Q5GdnInputPlan {
    Q4Q5GdnInputScheduleId schedule;
    std::size_t workspace_bytes;
};

const char* q4_q5_gdn_input_schedule_name(Q4Q5GdnInputScheduleId schedule) noexcept;

bool q4_q5_gdn_input_admits(const Q4Q5GdnInputProblem& problem) noexcept;
Q4Q5GdnInputPlan q4_q5_gdn_input_resolve_plan(const Q4Q5GdnInputProblem& problem,
                                              LinearPolicy policy);

std::size_t q4_q5_gdn_input_capacity_workspace_bytes(std::int32_t min_tokens,
                                                     std::int32_t max_tokens, LinearPolicy policy);

void q4_q5_gdn_input_execute_plan(const Q4Q5GdnInputPlan& plan, const Tensor& x,
                                  const Weight& qk_weight, const Weight& value_z_weight,
                                  Tensor& qkv, Tensor& z, WorkspaceArena* ws, LinearPolicy policy,
                                  cudaStream_t stream);
void q4_q5_gdn_input_dispatch(const Tensor& x, const Weight& qk_weight,
                              const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                              WorkspaceArena* ws, LinearPolicy policy, cudaStream_t stream);

} // namespace ninfer::ops::detail
