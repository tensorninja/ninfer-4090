#include "ninfer/ops/pointer_head.h"

#include "ops/launcher/pointer_head.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kMaximumPointerRows = 65536;

bool is_rank2(const Tensor& tensor) { return tensor.ne[2] == 1 && tensor.ne[3] == 1; }

bool is_rank1(const Tensor& tensor) { return tensor.ne[1] == 1 && is_rank2(tensor); }

bool aligned16(const Tensor& tensor) {
    return (reinterpret_cast<std::uintptr_t>(tensor.data) & 15U) == 0U;
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

bool usable(const Tensor& tensor) { return tensor.data != nullptr && tensor.is_contiguous(); }

} // namespace

void pointer_head_project(const Tensor& hidden, const Tensor& weight, const Tensor& bias,
                          Tensor& out, cudaStream_t stream) {
    if (hidden.dtype != DType::BF16 || weight.dtype != DType::BF16 || bias.dtype != DType::BF16 ||
        out.dtype != DType::FP32) {
        throw std::invalid_argument(
            "pointer_head_project: hidden, weight, and bias must be BF16 and out FP32");
    }
    const std::int32_t d = hidden.ne[0];
    const std::int32_t r = hidden.ne[1];
    const std::int32_t p = weight.ne[1];
    if (!is_rank2(hidden) || !is_rank2(weight) || !is_rank1(bias) || !is_rank2(out) ||
        weight.ne[0] != d || bias.ne[0] != p || out.ne[0] != p || out.ne[1] != r) {
        throw std::invalid_argument(
            "pointer_head_project: expected hidden [D,R], weight [D,P], bias [P], out [P,R]");
    }
    constexpr auto kOffsetLimit =
        static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max());
    if (d <= 0 || (d % 32) != 0 || p <= 0 || p > kMaximumPointerRows || r <= 0 ||
        static_cast<std::int64_t>(p) * d > kOffsetLimit ||
        static_cast<std::int64_t>(r) * d > kOffsetLimit) {
        throw std::invalid_argument("pointer_head_project: unsupported D, P, or R");
    }
    if (!usable(hidden) || !usable(weight) || !usable(bias) || !usable(out)) {
        throw std::invalid_argument(
            "pointer_head_project: tensors must be contiguous and non-null");
    }
    if (!aligned16(hidden) || !aligned16(weight)) {
        throw std::invalid_argument(
            "pointer_head_project: hidden and weight must be 16-byte aligned");
    }
    if (overlaps(out, hidden) || overlaps(out, weight) || overlaps(out, bias)) {
        throw std::invalid_argument("pointer_head_project: out must not overlap an input");
    }
    detail::pointer_head_project_launch(hidden, weight, bias, out, stream);
}

void pointer_head_score(const Tensor& queries, const Tensor& keys, const Tensor& questions,
                        float scale, Tensor& probabilities, cudaStream_t stream) {
    if (queries.dtype != DType::FP32 || keys.dtype != DType::FP32 ||
        questions.dtype != DType::I32 || probabilities.dtype != DType::FP32) {
        throw std::invalid_argument(
            "pointer_head_score: queries, keys, and probabilities must be FP32 and questions I32");
    }
    const std::int32_t p = queries.ne[0];
    if (!is_rank2(queries) || !is_rank2(keys) || !is_rank2(questions) || !is_rank1(probabilities) ||
        keys.ne[0] != p || questions.ne[0] != 3 || probabilities.ne[0] != keys.ne[1]) {
        throw std::invalid_argument(
            "pointer_head_score: expected queries [P,Nq], keys [P,Nk], questions [3,Qn], "
            "probabilities [Nk]");
    }
    if (p <= 0 || (p % 4) != 0 || queries.ne[1] <= 0 || keys.ne[1] <= 0 || questions.ne[1] <= 0) {
        throw std::invalid_argument("pointer_head_score: unsupported P or empty operand");
    }
    if (!std::isfinite(scale)) {
        throw std::invalid_argument("pointer_head_score: scale must be finite");
    }
    if (!usable(queries) || !usable(keys) || !usable(questions) || !usable(probabilities)) {
        throw std::invalid_argument("pointer_head_score: tensors must be contiguous and non-null");
    }
    if (!aligned16(queries) || !aligned16(keys)) {
        throw std::invalid_argument("pointer_head_score: queries and keys must be 16-byte aligned");
    }
    if (overlaps(probabilities, queries) || overlaps(probabilities, keys) ||
        overlaps(probabilities, questions)) {
        throw std::invalid_argument("pointer_head_score: probabilities must not overlap an input");
    }
    detail::pointer_head_score_launch(queries, keys, questions, scale, probabilities, stream);
}

} // namespace ninfer::ops
