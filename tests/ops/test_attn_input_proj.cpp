#include "ninfer/ops/attn_input_proj.h"

#include "core/arena.h"
#include "ops/direct_bf16_weight.h"
#include "ops/input_projection_test_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::direct_bf16_weight;
using namespace ninfer::test::input_projection;

namespace {

// This criterion belongs to the complete A16 attention-input-projection Op.
constexpr ReductionCriterion kAttnInputProjA16Tolerance{2.9e-3, 4.0e-3, 4.5e-3};
constexpr ReductionCriterion kAttnInputProjA4Tolerance{0.16, 1.0 / 256.0, 0.16};
// Group-64 symmetric INT8 activations. The criterion belongs to the A8 compute
// profile, not to A16: quantizing the activation to 8 bits is a declared
// semantic boundary of that profile.
constexpr ReductionCriterion kAttnInputProjA8Tolerance{4.0e-2, 2.0e-2, 5.0e-2};
// Retain the original seven grid points while stabilizing the distribution-level A4 criterion.
constexpr std::int32_t kA4SampleRows = 31;

int verify_output(std::string_view label, const GuardedBf16Tensor& output,
                  const quantized_weight::PackedWeight& weight, std::int32_t weight_row_offset,
                  std::int32_t output_rows, const std::vector<float>& activation,
                  std::int32_t hidden, std::int32_t tokens,
                  const ReductionCriterion& criterion = kAttnInputProjA16Tolerance,
                  std::int32_t sample_count           = 7) {
    int failures = output.verify_guards(label);
    failures += output.verify_fully_written(label);
    const std::vector<double> actual =
        gather_rows(output.values(), output_rows, 0, output_rows, tokens, sample_count);
    const std::vector<double> expected = projection_oracle(
        weight, weight_row_offset, output_rows, activation, hidden, tokens, sample_count);
    failures += compare(label, actual, expected, criterion);
    return failures;
}

int run_q4_q5_case(DevicePackedWeight& query_key, DevicePackedWeight& gate_value,
                   std::int32_t tokens, ops::LinearPolicy policy = ops::LinearPolicy::A16Only) {
    constexpr std::int32_t kHidden      = 5120;
    constexpr std::int32_t kQRows       = 6144;
    constexpr std::int32_t kKvRows      = 1024;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 101U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);

    GuardedBf16Tensor query(kQRows, tokens);
    GuardedBf16Tensor gate(kQRows, tokens);
    GuardedBf16Tensor key(kKvRows, tokens);
    GuardedBf16Tensor value(kKvRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor q = query.tensor();
    Tensor g = gate.tensor();
    Tensor k = key.tensor();
    Tensor v = value.tensor();
    const bool a8 = policy == ops::LinearPolicy::AllowA8;
    if (a8) {
        const std::size_t bytes =
            ops::attn_input_proj_workspace_capacity_bytes(kHidden, policy, tokens, tokens);
        WorkspaceArena workspace(std::max<std::size_t>(bytes, 256));
        ops::attn_input_proj(x, query_key.view(), gate_value.view(), q, g, k, v, policy, workspace,
                             nullptr);
    } else {
        ops::attn_input_proj(x, query_key.view(), gate_value.view(), q, g, k, v, nullptr);
    }
    cuda_synchronize();

    const ReductionCriterion& criterion =
        a8 ? kAttnInputProjA8Tolerance : kAttnInputProjA16Tolerance;
    const std::string suffix =
        std::string(a8 ? " Q4/Q5 A8 T=" : " Q4/Q5 A16 T=") + std::to_string(tokens);
    int failures = 0;
    failures += verify_output("attn q" + suffix, query, query_key.host, 0, kQRows, activation,
                              kHidden, tokens, criterion);
    failures += verify_output("attn k" + suffix, key, query_key.host, kQRows, kKvRows, activation,
                              kHidden, tokens, criterion);
    failures += verify_output("attn gate" + suffix, gate, gate_value.host, 0, kQRows, activation,
                              kHidden, tokens, criterion);
    failures += verify_output("attn value" + suffix, value, gate_value.host, kQRows, kKvRows,
                              activation, kHidden, tokens, criterion);
    failures += verify_preserved("attn x" + suffix, device_activation, activation_bits);
    failures += query_key.verify_preserved("attn query/key" + suffix);
    failures += gate_value.verify_preserved("attn gate/value" + suffix);
    return failures;
}

// Prefix reuse replays a cached prefix by re-indexing the first uncached token
// to column 0, so the A8 route must be bitwise independent of both the width of
// the call and the column a token lands in. A route catalog that split A8 by
// token count, or a schedule chosen from T, would break this.
int verify_call_invariance(DevicePackedWeight& query_key, DevicePackedWeight& gate_value) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQRows = 6144, kKvRows = 1024;
    constexpr std::int32_t kWide = 260, kNarrow = 128;
    const std::vector<float> activation = make_bf16_activation(kHidden, kWide, 7U);
    const std::vector<std::uint16_t> bits = bf16_bits(activation);
    DeviceBuffer dev = to_device(bits);
    std::vector<std::vector<std::uint16_t>> captured;
    // Arm A: tokens 4..131 sitting at columns 4..131 of a 260-wide call.
    // Arm B: the same tokens re-indexed to columns 0..127 of a 128-wide call,
    // exactly as prefix reuse re-indexes the first uncached token to column 0.
    constexpr std::int32_t kShift = 4;
    std::vector<std::uint16_t> shifted(
        bits.begin() + static_cast<std::ptrdiff_t>(kShift) * kHidden, bits.end());
    DeviceBuffer dev_shifted = to_device(shifted);
    for (int arm = 0; arm < 2; ++arm) {
        const std::int32_t T = arm == 0 ? kWide : kNarrow;
        GuardedBf16Tensor query(kQRows, T), gate(kQRows, T), key(kKvRows, T), value(kKvRows, T);
        Tensor x(arm == 0 ? dev.p : dev_shifted.p, DType::BF16, {kHidden, T});
        Tensor q = query.tensor(), g = gate.tensor(), k = key.tensor(), v = value.tensor();
        const std::size_t bytes = ops::attn_input_proj_workspace_capacity_bytes(
            kHidden, ops::LinearPolicy::AllowA8, T, T);
        WorkspaceArena ws(std::max<std::size_t>(bytes, 256));
        ops::attn_input_proj(x, query_key.view(), gate_value.view(), q, g, k, v,
                             ops::LinearPolicy::AllowA8, ws, nullptr);
        cuda_synchronize();
        captured.push_back(query.bits());
    }
    std::size_t diff = 0;
    const std::size_t n = static_cast<std::size_t>(kQRows) * kNarrow;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t a = static_cast<std::size_t>(kQRows) * kShift + i;
        if (captured[0][a] != captured[1][i]) { ++diff; }
    }
    if (diff != 0) {
        std::cerr << "attn Q4/Q5 A8: " << diff << " of " << n
                  << " outputs changed when the same tokens were re-indexed\n";
        return 1;
    }
    return 0;
}

int run_q4_q5() {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kParent = 7168;
    DevicePackedWeight query_key(
        quantized_weight::make_patterned_weight(QType::Q4G64_F16S, kParent, kHidden, 103U));
    DevicePackedWeight gate_value(
        quantized_weight::make_patterned_weight(QType::Q5G64_F16S, kParent, kHidden, 107U));

    int failures = verify_call_invariance(query_key, gate_value);
    // A16: the small-T tensor-core route over T <= 32 (every n-tile width and a masked
    // column in each), then the grouped r32 schedule.
    for (const std::int32_t tokens : {1, 2, 3, 8, 12, 16, 17, 24, 25, 32, 33, 48}) {
        failures += run_q4_q5_case(query_key, gate_value, tokens);
    }
    // AllowA8 is one INT8 route over every T; both the exact (64-row/64-column) and
    // masked tile shapes are covered.
    for (const std::int32_t tokens : {1, 16, 17, 21, 48, 64, 128, 257}) {
        failures += run_q4_q5_case(query_key, gate_value, tokens, ops::LinearPolicy::AllowA8);
    }
    return failures;
}

std::vector<double> bf16_attention_oracle(const HostWeight& weight,
                                          std::span<const float> activation) {
    std::vector<double> result(static_cast<std::size_t>(weight.n));
    const unsigned available   = std::max(1U, std::thread::hardware_concurrency());
    const std::int32_t threads = std::min(weight.n, static_cast<std::int32_t>(available));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));
    for (std::int32_t thread = 0; thread < threads; ++thread) {
        const std::int32_t begin =
            static_cast<std::int32_t>((static_cast<std::int64_t>(weight.n) * thread) / threads);
        const std::int32_t end = static_cast<std::int32_t>(
            (static_cast<std::int64_t>(weight.n) * (thread + 1)) / threads);
        workers.emplace_back([&, begin, end] {
            for (std::int32_t row = begin; row < end; ++row) {
                result[static_cast<std::size_t>(row)] = dot_fp64(weight, row, activation);
            }
        });
    }
    for (std::thread& worker : workers) { worker.join(); }
    return result;
}

int verify_direct_output(std::string_view label, const GuardedBf16Tensor& output,
                         std::span<const double> expected) {
    int failures = output.verify_guards(label);
    failures += output.verify_fully_written(label);
    failures +=
        compare(label, output.values(), std::vector<double>(expected.begin(), expected.end()),
                kAttnInputProjA16Tolerance);
    return failures;
}

std::vector<std::int32_t> sampled_tokens(std::int32_t tokens) {
    if (tokens <= 32) {
        std::vector<std::int32_t> result(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) {
            result[static_cast<std::size_t>(token)] = token;
        }
        return result;
    }
    std::vector<std::int32_t> result{0, 1, tokens / 2, tokens - 2, tokens - 1};
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

int verify_direct_output_sampled(std::string_view label, const GuardedBf16Tensor& output,
                                 const HostWeight& weight, std::int32_t parent_row_offset,
                                 std::int32_t output_rows, const std::vector<float>& activation,
                                 std::int32_t hidden, std::int32_t tokens) {
    int failures = output.verify_guards(label);
    failures += output.verify_fully_written(label);
    const std::vector<std::int32_t> rows          = sampled_rows(output_rows);
    const std::vector<std::int32_t> token_samples = sampled_tokens(tokens);
    const std::vector<double> values              = output.values();
    std::vector<double> actual;
    std::vector<double> expected;
    actual.reserve(rows.size() * token_samples.size());
    expected.reserve(actual.capacity());
    for (const std::int32_t local_row : rows) {
        for (const std::int32_t token : token_samples) {
            actual.push_back(values[static_cast<std::size_t>(token) * output_rows + local_row]);
            expected.push_back(dot_fp64(
                weight, parent_row_offset + local_row,
                std::span<const float>(activation.data() + static_cast<std::size_t>(token) * hidden,
                                       hidden)));
        }
    }
    failures += compare(label, actual, expected, kAttnInputProjA16Tolerance);
    return failures;
}

int run_bf16_target_case(DeviceWeight& parent, std::int32_t tokens) {
    constexpr std::int32_t kHidden      = 5120;
    constexpr std::int32_t kQRows       = 6144;
    constexpr std::int32_t kKvRows      = 1024;
    constexpr std::int32_t kParentRows  = 14336;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 317U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);

    GuardedBf16Tensor query(kQRows, tokens);
    GuardedBf16Tensor gate(kQRows, tokens);
    GuardedBf16Tensor key(kKvRows, tokens);
    GuardedBf16Tensor value(kKvRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor q = query.tensor();
    Tensor g = gate.tensor();
    Tensor k = key.tensor();
    Tensor v = value.tensor();
    ops::attn_input_proj(x, parent.view(), q, g, k, v, nullptr);
    cuda_synchronize();

    constexpr std::int32_t kKeyBegin   = kQRows;
    constexpr std::int32_t kGateBegin  = kKeyBegin + kKvRows;
    constexpr std::int32_t kValueBegin = kGateBegin + kQRows;
    const std::string suffix           = " BF16 A16 T=" + std::to_string(tokens);
    int failures                       = 0;
    if (tokens == 1) {
        const std::vector<double> expected = bf16_attention_oracle(parent.host, activation);
        failures += verify_direct_output(
            "attn q" + suffix, query,
            std::span<const double>(expected.data(), static_cast<std::size_t>(kQRows)));
        failures +=
            verify_direct_output("attn k" + suffix, key,
                                 std::span<const double>(expected.data() + kKeyBegin,
                                                         static_cast<std::size_t>(kKvRows)));
        failures += verify_direct_output("attn gate" + suffix, gate,
                                         std::span<const double>(expected.data() + kGateBegin,
                                                                 static_cast<std::size_t>(kQRows)));
        failures +=
            verify_direct_output("attn value" + suffix, value,
                                 std::span<const double>(expected.data() + kValueBegin,
                                                         static_cast<std::size_t>(kKvRows)));
    } else {
        failures += verify_direct_output_sampled("attn q" + suffix, query, parent.host, 0, kQRows,
                                                 activation, kHidden, tokens);
        failures += verify_direct_output_sampled("attn k" + suffix, key, parent.host, kKeyBegin,
                                                 kKvRows, activation, kHidden, tokens);
        failures += verify_direct_output_sampled("attn gate" + suffix, gate, parent.host,
                                                 kGateBegin, kQRows, activation, kHidden, tokens);
        failures += verify_direct_output_sampled("attn value" + suffix, value, parent.host,
                                                 kValueBegin, kKvRows, activation, kHidden, tokens);
    }
    failures += verify_preserved("attn x" + suffix, device_activation, activation_bits);
    failures += parent.verify_preserved("attn parent" + suffix);
    return failures;
}

int run_bf16_target() {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kParentRows = 14336;
    DeviceWeight parent(make_patterned(kParentRows, kHidden, 313U));
    int failures = 0;
    if (ops::attn_input_proj_workspace_capacity_bytes(QType::BF16_CTRL, kParentRows, kHidden,
                                                      ops::LinearPolicy::A16Only, 1, 1024) != 0) {
        std::cerr << "BF16 attention input workspace interval is not zero-capacity\n";
        ++failures;
    }
    for (const std::int32_t tokens : {1, 2, 4, 8, 16, 17, 22, 23, 32, 33, 128, 129, 1024}) {
        failures += run_bf16_target_case(parent, tokens);
    }
    return failures;
}

int run_nvfp4_target_case(DevicePackedWeight& parent, std::int32_t tokens,
                          ops::LinearPolicy policy = ops::LinearPolicy::A16Only) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    const std::vector<float> activation =
        make_bf16_activation(kHidden, tokens, 337U + static_cast<std::uint32_t>(tokens));
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);

    GuardedBf16Tensor query(kQRows, tokens);
    GuardedBf16Tensor gate(kQRows, tokens);
    GuardedBf16Tensor key(kKvRows, tokens);
    GuardedBf16Tensor value(kKvRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor q                   = query.tensor();
    Tensor g                   = gate.tensor();
    Tensor k                   = key.tensor();
    Tensor v                   = value.tensor();
    const std::size_t capacity = ops::attn_input_proj_workspace_capacity_bytes(
        QType::NVFP4, 14336, kHidden, policy, tokens, tokens);
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    ops::attn_input_proj(x, parent.view(), q, g, k, v, policy, workspace, nullptr);
    cuda_synchronize();

    constexpr std::int32_t kKeyBegin   = kQRows;
    constexpr std::int32_t kGateBegin  = kKeyBegin + kKvRows;
    constexpr std::int32_t kValueBegin = kGateBegin + kQRows;
    int failures                       = 0;
    const bool a4                      = policy == ops::LinearPolicy::AllowA4;
    const ReductionCriterion& criterion =
        a4 ? kAttnInputProjA4Tolerance : kAttnInputProjA16Tolerance;
    const std::int32_t sample_count = a4 ? kA4SampleRows : 7;
    const std::string suffix =
        std::string(" NVFP4 ") + (a4 ? "A4" : "A16") + " T=" + std::to_string(tokens);
    failures += verify_output("attn q" + suffix, query, parent.host, 0, kQRows, activation, kHidden,
                              tokens, criterion, sample_count);
    failures += verify_output("attn k" + suffix, key, parent.host, kKeyBegin, kKvRows, activation,
                              kHidden, tokens, criterion, sample_count);
    failures += verify_output("attn gate" + suffix, gate, parent.host, kGateBegin, kQRows,
                              activation, kHidden, tokens, criterion, sample_count);
    failures += verify_output("attn value" + suffix, value, parent.host, kValueBegin, kKvRows,
                              activation, kHidden, tokens, criterion, sample_count);
    failures += verify_preserved("attn x" + suffix, device_activation, activation_bits);
    failures += parent.verify_preserved("attn parent" + suffix);
    return failures;
}

int run_nvfp4_target() {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kParentRows = 14336;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::NVFP4, kParentRows, kHidden, 331U, options));

    int failures = 0;
    for (const std::int32_t tokens : {1, 2, 4, 8, 16, 20, 32, 33}) {
        failures += run_nvfp4_target_case(parent, tokens);
    }
#ifndef NINFER_SM86
    failures += run_nvfp4_target_case(parent, 4, ops::LinearPolicy::AllowA4);
    failures += run_nvfp4_target_case(parent, 17, ops::LinearPolicy::AllowA4);
    failures += run_nvfp4_target_case(parent, 1024, ops::LinearPolicy::AllowA4);
#endif
    return failures;
}

int run_w8_target_case(DevicePackedWeight& parent, std::int32_t tokens) {
    constexpr std::int32_t kHidden      = 2048;
    constexpr std::int32_t kQRows       = 4096;
    constexpr std::int32_t kKvRows      = 512;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 201U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);

    GuardedBf16Tensor query(kQRows, tokens);
    GuardedBf16Tensor gate(kQRows, tokens);
    GuardedBf16Tensor key(kKvRows, tokens);
    GuardedBf16Tensor value(kKvRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor q = query.tensor();
    Tensor g = gate.tensor();
    Tensor k = key.tensor();
    Tensor v = value.tensor();
    ops::attn_input_proj(x, parent.view(), q, g, k, v, nullptr);
    cuda_synchronize();

    const std::string suffix = " W8 target A16 T=" + std::to_string(tokens);
    int failures             = 0;
    failures += verify_output("attn q" + suffix, query, parent.host, 0, kQRows, activation, kHidden,
                              tokens);
    failures += verify_output("attn k" + suffix, key, parent.host, kQRows, kKvRows, activation,
                              kHidden, tokens);
    failures += verify_output("attn gate" + suffix, gate, parent.host, kQRows + kKvRows, kQRows,
                              activation, kHidden, tokens);
    failures += verify_output("attn value" + suffix, value, parent.host, 2 * kQRows + kKvRows,
                              kKvRows, activation, kHidden, tokens);
    failures += verify_preserved("attn x" + suffix, device_activation, activation_bits);
    failures += parent.verify_preserved("attn parent weight" + suffix);
    return failures;
}

int run_w8_target() {
    constexpr std::int32_t kHidden = 2048;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::W8G32_F16S, 9216, kHidden, 211U));
    int failures = 0;
    for (const std::int32_t tokens : {1, 2, 17, 48, 64, 65, 129}) {
        failures += run_w8_target_case(parent, tokens);
    }
    return failures;
}

int run_w8_companion_case(DevicePackedWeight& parent, std::int32_t tokens) {
    constexpr std::int32_t kHidden      = 2048;
    constexpr std::int32_t kQRows       = 4096;
    constexpr std::int32_t kKvRows      = 1024;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 301U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);

    GuardedBf16Tensor query(kQRows, tokens);
    GuardedBf16Tensor key(kKvRows, tokens);
    GuardedBf16Tensor value(kKvRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor q = query.tensor();
    Tensor k = key.tensor();
    Tensor v = value.tensor();
    ops::attn_input_proj(x, parent.view(), q, k, v, nullptr);
    cuda_synchronize();

    const std::string suffix = " W8 companion A16 T=" + std::to_string(tokens);
    int failures             = 0;
    failures += verify_output("attn q" + suffix, query, parent.host, 0, kQRows, activation, kHidden,
                              tokens);
    failures += verify_output("attn k" + suffix, key, parent.host, kQRows, kKvRows, activation,
                              kHidden, tokens);
    failures += verify_output("attn value" + suffix, value, parent.host, kQRows + kKvRows, kKvRows,
                              activation, kHidden, tokens);
    failures += verify_preserved("attn x" + suffix, device_activation, activation_bits);
    failures += parent.verify_preserved("attn parent weight" + suffix);
    return failures;
}

int run_w8_companion() {
    constexpr std::int32_t kHidden = 2048;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::W8G32_F16S, 6144, kHidden, 307U));
    int failures = 0;
    // One public numerical case from every registered companion A16 T region.
    for (const std::int32_t tokens : {1, 2, 97, 193, 289, 321, 385, 449}) {
        failures += run_w8_companion_case(parent, tokens);
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_q4_q5();
    failures += run_bf16_target();
    failures += run_nvfp4_target();
    failures += run_w8_target();
    failures += run_w8_companion();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " attn_input_proj\n";
    return failures == 0 ? 0 : 1;
}
