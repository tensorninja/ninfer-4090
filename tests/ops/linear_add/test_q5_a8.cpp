#include "ops/linear_add/linear_add_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using ninfer::test::linear_add::ActivationCompute;
using ninfer::test::linear_add::ShapeCase;
using ninfer::test::linear_add::WeightFormat;

int q5_a8_conformance() {
    // Same registered region boundaries as A16: AllowA8 replaces only the widest
    // interval, so every start below 129 must still select the shared A16 route
    // and the crossing at 128/129/130 must select the INT8 one. The interiors
    // carry T past the boundary in both the masked (129, 130, 192) and exact
    // (256, 512) tile shapes of the INT8 schedule.
    constexpr std::array<std::int32_t, 4> kK6144RouteStarts{2, 33, 49, 129};
    constexpr std::array<std::int32_t, 9> kK6144RouteInteriors{1, 8, 16, 24, 40, 96, 192, 256, 512};

    int failures = 0;
    failures += ninfer::test::linear_add::run_shape(
        "Q5_A8 LinearAdd", WeightFormat::Q5G64F16S, ActivationCompute::A8,
        ShapeCase{5120, 6144, 401U, kK6144RouteStarts, kK6144RouteInteriors});
    constexpr std::array<std::int32_t, 4> kK17408RouteStarts{2, 33, 49, 129};
    constexpr std::array<std::int32_t, 9> kK17408RouteInteriors{1, 8, 16, 24, 40, 96, 192, 256, 512};
    failures += ninfer::test::linear_add::run_shape(
        "Q5_A8 LinearAdd", WeightFormat::Q5G64F16S, ActivationCompute::A8,
        ShapeCase{5120, 17408, 409U, kK17408RouteStarts, kK17408RouteInteriors});
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear_add::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = q5_a8_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q5_A8 LinearAdd\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q5_A8 LinearAdd: " << error.what() << '\n';
        return 1;
    }
}
