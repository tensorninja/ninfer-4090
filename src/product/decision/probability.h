#pragma once

#include <cstddef>
#include <span>

namespace ninfer::product::decision {

[[nodiscard]] std::size_t first_argmax(std::span<const double> values);
[[nodiscard]] double sum(std::span<const double> values);
[[nodiscard]] double weighted_score(std::span<const double> probabilities);
[[nodiscard]] double choice_confidence(std::span<const double> probabilities);
[[nodiscard]] double score_confidence(std::span<const double> probabilities);

}
