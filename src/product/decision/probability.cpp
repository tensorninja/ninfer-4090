#include "product/decision/probability.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ninfer::product::decision {
namespace {

std::vector<double> normalize(std::span<const double> values) {
    const double total = sum(values);
    std::vector<double> normalized(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        normalized[i] = total == 0.0 ? 1.0 / static_cast<double>(values.size()) : values[i] / total;
    }
    return normalized;
}

void require_options(std::span<const double> probabilities) {
    if (probabilities.empty()) {
        throw std::invalid_argument("decision probabilities need at least one option");
    }
}

}

std::size_t first_argmax(std::span<const double> values) {
    require_options(values);
    std::size_t best = 0;
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[best]) { best = i; }
    }
    return best;
}

double sum(std::span<const double> values) {
    if (values.empty()) { return 0.0; }
    double sum          = 0.0 + values.front();
    double compensation = 0.0;
    for (const double value : values.subspan(1)) {
        const double total = sum + value;
        if (std::fabs(sum) >= std::fabs(value)) {
            compensation += (sum - total) + value;
        } else {
            compensation += (value - total) + sum;
        }
        sum = total;
    }
    if (compensation != 0.0 && std::isfinite(compensation)) { sum += compensation; }
    return sum;
}

double weighted_score(std::span<const double> probabilities) {
    require_options(probabilities);
    std::vector<double> weighted(probabilities.size());
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        weighted[i] = static_cast<double>(i) * probabilities[i];
    }
    return sum(weighted);
}

double choice_confidence(std::span<const double> probabilities) {
    require_options(probabilities);
    const std::size_t options = probabilities.size();
    if (options == 1) { return 1.0; }
    const std::vector<double> normalized = normalize(probabilities);
    const double uniform                 = 1.0 / static_cast<double>(options);
    return (normalized[first_argmax(normalized)] - uniform) / (1.0 - uniform);
}

double score_confidence(std::span<const double> probabilities) {
    require_options(probabilities);
    const std::size_t levels = probabilities.size();
    if (levels == 1) { return 1.0; }
    const std::vector<double> normalized = normalize(probabilities);
    const std::size_t mode               = first_argmax(normalized);
    const double center                  = static_cast<double>(levels - 1) / 2.0;
    std::vector<double> terms(levels);
    for (std::size_t i = 0; i < levels; ++i) {
        terms[i] = std::fabs(static_cast<double>(i) - center);
    }
    const double spread = sum(terms) / static_cast<double>(levels);
    for (std::size_t i = 0; i < levels; ++i) {
        terms[i] = normalized[i] * static_cast<double>(i > mode ? i - mode : mode - i);
    }
    const double confidence = 1.0 - sum(terms) / spread;
    return confidence > 0.0 ? confidence : 0.0;
}

}
