#pragma once

#include "ninfer/types.h"

#include <span>
#include <string_view>
#include <vector>

namespace ninfer::serve {

class DecisionModels {
public:
    explicit DecisionModels(std::span<const LoraAdapterInfo> adapters);

    [[nodiscard]] const LoraAdapterInfo* find(std::string_view name) const;

    [[nodiscard]] std::span<const LoraAdapterInfo> adapters() const { return adapters_; }

private:
    std::vector<LoraAdapterInfo> adapters_;
};

}
