#include "serve/decision_models.h"

namespace ninfer::serve {

DecisionModels::DecisionModels(std::span<const LoraAdapterInfo> adapters) {
    for (const auto& adapter : adapters) {
        if (adapter.kind == LoraAdapterKind::Decision) { adapters_.push_back(adapter); }
    }
}

const LoraAdapterInfo* DecisionModels::find(std::string_view name) const {
    for (const auto& adapter : adapters_) {
        if (adapter.name == name) { return &adapter; }
    }
    return nullptr;
}

}
