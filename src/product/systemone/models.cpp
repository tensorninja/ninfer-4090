#include "product/systemone/models.h"

#include "product/systemone/request.h"

#include <stdexcept>

namespace ninfer::product::systemone {

std::string alias_binding(std::span<const LoraAdapterInfo> adapters,
                          std::string_view configured) {
    if (!configured.empty()) {
        for (const LoraAdapterInfo& adapter : adapters) {
            if (adapter.name == configured && adapter.kind == LoraAdapterKind::Decision) {
                return adapter.name;
            }
        }
        throw std::invalid_argument("System One default '" + std::string(configured) +
                                    "' is not a decision adapter of the LoRA pool");
    }
    const LoraAdapterInfo* only = nullptr;
    for (const LoraAdapterInfo& adapter : adapters) {
        if (adapter.kind != LoraAdapterKind::Decision) { continue; }
        if (only != nullptr) { return {}; }
        only = &adapter;
    }
    return only != nullptr ? only->name : std::string();
}

std::optional<std::string> resolve_model(std::string_view model,
                                         std::span<const LoraAdapterInfo> adapters,
                                         std::string_view binding) {
    for (const LoraAdapterInfo& adapter : adapters) {
        if (adapter.name == model && adapter.kind == LoraAdapterKind::Decision) {
            return adapter.name;
        }
    }
    if (model == kDefaultModel && !binding.empty()) { return std::string(binding); }
    return std::nullopt;
}

std::string unknown_model_detail(std::string_view model,
                                 std::span<const LoraAdapterInfo> adapters,
                                 std::string_view binding) {
    std::string served;
    for (const LoraAdapterInfo& adapter : adapters) {
        if (adapter.kind != LoraAdapterKind::Decision) { continue; }
        served += served.empty() ? "" : ", ";
        served += adapter.name;
    }
    if (!binding.empty()) {
        served += ", ";
        served += kDefaultModel;
        served += " (";
        served += binding;
        served += ')';
    }
    return "model '" + std::string(model) + "' not found; " +
           (served.empty() ? std::string("this server has no System One models")
                           : "System One models: " + served);
}

} // namespace ninfer::product::systemone
