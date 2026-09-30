#pragma once

#include "ninfer/types.h"

#include <optional>
#include <span>
#include <string>
#include <string_view>

// System One model names over the Engine's adapter pool. A System One model is a decision adapter;
// the base weights and generative adapters are not System One models.
namespace ninfer::product::systemone {

// The adapter the SDK-default alias (kDefaultModel) answers with: `configured` when it is
// non-empty, else the pool's only decision adapter. Empty when the alias is unbound. Throws
// std::invalid_argument when `configured` is not a decision adapter of the pool.
[[nodiscard]] std::string alias_binding(std::span<const LoraAdapterInfo> adapters,
                                        std::string_view configured);

// The decision adapter a request's `model` names: a decision adapter of that name, else the
// alias's binding. Nullopt when nothing answers to the name.
[[nodiscard]] std::optional<std::string>
resolve_model(std::string_view model, std::span<const LoraAdapterInfo> adapters,
              std::string_view binding);

// The 404 detail for a `model` nothing answers to, naming what is served.
[[nodiscard]] std::string unknown_model_detail(std::string_view model,
                                               std::span<const LoraAdapterInfo> adapters,
                                               std::string_view binding);

} // namespace ninfer::product::systemone
