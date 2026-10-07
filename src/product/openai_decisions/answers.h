#pragma once

#include "product/openai_decisions/request.h"

#include <span>
#include <string>
#include <string_view>

namespace ninfer::product::openai_decisions {

[[nodiscard]] std::string response_body(std::string_view model, const DecisionResult& result,
                                        std::span<const QuestionMeta> questions);

}
