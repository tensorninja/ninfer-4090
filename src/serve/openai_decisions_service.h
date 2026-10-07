#pragma once

#include "product/openai_decisions/request.h"
#include "serve/decision_models.h"
#include "serve/generation_service.h"

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

struct PreparedOpenAIDecision {
    std::string model;
    std::vector<product::openai_decisions::QuestionMeta> questions;
    PreparedDecisionRequest decision;
};

struct OpenAIDecisionOutcome {
    std::string body;
    DecisionResult result;
};

class OpenAIDecisionsService {
public:
    OpenAIDecisionsService(GenerationService& generation, const DecisionModels& models);
    [[nodiscard]] PreparedOpenAIDecision prepare(std::string_view body) const;
    [[nodiscard]] OpenAIDecisionOutcome run(PreparedOpenAIDecision& prepared,
                                            std::function<bool()> is_cancelled);

private:
    GenerationService& generation_;
    const DecisionModels& models_;
};

}
