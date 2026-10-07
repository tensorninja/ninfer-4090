#include "product/openai_decisions/answers.h"
#include "product/decision/probability.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::product::openai_decisions {

std::string response_body(std::string_view model, const DecisionResult& result,
                          std::span<const QuestionMeta> questions) {
    using Json = nlohmann::ordered_json;
    if (result.probabilities.size() != questions.size()) {
        throw std::invalid_argument("Decisions answers need one probability row per question");
    }
    Json answers = Json::array();
    for (std::size_t q = 0; q < questions.size(); ++q) {
        const QuestionMeta& meta = questions[q];
        const std::vector<double> p(result.probabilities[q].begin(), result.probabilities[q].end());
        const std::size_t options = meta.type == QuestionType::Predicate ? 2
                                    : meta.type == QuestionType::Choice  ? meta.choices.size()
                                                                         : meta.labels.size();
        if (p.size() != options || p.empty()) {
            throw std::invalid_argument(
                "Decisions question has a probability row of the wrong size");
        }
        for (const double probability : p) {
            if (!std::isfinite(probability)) {
                throw std::domain_error("Decisions probability must be finite");
            }
        }
        Json answer = Json::object();
        switch (meta.type) {
        case QuestionType::Predicate:
            answer["type"] = "predicate";
            break;
        case QuestionType::Choice:
            answer["type"] = "choice";
            break;
        case QuestionType::Score:
            answer["type"] = "score";
            break;
        }
        answer["name"] = meta.name ? Json(*meta.name) : Json(nullptr);
        switch (meta.type) {
        case QuestionType::Predicate:
            answer["probability"] = p[1];
            break;
        case QuestionType::Choice: {
            answer["choice"]  = meta.choices[decision::first_argmax(p)];
            Json distribution = Json::array();
            for (std::size_t i = 0; i < p.size(); ++i) {
                distribution.push_back(Json{{"value", meta.choices[i]}, {"probability", p[i]}});
            }
            answer["probabilities"] = std::move(distribution);
            answer["confidence"]    = decision::choice_confidence(p);
            break;
        }
        case QuestionType::Score: {
            answer["score"]   = decision::weighted_score(p);
            Json distribution = Json::array();
            for (std::size_t i = 0; i < p.size(); ++i) {
                distribution.push_back(
                    Json{{"value", i}, {"label", meta.labels[i]}, {"probability", p[i]}});
            }
            answer["probabilities"] = std::move(distribution);
            answer["confidence"]    = decision::score_confidence(p);
            break;
        }
        }
        answers.push_back(std::move(answer));
    }
    const auto input_tokens = result.summary.input_tokens();
    Json usage              = {{"input_tokens", input_tokens},
                               {"input_tokens_details",
                                {{"cached_tokens", result.reused_state_tokens}, {"cache_write_tokens", 0}}},
                               {"output_tokens", 0},
                               {"output_tokens_details", {{"reasoning_tokens", 0}}},
                               {"total_tokens", input_tokens}};
    Json body               = {{"model", std::string(model)},
                               {"answers", std::move(answers)},
                               {"usage", std::move(usage)}};
    return body.dump();
}

}
