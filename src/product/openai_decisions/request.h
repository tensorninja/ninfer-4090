#pragma once

#include "ninfer/types.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::product::openai_decisions {

enum class QuestionType : std::uint8_t { Predicate, Choice, Score };

struct QuestionMeta {
    QuestionType type = QuestionType::Predicate;
    std::optional<std::string> name;
    std::vector<nlohmann::ordered_json> choices;
    std::vector<std::string> labels;
};

struct Request {
    std::string model;
    DecisionInput input;
    std::vector<QuestionMeta> questions;
};

class RequestValidationError final : public std::runtime_error {
public:
    RequestValidationError(int status, std::string message, std::string param, std::string code);

    [[nodiscard]] int status() const noexcept { return status_; }

    [[nodiscard]] const std::string& param() const noexcept { return param_; }

    [[nodiscard]] const std::string& code() const noexcept { return code_; }

private:
    int status_;
    std::string param_;
    std::string code_;
};

[[nodiscard]] Request parse_request(std::string_view body);

}
