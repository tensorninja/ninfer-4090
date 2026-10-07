#pragma once

#include "ninfer/types.h"
#include "product/media_acquire/source.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
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

struct SourcePart {
    DecisionPartKind kind = DecisionPartKind::Text;
    std::string text;
    media_acquire::Source image;
    ImageDetail detail = ImageDetail::Auto;
    std::string param;
};

struct Input {
    std::vector<SourcePart> state;
    std::vector<DecisionQuestion> questions;
    std::size_t images = 0;
};

struct Request {
    std::string model;
    Input input;
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
[[nodiscard]] DecisionInput
to_decision_input(Input input, const std::function<OwnedMedia(const SourcePart&)>& acquire);
}
