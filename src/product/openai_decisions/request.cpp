#include "product/openai_decisions/request.h"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <utility>

namespace ninfer::product::openai_decisions {
namespace {

using Json = nlohmann::ordered_json;

constexpr std::size_t kStringLimit = 1048576;
constexpr std::size_t kInputLimit  = 10485760;

[[noreturn]] void invalid(std::string_view param, std::string_view message,
                          std::string_view code = "invalid_value") {
    throw RequestValidationError(400, std::string(message), std::string(param), std::string(code));
}

std::string field(std::string_view parent, std::string_view key) {
    return parent.empty() ? std::string(key) : std::string(parent) + "." + std::string(key);
}

std::string index(std::string_view parent, std::size_t i) {
    return std::string(parent) + "[" + std::to_string(i) + "]";
}

void fields(const Json& object, std::string_view param,
            std::initializer_list<std::string_view> allowed) {
    if (!object.is_object()) { invalid(param, "Expected an object.", "invalid_type"); }
    for (const auto& [key, value] : object.items()) {
        if (std::find(allowed.begin(), allowed.end(), key) == allowed.end()) {
            invalid(field(param, key), "Unknown parameter.", "unknown_parameter");
        }
    }
}

const Json& required(const Json& object, std::string_view key, std::string_view parent) {
    const auto value = object.find(key);
    if (value == object.end()) {
        invalid(field(parent, key), "Missing required parameter.", "missing_required_parameter");
    }
    return *value;
}

const std::string& text(const Json& value, std::string_view param,
                        std::size_t limit = kStringLimit) {
    if (!value.is_string()) { invalid(param, "Expected a string.", "invalid_type"); }
    const std::string& string = value.get_ref<const std::string&>();
    std::size_t length        = 0;
    for (const unsigned char byte : string) {
        if ((byte & 0xC0U) != 0x80U && ++length > limit) {
            invalid(param, "String exceeds the maximum of " + std::to_string(limit) +
                               " Unicode characters.");
        }
    }
    return string;
}

void array(const Json& value, std::string_view param, std::size_t minimum, std::size_t maximum) {
    if (!value.is_array()) { invalid(param, "Expected an array.", "invalid_type"); }
    if (value.size() < minimum || value.size() > maximum) {
        invalid(param, "Expected between " + std::to_string(minimum) + " and " +
                           std::to_string(maximum) + " items.");
    }
}

void append_content(std::string& state, const Json& content, std::string_view param) {
    if (content.is_string()) {
        state += text(content, param, kInputLimit);
        return;
    }
    array(content, param, 0, 16384);
    for (std::size_t i = 0; i < content.size(); ++i) {
        const Json& part       = content[i];
        const std::string path = index(param, i);
        if (!part.is_object()) { invalid(path, "Expected an object.", "invalid_type"); }
        const std::string& type = text(required(part, "type", path), field(path, "type"));
        if (type == "input_image") {
            invalid(field(path, "type"), "Decisions supports text input only.",
                    "unsupported_modality");
        }
        fields(part, path, {"type", "text"});
        if (type != "input_text") { invalid(field(path, "type"), "Expected input_text."); }
        state += text(required(part, "text", path), field(path, "text"), kInputLimit);
    }
}

std::string input_text(const Json& input) {
    if (input.is_string()) { return text(input, "input", kInputLimit); }
    array(input, "input", 0, 131072);
    std::string state;
    for (std::size_t i = 0; i < input.size(); ++i) {
        const Json& message    = input[i];
        const std::string path = index("input", i);
        fields(message, path, {"role", "content", "type"});
        if (text(required(message, "role", path), field(path, "role")) != "user") {
            invalid(field(path, "role"), "Only user messages are supported.");
        }
        if (message.contains("type") && text(message["type"], field(path, "type")) != "message") {
            invalid(field(path, "type"), "Expected message.");
        }
        if (i != 0) { state += "\n\n"; }
        append_content(state, required(message, "content", path), field(path, "content"));
    }
    return state;
}

std::string option_text(std::string label, const Json& option, std::string_view path) {
    if (option.contains("description")) {
        const std::string& description = text(option["description"], field(path, "description"));
        if (!description.empty()) { label += ": " + description; }
    }
    return label;
}

void append_question(Request& request, const Json& question, std::size_t q) {
    const std::string path = index("questions", q);
    if (!question.is_object()) { invalid(path, "Expected an object.", "invalid_type"); }
    const std::string& type = text(required(question, "type", path), field(path, "type"));
    QuestionMeta meta;
    if (type == "predicate") {
        fields(question, path, {"type", "name", "instructions"});
        meta.type = QuestionType::Predicate;
    } else if (type == "choice") {
        fields(question, path, {"type", "name", "instructions", "choices"});
        meta.type = QuestionType::Choice;
    } else if (type == "score") {
        fields(question, path, {"type", "name", "instructions", "levels"});
        meta.type = QuestionType::Score;
    } else {
        invalid(field(path, "type"), "Expected predicate, choice, or score.");
    }
    if (question.contains("name")) { meta.name = text(question["name"], field(path, "name")); }
    DecisionQuestion rendered;
    rendered.instructions =
        text(required(question, "instructions", path), field(path, "instructions"));
    if (meta.type == QuestionType::Predicate) {
        rendered.options = {"no", "yes"};
    } else if (meta.type == QuestionType::Choice) {
        const std::string choices_path = field(path, "choices");
        const Json& choices            = required(question, "choices", path);
        array(choices, choices_path, 2, kMaximumDecisionOptions);
        bool has_boolean = false;
        for (std::size_t i = 0; i < choices.size(); ++i) {
            const std::string choice_path = index(choices_path, i);
            fields(choices[i], choice_path, {"value", "description"});
            const Json& value = required(choices[i], "value", choice_path);
            if (value.is_boolean()) {
                has_boolean = true;
            } else {
                static_cast<void>(text(value, field(choice_path, "value")));
            }
            meta.choices.push_back(value);
        }
        for (std::size_t i = 0; i < choices.size(); ++i) {
            std::string label =
                has_boolean ? meta.choices[i].dump() : meta.choices[i].get<std::string>();
            rendered.options.push_back(
                option_text(std::move(label), choices[i], index(choices_path, i)));
        }
    } else {
        const std::string levels_path = field(path, "levels");
        const Json& levels            = required(question, "levels", path);
        array(levels, levels_path, 2, 10);
        for (std::size_t i = 0; i < levels.size(); ++i) {
            const std::string level_path = index(levels_path, i);
            fields(levels[i], level_path, {"label", "description"});
            const std::string& label =
                text(required(levels[i], "label", level_path), field(level_path, "label"));
            meta.labels.push_back(label);
            rendered.options.push_back(option_text(label, levels[i], level_path));
        }
    }
    request.input.questions.push_back(std::move(rendered));
    request.questions.push_back(std::move(meta));
}

}

RequestValidationError::RequestValidationError(int status, std::string message, std::string param,
                                               std::string code)
    : std::runtime_error(std::move(message)), status_(status), param_(std::move(param)),
      code_(std::move(code)) {}

Request parse_request(std::string_view body) {
    Json root;
    try {
        root = Json::parse(body.begin(), body.end());
    } catch (const Json::exception&) { invalid("", "Invalid JSON body.", "invalid_json"); }
    fields(root, "", {"model", "input", "questions", "safety_identifier"});
    Request request;
    request.model = text(required(root, "model", ""), "model");
    if (root.contains("safety_identifier") && !root["safety_identifier"].is_null()) {
        static_cast<void>(text(root["safety_identifier"], "safety_identifier", 128));
    }
    request.input.state   = input_text(required(root, "input", ""));
    const Json& questions = required(root, "questions", "");
    array(questions, "questions", 1, 200);
    request.input.questions.reserve(questions.size());
    request.questions.reserve(questions.size());
    for (std::size_t q = 0; q < questions.size(); ++q) {
        append_question(request, questions[q], q);
    }
    return request;
}

}
