#include "product/decision/probability.h"
#include "product/systemone/answers.h"
#include "product/systemone/render.h"
#include "product/systemone/request.h"

#include <nlohmann/json.hpp>

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

// Exact parity of the System One product translation with kev's Python, against golden fixtures
// computed by kev's own functions (tools/parity/qwen3_8_27b/decision.py fixtures).
namespace {

using namespace ninfer::product::systemone;
using Json = nlohmann::ordered_json;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    ++failures;
    std::cerr << message << '\n';
}

// A diagnostic JSON string literal of arbitrary bytes (bodies are deliberately not always UTF-8).
std::string json_literal(const std::string& text) {
    return Json(text).dump(-1, ' ', false, Json::error_handler_t::replace);
}

Json load_fixture(const std::string& name) {
    const std::string path = NINFER_SOURCE_DIR "/tests/fixtures/systemone/" + name;
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("cannot open " + path); }
    return Json::parse(stream);
}

// A double stored as its Python repr.
double number(const Json& text) {
    const std::string& digits = text.get_ref<const std::string&>();
    double value              = 0.0;
    const auto result = std::from_chars(digits.data(), digits.data() + digits.size(), value);
    if (result.ec != std::errc{} || result.ptr != digits.data() + digits.size()) {
        throw std::runtime_error("fixture number is not a float repr: " + digits);
    }
    return value;
}

std::vector<double> numbers(const Json& texts) {
    std::vector<double> values;
    for (const Json& text : texts) { values.push_back(number(text)); }
    return values;
}

std::string body_of(const Json& request) {
    if (request.contains("body")) { return request["body"].get<std::string>(); }
    const std::string hex = request["body_hex"].get<std::string>();
    std::string bytes;
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
        bytes += static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16));
    }
    return bytes;
}

QuestionType question_type(const std::string& name) {
    if (name == "noul") { return QuestionType::Noul; }
    if (name == "choice") { return QuestionType::Choice; }
    if (name == "score") { return QuestionType::Score; }
    throw std::runtime_error("unknown question type " + name);
}

void test_text_rules(const Json& golden) {
    const std::set<char32_t> whitespace(golden["whitespace"].begin(), golden["whitespace"].end());
    std::size_t isspace_mismatches = 0;
    for (char32_t c = 0; c < 0x110000; ++c) {
        isspace_mismatches += python_isspace(c) != whitespace.contains(c) ? 1 : 0;
    }
    check(isspace_mismatches == 0, "python_isspace differs from str.isspace() at " +
                                       std::to_string(isspace_mismatches) + " code points");

    for (const Json& item : golden["lstrip"]) {
        const std::string text = item["text"].get<std::string>();
        check(python_lstrip(text) == item["expected"].get<std::string>(),
              "lstrip mismatch for " + json_literal(text));
    }

    for (const Json& repr : golden["float_repr"]) {
        const std::string expected = repr.get<std::string>();
        const std::string actual   = python_float_repr(number(repr));
        check(actual == expected, "float repr " + actual + " != " + expected);
    }

    for (const Json& item : golden["render"]) {
        const std::string text = item["json"].get<std::string>();
        try {
            const std::string actual = render(parse_json(text), item["indent"].get<std::size_t>());
            check(actual == item["expected"].get<std::string>(),
                  "render mismatch for " + text.substr(0, 80) + ": " + json_literal(actual));
        } catch (const std::exception& error) {
            check(false, "render input rejected: " + text.substr(0, 80) + ": " + error.what());
        }
    }

    for (const Json& item : golden["option_text"]) {
        const std::string name = item["name"].get<std::string>();
        std::string actual;
        if (item.contains("description")) {
            const Value description = parse_json(item["description"].get<std::string>());
            actual                  = option_text(name, &description);
        } else {
            actual = option_text(name, nullptr);
        }
        check(actual == item["expected"].get<std::string>(),
              "option_text mismatch: " + json_literal(actual));
    }
}

void test_numerics(const Json& golden) {
    for (const Json& item : golden["round"]) {
        const std::string actual =
            python_float_repr(python_round(number(item["x"]), item["ndigits"].get<int>()));
        check(actual == item["expected"].get<std::string>(),
              "round(" + item["x"].get<std::string>() + ", " + item["ndigits"].dump() +
                  ") = " + actual + ", Python " + item["expected"].get<std::string>());
    }

    for (const Json& item : golden["sum"]) {
        const std::string actual =
            python_float_repr(ninfer::product::decision::sum(numbers(item["values"])));
        check(actual == item["expected"].get<std::string>(),
              "sum = " + actual + ", Python " + item["expected"].get<std::string>());
    }

    for (const Json& item : golden["confidence"]) {
        const std::vector<double> p = numbers(item["p"]);
        const std::string choice =
            python_float_repr(ninfer::product::decision::choice_confidence(p));
        const std::string score = python_float_repr(ninfer::product::decision::score_confidence(p));
        check(choice == item["choice"].get<std::string>(),
              "choice_confidence = " + choice + ", Python " + item["choice"].get<std::string>());
        check(score == item["score"].get<std::string>(),
              "score_confidence = " + score + ", Python " + item["score"].get<std::string>());
    }
}

void test_requests(const Json& golden) {
    std::size_t index = 0;
    for (const Json& item : golden["requests"]) {
        const std::string label = "request " + std::to_string(index++) + " (" +
                                  json_literal(body_of(item).substr(0, 60)) + ")";
        const int status        = item["status"].get<int>();
        try {
            const Request request = parse_request(body_of(item));
            if (status != 200) {
                check(false, label + ": accepted, FastAPI answered " + std::to_string(status));
                continue;
            }
            check(request.model == item["model"].get<std::string>(), label + ": model");
            check(request.input.state == item["state"].get<std::string>(), label + ": state");
            const Json& questions = item["questions"];
            const Json& meta      = item["meta"];
            check(request.input.questions.size() == questions.size() &&
                      request.questions.size() == meta.size(),
                  label + ": question count");
            for (std::size_t q = 0; q < questions.size() && q < request.questions.size(); ++q) {
                const ninfer::DecisionQuestion& question = request.input.questions[q];
                const QuestionMeta& actual               = request.questions[q];
                check(question.instructions == questions[q]["instructions"].get<std::string>(),
                      label + ": instructions of question " + std::to_string(q));
                check(question.options == questions[q]["options"].get<std::vector<std::string>>(),
                      label + ": options of question " + std::to_string(q));
                check(actual.id == meta[q]["id"].get<std::string>() &&
                          question_type_name(actual.type) == meta[q]["type"].get<std::string>() &&
                          actual.keys == meta[q]["keys"].get<std::vector<std::string>>() &&
                          actual.legend == meta[q].value("legend", std::vector<std::string>{}),
                      label + ": metadata of question " + std::to_string(q));
            }
        } catch (const RequestValidationError& error) {
            check(error.status() == status && error.detail() == item["detail"],
                  label + ": answered " + std::to_string(error.status()) + " " +
                      python_json_dumps(error.detail(), kFastApi).substr(0, 600) +
                      "\n  FastAPI " + std::to_string(status) + " " +
                      python_json_dumps(item.value("detail", Json()), kFastApi).substr(0, 600));
        }
    }
}

void test_answers(const Json& golden) {
    std::size_t index = 0;
    for (const Json& item : golden["answers"]) {
        const std::string label = "answers " + std::to_string(index++);
        std::vector<QuestionMeta> questions;
        for (const Json& meta : item["questions"]) {
            questions.push_back(QuestionMeta{
                .id     = meta["id"].get<std::string>(),
                .type   = question_type(meta["type"].get<std::string>()),
                .keys   = meta["keys"].get<std::vector<std::string>>(),
                .legend = meta.value("legend", std::vector<std::string>{}),
            });
        }
        std::vector<std::vector<float>> probabilities;
        for (const Json& row : item["probabilities"]) {
            std::vector<float>& values = probabilities.emplace_back();
            for (const double value : numbers(row)) {
                values.push_back(static_cast<float>(value));
                check(static_cast<double>(values.back()) == value,
                      label + ": fixture probability is not a float32 value");
            }
        }
        const Json answers = to_answers(probabilities, questions);
        const std::string dumped = python_json_dumps(answers);
        check(dumped == item["answers"].get<std::string>(),
              label + ": json.dumps\n  " + dumped + "\n  Python " +
                  item["answers"].get<std::string>());
        check(python_json_dumps(answers, kFastApi) == item["answers_fastapi"].get<std::string>(),
              label + ": FastAPI rendering");
        const std::string body = response_body(
            item["model"].get<std::string>(), answers, item["input_tokens"].get<std::uint32_t>(),
            item["output_tokens"].get<std::uint32_t>(), number(item["latency_ms"]));
        check(body == item["response"].get<std::string>(),
              label + ": response body\n  " + body + "\n  FastAPI " +
                  item["response"].get<std::string>());
    }

    // Starlette renders responses with allow_nan=False: a NaN answer is an error, not a body.
    bool refused = false;
    try {
        static_cast<void>(python_json_dumps(Json(std::numeric_limits<double>::quiet_NaN()),
                                            kFastApi));
    } catch (const std::domain_error&) { refused = true; }
    check(refused && python_json_dumps(Json(std::numeric_limits<double>::infinity())) ==
                         "Infinity",
          "non-finite floats must follow json.dumps' allow_nan");
}

} // namespace

int main() {
    try {
        const Json golden = load_fixture("kev_golden.json");
        test_text_rules(golden);
        test_numerics(golden);
        test_requests(golden);
        test_answers(golden);
    } catch (const std::exception& error) {
        std::cerr << "systemone product test aborted: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " System One parity check(s) failed\n";
        return 1;
    }
    return 0;
}
