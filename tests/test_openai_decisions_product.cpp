#include "product/decision/probability.h"
#include "product/openai_decisions/answers.h"
#include "product/openai_decisions/request.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace api  = ninfer::product::openai_decisions;
namespace math = ninfer::product::decision;
using Json     = nlohmann::ordered_json;

int failures = 0;
int checks   = 0;

void check(bool condition, const std::string& message) {
    ++checks;
    if (condition) { return; }
    ++failures;
    std::cerr << message << '\n';
}

Json request_body() {
    return {{"model", "decision-adapter"},
            {"input", "evidence"},
            {"questions", Json::array({{{"type", "predicate"}, {"instructions", "Classify."}}})}};
}

Json question_set() {
    Json body                    = request_body();
    body["questions"][0]["name"] = "name";
    body["questions"].push_back(
        {{"type", "choice"},
         {"instructions", "Select."},
         {"choices", {{{"value", "a"}, {"description", "first"}}, {{"value", "b"}}}}});
    body["questions"].push_back(
        {{"type", "score"},
         {"instructions", "Rate."},
         {"levels", {{{"label", "low"}, {"description", "bottom"}}, {{"label", "high"}}}}});
    return body;
}

void rejected_body(std::string_view body, std::string_view param, std::string_view code) {
    try {
        static_cast<void>(api::parse_request(body));
        check(false, "accepted invalid request at " + std::string(param));
    } catch (const api::RequestValidationError& error) {
        check(error.status() == 400 && error.param() == param && error.code() == code,
              "wrong validation error at " + std::string(param) + ": " + error.param() + " / " +
                  error.code());
        check(!std::string_view(error.what()).empty(), "validation error needs a message");
    }
}

void rejected(const Json& body, std::string_view param, std::string_view code = "invalid_value") {
    rejected_body(body.dump(), param, code);
}

void test_rendering() {
    Json body                            = question_set();
    body["input"]                        = " \nCafé 雪 \U00010437\t";
    body["safety_identifier"]            = "DO_NOT_RENDER_OR_LOG";
    body["questions"][0]["name"]         = "duplicate";
    body["questions"][0]["instructions"] = " \nIs it true?\t";
    body["questions"][1]["name"]         = "duplicate";
    body["questions"][1]["choices"]      = {{{"value", "true"}},
                                            {{"value", "false"}, {"description", ""}},
                                            {{"value", "a\n\"b"}, {"description", "  spaced\n"}}};
    body["questions"][2]["levels"]       = {{{"label", "low"}, {"description", "bottom"}},
                                            {{"label", "low"}, {"description", ""}},
                                            {{"label", "high"}}};
    const api::Request strings           = api::parse_request(body.dump());
    check(strings.model == "decision-adapter" && strings.input.state.size() == 1 &&
              strings.input.state[0].kind == ninfer::DecisionPartKind::Text &&
              strings.input.state[0].text == body["input"].get<std::string>(),
          "model and evidence must be verbatim, without safety identifier");
    check(strings.input.questions[0].instructions == " \nIs it true?\t" &&
              strings.input.questions[0].options == std::vector<std::string>{"no", "yes"},
          "predicate instructions and no/yes option order");
    check(strings.input.questions[1].options ==
              std::vector<std::string>{"true", "false", "a\n\"b:   spaced\n"},
          "string-only choices must retain TypeSafe rendering");
    check(strings.input.questions[2].options ==
                  std::vector<std::string>{"low: bottom", "low", "high"} &&
              strings.questions[2].labels == std::vector<std::string>{"low", "low", "high"},
          "score descriptions render but metadata keeps original repeated labels");
    check(strings.questions[0].name == strings.questions[1].name &&
              !strings.questions[2].name.has_value(),
          "duplicate and missing names");
    body["questions"][1]["choices"].push_back({{"value", true}, {"description", "boolean"}});
    body["questions"][1]["choices"].push_back({{"value", false}});
    body["questions"][1]["choices"].push_back({{"value", true}});
    const api::Request mixed = api::parse_request(body.dump());
    check(mixed.input.questions[1].options ==
              std::vector<std::string>{"\"true\"", "\"false\"", "\"a\\n\\\"b\":   spaced\n",
                                       "true: boolean", "false", "true"},
          "mixed choices must use JSON scalar spelling for every value");
    check(mixed.questions[1].choices[0].is_string() && mixed.questions[1].choices[3].is_boolean() &&
              mixed.questions[1].choices[3] == mixed.questions[1].choices[5],
          "typed and repeated choices retain ordinal identity");

    const std::string evidence = body["input"].get<std::string>();
    body["input"]              = {{{"role", "user"}, {"content", evidence}}};
    check(api::parse_request(body.dump()).input.state[0].text == evidence,
          "single message equals string");
    body["input"] = {{{"role", "user"},
                      {"type", "message"},
                      {"content",
                       {{{"type", "input_text"}, {"text", "ab "}},
                        {{"type", "input_text"}, {"text", ""}},
                        {{"type", "input_text"}, {"text", " c\n"}}}}},
                     {{"role", "user"}, {"content", "next"}},
                     {{"role", "user"}, {"content", Json::array()}}};
    check(api::parse_request(body.dump()).input.state[0].text == "ab  c\n\n\nnext\n\n",
          "text parts concatenate without separators; messages join with two newlines");
    for (const Json& empty :
         {Json(""), Json::array(), Json::array({{{"role", "user"}, {"content", Json::array()}}})}) {
        body["input"] = empty;
        check(api::parse_request(body.dump()).input.state[0].text.empty(),
              "empty supported evidence");
    }
    body["input"] = std::string("a\0b", 3);
    check(api::parse_request(body.dump()).input.state[0].text == std::string("a\0b", 3),
          "escaped NUL is valid text");
    body["safety_identifier"] = nullptr;
    check(api::parse_request(body.dump()).input.state[0].text == std::string("a\0b", 3),
          "null safety identifier is discarded");
}

void test_validation() {
    for (const std::string key : {"model", "input", "questions"}) {
        Json body = request_body();
        body.erase(key);
        rejected(body, key, "missing_required_parameter");
    }
    for (const Json& value : {Json(nullptr), Json(false), Json(1), Json::array(), Json::object()}) {
        Json body     = request_body();
        body["model"] = value;
        rejected(body, "model", "invalid_type");
        body                         = request_body();
        body["questions"][0]["name"] = value;
        rejected(body, "questions[0].name", "invalid_type");
        body["questions"][0].erase("name");
        body["questions"][0]["instructions"] = value;
        rejected(body, "questions[0].instructions", "invalid_type");
    }
    for (const std::string key : {"type", "instructions"}) {
        Json body = request_body();
        body["questions"][0].erase(key);
        rejected(body, "questions[0]." + key, "missing_required_parameter");
    }
    Json body                    = request_body();
    body["questions"][0]["type"] = "noul";
    rejected(body, "questions[0].type");
    body["questions"][0]["type"] = nullptr;
    rejected(body, "questions[0].type", "invalid_type");
    body["questions"][0] = "predicate";
    rejected(body, "questions[0]", "invalid_type");
    body["questions"] = Json::object();
    rejected(body, "questions", "invalid_type");
    for (const auto& [pointer, param] : std::vector<std::pair<std::string, std::string>>{
             {"/extra", "extra"},
             {"/questions/0/criteria", "questions[0].criteria"},
             {"/questions/1/levels", "questions[1].levels"},
             {"/questions/2/choices", "questions[2].choices"},
             {"/questions/1/choices/0/extra", "questions[1].choices[0].extra"},
             {"/questions/2/levels/0/extra", "questions[2].levels[0].extra"}}) {
        body                              = question_set();
        body[Json::json_pointer(pointer)] = nullptr;
        rejected(body, param, "unknown_parameter");
    }
    for (const auto& [pointer, param] : std::vector<std::pair<std::string, std::string>>{
             {"/questions/1/choices", "questions[1].choices"},
             {"/questions/1/choices/0/value", "questions[1].choices[0].value"},
             {"/questions/2/levels", "questions[2].levels"},
             {"/questions/2/levels/0/label", "questions[2].levels[0].label"}}) {
        body = question_set();
        const Json::json_pointer path(pointer);
        body[path.parent_pointer()].erase(path.back());
        rejected(body, param, "missing_required_parameter");
    }
    for (const Json& value : {Json(nullptr), Json(1), Json(1.5), Json::array(), Json::object()}) {
        body                                        = question_set();
        body["questions"][1]["choices"][0]["value"] = value;
        rejected(body, "questions[1].choices[0].value", "invalid_type");
    }
    for (const auto& [pointer, param] : std::vector<std::pair<std::string, std::string>>{
             {"/questions/1/choices", "questions[1].choices"},
             {"/questions/1/choices/0", "questions[1].choices[0]"},
             {"/questions/1/choices/0/description", "questions[1].choices[0].description"},
             {"/questions/2/levels", "questions[2].levels"},
             {"/questions/2/levels/0", "questions[2].levels[0]"},
             {"/questions/2/levels/0/label", "questions[2].levels[0].label"},
             {"/questions/2/levels/0/description", "questions[2].levels[0].description"}}) {
        body                              = question_set();
        body[Json::json_pointer(pointer)] = nullptr;
        rejected(body, param, "invalid_type");
    }
    body                      = request_body();
    body["safety_identifier"] = 1;
    rejected(body, "safety_identifier", "invalid_type");
    for (const Json& input : {Json(nullptr), Json(3), Json::object(), Json(false)}) {
        body          = request_body();
        body["input"] = input;
        rejected(body, "input", "invalid_type");
    }
    body          = request_body();
    body["input"] = {{{"role", "user"}, {"content", {{{"type", "input_text"}, {"text", "abc"}}}}}};
    const Json message_body = body;
    for (const std::string role : {"system", "developer", "assistant", "tool"}) {
        body["input"][0]["role"] = role;
        rejected(body, "input[0].role");
    }
    for (const std::string key : {"role", "content"}) {
        body = message_body;
        body["input"][0].erase(key);
        rejected(body, "input[0]." + key, "missing_required_parameter");
    }
    for (const std::string key : {"type", "text"}) {
        body = message_body;
        body["input"][0]["content"][0].erase(key);
        rejected(body, "input[0].content[0]." + key, "missing_required_parameter");
    }
    for (const auto& [pointer, param] : std::vector<std::pair<std::string, std::string>>{
             {"/input/0/extra", "input[0].extra"},
             {"/input/0/content/0/extra", "input[0].content[0].extra"}}) {
        body                              = message_body;
        body[Json::json_pointer(pointer)] = nullptr;
        rejected(body, param, "unknown_parameter");
    }
    for (const auto& [pointer, param] : std::vector<std::pair<std::string, std::string>>{
             {"/input/0", "input[0]"},
             {"/input/0/role", "input[0].role"},
             {"/input/0/type", "input[0].type"},
             {"/input/0/content", "input[0].content"},
             {"/input/0/content/0", "input[0].content[0]"},
             {"/input/0/content/0/type", "input[0].content[0].type"},
             {"/input/0/content/0/text", "input[0].content[0].text"}}) {
        body                              = message_body;
        body[Json::json_pointer(pointer)] = nullptr;
        rejected(body, param, "invalid_type");
    }
    body                     = message_body;
    body["input"][0]["type"] = "function_call";
    rejected(body, "input[0].type");
    for (const std::string type : {"text", "input_audio", "input_file", "item_reference"}) {
        body                                   = message_body;
        body["input"][0]["content"][0]["type"] = type;
        rejected(body, "input[0].content[0].type");
    }
    for (const std::string url : {"data:image/png;base64,ignored", "https://invalid.example/a.png",
                                  "file:///not-read.png"}) {
        body                           = message_body;
        body["input"][0]["content"][0] = {{"type", "input_image"}, {"image_url", url}};
        rejected(body, "input[0].content[0].image_url", "invalid_media");
    }
    rejected(Json::array(), "", "invalid_type");
    for (const std::string_view malformed :
         {"", "{", "null true", "{\"input\":NaN}", "{\"input\":\"\\ud800\"}",
          "{\"input\":\"\\udfff\"}"}) {
        rejected_body(malformed, "", "invalid_json");
    }
    for (const std::string& bytes :
         {std::string("\x80"), std::string("\xC0\xAF"), std::string("\xE2\x82"),
          std::string("\xED\xA0\x80"), std::string("\xF4\x90\x80\x80")}) {
        rejected_body("{\"model\":\"" + bytes + "\"}", "", "invalid_json");
    }
}

void string_limit(Json body, const std::string& pointer, const std::string& param,
                  std::size_t maximum) {
    const Json::json_pointer path(pointer);
    body[path] = std::string(maximum - 1, 'a') + "\U00010437";
    static_cast<void>(api::parse_request(body.dump()));
    check(true, param + " allows its maximum Unicode character count");
    body[path].get_ref<std::string&>() += "b";
    rejected(body, param);
    body[path] = "";
    static_cast<void>(api::parse_request(body.dump()));
    check(true, param + " allows zero Unicode characters");
}

void array_limit(Json body, const std::string& pointer, const std::string& param,
                 const Json& element, std::size_t minimum, std::size_t maximum) {
    const Json::json_pointer path(pointer);
    for (const std::size_t count : {minimum, maximum}) {
        body[path] = std::vector<Json>(count, element);
        static_cast<void>(api::parse_request(body.dump()));
        check(true, param + " allows boundary count " + std::to_string(count));
    }
    body[path].push_back(element);
    rejected(body, param);
    if (minimum != 0) {
        body[path] = std::vector<Json>(minimum - 1, element);
        rejected(body, param);
    }
}

void test_limits() {
    for (const auto& [pointer, param] : std::vector<std::pair<std::string, std::string>>{
             {"/model", "model"},
             {"/questions/0/name", "questions[0].name"},
             {"/questions/0/instructions", "questions[0].instructions"},
             {"/questions/1/choices/0/value", "questions[1].choices[0].value"},
             {"/questions/1/choices/0/description", "questions[1].choices[0].description"},
             {"/questions/2/levels/0/label", "questions[2].levels[0].label"},
             {"/questions/2/levels/0/description", "questions[2].levels[0].description"}}) {
        string_limit(question_set(), pointer, param, 1048576);
    }
    string_limit(request_body(), "/safety_identifier", "safety_identifier", 128);
    string_limit(request_body(), "/input", "input", 10485760);
    Json body     = request_body();
    body["input"] = {{{"role", "user"}, {"content", ""}}};
    string_limit(body, "/input/0/content", "input[0].content", 10485760);
    body["input"][0]["content"] = {{{"type", "input_text"}, {"text", ""}}};
    string_limit(body, "/input/0/content/0/text", "input[0].content[0].text", 10485760);
    array_limit(body, "/input/0/content", "input[0].content",
                {{"type", "input_text"}, {"text", ""}}, 0, 16384);
    array_limit(request_body(), "/input", "input", {{"role", "user"}, {"content", ""}}, 0, 131072);
    array_limit(request_body(), "/questions", "questions",
                {{"type", "predicate"}, {"instructions", ""}}, 1, 200);
    array_limit(question_set(), "/questions/1/choices", "questions[1].choices",
                {{"value", "repeated"}}, 2, 255);
    array_limit(question_set(), "/questions/2/levels", "questions[2].levels",
                {{"label", "repeated"}}, 2, 10);
    body["input"][0]["content"] = {{{"type", "input_text"}, {"text", std::string(10485760, 'a')}},
                                   {{"type", "input_text"}, {"text", "b"}}};
    check(api::parse_request(body.dump()).input.state[0].text.size() == 10485761,
          "input text limit applies to each part, not concatenated evidence");
}

void test_images() {
    using ninfer::DecisionPartKind;
    using ninfer::ImageDetail;
    const Json image    = {{"type", "input_image"}, {"image_url", "data:image/png;base64,AA=="}};
    Json body           = request_body();
    body["input"]       = {{{"role", "user"},
                            {"content",
                             {{{"type", "input_text"}, {"text", "before"}},
                              image,
                              {{"type", "input_text"}, {"text", "after"}}}}},
                           {{"role", "user"}, {"content", {image}}}};
    api::Request parsed = api::parse_request(body.dump());
    check(parsed.input.images == 2 && parsed.input.state.size() == 4 &&
              parsed.input.state[0].text == "before" &&
              parsed.input.state[1].kind == DecisionPartKind::Image &&
              parsed.input.state[2].text == "after\n\n" &&
              parsed.input.state[3].param == "input[1].content[0].image_url",
          "image order, adjacent text, message separators and source paths are preserved");
    check(parsed.input.state[1].image.bytes.empty() &&
              parsed.input.state[1].image.value == image["image_url"].get<std::string>(),
          "pure parsing retains encoded source without acquiring image bytes");
    int acquired = 0;
    const auto owned =
        api::to_decision_input(std::move(parsed.input), [&](const api::SourcePart& part) {
            check(part.image.kind == ninfer::product::media_acquire::SourceKind::Data &&
                      part.image.media_type == "image/png",
                  "acquisition gets only inline image sources");
            ++acquired;
            return ninfer::OwnedMedia{.kind = ninfer::MediaKind::Image, .bytes = {0}};
        });
    check(acquired == 2 && owned.state.size() == 4 && owned.state[1].image.bytes.size() == 1 &&
              owned.state[2].text == "after\n\n" && owned.questions[0].options.size() == 2 &&
              owned.overflow == ninfer::DecisionStateOverflow::Reject,
          "acquisition materializes ordered public inputs and rejects state truncation");
    const auto text_only =
        api::to_decision_input(api::parse_request(request_body().dump()).input, {});
    check(text_only.state[0].text == "evidence" &&
              text_only.overflow == ninfer::DecisionStateOverflow::Reject,
          "OpenAI text also rejects overflow before Engine preparation");
    body["input"] = {{{"role", "user"}, {"content", {image}}}};
    for (const auto& [detail, expected] :
         std::vector<std::pair<Json, ImageDetail>>{{nullptr, ImageDetail::Auto},
                                                   {"auto", ImageDetail::Auto},
                                                   {"low", ImageDetail::Low},
                                                   {"high", ImageDetail::High},
                                                   {"original", ImageDetail::Original}}) {
        body["input"][0]["content"][0]["detail"] = detail;
        const auto input = api::to_decision_input(api::parse_request(body.dump()).input,
                                                  [](const auto&) { return ninfer::OwnedMedia{}; });
        check(input.state.back().detail == expected, "detail survives source acquisition");
    }
    for (const Json& detail :
         {Json("invalid"), Json(false), Json(4), Json::array(), Json::object()}) {
        body["input"][0]["content"][0]["detail"] = detail;
        rejected(body, "input[0].content[0].detail",
                 detail.is_string() ? "invalid_value" : "invalid_type");
    }
    for (const std::string& source :
         {"", "data:image/png,AA==", "data:video/mp4;base64,AA==", "data:image/;base64,AA==",
          "data:image/png;base64,", "data:image/png;base64,AA", "data:image/png;base64,=AAA",
          "data:image/png;base64,A===", "data:image/png;base64,!!!!"}) {
        body["input"][0]["content"][0]              = image;
        body["input"][0]["content"][0]["image_url"] = source;
        rejected(body, "input[0].content[0].image_url", "invalid_media");
    }
    for (const Json& source : {Json(nullptr), Json(false), Json(3), Json::object()}) {
        body["input"][0]["content"][0]["image_url"] = source;
        rejected(body, "input[0].content[0].image_url", "invalid_type");
    }
    body["input"][0]["content"][0] = {{"type", "input_image"}};
    rejected(body, "input[0].content[0].image_url", "missing_required_parameter");
    body["input"][0]["content"][0]            = image;
    body["input"][0]["content"][0]["file_id"] = "file-not-supported";
    rejected(body, "input[0].content[0].file_id", "unknown_parameter");
    body["input"][0]["content"][0] = {{"type", "input_video"},
                                      {"video_url", "data:video/mp4;base64,AA=="}};
    rejected(body, "input[0].content[0].type");
    body["input"][0]["content"] = std::vector<Json>(128, image);
    check(api::parse_request(body.dump()).input.images == 128, "128 image sources are accepted");
    body["input"].push_back({{"role", "user"}, {"content", {image}}});
    rejected(body, "input[1].content[0]");
    body["input"] = {{{"role", "user"}, {"content", {image}}}};
    body["input"][0]["content"][0]["image_url"] =
        "data:image/png;base64," + std::string(1048580, 'A');
    check(api::parse_request(body.dump()).input.images == 1,
          "image source strings are not capped by the default text string limit");
}

struct Oracle {
    std::size_t best;
    double score;
    double choice_confidence;
    double score_confidence;
};

Oracle oracle(std::span<const float> row) {
    const std::size_t best =
        static_cast<std::size_t>(std::max_element(row.begin(), row.end()) - row.begin());
    double total    = 0.0;
    double score    = 0.0;
    double distance = 0.0;
    for (std::size_t i = 0; i < row.size(); ++i) {
        const double probability = static_cast<double>(row[i]);
        total += probability;
        score += static_cast<double>(i) * probability;
        distance += std::abs(static_cast<double>(i) - static_cast<double>(best)) * probability;
    }
    if (row.size() == 1) { return {best, score, 1.0, 1.0}; }
    const double count   = static_cast<double>(row.size());
    const double uniform = 1.0 / count;
    const double maximum = total == 0.0 ? uniform : static_cast<double>(row[best]) / total;
    if (total == 0.0) {
        for (std::size_t i = 0; i < row.size(); ++i) {
            distance += std::abs(static_cast<double>(i) - static_cast<double>(best)) / count;
        }
    } else {
        distance /= total;
    }
    const double spread = static_cast<double>(row.size() * row.size() / 4) / count;
    return {best, score, (maximum - uniform) / (1.0 - uniform),
            std::max(0.0, 1.0 - distance / spread)};
}

void near(double actual, double expected, const std::string& label) {
    check(std::abs(actual - expected) <= 2e-14 * std::max(1.0, std::abs(expected)), label);
}

void test_probability_math() {
    const std::vector<double> cancellation{1e16, 1.0, -1e16};
    check(math::sum(cancellation) == 1.0, "sum must retain compensation");
    for (const std::size_t count : {1, 2, 3, 4, 9, 10, 64, 255}) {
        for (int pattern = 0; pattern < 6; ++pattern) {
            std::vector<float> row(count);
            for (std::size_t i = 0; i < count; ++i) {
                switch (pattern) {
                case 0:
                    row[i] = 0.0F;
                    break;
                case 1:
                    row[i] = 1.0F / static_cast<float>(count);
                    break;
                case 2:
                    row[i] = i == count / 2 ? 1.0F : 0.0F;
                    break;
                case 3:
                    row[i] = i == 0 || i == count - 1 ? 0.5F : 0.0F;
                    break;
                case 4:
                    row[i] = static_cast<float>((i * 37 + 11) % 101) / 100.0F;
                    break;
                case 5:
                    row[i] = i == count / 2 ? 1.0F : 1e-15F;
                    break;
                }
            }
            const Oracle expected = oracle(row);
            const std::vector<double> widened(row.begin(), row.end());
            check(math::first_argmax(widened) == expected.best, "first maximum option wins");
            near(math::weighted_score(widened), expected.score, "independent FP64 weighted score");
            near(math::choice_confidence(widened), expected.choice_confidence,
                 "independent FP64 choice confidence");
            near(math::score_confidence(widened), expected.score_confidence,
                 "independent FP64 score confidence");
            if (count < 2) { continue; }
            Json body         = request_body();
            body["questions"] = Json::array();
            Json choices      = Json::array();
            Json levels       = Json::array();
            for (std::size_t i = 0; i < count; ++i) {
                choices.push_back({{"value", std::to_string(i)}});
                levels.push_back({{"label", std::to_string(i)}});
            }
            body["questions"].push_back(
                {{"type", "choice"}, {"instructions", ""}, {"choices", choices}});
            if (count <= 10) {
                body["questions"].push_back(
                    {{"type", "score"}, {"instructions", ""}, {"levels", levels}});
            }
            const api::Request request = api::parse_request(body.dump());
            ninfer::DecisionResult result;
            result.probabilities = std::vector<std::vector<float>>(request.questions.size(), row);
            const Json answers   = Json::parse(
                api::response_body(request.model, result, request.questions))["answers"];
            check(answers[0]["choice"] == std::to_string(expected.best), "choice from FP32 row");
            near(answers[0]["confidence"], expected.choice_confidence,
                 "choice confidence wire math");
            for (std::size_t i = 0; i < count; ++i) {
                check(answers[0]["probabilities"][i]["probability"].get<double>() ==
                          static_cast<double>(row[i]),
                      "choice FP32 probability loses no precision");
            }
            if (count <= 10) {
                near(answers[1]["score"], expected.score, "score wire math");
                near(answers[1]["confidence"], expected.score_confidence,
                     "score confidence wire math");
                for (std::size_t i = 0; i < count; ++i) {
                    check(answers[1]["probabilities"][i]["probability"].get<double>() ==
                              static_cast<double>(row[i]),
                          "score FP32 probability loses no precision");
                }
            }
        }
    }
}

void test_answers() {
    const api::Request request = api::parse_request(R"({
        "model":"decision-adapter", "input":"evidence", "questions":[
            {"type":"predicate","instructions":""},
            {"type":"choice","name":"same","instructions":"","choices":[
                {"value":true},{"value":"true"},{"value":true},{"value":"false"}]},
            {"type":"choice","name":"same","instructions":"","choices":[
                {"value":true},{"value":"true"}]},
            {"type":"score","name":"","instructions":"","levels":[
                {"label":"low","description":"bottom"},{"label":"mid"},{"label":"low"}]}
        ]})");
    ninfer::DecisionResult result;
    result.probabilities = {
        {0.75F, 0.25F}, {0.25F, 0.5F, 0.125F, 0.125F}, {0.5F, 0.5F}, {0.25F, 0.5F, 0.25F}};
    result.summary.state_tokens  = 71;
    result.summary.branch_tokens = 29;
    result.reused_state_tokens   = 71;
    const Json expected          = Json::parse(R"({"model":"served-name","answers":[
        {"type":"predicate","name":null,"probability":0.25},
        {"type":"choice","name":"same","choice":"true","probabilities":[
            {"value":true,"probability":0.25},{"value":"true","probability":0.5},
            {"value":true,"probability":0.125},{"value":"false","probability":0.125}],
            "confidence":0.3333333333333333},
        {"type":"choice","name":"same","choice":true,"probabilities":[
            {"value":true,"probability":0.5},{"value":"true","probability":0.5}],"confidence":0.0},
        {"type":"score","name":"","score":1.0,"probabilities":[
            {"value":0,"label":"low","probability":0.25},
            {"value":1,"label":"mid","probability":0.5},
            {"value":2,"label":"low","probability":0.25}],"confidence":0.25}
        ],"usage":{"input_tokens":100,"input_tokens_details":{"cached_tokens":71,"cache_write_tokens":0},
        "output_tokens":0,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":100}})");
    check(api::response_body("served-name", result, request.questions) == expected.dump(),
          "exact ordered schema, typed values, names, labels, and cached usage");
    result.reused_state_tokens = 0;
    const Json cold = Json::parse(api::response_body("served-name", result, request.questions));
    check(cold["usage"]["input_tokens_details"]["cached_tokens"] == 0 &&
              cold["usage"]["input_tokens"] == 100 && cold["usage"]["total_tokens"] == 100,
          "cold usage counts prefill only");
    result.summary.images        = 2;
    result.summary.vision_tokens = 512;
    result.summary.state_tokens += 516;
    const Json visual = Json::parse(api::response_body("served-name", result, request.questions));
    check(visual["usage"]["input_tokens"] == 616 && visual["usage"]["total_tokens"] == 616 &&
              visual["usage"]["output_tokens"] == 0,
          "expanded state already includes vision tokens and delimiters without double counting");
    result.probabilities[0][1] = 0.123456789F;
    const Json precise = Json::parse(api::response_body("served-name", result, request.questions));
    check(precise["answers"][0]["probability"].get<double>() == static_cast<double>(0.123456789F),
          "predicate has no TypeSafe decimal rounding");
    for (const bool wrong_count : {false, true}) {
        ninfer::DecisionResult invalid = result;
        if (wrong_count) {
            invalid.probabilities.pop_back();
        } else {
            invalid.probabilities[0].pop_back();
        }
        bool threw = false;
        try {
            static_cast<void>(api::response_body("m", invalid, request.questions));
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "inconsistent Engine probability shape must not produce an answer");
    }
    result.probabilities[0][1] = std::numeric_limits<float>::quiet_NaN();
    bool threw                 = false;
    try {
        static_cast<void>(api::response_body("m", result, request.questions));
    } catch (const std::domain_error&) { threw = true; }
    check(threw, "non-finite probability must not become a null probability or fake refusal");
}

}

int main() {
    try {
        test_rendering();
        test_validation();
        test_limits();
        test_images();
        test_probability_math();
        test_answers();
    } catch (const std::exception& error) {
        std::cerr << "OpenAI Decisions product test aborted: " << error.what() << '\n';
        return 1;
    }
    std::cout << checks << " OpenAI Decisions checks, " << failures << " failed\n";
    return failures == 0 ? 0 : 1;
}
