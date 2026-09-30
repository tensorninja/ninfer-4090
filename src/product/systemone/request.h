#pragma once

#include "ninfer/types.h"
#include "product/systemone/render.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// The TypeSafe System One request (kev/api.py `SystemOneRequest`, served by kev/serve.py through
// FastAPI) translated into the Engine's DecisionInput. Decoding and validation reproduce what
// FastAPI answers for the same body: Python's json.loads over the raw bytes, then pydantic v2
// validation with FastAPI's 422 error shape, then kev's `to_record` rendering.
namespace ninfer::product::systemone {

// The model a request without `model` is answered under: the TypeSafe SDK's default model. kev
// itself defaults to "kev-latest"; both names serve its checkpoint.
inline constexpr std::string_view kDefaultModel = "jev-latest";

enum class QuestionType : std::uint8_t { Noul, Choice, Score };

// kev `to_record` per-question metadata that maps option probabilities back to answers.
struct QuestionMeta {
    std::string id;
    QuestionType type = QuestionType::Noul;
    // The answer keys in option order: ["false", "true"] (noul), the criteria names (choice), the
    // level indices "0".."L-1" (score).
    std::vector<std::string> keys;
    // Score only: the rendered level text of each key, in key order.
    std::vector<std::string> legend;
};

struct Request {
    std::string model;
    // The rendered state and questions, in request order.
    DecisionInput input;
    std::vector<QuestionMeta> questions;
};

// What FastAPI answers for a body kev would refuse: status 422 with `detail` the pydantic error
// list ({"type", "loc", "msg", "input"[, "ctx"]}, `loc` starting at "body"), or status 400 with
// `detail` the string FastAPI uses when the body cannot be decoded at all. The response body is
// {"detail": detail}. Error inputs are the request values; an integer beyond 64 bits appears as the
// nearest double, where FastAPI echoes its exact digits.
class RequestValidationError final : public std::runtime_error {
public:
    RequestValidationError(int status, nlohmann::ordered_json detail);

    [[nodiscard]] int status() const noexcept { return status_; }

    [[nodiscard]] const nlohmann::ordered_json& detail() const noexcept { return detail_; }

private:
    int status_;
    nlohmann::ordered_json detail_;
};

// Python json.loads(body) for a request body (UTF-8, optionally BOM-prefixed). A JSON syntax
// error is FastAPI's 422 `json_invalid` with Python's message and code-point position. Bodies
// json.loads cannot decode for another reason are FastAPI's 400: ill-formed UTF-8, an integer of
// more than 4,300 digits, or containers nested deeper than 1,000 levels (CPython's own limit is
// its C recursion limit, several thousand; kev fails rendering far below it).
//
// One divergence: text that decodes to a lone UTF-16 surrogate (a \uD800-\uDFFF escape without its
// pair, or surrogate code units encoded as UTF-8) is also refused with the 400, since it has no
// UTF-8 form. Python keeps it in the str, and kev then fails with a 500 once it reaches the
// tokenizer or the response. json.loads's UTF-16/UTF-32 detection is not reproduced either: every
// body is read as UTF-8, as RFC 8259 requires.
[[nodiscard]] Value parse_json(std::string_view body);

// The whole System One request: decoding as parse_json, pydantic validation of SystemOneRequest
// (unknown fields ignored, `model` defaulting to kDefaultModel), and kev's rendering of the state,
// instructions and options. Throws RequestValidationError.
[[nodiscard]] Request parse_request(std::string_view body);

// "noul", "choice" or "score".
[[nodiscard]] std::string_view question_type_name(QuestionType type) noexcept;

} // namespace ninfer::product::systemone
