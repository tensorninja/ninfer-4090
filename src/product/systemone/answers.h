#pragma once

#include "product/systemone/request.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// kev's System One answers (kev/api.py `to_answers` and the confidence formulas it mirrors from
// TypeSafe's system-one-adapter) and the Python JSON text kev sends and bills them as. Every result
// equals CPython 3.12+ bit for bit: the probabilities are the float32 values the pointer head
// produced, widened to double exactly as `tensor.tolist()` does.
namespace ninfer::product::systemone {

// Python round(x, ndigits) for a float and ndigits >= 0: x correctly rounded to ndigits decimals,
// exact ties to even, then read back as the nearest double. Non-finite values round to themselves.
[[nodiscard]] double python_round(double value, int ndigits);

// kev round_prob: round(x, 4).
[[nodiscard]] double round_prob(double value);

// kev to_answers: per question, in request order, its answer object keyed by question id. Each
// probability row has one entry per key of its question. Throws std::invalid_argument otherwise.
[[nodiscard]] nlohmann::ordered_json to_answers(std::span<const std::vector<float>> probabilities,
                                                std::span<const QuestionMeta> questions);

// The json.dumps options kev's text depends on (indent is always None).
struct DumpStyle {
    bool ensure_ascii = true;
    bool allow_nan    = true;
    std::string_view item_separator;
    std::string_view key_separator;
};

// json.dumps(value) defaults: the text kev tokenizes for usage.output_tokens.
inline constexpr DumpStyle kPythonDefault{
    .ensure_ascii = true, .allow_nan = true, .item_separator = ", ", .key_separator = ": "};
// FastAPI's JSONResponse rendering of a response body.
inline constexpr DumpStyle kFastApi{
    .ensure_ascii = false, .allow_nan = false, .item_separator = ",", .key_separator = ":"};

// Python json.dumps over null, booleans, integers, doubles (repr), UTF-8 strings, arrays and
// objects in insertion order. A non-finite double is NaN/Infinity/-Infinity when allowed, else
// std::domain_error as Python's ValueError.
[[nodiscard]] std::string python_json_dumps(const nlohmann::ordered_json& value,
                                            const DumpStyle& style = kPythonDefault);

// The /v1/systemone response body kev sends: {"model", "answers", "usage": {"input_tokens",
// "output_tokens"}, "latency_ms"} with latency_ms = round(latency_ms, 1), rendered as FastAPI does.
[[nodiscard]] std::string response_body(std::string_view model,
                                        const nlohmann::ordered_json& answers,
                                        std::uint32_t input_tokens, std::uint32_t output_tokens,
                                        double latency_ms);

// A failed request's body {"detail": detail}, rendered as FastAPI renders an HTTPException.
[[nodiscard]] std::string error_body(const nlohmann::ordered_json& detail);

// The status a failed decision is answered with, its detail being the error's message: 404 for
// an adapter that is not a decision adapter of the pool, 422 for a decision beyond a budget, 429
// when the pending queue is full, 503 for a pending timeout or an engine that cannot serve
// decisions, 499 for a request whose client went away.
[[nodiscard]] int request_error_status(RequestErrorKind kind) noexcept;

} // namespace ninfer::product::systemone
