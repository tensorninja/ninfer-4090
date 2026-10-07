#include "product/systemone/answers.h"
#include "product/decision/probability.h"

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::product::systemone {
namespace {

using Json = nlohmann::ordered_json;

void append_hex4(std::string& out, unsigned value) {
    constexpr std::string_view digits = "0123456789abcdef";
    out += "\\u";
    for (int shift = 12; shift >= 0; shift -= 4) { out += digits[(value >> shift) & 0xFU]; }
}

// json's py_encode_basestring_ascii / py_encode_basestring.
void dump_string(std::string& out, std::string_view text, bool ensure_ascii) {
    out += '"';
    for (std::size_t i = 0; i < text.size();) {
        const auto byte = static_cast<unsigned char>(text[i]);
        const char* escape = nullptr;
        switch (byte) {
        case '"': escape = "\\\""; break;
        case '\\': escape = "\\\\"; break;
        case '\b': escape = "\\b"; break;
        case '\f': escape = "\\f"; break;
        case '\n': escape = "\\n"; break;
        case '\r': escape = "\\r"; break;
        case '\t': escape = "\\t"; break;
        default: break;
        }
        if (escape != nullptr) {
            out += escape;
            ++i;
        } else if (byte < 0x20) {
            append_hex4(out, byte);
            ++i;
        } else if (byte < 0x7F || !ensure_ascii) {
            out += static_cast<char>(byte);
            ++i;
        } else {
            char32_t code_point = 0;
            if (!next_code_point(text, i, code_point)) {
                throw std::invalid_argument("python_json_dumps: a string is not valid UTF-8");
            }
            if (code_point >= 0x10000) {
                const char32_t offset = code_point - 0x10000;
                append_hex4(out, 0xD800U | (offset >> 10U));
                append_hex4(out, 0xDC00U | (offset & 0x3FFU));
            } else {
                append_hex4(out, code_point);
            }
        }
    }
    out += '"';
}

void dump(std::string& out, const Json& value, const DumpStyle& style) {
    switch (value.type()) {
    case Json::value_t::null: out += "null"; return;
    case Json::value_t::boolean: out += value.get<bool>() ? "true" : "false"; return;
    case Json::value_t::number_integer: out += std::to_string(value.get<std::int64_t>()); return;
    case Json::value_t::number_unsigned: out += std::to_string(value.get<std::uint64_t>()); return;
    case Json::value_t::number_float: {
        const double number = value.get<double>();
        if (std::isfinite(number)) {
            out += python_float_repr(number);
        } else if (!style.allow_nan) {
            throw std::domain_error("Out of range float values are not JSON compliant: " +
                                    python_float_repr(number));
        } else {
            out += std::isnan(number) ? "NaN" : number > 0.0 ? "Infinity" : "-Infinity";
        }
        return;
    }
    case Json::value_t::string:
        dump_string(out, value.get_ref<const std::string&>(), style.ensure_ascii);
        return;
    case Json::value_t::array: {
        out += '[';
        bool first = true;
        for (const Json& item : value) {
            if (!first) { out += style.item_separator; }
            first = false;
            dump(out, item, style);
        }
        out += ']';
        return;
    }
    case Json::value_t::object: {
        out += '{';
        bool first = true;
        for (const auto& [key, item] : value.items()) {
            if (!first) { out += style.item_separator; }
            first = false;
            dump_string(out, key, style.ensure_ascii);
            out += style.key_separator;
            dump(out, item, style);
        }
        out += '}';
        return;
    }
    case Json::value_t::binary:
    case Json::value_t::discarded: break;
    }
    throw std::invalid_argument("python_json_dumps: the value has no Python JSON form");
}

} // namespace

double python_round(double value, int ndigits) {
    if (!std::isfinite(value)) { return value; }
    if (ndigits < 0) { throw std::invalid_argument("python_round needs ndigits >= 0"); }
    // float.__round__'s NDIGITS_MAX: every double is exact at this many decimals.
    if (ndigits > 323) { return value; }
    std::array<char, 700> buffer{};
    const std::to_chars_result decimal = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::fixed, ndigits);
    double rounded = 0.0;
    std::from_chars(buffer.data(), decimal.ptr, rounded);
    return rounded;
}

double round_prob(double value) { return python_round(value, 4); }

nlohmann::ordered_json to_answers(std::span<const std::vector<float>> probabilities,
                                  std::span<const QuestionMeta> questions) {
    if (probabilities.size() != questions.size()) {
        throw std::invalid_argument("System One answers need one probability row per question");
    }
    Json answers = Json::object();
    for (std::size_t q = 0; q < questions.size(); ++q) {
        const QuestionMeta& meta = questions[q];
        const std::vector<double> p(probabilities[q].begin(), probabilities[q].end());
        if (p.size() != meta.keys.size() ||
            (meta.type == QuestionType::Score && meta.legend.size() != meta.keys.size())) {
            throw std::invalid_argument("System One question '" + meta.id +
                                        "' has a probability row of the wrong size");
        }
        Json answer    = Json::object();
        answer["type"] = std::string(question_type_name(meta.type));
        switch (meta.type) {
        case QuestionType::Noul: answer["noul"] = round_prob(p[1]); break;
        case QuestionType::Choice: {
            Json distribution = Json::object();
            for (std::size_t i = 0; i < p.size(); ++i) {
                distribution[meta.keys[i]] = round_prob(p[i]);
            }
            answer["choice"]        = meta.keys[decision::first_argmax(p)];
            answer["confidence"]    = round_prob(decision::choice_confidence(p));
            answer["probabilities"] = std::move(distribution);
            break;
        }
        case QuestionType::Score: {
            Json legend       = Json::object();
            Json distribution = Json::object();
            for (std::size_t i = 0; i < p.size(); ++i) {
                legend[meta.keys[i]]       = meta.legend[i];
                distribution[meta.keys[i]] = round_prob(p[i]);
            }
            answer["score"]         = round_prob(decision::weighted_score(p));
            answer["legend"]        = std::move(legend);
            answer["probabilities"] = std::move(distribution);
            answer["confidence"]    = round_prob(decision::score_confidence(p));
            break;
        }
        }
        answers[meta.id] = std::move(answer);
    }
    return answers;
}

std::string python_json_dumps(const nlohmann::ordered_json& value, const DumpStyle& style) {
    std::string out;
    dump(out, value, style);
    return out;
}

std::string response_body(std::string_view model, const nlohmann::ordered_json& answers,
                          std::uint32_t input_tokens, std::uint32_t output_tokens,
                          double latency_ms) {
    Json usage             = Json::object();
    usage["input_tokens"]  = input_tokens;
    usage["output_tokens"] = output_tokens;
    Json body              = Json::object();
    body["model"]          = std::string(model);
    body["answers"]        = answers;
    body["usage"]          = std::move(usage);
    body["latency_ms"]     = python_round(latency_ms, 1);
    return python_json_dumps(body, kFastApi);
}

std::string error_body(const nlohmann::ordered_json& detail) {
    Json body      = Json::object();
    body["detail"] = detail;
    return python_json_dumps(body, kFastApi);
}

int request_error_status(RequestErrorKind kind) noexcept {
    switch (kind) {
    case RequestErrorKind::UnknownAdapter: return 404;
    case RequestErrorKind::ContextLengthExceeded:
    case RequestErrorKind::MediaBudgetExceeded: return 422;
    case RequestErrorKind::Overloaded: return 429;
    case RequestErrorKind::QueueTimeout:
    case RequestErrorKind::Unavailable: return 503;
    case RequestErrorKind::Cancelled: return 499;
    }
    return 500;
}

} // namespace ninfer::product::systemone
