#include "product/systemone/request.h"

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::product::systemone {
namespace {

using Json = nlohmann::ordered_json;

// CPython's default sys.get_int_max_str_digits(): int() of a longer literal raises the ValueError
// that json.loads lets through and FastAPI answers 400.
constexpr std::size_t kMaximumIntegerDigits = 4300;
// Deeper containers are refused like CPython's json RecursionError (also a FastAPI 400); the bound
// keeps every recursive pass over a value well inside a thread stack.
constexpr int kMaximumNesting = 1000;
constexpr std::string_view kUndecodableBody = "There was an error parsing the body";
// An object with this many entries indexes its keys instead of scanning them for duplicates.
constexpr std::size_t kIndexedObjectEntries = 16;

constexpr std::string_view kFieldRequired  = "Field required";
constexpr std::string_view kNotAnObject    = "Input should be a valid dictionary or object to "
                                             "extract fields from";
constexpr std::string_view kNotADictionary = "Input should be a valid dictionary";

// Python's JSONDecodeError: its message and the byte offset of its position.
struct SyntaxError {
    std::string_view message;
    std::size_t offset = 0;
};

// json.loads failed with another exception, which FastAPI answers 400.
struct UndecodableBody {};

// bytes.decode("utf-8", "surrogatepass"): well-formed UTF-8 plus encoded surrogates. Returns
// whether the text holds an encoded surrogate.
bool validate_utf8(std::string_view text) {
    bool surrogate = false;
    for (std::size_t i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i]);
        if (lead < 0x80) {
            ++i;
            continue;
        }
        std::size_t length = 0;
        unsigned char low  = 0x80;
        unsigned char high = 0xBF;
        if (lead >= 0xC2 && lead <= 0xDF) {
            length = 2;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            length = 3;
            if (lead == 0xE0) { low = 0xA0; }
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            length = 4;
            if (lead == 0xF0) { low = 0x90; }
            if (lead == 0xF4) { high = 0x8F; }
        } else {
            throw UndecodableBody{};
        }
        if (text.size() - i < length) { throw UndecodableBody{}; }
        for (std::size_t k = 1; k < length; ++k) {
            const auto byte = static_cast<unsigned char>(text[i + k]);
            if (byte < (k == 1 ? low : 0x80) || byte > (k == 1 ? high : 0xBF)) {
                throw UndecodableBody{};
            }
        }
        surrogate = surrogate || (lead == 0xED && static_cast<unsigned char>(text[i + 1]) >= 0xA0);
        i += length;
    }
    return surrogate;
}

// The code-point index of a byte offset in decoded text: Python's JSONDecodeError.pos.
std::size_t code_point_index(std::string_view text, std::size_t offset) {
    std::size_t index = 0;
    for (std::size_t i = 0; i < offset && i < text.size(); ++i) {
        if ((static_cast<unsigned char>(text[i]) & 0xC0U) != 0x80U) { ++index; }
    }
    return index;
}

bool is_whitespace(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

bool is_digit(char c) { return c >= '0' && c <= '9'; }

int hex_digit(char c) {
    if (c >= '0' && c <= '9') { return c - '0'; }
    if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
    if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
    return -1;
}

bool is_surrogate(char32_t c) { return c >= 0xD800 && c <= 0xDFFF; }

// UTF-8, generalized to surrogates, which a decoded body refuses afterwards.
void append_code_point(std::string& out, char32_t c) {
    if (c < 0x80) {
        out += static_cast<char>(c);
    } else if (c < 0x800) {
        out += static_cast<char>(0xC0U | (c >> 6U));
        out += static_cast<char>(0x80U | (c & 0x3FU));
    } else if (c < 0x10000) {
        out += static_cast<char>(0xE0U | (c >> 12U));
        out += static_cast<char>(0x80U | ((c >> 6U) & 0x3FU));
        out += static_cast<char>(0x80U | (c & 0x3FU));
    } else {
        out += static_cast<char>(0xF0U | (c >> 18U));
        out += static_cast<char>(0x80U | ((c >> 12U) & 0x3FU));
        out += static_cast<char>(0x80U | ((c >> 6U) & 0x3FU));
        out += static_cast<char>(0x80U | (c & 0x3FU));
    }
}

// Python float() of a JSON number token: correctly rounded, saturating to +-inf above DBL_MAX and
// to +-0.0 below the smallest subnormal.
double parse_float(std::string_view token) {
    double value                    = 0.0;
    const std::from_chars_result result = std::from_chars(token.data(), token.data() + token.size(),
                                                          value);
    if (result.ec != std::errc::result_out_of_range) { return value; }

    // The decimal power of the leading significant digit decides overflow from underflow.
    std::size_t i       = token.front() == '-' ? 1 : 0;
    std::int64_t power  = 0;
    bool significant    = false;
    const std::size_t integer_begin = i;
    while (i < token.size() && is_digit(token[i])) { ++i; }
    if (token[integer_begin] != '0') {
        power       = static_cast<std::int64_t>(i - integer_begin) - 1;
        significant = true;
    }
    if (i < token.size() && token[i] == '.') {
        for (std::int64_t place = -1; ++i < token.size() && is_digit(token[i]); --place) {
            if (!significant && token[i] != '0') {
                power       = place;
                significant = true;
            }
        }
    }
    std::int64_t exponent = 0;
    if (i < token.size()) {
        const bool negative = token[++i] == '-';
        if (token[i] == '-' || token[i] == '+') { ++i; }
        for (; i < token.size(); ++i) {
            exponent = std::min<std::int64_t>(exponent * 10 + (token[i] - '0'), 1'000'000'000);
        }
        if (negative) { exponent = -exponent; }
    }
    const double magnitude =
        significant && power + exponent >= 0 ? std::numeric_limits<double>::infinity() : 0.0;
    return token.front() == '-' ? -magnitude : magnitude;
}

// CPython's C json scanner (Modules/_json.c scan_once_unicode and its callees) over decoded
// UTF-8, with the same acceptance, values and error messages and positions. Every structural
// comparison is against ASCII, so byte offsets map to Python's code-point indices exactly.
class Scanner {
public:
    explicit Scanner(std::string_view text) : text_(text) {}

    // json.loads: one value between optional whitespace.
    Value decode() {
        std::size_t index = skip(0);
        Value value       = scan(index, 0);
        index             = skip(index);
        if (index != text_.size()) { throw SyntaxError{"Extra data", index}; }
        return value;
    }

    [[nodiscard]] bool decoded_surrogate() const noexcept { return surrogate_; }

private:
    [[nodiscard]] std::size_t skip(std::size_t index) const {
        while (index < text_.size() && is_whitespace(text_[index])) { ++index; }
        return index;
    }

    [[nodiscard]] bool at(std::size_t index, char c) const {
        return index < text_.size() && text_[index] == c;
    }

    // Reads the value starting at `index` and advances `index` past it.
    Value scan(std::size_t& index, int depth) {
        if (index >= text_.size()) { throw SyntaxError{"Expecting value", index}; }
        Value value;
        const std::string_view rest = text_.substr(index);
        switch (text_[index]) {
        case '"':
            ++index;
            value.kind = Value::Kind::String;
            value.text = string(index);
            return value;
        case '{':
        case '[':
            if (depth >= kMaximumNesting) { throw UndecodableBody{}; }
            ++index;
            return text_[index - 1] == '{' ? object(index, depth + 1) : array(index, depth + 1);
        case 'n':
            if (rest.starts_with("null")) {
                index += 4;
                return value;
            }
            break;
        case 't':
        case 'f':
            if (rest.starts_with("true") || rest.starts_with("false")) {
                value.kind    = Value::Kind::Bool;
                value.boolean = rest.front() == 't';
                index += value.boolean ? 4 : 5;
                return value;
            }
            break;
        case 'N':
        case 'I':
        case '-':
            for (const auto& [name, number] :
                 std::array<std::pair<std::string_view, double>, 3>{{
                     {"NaN", std::numeric_limits<double>::quiet_NaN()},
                     {"Infinity", std::numeric_limits<double>::infinity()},
                     {"-Infinity", -std::numeric_limits<double>::infinity()},
                 }}) {
                if (rest.starts_with(name)) {
                    value.kind   = Value::Kind::Float;
                    value.number = number;
                    index += name.size();
                    return value;
                }
            }
            break;
        default: break;
        }
        return number(index);
    }

    // `index` follows the opening brace.
    Value object(std::size_t& index, int depth) {
        Value object;
        object.kind = Value::Kind::Dict;
        std::unordered_map<std::string, std::size_t> positions;
        std::size_t i = skip(index);
        if (!at(i, '}')) {
            while (true) {
                if (!at(i, '"')) {
                    throw SyntaxError{"Expecting property name enclosed in double quotes", i};
                }
                ++i;
                std::string key = string(i);
                i               = skip(i);
                if (!at(i, ':')) { throw SyntaxError{"Expecting ':' delimiter", i}; }
                i = skip(i + 1);
                insert(object, positions, std::move(key), scan(i, depth));
                i = skip(i);
                if (at(i, '}')) { break; }
                if (!at(i, ',')) { throw SyntaxError{"Expecting ',' delimiter", i}; }
                const std::size_t comma = i;
                i                       = skip(i + 1);
                if (at(i, '}')) {
                    throw SyntaxError{"Illegal trailing comma before end of object", comma};
                }
            }
        }
        index = i + 1;
        return object;
    }

    // `index` follows the opening bracket.
    Value array(std::size_t& index, int depth) {
        Value array;
        array.kind    = Value::Kind::List;
        std::size_t i = skip(index);
        if (!at(i, ']')) {
            while (true) {
                array.items.push_back(scan(i, depth));
                i = skip(i);
                if (at(i, ']')) { break; }
                if (!at(i, ',')) { throw SyntaxError{"Expecting ',' delimiter", i}; }
                const std::size_t comma = i;
                i                       = skip(i + 1);
                if (at(i, ']')) {
                    throw SyntaxError{"Illegal trailing comma before end of array", comma};
                }
            }
        }
        index = i + 1;
        return array;
    }

    // Python dict assignment: a repeated key keeps its first position and takes the new value.
    static void insert(Value& object, std::unordered_map<std::string, std::size_t>& positions,
                       std::string key, Value value) {
        std::vector<Value::Entry>& entries = object.entries;
        if (entries.size() < kIndexedObjectEntries) {
            for (Value::Entry& entry : entries) {
                if (entry.key == key) {
                    entry.value = std::move(value);
                    return;
                }
            }
            entries.push_back(Value::Entry{std::move(key), std::move(value)});
            if (entries.size() == kIndexedObjectEntries) {
                for (std::size_t i = 0; i < entries.size(); ++i) {
                    positions.emplace(entries[i].key, i);
                }
            }
            return;
        }
        const auto [position, inserted] = positions.try_emplace(key, entries.size());
        if (!inserted) {
            entries[position->second].value = std::move(value);
            return;
        }
        entries.push_back(Value::Entry{std::move(key), std::move(value)});
    }

    // scanstring_unicode: `index` follows the opening quote and ends past the closing one.
    std::string string(std::size_t& index) {
        const std::size_t length = text_.size();
        const std::size_t begin  = index - 1;
        std::string out;
        std::size_t end = index;
        while (true) {
            std::size_t next = end;
            for (; next < length; ++next) {
                const char c = text_[next];
                if (c == '"' || c == '\\') { break; }
                if (static_cast<unsigned char>(c) <= 0x1F) {
                    throw SyntaxError{"Invalid control character at", next};
                }
            }
            if (next >= length) { throw SyntaxError{"Unterminated string starting at", begin}; }
            out.append(text_.substr(end, next - end));
            if (text_[next] == '"') {
                index = next + 1;
                return out;
            }
            ++next;
            if (next == length) { throw SyntaxError{"Unterminated string starting at", begin}; }
            char32_t c = 0;
            if (text_[next] != 'u') {
                end = next + 1;
                switch (text_[next]) {
                case '"': c = '"'; break;
                case '\\': c = '\\'; break;
                case '/': c = '/'; break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                default: throw SyntaxError{"Invalid \\escape", end - 2};
                }
            } else {
                ++next;
                end = next + 4;
                if (end >= length) { throw SyntaxError{"Invalid \\uXXXX escape", next - 1}; }
                for (; next < end; ++next) {
                    const int digit = hex_digit(text_[next]);
                    if (digit < 0) { throw SyntaxError{"Invalid \\uXXXX escape", end - 5}; }
                    c = (c << 4U) | static_cast<char32_t>(digit);
                }
                // A high surrogate joins an immediately following \u low surrogate.
                if (c >= 0xD800 && c <= 0xDBFF && end + 6 < length && text_[next] == '\\' &&
                    text_[next + 1] == 'u') {
                    next += 2;
                    end += 6;
                    char32_t low = 0;
                    for (; next < end; ++next) {
                        const int digit = hex_digit(text_[next]);
                        if (digit < 0) { throw SyntaxError{"Invalid \\uXXXX escape", end - 5}; }
                        low = (low << 4U) | static_cast<char32_t>(digit);
                    }
                    if (low >= 0xDC00 && low <= 0xDFFF) {
                        c = 0x10000 + (((c - 0xD800) << 10U) | (low - 0xDC00));
                    } else {
                        end -= 6;
                    }
                }
            }
            surrogate_ = surrogate_ || is_surrogate(c);
            append_code_point(out, c);
        }
    }

    // _match_number_unicode: the longest JSON number prefix at `index`, an int unless it has a
    // fraction or an exponent.
    Value number(std::size_t& index) {
        const std::size_t start  = index;
        const std::size_t length = text_.size();
        std::size_t i            = start;
        if (text_[i] == '-') {
            ++i;
            if (i >= length) { throw SyntaxError{"Expecting value", start}; }
        }
        if (text_[i] >= '1' && text_[i] <= '9') {
            ++i;
            while (i < length && is_digit(text_[i])) { ++i; }
        } else if (text_[i] == '0') {
            ++i;
        } else {
            throw SyntaxError{"Expecting value", start};
        }
        bool is_float = false;
        if (i + 1 < length && text_[i] == '.' && is_digit(text_[i + 1])) {
            is_float = true;
            i += 2;
            while (i < length && is_digit(text_[i])) { ++i; }
        }
        if (i + 1 < length && (text_[i] == 'e' || text_[i] == 'E')) {
            const std::size_t exponent_start = i++;
            if (i + 1 < length && (text_[i] == '-' || text_[i] == '+')) { ++i; }
            while (i < length && is_digit(text_[i])) { ++i; }
            if (is_digit(text_[i - 1])) {
                is_float = true;
            } else {
                i = exponent_start;
            }
        }
        const std::string_view token = text_.substr(start, i - start);
        index                        = i;
        Value value;
        if (is_float) {
            value.kind   = Value::Kind::Float;
            value.number = parse_float(token);
            return value;
        }
        const bool negative = token.front() == '-';
        if (token.size() - (negative ? 1 : 0) > kMaximumIntegerDigits) { throw UndecodableBody{}; }
        value.kind = Value::Kind::Int;
        value.text = token == "-0" ? "0" : std::string(token);
        return value;
    }

    std::string_view text_;
    bool surrogate_ = false;
};

// The error input as FastAPI echoes it.
Json to_json(const Value& value) {
    switch (value.kind) {
    case Value::Kind::Null: return nullptr;
    case Value::Kind::Bool: return value.boolean;
    case Value::Kind::Int: {
        const char* first = value.text.data();
        const char* last  = first + value.text.size();
        std::int64_t signed_value = 0;
        if (const auto result = std::from_chars(first, last, signed_value);
            result.ec == std::errc{} && result.ptr == last) {
            return signed_value;
        }
        std::uint64_t unsigned_value = 0;
        if (const auto result = std::from_chars(first, last, unsigned_value);
            result.ec == std::errc{} && result.ptr == last) {
            return unsigned_value;
        }
        return parse_float(value.text);
    }
    case Value::Kind::Float: return value.number;
    case Value::Kind::String: return value.text;
    case Value::Kind::List: {
        Json array = Json::array();
        for (const Value& item : value.items) { array.push_back(to_json(item)); }
        return array;
    }
    case Value::Kind::Dict: {
        Json object = Json::object();
        for (const Value::Entry& entry : value.entries) { object[entry.key] = to_json(entry.value); }
        return object;
    }
    }
    return nullptr;
}

// Pydantic's error list in FastAPI's shape.
class Errors {
public:
    void add(std::string_view type, Json loc, std::string_view message, Json input,
             Json context = nullptr) {
        Json error      = Json::object();
        error["type"]   = std::string(type);
        error["loc"]    = std::move(loc);
        error["msg"]    = std::string(message);
        error["input"]  = std::move(input);
        if (!context.is_null()) { error["ctx"] = std::move(context); }
        detail_.push_back(std::move(error));
    }

    void append(Errors&& other) {
        for (Json& error : other.detail_) { detail_.push_back(std::move(error)); }
    }

    [[nodiscard]] bool empty() const noexcept { return detail_.empty(); }

    [[noreturn]] void raise() { throw RequestValidationError(422, std::move(detail_)); }

private:
    Json detail_ = Json::array();
};

Json location(const Json& prefix, std::initializer_list<std::string_view> tail) {
    Json loc = prefix;
    for (const std::string_view part : tail) { loc.push_back(std::string(part)); }
    return loc;
}

Json length_context(std::string_view field_type, std::string_view bound_name, std::size_t bound,
                    std::size_t actual) {
    Json context                 = Json::object();
    context["field_type"]        = std::string(field_type);
    context[std::string(bound_name)] = bound;
    context["actual_length"]     = actual;
    return context;
}

// One member of the non-discriminated Noul | Choice | Score union, named in `loc` as pydantic
// names it.
struct UnionMember {
    QuestionType type;
    std::string_view literal;
    std::string_view tag;
};

constexpr std::array<UnionMember, 3> kQuestionMembers{{
    {QuestionType::Noul, "noul", "Noul"},
    {QuestionType::Choice, "choice", "function-after[_check(), Choice]"},
    {QuestionType::Score, "score", "Score"},
}};

// Validates a question as one union member: fields in declaration order (type, instructions,
// criteria), then Choice's after-validator. Returns whether the member accepts it; otherwise its
// errors are appended.
bool validate_member(const UnionMember& member, const Value& question, const Json& prefix,
                     Errors& errors) {
    const Json member_loc = location(prefix, {member.tag});
    if (question.kind != Value::Kind::Dict) {
        errors.add("model_attributes_type", member_loc, kNotAnObject, to_json(question));
        return false;
    }
    bool valid        = true;
    const Value* type = question.find("type");
    if (type == nullptr) {
        errors.add("missing", location(member_loc, {"type"}), kFieldRequired, to_json(question));
        valid = false;
    } else if (type->kind != Value::Kind::String || type->text != member.literal) {
        const std::string expected = "'" + std::string(member.literal) + "'";
        errors.add("literal_error", location(member_loc, {"type"}), "Input should be " + expected,
                   to_json(*type), Json::object({{"expected", expected}}));
        valid = false;
    }
    const Json criteria_loc = location(member_loc, {"criteria"});
    const Value* criteria   = question.find("criteria");
    switch (member.type) {
    case QuestionType::Noul:
        if (criteria != nullptr && criteria->kind != Value::Kind::Null &&
            criteria->kind != Value::Kind::Dict) {
            errors.add("dict_type", criteria_loc, kNotADictionary, to_json(*criteria));
            valid = false;
        }
        break;
    case QuestionType::Choice:
        if (criteria == nullptr) {
            errors.add("missing", criteria_loc, kFieldRequired, to_json(question));
            valid = false;
        } else if (criteria->kind != Value::Kind::Dict) {
            errors.add("dict_type", criteria_loc, kNotADictionary, to_json(*criteria));
            valid = false;
        }
        break;
    case QuestionType::Score:
        if (criteria == nullptr) {
            errors.add("missing", criteria_loc, kFieldRequired, to_json(question));
            valid = false;
        } else if (criteria->kind != Value::Kind::List) {
            errors.add("list_type", criteria_loc, "Input should be a valid list", to_json(*criteria));
            valid = false;
        } else if (criteria->items.empty()) {
            errors.add("too_short", criteria_loc,
                       "List should have at least 1 item after validation, not 0",
                       to_json(*criteria), length_context("List", "min_length", 1, 0));
            valid = false;
        } else if (criteria->items.size() > kMaximumDecisionOptions) {
            const std::size_t count = criteria->items.size();
            errors.add("too_long", criteria_loc,
                       "List should have at most " + std::to_string(kMaximumDecisionOptions) +
                           " items after validation, not " + std::to_string(count),
                       to_json(*criteria),
                       length_context("List", "max_length", kMaximumDecisionOptions, count));
            valid = false;
        }
        break;
    }
    if (valid && member.type == QuestionType::Choice) {
        const std::size_t count = criteria->entries.size();
        if (count < 1 || count > kMaximumDecisionOptions) {
            errors.add("value_error", member_loc,
                       "Value error, criteria must have 1.." +
                           std::to_string(kMaximumDecisionOptions) + " options",
                       to_json(question), Json::object({{"error", Json::object()}}));
            valid = false;
        }
    }
    return valid;
}

// Pydantic's smart-mode union: the member that accepts the question (the type literals exclude
// each other), else every member's errors in declaration order.
std::optional<QuestionType> validate_question(const std::string& id, const Value& question,
                                              Errors& errors) {
    const Json prefix = Json::array({"body", "questions", id});
    Errors member_errors;
    for (const UnionMember& member : kQuestionMembers) {
        if (validate_member(member, question, prefix, member_errors)) { return member.type; }
    }
    errors.append(std::move(member_errors));
    return std::nullopt;
}

[[noreturn]] void raise_body_missing() {
    Errors errors;
    errors.add("missing", Json::array({"body"}), kFieldRequired, nullptr);
    errors.raise();
}

std::string summary(const Json& detail) {
    if (detail.is_string()) { return detail.get<std::string>(); }
    if (detail.is_array() && !detail.empty() && detail.front().contains("msg")) {
        return "invalid System One request: " + detail.front()["msg"].get<std::string>();
    }
    return "invalid System One request";
}

} // namespace

RequestValidationError::RequestValidationError(int status, nlohmann::ordered_json detail)
    : std::runtime_error(summary(detail)), status_(status), detail_(std::move(detail)) {}

std::string_view question_type_name(QuestionType type) noexcept {
    switch (type) {
    case QuestionType::Noul: return "noul";
    case QuestionType::Choice: return "choice";
    case QuestionType::Score: return "score";
    }
    return {};
}

Value parse_json(std::string_view body) {
    std::string_view text = body;
    // bytes.decode("utf-8-sig") drops one byte order mark; positions count from after it.
    if (text.starts_with("\xEF\xBB\xBF")) { text.remove_prefix(3); }
    try {
        const bool encoded_surrogate = validate_utf8(text);
        Scanner scanner(text);
        Value value = scanner.decode();
        if (encoded_surrogate || scanner.decoded_surrogate()) { throw UndecodableBody{}; }
        return value;
    } catch (const SyntaxError& error) {
        Json invalid     = Json::object();
        invalid["type"]  = "json_invalid";
        invalid["loc"]   = Json::array({"body", code_point_index(text, error.offset)});
        invalid["msg"]   = "JSON decode error";
        invalid["input"] = Json::object();
        invalid["ctx"]   = Json::object({{"error", std::string(error.message)}});
        throw RequestValidationError(422, Json::array({std::move(invalid)}));
    } catch (const UndecodableBody&) {
        throw RequestValidationError(400, std::string(kUndecodableBody));
    }
}

Request parse_request(std::string_view body) {
    // FastAPI validates an empty body and a JSON null alike: the required body is missing.
    if (body.empty()) { raise_body_missing(); }
    const Value root = parse_json(body);
    if (root.kind == Value::Kind::Null) { raise_body_missing(); }
    if (root.kind != Value::Kind::Dict) {
        Errors errors;
        errors.add("model_attributes_type", Json::array({"body"}), kNotAnObject, to_json(root));
        errors.raise();
    }

    // SystemOneRequest fields in declaration order: state, model, questions.
    Errors errors;
    const Value* state = root.find("state");
    if (state == nullptr) {
        errors.add("missing", Json::array({"body", "state"}), kFieldRequired, to_json(root));
    }
    const Value* model = root.find("model");
    if (model != nullptr && model->kind != Value::Kind::String) {
        errors.add("string_type", Json::array({"body", "model"}), "Input should be a valid string",
                   to_json(*model));
    }
    const Value* questions = root.find("questions");
    std::vector<QuestionType> types;
    if (questions == nullptr) {
        errors.add("missing", Json::array({"body", "questions"}), kFieldRequired, to_json(root));
    } else if (questions->kind != Value::Kind::Dict) {
        errors.add("dict_type", Json::array({"body", "questions"}), kNotADictionary,
                   to_json(*questions));
    } else if (questions->entries.empty()) {
        errors.add("too_short", Json::array({"body", "questions"}),
                   "Dictionary should have at least 1 item after validation, not 0", Json::object(),
                   length_context("Dictionary", "min_length", 1, 0));
    } else {
        types.reserve(questions->entries.size());
        for (const Value::Entry& entry : questions->entries) {
            const std::optional<QuestionType> type =
                validate_question(entry.key, entry.value, errors);
            if (type) { types.push_back(*type); }
        }
    }
    if (!errors.empty()) { errors.raise(); }

    // kev to_record.
    Request request;
    request.model       = model != nullptr ? model->text : std::string(kDefaultModel);
    request.input.state = render(*state);
    request.input.questions.reserve(types.size());
    request.questions.reserve(types.size());
    for (std::size_t q = 0; q < types.size(); ++q) {
        const Value::Entry& entry = questions->entries[q];
        const Value* instructions = entry.value.find("instructions");
        const Value* criteria     = entry.value.find("criteria");
        DecisionQuestion question{
            .instructions = instructions != nullptr ? render(*instructions) : std::string(),
            .options      = {},
        };
        QuestionMeta meta{.id = entry.key, .type = types[q], .keys = {}, .legend = {}};
        switch (types[q]) {
        case QuestionType::Noul: {
            const bool described = criteria != nullptr && criteria->kind == Value::Kind::Dict;
            question.options     = {
                option_text("no", described ? criteria->find("false") : nullptr),
                option_text("yes", described ? criteria->find("true") : nullptr),
            };
            meta.keys = {"false", "true"};
            break;
        }
        case QuestionType::Choice:
            for (const Value::Entry& option : criteria->entries) {
                question.options.push_back(option_text(option.key, &option.value));
                meta.keys.push_back(option.key);
            }
            break;
        case QuestionType::Score:
            for (std::size_t level = 0; level < criteria->items.size(); ++level) {
                question.options.push_back(render(criteria->items[level]));
                meta.keys.push_back(std::to_string(level));
            }
            meta.legend = question.options;
            break;
        }
        request.input.questions.push_back(std::move(question));
        request.questions.push_back(std::move(meta));
    }
    return request;
}

} // namespace ninfer::product::systemone
