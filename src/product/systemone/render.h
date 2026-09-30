#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// kev's text rendering of TypeSafe System One content (external_repos/kev, kev/api.py `render` and
// `option_text`) over the values Python's json.loads builds, together with the Python formatting
// rules it depends on. Every function reproduces the CPython 3.12+ result exactly.
namespace ninfer::product::systemone {

// One value as Python's json.loads builds it and pydantic's JSONContent keeps it. Integers keep
// their exact decimal digits (Python ints are unbounded); objects keep Python dict semantics: the
// first occurrence of a key fixes its position, the last one its value.
struct Value {
    enum class Kind : std::uint8_t { Null, Bool, Int, Float, String, List, Dict };
    struct Entry;

    Kind kind    = Kind::Null;
    bool boolean = false;     // Bool
    double number = 0.0;      // Float, any IEEE double including NaN and infinities
    std::string text;         // String: UTF-8 text. Int: canonical decimal digits ("-0" is "0").
    std::vector<Value> items; // List
    std::vector<Entry> entries; // Dict, in insertion order with unique keys

    // The value of a Dict key, or nullptr when the key is absent (Python's dict.get).
    [[nodiscard]] const Value* find(std::string_view key) const noexcept;
};

struct Value::Entry {
    std::string key;
    Value value;
};

// kev `render(v, indent)`: None is empty, scalars are Python str(), a list is one "- item" line per
// element and an object one "key: value" line per entry, nested containers indented two spaces per
// level.
[[nodiscard]] std::string render(const Value& value, std::size_t indent = 0);

// kev `option_text(name, desc)`: the bare name when the description is absent, null or the empty
// string, else "name: render(desc)". `description == nullptr` is an absent key.
[[nodiscard]] std::string option_text(std::string_view name, const Value* description);

// Python repr(float): shortest round-trip digits, fixed notation iff 1e-4 <= |x| < 1e16 (with ".0"
// appended to integral values), else d[.ddd]e[+-]XX; "nan", "inf", "-inf", "-0.0".
[[nodiscard]] std::string python_float_repr(double value);

// Python str.isspace() for one code point, which is the set str.lstrip() and str.strip() remove.
[[nodiscard]] bool python_isspace(char32_t code_point) noexcept;

// Python str.lstrip() without arguments over valid UTF-8.
[[nodiscard]] std::string_view python_lstrip(std::string_view text) noexcept;

// Decodes the code point of well-formed UTF-8 at `offset` and advances past it. Returns false,
// leaving `offset` unchanged, at the end of `text` or at an ill-formed sequence (including an
// encoded surrogate).
[[nodiscard]] bool next_code_point(std::string_view text, std::size_t& offset,
                                   char32_t& code_point) noexcept;

} // namespace ninfer::product::systemone
