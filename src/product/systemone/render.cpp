#include "product/systemone/render.h"

#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <string>
#include <string_view>
#include <system_error>

namespace ninfer::product::systemone {

const Value* Value::find(std::string_view key) const noexcept {
    for (const Entry& entry : entries) {
        if (entry.key == key) { return &entry.value; }
    }
    return nullptr;
}

std::string render(const Value& value, std::size_t indent) {
    switch (value.kind) {
    case Value::Kind::Null: return {};
    case Value::Kind::Bool: return value.boolean ? "True" : "False";
    case Value::Kind::Int:
    case Value::Kind::String: return value.text;
    case Value::Kind::Float: return python_float_repr(value.number);
    case Value::Kind::List: {
        // "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
        const std::string pad(2 * indent, ' ');
        std::string out;
        for (std::size_t i = 0; i < value.items.size(); ++i) {
            if (i != 0) { out += '\n'; }
            const std::string item = render(value.items[i], indent + 1);
            out += pad;
            out += "- ";
            out += python_lstrip(item);
        }
        return out;
    }
    case Value::Kind::Dict: {
        // Containers go below their key one level deeper; scalars follow it, rendered unindented.
        const std::string pad(2 * indent, ' ');
        std::string out;
        for (std::size_t i = 0; i < value.entries.size(); ++i) {
            const Value::Entry& entry = value.entries[i];
            if (i != 0) { out += '\n'; }
            out += pad;
            out += entry.key;
            const Value::Kind kind = entry.value.kind;
            if (kind == Value::Kind::Dict || kind == Value::Kind::List) {
                out += ":\n";
                out += render(entry.value, indent + 1);
            } else {
                out += ": ";
                out += render(entry.value);
            }
        }
        return out;
    }
    }
    return {};
}

std::string option_text(std::string_view name, const Value* description) {
    // `desc is None or desc == ""`: only null and the empty string compare so in Python.
    if (description == nullptr || description->kind == Value::Kind::Null ||
        (description->kind == Value::Kind::String && description->text.empty())) {
        return std::string(name);
    }
    std::string out(name);
    out += ": ";
    out += render(*description);
    return out;
}

std::string python_float_repr(double value) {
    if (std::isnan(value)) { return "nan"; }
    if (std::isinf(value)) { return value > 0.0 ? "inf" : "-inf"; }
    if (value == 0.0) { return std::signbit(value) ? "-0.0" : "0.0"; }

    // The shortest round-trip digits (the ones repr uses) in scientific form "-d.ddde+XX".
    std::array<char, 64> buffer{};
    const std::to_chars_result result = std::to_chars(buffer.data(), buffer.data() + buffer.size(),
                                                      value, std::chars_format::scientific);
    std::string_view scientific(buffer.data(), static_cast<std::size_t>(result.ptr - buffer.data()));
    std::string out;
    if (scientific.front() == '-') {
        out += '-';
        scientific.remove_prefix(1);
    }
    const std::size_t e = scientific.find('e');
    std::string digits(1, scientific.front());
    if (e > 1) { digits.append(scientific.substr(2, e - 2)); }
    const bool negative_exponent = scientific[e + 1] == '-';
    const std::string_view exponent_digits = scientific.substr(e + 2);
    int exponent = 0;
    std::from_chars(exponent_digits.data(), exponent_digits.data() + exponent_digits.size(),
                    exponent);
    if (negative_exponent) { exponent = -exponent; }

    // CPython format_float_short, 'r' mode: decpt is the decimal point position after the first
    // digit's power of ten, exponent notation for decpt <= -4 or decpt > 16.
    const int decpt = exponent + 1;
    const int count = static_cast<int>(digits.size());
    if (decpt <= -4 || decpt > 16) {
        out += digits.front();
        if (count > 1) {
            out += '.';
            out.append(digits, 1);
        }
        out += 'e';
        out += negative_exponent ? '-' : '+';
        const int magnitude = std::abs(exponent);
        if (magnitude < 10) { out += '0'; }
        out += std::to_string(magnitude);
    } else if (decpt <= 0) {
        out += "0.";
        out.append(static_cast<std::size_t>(-decpt), '0');
        out += digits;
    } else if (decpt < count) {
        out.append(digits, 0, static_cast<std::size_t>(decpt));
        out += '.';
        out.append(digits, static_cast<std::size_t>(decpt));
    } else {
        out += digits;
        out.append(static_cast<std::size_t>(decpt - count), '0');
        out += ".0";
    }
    return out;
}

bool python_isspace(char32_t code_point) noexcept {
    // Py_UNICODE_ISSPACE: bidirectional class WS, B or S, or general category Zs.
    switch (code_point) {
    case 0x09:
    case 0x0A:
    case 0x0B:
    case 0x0C:
    case 0x0D:
    case 0x1C:
    case 0x1D:
    case 0x1E:
    case 0x1F:
    case 0x20:
    case 0x85:
    case 0xA0:
    case 0x1680:
    case 0x2028:
    case 0x2029:
    case 0x202F:
    case 0x205F:
    case 0x3000: return true;
    default: return code_point >= 0x2000 && code_point <= 0x200A;
    }
}

std::string_view python_lstrip(std::string_view text) noexcept {
    std::size_t offset = 0;
    while (offset < text.size()) {
        std::size_t next     = offset;
        char32_t code_point  = 0;
        if (!next_code_point(text, next, code_point) || !python_isspace(code_point)) { break; }
        offset = next;
    }
    return text.substr(offset);
}

bool next_code_point(std::string_view text, std::size_t& offset, char32_t& code_point) noexcept {
    if (offset >= text.size()) { return false; }
    const auto byte = [&](std::size_t i) { return static_cast<unsigned char>(text[offset + i]); };
    const unsigned char lead = byte(0);
    if (lead < 0x80) {
        code_point = lead;
        ++offset;
        return true;
    }
    std::size_t length   = 0;
    unsigned char low    = 0x80;
    unsigned char high   = 0xBF;
    char32_t accumulated = 0;
    if (lead >= 0xC2 && lead <= 0xDF) {
        length      = 2;
        accumulated = lead & 0x1FU;
    } else if (lead >= 0xE0 && lead <= 0xEF) {
        length      = 3;
        accumulated = lead & 0x0FU;
        if (lead == 0xE0) { low = 0xA0; }
        if (lead == 0xED) { high = 0x9F; }
    } else if (lead >= 0xF0 && lead <= 0xF4) {
        length      = 4;
        accumulated = lead & 0x07U;
        if (lead == 0xF0) { low = 0x90; }
        if (lead == 0xF4) { high = 0x8F; }
    } else {
        return false;
    }
    if (text.size() - offset < length) { return false; }
    for (std::size_t i = 1; i < length; ++i) {
        const unsigned char continuation = byte(i);
        const unsigned char minimum      = i == 1 ? low : 0x80;
        const unsigned char maximum      = i == 1 ? high : 0xBF;
        if (continuation < minimum || continuation > maximum) { return false; }
        accumulated = (accumulated << 6U) | (continuation & 0x3FU);
    }
    code_point = accumulated;
    offset += length;
    return true;
}

} // namespace ninfer::product::systemone
