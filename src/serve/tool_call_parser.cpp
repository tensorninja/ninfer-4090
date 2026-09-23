#include "serve/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <utility>

namespace ninfer::serve {
namespace {

using Json        = nlohmann::json;
using OrderedJson = nlohmann::ordered_json;

constexpr std::string_view kToolOpen           = "<tool_call>";
constexpr std::string_view kToolClose          = "</tool_call>";
constexpr std::string_view kFunctionOpen       = "<function=";
constexpr std::string_view kFunctionClose      = "</function>";
constexpr std::string_view kParameterOpen      = "<parameter=";
constexpr std::string_view kParameterClose     = "</parameter>";
constexpr std::string_view kParameterCloseLine = "\n</parameter>";

bool is_space(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
}

std::size_t leading_space(std::string_view text) noexcept {
    std::size_t size = 0;
    while (size < text.size() && is_space(text[size])) { ++size; }
    return size;
}

enum class Match : std::uint8_t { None, Partial, Full };

// Whether `text` starts with `marker`, or could once more input arrives.
Match match_prefix(std::string_view text, std::string_view marker, bool at_end) noexcept {
    if (text.size() >= marker.size()) {
        return text.substr(0, marker.size()) == marker ? Match::Full : Match::None;
    }
    return !at_end && marker.substr(0, text.size()) == text ? Match::Partial : Match::None;
}

// Longest suffix of `text` that is a proper prefix of `marker`.
std::size_t partial_suffix(std::string_view text, std::string_view marker) noexcept {
    for (std::size_t size = std::min(text.size(), marker.size() - 1); size != 0; --size) {
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

bool function_name_char(char c) noexcept {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
           c == '_' || c == '-';
}

bool valid_function_name(std::string_view name, std::size_t max_name_length) noexcept {
    return !name.empty() && name.size() <= max_name_length &&
           std::all_of(name.begin(), name.end(), function_name_char);
}

// JSON string escaping is a per-byte map, so escaping each piece of a split value concatenates
// to the escaping of the whole value.
void append_escaped(std::string& out, std::string_view text) {
    for (const char c : text) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                std::array<char, 8> escaped{};
                std::snprintf(escaped.data(), escaped.size(), "\\u%04x",
                              static_cast<unsigned int>(static_cast<unsigned char>(c)));
                out += escaped.data();
            } else {
                out += c;
            }
        }
    }
}

std::string json_string(std::string_view text) {
    std::string out = "\"";
    append_escaped(out, text);
    out += '"';
    return out;
}

std::string new_tool_call_id() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> dist;
    std::array<char, 32> buf{};
    std::snprintf(buf.data(), buf.size(), "call_%016llx",
                  static_cast<unsigned long long>(dist(rng)));
    return std::string(buf.data());
}

// The inverse of the template's `<parameter=K>\nVALUE\n</parameter>` rendering: exactly one
// newline on each side belongs to the markup, everything else to the value.
std::string_view strip_value_frame(std::string_view value) noexcept {
    if (!value.empty() && value.front() == '\n') { value.remove_prefix(1); }
    if (!value.empty() && value.back() == '\n') { value.remove_suffix(1); }
    return value;
}

std::string buffered_value_json(std::string_view value) {
    const OrderedJson parsed = OrderedJson::parse(value, nullptr, false);
    return parsed.is_discarded() ? json_string(value) : parsed.dump();
}

// Qwen3 weights are trained on the Hermes convention, so the model emits a JSON
// body inside <tool_call> even when the prompt teaches the XML dialect.
// Accepting only XML turned those calls back into prose, which reads to a client
// as a turn that ended without calling anything.
bool parse_hermes_block(std::string_view block, std::size_t max_name_length, std::string& name,
                        std::string& arguments) {
    const OrderedJson parsed = OrderedJson::parse(block, nullptr, false);
    if (!parsed.is_object()) { return false; }

    const auto name_field = parsed.find("name");
    if (name_field == parsed.end() || !name_field->is_string()) { return false; }
    const auto& candidate = name_field->get_ref<const std::string&>();
    if (!valid_function_name(candidate, max_name_length)) { return false; }

    OrderedJson args           = OrderedJson::object();
    const auto arguments_field = parsed.find("arguments");
    if (arguments_field != parsed.end() && !arguments_field->is_null()) {
        if (arguments_field->is_object()) {
            args = *arguments_field;
        } else if (arguments_field->is_string()) {
            // Double-encoded arguments: the object arrives as a JSON string.
            args = OrderedJson::parse(arguments_field->get_ref<const std::string&>(), nullptr,
                                      false);
            if (!args.is_object()) { return false; }
        } else {
            return false;
        }
    }
    name      = candidate;
    arguments = args.dump();
    return true;
}

enum SchemaType : unsigned {
    kSchemaString = 1U << 0U,
    kSchemaObject = 1U << 1U,
    kSchemaArray  = 1U << 2U,
    kSchemaNull   = 1U << 3U,
    kSchemaOther  = 1U << 4U,
};

unsigned type_name_bit(const std::string& name) noexcept {
    if (name == "string") { return kSchemaString; }
    if (name == "object") { return kSchemaObject; }
    if (name == "array") { return kSchemaArray; }
    if (name == "null") { return kSchemaNull; }
    return kSchemaOther;
}

unsigned value_type_bit(const Json& value) noexcept {
    if (value.is_string()) { return kSchemaString; }
    if (value.is_object()) { return kSchemaObject; }
    if (value.is_array()) { return kSchemaArray; }
    if (value.is_null()) { return kSchemaNull; }
    return kSchemaOther;
}

// The set of JSON types a property schema admits, or 0 when the schema does not say.
unsigned schema_types(const Json& schema, int depth) {
    constexpr int kMaximumDepth = 8;
    if (!schema.is_object() || depth > kMaximumDepth) { return 0; }
    if (const auto type = schema.find("type"); type != schema.end()) {
        if (type->is_string()) { return type_name_bit(type->get_ref<const std::string&>()); }
        if (!type->is_array()) { return 0; }
        unsigned types = 0;
        for (const Json& entry : *type) {
            if (!entry.is_string()) { return 0; }
            types |= type_name_bit(entry.get_ref<const std::string&>());
        }
        return types;
    }
    for (const char* key : {"anyOf", "oneOf"}) {
        const auto branches = schema.find(key);
        if (branches == schema.end()) { continue; }
        if (!branches->is_array() || branches->empty()) { return 0; }
        unsigned types = 0;
        for (const Json& branch : *branches) {
            const unsigned branch_types = schema_types(branch, depth + 1);
            if (branch_types == 0) { return 0; }
            types |= branch_types;
        }
        return types;
    }
    if (const auto constant = schema.find("const"); constant != schema.end()) {
        return value_type_bit(*constant);
    }
    if (const auto values = schema.find("enum"); values != schema.end() && values->is_array()) {
        unsigned types = 0;
        for (const Json& value : *values) { types |= value_type_bit(value); }
        return types;
    }
    return 0;
}

ToolParameterKind classify_property(const Json& schema) {
    const unsigned types = schema_types(schema, 0) & ~static_cast<unsigned>(kSchemaNull);
    if (types == kSchemaString) { return ToolParameterKind::String; }
    if (types != 0 && (types & ~static_cast<unsigned>(kSchemaObject | kSchemaArray)) == 0) {
        return ToolParameterKind::JsonContainer;
    }
    return ToolParameterKind::Buffered;
}

} // namespace

ToolParameterKinds ToolParameterKinds::from_tools(const std::vector<ToolDefinition>& tools) {
    ToolParameterKinds kinds;
    for (const ToolDefinition& tool : tools) {
        const Json schema = Json::parse(tool.parameters_json, nullptr, false);
        if (!schema.is_object()) { continue; }
        const auto properties = schema.find("properties");
        if (properties == schema.end() || !properties->is_object()) { continue; }
        auto& parameters = kinds.tools_[tool.name];
        for (auto property = properties->begin(); property != properties->end(); ++property) {
            parameters[property.key()] = classify_property(property.value());
        }
    }
    return kinds;
}

ToolParameterKind ToolParameterKinds::kind(std::string_view tool,
                                           std::string_view parameter) const {
    const auto parameters = tools_.find(tool);
    if (parameters == tools_.end()) { return ToolParameterKind::Buffered; }
    const auto kind = parameters->second.find(parameter);
    return kind == parameters->second.end() ? ToolParameterKind::Buffered : kind->second;
}

QwenToolCallStream::QwenToolCallStream(ToolParameterKinds kinds, std::size_t max_name_length,
                                       ToolCallEvents* events)
    : kinds_(std::move(kinds)), max_name_length_(max_name_length), events_(events) {}

void QwenToolCallStream::feed(std::string_view text) {
    if (finished_) { throw std::logic_error("tool-call stream is already finished"); }
    // A whole output fed at once is parsed in bounded pieces, so no step moves an unbounded
    // buffer. Pieces end on character boundaries, and the result does not depend on the split.
    constexpr std::size_t kPiece = 4096;
    while (!text.empty()) {
        std::size_t size = std::min(text.size(), kPiece);
        while (size < text.size() && (static_cast<unsigned char>(text[size]) & 0xC0U) == 0x80U) {
            ++size;
        }
        buffer_.append(text.substr(0, size));
        text.remove_prefix(size);
        while (step()) {}
    }
}

ToolCallStreamResult QwenToolCallStream::finish() {
    if (finished_) { throw std::logic_error("tool-call stream is already finished"); }
    at_end_ = true;
    while (step()) {}
    finished_ = true;
    return std::move(result_);
}

// Each step either makes a decision and returns true, or returns false when the buffered bytes
// cannot be decided before more input (or, at the end, when nothing is left to decide).
bool QwenToolCallStream::step() {
    switch (state_) {
    case State::Text:
        return step_text();
    case State::Header:
        return step_header();
    case State::FunctionName:
        return step_function_name();
    case State::Hermes:
        return step_hermes();
    case State::Body:
        return step_body();
    case State::ParameterName:
        return step_parameter_name();
    case State::StringValue:
        return step_streamed_value(true);
    case State::ContainerStart:
        return step_container_start();
    case State::RawValue:
        return step_streamed_value(false);
    case State::BufferedValue:
        return step_buffered_value();
    case State::CallClose:
        return step_call_close();
    case State::AfterCall:
        return step_after_call();
    }
    return false;
}

// Whitespace before <tool_call> belongs to the markup, so trailing whitespace is held with any
// partial marker until the next byte shows which it is.
bool QwenToolCallStream::step_text() {
    const std::size_t marker = buffer_.find(kToolOpen);
    if (marker != std::string::npos) {
        std::size_t end = marker;
        while (end != 0 && is_space(buffer_[end - 1])) { --end; }
        emit_content(std::string_view(buffer_).substr(0, end));
        markup_.assign(buffer_, end, marker + kToolOpen.size() - end);
        buffer_.erase(0, marker + kToolOpen.size());
        set_state(State::Header);
        return true;
    }
    std::size_t end = buffer_.size();
    if (!at_end_) {
        end -= partial_suffix(buffer_, kToolOpen);
        while (end != 0 && is_space(buffer_[end - 1])) { --end; }
    }
    emit_content(std::string_view(buffer_).substr(0, end));
    buffer_.erase(0, end);
    return false;
}

bool QwenToolCallStream::step_header() {
    consume_markup(leading_space(buffer_));
    if (buffer_.empty()) {
        if (!at_end_) { return false; }
        reject_markup();
        return true;
    }
    if (buffer_.front() == '{') {
        set_state(State::Hermes);
        return true;
    }
    switch (match_prefix(buffer_, kFunctionOpen, at_end_)) {
    case Match::Full:
        consume_markup(kFunctionOpen.size());
        name_.clear();
        set_state(State::FunctionName);
        return true;
    case Match::Partial:
        return false;
    case Match::None:
        break;
    }
    reject_markup();
    return true;
}

bool QwenToolCallStream::step_function_name() {
    std::size_t index = 0;
    for (; index < buffer_.size(); ++index) {
        const char c = buffer_[index];
        if (c == '>' && !name_.empty()) {
            consume_markup(index + 1);
            markup_.clear();
            begin_call(std::exchange(name_, {}));
            first_parameter_ = true;
            emit_arguments("{");
            set_state(State::Body);
            return true;
        }
        if (!function_name_char(c) || name_.size() == max_name_length_) {
            consume_markup(index);
            reject_markup();
            return true;
        }
        name_ += c;
    }
    consume_markup(index);
    if (!at_end_) { return false; }
    reject_markup();
    return true;
}

bool QwenToolCallStream::step_hermes() {
    const std::size_t close = find_unscanned(kToolClose);
    if (close == std::string::npos) {
        if (!at_end_) { return false; }
        consume_markup(buffer_.size());
        reject_markup();
        return true;
    }
    std::string name;
    std::string arguments;
    if (!parse_hermes_block(std::string_view(buffer_).substr(0, close), max_name_length_, name,
                            arguments)) {
        consume_markup(close + kToolClose.size());
        reject_markup();
        return true;
    }
    buffer_.erase(0, close + kToolClose.size());
    markup_.clear();
    begin_call(std::move(name));
    emit_arguments(arguments);
    complete_call();
    set_state(State::AfterCall);
    return true;
}

// Stray bytes between parameters are dropped. A marker still incomplete at the end of output is
// part of the unfinished call rather than stray text.
bool QwenToolCallStream::step_body() {
    std::size_t index = 0;
    for (;;) {
        const std::size_t tag = std::min(buffer_.find('<', index), buffer_.size());
        discard(std::string_view(buffer_).substr(index, tag - index));
        index = tag;
        if (index == buffer_.size()) {
            buffer_.clear();
            return false;
        }
        const std::string_view rest = std::string_view(buffer_).substr(index);
        bool partial                = false;
        for (const std::string_view marker : {kParameterOpen, kFunctionClose, kToolClose}) {
            const Match match = match_prefix(rest, marker, false);
            if (match == Match::Partial) { partial = true; }
            if (match != Match::Full) { continue; }
            buffer_.erase(0, index + marker.size());
            if (marker == kParameterOpen) {
                set_state(State::ParameterName);
            } else {
                emit_arguments("}");
                complete_call();
                set_state(marker == kFunctionClose ? State::CallClose : State::AfterCall);
            }
            return true;
        }
        if (partial) {
            buffer_.erase(0, index);
            return false;
        }
        discard(rest.substr(0, 1));
        ++index;
    }
}

bool QwenToolCallStream::step_parameter_name() {
    const std::size_t close = find_unscanned(">");
    if (close == std::string::npos) { return false; }
    const std::string name = buffer_.substr(0, close);
    buffer_.erase(0, close + 1);
    begin_parameter(name);
    return true;
}

bool QwenToolCallStream::step_container_start() {
    const std::size_t start = leading_space(buffer_);
    if (start == buffer_.size()) { return false; }
    if (buffer_[start] == '{' || buffer_[start] == '[') {
        buffer_.erase(0, start);
        value_started_ = true;
        set_state(State::RawValue);
    } else {
        set_state(State::BufferedValue);
    }
    return true;
}

// Streams a value up to any suffix that could still become `\n</parameter>`. The held bytes
// always start at an ASCII byte, so every emitted piece ends on a character boundary.
bool QwenToolCallStream::step_streamed_value(bool escape) {
    if (!value_started_) {
        if (buffer_.empty()) { return false; }
        value_started_ = true;
        if (buffer_.front() == '\n') { buffer_.erase(0, 1); }
    }
    const std::size_t close = buffer_.find(kParameterClose);
    std::size_t end         = close;
    if (close != std::string::npos) {
        if (end != 0 && buffer_[end - 1] == '\n') { --end; }
    } else {
        if (at_end_) { return false; }
        end = buffer_.size() - std::max(partial_suffix(buffer_, kParameterCloseLine),
                                        partial_suffix(buffer_, kParameterClose));
    }
    std::string delta;
    if (escape) {
        append_escaped(delta, std::string_view(buffer_).substr(0, end));
        if (close != std::string::npos) { delta += '"'; }
    } else {
        delta.assign(buffer_, 0, end);
    }
    if (close == std::string::npos) {
        buffer_.erase(0, end);
        emit_arguments(delta);
        return false;
    }
    buffer_.erase(0, close + kParameterClose.size());
    set_state(State::Body);
    emit_arguments(delta);
    return true;
}

bool QwenToolCallStream::step_buffered_value() {
    const std::size_t close = find_unscanned(kParameterClose);
    if (close == std::string::npos) { return false; }
    const std::string value =
        buffered_value_json(strip_value_frame(std::string_view(buffer_).substr(0, close)));
    buffer_.erase(0, close + kParameterClose.size());
    set_state(State::Body);
    emit_arguments(value);
    return true;
}

bool QwenToolCallStream::step_call_close() {
    buffer_.erase(0, leading_space(buffer_));
    if (buffer_.empty()) { return false; }
    switch (match_prefix(buffer_, kToolClose, false)) {
    case Match::Full:
        buffer_.erase(0, kToolClose.size());
        set_state(State::AfterCall);
        return true;
    case Match::Partial:
        return false;
    case Match::None:
        break;
    }
    set_state(State::AfterCall);
    return true;
}

bool QwenToolCallStream::step_after_call() {
    const std::size_t marker = buffer_.find(kToolOpen);
    if (marker != std::string::npos) {
        discard(std::string_view(buffer_).substr(0, marker));
        buffer_.erase(0, marker + kToolOpen.size());
        markup_.assign(kToolOpen);
        set_state(State::Header);
        return true;
    }
    const std::size_t end =
        at_end_ ? buffer_.size() : buffer_.size() - partial_suffix(buffer_, kToolOpen);
    discard(std::string_view(buffer_).substr(0, end));
    buffer_.erase(0, end);
    return false;
}

void QwenToolCallStream::set_state(State state) noexcept {
    state_   = state;
    scanned_ = 0;
}

void QwenToolCallStream::consume_markup(std::size_t size) {
    markup_.append(buffer_, 0, size);
    buffer_.erase(0, size);
}

// The block turned out not to be a call. Before any call its bytes are ordinary text; after one
// they are dropped, like any other text after a call.
void QwenToolCallStream::reject_markup() {
    if (announced_) {
        discard(markup_);
    } else {
        emit_content(markup_);
    }
    markup_.clear();
    name_.clear();
    set_state(announced_ ? State::AfterCall : State::Text);
}

void QwenToolCallStream::begin_call(std::string name) {
    ToolCall call;
    call.id    = new_tool_call_id();
    call.name  = std::move(name);
    call.state = ToolCallState::Incomplete;
    result_.tool_calls.push_back(std::move(call));
    announced_ = true;
    if (events_ != nullptr) {
        events_->tool_call_begin(result_.tool_calls.size() - 1, result_.tool_calls.back());
    }
}

void QwenToolCallStream::begin_parameter(const std::string& name) {
    std::string opening = first_parameter_ ? "" : ",";
    first_parameter_    = false;
    opening += json_string(name);
    opening += ':';
    value_started_ = false;
    switch (kinds_.kind(result_.tool_calls.back().name, name)) {
    case ToolParameterKind::String:
        opening += '"';
        set_state(State::StringValue);
        break;
    case ToolParameterKind::JsonContainer:
        set_state(State::ContainerStart);
        break;
    case ToolParameterKind::Buffered:
        set_state(State::BufferedValue);
        break;
    }
    emit_arguments(opening);
}

void QwenToolCallStream::complete_call() {
    ToolCall& call = result_.tool_calls.back();
    call.state     = ToolCallState::Complete;
    if (events_ != nullptr) { events_->tool_call_end(result_.tool_calls.size() - 1, call); }
}

void QwenToolCallStream::emit_content(std::string_view text) {
    if (text.empty()) { return; }
    result_.content.append(text);
    if (events_ != nullptr) { events_->content(text); }
}

void QwenToolCallStream::emit_arguments(std::string_view delta) {
    if (delta.empty()) { return; }
    result_.tool_calls.back().arguments_json.append(delta);
    if (events_ != nullptr) { events_->tool_call_arguments(result_.tool_calls.size() - 1, delta); }
}

void QwenToolCallStream::discard(std::string_view text) noexcept {
    result_.discarded_bytes += static_cast<std::size_t>(
        std::count_if(text.begin(), text.end(), [](char c) { return !is_space(c); }));
}

// Finds `marker` in a buffer that only grows while the state holds it, without rescanning the
// prefix already known not to contain it.
std::size_t QwenToolCallStream::find_unscanned(std::string_view marker) {
    const std::size_t found = buffer_.find(marker, scanned_);
    if (found == std::string::npos) {
        scanned_ = buffer_.size() < marker.size() ? 0 : buffer_.size() - marker.size() + 1;
    }
    return found;
}

} // namespace ninfer::serve
