#include "serve/openai_schema.h"

#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <random>
#include <string>
#include <string_view>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {}) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

const Json& require_object(const Json& body) {
    if (!body.is_object()) { bad_request("request body must be a JSON object"); }
    return body;
}

void reject_unknown_fields(const Json& obj, std::initializer_list<std::string_view> allowed,
                           std::string_view path = {}) {
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        bool known = false;
        for (const std::string_view key : allowed) {
            if (it.key() == key) {
                known = true;
                break;
            }
        }
        if (!known) {
            const std::string param = path.empty() ? it.key() : std::string(path) + "." + it.key();
            bad_request("unknown parameter: " + param, param, "unknown_parameter");
        }
    }
}

void reject_unknown_top_level(const Json& body) {
    // Keep standard client metadata/routing fields in the compatibility surface even when they do
    // not alter local inference. Truly unknown fields must never look successfully supported.
    reject_unknown_fields(body, {"audio",
                                 "chat_template_kwargs",
                                 "enable_thinking",
                                 "frequency_penalty",
                                 "function_call",
                                 "functions",
                                 "logit_bias",
                                 "logprobs",
                                 "max_completion_tokens",
                                 "max_tokens",
                                 "messages",
                                 "metadata",
                                 "modalities",
                                 "model",
                                 "n",
                                 "parallel_tool_calls",
                                 "prediction",
                                 "presence_penalty",
                                 "preserve_thinking",
                                 "prompt_cache_key",
                                 "reasoning_effort",
                                 "response_format",
                                 "safety_identifier",
                                 "seed",
                                 "service_tier",
                                 "stop",
                                 "store",
                                 "stream",
                                 "stream_options",
                                 "temperature",
                                 "tool_choice",
                                 "tools",
                                 "top_k",
                                 "top_logprobs",
                                 "top_p",
                                 "user",
                                 "verbosity",
                                 "web_search_options"});
}

bool get_bool(const Json& obj, const char* key, bool fallback) {
    if (!obj.contains(key) || obj.at(key).is_null()) { return fallback; }
    if (!obj.at(key).is_boolean()) { bad_request(std::string(key) + " must be a boolean", key); }
    return obj.at(key).get<bool>();
}

std::optional<double> get_number(const Json& obj, const char* key) {
    if (!obj.contains(key) || obj.at(key).is_null()) { return std::nullopt; }
    if (!obj.at(key).is_number()) { bad_request(std::string(key) + " must be a number", key); }
    return obj.at(key).get<double>();
}

std::optional<int> get_int(const Json& obj, const char* key) {
    if (!obj.contains(key) || obj.at(key).is_null()) { return std::nullopt; }
    if (!obj.at(key).is_number_integer()) {
        bad_request(std::string(key) + " must be an integer", key);
    }
    if (obj.at(key).is_number_unsigned()) {
        const std::uint64_t value = obj.at(key).get<std::uint64_t>();
        if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            bad_request(std::string(key) + " is out of range", key);
        }
        return static_cast<int>(value);
    }
    const std::int64_t value = obj.at(key).get<std::int64_t>();
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
        bad_request(std::string(key) + " is out of range", key);
    }
    return static_cast<int>(value);
}

std::optional<std::uint64_t> get_u64(const Json& obj, const char* key) {
    if (!obj.contains(key) || obj.at(key).is_null()) { return std::nullopt; }
    if (!obj.at(key).is_number_integer()) {
        bad_request(std::string(key) + " must be a nonnegative integer", key);
    }
    if (obj.at(key).is_number_unsigned()) { return obj.at(key).get<std::uint64_t>(); }
    const std::int64_t value = obj.at(key).get<std::int64_t>();
    if (value < 0) { bad_request(std::string(key) + " must be nonnegative", key); }
    return static_cast<std::uint64_t>(value);
}

bool is_valid_function_name(const std::string& name) {
    if (name.empty() || name.size() > 64) { return false; }
    for (const unsigned char c : name) {
        if (std::isalnum(c) == 0 && c != '_' && c != '-') { return false; }
    }
    return true;
}

std::string require_function_name(const Json& obj, const char* param) {
    if (!obj.contains("name") || !obj.at("name").is_string()) {
        bad_request("function name must be a string", param);
    }
    std::string name = obj.at("name").get<std::string>();
    if (!is_valid_function_name(name)) {
        bad_request("function name must match [A-Za-z0-9_-]{1,64}", param);
    }
    return name;
}

ninfer::product::media_acquire::Source parse_media_url(const Json& part, const char* field) {
    if (!part.contains(field)) {
        bad_request(std::string(field) + " content part must contain " + field, "messages");
    }
    const Json& value = part.at(field);
    std::string url;
    if (value.is_string()) {
        url = value.get<std::string>();
    } else if (value.is_object()) {
        reject_unknown_fields(value, {"url", "detail"}, std::string("messages.content.") + field);
        if (!value.contains("url") || !value.at("url").is_string()) {
            bad_request(std::string(field) + " must contain a string url", "messages");
        }
        url = value.at("url").get<std::string>();
    } else {
        bad_request(std::string(field) + " must be a URL string or object containing url",
                    "messages");
    }
    if (url.empty()) { bad_request(std::string(field) + " URL must not be empty", "messages"); }
    ninfer::product::media_acquire::Source source;
    source.value = std::move(url);
    if (source.value.starts_with("data:")) {
        source.kind = ninfer::product::media_acquire::SourceKind::Data;
    } else if (source.value.starts_with("http://") || source.value.starts_with("https://")) {
        source.kind = ninfer::product::media_acquire::SourceKind::Url;
    } else {
        bad_request(std::string(field) + " must use HTTP(S) or a data URI", "messages");
    }
    return source;
}

void parse_content_parts(const Json& content, ChatTurn& turn, std::size_t index) {
    if (content.is_string()) {
        turn.content.push_back(ContentPart{ContentKind::Text, content.get<std::string>(), "text"});
        return;
    }
    if (!content.is_array()) {
        bad_request("message " + std::to_string(index) + " content must be a string or array",
                    "messages");
    }
    for (const Json& part : content) {
        if (!part.is_object() || !part.contains("type") || !part.at("type").is_string()) {
            bad_request("message " + std::to_string(index) +
                            " content parts must be objects with a string 'type'",
                        "messages");
        }
        const std::string type = part.at("type").get<std::string>();
        ContentPart out;
        out.type_raw = type;
        if (type == "text") {
            reject_unknown_fields(part, {"type", "text"}, "messages.content");
            if (!part.contains("text") || !part.at("text").is_string()) {
                bad_request("text content part must contain a string 'text'", "messages");
            }
            out.kind = ContentKind::Text;
            out.text = part.at("text").get<std::string>();
        } else if (type == "image_url") {
            reject_unknown_fields(part, {"type", "image_url"}, "messages.content");
            out.kind   = ContentKind::Image;
            out.source = parse_media_url(part, "image_url");
        } else if (type == "video_url") {
            reject_unknown_fields(part, {"type", "video_url"}, "messages.content");
            out.kind   = ContentKind::Video;
            out.source = parse_media_url(part, "video_url");
        } else if (type == "input_audio") {
            reject_unknown_fields(part, {"type", "input_audio"}, "messages.content");
            out.kind = ContentKind::InputAudio;
        } else {
            out.kind = ContentKind::Unsupported;
        }
        turn.content.push_back(std::move(out));
    }
    if (turn.content.empty()) {
        bad_request("message " + std::to_string(index) + " content must not be empty", "messages");
    }
}

std::vector<ToolCall> parse_assistant_tool_calls(const Json& item, std::size_t index) {
    std::vector<ToolCall> calls;
    if (!item.contains("tool_calls") || item.at("tool_calls").is_null()) { return calls; }
    const Json& tool_calls = item.at("tool_calls");
    if (!tool_calls.is_array() || tool_calls.empty()) {
        bad_request("assistant message " + std::to_string(index) +
                        " tool_calls must be a non-empty array",
                    "messages");
    }
    calls.reserve(tool_calls.size());
    for (std::size_t i = 0; i < tool_calls.size(); ++i) {
        const Json& call = tool_calls.at(i);
        if (!call.is_object()) { bad_request("tool_calls entries must be objects", "messages"); }
        reject_unknown_fields(call, {"id", "type", "function", "index"}, "messages.tool_calls");
        if (!call.contains("id") || !call.at("id").is_string() ||
            call.at("id").get<std::string>().empty()) {
            bad_request("tool_calls entries must contain a string id", "messages");
        }
        if (!call.contains("type") || !call.at("type").is_string() ||
            call.at("type").get<std::string>() != "function") {
            bad_request("only function tool_calls are supported", "messages",
                        "tool_type_not_supported");
        }
        if (!call.contains("function") || !call.at("function").is_object()) {
            bad_request("tool_calls entries must contain a function object", "messages");
        }
        const Json& fn = call.at("function");
        reject_unknown_fields(fn, {"name", "arguments"}, "messages.tool_calls.function");
        ToolCall out;
        out.id   = call.at("id").get<std::string>();
        out.name = require_function_name(fn, "messages");
        if (!fn.contains("arguments") || !fn.at("arguments").is_string()) {
            bad_request("function tool_calls must contain string arguments", "messages");
        }
        out.arguments_json = fn.at("arguments").get<std::string>();
        const Json parsed  = Json::parse(out.arguments_json, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object()) {
            bad_request("function tool_call arguments must be a JSON object string", "messages");
        }
        calls.push_back(std::move(out));
    }
    return calls;
}

void parse_messages(const Json& body, GenerationRequest& out) {
    if (!body.contains("messages")) { bad_request("missing required field: messages", "messages"); }
    const Json& messages = body.at("messages");
    if (!messages.is_array() || messages.empty()) {
        bad_request("messages must be a non-empty array", "messages");
    }
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const Json& item = messages.at(i);
        if (!item.is_object()) {
            bad_request("message " + std::to_string(i) + " must be an object", "messages");
        }
        reject_unknown_fields(item,
                              {"role", "content", "name", "tool_calls", "tool_call_id",
                               "function_call", "reasoning_content", "refusal", "audio"},
                              "messages");
        if (!item.contains("role") || !item.at("role").is_string()) {
            bad_request("message " + std::to_string(i) + " must have a string role", "messages");
        }
        const std::string role = item.at("role").get<std::string>();
        if (item.contains("function_call") && !item.at("function_call").is_null()) {
            bad_request("message function_call is not supported", "messages",
                        "tools_not_supported");
        }
        if (role == "function") {
            ApiError error;
            error.message = "role '" + role + "' is not supported yet";
            error.param   = "messages";
            error.code    = "unsupported_role";
            throw ApiException(std::move(error));
        }
        ChatTurn turn;
        turn.role = role;
        if (role == "tool") {
            if (item.contains("tool_calls") && !item.at("tool_calls").is_null()) {
                bad_request("tool messages must not contain tool_calls", "messages");
            }
            if (!item.contains("tool_call_id") || !item.at("tool_call_id").is_string() ||
                item.at("tool_call_id").get<std::string>().empty()) {
                bad_request("tool messages must contain a string tool_call_id", "messages");
            }
            if (!item.contains("content") || item.at("content").is_null()) {
                bad_request("tool messages must contain content", "messages");
            }
            turn.tool_call_id   = item.at("tool_call_id").get<std::string>();
            const Json& content = item.at("content");
            if (content.is_array() && content.empty()) {
                turn.content.push_back(ContentPart{ContentKind::Text, {}, "text"});
            } else {
                parse_content_parts(content, turn, i);
            }
            out.messages.push_back(std::move(turn));
            continue;
        }
        if (item.contains("tool_call_id") && !item.at("tool_call_id").is_null()) {
            bad_request("tool_call_id is only valid on tool messages", "messages");
        }
        if (role == "assistant") {
            turn.tool_calls = parse_assistant_tool_calls(item, i);
            if (item.contains("content") && !item.at("content").is_null()) {
                parse_content_parts(item.at("content"), turn, i);
            } else if (turn.tool_calls.empty()) {
                bad_request("assistant message " + std::to_string(i) +
                                " must have content or tool_calls",
                            "messages");
            }
            if (item.contains("reasoning_content") && !item.at("reasoning_content").is_null()) {
                if (!item.at("reasoning_content").is_string()) {
                    bad_request("assistant message " + std::to_string(i) +
                                    " reasoning_content must be a string",
                                "messages");
                }
                turn.reasoning_content = item.at("reasoning_content").get<std::string>();
            }
            out.messages.push_back(std::move(turn));
            continue;
        }
        if (item.contains("tool_calls") && !item.at("tool_calls").is_null()) {
            bad_request("tool_calls are only valid on assistant messages", "messages");
        }
        if (!item.contains("content") || item.at("content").is_null()) {
            bad_request("message " + std::to_string(i) + " must have content", "messages");
        }
        parse_content_parts(item.at("content"), turn, i);
        out.messages.push_back(std::move(turn));
    }
}

void parse_tools(const Json& body, GenerationRequest& out) {
    if (!body.contains("tools") || body.at("tools").is_null()) { return; }
    const Json& tools = body.at("tools");
    if (!tools.is_array()) { bad_request("tools must be an array", "tools"); }
    out.tools.reserve(tools.size());
    for (std::size_t i = 0; i < tools.size(); ++i) {
        const Json& item = tools.at(i);
        if (!item.is_object()) { bad_request("tools entries must be objects", "tools"); }
        reject_unknown_fields(item, {"type", "function"}, "tools");
        if (!item.contains("type") || !item.at("type").is_string()) {
            bad_request("tools entries must contain a string type", "tools");
        }
        if (item.at("type").get<std::string>() != "function") {
            bad_request("only function tools are supported", "tools", "tool_type_not_supported");
        }
        if (!item.contains("function") || !item.at("function").is_object()) {
            bad_request("function tools must contain a function object", "tools");
        }
        Json normalized = item;
        Json& fn        = normalized["function"];
        reject_unknown_fields(fn, {"name", "description", "parameters", "strict"},
                              "tools.function");
        ToolDefinition tool;
        tool.name = require_function_name(fn, "tools");
        if (fn.contains("description") && !fn.at("description").is_null()) {
            if (!fn.at("description").is_string()) {
                bad_request("function description must be a string", "tools");
            }
            tool.description = fn.at("description").get<std::string>();
        }
        if (!fn.contains("parameters") || fn.at("parameters").is_null()) {
            fn["parameters"] = Json{{"type", "object"}, {"properties", Json::object()}};
        }
        if (!fn.at("parameters").is_object()) {
            bad_request("function parameters must be a JSON object", "tools");
        }
        tool.parameters_json = fn.at("parameters").dump();
        if (fn.contains("strict") && !fn.at("strict").is_null()) {
            if (!fn.at("strict").is_boolean()) {
                bad_request("function strict must be a boolean", "tools");
            }
            if (fn.at("strict").get<bool>()) {
                bad_request("strict function schema enforcement is not supported", "tools",
                            "strict_tools_not_supported");
            }
        }
        // The prompt receives the same non-strict guarantee the API accepted; null and omission
        // are canonicalized rather than preserving ambiguous wire metadata.
        fn["strict"]         = false;
        tool.strict          = false;
        tool.definition_json = normalized.dump();
        out.tools.push_back(std::move(tool));
    }
}

void parse_tool_choice(const Json& body, GenerationRequest& out) {
    if (!body.contains("tool_choice") || body.at("tool_choice").is_null()) { return; }
    const Json& choice = body.at("tool_choice");
    if (choice.is_string()) {
        const std::string value = choice.get<std::string>();
        if (value == "none") {
            out.tool_choice.mode = ToolChoiceMode::None;
        } else if (value == "auto") {
            out.tool_choice.mode = ToolChoiceMode::Auto;
        } else if (value == "required") {
            bad_request("tool_choice 'required' is not supported", "tool_choice",
                        "tool_choice_not_supported");
        } else {
            bad_request("tool_choice must be 'none', 'auto', 'required', or a function choice",
                        "tool_choice");
        }
    } else if (choice.is_object()) {
        reject_unknown_fields(choice, {"type", "function"}, "tool_choice");
        if (!choice.contains("type") || !choice.at("type").is_string() ||
            choice.at("type").get<std::string>() != "function") {
            bad_request("only function tool_choice objects are supported", "tool_choice",
                        "tool_type_not_supported");
        }
        if (!choice.contains("function") || !choice.at("function").is_object()) {
            bad_request("function tool_choice must contain a function object", "tool_choice");
        }
        reject_unknown_fields(choice.at("function"), {"name"}, "tool_choice.function");
        (void)require_function_name(choice.at("function"), "tool_choice");
        bad_request("named tool_choice is not supported", "tool_choice",
                    "tool_choice_not_supported");
    } else {
        bad_request("tool_choice must be a string or object", "tool_choice");
    }
}

void parse_stop(const Json& body, GenerationRequest& out) {
    if (!body.contains("stop") || body.at("stop").is_null()) { return; }
    const Json& stop = body.at("stop");
    if (stop.is_string()) {
        if (!stop.get<std::string>().empty()) {
            out.stop_strings.push_back(stop.get<std::string>());
        }
        return;
    }
    if (stop.is_array()) {
        for (const Json& s : stop) {
            if (!s.is_string()) { bad_request("stop entries must be strings", "stop"); }
            if (!s.get<std::string>().empty()) { out.stop_strings.push_back(s.get<std::string>()); }
        }
        return;
    }
    bad_request("stop must be a string or array of strings", "stop");
}

void parse_sampling(const Json& body, GenerationRequest& out) {
    SamplingParams& s   = out.sampling;
    s.temperature       = get_number(body, "temperature");
    s.top_p             = get_number(body, "top_p");
    s.top_k             = get_int(body, "top_k");
    s.presence_penalty  = get_number(body, "presence_penalty");
    s.frequency_penalty = get_number(body, "frequency_penalty");
    s.seed              = get_u64(body, "seed");
    if (body.contains("logit_bias") && !body.at("logit_bias").is_null()) {
        const Json& bias = body.at("logit_bias");
        if (!bias.is_object()) { bad_request("logit_bias must be an object", "logit_bias"); }
        // Empty maps are emitted by some clients as a default. A nonempty map would promise a
        // sampler effect the engine cannot currently provide, so reject it instead of ignoring it.
        if (!bias.empty()) {
            bad_request("nonempty logit_bias is not supported", "logit_bias",
                        "logit_bias_not_supported");
        }
    }
    if (const std::optional<int> n = get_int(body, "n")) {
        s.n = *n;
        if (s.n != 1) {
            ApiError error;
            error.message = "n>1 is not supported yet";
            error.param   = "n";
            error.code    = "n_not_supported";
            throw ApiException(std::move(error));
        }
    }
}

void reject_unsupported_features(const Json& body) {
    for (const char* key : {"functions", "function_call"}) {
        if (body.contains(key) && !body.at(key).is_null()) {
            ApiError error;
            error.message = std::string(key) + " is not supported yet";
            error.param   = key;
            error.code    = "tools_not_supported";
            throw ApiException(std::move(error));
        }
    }
    if (body.contains("response_format") && !body.at("response_format").is_null()) {
        const Json& fmt = body.at("response_format");
        if (fmt.is_object()) { reject_unknown_fields(fmt, {"type"}, "response_format"); }
        std::string type = fmt.is_object() && fmt.contains("type") && fmt.at("type").is_string()
                               ? fmt.at("type").get<std::string>()
                               : std::string();
        if (type != "text") {
            ApiError error;
            error.message = "only response_format {type:text} is supported";
            error.param   = "response_format";
            error.code    = "response_format_not_supported";
            throw ApiException(std::move(error));
        }
    }

    if (body.contains("logprobs") && !body.at("logprobs").is_null()) {
        if (!body.at("logprobs").is_boolean()) {
            bad_request("logprobs must be a boolean", "logprobs");
        }
        if (body.at("logprobs").get<bool>()) {
            bad_request("logprobs are not supported", "logprobs", "logprobs_not_supported");
        }
    }
    if (body.contains("top_logprobs") && !body.at("top_logprobs").is_null()) {
        const std::optional<int> value = get_int(body, "top_logprobs");
        if (!value || *value != 0) {
            bad_request("top_logprobs are not supported", "top_logprobs", "logprobs_not_supported");
        }
    }
    if (body.contains("parallel_tool_calls") && !body.at("parallel_tool_calls").is_null()) {
        if (!body.at("parallel_tool_calls").is_boolean()) {
            bad_request("parallel_tool_calls must be a boolean", "parallel_tool_calls");
        }
        if (!body.at("parallel_tool_calls").get<bool>()) {
            bad_request("parallel_tool_calls=false cannot be enforced", "parallel_tool_calls",
                        "parallel_tool_calls_not_supported");
        }
    }
    if (body.contains("modalities") && !body.at("modalities").is_null()) {
        const Json& modalities = body.at("modalities");
        if (!modalities.is_array() || modalities.size() != 1 || !modalities.at(0).is_string() ||
            modalities.at(0).get<std::string>() != "text") {
            bad_request("modalities only supports ['text']", "modalities",
                        "modality_not_supported");
        }
    }
    for (const char* key : {"audio", "prediction", "web_search_options"}) {
        if (body.contains(key) && !body.at(key).is_null()) {
            bad_request(std::string(key) + " is not supported", key, "parameter_not_supported");
        }
    }
    if (body.contains("store") && !body.at("store").is_null()) {
        if (!body.at("store").is_boolean()) { bad_request("store must be a boolean", "store"); }
        if (body.at("store").get<bool>()) {
            bad_request("store=true is not supported", "store", "store_not_supported");
        }
    }

    if (body.contains("service_tier") && !body.at("service_tier").is_null()) {
        if (!body.at("service_tier").is_string()) {
            bad_request("service_tier must be a string", "service_tier");
        }
        const std::string tier = body.at("service_tier").get<std::string>();
        if (tier != "auto" && tier != "default") {
            bad_request("only service_tier 'auto' or 'default' is supported", "service_tier",
                        "service_tier_not_supported");
        }
        // Local serving has one service tier; these default spellings are harmless SDK metadata.
    }
    if (body.contains("metadata") && !body.at("metadata").is_null()) {
        const Json& metadata = body.at("metadata");
        if (!metadata.is_object()) { bad_request("metadata must be an object", "metadata"); }
        if (metadata.size() > 16) {
            bad_request("metadata supports at most 16 entries", "metadata");
        }
        for (auto it = metadata.begin(); it != metadata.end(); ++it) {
            if (it.key().size() > 64 || !it.value().is_string() ||
                it.value().get_ref<const std::string&>().size() > 512) {
                bad_request("metadata keys must be at most 64 characters and string values at "
                            "most 512 characters",
                            "metadata");
            }
        }
        // Metadata is accepted for client correlation only and deliberately does not affect output.
    }
    if (body.contains("user") && !body.at("user").is_null()) {
        if (!body.at("user").is_string()) { bad_request("user must be a string", "user"); }
        // User identifiers are compatibility metadata; this local server has no attribution layer.
    }
    if (body.contains("prompt_cache_key") && !body.at("prompt_cache_key").is_null() &&
        !body.at("prompt_cache_key").is_string()) {
        bad_request("prompt_cache_key must be a string", "prompt_cache_key");
    }
    for (const char* key : {"safety_identifier", "verbosity"}) {
        if (body.contains(key) && !body.at(key).is_null()) {
            bad_request(std::string(key) + " is not supported", key, "parameter_not_supported");
        }
    }
}

Json base_chunk(const std::string& id, const std::string& model, std::int64_t created) {
    return Json{
        {"id", id}, {"object", "chat.completion.chunk"}, {"created", created}, {"model", model}};
}

// llama.cpp-compatible `timings` block: consumed by proxies (llama-swap) to
// populate per-request Prefill/Decode rates, draft acceptance, cache hits and
// phase durations. Shape matches llama.cpp's server `timings` JSON so existing
// metric parsers work unchanged.
// Slot identity next to `timings`: which lane served the request and, when that lane retained
// the finished session, its digest - what a client needs to target /slots operations safely.
void add_slot_identity(Json& payload, const CompletionUsage& usage) {
    if (usage.id_slot >= 0) { payload["id_slot"] = usage.id_slot; }
    if (!usage.session_digest.empty()) { payload["session_digest"] = usage.session_digest; }
}

void add_timings(Json& payload, const CompletionUsage& usage) {
    if (!usage.has_timings) { return; }
    Json timings = {
        {"prompt_n", usage.prompt_tokens},
        {"predicted_n", usage.completion_tokens},
        {"prompt_ms", usage.prefill_seconds * 1000.0},
        {"predicted_ms", usage.decode_seconds * 1000.0},
    };
    if (usage.prefill_seconds > 0.0) {
        timings["prompt_per_second"] = usage.prompt_tokens / usage.prefill_seconds;
    }
    if (usage.decode_seconds > 0.0) {
        timings["predicted_per_second"] = usage.completion_tokens / usage.decode_seconds;
    }
    if (usage.ttft_seconds > 0.0) {
        timings["ttft_ms"] = usage.ttft_seconds * 1000.0;
    }
    if (usage.cache_hit_tokens > 0) {
        timings["cache_n"] = usage.cache_hit_tokens;
    }
    if (usage.draft_tokens > 0) {
        timings["draft_n"]     = usage.draft_tokens;
        timings["draft_n_accepted"] = usage.accepted_tokens;
    }
    payload["timings"] = std::move(timings);
}

Json tool_calls_json(const std::vector<ToolCall>& tool_calls) {
    Json out = Json::array();
    for (const ToolCall& call : tool_calls) {
        out.push_back(Json{{"id", call.id},
                           {"type", "function"},
                           {"function", Json{{"name", call.name}, {"arguments", call.arguments_json}}}});
    }
    return out;
}

Json chat_tool_call_chunk(const std::string& id, const std::string& model, std::int64_t created,
                          Json tool_call, bool include_usage) {
    Json payload       = base_chunk(id, model, created);
    payload["choices"] = Json::array(
        {Json{{"index", 0},
              {"delta", Json{{"tool_calls", Json::array({std::move(tool_call)})}}},
              {"finish_reason", nullptr}}});
    if (include_usage) { payload["usage"] = nullptr; }
    return payload;
}

std::string sse_event(const Json& payload) { return "data: " + payload.dump() + "\n\n"; }

} // namespace

std::optional<bool> parse_openai_preserve_thinking(const Json& body) {
    std::optional<bool> top_level;
    if (body.contains("preserve_thinking") && !body.at("preserve_thinking").is_null()) {
        if (!body.at("preserve_thinking").is_boolean()) {
            bad_request("preserve_thinking must be a boolean or null", "preserve_thinking");
        }
        top_level = body.at("preserve_thinking").get<bool>();
    }

    std::optional<bool> template_value;
    if (body.contains("chat_template_kwargs")) {
        const Json& kwargs = body.at("chat_template_kwargs");
        if (!kwargs.is_object()) {
            bad_request("chat_template_kwargs must be an object", "chat_template_kwargs");
        }
        for (auto it = kwargs.begin(); it != kwargs.end(); ++it) {
            if (it.key() != "preserve_thinking" && !it.value().is_null()) {
                bad_request("chat_template_kwargs." + it.key() + " is not supported",
                            "chat_template_kwargs", "chat_template_option_not_supported");
            }
        }
        if (kwargs.contains("preserve_thinking") && !kwargs.at("preserve_thinking").is_null()) {
            if (!kwargs.at("preserve_thinking").is_boolean()) {
                bad_request("chat_template_kwargs.preserve_thinking must be a boolean or null",
                            "chat_template_kwargs");
            }
            template_value = kwargs.at("preserve_thinking").get<bool>();
        }
    }

    if (top_level && template_value && *top_level != *template_value) {
        bad_request("conflicting preserve_thinking values", "preserve_thinking",
                    "conflicting_template_option");
    }
    return template_value ? template_value : top_level;
}

void parse_openai_reasoning_effort(const Json& body, GenerationRequest& out) {
    if (!body.contains("reasoning_effort") || body.at("reasoning_effort").is_null()) { return; }
    if (!body.at("reasoning_effort").is_string()) {
        bad_request("reasoning_effort must be a string or null", "reasoning_effort");
    }
    const std::string value = body.at("reasoning_effort").get<std::string>();
    const std::optional<RequestedReasoningEffort> effort = parse_requested_reasoning_effort(value);
    if (!effort) {
        bad_request("reasoning_effort must be one of none, minimal, low, medium, high, xhigh, or "
                    "max",
                    "reasoning_effort");
    }
    out.reasoning_effort       = *effort;
    out.reasoning_effort_param = "reasoning_effort";
}

GenerationRequest parse_chat_completion_request(const Json& body, const RequestLimits& limits) {
    require_object(body);
    reject_unknown_top_level(body);
    reject_unsupported_features(body);

    GenerationRequest out;
    if (!body.contains("model") || !body.at("model").is_string() ||
        body.at("model").get<std::string>().empty()) {
        bad_request("missing required field: model", "model");
    }
    out.model = body.at("model").get<std::string>();
    if (body.contains("prompt_cache_key") && !body.at("prompt_cache_key").is_null()) {
        out.prompt_cache_routing_hint = body.at("prompt_cache_key").get<std::string>();
    }

    parse_tools(body, out);
    parse_tool_choice(body, out);
    parse_messages(body, out);
    parse_stop(body, out);
    parse_sampling(body, out);

    out.stream = get_bool(body, "stream", false);
    if (body.contains("stream_options") && !body.at("stream_options").is_null()) {
        if (!body.at("stream_options").is_object()) {
            bad_request("stream_options must be an object", "stream_options");
        }
        reject_unknown_fields(body.at("stream_options"), {"include_usage"}, "stream_options");
        out.include_usage = get_bool(body.at("stream_options"), "include_usage", false);
    }
    if (body.contains("enable_thinking") && !body.at("enable_thinking").is_null()) {
        out.enable_thinking = get_bool(body, "enable_thinking", false);
    }
    parse_openai_reasoning_effort(body, out);
    out.preserve_thinking = parse_openai_preserve_thinking(body);

    std::optional<int> max_tokens = get_int(body, "max_completion_tokens");
    if (!max_tokens) { max_tokens = get_int(body, "max_tokens"); }
    if (max_tokens) {
        if (*max_tokens <= 0) { bad_request("max_tokens must be positive", "max_tokens"); }
        out.max_tokens     = *max_tokens;
        out.max_tokens_set = true;
    } else {
        out.max_tokens     = limits.default_max_tokens;
        out.max_tokens_set = false;
    }
    return out;
}

std::string make_chat_completion_response(const std::string& id, const std::string& model,
                                          std::int64_t created, const std::string& content,
                                          const std::string& reasoning, const char* finish_reason,
                                          const CompletionUsage& usage) {
    Json message = {{"role", "assistant"}, {"content", content}};
    if (!reasoning.empty()) { message["reasoning_content"] = reasoning; }
    Json payload = {
        {"id", id},
        {"object", "chat.completion"},
        {"created", created},
        {"model", model},
        {"choices",
         Json::array({Json{
             {"index", 0}, {"message", std::move(message)}, {"finish_reason", finish_reason}}})},
        {"usage", Json{{"prompt_tokens", usage.prompt_tokens},
                       {"completion_tokens", usage.completion_tokens},
                       {"total_tokens", usage.prompt_tokens + usage.completion_tokens}}}};
    add_timings(payload, usage);
    add_slot_identity(payload, usage);
    return payload.dump();
}

std::string make_chat_completion_tool_response(const std::string& id, const std::string& model,
                                               std::int64_t created, const std::string& content,
                                               const std::string& reasoning,
                                               const std::vector<ToolCall>& tool_calls,
                                               const char* finish_reason,
                                               const CompletionUsage& usage) {
    Json message = {{"role", "assistant"},
                    {"content", content.empty() ? Json(nullptr) : Json(content)},
                    {"tool_calls", tool_calls_json(tool_calls)}};
    if (!reasoning.empty()) { message["reasoning_content"] = reasoning; }
    Json payload = {
        {"id", id},
        {"object", "chat.completion"},
        {"created", created},
        {"model", model},
        {"choices",
         Json::array({Json{
             {"index", 0}, {"message", std::move(message)}, {"finish_reason", finish_reason}}})},
        {"usage", Json{{"prompt_tokens", usage.prompt_tokens},
                       {"completion_tokens", usage.completion_tokens},
                       {"total_tokens", usage.prompt_tokens + usage.completion_tokens}}}};
    add_timings(payload, usage);
    add_slot_identity(payload, usage);
    return payload.dump();
}

std::string make_chat_chunk_role(const std::string& id, const std::string& model,
                                 std::int64_t created, bool include_usage) {
    Json payload       = base_chunk(id, model, created);
    payload["choices"] = Json::array({Json{{"index", 0},
                                           {"delta", Json{{"role", "assistant"}, {"content", ""}}},
                                           {"finish_reason", nullptr}}});
    if (include_usage) { payload["usage"] = nullptr; }
    return sse_event(payload);
}

std::string make_chat_chunk_reasoning(const std::string& id, const std::string& model,
                                      std::int64_t created, const std::string& delta_text,
                                      bool include_usage) {
    Json payload       = base_chunk(id, model, created);
    payload["choices"] = Json::array({Json{{"index", 0},
                                           {"delta", Json{{"reasoning_content", delta_text}}},
                                           {"finish_reason", nullptr}}});
    if (include_usage) { payload["usage"] = nullptr; }
    return sse_event(payload);
}

std::string make_chat_chunk_content(const std::string& id, const std::string& model,
                                    std::int64_t created, const std::string& delta_text,
                                    bool include_usage) {
    Json payload       = base_chunk(id, model, created);
    payload["choices"] = Json::array(
        {Json{{"index", 0}, {"delta", Json{{"content", delta_text}}}, {"finish_reason", nullptr}}});
    if (include_usage) { payload["usage"] = nullptr; }
    return sse_event(payload);
}

std::string make_chat_chunk_tool_call_begin(const std::string& id, const std::string& model,
                                            std::int64_t created, std::size_t index,
                                            const ToolCall& call, bool include_usage) {
    return sse_event(chat_tool_call_chunk(
        id, model, created,
        Json{{"index", index},
             {"id", call.id},
             {"type", "function"},
             {"function", Json{{"name", call.name}, {"arguments", ""}}}},
        include_usage));
}

std::string make_chat_chunk_tool_call_arguments(const std::string& id, const std::string& model,
                                                std::int64_t created, std::size_t index,
                                                const std::string& delta, bool include_usage) {
    return sse_event(chat_tool_call_chunk(
        id, model, created, Json{{"index", index}, {"function", Json{{"arguments", delta}}}},
        include_usage));
}

std::string make_chat_chunk_final(const std::string& id, const std::string& model,
                                  std::int64_t created, const char* finish_reason,
                                  bool include_usage, const CompletionUsage& usage) {
    Json payload       = base_chunk(id, model, created);
    payload["choices"] = Json::array(
        {Json{{"index", 0}, {"delta", Json::object()}, {"finish_reason", finish_reason}}});
    if (include_usage) { payload["usage"] = nullptr; }
    add_timings(payload, usage);
    add_slot_identity(payload, usage);
    return sse_event(payload);
}

std::string make_chat_chunk_usage(const std::string& id, const std::string& model,
                                  std::int64_t created, const CompletionUsage& usage) {
    Json payload       = base_chunk(id, model, created);
    payload["choices"] = Json::array();
    payload["usage"]   = Json{{"prompt_tokens", usage.prompt_tokens},
                              {"completion_tokens", usage.completion_tokens},
                              {"total_tokens", usage.prompt_tokens + usage.completion_tokens}};
    add_timings(payload, usage);
    add_slot_identity(payload, usage);
    return sse_event(payload);
}

std::string sse_done() { return "data: [DONE]\n\n"; }

std::string make_models_list(const std::string& model_id,
                             const std::vector<std::string>& adapter_model_ids,
                             std::int64_t created, std::uint32_t context_window, bool vision) {
    const auto entry = [&](const std::string& id) {
        return Json{{"id", id},
                    {"object", "model"},
                    {"created", created},
                    {"owned_by", "ninfer"},
                    {"context_window", context_window},
                    {"modalities", Json{{"vision", vision}}}};
    };
    Json data = Json::array({entry(model_id)});
    for (const std::string& id : adapter_model_ids) { data.push_back(entry(id)); }
    const Json payload = {{"object", "list"}, {"data", std::move(data)}};
    return payload.dump();
}

std::string make_model_object(const std::string& model_id, std::int64_t created,
                              std::uint32_t context_window, bool vision) {
    const Json payload = {{"id", model_id},
                          {"object", "model"},
                          {"created", created},
                          {"owned_by", "ninfer"},
                          {"context_window", context_window},
                          {"modalities", Json{{"vision", vision}}}};
    return payload.dump();
}

std::string make_error_body(const ApiError& error) {
    Json err     = {{"message", error.message}, {"type", error.type}};
    err["param"] = error.param.empty() ? Json(nullptr) : Json(error.param);
    err["code"]  = error.code.empty() ? Json(nullptr) : Json(error.code);
    return Json{{"error", err}}.dump();
}

std::string new_chat_completion_id() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> dist;
    std::array<char, 32> buf{};
    std::snprintf(buf.data(), buf.size(), "%016llx", static_cast<unsigned long long>(dist(rng)));
    return "chatcmpl-" + std::string(buf.data());
}

std::int64_t unix_time_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

} // namespace ninfer::serve
