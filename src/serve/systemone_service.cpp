#include "serve/systemone_service.h"

#include "product/systemone/answers.h"
#include "product/systemone/models.h"
#include "serve/request_log.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <optional>
#include <random>
#include <utility>

namespace ninfer::serve {
namespace {

namespace so = ninfer::product::systemone;
using Json   = nlohmann::ordered_json;
using Clock  = std::chrono::steady_clock;

std::string error_message(const Json& detail) {
    return detail.is_string() ? detail.get<std::string>() : detail.dump();
}

[[noreturn]] void throw_engine_error(const ninfer::RequestError& error) {
    throw SystemOneError(so::request_error_status(error.kind()), error.what());
}

} // namespace

SystemOneError::SystemOneError(int status, nlohmann::ordered_json detail)
    : std::runtime_error(error_message(detail)), status_(status), detail_(std::move(detail)) {}

std::string SystemOneError::body() const { return so::error_body(detail_); }

std::string new_typesafe_request_id() {
    static thread_local std::mt19937_64 generator{std::random_device{}()};
    std::array<std::uint8_t, 16> bytes{};
    for (std::size_t index = 0; index < bytes.size(); index += 8) {
        const std::uint64_t word = generator();
        for (std::size_t byte = 0; byte < 8; ++byte) {
            bytes[index + byte] = static_cast<std::uint8_t>(word >> (8 * byte));
        }
    }
    bytes[6] = static_cast<std::uint8_t>((bytes[6] & 0x0FU) | 0x40U);
    bytes[8] = static_cast<std::uint8_t>((bytes[8] & 0x3FU) | 0x80U);
    static constexpr char kHex[] = "0123456789abcdef";
    std::string id;
    id.reserve(32);
    for (const std::uint8_t value : bytes) {
        id += kHex[value >> 4U];
        id += kHex[value & 0x0FU];
    }
    return id;
}

SystemOneService::SystemOneService(GenerationService& generation, const ServeOptions& options,
                                   std::string public_model_id, const DecisionModels& models)
    : generation_(generation), models_(models), public_model_id_(std::move(public_model_id)) {
    binding_ = so::alias_binding(models_.adapters(), options.systemone_default);
    const ninfer::MemorySummary memory = generation.memory_summary();
    kv_cache_                          = memory.kv_cache;
    max_context_                       = memory.max_context;
    prefix_reuse_                      = options.allow_prefix_reuse;
}

nlohmann::ordered_json SystemOneService::card(std::string_view name,
                                              const ninfer::LoraAdapterInfo& adapter) const {
    std::string description = adapter.description;
    if (description.empty()) {
        std::array<char, 32> temperature{};
        std::snprintf(temperature.data(), temperature.size(), "%.2f",
                      static_cast<double>(adapter.temperature));
        description = "System One pointer head on " + public_model_id_ + ", serving " +
                      adapter.name + " at temperature " + temperature.data();
    }
    Json card            = Json::object();
    card["name"]         = std::string(name);
    card["description"]  = std::move(description);
    card["release_date"] = adapter.release_date;
    card["adapter"]      = adapter.name;
    card["base"]         = public_model_id_;
    card["rank"]         = adapter.rank;
    card["temperature"]  = shortest_double(adapter.temperature);
    card["kv_cache"]     = kv_cache_name(kv_cache_);
    card["max_context"]  = max_context_;
    card["prefix_reuse"] = prefix_reuse_;
    return card;
}

std::string SystemOneService::models_body() const {
    Json models = Json::array();
    for (const ninfer::LoraAdapterInfo& adapter : models_.adapters()) {
        models.push_back(card(adapter.name, adapter));
    }
    for (const ninfer::LoraAdapterInfo& adapter : models_.adapters()) {
        if (adapter.name == binding_) { models.push_back(card(so::kDefaultModel, adapter)); }
    }
    Json body      = Json::object();
    body["models"] = std::move(models);
    return so::python_json_dumps(body, so::kFastApi);
}

PreparedSystemOne SystemOneService::prepare(std::string_view body) const {
    const Clock::time_point received = Clock::now();
    so::Request request;
    try {
        request = so::parse_request(body);
    } catch (const so::RequestValidationError& error) {
        throw SystemOneError(error.status(), error.detail());
    }
    const LoraAdapterInfo* adapter = models_.find(request.model);
    if (adapter == nullptr && request.model == so::kDefaultModel && !binding_.empty()) {
        adapter = models_.find(binding_);
    }
    if (adapter == nullptr) {
        throw SystemOneError(404,
                             so::unknown_model_detail(request.model, models_.adapters(), binding_));
    }

    PreparedSystemOne prepared;
    prepared.model     = std::move(request.model);
    prepared.adapter           = adapter->name;
    prepared.questions = std::move(request.questions);
    const double parse_seconds = std::chrono::duration<double>(Clock::now() - received).count();
    try {
        prepared.decision = generation_.prepare_decision(std::move(request.input), adapter->name);
    } catch (const ninfer::DecisionInputError& error) {
        throw SystemOneError(422, error.what());
    } catch (const ninfer::RequestError& error) { throw_engine_error(error); }
    prepared.decision.prepare_seconds += parse_seconds;
    return prepared;
}

SystemOneOutcome SystemOneService::run(PreparedSystemOne& prepared,
                                       std::function<bool()> is_cancelled) {
    SystemOneOutcome outcome;
    try {
        outcome.result = generation_.decide(prepared.decision, std::move(is_cancelled));
    } catch (const ninfer::DecisionInputError& error) {
        throw SystemOneError(422, error.what());
    } catch (const ninfer::RequestError& error) { throw_engine_error(error); }
    const Json answers    = so::to_answers(outcome.result.probabilities, prepared.questions);
    outcome.output_tokens = generation_.count_text_tokens(so::python_json_dumps(answers));
    outcome.body          = so::response_body(prepared.adapter, answers,
                                              outcome.result.summary.input_tokens(),
                                              outcome.output_tokens,
                                              outcome.result.timings.engine_seconds() * 1000.0);
    return outcome;
}

} // namespace ninfer::serve
