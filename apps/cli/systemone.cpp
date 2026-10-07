#include "systemone.h"

#include "product/systemone/answers.h"
#include "product/systemone/models.h"
#include "product/systemone/request.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::cli {
namespace {

namespace so = ninfer::product::systemone;
using Json   = nlohmann::ordered_json;

std::string read_input(const std::filesystem::path& path) {
    std::ostringstream buffer;
    if (path == "-") {
        buffer << std::cin.rdbuf();
    } else {
        std::ifstream file(path, std::ios::binary);
        if (!file) { throw std::invalid_argument("cannot read System One input " + path.string()); }
        buffer << file.rdbuf();
    }
    return buffer.str();
}

std::vector<std::string> request_bodies(const std::string& text, bool jsonl) {
    if (!jsonl) { return {text}; }
    std::vector<std::string> bodies;
    std::size_t begin = 0;
    while (begin < text.size()) {
        std::size_t end = text.find('\n', begin);
        if (end == std::string::npos) { end = text.size(); }
        std::string_view line(text.data() + begin, end - begin);
        if (!line.empty() && line.back() == '\r') { line.remove_suffix(1); }
        if (line.find_first_not_of(" \t") != std::string_view::npos) { bodies.emplace_back(line); }
        begin = end + 1;
    }
    return bodies;
}

// A failed request: its status and the FastAPI body {"detail": detail}.
struct Failure {
    int status = 500;
    Json detail;
};

Json prepared_json(const PreparedDecision& prepared) {
    Json branches = Json::array();
    for (const DecisionBranch& branch : prepared.branches()) {
        Json entry               = Json::object();
        entry["begin"]           = prepared.state_token_ids().size() + branch.begin;
        entry["length"]          = branch.length;
        entry["option_readouts"] = branch.option_readouts;
        branches.push_back(std::move(entry));
    }
    const auto state           = prepared.state_token_ids();
    const auto branches_tokens = prepared.branch_token_ids();
    std::vector<TokenId> tokens(state.begin(), state.end());
    tokens.insert(tokens.end(), branches_tokens.begin(), branches_tokens.end());
    Json out                              = Json::object();
    out["format"]                         = "ninfer-decision-prepared";
    out["format_version"]                 = 1;
    out["state_tokens"]                   = prepared.summary().state_tokens;
    out["state_truncated"]                = prepared.summary().state_truncated;
    out["tokens"]                         = std::move(tokens);
    out["branches"]                       = std::move(branches);
    return out;
}

std::string_view source_name(ContinuationSource source) noexcept {
    switch (source) {
    case ContinuationSource::L1: return "l1";
    case ContinuationSource::L2: return "l2";
    case ContinuationSource::L3: return "l3";
    case ContinuationSource::None: return "none";
    }
    return "none";
}

// The parity-tool fields beside a successful response: the unrounded FP32 probabilities and the
// engine's accounting of the decision.
Json decision_json(const DecisionResult& result) {
    Json probabilities = Json::array();
    for (const std::vector<float>& question : result.probabilities) {
        Json row = Json::array();
        for (const float probability : question) { row.push_back(static_cast<double>(probability)); }
        probabilities.push_back(std::move(row));
    }
    Json timings                 = Json::object();
    timings["prepare_seconds"]   = result.timings.prepare_seconds;
    timings["queue_seconds"]     = result.timings.queue_seconds;
    timings["restore_seconds"]   = result.timings.restore_seconds;
    timings["state_seconds"]     = result.timings.state_seconds;
    timings["branch_seconds"]    = result.timings.branch_seconds;
    timings["execution_seconds"] = result.timings.execution_seconds;
    timings["total_seconds"]     = result.timings.total_seconds;

    Json out                   = Json::object();
    out["probabilities"]       = std::move(probabilities);
    out["adapter"]             = result.adapter;
    out["slot"]                = result.slot;
    out["state_tokens"]        = result.summary.state_tokens;
    out["branch_tokens"]       = result.summary.branch_tokens;
    out["state_truncated"]     = result.summary.state_truncated;
    out["reused_state_tokens"] = result.reused_state_tokens;
    out["state_source"]        = source_name(result.state_source);
    out["branch_passes"]       = result.branch_passes;
    out["long_branch_chunks"]  = result.long_branch_chunks;
    out["timings"]             = std::move(timings);
    return out;
}

// {"status": status, "response": <body text>[, decision fields]}: the response text is spliced in
// unchanged so it stays byte-identical to what the server sends.
std::string tool_line(int status, const std::string& body, const DecisionResult* result) {
    std::string line = "{\"status\":" + std::to_string(status) + ",\"response\":" + body;
    if (result != nullptr) {
        const std::string fields = decision_json(*result).dump();
        line += ',';
        line.append(fields, 1, std::string::npos);
    } else {
        line += '}';
    }
    return line;
}

} // namespace

int run_systemone(Engine& engine, const Options& options) {
    const LoadSummary load = engine.load_summary();
    if (!load.decisions_supported) {
        throw std::invalid_argument("target '" + load.target + "' does not serve decisions");
    }
    const std::string binding = so::alias_binding(load.lora_adapters, options.systemone_default);
    std::ofstream dump;
    if (!options.systemone_dump_prepared.empty()) {
        dump.open(options.systemone_dump_prepared, std::ios::binary | std::ios::trunc);
        if (!dump) {
            throw std::invalid_argument("cannot write " +
                                        options.systemone_dump_prepared.string());
        }
    }
    const std::vector<std::string> bodies =
        request_bodies(read_input(options.systemone_path), options.systemone_jsonl);
    if (bodies.empty()) { throw std::invalid_argument("System One input holds no request"); }

    int exit_status = 0;
    for (std::size_t index = 0; index < bodies.size(); ++index) {
        std::optional<Failure> failure;
        std::optional<DecisionResult> result;
        std::string response;
        try {
            so::Request request = so::parse_request(bodies[index]);
            const std::optional<std::string> adapter =
                so::resolve_model(request.model, load.lora_adapters, binding);
            if (!adapter) {
                failure = Failure{404, so::unknown_model_detail(request.model, load.lora_adapters,
                                                                binding)};
            } else {
                PreparedDecision prepared = engine.prepare_decision(std::move(request.input));
                if (dump.is_open()) { dump << prepared_json(prepared).dump() << '\n'; }
                result = engine.decide(std::move(prepared),
                                       DecisionOptions{.adapter            = *adapter,
                                                       .allow_prefix_reuse = !options.systemone_cold});
                const Json answers = so::to_answers(result->probabilities, request.questions);
                const std::uint32_t output_tokens =
                    engine.count_text_tokens(so::python_json_dumps(answers));
                response = so::response_body(*adapter, answers, result->summary.input_tokens(),
                                             output_tokens,
                                             result->timings.engine_seconds() * 1000.0);
            }
        } catch (const so::RequestValidationError& error) {
            failure = Failure{error.status(), error.detail()};
        } catch (const DecisionInputError& error) {
            failure = Failure{422, error.what()};
        } catch (const RequestError& error) {
            failure = Failure{so::request_error_status(error.kind()), error.what()};
        }

        const int status        = failure ? failure->status : 200;
        const std::string body  = failure ? so::error_body(failure->detail) : response;
        std::cout << (options.systemone_probabilities
                          ? tool_line(status, body, failure ? nullptr : &*result)
                          : body)
                  << '\n'
                  << std::flush;
        std::cerr << std::left << std::setw(12) << "systemone" << std::setw(26)
                  << ("request " + std::to_string(index + 1)) << status;
        if (result) {
            std::cerr << " " << result->adapter << " questions " << result->summary.questions
                      << " state " << result->summary.state_tokens << " (reused "
                      << result->reused_state_tokens << ") branches "
                      << result->summary.branch_tokens << " passes " << result->branch_passes
                      << " long chunks " << result->long_branch_chunks << " latency "
                      << std::fixed << std::setprecision(1)
                      << result->timings.engine_seconds() * 1000.0 << " ms"
                      << std::defaultfloat;
        } else if (failure) {
            std::cerr << ' ' << failure->detail.dump();
        }
        std::cerr << '\n';
        if (failure) { exit_status = 1; }
    }
    return exit_status;
}

} // namespace ninfer::cli
