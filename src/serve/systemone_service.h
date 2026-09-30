#pragma once

// The System One (TypeSafe) decision protocol over the served Engine: kev's request semantics and
// response bodies (src/product/systemone), model names over the pool's decision adapters, and the
// FastAPI statuses and {"detail": ...} bodies the TypeSafe SDK reads.

#include "ninfer/types.h"
#include "product/systemone/request.h"
#include "serve/generation_service.h"
#include "serve/serve_options.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// A failed System One request: its status and FastAPI's `detail`.
class SystemOneError final : public std::runtime_error {
public:
    SystemOneError(int status, nlohmann::ordered_json detail);

    [[nodiscard]] int status() const noexcept { return status_; }

    [[nodiscard]] const nlohmann::ordered_json& detail() const noexcept { return detail_; }

    // {"detail": detail}.
    [[nodiscard]] std::string body() const;

private:
    int status_;
    nlohmann::ordered_json detail_;
};

// A fresh x-typesafe-request-id for a request that brought none: uuid4 hex, as kev's
// uuid.uuid4().hex.
[[nodiscard]] std::string new_typesafe_request_id();

// A request parsed, answered by a decision adapter and submitted to the Engine FIFO.
struct PreparedSystemOne {
    // The model the request names, and the decision adapter that answers it.
    std::string model;
    std::string adapter;
    std::vector<product::systemone::QuestionMeta> questions;
    PreparedDecisionRequest decision;
};

struct SystemOneOutcome {
    std::string body;
    ninfer::DecisionResult result;
    // kev's usage.output_tokens: the token count of json.dumps(answers).
    std::uint32_t output_tokens = 0;
};

class SystemOneService {
public:
    // Binds the SDK-default model name to options.systemone_default, else to the pool's only
    // decision adapter. Throws std::invalid_argument when options.systemone_default names no
    // decision adapter of the pool.
    SystemOneService(GenerationService& generation, const ServeOptions& options,
                     std::string public_model_id);

    // The decision adapter the SDK-default model name answers with; empty when unbound.
    [[nodiscard]] const std::string& alias_binding() const noexcept { return binding_; }

    // GET /v1/models: one card per decision adapter, then the bound alias's.
    [[nodiscard]] std::string models_body() const;

    // Validates the body as kev does, resolves its model and submits the decision. Throws
    // SystemOneError.
    [[nodiscard]] PreparedSystemOne prepare(std::string_view body) const;

    // Waits for the decision and renders kev's response. Throws SystemOneError.
    [[nodiscard]] SystemOneOutcome run(PreparedSystemOne& prepared,
                                       std::function<bool()> is_cancelled);

private:
    [[nodiscard]] nlohmann::ordered_json card(std::string_view name,
                                              const ninfer::LoraAdapterInfo& adapter) const;

    GenerationService& generation_;
    std::vector<ninfer::LoraAdapterInfo> adapters_;
    std::string public_model_id_;
    std::string binding_;
    KvCacheStorage kv_cache_   = KvCacheStorage::BFloat16;
    std::uint32_t max_context_ = 0;
    bool prefix_reuse_         = true;
};

} // namespace ninfer::serve
