#include "serve/openai_decisions_service.h"

#include "product/openai_decisions/answers.h"
#include "serve/request.h"

#include <chrono>
#include <utility>

namespace ninfer::serve {
namespace {

[[noreturn]] void throw_media_error(const DecisionMediaError& exception,
                                    const std::vector<std::string>& params) {
    const bool budget = exception.kind() == DecisionMediaErrorKind::BudgetExceeded;
    throw ApiException({.status  = budget ? 413 : 400,
                        .type    = "invalid_request_error",
                        .message = exception.what(),
                        .param   = params.at(exception.image_index()),
                        .code    = budget ? "media_budget_exceeded" : "invalid_media"});
}

[[noreturn]] void throw_input_error(const DecisionInputError& exception) {
    throw ApiException({.status  = 400,
                        .type    = "invalid_request_error",
                        .message = exception.what(),
                        .param   = "input",
                        .code    = "context_length_exceeded"});
}

[[noreturn]] void throw_engine_error(const RequestError& exception) {
    ApiError error;
    error.message = exception.what();
    switch (exception.kind()) {
    case RequestErrorKind::ContextLengthExceeded:
        error.param = "input";
        error.code  = "context_length_exceeded";
        break;
    case RequestErrorKind::MediaBudgetExceeded:
        error.status = 413;
        error.param  = "input";
        error.code   = "media_budget_exceeded";
        break;
    case RequestErrorKind::UnknownAdapter:
        error.status = 404;
        error.param  = "model";
        error.code   = "model_not_found";
        break;
    case RequestErrorKind::Overloaded:
        error.status = 429;
        error.type   = "rate_limit_error";
        error.code   = "server_overloaded";
        break;
    case RequestErrorKind::QueueTimeout:
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case RequestErrorKind::Unavailable:
        error.status = 503;
        error.type   = "server_error";
        error.code   = "service_unavailable";
        break;
    case RequestErrorKind::Cancelled:
        error.status = 499;
        error.type   = "request_cancelled";
        error.code   = "client_disconnected";
        break;
    }
    throw ApiException(std::move(error));
}

}

OpenAIDecisionsService::OpenAIDecisionsService(GenerationService& generation,
                                               const DecisionModels& models)
    : generation_(generation), models_(models) {}

PreparedOpenAIDecision OpenAIDecisionsService::prepare(std::string_view body,
                                                       std::function<bool()> is_cancelled) const {
    namespace oa        = product::openai_decisions;
    const auto received = std::chrono::steady_clock::now();
    oa::Request request;
    try {
        request = oa::parse_request(body);
    } catch (const oa::RequestValidationError& error) {
        throw ApiException({.status  = error.status(),
                            .type    = "invalid_request_error",
                            .message = error.what(),
                            .param   = error.param(),
                            .code    = error.code()});
    }
    const LoraAdapterInfo* adapter = models_.find(request.model);
    if (adapter == nullptr) {
        throw ApiException({.status  = 404,
                            .type    = "invalid_request_error",
                            .message = "decision model '" + request.model + "' not found",
                            .param   = "model",
                            .code    = "model_not_found"});
    }
    PreparedOpenAIDecision prepared;
    prepared.model     = adapter->name;
    prepared.questions = std::move(request.questions);
    for (const auto& part : request.input.state) {
        if (part.kind == DecisionPartKind::Image) { prepared.image_params.push_back(part.param); }
    }
    const double parse_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - received).count();
    try {
        prepared.decision = generation_.prepare_decision(std::move(request.input), adapter->name,
                                                         std::move(is_cancelled));
    } catch (const DecisionMediaError& error) {
        throw_media_error(error, prepared.image_params);
    } catch (const DecisionInputError& error) {
        throw_input_error(error);
    } catch (const RequestError& error) { throw_engine_error(error); }
    prepared.decision.prepare_seconds += parse_seconds;
    return prepared;
}

OpenAIDecisionOutcome OpenAIDecisionsService::run(PreparedOpenAIDecision& prepared,
                                                  std::function<bool()> is_cancelled) {
    OpenAIDecisionOutcome outcome;
    try {
        outcome.result = generation_.decide(prepared.decision, std::move(is_cancelled));
    } catch (const DecisionMediaError& error) {
        throw_media_error(error, prepared.image_params);
    } catch (const DecisionInputError& error) {
        throw_input_error(error);
    } catch (const RequestError& error) { throw_engine_error(error); }
    outcome.body = product::openai_decisions::response_body(prepared.model, outcome.result,
                                                            prepared.questions);
    return outcome;
}

}
