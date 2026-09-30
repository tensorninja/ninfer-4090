// System One decisions against the real 27B artifact and a trained decision adapter.
//
// These are the engine-level P3 gates that tools/parity/qwen3_8_27b/decision.py cannot reach
// through the CLI: a repeated cold decision is bit-identical; a state published to L2 at completion
// restores bit-identically once L1 lost it; a decision that continues a retained
// state, whole or as a prefix of a longer one, took the restore path and is behaviourally
// equivalent to computing it cold; chat greedy output is bit-identical while decisions prefill
// beside it, and so are those decisions; a cancelled or abandoned decision releases its lane;
// adapter kinds do not cross; a state that cannot fit a lane is rejected before submission.
//
// The pass width is 256 tokens, so a modest state spans several chunks, a question of many options
// takes the long-branch path, and a chat prompt's chunks interleave with a decision's.
#include "ninfer/engine.h"

#include <unistd.h> // getpid, for a per-process temporary pool directory

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr const char* kDeciderName     = "decider";
constexpr std::uint32_t kPrefillChunk  = 256;
constexpr std::uint32_t kMaxContext    = 4096;
constexpr std::uint32_t kConcurrency   = 4;
constexpr std::uint32_t kChatTokens    = 16;
// The plan's served-vs-reference bar (kev's): max |dp| of any option of any question.
constexpr float kBehaviouralBar = 0.03F;

class PoolDirectory {
public:
    explicit PoolDirectory(const char* label) {
        root_ = std::filesystem::temp_directory_path() /
                ("ninfer_decision_pool_" + std::string(label) + "_" + std::to_string(::getpid()));
        std::error_code error;
        std::filesystem::remove_all(root_, error);
        std::filesystem::create_directories(root_);
    }

    ~PoolDirectory() {
        std::error_code error;
        std::filesystem::remove_all(root_, error);
    }

    PoolDirectory(const PoolDirectory&)            = delete;
    PoolDirectory& operator=(const PoolDirectory&) = delete;

    PoolDirectory& add(const char* name, const char* path) {
        std::filesystem::create_symlink(std::filesystem::absolute(path),
                                        root_ / (std::string(name) + ".lora.ninfer"));
        return *this;
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return root_; }

private:
    std::filesystem::path root_;
};

int fail(const std::string& message) {
    std::cerr << message << '\n';
    return 1;
}

// Distinct sentences joined by single spaces with no trailing space, so a longer text built from
// the same seed extends the token sequence of a shorter one.
std::string ticket_text(std::uint32_t sentences, std::uint32_t seed) {
    static constexpr const char* kSubjects[] = {"The customer", "Our courier", "The warehouse",
                                                "Support", "The card issuer"};
    static constexpr const char* kVerbs[]    = {"reported", "confirmed", "disputed", "escalated",
                                                "closed"};
    static constexpr const char* kObjects[]  = {"a late delivery", "two charges on one card",
                                                "a crushed box", "the wrong shoe size",
                                                "a missing invoice"};
    std::string text;
    for (std::uint32_t index = 0; index < sentences; ++index) {
        const std::uint32_t key = index * 7U + seed * 13U;
        if (index != 0) { text += ' '; }
        text += std::string(kSubjects[key % 5U]) + ' ' + kVerbs[(key / 5U) % 5U] + ' ' +
                kObjects[(key / 25U) % 5U] + " on day " + std::to_string(index + 1U) + '.';
    }
    return text;
}

std::vector<ninfer::DecisionQuestion> triage_questions() {
    return {
        {.instructions = "Which team should handle this?",
         .options      = {"returns: Exchanges, refunds, wrong or damaged items",
                          "shipping: Delivery status, delays, lost packages",
                          "billing: Charges, invoices, payment problems"}},
        {.instructions = "Does this message require urgent human attention?",
         .options      = {"false", "true"}},
        {.instructions = "How frustrated is the customer?",
         .options      = {"Calm", "Frustrated", "Very angry"}},
    };
}

// A question whose branch is several pass widths long.
ninfer::DecisionQuestion catalogue_question() {
    ninfer::DecisionQuestion question{.instructions = "Which catalogue entry matches the ticket?",
                                      .options      = {}};
    for (int entry = 0; entry < 48; ++entry) {
        question.options.push_back("entry_" + std::to_string(entry) +
                                   ": Issue concerning catalogue entry number " +
                                   std::to_string(entry) + " and its follow-up");
    }
    return question;
}

// A multi-chunk state, three packed questions and one long branch.
ninfer::DecisionInput mixed_decision(std::uint32_t sentences = 40, std::uint32_t seed = 1) {
    ninfer::DecisionInput input{.state = ticket_text(sentences, seed), .questions = triage_questions()};
    input.questions.push_back(catalogue_question());
    return input;
}

// A state of about 3,000 tokens: long enough that a decision cannot finish before a cancellation
// is observed, short enough to leave a lane room for its branch scratch.
ninfer::DecisionInput long_decision() {
    return {.state = ticket_text(250, 4), .questions = triage_questions()};
}

ninfer::DecisionOptions decider(bool reuse) {
    return {.adapter = kDeciderName, .allow_prefix_reuse = reuse};
}

ninfer::DecisionResult decide(ninfer::Engine& engine, const ninfer::DecisionInput& input,
                              bool reuse) {
    return engine.decide(engine.prepare_decision(input), decider(reuse));
}

bool well_formed(const ninfer::DecisionResult& result, const ninfer::DecisionInput& input) {
    if (result.probabilities.size() != input.questions.size()) { return false; }
    for (std::size_t question = 0; question < input.questions.size(); ++question) {
        const std::vector<float>& row = result.probabilities[question];
        if (row.size() != input.questions[question].options.size()) { return false; }
        double sum = 0.0;
        for (const float probability : row) {
            if (!std::isfinite(probability) || probability < 0.0F || probability > 1.0F) {
                return false;
            }
            sum += probability;
        }
        if (std::fabs(sum - 1.0) > 1e-4) { return false; }
    }
    return true;
}

float max_difference(const ninfer::DecisionResult& left, const ninfer::DecisionResult& right) {
    float worst = 0.0F;
    for (std::size_t question = 0; question < left.probabilities.size(); ++question) {
        for (std::size_t option = 0; option < left.probabilities[question].size(); ++option) {
            worst = std::max(worst, std::fabs(left.probabilities[question][option] -
                                              right.probabilities[question][option]));
        }
    }
    return worst;
}

ninfer::PromptInput chat_prompt() {
    ninfer::ChatMessage message;
    message.role = "user";
    message.parts.push_back(ninfer::MessagePart{
        .kind  = ninfer::MessagePartKind::Text,
        .text  = ticket_text(60, 5) + "\n\nIn one sentence, what went wrong for the customer?",
        .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

ninfer::RequestOptions greedy(std::uint32_t output_tokens, std::optional<std::string> adapter) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = output_tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.adapter                 = std::move(adapter);
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    return options;
}

int verify_registration(const ninfer::Engine& engine) {
    const ninfer::LoadSummary load = engine.load_summary();
    if (!load.decisions_supported) { return fail("the 27B target does not report decisions"); }
    for (const ninfer::LoraAdapterInfo& adapter : load.lora_adapters) {
        if (adapter.name != kDeciderName) { continue; }
        if (adapter.kind != ninfer::LoraAdapterKind::Decision || adapter.pointer_dim != 256 ||
            !std::isfinite(adapter.temperature) || adapter.temperature <= 0.0F) {
            return fail("the decision adapter is not registered as one");
        }
        return 0;
    }
    return fail("the decision adapter was not discovered");
}

// Cold decisions share nothing but the weights, so the same request is the same arithmetic.
int verify_cold_repeat_is_bit_identical(ninfer::Engine& engine) {
    const ninfer::DecisionInput input  = mixed_decision();
    const ninfer::DecisionResult first  = decide(engine, input, false);
    const ninfer::DecisionResult second = decide(engine, input, false);
    if (!well_formed(first, input) || !well_formed(second, input)) {
        return fail("a cold decision returned malformed probabilities");
    }
    if (first.summary.state_tokens <= kPrefillChunk || first.long_branch_chunks < 2 ||
        first.branch_passes < 1) {
        return fail("the mixed decision does not span several state chunks and a long branch");
    }
    if (first.reused_state_tokens != 0 || second.reused_state_tokens != 0) {
        return fail("a cold decision reused a state");
    }
    if (first.probabilities != second.probabilities) {
        return fail("a repeated cold decision is not bit-identical");
    }
    return 0;
}

// A state computed cold is published to L2 when its decision completes, and once L1 no longer
// holds it the same decision imports it from L2. The image is a byte copy of the state the cold
// run's branches read, so the probabilities are bit-identical rather than within the bar. Runs
// before any other decision that publishes, so the publication it waits for is its own.
int verify_state_restores_from_l2(ninfer::Engine& engine) {
    const ninfer::DecisionInput input = mixed_decision(40, 9);
    const std::uint64_t published     = engine.runtime_stats().continuation_publication_successes;
    const ninfer::DecisionResult cold = decide(engine, input, true);
    if (!well_formed(cold, input) || cold.state_source != ninfer::ContinuationSource::None ||
        cold.reused_state_tokens != 0 || cold.slot < 0) {
        return fail("the first decision on a state did not compute and retain it cold");
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (engine.runtime_stats().continuation_publication_successes == published) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return fail("the decision state was not published to L2");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (engine.erase_slot(static_cast<std::uint32_t>(cold.slot)) != cold.summary.state_tokens) {
        return fail("the retained decision state was not evicted from L1");
    }

    const std::uint64_t l2_before = engine.runtime_stats().continuation_l2_restore_successes;
    const ninfer::DecisionResult restored = decide(engine, input, true);
    if (restored.state_source != ninfer::ContinuationSource::L2 ||
        restored.reused_state_tokens != restored.summary.state_tokens ||
        restored.timings.restore_seconds <= 0.0) {
        return fail("a decision whose state left L1 did not restore it from L2");
    }
    if (engine.runtime_stats().continuation_l2_restore_successes != l2_before + 1) {
        return fail("the L2 decision restore was not counted");
    }
    if (restored.probabilities != cold.probabilities) {
        return fail("a decision on a state restored from L2 is not bit-identical to its cold run");
    }
    return 0;
}

// A retained state continues whole (the same state) and as the prefix of a longer state. Restore
// reproduces the input semantics of computing the state cold, not its chunking, so the bar is
// behavioural.
int verify_restored_state_is_equivalent(ninfer::Engine& engine) {
    const ninfer::DecisionInput shorter = mixed_decision(40, 1);
    ninfer::DecisionInput longer        = mixed_decision(40, 1);
    longer.state += ' ' + ticket_text(30, 2);

    const ninfer::DecisionResult cold_shorter = decide(engine, shorter, false);
    const ninfer::DecisionResult cold_longer  = decide(engine, longer, false);
    static_cast<void>(decide(engine, shorter, true)); // Computes and retains the shorter state.

    const ninfer::DecisionResult whole = decide(engine, shorter, true);
    if (whole.state_source != ninfer::ContinuationSource::L1 ||
        whole.reused_state_tokens != whole.summary.state_tokens) {
        return fail("a repeated state did not continue the retained one whole");
    }
    if (max_difference(whole, cold_shorter) > kBehaviouralBar) {
        return fail("a decision on a whole retained state moved beyond the bar");
    }

    const ninfer::DecisionResult prefix = decide(engine, longer, true);
    if (prefix.state_source != ninfer::ContinuationSource::L1 ||
        prefix.reused_state_tokens != cold_shorter.summary.state_tokens ||
        prefix.reused_state_tokens >= prefix.summary.state_tokens) {
        return fail("a longer state did not continue the retained shorter one");
    }
    if (!well_formed(prefix, longer) || max_difference(prefix, cold_longer) > kBehaviouralBar) {
        return fail("a decision continuing a retained prefix moved beyond the bar");
    }
    return 0;
}

// Decisions are prefill-only lanes: their chunks interleave with a chat prompt's chunks and decode
// rounds, but never join a call of another lane, so neither side's arithmetic may change. The
// long decision is submitted first so that its later chunks run after the chat lane has started.
int verify_chat_and_decisions_are_independent(ninfer::Engine& engine) {
    const std::vector<ninfer::DecisionInput> inputs{long_decision(), mixed_decision(40, 3),
                                                    mixed_decision(24, 6)};
    const std::vector<ninfer::TokenId> chat_alone =
        engine.generate(engine.prepare(chat_prompt()), greedy(kChatTokens, std::nullopt))
            .generated_token_ids;
    if (chat_alone.size() != kChatTokens) { return fail("the chat reference stopped early"); }
    std::vector<ninfer::DecisionResult> alone;
    for (const ninfer::DecisionInput& input : inputs) { alone.push_back(decide(engine, input, false)); }

    std::vector<ninfer::DecisionHandle> handles;
    handles.push_back(engine.submit_decision(engine.prepare_decision(inputs[0]), decider(false)));
    ninfer::GenerationHandle chat =
        engine.submit(engine.prepare(chat_prompt()), greedy(kChatTokens, std::nullopt));
    for (std::size_t index = 1; index < inputs.size(); ++index) {
        handles.push_back(
            engine.submit_decision(engine.prepare_decision(inputs[index]), decider(false)));
    }
    if (chat.wait().generated_token_ids != chat_alone) {
        return fail("chat greedy output changed while decisions ran beside it");
    }
    for (std::size_t index = 0; index < handles.size(); ++index) {
        if (handles[index].wait().probabilities != alone[index].probabilities) {
            return fail("decision " + std::to_string(index) +
                        " changed while it ran beside chat and other decisions");
        }
    }
    return 0;
}

int verify_cancellation_releases_the_lane(ninfer::Engine& engine) {
    ninfer::DecisionHandle cancelled =
        engine.submit_decision(engine.prepare_decision(long_decision()), decider(false));
    try {
        static_cast<void>(cancelled.wait(ninfer::CancellationView([] { return true; })));
        return fail("a cancelled decision completed");
    } catch (const ninfer::RequestError& error) {
        if (error.kind() != ninfer::RequestErrorKind::Cancelled) {
            return fail(std::string("a cancelled decision failed with: ") + error.what());
        }
    }
    {
        // Destroying an unconsumed handle cancels its decision.
        const ninfer::DecisionHandle abandoned =
            engine.submit_decision(engine.prepare_decision(long_decision()), decider(false));
    }

    // Every lane is free again: a full set of decisions is admitted before the deadline, and each
    // matches the same decision run alone.
    const ninfer::DecisionInput input     = mixed_decision(40, 7);
    const ninfer::DecisionResult expected = decide(engine, input, false);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    std::vector<ninfer::DecisionHandle> handles;
    for (std::uint32_t lane = 0; lane < kConcurrency; ++lane) {
        handles.push_back(
            engine.submit_decision(engine.prepare_decision(input), decider(false), deadline));
    }
    for (ninfer::DecisionHandle& handle : handles) {
        if (handle.wait().probabilities != expected.probabilities) {
            return fail("a decision after cancellations does not match the same decision alone");
        }
    }
    return 0;
}

int verify_adapter_kinds_do_not_cross(ninfer::Engine& engine) {
    try {
        static_cast<void>(engine.submit(engine.prepare(chat_prompt()), greedy(4, kDeciderName)));
        return fail("chat accepted a decision adapter");
    } catch (const ninfer::RequestError& error) {
        if (error.kind() != ninfer::RequestErrorKind::UnknownAdapter) {
            return fail(std::string("chat rejected a decision adapter with: ") + error.what());
        }
    }
    try {
        static_cast<void>(engine.submit_decision(engine.prepare_decision(mixed_decision()),
                                                 {.adapter = "missing", .allow_prefix_reuse = true}));
        return fail("a decision accepted an adapter outside the pool");
    } catch (const ninfer::RequestError& error) {
        if (error.kind() != ninfer::RequestErrorKind::UnknownAdapter) {
            return fail(std::string("an unknown decision adapter failed with: ") + error.what());
        }
    }
    return 0;
}

int verify_capacity_is_enforced(const ninfer::Engine& engine) {
    try {
        static_cast<void>(engine.prepare_decision(
            {.state = ticket_text(400, 8), .questions = triage_questions()}));
        return fail("a state longer than a lane was prepared");
    } catch (const ninfer::DecisionInputError&) {
        return 0;
    }
}

int exercise(const char* artifact, const char* decision_path) {
    PoolDirectory pool("exercise");
    pool.add(kDeciderName, decision_path);

    ninfer::EngineOptions options;
    options.artifact_path            = artifact;
    options.max_context              = kMaxContext;
    options.kv_capacity              = ninfer::KvCapacityPolicy::explicit_capacity(4 * kMaxContext);
    options.kv_cache                 = ninfer::KvCacheStorage::RK4V4E8;
    options.prefill_chunk            = kPrefillChunk;
    options.max_concurrency          = kConcurrency;
    options.lora.directory           = pool.path();
    options.lora.slots               = 1;
    options.continuation_cache.tiers = ninfer::ContinuationCacheTiers::L1L2;

    ninfer::Engine engine(options);
    if (const int result = verify_registration(engine); result != 0) { return result; }
    if (const int result = verify_cold_repeat_is_bit_identical(engine); result != 0) {
        return result;
    }
    if (const int result = verify_state_restores_from_l2(engine); result != 0) { return result; }
    if (const int result = verify_restored_state_is_equivalent(engine); result != 0) {
        return result;
    }
    if (const int result = verify_chat_and_decisions_are_independent(engine); result != 0) {
        return result;
    }
    if (const int result = verify_cancellation_releases_the_lane(engine); result != 0) {
        return result;
    }
    if (const int result = verify_adapter_kinds_do_not_cross(engine); result != 0) {
        return result;
    }
    return verify_capacity_is_enforced(engine);
}

const char* env_or_null(const char* name) {
    const char* value = std::getenv(name);
    return (value != nullptr && *value != '\0') ? value : nullptr;
}

} // namespace

int main() {
    const char* artifact = env_or_null("NINFER_QWEN3_8_27B_WEIGHTS");
    const char* decision = env_or_null("NINFER_QWEN3_8_27B_LORA_DECISION");
    if (artifact == nullptr || decision == nullptr) {
        std::cout << "skip: NINFER_QWEN3_8_27B_WEIGHTS and NINFER_QWEN3_8_27B_LORA_DECISION (a "
                     "trained decision .lora.ninfer) are required\n";
        return 77;
    }
    if (const int result = exercise(artifact, decision); result != 0) { return result; }
    std::cout << "ok\n";
    return 0;
}
