#pragma once

#include "ninfer/types.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer {

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();

    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] const PromptSummary& summary() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    class Impl;
    explicit PreparedPrompt(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class GenerationHandle {
public:
    GenerationHandle() noexcept;
    ~GenerationHandle();

    GenerationHandle(GenerationHandle&&) noexcept;
    GenerationHandle& operator=(GenerationHandle&&) noexcept;

    GenerationHandle(const GenerationHandle&)            = delete;
    GenerationHandle& operator=(const GenerationHandle&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept;

    GenerationResult wait(OutputSink* sink = nullptr, const CancellationView& cancellation = {});

private:
    class Impl;
    explicit GenerationHandle(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

// An escaped, tokenized and laid-out decision, ready to submit.
class PreparedDecision {
public:
    PreparedDecision() noexcept;
    ~PreparedDecision();

    PreparedDecision(PreparedDecision&&) noexcept;
    PreparedDecision& operator=(PreparedDecision&&) noexcept;

    PreparedDecision(const PreparedDecision&)            = delete;
    PreparedDecision& operator=(const PreparedDecision&) = delete;

    [[nodiscard]] const DecisionSummary& summary() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

    // The laid-out tokens (the state row, then every branch in request order) and each
    // question's branch, retained for parity tools. Empty when the decision is empty.
    [[nodiscard]] std::span<const TokenId> token_ids() const noexcept;
    [[nodiscard]] std::span<const DecisionBranch> branches() const noexcept;

private:
    class Impl;
    explicit PreparedDecision(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class DecisionHandle {
public:
    DecisionHandle() noexcept;
    ~DecisionHandle();

    DecisionHandle(DecisionHandle&&) noexcept;
    DecisionHandle& operator=(DecisionHandle&&) noexcept;

    DecisionHandle(const DecisionHandle&)            = delete;
    DecisionHandle& operator=(const DecisionHandle&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;

    DecisionResult wait(const CancellationView& cancellation = {});

private:
    class Impl;
    explicit DecisionHandle(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class Engine {
public:
    explicit Engine(EngineOptions options);
    ~Engine();

    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    [[nodiscard]] PreparedPrompt prepare(PromptInput input) const;

    // Raw token input is retained for parity tools and repeatable performance measurement.
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;

    [[nodiscard]] std::uint32_t count_tokens(PromptInput input) const;
    [[nodiscard]] PromptCapabilities prompt_capabilities() const;
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const;

    // Establishes queue membership synchronously. Destroying an unconsumed handle cancels its
    // request; wait() owns result consumption and may run independently from GPU execution.
    [[nodiscard]] GenerationHandle
    submit(PreparedPrompt prompt, RequestOptions options,
           std::chrono::steady_clock::time_point pending_deadline = {},
           HostInputLease host_input                              = {});

    GenerationResult generate(PreparedPrompt prompt, RequestOptions options,
                              OutputSink* sink                     = nullptr,
                              const CancellationView& cancellation = {});

    // System One decisions. prepare_decision escapes, tokenizes and lays out the rendered input
    // and throws DecisionInputError when it does not fit its budgets; submit_decision joins the
    // same bounded FIFO as generation. Both throw RequestError(Unavailable) on a target without
    // decision support; submit_decision throws RequestError(UnknownAdapter) unless the options
    // name a decision adapter.
    [[nodiscard]] PreparedDecision prepare_decision(DecisionInput input) const;
    [[nodiscard]] DecisionHandle
    submit_decision(PreparedDecision decision, DecisionOptions options,
                    std::chrono::steady_clock::time_point pending_deadline = {});
    DecisionResult decide(PreparedDecision decision, DecisionOptions options,
                          const CancellationView& cancellation = {});

    // Tokens of plain text encoded with added-token parsing and no template.
    [[nodiscard]] std::uint32_t count_text_tokens(std::string_view text) const;

    [[nodiscard]] const EngineOptions& options() const;
    [[nodiscard]] LoadSummary load_summary() const;
    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    void reset_memory_peaks() noexcept;

    // Session persistence for one Engine lane ("slot"). save_slot writes the lane's retained
    // session to `path`; restore_slot rebuilds a lane from a saved file, evicting whatever the
    // lane retained; erase_slot evicts the lane's retained session and reports its depth. A
    // busy lane raises RequestError(Overloaded); incompatible or missing files raise
    // std::invalid_argument; a non-empty expected_digest that does not match the lane's
    // resident session raises SlotSessionMismatch, checked atomically with the operation. GPU
    // work runs at a request boundary; file I/O runs outside it.
    [[nodiscard]] SlotSaveResult save_slot(std::uint32_t lane, const std::string& path,
                                           const std::string& expected_digest = {});
    [[nodiscard]] SlotRestoreResult restore_slot(std::uint32_t lane, const std::string& path);
    std::uint32_t erase_slot(std::uint32_t lane, const std::string& expected_digest = {});

    // Truthful per-lane occupancy, read at a request boundary.
    [[nodiscard]] std::vector<SlotState> slot_states() const;
    // The LoRA bank's device slots, one entry per `--lora-slots`, read at the same boundary.
    // Empty when the Engine was loaded without a LoRA pool.
    [[nodiscard]] std::vector<AdapterSlotState> adapter_slot_states() const;

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace ninfer
