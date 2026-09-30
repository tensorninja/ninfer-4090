#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer {

using TokenId = std::int32_t;

inline constexpr std::uint32_t kMaximumConcurrency = 8;

enum class KvCacheStorage : std::uint8_t {
    BFloat16,
    Int8Group64,
    RotatedInt8KeyInt4ValueGroup64,
    RotatedInt4KeyInt4ValueGroup64,
    RK4V4E8,
    RK2V4E8,
};

enum class KvCapacityMode : std::uint8_t {
    Explicit,
    Automatic,
};

inline constexpr std::size_t kDefaultKvCapacityHeadroomBytes = 1024ULL * 1024ULL * 1024ULL;

struct KvCapacityPolicy {
    KvCapacityMode mode                  = KvCapacityMode::Explicit;
    std::uint32_t explicit_tokens        = 2048;
    std::size_t automatic_headroom_bytes = 0;

    [[nodiscard]] static constexpr KvCapacityPolicy
    explicit_capacity(std::uint32_t tokens) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Explicit, tokens, 0};
    }

    [[nodiscard]] static constexpr KvCapacityPolicy
    automatic(std::size_t headroom_bytes = kDefaultKvCapacityHeadroomBytes) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Automatic, 0, headroom_bytes};
    }
};

enum class ProposalHead : std::uint8_t {
    Full,
    Optimized,
};

enum class SpeculativeBackend : std::uint8_t {
    None,
    Mtp,
    DFlash,
};

struct SpeculativeOptions {
    SpeculativeBackend backend = SpeculativeBackend::None;
    std::uint32_t draft_tokens = 0;
    ProposalHead proposal_head = ProposalHead::Full;
};

// Terminates a generation that has locked into a repeating token cycle. A failed tool call often
// leaves a model narrating its next step, reading that narration back, and emitting it again; the
// cycle then runs to the output limit. The guard observes committed tokens only and never changes
// the sampled distribution, so a generation that does not loop is bit-identical to one produced
// without it.
struct RepetitionGuardOptions {
    bool enabled = true;
    // How far back a repeat may reach, in committed tokens. Also bounds the detectable period.
    std::uint32_t window = 512;
    // Length of the token n-gram whose recurrence proposes a period.
    std::uint32_t ngram = 24;
    // Full periods that must be confirmed before the cycle counts as locked.
    std::uint32_t cycles = 3;
    // Confirmed tokens required regardless of period, so a short period needs a long run. This is
    // what keeps a divider rule or a repeated indent from tripping a period-1 match.
    std::uint32_t min_tokens = 64;
};

struct LoadProgress {
    std::function<void(std::string_view phase, std::uint64_t done, std::uint64_t total)> callback;
};

enum class PrefixCheckpointPolicy : std::uint8_t {
    StableTurn,
    RollingTool,
};

enum class ContinuationCacheTiers : std::uint8_t {
    Off,
    L1,
    L1L2,
    L1L2L3,
};

enum class ContinuationCachePolicy : std::uint8_t {
    Adaptive,
};

struct ContinuationCacheOptions {
    ContinuationCacheTiers tiers   = ContinuationCacheTiers::L1L2;
    ContinuationCachePolicy policy = ContinuationCachePolicy::Adaptive;
    std::size_t l1_capacity_mib    = 768;
    std::size_t l2_capacity_mib    = 16384;
    std::size_t l3_capacity_mib    = 49152;
    std::filesystem::path directory;
    std::string cache_namespace             = "local";
    std::uint32_t l1_idle_ttl_seconds       = 600;
    std::uint32_t l2_idle_ttl_seconds       = 7200;
    std::uint32_t l3_idle_ttl_seconds       = 86400;
    std::uint32_t persist_interval_seconds  = 60;
    std::uint32_t persist_min_tokens        = 8192;
    std::size_t filesystem_reserve_mib      = 0;
    std::uint32_t prefix_checkpoint_history = 4;
};

// LoRA adapter pool. Every conforming `.ninfer` adapter under `directory` is discovered at
// startup and selectable by name for the process lifetime; the pool size is unbounded. Only
// `slots` of them are device-resident at once, and the engine swaps an adapter into a slot at
// admission when a request selects one that is not resident.
//
// Discovery validates every descriptor and computes the adapter's SHA-256 content fingerprint,
// which namespaces its continuation state. The configured artifact sidecar cache avoids rescans
// for unchanged files.
struct LoraOptions {
    // Empty disables LoRA entirely: no bank is committed and the captured graph is topologically
    // identical to a build without adapter support.
    std::filesystem::path directory;
    // Resident device slots. Bounded only by device memory, which the KV capacity resolver sees
    // after the bank is committed. A pool smaller than this commits only what it needs.
    std::uint32_t slots = 2;
    // Bank rank ceiling. Zero takes the maximum rank present in the pool. An adapter above the
    // ceiling is rejected at discovery rather than silently taxing every other adapter.
    std::int32_t rank_ceiling = 0;
};

// What an adapter serves. A generative adapter is a chat model of the OpenAI/Anthropic surface;
// a decision adapter carries a System One pointer head and is served only through decisions.
enum class LoraAdapterKind : std::uint8_t {
    Generative,
    Decision,
};

// Host metadata of one discovered adapter.
struct LoraAdapterInfo {
    std::string name;
    LoraAdapterKind kind = LoraAdapterKind::Generative;
    // The adapter's own rank before bank padding.
    std::int32_t rank = 0;
    // Decision adapters only: the calibrated temperature the pointer logits are divided by, the
    // pointer dimension, and the model-card description and release date (YYYY-MM-DD).
    float temperature         = 1.0F;
    std::uint32_t pointer_dim = 0;
    std::string description;
    std::string release_date;
};

// What a target package committed when it attached the adapter pool. Returned by
// `Package::attach_lora` so the loader publishes the pool and the resident bank's cost once, at
// the point that knows it, instead of discarding it and leaving the bytes unattributable later.
struct LoraAttachment {
    // Discovered adapters in pool order; empty when no adapter was discovered.
    std::vector<LoraAdapterInfo> adapters;
    // Bank rank. Every pool adapter executes at this rank; a lower-rank adapter is zero-padded
    // into it, which is exact because the padded factors contribute nothing.
    std::int32_t rank = 0;
    // Device-resident slots the bank committed.
    std::uint32_t slots = 0;
    // Device bytes committed for the bank, and artifact bytes the pool occupies on disk.
    std::uint64_t device_bytes = 0;
    std::uint64_t file_bytes   = 0;
};

// Outcome of one background auto-save of an involuntarily evicted session. Delivered on the
// Engine's writer thread; the listener must be thread-safe.
struct SlotAutoSaveEvent {
    std::string path;
    std::uint32_t tokens = 0;
    std::size_t bytes    = 0;
    double seconds       = 0.0;
    std::string error; // empty on success
};

struct EngineOptions {
    std::filesystem::path artifact_path;
    int device                         = 0;
    std::uint32_t max_context          = 2048; // Exact logical ceiling of each request.
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 30000;
    std::uint32_t prefill_chunk        = 1024;
    // While a prompt is still being consumed, the execution thread alternates prefill chunks with
    // decode rounds. A chunk costs several rounds, so a one-for-one alternation gives the thread
    // almost entirely to the prefilling lane. This is the share of thread time granted to decode
    // relative to prefill while a prefill is in flight: 0 keeps one round per chunk, 1.0 splits
    // the thread evenly. Raising it speeds up already-generating requests at the cost of the
    // latency of the request being prefilled.
    double prefill_decode_balance      = 0.0;
    // Retained host-side turn checkpoints per lane (0 disables the ring). Each entry snapshots
    // the linear-attention state at a past turn boundary so a prompt that diverges mid-history
    // re-prefills from the nearest checkpoint instead of from zero. Host memory cost per entry
    // is the model's full GDN state image (~147 MiB on Qwen3.8-27B).
    std::uint32_t turn_checkpoint_ring = 0;
    // Before an involuntary eviction destroys a retained session, snapshot it back to the slot
    // file it was last saved to or restored from (sessions that never touched a slot file are
    // not covered). The device snapshot runs on the eviction path; the file write runs on a
    // background writer thread. Explicit erase never auto-saves.
    bool auto_save_evicted = false;
    // Optional observer for auto-save outcomes; called on the writer thread.
    std::function<void(const SlotAutoSaveEvent&)> auto_save_listener;
    KvCacheStorage kv_cache            = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    RepetitionGuardOptions repetition_guard;
    PrefixCheckpointPolicy prefix_checkpoint_policy = PrefixCheckpointPolicy::RollingTool;
    ContinuationCacheOptions continuation_cache;
    std::uint32_t vision_max_tokens = 8192;
    bool enable_vision  = false;
    bool use_cuda_graph = true;
    LoraOptions lora;
    LoadProgress load_progress;
};

enum class SamplingMode : std::uint8_t {
    Thinking,
    NonThinking,
};

// Immutable model-owned values used when a request does not override a sampling field. Seed is
// deliberately excluded: it is an execution choice rather than a model recommendation.
struct SamplingPreset {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
};

struct ModelSamplingDefaults {
    SamplingPreset thinking;
    SamplingPreset non_thinking;

    [[nodiscard]] constexpr const SamplingPreset& for_mode(SamplingMode mode) const noexcept {
        return mode == SamplingMode::Thinking ? thinking : non_thinking;
    }
};

// Public request-side overrides. std::nullopt means "use the registered model/mode default";
// explicit zero remains a real override (including temperature=0 for exact argmax).
struct SamplingOverrides {
    std::optional<float> temperature;
    std::optional<std::int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// Complete parameters after Engine resolution. Target runtimes consume only this type.
struct ResolvedSamplingParameters {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
    std::uint64_t seed      = 0;
};

enum class OutputChannel : std::uint8_t {
    Content,
    Reasoning,
};

struct StopString {
    std::string text;
    OutputChannel channel  = OutputChannel::Content;
    bool include_in_output = false;
};

struct StopPolicy {
    std::vector<TokenId> token_ids;
    std::vector<StopString> strings;
    bool include_model_defaults = true;
    bool publish_stop_token     = false;
};

struct ExecutionOptions {
    SamplingOverrides sampling;
    // Client-provided routing hint only; exact prepared-prefix identity must authorize reuse.
    std::optional<std::string> routing_hint;
    // Pool adapter name; nullopt selects the base weights. A name outside the pool is a request
    // error, never a silent fallback to base. Residency is invisible here: a name in the pool is
    // always selectable, and the engine stages it into a slot if it is not resident.
    std::optional<std::string> adapter;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
};

struct OutputOptions {
    bool raw                     = false;
    bool preserve_special_tokens = false;
};

struct RequestOptions {
    ExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

// Owns a bounded host-input reservation whose lifetime may cross from request preparation into
// Engine execution. The concrete reservation is product-owned; Engine only releases it once the
// target reports that the retained host payload is no longer needed.
class HostInputLease {
public:
    HostInputLease() noexcept = default;

    explicit HostInputLease(std::shared_ptr<void> owner) noexcept : owner_(std::move(owner)) {}

    HostInputLease(HostInputLease&&) noexcept            = default;
    HostInputLease& operator=(HostInputLease&&) noexcept = default;

    HostInputLease(const HostInputLease&)            = delete;
    HostInputLease& operator=(const HostInputLease&) = delete;

    void reset() noexcept { owner_.reset(); }

    [[nodiscard]] explicit operator bool() const noexcept { return owner_ != nullptr; }

private:
    std::shared_ptr<void> owner_;
};

enum class MediaKind : std::uint8_t {
    Image,
    Video,
};

struct OwnedMedia {
    MediaKind kind = MediaKind::Image;
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

enum class MessagePartKind : std::uint8_t {
    Text,
    Media,
};

struct MessagePart {
    MessagePartKind kind = MessagePartKind::Text;
    std::string text;
    OwnedMedia media;
    // An explicit prompt-cache breakpoint: the rendered prefix through the end of this part is
    // published as a content-addressed continuation any later request with the same prefix can
    // restore. Placement follows OpenAI `prompt_cache_breakpoint` and Anthropic `cache_control`.
    bool cache_breakpoint = false;
};

struct ChatMessage {
    std::string role;
    std::vector<MessagePart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
};

enum class ReasoningEffort : std::uint8_t {
    Low,
    Medium,
    XHigh,
};

struct ReasoningEffortCapabilities {
    bool low    = false;
    bool medium = false;
    bool xhigh  = false;
    std::optional<ReasoningEffort> default_effort;

    [[nodiscard]] constexpr bool supports(ReasoningEffort effort) const noexcept {
        switch (effort) {
        case ReasoningEffort::Low:
            return low;
        case ReasoningEffort::Medium:
            return medium;
        case ReasoningEffort::XHigh:
            return xhigh;
        }
        return false;
    }
};

struct PromptCapabilities {
    bool enable_thinking = false;
    ReasoningEffortCapabilities reasoning_effort;
};

// Which prompt-cache boundaries a request publishes and looks up. Implicit boundaries are the
// system/tools prefix and the turn openers the server derives from the rendered chat; explicit
// ones are the client's breakpoints. Explicit mode uses only the latter, as OpenAI's
// `prompt_cache_options.mode` does. Routed sessions (`prompt_cache_key`) are a separate axis.
enum class PromptCacheMode : std::uint8_t {
    Implicit,
    Explicit,
};

struct PromptOptions {
    bool add_generation_prompt = true;
    bool enable_thinking       = true;
    std::optional<ReasoningEffort> reasoning_effort;
    bool preserve_thinking = false;
    bool add_vision_id     = false;
    PromptCacheMode prompt_cache_mode = PromptCacheMode::Implicit;
    std::vector<std::string> tool_jsons;
};

struct PromptInput {
    std::vector<ChatMessage> messages;
    PromptOptions options;
};

enum class RequestErrorKind : std::uint8_t {
    ContextLengthExceeded,
    MediaBudgetExceeded,
    Overloaded,
    QueueTimeout,
    Unavailable,
    // The request named a LoRA adapter that was not registered at startup.
    UnknownAdapter,
    // A decision was cancelled before it produced its result. A cancelled generation instead
    // finishes with FinishReason::Cancelled and keeps what it had produced.
    Cancelled,
};

class RequestError final : public std::invalid_argument {
public:
    RequestError(RequestErrorKind kind, std::string message)
        : std::invalid_argument(std::move(message)), kind_(kind) {}

    [[nodiscard]] RequestErrorKind kind() const noexcept { return kind_; }

private:
    RequestErrorKind kind_;
};

// A fault that invalidates the generation round in flight and nothing beyond it. A licensed
// prefix disagreeing with the state that produced it says nothing about the device, the
// scheduler, or the next round, so the executor fails the requests that round was serving and
// keeps accepting work. Every other exception escaping the worker loop retires the engine.
class RoundFault final : public std::logic_error {
public:
    explicit RoundFault(std::string message) : std::logic_error(std::move(message)) {}
};

struct PromptSummary {
    std::uint32_t prompt_tokens = 0;
    bool has_media              = false;
};

enum class FinishReason : std::uint8_t {
    None,
    OutputLimit,
    ContextCapacity,
    StopToken,
    StopString,
    Cancelled,
    RepetitionCycle,
};

struct OutputDelta {
    OutputChannel channel = OutputChannel::Content;
    std::string text;
};

// Prefill produces no output, so a long prompt leaves a streaming transport
// with nothing to send for the whole computation. This carries the one fact a
// client can act on while it waits.
struct PromptProgress {
    std::uint32_t processed_prompt_tokens = 0;
    std::uint32_t prompt_tokens           = 0;
    std::uint32_t reused_prompt_tokens    = 0;
};

class OutputSink {
public:
    virtual ~OutputSink()                   = default;
    virtual void publish(OutputDelta delta) = 0;
    // Only transports that keep a socket open while the prompt is computed
    // have anywhere to put this, so it defaults to discarding it.
    virtual void publish_prompt_progress(PromptProgress /*progress*/) {}
};

class CancellationView {
public:
    CancellationView() = default;
    explicit CancellationView(std::function<bool()> requested);

    [[nodiscard]] bool requested() const;

private:
    std::function<bool()> requested_;
};

struct GenerationTimings {
    double prepare_seconds     = 0.0;
    double first_token_seconds = 0.0;
    double vision_seconds      = 0.0;
    // Wall time the request spent in the bounded FIFO between submission and the admission that
    // gave it a lane. Under concurrent load this, not prefill, dominates TTFT, so it is a
    // first-class timing rather than a cache diagnostic.
    double queue_seconds       = 0.0;
    // Wall time spent importing a continuation image into the admitted lane. Charged to the
    // request that consumed it; an L1 hit needs no import and reports zero.
    double restore_seconds     = 0.0;
    // Completion-path cost of handing the retained lane to the publication worker: fencing the
    // lane and queueing the export job. The export itself runs on the worker and is reported
    // through the continuation export counters, not here.
    double publish_seconds     = 0.0;
    double prefill_seconds     = 0.0;
    double decode_seconds      = 0.0;
    double total_seconds       = 0.0;
};

struct SpeculativeStats {
    SpeculativeBackend backend    = SpeculativeBackend::None;
    bool enabled                  = false;
    std::uint32_t draft_window    = 0;
    std::uint64_t rounds          = 0;
    std::uint64_t drafted_tokens  = 0;
    std::uint64_t accepted_tokens = 0;
    std::uint64_t fallback_steps  = 0;
    std::vector<std::uint64_t> accepted_per_position;
};

enum class PrefixReusePath : std::uint8_t {
    FullReset,
    AppendAtFrontier,
    RestoreTurnCheckpoint,
    // Restored from the host-resident anchor at the last user query's opener. Shallower than a
    // turn checkpoint, but it survives a client rewriting the tail of that message.
    RestoreUserTurnAnchor,
};

enum class ContinuationSource : std::uint8_t { None, L1, L2, L3 };
enum class ContinuationAliasKind : std::uint8_t { None, Session, StablePrefix };
enum class ContinuationMissReason : std::uint8_t {
    None,
    Disabled,
    // No alias could be derived for the request at all: neither a client routing hint nor a
    // reusable stable prefix. Genuinely uncacheable, distinct from NotAttempted.
    NoAlias,
    // An alias existed but no candidate was ever evaluated - the lookup found nothing under it,
    // or restoration never ran. Separating this from NoAlias is what makes a cold-start miss
    // distinguishable from a cache that simply had no entry yet.
    NotAttempted,
    EntryUnavailableOrCorrupt,
    NotDeeper,
    PreflightRejected,
    RollbackConflict,
    NoLane,
    RestoreFailed,
};

// Why an import of an otherwise viable continuation image into a lane did not complete. The
// engine cannot choose a better candidate without knowing this, and a swallowed cause is what
// made a KV-exhaustion restore failure look identical to a corrupt image.
enum class ContinuationRestoreFailure : std::uint8_t {
    None,
    // The shared paged-KV pool could not satisfy the restored lane's page reservation.
    KvReservationExhausted,
    // The independent re-verification inside the import disagreed with the preflight depth.
    VerifyDepthMismatch,
    // The image's segment inventory did not match what this backend configuration requires.
    SegmentInventoryMismatch,
    // Frontier, ledger, backend, or geometry metadata was inconsistent with the image payload.
    MetadataMismatch,
    // A segment failed to decode, or a host-to-device transfer raised.
    DecodeFailed,
    // The lane was not in a state that can accept an import.
    LaneUnavailable,
};

[[nodiscard]] constexpr std::string_view continuation_source_name(ContinuationSource value) {
    switch (value) {
    case ContinuationSource::None: return "none";
    case ContinuationSource::L1: return "l1";
    case ContinuationSource::L2: return "l2";
    case ContinuationSource::L3: return "l3";
    }
    return "none";
}

[[nodiscard]] constexpr std::string_view continuation_alias_kind_name(
    ContinuationAliasKind value) {
    switch (value) {
    case ContinuationAliasKind::None: return "none";
    case ContinuationAliasKind::Session: return "routed_session";
    case ContinuationAliasKind::StablePrefix: return "stable_prefix";
    }
    return "none";
}

[[nodiscard]] constexpr std::string_view continuation_miss_reason_name(
    ContinuationMissReason value) {
    switch (value) {
    case ContinuationMissReason::None: return "none";
    case ContinuationMissReason::Disabled: return "disabled";
    case ContinuationMissReason::NoAlias: return "no_alias";
    case ContinuationMissReason::NotAttempted: return "not_attempted";
    case ContinuationMissReason::EntryUnavailableOrCorrupt:
        return "entry_unavailable_or_corrupt";
    case ContinuationMissReason::NotDeeper: return "not_deeper";
    case ContinuationMissReason::PreflightRejected: return "preflight_rejected";
    case ContinuationMissReason::RollbackConflict: return "rollback_conflict";
    case ContinuationMissReason::NoLane: return "no_lane";
    case ContinuationMissReason::RestoreFailed: return "restore_failed";
    }
    return "none";
}

[[nodiscard]] constexpr std::string_view continuation_restore_failure_name(
    ContinuationRestoreFailure value) {
    switch (value) {
    case ContinuationRestoreFailure::None: return "none";
    case ContinuationRestoreFailure::KvReservationExhausted: return "kv_reservation_exhausted";
    case ContinuationRestoreFailure::VerifyDepthMismatch: return "verify_depth_mismatch";
    case ContinuationRestoreFailure::SegmentInventoryMismatch:
        return "segment_inventory_mismatch";
    case ContinuationRestoreFailure::MetadataMismatch: return "metadata_mismatch";
    case ContinuationRestoreFailure::DecodeFailed: return "decode_failed";
    case ContinuationRestoreFailure::LaneUnavailable: return "lane_unavailable";
    }
    return "none";
}

struct ContinuationDiagnostics {
    ContinuationSource source                 = ContinuationSource::None;
    ContinuationAliasKind alias_kind         = ContinuationAliasKind::None;
    ContinuationMissReason final_miss_reason = ContinuationMissReason::NotAttempted;
    // Set only when final_miss_reason is RestoreFailed; names the import step that refused.
    ContinuationRestoreFailure restore_failure = ContinuationRestoreFailure::None;
    std::uint64_t lookup_microseconds         = 0;
    std::uint64_t preflight_microseconds      = 0;
    std::uint64_t restore_microseconds        = 0;
    std::uint64_t restored_tokens             = 0;
    std::uint64_t restored_bytes              = 0;
    // Tokens this request captured for content-addressed publication beyond what it restored:
    // the deepest boundary it published minus its restored depth. OpenAI's `cache_write_tokens`.
    std::uint64_t cache_write_tokens          = 0;
    // Deepest prefix any preflighted candidate agreed with this prompt on, whether or not the
    // candidate was usable. On a miss this separates a tail rewrite, which a deeper checkpoint
    // ladder can recover, from an early rewrite, which no ladder can.
    std::uint64_t deepest_candidate_agreement = 0;
    bool candidate_agreement_observed         = false;
    // Why a restore that was deferred for shared-KV capacity never came back. `gate_checks`
    // counts how often the deferred retry was consulted at all, `gate_passes` how often it found
    // a lane that could host the restore, and the two lane numbers record which lane the restore
    // chose against the one admission actually used. A retry that never runs, one that runs but
    // never sees capacity, and one that sees capacity yet still loses the lane are three
    // different defects, and only these fields separate them.
    std::uint32_t restore_gate_checks         = 0;
    std::uint32_t restore_gate_passes         = 0;
    std::int32_t restore_target_lane          = -1;
    std::int32_t admitted_lane                = -1;
    bool destructive_rollback                 = false;
    bool completion_publication_queued        = false;
};

struct GenerationResult {
    PromptSummary prompt;
    std::vector<TokenId> generated_token_ids;
    std::string content;
    std::string reasoning;
    std::uint32_t reasoning_tokens     = 0;
    FinishReason finish_reason         = FinishReason::None;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::FullReset;
    GenerationTimings timings;
    SpeculativeStats speculative;
    ContinuationDiagnostics continuation;
    // Lane that served the request and, when it retained the finished session, that session's
    // identifying digest (see SlotState) - the handle a client needs for /slots operations.
    std::int32_t slot = -1;
    std::string session_digest;
};

// System One decisions. A decision is a prefill-only request of a decision adapter: the state,
// then one branch per question whose option and decide readouts feed the adapter's pointer head.
// It never samples, decodes or drafts.
//
// One question after product-side rendering: its instruction text and its option texts in option
// order (1 to kMaximumDecisionOptions). The engine escapes, tokenizes and lays the texts out; it
// never interprets question types.
inline constexpr std::uint32_t kMaximumDecisionOptions = 255;

struct DecisionQuestion {
    std::string instructions;
    std::vector<std::string> options;
};

// The rendered state and at least one question, in request order.
struct DecisionInput {
    std::string state;
    std::vector<DecisionQuestion> questions;
};

// Token accounting of one prepared decision.
struct DecisionSummary {
    // The state including its delimiter (the branch position origin).
    std::uint32_t state_tokens = 0;
    // Every question branch, and the longest one.
    std::uint32_t branch_tokens  = 0;
    std::uint32_t longest_branch = 0;
    std::uint32_t questions      = 0;
    std::uint32_t options        = 0;
    // The state text encoded to more tokens than a state holds and was cut to its head.
    bool state_truncated = false;

    [[nodiscard]] std::uint32_t input_tokens() const noexcept {
        return state_tokens + branch_tokens;
    }
};

// One question's branch in a prepared decision's token layout (kev model.encode): the state row
// [<|fim_prefix|>, state...] is followed, per question in request order, by the branch
// [<|fim_middle|>, instructions..., (<|box_start|>, option..., <|box_end|>)..., <|fim_suffix|>].
struct DecisionBranch {
    // Offset of the branch's first token in the layout, and its length.
    std::uint32_t begin  = 0;
    std::uint32_t length = 0;
    // Branch-relative offsets of each option's closing delimiter, in option order. The decide
    // readout is the branch's last token.
    std::vector<std::uint32_t> option_readouts;
};

// A decision that does not fit its token budgets. `what()` is the client-visible detail.
class DecisionInputError final : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

struct DecisionOptions {
    // The decision adapter to answer with. Required.
    std::string adapter;
    // Restore a cached state and retain/publish this one. Off computes the state cold.
    bool allow_prefix_reuse = true;
};

struct DecisionTimings {
    double prepare_seconds = 0.0; // Escape, tokenize and lay out (host, before submission).
    // Submission to admission. It includes restore_seconds: a cached image is imported into the
    // lane just before admission.
    double queue_seconds   = 0.0;
    double restore_seconds = 0.0; // Import of an L2/L3 state image; zero for L1 or a cold state.
    double state_seconds   = 0.0; // State prefill.
    double branch_seconds  = 0.0; // Branch passes, long-branch chunks, readout and head.
    double execution_seconds = 0.0; // Admission to result.
    double total_seconds     = 0.0; // Submission to result.

    // The engine's time on this decision without its queue wait: the restore plus execution.
    // System One reports it as `latency_ms`.
    [[nodiscard]] double engine_seconds() const noexcept {
        return restore_seconds + execution_seconds;
    }
};

struct DecisionResult {
    // Per question in request order, the option probabilities in option order.
    std::vector<std::vector<float>> probabilities;
    DecisionSummary summary;
    // State tokens restored instead of computed, and where they came from.
    std::uint32_t reused_state_tokens = 0;
    ContinuationSource state_source   = ContinuationSource::None;
    std::uint32_t branch_passes       = 0;
    std::uint32_t long_branch_chunks  = 0;
    DecisionTimings timings;
    std::string adapter;
    std::int32_t slot = -1;
};

struct ArenaMemorySummary {
    std::size_t capacity_bytes  = 0;
    std::size_t used_bytes      = 0;
    std::size_t peak_used_bytes = 0;
};

struct MemorySummary {
    int device                                = 0;
    std::uint32_t max_context                 = 0;
    KvCapacityMode kv_capacity_mode           = KvCapacityMode::Explicit;
    std::uint32_t kv_capacity                 = 0; // Resolved page-aligned Main KV capacity.
    std::uint32_t kv_capacity_page_groups     = 0;
    std::uint32_t kv_capacity_max_page_groups = 0;
    KvCacheStorage kv_cache                   = KvCacheStorage::BFloat16;
    ArenaMemorySummary weights;
    ArenaMemorySummary sequence;
    ArenaMemorySummary workspace;
    ArenaMemorySummary request_transient;
    std::size_t minimum_runtime_reservation_bytes = 0;
    std::size_t kv_capacity_increment_bytes       = 0;
    std::size_t runtime_reservation_bytes         = 0;
    std::size_t available_after_weights_bytes     = 0;
    std::size_t available_after_startup_bytes     = 0;
    std::size_t kv_capacity_headroom_bytes        = 0;
    std::size_t planned_slack_bytes               = 0;
    std::size_t workspace_logical_peak_bytes      = 0;
    std::size_t cuda_graph_allowance_bytes        = 0;
    std::size_t cuda_graph_observed_bytes         = 0;
    std::size_t kv_payload_bytes                  = 0;
    std::size_t text_kv_bytes                     = 0;
    std::size_t mtp_kv_bytes                      = 0;
    std::size_t gdn_state_bytes                   = 0;
    std::size_t dflash_kv_bytes                   = 0;
    std::size_t replay_records_bytes              = 0;
    // Resident LoRA bank. Held in its own arena outside `weights`, so without this field the
    // bank appears only as a reduction in `available_after_startup_bytes` and the reported
    // division of the board does not account for it.
    std::size_t lora_bank_bytes                   = 0;
};

// Monotonic execution counters plus one boundary-consistent scheduler snapshot. Consumers derive
// interval throughput by subtracting two snapshots and dividing by their own monotonic wall time.
struct RuntimeStats {
    // Actual prompt tokens evaluated by prefill; resident prefix hits are excluded.
    std::uint64_t computed_prefill_tokens = 0;
    // Tokens committed by decode rounds; the first token emitted by prefill is excluded.
    std::uint64_t committed_decode_tokens = 0;
    // Cumulative wall time of prefill and decode execution units. Advances with every unit,
    // so counter scrapers see rates move during a request rather than at its completion.
    double prefill_seconds_total = 0.0;
    double decode_seconds_total  = 0.0;
    // Board energy charged to prefill and decode units, in joules. Each unit is bracketed by two
    // instantaneous board-power reads and integrated trapezoidally, so only the interval a unit
    // actually occupied is attributed to it and time spent between units is left unaccounted here.
    //
    // These are estimates. The board refreshes power at roughly 50 Hz, so a single unit is not
    // resolved; only the aggregate over many units converges. Their sum is reconciled against the
    // exact NVML energy counter by the observer that samples it, and the leftover is published as
    // a residual rather than folded silently into a phase. `energy_samples` is zero where the
    // board exposes no power reading at all, which makes every derived figure absent, not zero.
    double prefill_energy_joules = 0.0;
    double decode_energy_joules  = 0.0;
    // Wall time actually covered by those brackets. An observer subtracts it from its own interval
    // to learn how much of the interval no phase claims, which is the only sound way to price the
    // remainder at an idle baseline. It is tracked separately from the worker_* seconds because
    // the first prefill chunk of a request runs inside admission and is timed there.
    double energy_accounted_seconds = 0.0;
    std::uint64_t energy_samples    = 0;
    // Decode batch executions and the sum of their batch sizes.
    std::uint64_t decode_rounds                      = 0;
    std::uint64_t decode_row_rounds                  = 0;
    // Rounds abandoned to a RoundFault, failing the requests they served without retiring the
    // engine. Any sustained rate here is a generation-state bug, not a client problem.
    std::uint64_t decode_rounds_abandoned            = 0;
    std::uint64_t continuation_lookup_hits           = 0;
    std::uint64_t continuation_lookup_misses         = 0;
    std::uint64_t continuation_preflight_rejections  = 0;
    // Continuation decodes served by the preparation thread, versus decoded inline by the
    // executor because no prepared payload matched the image it chose.
    std::uint64_t continuation_preparation_hits   = 0;
    std::uint64_t continuation_preparation_decoded = 0;
    std::uint64_t continuation_preparation_inline = 0;
    // Aggregate useful restores. These equal the corresponding L1 + L2 + L3 tier totals.
    std::uint64_t continuation_restore_successes     = 0;
    std::uint64_t continuation_restore_failures      = 0;
    // Restores refused only for shared-KV capacity and retried afterwards. A deferral is not a
    // failure: counting it as one conflates a recovered restore with a lost one.
    std::uint64_t continuation_restore_deferrals     = 0;
    std::uint64_t continuation_publication_successes = 0;
    std::uint64_t continuation_publication_failures  = 0;
    std::uint64_t continuation_publication_superseded = 0;
    // Publications dropped at enqueue because a newer snapshot for the same session replaced them.
    std::uint64_t continuation_publication_coalesced = 0;
    // Attribution for continuation_publication_failures; these five sum to it.
    std::uint64_t continuation_publication_failed_capacity    = 0;
    std::uint64_t continuation_publication_failed_evicted     = 0;
    std::uint64_t continuation_publication_failed_alias_moved = 0;
    std::uint64_t continuation_publication_failed_lineage     = 0;
    std::uint64_t continuation_publication_failed_error       = 0;
    std::uint64_t continuation_restored_tokens       = 0;
    std::uint64_t continuation_restored_bytes        = 0;
    std::uint64_t continuation_l1_restore_successes   = 0;
    std::uint64_t continuation_l2_restore_successes   = 0;
    std::uint64_t continuation_l3_restore_successes   = 0;
    std::uint64_t continuation_l1_restored_tokens     = 0;
    std::uint64_t continuation_l2_restored_tokens     = 0;
    std::uint64_t continuation_l3_restored_tokens     = 0;
    std::uint64_t continuation_l1_restored_bytes      = 0;
    std::uint64_t continuation_l2_restored_bytes      = 0;
    std::uint64_t continuation_l3_restored_bytes      = 0;
    std::uint64_t continuation_session_restores       = 0;
    std::uint64_t continuation_stable_prefix_restores = 0;
    std::uint64_t continuation_miss_disabled          = 0;
    std::uint64_t continuation_miss_no_alias          = 0;
    std::uint64_t continuation_miss_not_attempted     = 0;
    std::uint64_t continuation_miss_entry_unavailable_or_corrupt = 0;
    std::uint64_t continuation_miss_not_deeper        = 0;
    std::uint64_t continuation_miss_preflight_rejected = 0;
    std::uint64_t continuation_miss_rollback_conflict = 0;
    std::uint64_t continuation_miss_no_lane           = 0;
    std::uint64_t continuation_miss_restore_failed    = 0;
    // Restore-failure attribution. These sum to continuation_restore_failures.
    std::uint64_t continuation_restore_failed_kv_reservation = 0;
    std::uint64_t continuation_restore_failed_verify_depth   = 0;
    std::uint64_t continuation_restore_failed_inventory      = 0;
    std::uint64_t continuation_restore_failed_metadata       = 0;
    std::uint64_t continuation_restore_failed_decode         = 0;
    std::uint64_t continuation_restore_failed_lane           = 0;
    std::uint64_t continuation_l2_lookup_microseconds = 0;
    std::uint64_t continuation_l2_lookup_operations   = 0;
    std::uint64_t continuation_l3_lookup_microseconds = 0;
    std::uint64_t continuation_l3_lookup_operations   = 0;
    std::uint64_t continuation_preflight_microseconds = 0;
    std::uint64_t continuation_preflight_operations   = 0;
    std::uint64_t continuation_l2_restore_microseconds = 0;
    std::uint64_t continuation_l2_restore_operations  = 0;
    std::uint64_t continuation_l3_restore_microseconds = 0;
    std::uint64_t continuation_l3_restore_operations  = 0;
    std::uint64_t continuation_l2_admission_microseconds = 0;
    std::uint64_t continuation_l2_admission_operations = 0;
    // Device-to-host export of a retained lane into a continuation image, performed by the
    // publication worker after the request completed; it holds the lane but not the execution
    // thread.
    std::uint64_t continuation_export_microseconds     = 0;
    std::uint64_t continuation_export_operations       = 0;
    std::uint64_t continuation_l3_persistence_microseconds = 0;
    std::uint64_t continuation_l3_persistence_operations = 0;
    std::uint64_t continuation_persistence_queued    = 0;
    std::uint64_t continuation_persistence_coalesced = 0;
    std::uint64_t continuation_persistence_successes = 0;
    std::uint64_t continuation_persistence_failures  = 0;
    std::uint64_t continuation_l2_bytes              = 0;
    std::uint64_t continuation_l3_bytes              = 0;
    std::uint32_t continuation_l2_entries            = 0;
    std::uint32_t continuation_l3_entries            = 0;
    // Entries pushed out of a tier because the live working set exceeded its byte budget. A TTL
    // expiry is not counted: reclaiming state that went cold is the cache working, while a
    // capacity eviction is the tier being too small for what is actually in use.
    std::uint64_t continuation_l2_evictions          = 0;
    std::uint64_t continuation_l2_evicted_bytes      = 0;
    std::uint64_t continuation_l3_evictions          = 0;
    std::uint64_t continuation_l3_evicted_bytes      = 0;
    std::uint64_t l1_evictions                        = 0;
    std::uint64_t l1_demotions                        = 0;
    // On-demand KV growth. A request reserves a bounded decode window at admission and acquires
    // the rest of its output as it generates: `attempts` counts round boundaries that asked,
    // `forced_spills` counts retained sessions demoted to make room, and `curtailed` counts
    // requests that ended early at `length` because neither rung found pages.
    // Retained sessions demoted to make room for a restore that would otherwise have been
    // refused and cold-prefilled at full cost.
    std::uint64_t kv_restore_reclaimed_lanes          = 0;
    std::uint64_t kv_growth_attempts                  = 0;
    std::uint64_t kv_growth_forced_spills             = 0;
    std::uint64_t kv_growth_curtailed                 = 0;
    std::uint64_t l1_resident_bytes                   = 0;
    std::uint32_t l1_resident_entries                 = 0;
    // Requests refused before they could be logged: the bounded FIFO was full (429) or the
    // admission deadline expired while they waited (503). A rejected request produces no
    // request-log record, so without these counters it is invisible.
    std::uint64_t admission_rejected_overloaded      = 0;
    std::uint64_t admission_rejected_queue_timeout   = 0;
    // Cumulative wall time requests spent between submission and admission, and the number of
    // admissions that contributed. Their ratio is the mean queue delay, the quantity that
    // dominates TTFT once the cache is working.
    double queue_seconds_total                       = 0.0;
    std::uint64_t admitted_requests                  = 0;
    // How the single execution thread spent its wall clock. Every unit below runs behind one
    // mutex, so a second spent in any one of them is a second in which every other resident lane
    // makes no progress. Decode rounds are short and prefill chunks are long, so their ratio is
    // what decides whether a decoding request is starved by someone else's prompt.
    double worker_decode_seconds                     = 0.0;
    double worker_prefill_seconds                    = 0.0;
    double worker_admission_seconds                  = 0.0;
    // Decomposition of worker_admission_seconds. `calls` counts scheduler iterations that entered
    // the admission attempt, most of which admit nothing, so the per-call cost of the three phases
    // is what determines how much of the execution thread admission consumes.
    std::uint64_t worker_admission_calls             = 0;
    double worker_admission_plan_seconds             = 0.0;
    double worker_admission_restore_seconds          = 0.0;
    double worker_admission_commit_seconds           = 0.0;
    double worker_publish_seconds                    = 0.0;
    double worker_upkeep_seconds                     = 0.0;
    std::uint64_t worker_decode_rounds               = 0;
    std::uint64_t worker_prefill_steps               = 0;
    std::uint32_t running_requests                   = 0;
    std::uint32_t prefilling_requests                = 0;
    std::uint32_t decode_ready_requests              = 0;
    std::uint32_t waiting_requests                   = 0;
};

// Session persistence outcomes. Tokens count the resident session depth moved; bytes count the
// snapshot file payload on disk; session_digest identifies the session (see SlotState).
struct SlotSaveResult {
    std::uint32_t tokens = 0;
    std::uint64_t bytes  = 0;
    double seconds       = 0.0;
    std::string session_digest;
};

struct SlotRestoreResult {
    std::uint32_t tokens = 0;
    std::uint64_t bytes  = 0;
    double seconds       = 0.0;
    std::string session_digest;
};

// One retained turn checkpoint of a resident session: the ledger depth it rewinds to and the
// digest of the ledger prefix up to that frontier (same FNV-1a 64 hex encoding as
// SlotState::session_digest, computed over the prefix only).
struct SlotCheckpoint {
    std::uint32_t frontier = 0;
    std::string session_digest;
};

// One Engine lane's occupancy for /slots-style reporting: an active request's prompt size, or
// the retained resident session. session_digest is a stable identifier of the exact resident
// token ledger (FNV-1a 64 as 16 hex chars) - equal digests mean the identical session; clients
// treat it as opaque and may pass it back as a slot-operation precondition. checkpoints lists
// the retained turn checkpoints (oldest first) a diverging prompt can restore from.
struct SlotState {
    bool processing              = false;
    bool retained                = false;
    std::uint32_t prompt_tokens  = 0;
    std::uint32_t cached_tokens  = 0;
    std::string session_digest;
    std::vector<SlotCheckpoint> checkpoints;
};

// Raised when a slot operation's session precondition (if_digest) does not match the lane's
// resident session.
class SlotSessionMismatch final : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

struct LoadSummary {
    std::string target;
    std::string model_id;
    std::string weights_id;
    // Complete base-artifact identity. Session images bind to this rather than only to the
    // registered model/weights strings, which are not unique to one set of weight bytes.
    std::array<std::uint8_t, 32> artifact_fingerprint{};
    double load_seconds                = 0.0;
    double upload_seconds              = 0.0;
    std::uint64_t artifact_bytes_read  = 0;
    std::uint64_t host_to_device_bytes = 0;
    std::uint64_t peak_staging_bytes   = 0;
    std::size_t tensor_count           = 0;
    std::size_t resource_count         = 0;
    // Discovered adapters in pool order; empty when no adapter was discovered. Every adapter
    // here is selectable by a request of its kind regardless of what is currently resident.
    std::vector<LoraAdapterInfo> lora_adapters;
    // The loaded target serves System One decisions (a compile-time trait of its package).
    bool decisions_supported = false;
    // Bank rank, resident slot count, the device bytes the bank cost, and the disk bytes the
    // whole pool occupies. A lower-rank pool adapter is zero-padded into the bank rank.
    std::int32_t lora_rank             = 0;
    std::uint32_t lora_slots           = 0;
    std::uint64_t lora_device_bytes    = 0;
    std::uint64_t lora_file_bytes      = 0;
};

} // namespace ninfer
