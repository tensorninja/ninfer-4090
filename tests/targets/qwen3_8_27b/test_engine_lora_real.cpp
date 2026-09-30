// Runtime LoRA against the real 27B artifact.
//
// The gate that matters here is that prefill applies the selected adapter. A regression that
// bound the adapter only for the batched decode paths still produced adapted-looking output
// after the first token, so an "output differs from base" check over a long completion does not
// catch it. Asking for exactly one greedy token does: that token is the argmax of the logits the
// prefill chunk produced, so it can only move if prefill ran with the adapter.
#include "ninfer/engine.h"

#include <unistd.h> // getpid, for a per-process temporary pool directory

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <system_error>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

// The pool is discovered from a directory and an adapter's name is its file stem. The trained arm
// carries a behavioural signature; a synthetic random delta is unstructured noise that does not
// reliably move a confident argmax, and its numerics are already covered by
// tests/ops/test_lora_delta_add.cpp.
constexpr const char* kZeroName    = "zero";
constexpr const char* kTrainedName = "trained";
constexpr const char* kGdnName     = "gdn-only";

// A temporary adapter directory built from the fixtures the environment names. Discovery orders
// the pool by file name, so the pool index of a name here is its sorted position.
class PoolDirectory {
public:
    explicit PoolDirectory(const char* label) {
        root_ = std::filesystem::temp_directory_path() /
                ("ninfer_lora_pool_" + std::string(label) + "_" + std::to_string(::getpid()));
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

// A real chat turn, not a synthetic token run. The first assistant token has to be genuinely
// contested for an argmax comparison to have any sensitivity, and the public API exposes no
// logprobs to compare instead.
constexpr const char* kMathPrompt =
    "Let a and b be real numbers with a + b = 5 and a * b = 3. Find a^5 - b^5.";

ninfer::PromptInput chat_prompt(const char* text = kMathPrompt) {
    ninfer::ChatMessage message;
    message.role = "user";
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = text, .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

// A self-contained image, so the check needs no fixture file.
std::vector<std::uint8_t> gradient_ppm() {
    std::vector<std::uint8_t> ppm;
    const std::string header = "P6\n64 64\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < 64 * 64; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

ninfer::PromptInput vision_prompt() {
    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = gradient_ppm();
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "inline.ppm";

    ninfer::ChatMessage message;
    message.role = "user";
    message.parts.push_back(std::move(image));
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "What is visible?", .media = {}});
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
    // The arms share one prompt. Reuse would let a later arm restore a prefix another arm
    // prefilled instead of prefilling under its own adapter.
    options.execution.allow_prefix_reuse = false;
    options.stop.include_model_defaults  = false;
    return options;
}

std::vector<ninfer::TokenId> run(ninfer::Engine& engine, std::uint32_t output_tokens,
                                 std::optional<std::string> adapter) {
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare(chat_prompt()), greedy(output_tokens, adapter));
    return result.generated_token_ids;
}

// Discovery names an adapter by its file stem and orders the pool by file name.
int verify_registration(const ninfer::Engine& engine) {
    const ninfer::LoadSummary& load = engine.load_summary();
    std::vector<std::string> names;
    for (const ninfer::LoraAdapterInfo& adapter : load.lora_adapters) {
        if (adapter.kind != ninfer::LoraAdapterKind::Generative) {
            std::cerr << "adapter " << adapter.name << " is not generative\n";
            return 1;
        }
        names.push_back(adapter.name);
    }
    if (names.size() != 2 || names[0] != kTrainedName || names[1] != kZeroName) {
        std::cerr << "discovered adapter names are wrong: " << names.size() << " entries\n";
        return 1;
    }
    // The whole pool is selectable while only `slots` of it is resident; a client never sees the
    // difference, so the summary has to report both.
    if (load.lora_slots == 0 || load.lora_slots > names.size()) {
        std::cerr << "resident slot count " << load.lora_slots << " does not fit a pool of "
                  << names.size() << '\n';
        return 1;
    }
    return 0;
}

// The regression gate: one greedy token, produced entirely by prefill.
int verify_prefill_applies_adapter(ninfer::Engine& engine) {
    const std::vector<ninfer::TokenId> base    = run(engine, 1, std::nullopt);
    const std::vector<ninfer::TokenId> adapted = run(engine, 1, kTrainedName);
    if (base.size() != 1 || adapted.size() != 1) {
        std::cerr << "single-token requests did not generate exactly one token\n";
        return 1;
    }
    if (base[0] == adapted[0]) {
        std::cerr << "prefill ignored the selected LoRA adapter: the first greedy token is "
                  << base[0] << " both with and without the adapter\n";
        return 1;
    }
    return 0;
}

// A zero-delta adapter must leave greedy decoding byte-identical to the base weights, and must
// not leave the bank selected for the request that follows it.
int verify_zero_adapter_is_neutral(ninfer::Engine& engine) {
    constexpr std::uint32_t kTokens = 8;
    const std::vector<ninfer::TokenId> base = run(engine, kTokens, std::nullopt);
    const std::vector<ninfer::TokenId> zero = run(engine, kTokens, kZeroName);
    const std::vector<ninfer::TokenId> after = run(engine, kTokens, std::nullopt);
    if (base.size() != kTokens || zero.size() != kTokens || after.size() != kTokens) {
        std::cerr << "greedy requests did not generate the requested token count\n";
        return 1;
    }
    if (zero != base) {
        std::cerr << "a zero-delta adapter changed greedy output\n";
        return 1;
    }
    if (after != base) {
        std::cerr << "an adapter selection leaked into the following base request\n";
        return 1;
    }
    return 0;
}

// One compact decode batch carrying a different adapter on each lane.
//
// Per-row adapter routing is the only genuinely new mechanism in the feature, and it cannot be
// checked at batch 1: every row would carry the same value. Four equal-length requests are not
// enough either. `program_impl.h` fills the ingress over a compact `row` index and dereferences
// `lanes[row]`, so while the active lane set is {0,1,2,3} the row and the lane id coincide and
// indexing by the wrong one is indistinguishable. Staggering the output lengths retires lanes at
// different rounds, which leaves the surviving lane set non-contiguous and forces row != lane for
// most of the run.
int verify_mixed_adapter_batch(ninfer::Engine& engine) {
    struct Arm {
        std::optional<std::string> adapter;
        std::uint32_t tokens;
    };
    const std::vector<Arm> arms{{std::nullopt, 2}, {kTrainedName, 8}, {kZeroName, 4},
                                {kTrainedName, 6}};

    // Sequential references first, one request in flight at a time.
    std::vector<std::vector<ninfer::TokenId>> sequential;
    sequential.reserve(arms.size());
    for (const Arm& arm : arms) { sequential.push_back(run(engine, arm.tokens, arm.adapter)); }
    // Without a real difference between the arms, a routing fault would be undetectable below.
    if (sequential[0] == std::vector<ninfer::TokenId>(sequential[1].begin(),
                                                      sequential[1].begin() + 2)) {
        std::cerr << "mixed-batch check is inert: base and trained agree on this prompt\n";
        return 1;
    }

    // Submitting every request before waiting on any is what puts them in one round; `submit`
    // establishes queue membership synchronously and `wait` consumes independently.
    std::vector<ninfer::GenerationHandle> handles;
    handles.reserve(arms.size());
    for (const Arm& arm : arms) {
        handles.push_back(
            engine.submit(engine.prepare(chat_prompt()), greedy(arm.tokens, arm.adapter)));
    }
    for (std::size_t index = 0; index < handles.size(); ++index) {
        const ninfer::GenerationResult result = handles[index].wait();
        if (result.generated_token_ids != sequential[index]) {
            std::cerr << "lane " << index
                      << " of a mixed-adapter batch does not match its sequential "
                         "single-adapter run\n";
            return 1;
        }
    }
    return 0;
}

// Prefix reuse is isolated per adapter: KV and the FP32 GDN recurrent state encode the weights
// that produced them, so a prefix built under one adapter is not reusable under another.
//
// Each direction needs a prompt no other arm has ever prefilled. `allow_prefix_reuse` gates only
// whether a request *consumes* reuse, not whether it leaves a retained prefix behind, so a shared
// prompt would let the consumer legitimately reuse a same-adapter prefix planted earlier and the
// check would report a leak that is not one. The positive control matters as much as the negative
// one: asserting "no reuse" alone would pass in an engine where reuse never engages.
int verify_prefix_reuse_isolation(ninfer::Engine& engine, const char* text,
                                  const std::optional<std::string>& producer,
                                  const std::optional<std::string>& consumer, const char* label) {
    constexpr std::uint32_t kTokens = 4;
    auto warm = [&](const std::optional<std::string>& adapter) {
        ninfer::RequestOptions options       = greedy(kTokens, adapter);
        options.execution.allow_prefix_reuse = true;
        return engine.generate(engine.prepare(chat_prompt(text)), options);
    };

    warm(producer);
    const ninfer::GenerationResult repeat = warm(producer);
    if (repeat.reused_prompt_tokens == 0) {
        std::cerr << label << ": prefix reuse never engaged, so the isolation check is inert\n";
        return 1;
    }
    const ninfer::GenerationResult crossed = warm(consumer);
    if (crossed.reused_prompt_tokens != 0) {
        std::cerr << label << ": reused " << crossed.reused_prompt_tokens
                  << " prompt tokens from a prefix built under a different adapter\n";
        return 1;
    }
    std::cout << "  " << label << ": same-adapter reuse " << repeat.reused_prompt_tokens
              << " tokens, crossing reused 0\n";
    return 0;
}

int verify_prefix_reuse_is_adapter_isolated(ninfer::Engine& engine) {
    if (const int result = verify_prefix_reuse_isolation(
            engine, "Compute the sum of the first 40 positive odd integers.", kTrainedName,
            std::nullopt, "adapter prefix -> base request");
        result != 0) {
        return result;
    }
    return verify_prefix_reuse_isolation(
        engine, "Compute the product of the first 6 positive prime numbers.", std::nullopt,
        kTrainedName, "base prefix -> adapter request");
}

// Multimodal prefill applies the adapter through the same `configure_text_card` call as the text
// path, but the two had never been executed together, and that call site is the exact shape of
// the defect this file was written for. The vision tower itself carries no adapter - the bank
// holds decoder layers only - so what is checked here is that the adapter reaches the decoder
// while it consumes image tokens.
int verify_vision_applies_adapter(ninfer::Engine& engine) {
    constexpr std::uint32_t kTokens = 12;
    auto caption = [&](std::optional<std::string> adapter) {
        return engine
            .generate(engine.prepare(vision_prompt()), greedy(kTokens, std::move(adapter)))
            .generated_token_ids;
    };
    const std::vector<ninfer::TokenId> base    = caption(std::nullopt);
    const std::vector<ninfer::TokenId> adapted = caption(kTrainedName);
    if (base.empty() || adapted.empty()) {
        std::cerr << "a multimodal request generated nothing\n";
        return 1;
    }
    if (base == adapted) {
        std::cerr << "the adapter did not change multimodal output at all: prefill over image "
                     "tokens ignored the selected adapter\n";
        return 1;
    }
    // Whether the *first* token moves depends on how strongly this particular adapter bears on a
    // caption, which is a property of the adapter rather than of the engine. Report it instead of
    // asserting it, so the check stays a statement about adapter application.
    std::cout << "  vision: adapter changed the caption; first token "
              << (base[0] == adapted[0] ? "unchanged" : "moved") << '\n';
    return 0;
}

// A saved slot must carry the adapter that produced it. The image holds KV and FP32 GDN state
// that encode that adapter's weights, and the restored lane is a prefix-reuse candidate the
// moment it lands, so losing the identity hands one adapter's state to a request using other
// weights. The positive arm is what distinguishes a restored identity from a blanked one: if the
// lane came back as base, the negative arm alone would still pass.
int verify_slot_restore_keeps_adapter_identity(ninfer::Engine& engine) {
    constexpr std::uint32_t kTokens = 4;
    const char* text = "Compute the greatest common divisor of 1071 and 462.";
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "ninfer_lora_slot_identity.bin";
    auto warm = [&](const std::optional<std::string>& adapter) {
        ninfer::RequestOptions options       = greedy(kTokens, adapter);
        options.execution.allow_prefix_reuse = true;
        return engine.generate(engine.prepare(chat_prompt(text)), options);
    };

    const ninfer::GenerationResult produced = warm(kTrainedName);
    if (produced.slot < 0 || produced.session_digest.empty()) {
        std::cerr << "the adapter request did not retain a session to save\n";
        return 1;
    }
    int failures = 0;
    try {
        (void)engine.save_slot(static_cast<std::uint32_t>(produced.slot), path.string());
        // Drop the producing lane's copy, so the restored lane is the only holder of this
        // prompt's state and the positive arm below cannot be satisfied by some other lane.
        (void)engine.erase_slot(static_cast<std::uint32_t>(produced.slot));

        // Restore into a lane whose last request was base, so the lane's own adapter field is -1
        // and disagrees with the image. Restoring into the producing lane proves nothing: that
        // lane already carries the right value, so a restore that dropped the identity entirely
        // would still look correct.
        const ninfer::GenerationResult unadapted = warm(std::nullopt);
        if (unadapted.slot < 0) {
            std::cerr << "the base request did not report a slot\n";
            return 1;
        }
        const auto lane = static_cast<std::uint32_t>(unadapted.slot);
        (void)engine.restore_slot(lane, path.string());

        const ninfer::GenerationResult same = warm(kTrainedName);
        if (same.reused_prompt_tokens == 0) {
            std::cerr << "a restored slot did not reuse under the adapter that produced it: the "
                         "restore lost the adapter identity\n";
            failures = 1;
        }
        const ninfer::GenerationResult crossed = warm(std::nullopt);
        if (failures == 0 && crossed.reused_prompt_tokens != 0) {
            std::cerr << "a base request reused " << crossed.reused_prompt_tokens
                      << " prompt tokens from a slot saved under an adapter\n";
            failures = 1;
        }
        if (failures == 0) {
            std::cout << "  slot restore: adapter identity survived save/restore into slot " << lane
                      << '\n';
        }
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        throw;
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    return failures;
}

// A prompt long enough to reach the workspace plan's peak stage.
//
// The arena capacity is fitted exactly to the plan, so anything the plan does not reserve shows
// up only once a request actually reaches that peak - short prompts run inside the slack and look
// fine. The per-chunk bank selector was such an allocation: one int32, 256 bytes after alignment,
// enough to push the peak past the capacity and abort the request with std::bad_alloc while the
// same prompt served from the base weights succeeded.
// This needs its own Engine, and a deliberately minimal one. With several lanes or Vision
// enabled the decode stages dominate the workspace peak and leave prefill enough slack to hide an
// unreserved allocation; at one lane with no Vision, text prefill *is* the peak and the capacity
// is fitted to it. That single-lane long-context shape is also the product's default.
int verify_long_prompt_fits_the_workspace(const char* artifact, const char* zero_path,
                                          const char* trained_path) {
    PoolDirectory pool("workspace");
    pool.add(kZeroName, zero_path).add(kTrainedName, trained_path);

    ninfer::EngineOptions options;
    options.artifact_path   = artifact;
    options.max_context     = 40960;
    options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(40960);
    options.prefill_chunk   = 1024;
    options.max_concurrency = 1;
    options.lora.directory  = pool.path();
    options.lora.slots      = 2;
    ninfer::Engine engine(options);
    for (const std::uint32_t tokens : {1024U, 8192U, 32768U}) {
        std::vector<ninfer::TokenId> ids(tokens, 198);
        const ninfer::GenerationResult result =
            engine.generate(engine.prepare_tokens(ids), greedy(1, kTrainedName));
        if (result.generated_token_ids.size() != 1) {
            std::cerr << "an adapted " << tokens << "-token prefill did not generate its token\n";
            return 1;
        }
    }
    std::cout << "  long prompt: adapted prefill reached the workspace peak at 32768 tokens\n";
    return 0;
}

// The Gated DeltaNet output correction, in isolation.
//
// 48 of the 64 layers are GDN, and `gdn/output` is the only registered site on them besides
// `mlp/down`, so this correction carries most of the adapted token-mixing path. Every other check
// here registers a complete adapter, where six other sites move the output too: dropping the GDN
// correction entirely would leave all of them passing. This registers an adapter whose *only*
// site is `gdn/output`, so the base and adapted answers can differ for exactly one reason.
//
// It needs its own Engine because a one-site adapter cannot share a bank with a seven-site one -
// every registered adapter must present the same inventory.
//
// The fixture is generated at a deliberately large amplitude (`--kind random --sigma 0.2`). At the
// generator's default 0.02 the correction is real but too small to flip a greedy argmax on a
// short, confident prompt, which would make this gate report a defect that is not there. The
// question here is whether the delta reaches the residual at all, so the amplitude is chosen to
// answer that question unambiguously rather than to resemble a trained adapter.
std::vector<ninfer::TokenId> gdn_only_tokens(const char* artifact, const char* gdn_only_path,
                                             const char* trained_path, bool* moved) {
    PoolDirectory pool(trained_path == nullptr ? "gdn_alone" : "gdn_union");
    pool.add(kGdnName, gdn_only_path);
    if (trained_path != nullptr) { pool.add(kTrainedName, trained_path); }

    ninfer::EngineOptions options;
    options.artifact_path   = artifact;
    options.max_context     = 2048;
    options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    options.prefill_chunk   = 1024;
    options.max_concurrency = 1;
    options.lora.directory  = pool.path();
    options.lora.slots      = 1;

    ninfer::Engine engine(options);
    const std::vector<ninfer::TokenId> base    = run(engine, 24, std::nullopt);
    const std::vector<ninfer::TokenId> adapted = run(engine, 24, kGdnName);
    if (moved != nullptr) { *moved = base != adapted; }
    return adapted;
}

int verify_gdn_site_applies(const char* artifact, const char* gdn_only_path) {
    bool moved = false;
    (void)gdn_only_tokens(artifact, gdn_only_path, nullptr, &moved);
    if (!moved) {
        std::cerr << "a gdn/output-only adapter left the output unchanged, so the correction on "
                     "the 48 GDN layers never reached the residual\n";
        return 1;
    }
    std::cout << "  gdn/output: the 48-layer correction alone moves the output\n";
    return 0;
}

// Union-profile equivalence.
//
// The bank profile is the union of every pool adapter's sites, so putting a one-site adapter in a
// pool with a seven-site one widens the geometry the captured graph executes. The narrow adapter
// stages zeros into the six sites it never trained, and a zero factor contributes nothing, so its
// output must be identical to the same adapter loaded on its own. If it is not, the union either
// left residue from another adapter in the slab or mapped a factor to the wrong offset - both of
// which are silent, and neither of which any single-adapter check can see.
int verify_union_profile_preserves_narrow_adapter(const char* artifact, const char* gdn_only_path,
                                                  const char* trained_path) {
    const std::vector<ninfer::TokenId> alone =
        gdn_only_tokens(artifact, gdn_only_path, nullptr, nullptr);
    const std::vector<ninfer::TokenId> united =
        gdn_only_tokens(artifact, gdn_only_path, trained_path, nullptr);
    if (alone.empty() || united.empty()) {
        std::cerr << "a gdn/output-only request generated nothing\n";
        return 1;
    }
    if (alone != united) {
        std::cerr << "a one-site adapter produced different output in a union bank than on its "
                     "own, so the widened profile does not zero-fill the sites it never trained\n";
        return 1;
    }
    std::cout << "  union profile: a one-site adapter is unchanged by a seven-site pool peer\n";
    return 0;
}

int verify_unknown_adapter_is_rejected(ninfer::Engine& engine) {
    try {
        run(engine, 1, "not-registered");
    } catch (const ninfer::RequestError& error) {
        if (error.kind() != ninfer::RequestErrorKind::UnknownAdapter) {
            std::cerr << "unknown adapter raised the wrong request error kind\n";
            return 1;
        }
        return 0;
    }
    std::cerr << "an adapter name outside the pool was accepted\n";
    return 1;
}

// Slot swapping is transparent to output.
//
// With one slot and two pool adapters, every alternation evicts the resident adapter and stages
// the other over the same device address. That address is what the captured graph holds, so a
// swap must be invisible: each adapter has to produce exactly what it produced when it was the
// only occupant. This is the gate on staging - zeroing, rank padding, and the offsets a factor
// lands at - because a partial or misaligned stage still yields plausible text.
int verify_slot_swap_is_transparent(const char* artifact, const char* zero_path,
                                    const char* trained_path) {
    PoolDirectory pool("swap");
    pool.add(kZeroName, zero_path).add(kTrainedName, trained_path);

    constexpr std::uint32_t kTokens = 12;
    ninfer::EngineOptions options;
    options.artifact_path   = artifact;
    options.max_context     = 2048;
    options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    options.prefill_chunk   = 1024;
    options.max_concurrency = 1;
    options.lora.directory  = pool.path();
    options.lora.slots      = 1;

    ninfer::Engine engine(options);
    // References with no swap in between: each adapter is staged once, into an empty slot.
    const std::vector<ninfer::TokenId> trained_first = run(engine, kTokens, kTrainedName);
    const std::vector<ninfer::TokenId> zero_first    = run(engine, kTokens, kZeroName);
    if (trained_first.size() != kTokens || zero_first.size() != kTokens) {
        std::cerr << "a swap-arm request did not generate the requested token count\n";
        return 1;
    }
    if (trained_first == zero_first) {
        std::cerr << "swap check is inert: the two adapters agree on this prompt\n";
        return 1;
    }

    // Alternate, forcing an eviction and a restage before every request.
    for (int round = 0; round < 2; ++round) {
        if (run(engine, kTokens, kTrainedName) != trained_first) {
            std::cerr << "the trained adapter produced different output after being swapped back "
                         "into its slot\n";
            return 1;
        }
        if (run(engine, kTokens, kZeroName) != zero_first) {
            std::cerr << "the zero adapter produced different output after being swapped back "
                         "into its slot\n";
            return 1;
        }
    }
    // A base request in a swapping engine must still be exactly the base: the resident slot holds
    // whichever adapter ran last, and only the negative per-row index keeps it out.
    const std::vector<ninfer::TokenId> base_after = run(engine, kTokens, std::nullopt);
    if (base_after == trained_first) {
        std::cerr << "a base request returned the resident adapter's output\n";
        return 1;
    }
    std::cout << "  slot swap: both adapters reproduce their unswapped output at one slot after "
              << engine.load_summary().lora_slots << "-slot alternation\n";
    return 0;
}

int exercise(const char* artifact, const char* zero_path, const char* trained_path) {
    PoolDirectory pool("exercise");
    pool.add(kZeroName, zero_path).add(kTrainedName, trained_path);

    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context   = 2048;
    options.kv_capacity   = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    options.prefill_chunk = 1024;
    // Four lanes so a mixed-adapter decode batch is reachable; the prompts are short, so the
    // shared KV pool is far from its bound.
    options.max_concurrency = 4;
    options.enable_vision   = true;
    // Both adapters resident, so this file's routing checks are not also exercising residency.
    options.lora.directory = pool.path();
    options.lora.slots     = 2;

    ninfer::Engine engine(options);
    if (const int result = verify_registration(engine); result != 0) { return result; }
    if (const int result = verify_prefill_applies_adapter(engine); result != 0) { return result; }
    if (const int result = verify_zero_adapter_is_neutral(engine); result != 0) { return result; }
    if (const int result = verify_mixed_adapter_batch(engine); result != 0) { return result; }
    if (const int result = verify_prefix_reuse_is_adapter_isolated(engine); result != 0) {
        return result;
    }
    if (const int result = verify_vision_applies_adapter(engine); result != 0) { return result; }
    if (const int result = verify_slot_restore_keeps_adapter_identity(engine); result != 0) {
        return result;
    }
    if (const int result = verify_unknown_adapter_is_rejected(engine); result != 0) {
        return result;
    }
    return 0;
}

const char* env_or_null(const char* name) {
    const char* value = std::getenv(name);
    return (value != nullptr && *value != '\0') ? value : nullptr;
}

} // namespace

int main() {
    const char* artifact = env_or_null("NINFER_QWEN3_8_27B_WEIGHTS");
    const char* zero     = env_or_null("NINFER_QWEN3_8_27B_LORA_ZERO");
    const char* trained  = env_or_null("NINFER_QWEN3_8_27B_LORA_TRAINED");
    const char* gdn_only = env_or_null("NINFER_QWEN3_8_27B_LORA_GDN_ONLY");
    if (artifact == nullptr || zero == nullptr || trained == nullptr || gdn_only == nullptr) {
        std::cout << "skip: NINFER_QWEN3_8_27B_WEIGHTS, NINFER_QWEN3_8_27B_LORA_ZERO, "
                     "NINFER_QWEN3_8_27B_LORA_TRAINED and NINFER_QWEN3_8_27B_LORA_GDN_ONLY are "
                     "all required\n";
        return 77;
    }
    if (const int result = exercise(artifact, zero, trained); result != 0) { return result; }
    if (const int result = verify_slot_swap_is_transparent(artifact, zero, trained); result != 0) {
        return result;
    }
    if (const int result = verify_long_prompt_fits_the_workspace(artifact, zero, trained);
        result != 0) {
        return result;
    }
    if (const int result = verify_gdn_site_applies(artifact, gdn_only); result != 0) {
        return result;
    }
    if (const int result =
            verify_union_profile_preserves_narrow_adapter(artifact, gdn_only, trained);
        result != 0) {
        return result;
    }
    std::cout << "ok\n";
    return 0;
}
