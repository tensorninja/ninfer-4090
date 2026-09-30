#pragma once

// LoRA adapter pool discovery and slot residency for the Qwen3.8-27B identity.
//
// An adapter artifact is a normal `.ninfer` container carrying only BF16 `contiguous-le-v1`
// rank-two tensors under the identity `{qwen3.8-27b, lora-bf16}`. It restates no frontend
// resource, so the base artifact remains the sole authority for tokenizer, template and
// preprocessor configs.
//
// The pool is every conforming adapter under one directory and is unbounded. The device bank is
// a fixed number of slots, each a slab of one adapter, laid out slot-major in a single
// allocation. Every site's slot stride is then the same constant and every plane address is
// deterministic, which is what lets the bank participate in a captured graph: capture freezes
// the bank base pointer, the slab stride and the site geometry, while the selected slot is a
// device-resident per-row value the graph re-reads on every replay. Overwriting a slot's bytes
// is therefore invisible to the captured graph.
//
// One rank and one site inventory describe the whole bank because both are frozen into that
// capture. The profile is the union of every pool adapter's sites at the pool's maximum rank; an
// adapter that trained fewer sites, or trained at a lower rank, stages zeros into the remainder.
// That is exact rather than approximate - a zero factor contributes nothing - and it is the
// padding contract `ninfer/ops/lora.h` already states.
//
// An adapter is of one of two kinds. A generative adapter carries only factors. A decision adapter
// additionally carries the System One pointer head - the four `decision/head/*` BF16 objects - and
// the `decision/metadata` JSON resource, all five together. When the pool holds any decision
// adapter every slab gains the head region after its sites; a generative occupant stages zeros
// there, which no decision ever reads because decisions select only decision adapters.

#include <ninfer/targets/qwen3_8/model_view.h>
#include <ninfer/types.h>

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_27b/impl/load/bindings.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_8_27b::detail {

inline constexpr std::string_view kLoraWeightsId = "lora-bf16";

// One registered site inside a slab. Offsets are byte positions inside one adapter slab, so the
// same plan describes every slot. `present` is a property of the bank profile, not of any single
// adapter: a profile site an adapter never trained is staged as zeros.
struct LoraSitePlan {
    std::uint64_t a_offset = 0;
    std::uint64_t b_offset = 0;
    std::int32_t rows      = 0; // N of the destination this site corrects
    std::int32_t columns   = 0; // K of the activation this site reads
    bool present           = false;
};

struct LoraFullLayerPlan {
    LoraSitePlan query;
    LoraSitePlan gate;
    LoraSitePlan key;
    LoraSitePlan value;
    LoraSitePlan output;
    LoraSitePlan down;
};

struct LoraGdnLayerPlan {
    LoraSitePlan output;
    LoraSitePlan down;
};

// Which registered sites one artifact carries. `query_gate` covers the shared down-projection
// factor and both of its up-projections, which the attention parent stores as row selections of
// one module and so can only be present or absent together.
struct LoraFullLayerInventory {
    bool query_gate = false;
    bool key        = false;
    bool value      = false;
    bool output     = false;
    bool down       = false;
};

struct LoraGdnLayerInventory {
    bool output = false;
    bool down   = false;
};

struct LoraInventory {
    std::array<LoraFullLayerInventory, kFullAttentionLayers> full_layers{};
    std::array<LoraGdnLayerInventory, kGdnLayers> gdn_layers{};

    [[nodiscard]] bool any() const noexcept;
    // Union with another adapter's inventory. The bank profile is the union over the pool.
    void merge(const LoraInventory& other) noexcept;
    // Number of distinct sites, used only for diagnostics.
    [[nodiscard]] std::size_t site_count() const noexcept;
};

// Byte offsets of the pointer head inside one slab: BF16 [pointer_dim, hidden] weights and
// [pointer_dim] biases of the query and key projections.
struct DecisionHeadPlan {
    std::uint64_t query_weight = 0;
    std::uint64_t query_bias   = 0;
    std::uint64_t key_weight   = 0;
    std::uint64_t key_bias     = 0;
};

// The frozen bank geometry. Built once, before the first slot is staged.
struct LoraBankProfile {
    std::int32_t rank        = 0;
    std::uint64_t slab_bytes = 0;
    LoraInventory inventory;
    std::array<LoraFullLayerPlan, kFullAttentionLayers> full_layers;
    std::array<LoraGdnLayerPlan, kGdnLayers> gdn_layers;
    // Present when the pool holds a decision adapter.
    std::optional<DecisionHeadPlan> decision_head;
};

// Host metadata of a decision adapter, read from its `decision/metadata` resource.
struct DecisionAdapterMetadata {
    float temperature = 1.0F;
    std::string description;
    std::string release_date;
};

// One discovered adapter. `fingerprint` is the container's SHA-256 content identity and is what
// namespaces the adapter's continuation state; a pool position is a residency detail that must
// never reach a cache key or a persisted image. The fingerprint covers the head and its metadata,
// so a head-only retrain is a different adapter scope.
struct LoraPoolEntry {
    std::string name;
    std::filesystem::path path;
    std::int32_t rank        = 0;
    std::uint64_t file_bytes = 0;
    artifact::Sha256Digest fingerprint{};
    LoraInventory inventory;
    LoraAdapterKind kind = LoraAdapterKind::Generative;
    DecisionAdapterMetadata decision;
};

struct LoraDiscovery {
    std::vector<LoraPoolEntry> pool;
    LoraBankProfile profile;
    artifact::FingerprintCacheOptions fingerprint_cache;
    // Files under the directory that are not usable adapters, each with its reason. Reported at
    // load so a foreign, corrupt or over-rank file is visible rather than silently absent.
    std::vector<std::string> rejected;
};

// Scans `directory` for `*.ninfer` adapters, orders the pool by name, and builds the union
// profile. Rejects a foreign identity, an unregistered rank, a rank above `rank_ceiling`, an
// artifact carrying no registered site, and a malformed factor.
[[nodiscard]] LoraDiscovery
discover_lora_pool(const LoraOptions& options,
                   const artifact::FingerprintCacheOptions& fingerprint_cache);

// The device bank: `slots` slabs in one allocation, plus the pool that can occupy them and the
// pinned slab used to assemble one adapter for upload.
class LoraBank {
public:
    LoraBank(LoraDiscovery discovery, std::uint32_t slots, DeviceContext& device);

    [[nodiscard]] const std::vector<LoraPoolEntry>& pool() const noexcept { return pool_; }
    [[nodiscard]] const LoraBankProfile& profile() const noexcept { return profile_; }
    [[nodiscard]] std::uint32_t slots() const noexcept { return slots_; }
    [[nodiscard]] std::uint64_t device_bytes() const noexcept { return device_bytes_; }
    [[nodiscard]] std::uint64_t pool_file_bytes() const noexcept { return pool_file_bytes_; }
    [[nodiscard]] const RuntimeModelView::Lora& view() const noexcept { return view_; }

    // Overwrite `slot` with pool entry `index`. The slot's device address does not move, so a
    // captured graph replaying against the bank is unaffected; only the bytes change. The caller
    // owns the guarantee that no live continuation depends on the slot's previous occupant.
    void stage(std::uint32_t slot, std::size_t index, DeviceContext& device);
    // Two-phase form used by admission: all file access, fingerprint verification and host
    // assembly happen before retained lanes are displaced; commit is only the bounded upload.
    void prepare(std::size_t index);
    void commit(std::uint32_t slot, DeviceContext& device);

    [[nodiscard]] std::uint64_t stage_count() const noexcept { return stage_count_; }
    [[nodiscard]] double stage_seconds() const noexcept { return stage_seconds_; }

private:
    void assemble(const LoraPoolEntry& entry, const artifact::Reader& reader);

    std::vector<LoraPoolEntry> pool_;
    LoraBankProfile profile_;
    artifact::FingerprintCacheOptions fingerprint_cache_;
    std::uint32_t slots_           = 0;
    std::uint64_t device_bytes_    = 0;
    std::uint64_t pool_file_bytes_ = 0;
    std::unique_ptr<DeviceArena> arena_;
    unsigned char* base_ = nullptr;
    std::optional<PinnedHostBuffer> staging_;
    RuntimeModelView::Lora view_;
    std::optional<std::size_t> prepared_index_;
    std::chrono::steady_clock::time_point prepare_started_{};
    std::uint64_t stage_count_ = 0;
    double stage_seconds_      = 0.0;
};

} // namespace ninfer::targets::qwen3_8_27b::detail
