// LoRA pool discovery, union profile, and slot staging.
//
// This is the gate on the bytes a slot holds. Everything above it - routing, prefix isolation,
// serving - can be checked against a running model, but only here can the slab itself be read
// back and compared to the artifact. Three properties are silent when broken:
//
//   - a site the bank profile carries but the staged adapter never trained must be exactly zero,
//     because that is what makes the union of a narrow and a wide adapter exact rather than
//     approximate;
//   - the trailing rank rows of A and the trailing rank columns of every row of B must be exactly
//     zero when a lower-rank adapter is staged into a wider bank;
//   - restaging must leave no residue of the slot's previous occupant, since a slot is reused for
//     the lifetime of the process.
//
// All three produce plausible output when wrong, so a text-level check cannot see them. This test
// needs the adapters and about one slab of device memory; it never loads the base model.
//
// With a decision adapter it also gates the pointer-head region: planned only for a pool holding
// a decision adapter, verbatim head bytes for a decision occupant, zeros for a generative one.
#include "targets/qwen3_8_27b/impl/load/lora_bindings.h"

#include "artifact/reader.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <unistd.h> // getpid, for a per-process temporary pool directory

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

namespace {

using namespace ninfer::targets::qwen3_8_27b::detail;

int failures = 0;

void check(bool condition, const std::string& what) {
    if (condition) { return; }
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
}

const char* env_or_null(const char* name) {
    const char* value = std::getenv(name);
    return (value != nullptr && *value != '\0') ? value : nullptr;
}

// A temporary pool directory of symlinks, so the fixtures stay where they are and the pool names
// are whatever this test wants them to be.
class PoolDirectory {
public:
    explicit PoolDirectory(const char* label) {
        root_ = std::filesystem::temp_directory_path() /
                ("ninfer_lora_pool_test_" + std::string(label) + "_" +
                 std::to_string(::getpid()));
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

std::vector<unsigned char> read_slot(const LoraBank& bank, std::uint32_t slot,
                                     ninfer::DeviceContext& device) {
    const std::uint64_t slab = bank.profile().slab_bytes;
    std::vector<unsigned char> host(static_cast<std::size_t>(slab));
    // The bank's first plane address is the base of slot 0; every slot is one slab further on.
    const auto& first = bank.view().gdn_layers;
    const unsigned char* base = nullptr;
    for (const auto& layer : first) {
        if (layer.output.present()) {
            base = static_cast<const unsigned char*>(layer.output.a.data) -
                   bank.profile().gdn_layers[static_cast<std::size_t>(&layer - first.data())]
                       .output.a_offset;
            break;
        }
    }
    if (base == nullptr) {
        for (std::size_t index = 0; index < bank.view().full_layers.size(); ++index) {
            const auto& layer = bank.view().full_layers[index];
            if (layer.down.present()) {
                base = static_cast<const unsigned char*>(layer.down.a.data) -
                       bank.profile().full_layers[index].down.a_offset;
                break;
            }
        }
    }
    check(base != nullptr, "the bank exposes at least one present site");
    if (base == nullptr) { return host; }
    CUDA_CHECK(cudaMemcpy(host.data(), base + static_cast<std::uint64_t>(slot) * slab,
                          static_cast<std::size_t>(slab), cudaMemcpyDeviceToHost));
    (void)device;
    return host;
}

bool all_zero(const unsigned char* data, std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
        if (data[index] != 0) { return false; }
    }
    return true;
}

// Walks the profile against one staged adapter and checks every site against the artifact.
void verify_staged_slab(const LoraBank& bank, std::size_t pool_index,
                        const std::vector<unsigned char>& slab) {
    const LoraPoolEntry& entry     = bank.pool()[pool_index];
    const LoraBankProfile& profile = bank.profile();
    ninfer::artifact::Reader reader(entry.path, {}, {});

    const auto verify_site = [&](const LoraSitePlan& plan, const std::string& a_name,
                                 const std::string& b_name, bool trained) {
        if (!plan.present) { return; }
        const auto a_bytes = static_cast<std::size_t>(profile.rank) *
                             static_cast<std::size_t>(plan.columns) * 2U;
        const auto b_bytes = static_cast<std::size_t>(plan.rows) *
                             static_cast<std::size_t>(profile.rank) * 2U;
        if (!trained) {
            check(all_zero(slab.data() + plan.a_offset, a_bytes),
                  entry.name + ": untrained site " + a_name + " is not zero");
            check(all_zero(slab.data() + plan.b_offset, b_bytes),
                  entry.name + ": untrained site " + b_name + " is not zero");
            return;
        }
        const ninfer::artifact::PayloadSpan a = reader.payload(a_name);
        const ninfer::artifact::PayloadSpan b = reader.payload(b_name);
        // A is row-major [rank, columns]: the artifact's rows first, then a zero tail.
        const auto a_stored = static_cast<std::size_t>(entry.rank) *
                              static_cast<std::size_t>(plan.columns) * 2U;
        check(std::memcmp(slab.data() + plan.a_offset, a.data.data(), a_stored) == 0,
              entry.name + ": " + a_name + " does not match the artifact");
        check(all_zero(slab.data() + plan.a_offset + a_stored, a_bytes - a_stored),
              entry.name + ": " + a_name + " rank padding is not zero");
        // B is row-major [rows, rank]: every row carries the artifact's columns then a zero tail.
        const auto stored_pitch = static_cast<std::size_t>(entry.rank) * 2U;
        const auto bank_pitch   = static_cast<std::size_t>(profile.rank) * 2U;
        bool rows_match = true;
        bool tails_zero = true;
        for (std::int32_t row = 0; row < plan.rows; ++row) {
            const unsigned char* out =
                slab.data() + plan.b_offset + static_cast<std::size_t>(row) * bank_pitch;
            rows_match = rows_match &&
                         std::memcmp(out, b.data.data() + static_cast<std::size_t>(row) *
                                                              stored_pitch,
                                     stored_pitch) == 0;
            tails_zero = tails_zero && all_zero(out + stored_pitch, bank_pitch - stored_pitch);
        }
        check(rows_match, entry.name + ": " + b_name + " does not match the artifact");
        check(tails_zero, entry.name + ": " + b_name + " rank padding is not zero");
        check(b_bytes >= bank_pitch, entry.name + ": " + b_name + " is empty");
    };

    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const bool full = layer >= 3 && (layer - 3) % 4 == 0;
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        if (full) {
            const std::size_t index            = (layer - 3) / 4;
            const LoraFullLayerPlan& plan      = profile.full_layers[index];
            const LoraFullLayerInventory& have = entry.inventory.full_layers[index];
            verify_site(plan.query, prefix + "attention/query_gate/lora_a",
                        prefix + "attention/query/lora_b", have.query_gate);
            verify_site(plan.gate, prefix + "attention/query_gate/lora_a",
                        prefix + "attention/gate/lora_b", have.query_gate);
            verify_site(plan.key, prefix + "attention/key/lora_a", prefix + "attention/key/lora_b",
                        have.key);
            verify_site(plan.value, prefix + "attention/value/lora_a",
                        prefix + "attention/value/lora_b", have.value);
            verify_site(plan.output, prefix + "attention/output/lora_a",
                        prefix + "attention/output/lora_b", have.output);
            verify_site(plan.down, prefix + "mlp/down/lora_a", prefix + "mlp/down/lora_b",
                        have.down);
        } else {
            const std::size_t index = layer - (layer >= 3 ? (layer - 3) / 4 + 1 : 0);
            const LoraGdnLayerPlan& plan      = profile.gdn_layers[index];
            const LoraGdnLayerInventory& have = entry.inventory.gdn_layers[index];
            verify_site(plan.output, prefix + "gdn/output/lora_a", prefix + "gdn/output/lora_b",
                        have.output);
            verify_site(plan.down, prefix + "mlp/down/lora_a", prefix + "mlp/down/lora_b",
                        have.down);
        }
    }
}

// The pointer head of a staged slot: verbatim artifact bytes for a decision occupant, exact zeros
// for a generative one.
void verify_staged_head(const LoraBank& bank, std::size_t pool_index,
                        const std::vector<unsigned char>& slab) {
    const LoraPoolEntry& entry = bank.pool()[pool_index];
    const DecisionHeadPlan& head = *bank.profile().decision_head;
    constexpr std::size_t kWeightBytes = 256U * 5120U * 2U;
    constexpr std::size_t kBiasBytes   = 256U * 2U;
    const struct {
        const char* name;
        std::uint64_t offset;
        std::size_t bytes;
    } objects[] = {{"decision/head/query/weight", head.query_weight, kWeightBytes},
                   {"decision/head/query/bias", head.query_bias, kBiasBytes},
                   {"decision/head/key/weight", head.key_weight, kWeightBytes},
                   {"decision/head/key/bias", head.key_bias, kBiasBytes}};
    if (entry.kind != ninfer::LoraAdapterKind::Decision) {
        for (const auto& object : objects) {
            check(all_zero(slab.data() + object.offset, object.bytes),
                  entry.name + ": a generative occupant leaves " + object.name + " nonzero");
        }
        return;
    }
    ninfer::artifact::Reader reader(entry.path, {}, {});
    for (const auto& object : objects) {
        const ninfer::artifact::PayloadSpan payload = reader.payload(object.name);
        check(payload.data.size() == object.bytes,
              entry.name + ": " + object.name + " has the registered size");
        check(std::memcmp(slab.data() + object.offset, payload.data.data(), object.bytes) == 0,
              entry.name + ": " + object.name + " does not match the artifact");
    }
}

// A generative and a decision adapter share one slot. The head region exists because the pool
// holds a decision adapter; it carries the head only while the decision adapter occupies the slot,
// and restaging either occupant reproduces its slab byte for byte.
void check_decision_pool(const char* generative, const char* decision) {
    PoolDirectory pool("decision");
    pool.add("decider", decision).add("writer", generative);

    ninfer::LoraOptions options;
    options.directory = pool.path();
    options.slots     = 1;

    LoraDiscovery discovery = discover_lora_pool(options, {});
    check(discovery.rejected.empty(), "no decision-pool fixture was rejected");
    check(discovery.pool.size() == 2, "both decision-pool fixtures entered the pool");
    if (discovery.pool.size() != 2) { return; }
    const LoraPoolEntry& decider = discovery.pool[0];
    check(decider.kind == ninfer::LoraAdapterKind::Decision,
          "the head-carrying adapter is a decision adapter");
    check(discovery.pool[1].kind == ninfer::LoraAdapterKind::Generative,
          "the factor-only adapter is generative");
    check(std::isfinite(decider.decision.temperature) && decider.decision.temperature > 0.0F,
          "the decision temperature is a positive finite value");
    check(decider.decision.release_date.size() == 10, "the release date is YYYY-MM-DD");
    check(discovery.profile.decision_head.has_value(), "a decision pool plans the head region");
    if (!discovery.profile.decision_head) { return; }

    const DecisionHeadPlan head = *discovery.profile.decision_head;
    const std::uint64_t slab    = discovery.profile.slab_bytes;
    const std::uint64_t head_end =
        std::max({head.query_weight + 256U * 5120U * 2U, head.key_weight + 256U * 5120U * 2U,
                  head.query_bias + 512U, head.key_bias + 512U});
    check(head_end <= slab, "the head region lies inside the slab");
    std::cout << "  decision pool: slab " << slab / (1024 * 1024) << " MiB, temperature "
              << decider.decision.temperature << '\n';

    ninfer::DeviceContext device(0);
    LoraBank bank(std::move(discovery), 1, device);
    check(bank.view().kinds.size() == 2 &&
              bank.view().kinds[0] == ninfer::LoraAdapterKind::Decision &&
              bank.view().kinds[1] == ninfer::LoraAdapterKind::Generative,
          "the bank view carries each adapter's kind");
    check(bank.view().decision_temperatures.size() == 2 &&
              bank.view().decision_temperatures[0] == bank.pool()[0].decision.temperature,
          "the bank view carries the decision temperature");
    check(bank.view().decision_head.query_weight.data != nullptr &&
              bank.view().decision_head.slot_stride == bank.profile().slab_bytes,
          "the bank view binds the head region per slot");

    bank.stage(0, 0, device);
    const std::vector<unsigned char> decision_first = read_slot(bank, 0, device);
    verify_staged_slab(bank, 0, decision_first);
    verify_staged_head(bank, 0, decision_first);

    bank.stage(0, 1, device);
    const std::vector<unsigned char> generative_slab = read_slot(bank, 0, device);
    verify_staged_slab(bank, 1, generative_slab);
    verify_staged_head(bank, 1, generative_slab);

    bank.stage(0, 0, device);
    const std::vector<unsigned char> decision_again = read_slot(bank, 0, device);
    check(decision_again == decision_first,
          "restaging the decision adapter leaves no residue of the generative occupant");
}

} // namespace

int main() {
    const char* wide     = env_or_null("NINFER_QWEN3_8_27B_LORA_ZERO");
    const char* narrow   = env_or_null("NINFER_QWEN3_8_27B_LORA_GDN_ONLY");
    const char* decision = env_or_null("NINFER_QWEN3_8_27B_LORA_DECISION");
    if (wide == nullptr || (narrow == nullptr && decision == nullptr)) {
        std::cout << "skip: NINFER_QWEN3_8_27B_LORA_ZERO and NINFER_QWEN3_8_27B_LORA_GDN_ONLY or "
                     "NINFER_QWEN3_8_27B_LORA_DECISION are required\n";
        return 77;
    }
    if (decision != nullptr) { check_decision_pool(wide, decision); }
    if (narrow == nullptr) {
        if (failures != 0) {
            std::cerr << failures << " check(s) failed\n";
            return 1;
        }
        std::cout << "ok\n";
        return 0;
    }

    // A seven-site adapter and a one-site adapter in one pool. The profile must be their union,
    // which is exactly the case that cannot arise from a single adapter's own inventory.
    PoolDirectory pool("union");
    pool.add("narrow", narrow).add("wide", wide);

    ninfer::LoraOptions options;
    options.directory = pool.path();
    options.slots     = 1;

    LoraDiscovery discovery = discover_lora_pool(options, {});
    check(discovery.rejected.empty(), "no fixture was rejected");
    check(discovery.pool.size() == 2, "both fixtures entered the pool");
    if (discovery.pool.size() != 2) { return 1; }
    // Pool order is the sorted file name.
    check(discovery.pool[0].name == "narrow" && discovery.pool[1].name == "wide",
          "the pool is ordered by file name");
    check(discovery.pool[0].fingerprint != discovery.pool[1].fingerprint,
          "the two fixtures have distinct content fingerprints");

    const std::size_t narrow_sites = discovery.pool[0].inventory.site_count();
    const std::size_t wide_sites   = discovery.pool[1].inventory.site_count();
    const std::size_t union_sites  = discovery.profile.inventory.site_count();
    check(narrow_sites < wide_sites, "the fixtures carry different inventories");
    check(union_sites >= wide_sites && union_sites >= narrow_sites,
          "the profile is at least the union of both inventories");
    check(discovery.profile.rank >= discovery.pool[0].rank &&
              discovery.profile.rank >= discovery.pool[1].rank,
          "the bank rank covers every pool adapter");
    std::cout << "  discovery: pool of " << discovery.pool.size() << " at rank "
              << discovery.profile.rank << ", union of " << narrow_sites << " and " << wide_sites
              << " sites is " << union_sites << '\n';

    ninfer::DeviceContext device(0);
    LoraBank bank(std::move(discovery), 1, device);
    check(bank.slots() == 1, "the bank committed one slot");
    check(bank.device_bytes() == bank.profile().slab_bytes, "one slot costs one slab");

    // An unstaged slot must read as an exact no-op, not as whatever the allocator returned.
    const std::vector<unsigned char> empty = read_slot(bank, 0, device);
    check(all_zero(empty.data(), empty.size()), "an unstaged slot is zero");

    bank.stage(0, 0, device);
    const std::vector<unsigned char> narrow_first = read_slot(bank, 0, device);
    verify_staged_slab(bank, 0, narrow_first);

    bank.stage(0, 1, device);
    const std::vector<unsigned char> wide_slab = read_slot(bank, 0, device);
    verify_staged_slab(bank, 1, wide_slab);
    check(narrow_first != wide_slab, "the two adapters occupy the slot differently");

    // Restaging the narrow adapter over the wide one must reproduce its first staging byte for
    // byte. Anything else is residue: the sites the narrow adapter never trained would carry the
    // wide adapter's weights and be applied as if they were its own.
    bank.stage(0, 0, device);
    const std::vector<unsigned char> narrow_again = read_slot(bank, 0, device);
    check(narrow_again == narrow_first, "restaging leaves no residue of the previous occupant");
    std::cout << "  staging: " << bank.stage_count() << " stages in " << bank.stage_seconds()
              << " s (" << (bank.stage_seconds() / static_cast<double>(bank.stage_count())) * 1e3
              << " ms each) over one " << bank.profile().slab_bytes / (1024 * 1024)
              << " MiB slot\n";

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
