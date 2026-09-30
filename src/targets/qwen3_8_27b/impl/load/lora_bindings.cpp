#include "targets/qwen3_8_27b/impl/load/lora_bindings.h"

#include <ninfer/targets/qwen3_8/decision.h>

#include "artifact/typed_binding.h"
#include "targets/qwen3_8_27b/impl/config.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <variant>

namespace ninfer::targets::qwen3_8_27b::detail {
namespace {

using artifact::NumericFormat;

constexpr std::int32_t kHidden          = 5120;
constexpr std::int32_t kQuerySize       = 6144;
constexpr std::int32_t kKeyValueSize    = 1024;
constexpr std::int32_t kAttentionValues = 6144;
constexpr std::int32_t kGdnValues       = 6144;
constexpr std::int32_t kIntermediate    = 17408;
constexpr std::uint64_t kSlabAlignment  = 256;
constexpr std::int32_t kPointerDim      = DecisionConfig::pointer_dim;
static_assert(DecisionConfig::hidden == kHidden);

// The decision objects: the pointer head's four BF16 tensors and its metadata resource. A
// decision adapter carries all five; a generative adapter carries none.
constexpr std::string_view kHeadQueryWeight  = "decision/head/query/weight";
constexpr std::string_view kHeadQueryBias    = "decision/head/query/bias";
constexpr std::string_view kHeadKeyWeight    = "decision/head/key/weight";
constexpr std::string_view kHeadKeyBias      = "decision/head/key/bias";
constexpr std::string_view kDecisionMetadata = "decision/metadata";
constexpr std::array<std::string_view, 5> kDecisionObjects = {
    kHeadQueryWeight, kHeadQueryBias, kHeadKeyWeight, kHeadKeyBias, kDecisionMetadata};

// `decision/metadata` is one JSON object with exactly these members.
constexpr std::string_view kDecisionFormat        = "ninfer-decision-head";
constexpr std::int64_t kDecisionFormatVersion     = 1;
constexpr std::array<std::string_view, 5> kDelimiterKeys = {"state", "question", "option_open",
                                                            "option_close", "decide"};
constexpr std::array<std::string_view, 12> kDecisionMetadataKeys = {
    "format",      "format_version", "pointer_dim", "hidden_size", "logit_scale", "temperature",
    "readout",     "escape",         "delimiters",  "description", "release_date", "base_model"};

bool registered_rank(std::int32_t rank) {
    return rank == 8 || rank == 16 || rank == 32 || rank == 64;
}

bool is_full_attention_layer(std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; }

std::size_t full_attention_index(std::size_t layer) { return (layer - 3) / 4; }

// Layers 3, 7, 11, ... are full attention; every other layer is Gated DeltaNet.
std::size_t gdn_index(std::size_t layer) {
    return layer - (layer >= 3 ? full_attention_index(layer) + 1 : 0);
}

std::string layer_prefix(std::size_t layer) {
    return "text/layers/" + std::to_string(layer) + "/";
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// Object names of one registered site, and the shapes its factors must carry.
struct SiteNames {
    std::string a;
    std::string b;
    std::int32_t columns = 0; // K, the activation width lora_a reads
    std::int32_t rows    = 0; // N, the destination height lora_b writes
};

SiteNames query_gate_a_names(std::size_t layer) {
    return SiteNames{.a       = layer_prefix(layer) + "attention/query_gate/lora_a",
                     .b       = {},
                     .columns = kHidden,
                     .rows    = 0};
}

SiteNames query_site(std::size_t layer) {
    return SiteNames{.a       = layer_prefix(layer) + "attention/query_gate/lora_a",
                     .b       = layer_prefix(layer) + "attention/query/lora_b",
                     .columns = kHidden,
                     .rows    = kQuerySize};
}

SiteNames gate_site(std::size_t layer) {
    return SiteNames{.a       = layer_prefix(layer) + "attention/query_gate/lora_a",
                     .b       = layer_prefix(layer) + "attention/gate/lora_b",
                     .columns = kHidden,
                     .rows    = kQuerySize};
}

SiteNames key_site(std::size_t layer) {
    return SiteNames{.a       = layer_prefix(layer) + "attention/key/lora_a",
                     .b       = layer_prefix(layer) + "attention/key/lora_b",
                     .columns = kHidden,
                     .rows    = kKeyValueSize};
}

SiteNames value_site(std::size_t layer) {
    return SiteNames{.a       = layer_prefix(layer) + "attention/value/lora_a",
                     .b       = layer_prefix(layer) + "attention/value/lora_b",
                     .columns = kHidden,
                     .rows    = kKeyValueSize};
}

SiteNames attention_output_site(std::size_t layer) {
    return SiteNames{.a       = layer_prefix(layer) + "attention/output/lora_a",
                     .b       = layer_prefix(layer) + "attention/output/lora_b",
                     .columns = kAttentionValues,
                     .rows    = kHidden};
}

SiteNames gdn_output_site(std::size_t layer) {
    return SiteNames{.a       = layer_prefix(layer) + "gdn/output/lora_a",
                     .b       = layer_prefix(layer) + "gdn/output/lora_b",
                     .columns = kGdnValues,
                     .rows    = kHidden};
}

SiteNames mlp_down_site(std::size_t layer) {
    return SiteNames{.a       = layer_prefix(layer) + "mlp/down/lora_a",
                     .b       = layer_prefix(layer) + "mlp/down/lora_b",
                     .columns = kIntermediate,
                     .rows    = kHidden};
}

// Reads one factor's descriptor and checks it against the registered shape. Returns its rank,
// or nullopt when the object is absent. Shape errors are hard: an adapter that names a
// registered object must carry it at the registered width.
std::optional<std::int32_t> factor_rank(const artifact::Reader& reader, const std::string& name,
                                        std::int32_t expected_other, bool rank_is_leading) {
    const artifact::ObjectDescriptor* object = reader.find(name);
    if (object == nullptr) { return std::nullopt; }
    const auto* tensor = std::get_if<artifact::TensorDescriptor>(object);
    if (tensor == nullptr || tensor->shape.size() != 2) {
        throw artifact::ArtifactError(name + ": a LoRA factor must be a rank-two tensor");
    }
    if (tensor->format != NumericFormat::BF16 ||
        tensor->layout != artifact::StorageLayout::ContiguousLeV1) {
        throw artifact::ArtifactError(name + ": a LoRA factor must be BF16 contiguous-le-v1");
    }
    const std::uint64_t rank  = rank_is_leading ? tensor->shape[0] : tensor->shape[1];
    const std::uint64_t other = rank_is_leading ? tensor->shape[1] : tensor->shape[0];
    if (other != static_cast<std::uint64_t>(expected_other)) {
        throw artifact::ArtifactError(name + ": expected extent " +
                                      std::to_string(expected_other) + ", found " +
                                      std::to_string(other));
    }
    return static_cast<std::int32_t>(rank);
}

// Reads a private two-factor site. Both factors are required or neither may appear.
bool read_site(const artifact::Reader& reader, const SiteNames& site, std::int32_t& rank,
               std::string_view label) {
    const std::optional<std::int32_t> a = factor_rank(reader, site.a, site.columns, true);
    const std::optional<std::int32_t> b = factor_rank(reader, site.b, site.rows, false);
    if (!a && !b) { return false; }
    if (!a || !b) {
        throw artifact::ArtifactError("LoRA site '" + std::string(label) +
                                      "' is incomplete: both factors are required");
    }
    if (*a != *b) {
        throw artifact::ArtifactError("LoRA site '" + std::string(label) +
                                      "' disagrees on rank between its two factors");
    }
    if (rank != 0 && rank != *a) {
        throw artifact::ArtifactError("LoRA site '" + std::string(label) + "' has rank " +
                                      std::to_string(*a) + ", but the artifact already uses " +
                                      std::to_string(rank) +
                                      "; one artifact carries exactly one rank");
    }
    rank = *a;
    return true;
}

// Reads the shared query/gate group: one down-projection factor and two up-projections.
bool read_query_gate(const artifact::Reader& reader, std::size_t layer, std::int32_t& rank) {
    const SiteNames shared = query_gate_a_names(layer);
    const SiteNames query  = query_site(layer);
    const SiteNames gate   = gate_site(layer);
    const std::optional<std::int32_t> a = factor_rank(reader, shared.a, shared.columns, true);
    const std::optional<std::int32_t> q = factor_rank(reader, query.b, query.rows, false);
    const std::optional<std::int32_t> g = factor_rank(reader, gate.b, gate.rows, false);
    if (!a && !q && !g) { return false; }
    if (!a || !q || !g) {
        throw artifact::ArtifactError(
            "LoRA attention query/gate group in layer " + std::to_string(layer) +
            " is incomplete: the shared lora_a and both lora_b factors are required");
    }
    if (*a != *q || *a != *g) {
        throw artifact::ArtifactError("LoRA attention query/gate group in layer " +
                                      std::to_string(layer) + " disagrees on rank");
    }
    if (rank != 0 && rank != *a) {
        throw artifact::ArtifactError("LoRA attention query/gate group in layer " +
                                      std::to_string(layer) + " has rank " + std::to_string(*a) +
                                      ", but the artifact already uses " + std::to_string(rank) +
                                      "; one artifact carries exactly one rank");
    }
    rank = *a;
    return true;
}

// Walks the whole registered site table against one artifact, producing its inventory and rank.
void read_inventory(const artifact::Reader& reader, LoraInventory& inventory,
                    std::int32_t& rank) {
    std::set<std::string, std::less<>> registered_objects;
    const auto register_site = [&](const SiteNames& site) {
        registered_objects.insert(site.a);
        registered_objects.insert(site.b);
    };
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        if (is_full_attention_layer(layer)) {
            registered_objects.insert(query_gate_a_names(layer).a);
            registered_objects.insert(query_site(layer).b);
            registered_objects.insert(gate_site(layer).b);
            register_site(key_site(layer));
            register_site(value_site(layer));
            register_site(attention_output_site(layer));
            register_site(mlp_down_site(layer));
            LoraFullLayerInventory& full = inventory.full_layers[full_attention_index(layer)];
            full.query_gate              = read_query_gate(reader, layer, rank);
            full.key    = read_site(reader, key_site(layer), rank, "attention/key");
            full.value  = read_site(reader, value_site(layer), rank, "attention/value");
            full.output = read_site(reader, attention_output_site(layer), rank,
                                    "attention/output");
            full.down   = read_site(reader, mlp_down_site(layer), rank, "mlp/down");
        } else {
            register_site(gdn_output_site(layer));
            register_site(mlp_down_site(layer));
            LoraGdnLayerInventory& gdn = inventory.gdn_layers[gdn_index(layer)];
            gdn.output = read_site(reader, gdn_output_site(layer), rank, "gdn/output");
            gdn.down   = read_site(reader, mlp_down_site(layer), rank, "mlp/down");
        }
    }
    for (const std::string_view name : kDecisionObjects) { registered_objects.emplace(name); }
    for (const artifact::ObjectDescriptor& object : reader.objects()) {
        const std::string_view name = artifact::object_name(object);
        if (!registered_objects.contains(name)) {
            throw artifact::ArtifactError("unregistered LoRA object '" + std::string(name) + "'");
        }
    }
}

std::vector<std::uint64_t> head_shape(std::string_view name) {
    if (name == kHeadQueryWeight || name == kHeadKeyWeight) {
        return {static_cast<std::uint64_t>(kPointerDim), static_cast<std::uint64_t>(kHidden)};
    }
    return {static_cast<std::uint64_t>(kPointerDim)};
}

void require_head_tensor(const artifact::Reader& reader, std::string_view name) {
    const auto* tensor = std::get_if<artifact::TensorDescriptor>(reader.find(name));
    if (tensor == nullptr) {
        throw artifact::ArtifactError(std::string(name) + ": a decision head object is a tensor");
    }
    if (tensor->format != NumericFormat::BF16 ||
        tensor->layout != artifact::StorageLayout::ContiguousLeV1) {
        throw artifact::ArtifactError(std::string(name) +
                                      ": a decision head tensor must be BF16 contiguous-le-v1");
    }
    if (tensor->shape != head_shape(name)) {
        throw artifact::ArtifactError(std::string(name) + ": decision head shape is not [" +
                                      std::to_string(kPointerDim) +
                                      (name.ends_with("/weight") ? ", " + std::to_string(kHidden)
                                                                 : std::string()) +
                                      "]");
    }
}

// YYYY-MM-DD with a month and day in range. The converter checks the calendar.
bool valid_release_date(std::string_view date) {
    if (date.size() != 10 || date[4] != '-' || date[7] != '-') { return false; }
    for (const std::size_t index : {0U, 1U, 2U, 3U, 5U, 6U, 8U, 9U}) {
        if (date[index] < '0' || date[index] > '9') { return false; }
    }
    const int month = (date[5] - '0') * 10 + (date[6] - '0');
    const int day   = (date[8] - '0') * 10 + (date[9] - '0');
    return month >= 1 && month <= 12 && day >= 1 && day <= 31;
}

const nlohmann::json& metadata_member(const nlohmann::json& metadata, std::string_view key) {
    const auto found = metadata.find(key);
    if (found == metadata.end()) {
        throw artifact::ArtifactError("decision/metadata lacks '" + std::string(key) + "'");
    }
    return *found;
}

std::string metadata_string(const nlohmann::json& metadata, std::string_view key) {
    const nlohmann::json& value = metadata_member(metadata, key);
    if (!value.is_string()) {
        throw artifact::ArtifactError("decision/metadata '" + std::string(key) +
                                      "' is not a string");
    }
    return value.get<std::string>();
}

std::int64_t metadata_integer(const nlohmann::json& metadata, std::string_view key) {
    const nlohmann::json& value = metadata_member(metadata, key);
    if (!value.is_number_integer()) {
        throw artifact::ArtifactError("decision/metadata '" + std::string(key) +
                                      "' is not an integer");
    }
    return value.get<std::int64_t>();
}

double metadata_number(const nlohmann::json& metadata, std::string_view key) {
    const nlohmann::json& value = metadata_member(metadata, key);
    if (!value.is_number()) {
        throw artifact::ArtifactError("decision/metadata '" + std::string(key) +
                                      "' is not a number");
    }
    return value.get<double>();
}

// Validates the registered metadata contract: the head geometry and scale the family executes,
// kev's readout, escape rule and delimiters, and the host metadata the product layer publishes.
DecisionAdapterMetadata parse_decision_metadata(std::span<const std::byte> bytes) {
    const std::string_view text(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    nlohmann::json metadata;
    try {
        metadata = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw artifact::ArtifactError(std::string("decision/metadata is not JSON: ") +
                                      error.what());
    }
    if (!metadata.is_object()) {
        throw artifact::ArtifactError("decision/metadata is not a JSON object");
    }
    for (const auto& [key, value] : metadata.items()) {
        (void)value;
        if (std::find(kDecisionMetadataKeys.begin(), kDecisionMetadataKeys.end(), key) ==
            kDecisionMetadataKeys.end()) {
            throw artifact::ArtifactError("decision/metadata has unregistered member '" + key +
                                          "'");
        }
    }
    if (metadata_string(metadata, "format") != kDecisionFormat ||
        metadata_integer(metadata, "format_version") != kDecisionFormatVersion) {
        throw artifact::ArtifactError("decision/metadata is not " + std::string(kDecisionFormat) +
                                      " version " + std::to_string(kDecisionFormatVersion));
    }
    if (metadata_integer(metadata, "pointer_dim") != kPointerDim ||
        metadata_integer(metadata, "hidden_size") != kHidden) {
        throw artifact::ArtifactError("decision/metadata head geometry is not [" +
                                      std::to_string(kPointerDim) + ", " +
                                      std::to_string(kHidden) + "]");
    }
    // The pointer logit is (k . q) / sqrt(pointer_dim) / T; the scale is fixed by the geometry.
    if (metadata_number(metadata, "logit_scale") != 1.0 / std::sqrt(double{kPointerDim})) {
        throw artifact::ArtifactError("decision/metadata logit_scale is not 1/sqrt(pointer_dim)");
    }
    if (metadata_string(metadata, "readout") != "final_norm" ||
        metadata_string(metadata, "escape") != "kev-v1") {
        throw artifact::ArtifactError(
            "decision/metadata readout/escape is not the registered final_norm/kev-v1");
    }
    const nlohmann::json& delimiters = metadata_member(metadata, "delimiters");
    if (!delimiters.is_object() || delimiters.size() != kDelimiterKeys.size()) {
        throw artifact::ArtifactError("decision/metadata delimiters is not the five-member object");
    }
    for (std::size_t index = 0; index < kDelimiterKeys.size(); ++index) {
        const auto found = delimiters.find(kDelimiterKeys[index]);
        if (found == delimiters.end() || !found->is_string() ||
            found->get<std::string>() != qwen3_8::kDecisionDelimiterNames[index]) {
            throw artifact::ArtifactError("decision/metadata delimiter '" +
                                          std::string(kDelimiterKeys[index]) + "' is not " +
                                          std::string(qwen3_8::kDecisionDelimiterNames[index]));
        }
    }

    DecisionAdapterMetadata parsed;
    const double temperature = metadata_number(metadata, "temperature");
    parsed.temperature       = static_cast<float>(temperature);
    if (!std::isfinite(parsed.temperature) || parsed.temperature <= 0.0F ||
        static_cast<double>(parsed.temperature) != temperature) {
        throw artifact::ArtifactError(
            "decision/metadata temperature is not a positive finite FP32 value");
    }
    parsed.description  = metadata_string(metadata, "description");
    parsed.release_date = metadata_string(metadata, "release_date");
    if (!valid_release_date(parsed.release_date)) {
        throw artifact::ArtifactError("decision/metadata release_date is not YYYY-MM-DD");
    }
    (void)metadata_string(metadata, "base_model");
    return parsed;
}

// Reads the decision objects. Returns the adapter's kind; a partial set is malformed.
LoraAdapterKind read_decision(const artifact::Reader& reader, DecisionAdapterMetadata& metadata) {
    std::size_t present = 0;
    for (const std::string_view name : kDecisionObjects) {
        present += reader.find(name) != nullptr ? 1U : 0U;
    }
    if (present == 0) { return LoraAdapterKind::Generative; }
    if (present != kDecisionObjects.size()) {
        throw artifact::ArtifactError(
            "decision adapter is incomplete: the four decision/head tensors and "
            "decision/metadata are required together");
    }
    for (const std::string_view name : kDecisionObjects) {
        if (name != kDecisionMetadata) { require_head_tensor(reader, name); }
    }
    if (std::get_if<artifact::ResourceDescriptor>(reader.find(kDecisionMetadata)) == nullptr) {
        throw artifact::ArtifactError("decision/metadata is not a resource");
    }
    metadata = parse_decision_metadata(reader.payload(kDecisionMetadata).data);
    return LoraAdapterKind::Decision;
}

// Assigns slab offsets for the profile's inventory. A factor named by two sites - the shared
// query/gate down-projection - is placed exactly once.
class SlabBuilder {
public:
    std::uint64_t place(const std::string& name, std::int32_t rows, std::int32_t columns) {
        const auto existing = placed_.find(name);
        if (existing != placed_.end()) { return existing->second; }
        const auto bytes =
            static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(columns) * 2U;
        const std::uint64_t offset = cursor_;
        cursor_                    = align_up(cursor_ + bytes, kSlabAlignment);
        placed_.emplace(name, offset);
        return offset;
    }

    [[nodiscard]] std::uint64_t bytes() const noexcept { return cursor_; }

private:
    std::map<std::string, std::uint64_t> placed_;
    std::uint64_t cursor_ = 0;
};

LoraSitePlan place_site(SlabBuilder& slab, const SiteNames& site, std::int32_t rank) {
    return LoraSitePlan{
        .a_offset = slab.place(site.a, rank, site.columns),
        .b_offset = slab.place(site.b, site.rows, rank),
        .rows     = site.rows,
        .columns  = site.columns,
        .present  = true,
    };
}

qwen3_8::LoraSiteWeights bind_site_view(const LoraSitePlan& plan, unsigned char* slab_base,
                                        std::uint64_t slab_bytes, std::int32_t rank) {
    if (!plan.present) { return {}; }
    qwen3_8::LoraSiteWeights view;
    view.a = Tensor(slab_base + plan.a_offset, DType::BF16, {plan.columns, rank});
    view.b = Tensor(slab_base + plan.b_offset, DType::BF16, {rank, plan.rows});
    view.a_adapter_stride = slab_bytes;
    view.b_adapter_stride = slab_bytes;
    return view;
}

// Strips the conventional adapter suffixes so a pool name is the file's stem: a file named
// `math7.lora.ninfer` is selected as `math7`.
std::string pool_name(const std::filesystem::path& path) {
    std::string name = path.filename().string();
    static constexpr std::string_view kNinfer = ".ninfer";
    static constexpr std::string_view kLora   = ".lora";
    if (name.size() > kNinfer.size() && name.compare(name.size() - kNinfer.size(),
                                                     kNinfer.size(), kNinfer) == 0) {
        name.resize(name.size() - kNinfer.size());
    }
    if (name.size() > kLora.size() &&
        name.compare(name.size() - kLora.size(), kLora.size(), kLora) == 0) {
        name.resize(name.size() - kLora.size());
    }
    return name;
}

// Copies one factor into the pinned slab, zero-padding the rank. `stored_rank` rows or columns
// carry the artifact's values and the remainder is zero, which contributes nothing to the
// product and so reproduces the adapter's trained delta exactly at the bank rank.
void stage_a_factor(unsigned char* destination, const std::byte* source, std::int32_t stored_rank,
                    std::int32_t bank_rank, std::int32_t columns) {
    // A is row-major [rank, columns], so the padding is a contiguous tail.
    const std::size_t stored = static_cast<std::size_t>(stored_rank) *
                               static_cast<std::size_t>(columns) * 2U;
    const std::size_t total =
        static_cast<std::size_t>(bank_rank) * static_cast<std::size_t>(columns) * 2U;
    std::memcpy(destination, source, stored);
    if (total > stored) { std::memset(destination + stored, 0, total - stored); }
}

void stage_b_factor(unsigned char* destination, const std::byte* source, std::int32_t stored_rank,
                    std::int32_t bank_rank, std::int32_t rows) {
    // B is row-major [rows, rank], so the padding is a tail inside every row.
    const std::size_t stored_pitch = static_cast<std::size_t>(stored_rank) * 2U;
    const std::size_t bank_pitch   = static_cast<std::size_t>(bank_rank) * 2U;
    if (stored_pitch == bank_pitch) {
        std::memcpy(destination, source, bank_pitch * static_cast<std::size_t>(rows));
        return;
    }
    for (std::int32_t row = 0; row < rows; ++row) {
        unsigned char* out = destination + static_cast<std::size_t>(row) * bank_pitch;
        std::memcpy(out, source + static_cast<std::size_t>(row) * stored_pitch, stored_pitch);
        std::memset(out + stored_pitch, 0, bank_pitch - stored_pitch);
    }
}

} // namespace

bool LoraInventory::any() const noexcept {
    for (const LoraFullLayerInventory& full : full_layers) {
        if (full.query_gate || full.key || full.value || full.output || full.down) { return true; }
    }
    for (const LoraGdnLayerInventory& gdn : gdn_layers) {
        if (gdn.output || gdn.down) { return true; }
    }
    return false;
}

void LoraInventory::merge(const LoraInventory& other) noexcept {
    for (std::size_t index = 0; index < full_layers.size(); ++index) {
        LoraFullLayerInventory& into      = full_layers[index];
        const LoraFullLayerInventory& add = other.full_layers[index];
        into.query_gate                   = into.query_gate || add.query_gate;
        into.key                          = into.key || add.key;
        into.value                        = into.value || add.value;
        into.output                       = into.output || add.output;
        into.down                         = into.down || add.down;
    }
    for (std::size_t index = 0; index < gdn_layers.size(); ++index) {
        LoraGdnLayerInventory& into      = gdn_layers[index];
        const LoraGdnLayerInventory& add = other.gdn_layers[index];
        into.output                      = into.output || add.output;
        into.down                        = into.down || add.down;
    }
}

std::size_t LoraInventory::site_count() const noexcept {
    std::size_t count = 0;
    for (const LoraFullLayerInventory& full : full_layers) {
        count += (full.query_gate ? 2U : 0U) + (full.key ? 1U : 0U) + (full.value ? 1U : 0U) +
                 (full.output ? 1U : 0U) + (full.down ? 1U : 0U);
    }
    for (const LoraGdnLayerInventory& gdn : gdn_layers) {
        count += (gdn.output ? 1U : 0U) + (gdn.down ? 1U : 0U);
    }
    return count;
}

namespace {

LoraBankProfile build_bank_profile(const LoraInventory& inventory, std::int32_t rank,
                                   bool decision_head) {
    LoraBankProfile profile;
    profile.rank      = rank;
    profile.inventory = inventory;

    SlabBuilder slab;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        if (is_full_attention_layer(layer)) {
            const std::size_t index            = full_attention_index(layer);
            const LoraFullLayerInventory& have = inventory.full_layers[index];
            LoraFullLayerPlan& plan            = profile.full_layers[index];
            if (have.query_gate) {
                plan.query = place_site(slab, query_site(layer), rank);
                plan.gate  = place_site(slab, gate_site(layer), rank);
            }
            if (have.key) { plan.key = place_site(slab, key_site(layer), rank); }
            if (have.value) { plan.value = place_site(slab, value_site(layer), rank); }
            if (have.output) {
                plan.output = place_site(slab, attention_output_site(layer), rank);
            }
            if (have.down) { plan.down = place_site(slab, mlp_down_site(layer), rank); }
        } else {
            const std::size_t index           = gdn_index(layer);
            const LoraGdnLayerInventory& have = inventory.gdn_layers[index];
            LoraGdnLayerPlan& plan            = profile.gdn_layers[index];
            if (have.output) { plan.output = place_site(slab, gdn_output_site(layer), rank); }
            if (have.down) { plan.down = place_site(slab, mlp_down_site(layer), rank); }
        }
    }
    if (decision_head) {
        profile.decision_head = DecisionHeadPlan{
            .query_weight = slab.place(std::string(kHeadQueryWeight), kPointerDim, kHidden),
            .query_bias   = slab.place(std::string(kHeadQueryBias), 1, kPointerDim),
            .key_weight   = slab.place(std::string(kHeadKeyWeight), kPointerDim, kHidden),
            .key_bias     = slab.place(std::string(kHeadKeyBias), 1, kPointerDim),
        };
    }
    profile.slab_bytes = align_up(slab.bytes(), kSlabAlignment);
    return profile;
}

} // namespace

LoraDiscovery discover_lora_pool(const LoraOptions& options,
                                 const artifact::FingerprintCacheOptions& fingerprint_cache) {
    LoraDiscovery discovery;
    discovery.fingerprint_cache = fingerprint_cache;
    if (options.directory.empty()) { return discovery; }

    std::error_code error;
    if (!std::filesystem::is_directory(options.directory, error)) {
        throw std::invalid_argument("LoRA adapter directory '" + options.directory.string() +
                                    "' is not a directory");
    }
    if (options.rank_ceiling != 0 && !registered_rank(options.rank_ceiling)) {
        throw std::invalid_argument("LoRA rank ceiling " + std::to_string(options.rank_ceiling) +
                                    " is not registered; supported ranks are 8, 16, 32, 64");
    }

    std::vector<std::filesystem::path> candidates;
    for (const std::filesystem::directory_entry& entry :
         std::filesystem::directory_iterator(options.directory)) {
        if (!entry.is_regular_file()) { continue; }
        if (entry.path().extension() != ".ninfer") { continue; }
        candidates.push_back(entry.path());
    }
    // Pool order is the sorted file name, so a pool index is reproducible across processes even
    // though nothing persisted depends on it.
    std::sort(candidates.begin(), candidates.end());

    std::set<std::string> seen;
    std::int32_t bank_rank = 0;
    bool any_decision      = false;
    LoraInventory united;
    for (const std::filesystem::path& path : candidates) {
        const std::string name = pool_name(path);
        try {
            if (name.empty()) { throw artifact::ArtifactError("empty adapter name"); }
            artifact::Reader reader(path, fingerprint_cache, artifact::FingerprintProgress{});
            const artifact::ArtifactIdentity& identity = reader.identity();
            if (identity.model_id != std::string(Package::model_id) ||
                identity.weights_id != std::string(kLoraWeightsId)) {
                throw artifact::ArtifactError("identity is '" + identity.model_id + "/" +
                                              identity.weights_id + "', not '" +
                                              std::string(Package::model_id) + "/" +
                                              std::string(kLoraWeightsId) + "'");
            }
            LoraPoolEntry entry;
            entry.name = name;
            entry.path = path;
            read_inventory(reader, entry.inventory, entry.rank);
            if (!entry.inventory.any()) {
                throw artifact::ArtifactError("carries no registered site, so it corrects nothing");
            }
            entry.kind = read_decision(reader, entry.decision);
            if (!registered_rank(entry.rank)) {
                throw artifact::ArtifactError("rank " + std::to_string(entry.rank) +
                                              " is not registered; supported ranks are 8, 16, 32,"
                                              " 64");
            }
            if (options.rank_ceiling != 0 && entry.rank > options.rank_ceiling) {
                throw artifact::ArtifactError("rank " + std::to_string(entry.rank) +
                                              " exceeds the configured ceiling " +
                                              std::to_string(options.rank_ceiling));
            }
            if (!seen.insert(entry.name).second) {
                throw artifact::ArtifactError("adapter name '" + entry.name +
                                              "' is already in the pool");
            }
            entry.file_bytes  = reader.file_bytes();
            entry.fingerprint = reader.content_fingerprint();
            bank_rank         = std::max(bank_rank, entry.rank);
            any_decision      = any_decision || entry.kind == LoraAdapterKind::Decision;
            united.merge(entry.inventory);
            discovery.pool.push_back(std::move(entry));
        } catch (const std::exception& failure) {
            discovery.rejected.push_back(path.filename().string() + ": " + failure.what());
        }
    }

    if (!discovery.rejected.empty()) {
        std::string detail;
        for (const std::string& reason : discovery.rejected) {
            detail += detail.empty() ? "" : "; ";
            detail += reason;
        }
        throw std::invalid_argument("LoRA adapter directory '" + options.directory.string() +
                                    "' contains an unusable adapter: " + detail);
    }
    if (discovery.pool.empty()) {
        throw std::invalid_argument("LoRA adapter directory '" + options.directory.string() +
                                    "' holds no '.ninfer' adapter");
    }

    discovery.profile = build_bank_profile(united, bank_rank, any_decision);
    return discovery;
}

LoraBank::LoraBank(LoraDiscovery discovery, std::uint32_t slots, DeviceContext& device)
    : pool_(std::move(discovery.pool)), profile_(std::move(discovery.profile)),
      fingerprint_cache_(std::move(discovery.fingerprint_cache)) {
    if (pool_.empty()) { throw std::invalid_argument("a LoRA bank needs at least one adapter"); }
    if (slots == 0) { throw std::invalid_argument("a LoRA bank needs at least one slot"); }
    // Committing more slots than the pool can fill would reserve device memory the engine can
    // never use, and the KV resolver would silently shrink the context to pay for it.
    slots_ = static_cast<std::uint32_t>(std::min<std::size_t>(slots, pool_.size()));

    for (const LoraPoolEntry& entry : pool_) { pool_file_bytes_ += entry.file_bytes; }

    const std::uint64_t slab = profile_.slab_bytes;
    device_bytes_            = slab * slots_;
    arena_ = std::make_unique<DeviceArena>(static_cast<std::size_t>(device_bytes_) +
                                           kSlabAlignment);
    const DeviceSpan storage =
        arena_->alloc_bytes(static_cast<std::size_t>(device_bytes_), kSlabAlignment);
    base_ = static_cast<unsigned char*>(storage.data);
    // A slot the engine has not staged yet must read as an exact no-op rather than as whatever
    // the allocator handed back, because the captured graph reads every slot's geometry
    // unconditionally and only the per-row index decides which one contributes.
    CUDA_CHECK(cudaMemsetAsync(base_, 0, static_cast<std::size_t>(device_bytes_), device.stream));

    staging_.emplace(static_cast<std::size_t>(slab));

    view_.slots        = slots_;
    view_.pool_size    = static_cast<std::uint32_t>(pool_.size());
    view_.rank         = profile_.rank;
    view_.device_bytes = device_bytes_;
    view_.pool         = this;
    view_.fingerprints.reserve(pool_.size());
    view_.kinds.reserve(pool_.size());
    view_.decision_temperatures.reserve(pool_.size());
    for (const LoraPoolEntry& entry : pool_) {
        view_.fingerprints.push_back(entry.fingerprint);
        view_.kinds.push_back(entry.kind);
        view_.decision_temperatures.push_back(
            entry.kind == LoraAdapterKind::Decision ? entry.decision.temperature : 1.0F);
    }
    if (profile_.decision_head) {
        // Weights are row-major [pointer_dim, hidden]; shapes list the contiguous extent first.
        const DecisionHeadPlan& head          = *profile_.decision_head;
        qwen3_8::DecisionHeadWeights& weights = view_.decision_head;
        weights.query_weight =
            Tensor(base_ + head.query_weight, DType::BF16, {kHidden, kPointerDim});
        weights.query_bias = Tensor(base_ + head.query_bias, DType::BF16, {kPointerDim});
        weights.key_weight = Tensor(base_ + head.key_weight, DType::BF16, {kHidden, kPointerDim});
        weights.key_bias   = Tensor(base_ + head.key_bias, DType::BF16, {kPointerDim});
        weights.slot_stride = static_cast<std::size_t>(slab);
    }
    for (std::size_t index = 0; index < kFullAttentionLayers; ++index) {
        const LoraFullLayerPlan& source     = profile_.full_layers[index];
        qwen3_8::LoraFullLayerWeights& view = view_.full_layers[index];
        view.query  = bind_site_view(source.query, base_, slab, profile_.rank);
        view.gate   = bind_site_view(source.gate, base_, slab, profile_.rank);
        view.key    = bind_site_view(source.key, base_, slab, profile_.rank);
        view.value  = bind_site_view(source.value, base_, slab, profile_.rank);
        view.output = bind_site_view(source.output, base_, slab, profile_.rank);
        view.down   = bind_site_view(source.down, base_, slab, profile_.rank);
    }
    for (std::size_t index = 0; index < kGdnLayers; ++index) {
        const LoraGdnLayerPlan& source     = profile_.gdn_layers[index];
        qwen3_8::LoraGdnLayerWeights& view = view_.gdn_layers[index];
        view.output = bind_site_view(source.output, base_, slab, profile_.rank);
        view.down   = bind_site_view(source.down, base_, slab, profile_.rank);
    }
    device.synchronize();
}

// Fills the pinned slab with one adapter at the bank profile. Every profile byte is written
// exactly once - either from the artifact or as zero - so no residue of the slot's previous
// occupant can survive into the upload.
void LoraBank::assemble(const LoraPoolEntry& entry, const artifact::Reader& reader) {
    auto* slab = static_cast<unsigned char*>(staging_->data());
    std::memset(slab, 0, staging_->size());

    const auto copy_site = [&](const LoraSitePlan& plan, const SiteNames& names, bool have) {
        if (!plan.present || !have) { return; }
        const artifact::PayloadSpan a = reader.payload(names.a);
        const artifact::PayloadSpan b = reader.payload(names.b);
        stage_a_factor(slab + plan.a_offset, a.data.data(), entry.rank, profile_.rank,
                       plan.columns);
        stage_b_factor(slab + plan.b_offset, b.data.data(), entry.rank, profile_.rank, plan.rows);
    };

    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        if (is_full_attention_layer(layer)) {
            const std::size_t index            = full_attention_index(layer);
            const LoraFullLayerPlan& plan      = profile_.full_layers[index];
            const LoraFullLayerInventory& have = entry.inventory.full_layers[index];
            // Query and gate share one down-projection factor; staging it twice is harmless and
            // keeps the walk uniform.
            copy_site(plan.query, query_site(layer), have.query_gate);
            copy_site(plan.gate, gate_site(layer), have.query_gate);
            copy_site(plan.key, key_site(layer), have.key);
            copy_site(plan.value, value_site(layer), have.value);
            copy_site(plan.output, attention_output_site(layer), have.output);
            copy_site(plan.down, mlp_down_site(layer), have.down);
        } else {
            const std::size_t index           = gdn_index(layer);
            const LoraGdnLayerPlan& plan      = profile_.gdn_layers[index];
            const LoraGdnLayerInventory& have = entry.inventory.gdn_layers[index];
            copy_site(plan.output, gdn_output_site(layer), have.output);
            copy_site(plan.down, mlp_down_site(layer), have.down);
        }
    }

    // The head is copied verbatim: its geometry is fixed, so it needs no padding.
    if (profile_.decision_head && entry.kind == LoraAdapterKind::Decision) {
        const DecisionHeadPlan& head = *profile_.decision_head;
        const auto copy_head         = [&](std::string_view name, std::uint64_t offset) {
            const artifact::PayloadSpan payload = reader.payload(name);
            std::memcpy(slab + offset, payload.data.data(), payload.data.size());
        };
        copy_head(kHeadQueryWeight, head.query_weight);
        copy_head(kHeadQueryBias, head.query_bias);
        copy_head(kHeadKeyWeight, head.key_weight);
        copy_head(kHeadKeyBias, head.key_bias);
    }
}

void LoraBank::prepare(std::size_t index) {
    if (index >= pool_.size()) { throw std::invalid_argument("LoRA adapter is outside the pool"); }
    prepared_index_.reset();
    prepare_started_ = std::chrono::steady_clock::now();

    // Reopen through the existing fingerprint cache and verify content identity before copying.
    // A long-lived mmap is not an immutable snapshot: same-inode writes can alter its pages, which
    // would make executed bytes disagree with the fingerprint used by continuation state.
    const LoraPoolEntry& entry = pool_[index];
    artifact::Reader reader(entry.path, fingerprint_cache_, artifact::FingerprintProgress{});
    if (reader.content_fingerprint() != entry.fingerprint) {
        throw std::runtime_error("LoRA adapter '" + entry.name +
                                 "' changed on disk after discovery");
    }
    assemble(entry, reader);
    prepared_index_ = index;
}

void LoraBank::commit(std::uint32_t slot, DeviceContext& device) {
    if (slot >= slots_) { throw std::invalid_argument("LoRA slot is outside the bank"); }
    if (!prepared_index_) { throw std::logic_error("no LoRA adapter is prepared for staging"); }
    CUDA_CHECK(cudaMemcpyAsync(base_ + static_cast<std::uint64_t>(slot) * profile_.slab_bytes,
                               staging_->data(), static_cast<std::size_t>(profile_.slab_bytes),
                               cudaMemcpyHostToDevice, device.stream));
    device.synchronize();

    prepared_index_.reset();
    ++stage_count_;
    stage_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                     prepare_started_).count();
}

void LoraBank::stage(std::uint32_t slot, std::size_t index, DeviceContext& device) {
    prepare(index);
    commit(slot, device);
}

} // namespace ninfer::targets::qwen3_8_27b::detail
