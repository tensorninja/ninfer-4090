#include "targets/qwen3_8/impl/runtime/instance.h"
#include "targets/qwen3_8/impl/runtime/program.h"

#include "targets/qwen3_8/impl/runtime/schedule.h"
#include "targets/qwen3_8/impl/runtime/continuation_image.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"
#include "runtime/engine/l1_reuse_accounting.h"
#include "runtime/generation/row_commit.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <set>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::targets::qwen3_8::detail::NINFER_QWEN38_RUNTIME_NS {
namespace {

using Clock = std::chrono::steady_clock;
namespace image = qwen3_8::detail::continuation;

// Routes slot staging to the package through a dependent branch, so a target without a
// registered site table never names the entry point. Residency is family policy; the bytes,
// the pool and the bank profile are the package's.
template <class V>
void prepare_lora_slot(const LoadedModelData& model, std::size_t index) {
    if constexpr (V::supports_lora) {
        V::lora_prepare_slot(model, index);
    } else {
        (void)model;
        (void)index;
        throw std::logic_error("this target registers no LoRA site table");
    }
}

template <class V>
void commit_lora_slot(const LoadedModelData& model, std::uint32_t slot, DeviceContext& device) {
    if constexpr (V::supports_lora) {
        V::lora_commit_slot(model, slot, device);
    } else {
        (void)model;
        (void)slot;
        (void)device;
        throw std::logic_error("this target registers no LoRA site table");
    }
}

void write_paged_layout(image::Writer& out, const qwen3_8::PagedKVCacheLayout& layout) {
    out.u32(layout.layers);
    out.u32(layout.max_context);
    out.i32(layout.kv_heads);
    out.i32(layout.head_dim);
    out.u8(static_cast<std::uint8_t>(layout.dtype));
    out.i32(layout.quant_group);
    out.u8(layout.packed_v);
    out.u8(layout.rotate_k);
    out.u8(layout.rotate_v);
    out.u8(layout.packed_k);
    out.u8(layout.e8_root);
    out.u8(static_cast<std::uint8_t>(layout.pool.spec.plane_order));
    out.u32(layout.pool.spec.logical_page_capacity);
    out.u64(layout.pool.spec.planes.size());
    for (const auto& plane : layout.pool.spec.planes) { image::write_plane_spec(out, plane); }
}

cache::Bytes make_compatibility_key(const SequencePlanImpl& plan, std::string_view model_id,
                                     std::string_view weights_id,
                                     std::span<const std::uint8_t> artifact_fingerprint) {
    if (model_id.empty() || weights_id.empty() || artifact_fingerprint.size() != 32) {
        throw std::invalid_argument("continuation compatibility requires artifact identity");
    }
    image::Writer out;
    image::write_header(out, "qwen-continuation-abi");
    out.string(model_id);
    out.string(weights_id);
    out.raw(artifact_fingerprint);
    out.u32(static_cast<std::uint32_t>(plan.weights_profile));
    out.u32(plan.capacity);
    out.u32(plan.draft_window);
    out.u8(static_cast<std::uint8_t>(plan.speculative_backend));
    out.u8(static_cast<std::uint8_t>(plan.proposal_head));
    out.u8(plan.features.vision);
    write_paged_layout(out, plan.persistent.decoder.text_kv);
    out.u8(plan.persistent.decoder.mtp_kv.has_value());
    if (plan.persistent.decoder.mtp_kv) { write_paged_layout(out, *plan.persistent.decoder.mtp_kv); }
    const auto& linear = plan.persistent.decoder.linear_attention.spec;
    out.u32(linear.layers);
    out.i32(linear.conv_channels);
    out.i32(linear.conv_width);
    out.i32(linear.value_heads);
    out.i32(linear.value_head_dim);
    out.i32(linear.key_head_dim);
    out.u8(static_cast<std::uint8_t>(linear.conv_dtype));
    out.u8(plan.persistent.dflash.has_value());
    if (plan.persistent.dflash) {
        const auto write_cyclic = [&out](const CyclicKVCacheLayout& layout) {
            out.u32(static_cast<std::uint32_t>(layout.k.size()));
            out.u32(layout.capacity);
            out.u32(layout.padded_capacity);
            out.i32(layout.num_kv_heads);
            out.i32(layout.head_dim);
        };
        write_cyclic(plan.persistent.dflash->local);
        write_cyclic(plan.persistent.dflash->turn_checkpoint_local);
        write_paged_layout(out, plan.persistent.dflash->full);
    }
    out.u64(plan.persistent.tail_hidden.region.bytes);
    out.u64(plan.persistent.turn_checkpoint_hidden.region.bytes);
    return std::move(out).finish();
}

cache::Bytes copy_tensor_to_host(const Tensor& tensor, PinnedTransferBuffer& transfer,
                                 cudaStream_t stream) {
    if (!tensor.is_contiguous()) { throw std::logic_error("continuation tensor is not contiguous"); }
    cache::Bytes bytes(tensor.bytes());
    transfer.copy_device_to_host(bytes, tensor.data, bytes.size(), stream);
    return bytes;
}

void copy_tensor_to_device(const Tensor& tensor, const cache::Bytes& bytes,
                           PinnedTransferBuffer& transfer, cudaStream_t stream) {
    if (!tensor.is_contiguous() || bytes.size() != tensor.bytes()) {
        throw std::invalid_argument("continuation tensor does not match its destination");
    }
    transfer.copy_host_to_device(tensor.data, bytes, bytes.size(), stream);
}

// Session identity over a ledger prefix: FNV-1a 64 of the token bytes, rendered as 16 hex
// chars. The full-ledger form is the session_digest clients see; a ring checkpoint's digest is
// the same hash over the prefix its frontier covers (session_snapshot_impl.h calls through
// here for the full ledger).
std::string ledger_prefix_digest(std::span<const TokenId> ledger) {
    std::uint64_t hash = 1469598103934665603ULL;
    const auto* bytes  = reinterpret_cast<const unsigned char*>(ledger.data());
    const std::size_t count = ledger.size() * sizeof(TokenId);
    for (std::size_t index = 0; index < count; ++index) {
        hash = (hash ^ bytes[index]) * 1099511628211ULL;
    }
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(hash));
    return text;
}

std::int32_t checked_i32(std::uint32_t value, const char* label) {
    if (value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

std::array<std::int32_t, 3> prompt_rope_position(const PreparedPromptData& prompt,
                                                 std::uint32_t token) {
    const std::size_t tokens = prompt.token_ids.size();
    if (token >= tokens || prompt.positions.size() != 3 * tokens) {
        throw std::invalid_argument("MTP bridge position is outside prepared prompt metadata");
    }
    return {prompt.positions[token], prompt.positions[tokens + token],
            prompt.positions[2 * tokens + token]};
}

schedule::MtpGqaEnvelopes mtp_gqa_envelopes(std::uint32_t max_frontier, std::uint32_t k,
                                            std::uint32_t capacity) {
    const auto visible = [capacity](std::uint64_t value) {
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(capacity, value));
    };
    schedule::MtpGqaEnvelopes out;
    out.target_verify = {1, visible(static_cast<std::uint64_t>(max_frontier) + k + 1ULL)};
    out.batch         = out.target_verify;
    for (std::uint32_t step = 0; step + 1 < k; ++step) {
        out.ar[step] = {1, visible(static_cast<std::uint64_t>(max_frontier) + k + step + 2ULL)};
    }
    return out;
}

schedule::DFlashEnvelopes dflash_envelopes(std::uint32_t min_frontier, std::uint32_t max_frontier,
                                           std::uint32_t k) {
    (void)min_frontier;
    return schedule::DFlashEnvelopes{
        .local  = {0, max_frontier},
        .full   = {0, max_frontier},
        .append = {0, k + 1},
    };
}

DecodeGraphProfile& select_graph_profile(DecodeGraphFamily& family, std::uint32_t batch_size,
                                         std::uint32_t frontier, const char* label) {
    const auto it = std::find_if(
        family.profiles.begin(), family.profiles.end(), [&](const DecodeGraphProfile& profile) {
            return profile.batch_size == batch_size && profile.min_execution_frontier <= frontier &&
                   frontier <= profile.max_execution_frontier;
        });
    if (it == family.profiles.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage is incomplete");
    }
    return *it;
}

void validate_graph_profiles(const std::vector<GraphExecutionProfile>& profiles,
                             std::uint32_t max_frontier, const char* label) {
    if (profiles.empty() || profiles.front().min != 0 || profiles.back().max != max_frontier) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage has invalid endpoints");
    }
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        if (profiles[i].min > profiles[i].max ||
            (i != 0 && profiles[i].min != profiles[i - 1].max + 1)) {
            throw std::logic_error(std::string(label) + " CUDA Graph coverage has a gap");
        }
    }
}

DecodeGraphTopology& select_graph_topology(DecodeGraphFamily& family, std::uint32_t topology_class,
                                           const char* label) {
    const auto it = std::find_if(family.topologies.begin(), family.topologies.end(),
                                 [topology_class](const DecodeGraphTopology& topology) {
                                     return topology.topology_class == topology_class;
                                 });
    if (it == family.topologies.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph topology is unavailable");
    }
    return *it;
}

DecodeGraphExecutable& install_graph_profile(DecodeGraphFamily& family, DecodeGraphProfile& profile,
                                             const char* label) {
    DecodeGraphTopology& topology   = select_graph_topology(family, profile.topology_class, label);
    const std::size_t profile_index = static_cast<std::size_t>(&profile - family.profiles.data());
    if (topology.installed_profile != profile_index) {
        topology.executable.update(profile.definition);
        topology.installed_profile = profile_index;
    }
    return topology.executable;
}

template <class Prepare>
void instantiate_graph_family(DecodeGraphFamily& family, const char* label, DeviceContext& device,
                              Prepare&& prepare) {
    if (family.profiles.empty()) {
        throw std::logic_error(std::string(label) + " CUDA Graph family has no profiles");
    }

    for (std::size_t i = 0; i < family.profiles.size(); ++i) {
        DecodeGraphProfile& profile = family.profiles[i];
        if (!profile.definition.ready()) {
            throw std::logic_error(std::string(label) + " CUDA Graph definition is empty");
        }
        const auto existing =
            std::find_if(family.topologies.begin(), family.topologies.end(),
                         [&](const DecodeGraphTopology& topology) {
                             return topology.topology_class == profile.topology_class;
                         });
        if (existing != family.topologies.end()) { continue; }

        family.topologies.emplace_back();
        DecodeGraphTopology& topology = family.topologies.back();
        topology.topology_class       = profile.topology_class;
        topology.executable.instantiate(profile.definition);
        topology.installed_profile = i;
    }

    const auto install_and_upload = [&](DecodeGraphTopology& topology, std::size_t profile_index) {
        DecodeGraphProfile& profile = family.profiles[profile_index];
        if (topology.installed_profile != profile_index) {
            topology.executable.update(profile.definition);
            topology.installed_profile = profile_index;
        }
        topology.executable.upload(device.stream);
        device.synchronize();
    };

    for (DecodeGraphTopology& topology : family.topologies) {
        std::optional<std::size_t> first_profile;
        for (std::size_t i = 0; i < family.profiles.size(); ++i) {
            if (family.profiles[i].topology_class == topology.topology_class) {
                if (!first_profile) {
                    first_profile = i;
                    install_and_upload(topology, i);

                    DecodeGraphProfile& profile = family.profiles[i];
                    prepare(profile.min_execution_frontier, profile.batch_size);
                    device.synchronize();
                    topology.executable.launch(device.stream);
                    device.synchronize();
                    continue;
                }
                install_and_upload(topology, i);
            }
        }
        if (!first_profile) {
            throw std::logic_error(std::string(label) + " CUDA Graph topology has no definitions");
        }
        if (topology.installed_profile != *first_profile) {
            install_and_upload(topology, *first_profile);
        }
    }
}

} // namespace

ProgramImplCore::ProgramImplCore(const LoadedModelData& model_in, const SequencePlanImpl& plan,
                                   DeviceContext& device_in, std::string_view model_id,
                                   std::string_view weights_id,
                                   std::span<const std::uint8_t> artifact_fingerprint)
    : model(model_in), device(device_in), capacity(plan.capacity), kv_capacity(plan.kv_capacity),
      max_concurrency(plan.max_concurrency), prefill_chunk(plan.prefill_chunk),
      draft_window(plan.draft_window), speculative_backend(plan.speculative_backend),
      kv_dtype(plan.kv_dtype), kv_quant_group(plan.kv_quant_group),
      kv_packed_v(plan.kv_packed_v), kv_rotate_k(plan.kv_rotate_k), kv_rotate_v(plan.kv_rotate_v),
      kv_packed_k(plan.kv_packed_k), kv_e8_root(plan.kv_e8_root),
      proposal_head(plan.proposal_head),
      vision_enabled(plan.features.vision),
      use_cuda_graph(plan.use_cuda_graph), checkpoint_ring_capacity(plan.turn_checkpoint_ring),
      kv_payload_bytes(plan.persistent.kv_payload_bytes),
      text_kv_bytes(plan.persistent.decoder.text_kv.payload_bytes()),
      mtp_kv_bytes(plan.persistent.decoder.mtp_kv ? plan.persistent.decoder.mtp_kv->payload_bytes()
                                                  : 0),
      gdn_state_bytes(plan.persistent.decoder.linear_attention.payload_bytes()),
      dflash_kv_bytes(plan.persistent.dflash ? plan.persistent.dflash->kv_payload_bytes() : 0),
      replay_records_bytes(plan.persistent.replay_records
                               ? plan.persistent.replay_records->payload_bytes()
                               : 0),
      graph_allowance_bytes(plan.graph_allowance_bytes), workspace_plan(plan.workspace),
      continuation_compatibility_key(
          make_compatibility_key(plan, model_id, weights_id, artifact_fingerprint)),
      decision_state_compatibility_key(
          image::decision_state_compatibility_key(continuation_compatibility_key)),
      persistent(plan.persistent.bytes), workspace_storage(plan.workspace.capacity),
      work(DeviceSpan{workspace_storage.base(), workspace_storage.capacity()}),
      round_host(sizeof(TokenId)), continuation_transfer(16U * 1024U * 1024U),
      export_transfer(16U * 1024U * 1024U),
      ordinary_host(
          plan.speculative_backend == SpeculativeBackend::None
              ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_8::OrdinaryDecodeIngress) +
                                                     sizeof(qwen3_8::OrdinaryDecodeEgress))
              : std::nullopt),
      mtp_host(plan.speculative_backend == SpeculativeBackend::Mtp
                   ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_8::MtpDecodeIngress) +
                                                          sizeof(qwen3_8::MtpDecodeEgress))
                   : std::nullopt),
      dflash_host(plan.speculative_backend == SpeculativeBackend::DFlash
                      ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_8::DFlashDecodeIngress) +
                                                             sizeof(qwen3_8::DFlashDecodeEgress))
                      : std::nullopt) {
    if (model.weights_arena == nullptr) {
        throw std::invalid_argument("Qwen3.8 model view has no owning weight arena");
    }
    if (model.features != plan.features || model.mtp.has_value() != plan.features.mtp() ||
        model.dflash.has_value() != plan.features.dflash() ||
        model.optimized_proposal.has_value() != plan.features.optimized_proposal() ||
        model.vision.has_value() != plan.features.vision) {
        throw std::invalid_argument(
            "Qwen3.8 loaded weights do not match the frozen startup features");
    }
    if (model.mtp.has_value() && model.dflash.has_value()) {
        throw std::invalid_argument("MTP and DFlash model views are mutually exclusive");
    }
    if (model.dflash.has_value() && model.vision.has_value()) {
        throw std::invalid_argument("DFlash and Vision model views are mutually exclusive");
    }
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    decoder = std::make_unique<qwen3_8::DecoderState>(backing, plan.persistent.decoder);
    if (plan.persistent.replay_records) {
        replay_records.emplace(backing, *plan.persistent.replay_records);
    }
    if (replay_records.has_value() != (speculative_backend != SpeculativeBackend::None)) {
        throw std::logic_error("ReplaySSM records do not match the sequence plan");
    }
    if (plan.persistent.dflash) { dflash.emplace(backing, *plan.persistent.dflash); }
    if (dflash.has_value() != plan.features.dflash()) {
        throw std::logic_error("DFlash state does not match the frozen sequence plan");
    }

    io = qwen3_8::RoundState(backing, plan.persistent.round);
    if (io.mtp.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("round-state MTP extension does not match the sequence plan");
    }
    if (io.mtp_decode.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("MTP decode frame does not match the sequence plan");
    }
    if (io.ordinary.has_value() != (speculative_backend == SpeculativeBackend::None)) {
        throw std::logic_error("ordinary decode frame does not match the sequence plan");
    }
    if (io.dflash_prefill.has_value() != (speculative_backend == SpeculativeBackend::DFlash)) {
        throw std::logic_error("DFlash prefill scratch does not match the sequence plan");
    }
    if (io.dflash_decode.has_value() != (speculative_backend == SpeculativeBackend::DFlash)) {
        throw std::logic_error("DFlash decode frame does not match the sequence plan");
    }
    prefill_hidden               = plan.persistent.prefill_hidden.bind(backing);
    token_counts                 = plan.persistent.token_counts.bind(backing);
    sampling_config              = plan.persistent.sampling_config.bind(backing);
    tail_hidden_store            = plan.persistent.tail_hidden.bind(backing);
    turn_checkpoint_hidden_store = plan.persistent.turn_checkpoint_hidden.bind(backing);
    if (plan.persistent.decision_keys) {
        decision_keys = plan.persistent.decision_keys->bind(backing);
        decision_host.emplace(sizeof(float) *
                              std::max<std::size_t>(prefill_chunk, kMaximumDecisionOptions));
    }
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        SequenceState& sequence = sequences[lane];
        sequence.lane           = lane;
        sequence.tail_hidden    = tail_hidden_store.slice(1, static_cast<std::int32_t>(lane), 1);
        sequence.turn_checkpoint_hidden =
            turn_checkpoint_hidden_store.slice(1, static_cast<std::int32_t>(lane), 1);
        sequence.ledger.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_identity.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    }
    if (model.lora) {
        // Every slot starts empty and the bank starts zeroed, so a replay before the first stage
        // reads an exact no-op rather than uninitialized device memory.
        lora_slot_pool_.assign(static_cast<std::size_t>(model.lora->slots), -1);
        lora_slot_used_.assign(static_cast<std::size_t>(model.lora->slots), 0);
    }

    if (checkpoint_ring_capacity != 0) {
        if (speculative_backend == SpeculativeBackend::DFlash) {
            throw std::logic_error("turn checkpoint ring does not support the DFlash backend");
        }
        checkpoint_staging_store.emplace(checkpoint_entry_bytes() * max_concurrency);
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            CUDA_CHECK(cudaEventCreateWithFlags(&checkpoint_staging_events[lane],
                                                cudaEventDisableTiming));
        }
    }

    CUDA_CHECK(cudaStreamCreateWithFlags(&export_stream, cudaStreamNonBlocking));
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        CUDA_CHECK(cudaEventCreateWithFlags(&export_fence_events[lane], cudaEventDisableTiming));
    }

    set_device_i32(io.text_kv_table_row, 0);
    set_device_i32(io.backend_kv_table_row, 0);

    host_tokens = static_cast<TokenId*>(round_host.data());
    if (ordinary_host) {
        ordinary_host_ingress = static_cast<qwen3_8::OrdinaryDecodeIngress*>(ordinary_host->data());
        ordinary_host_egress  = reinterpret_cast<qwen3_8::OrdinaryDecodeEgress*>(
            static_cast<unsigned char*>(ordinary_host->data()) +
            sizeof(qwen3_8::OrdinaryDecodeIngress));
        *ordinary_host_ingress = {};
        *ordinary_host_egress  = {};
    }
    if (mtp_host) {
        mtp_host_ingress = static_cast<qwen3_8::MtpDecodeIngress*>(mtp_host->data());
        mtp_host_egress  = reinterpret_cast<qwen3_8::MtpDecodeEgress*>(
            static_cast<unsigned char*>(mtp_host->data()) + sizeof(qwen3_8::MtpDecodeIngress));
        *mtp_host_ingress = {};
        *mtp_host_egress  = {};
    }
    if (dflash_host) {
        dflash_host_ingress = static_cast<qwen3_8::DFlashDecodeIngress*>(dflash_host->data());
        dflash_host_egress  = reinterpret_cast<qwen3_8::DFlashDecodeEgress*>(
            static_cast<unsigned char*>(dflash_host->data()) +
            sizeof(qwen3_8::DFlashDecodeIngress));
        *dflash_host_ingress = {};
        *dflash_host_egress  = {};
    }
    if (io.dflash_prefill) {
        CUDA_CHECK(cudaMemsetAsync(io.dflash_prefill->produced_count.data, 0,
                                   io.dflash_prefill->produced_count.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(io.rope_delta.data, 0, io.rope_delta.bytes(), device.stream));
    if (io.mtp) {
        CUDA_CHECK(
            cudaMemsetAsync(io.mtp->position.data, 0, io.mtp->position.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(sampling_config.data, 0, sampling_config.bytes(), device.stream));
    device.synchronize();
    prepare_graphs();
    work.reset();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
}

ProgramImplCore::~ProgramImplCore() noexcept {
    if (device.stream != nullptr) { (void)cudaStreamSynchronize(device.stream); }
    if (export_stream != nullptr) {
        (void)cudaStreamSynchronize(export_stream);
        (void)cudaStreamDestroy(export_stream);
        export_stream = nullptr;
    }
    for (cudaEvent_t& event : checkpoint_staging_events) {
        if (event != nullptr) {
            (void)cudaEventDestroy(event);
            event = nullptr;
        }
    }
    for (cudaEvent_t& event : export_fence_events) {
        if (event != nullptr) {
            (void)cudaEventDestroy(event);
            event = nullptr;
        }
    }
}

std::int32_t ProgramImplCore::lora_slot(std::int32_t adapter) const {
    if (adapter < 0) { return -1; }
    for (std::size_t slot = 0; slot < lora_slot_pool_.size(); ++slot) {
        if (lora_slot_pool_[slot] == adapter) { return static_cast<std::int32_t>(slot); }
    }
    throw std::logic_error("a running sequence selects a LoRA adapter that is not resident");
}

bool ProgramImplCore::lora_slot_pinned(std::size_t slot) const noexcept {
    const std::int32_t occupant = lora_slot_pool_[slot];
    if (occupant < 0) { return false; }
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        if (sequences[lane].adapter != occupant) { continue; }
        const Lifecycle lifecycle = requests[lane].lifecycle;
        // A generating lane is executing against this slot's bytes right now. A merely retained
        // lane is an L1 optimization the caller can spend; it is soft, not pinned.
        if (lifecycle == Lifecycle::Prefilling || lifecycle == Lifecycle::Active ||
            lifecycle == Lifecycle::Pending) {
            return true;
        }
    }
    return false;
}

bool ProgramImplCore::ensure_adapter_resident(
    std::int32_t adapter, const std::function<void(std::uint32_t)>& release_retained) {
    if (adapter < 0) { return true; }
    if (lora_slot_pool_.empty() ||
        adapter >= static_cast<std::int32_t>(model.lora->pool_size)) {
        throw std::invalid_argument("selected LoRA adapter is outside the pool");
    }
    ++lora_clock_;
    for (std::size_t slot = 0; slot < lora_slot_pool_.size(); ++slot) {
        if (lora_slot_pool_[slot] == adapter) {
            lora_slot_used_[slot] = lora_clock_;
            return true;
        }
    }

    // An empty slot first, then the least recently used slot no generating lane depends on. A
    // pinned slot is never a candidate: its occupant's KV and GDN state would become
    // uninterpretable the moment the bytes changed.
    std::size_t victim = lora_slot_pool_.size();
    for (std::size_t slot = 0; slot < lora_slot_pool_.size(); ++slot) {
        if (lora_slot_pool_[slot] < 0) {
            victim = slot;
            break;
        }
        if (lora_slot_pinned(slot)) { continue; }
        if (victim == lora_slot_pool_.size() || lora_slot_used_[slot] < lora_slot_used_[victim]) {
            victim = slot;
        }
    }
    if (victim == lora_slot_pool_.size()) { return false; }

    // File open, identity verification and host assembly can fail for ordinary input reasons.
    // Complete them before demoting retained state; commit below is then only one device upload.
    prepare_lora_slot<Variant>(model, static_cast<std::size_t>(adapter));
    const std::int32_t evicted = lora_slot_pool_[victim];
    if (evicted >= 0) {
        // Retained lanes holding this adapter go out through the caller, which publishes their
        // sessions to L2/L3 and keeps its own L1 accounting straight. Doing it here would drop
        // the state instead of demoting it.
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            if (sequences[lane].adapter == evicted && sequences[lane].retained) {
                release_retained(lane);
            }
        }
        // Anything still naming the evicted adapter is an idle lane whose association would
        // silently hand it the new occupant's weights if it were ever reused.
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            if (sequences[lane].adapter == evicted) { sequences[lane].adapter = -1; }
        }
    }
    // Committed before the slot is recorded, so a failed upload leaves the slot unavailable rather
    // than claiming it holds bytes that did not arrive.
    lora_slot_pool_[victim] = -1;
    commit_lora_slot<Variant>(model, static_cast<std::uint32_t>(victim), device);
    lora_slot_pool_[victim] = adapter;
    lora_slot_used_[victim] = lora_clock_;
    ++lora_stage_count_;
    return true;
}

std::string ProgramImplCore::adapter_scope(std::int32_t adapter) const {
    if (adapter < 0) { return "base"; }
    if (!model.lora ||
        static_cast<std::size_t>(adapter) >= model.lora->fingerprints.size()) {
        throw std::invalid_argument("selected LoRA adapter is outside the pool");
    }
    static constexpr char kHex[] = "0123456789abcdef";
    const auto& fingerprint      = model.lora->fingerprints[static_cast<std::size_t>(adapter)];
    std::string scope;
    scope.reserve(fingerprint.size() * 2U);
    for (const std::uint8_t byte : fingerprint) {
        scope.push_back(kHex[byte >> 4]);
        scope.push_back(kHex[byte & 0x0FU]);
    }
    return scope;
}

bool ProgramImplCore::can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept {
    if (lane >= max_concurrency || plan.impl_ == nullptr) { return false; }
    const RequestControl& request = requests[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        return false;
    }
    const SequenceState& sequence = sequences[lane];
    const auto can_replace        = [](const PagedKVPool& pool, std::uint32_t old_pages,
                                std::uint32_t new_pages) {
        return old_pages <= pool.entitled_pages() && new_pages <= pool.logical_page_capacity() &&
               new_pages <= pool.page_group_count() - (pool.entitled_pages() - old_pages);
    };
    const std::uint32_t old_text = sequence.kv ? sequence.kv->text.page_entitlement() : 0;
    if (!can_replace(decoder->text_kv.pool(), old_text, plan.impl_->text_kv_page_entitlement)) {
        return false;
    }
    const qwen3_8::PagedKVCache* backend = backend_kv_cache();
    if (backend == nullptr) { return plan.impl_->backend_kv_page_entitlement == 0; }
    const std::uint32_t old_backend =
        sequence.kv && sequence.kv->backend ? sequence.kv->backend->page_entitlement() : 0;
    return can_replace(backend->pool(), old_backend, plan.impl_->backend_kv_page_entitlement);
}

bool ProgramImplCore::can_admit_lane_after_retained_eviction(
    std::uint32_t lane, const RequestPlan& plan) const noexcept {
    if (lane >= max_concurrency || plan.impl_ == nullptr) { return false; }
    const RequestControl& request = requests[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        return false;
    }

    std::uint32_t reclaimable_text    = 0;
    std::uint32_t reclaimable_backend = 0;
    for (std::uint32_t other = 0; other < max_concurrency; ++other) {
        if (other == lane || !sequences[other].retained || !sequences[other].kv) { continue; }
        reclaimable_text += sequences[other].kv->text.page_entitlement();
        if (sequences[other].kv->backend) {
            reclaimable_backend += sequences[other].kv->backend->page_entitlement();
        }
    }

    const auto can_replace = [](const PagedKVPool& pool, std::uint32_t old_pages,
                                std::uint32_t reclaimable_pages, std::uint32_t new_pages) {
        if (old_pages > pool.entitled_pages() ||
            reclaimable_pages > pool.entitled_pages() - old_pages ||
            new_pages > pool.logical_page_capacity()) {
            return false;
        }
        const std::uint32_t committed = pool.entitled_pages() - old_pages - reclaimable_pages;
        return new_pages <= pool.page_group_count() - committed;
    };

    const SequenceState& sequence = sequences[lane];
    const std::uint32_t old_text  = sequence.kv ? sequence.kv->text.page_entitlement() : 0;
    if (!can_replace(decoder->text_kv.pool(), old_text, reclaimable_text,
                     plan.impl_->text_kv_page_entitlement)) {
        return false;
    }

    const qwen3_8::PagedKVCache* backend = backend_kv_cache();
    if (backend == nullptr) { return plan.impl_->backend_kv_page_entitlement == 0; }
    const std::uint32_t old_backend =
        sequence.kv && sequence.kv->backend ? sequence.kv->backend->page_entitlement() : 0;
    return can_replace(backend->pool(), old_backend, reclaimable_backend,
                       plan.impl_->backend_kv_page_entitlement);
}

runtime::AdmissionResources ProgramImplCore::admission_capacity() const noexcept {
    const qwen3_8::PagedKVCache* backend = backend_kv_cache();
    return runtime::AdmissionResources{
        .active_lanes     = max_concurrency,
        .main_kv_pages    = decoder->text_kv.pool().page_group_count(),
        .backend_kv_pages = backend != nullptr ? backend->pool().page_group_count() : 0U,
    };
}

runtime::PrefillStepResult ProgramImplCore::start_prefill_lane(std::uint32_t lane,
                                                               PreparedPromptData&& prompt,
                                                               RequestPlan&& plan,
                                                               runtime::TransientRegion transient) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    SequenceState& sequence = sequences[lane];
    RequestControl& request = requests[lane];
    if (plan.impl_ == nullptr) { throw std::invalid_argument("request plan is empty"); }
    RequestPlanImpl& request_plan = *plan.impl_;
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("staged prefill requires a free request lane");
    }

    const std::uint32_t prompt_tokens = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (prompt_tokens != request_plan.summary.prompt_tokens ||
        (request_plan.vision.has_value() && !prompt.has_media())) {
        throw std::invalid_argument("request plan does not describe the prepared prompt");
    }
    const bool suffix_has_visual = std::any_of(
        prompt.token_types.begin() + static_cast<std::ptrdiff_t>(request_plan.reuse_base),
        prompt.token_types.end(), [](std::uint8_t type) { return type != 0; });
    if (suffix_has_visual != request_plan.vision.has_value()) {
        throw std::invalid_argument("request plan does not describe the prompt suffix modality");
    }
    if (request_plan.summary.transient_bytes != 0 &&
        (transient.data == nullptr || transient.size < request_plan.summary.transient_bytes ||
         transient.alignment < request_plan.summary.transient_alignment)) {
        throw std::invalid_argument("request transient region does not satisfy the plan");
    }
    if (request_plan.reuse != ReusePath::FullReset &&
        (!sequence.retained ||
         !qwen3_8::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                          request_plan.reuse_base))) {
        throw std::logic_error("planned resident prefix is no longer reusable");
    }
    if (request_plan.reuse != ReusePath::FullReset && sequence.adapter != request_plan.adapter) {
        throw std::logic_error("planned resident prefix belongs to a different LoRA adapter");
    }
    if (request_plan.reuse == ReusePath::RestoreTurnCheckpoint &&
        (!sequence.turn_checkpoint.valid ||
         sequence.turn_checkpoint.frontier != request_plan.reuse_base)) {
        // The planned checkpoint is not the resident one, so it must live in the host ring.
        // Landing it in the device checkpoint slot lets the restore below run unchanged.
        if (!upload_ring_checkpoint(sequence, request_plan.reuse_base)) {
            throw std::logic_error("planned turn checkpoint is unavailable");
        }
    }
    if (request_plan.reuse == ReusePath::RestoreUserTurnAnchor &&
        (!sequence.user_turn_anchor.valid ||
         sequence.user_turn_anchor.frontier != request_plan.reuse_base)) {
        throw std::logic_error("planned user turn anchor is unavailable");
    }
    if (request_plan.turn_checkpoint_action == TurnCheckpointAction::KeepExisting &&
        (!prompt.identity.turn_rewrite_boundary || !sequence.turn_checkpoint.valid ||
         sequence.turn_checkpoint.frontier != *prompt.identity.turn_rewrite_boundary)) {
        throw std::logic_error("planned turn checkpoint retention is unavailable");
    }
    if (request_plan.turn_checkpoint_action == TurnCheckpointAction::CaptureNew &&
        (!request_plan.turn_checkpoint_capture_frontier ||
         *request_plan.turn_checkpoint_capture_frontier <= request_plan.reuse_base ||
         *request_plan.turn_checkpoint_capture_frontier >= prompt_tokens)) {
        throw std::logic_error("planned turn checkpoint capture frontier is invalid");
    }
    {
        std::uint32_t previous_frontier = request_plan.reuse_base;
        for (const std::uint32_t frontier : request_plan.capture_frontiers) {
            if (frontier <= previous_frontier || frontier > prompt_tokens) {
                throw std::logic_error("planned boundary capture frontier is invalid");
            }
            previous_frontier = frontier;
        }
    }

    const auto started       = Clock::now();
    const std::uint32_t base = request_plan.reuse_base;
    const std::uint32_t initial_mtp_extent =
        speculative_backend == SpeculativeBackend::Mtp
            ? std::min({draft_window,
                        request_plan.summary.effective_output_tokens > 1
                            ? request_plan.summary.effective_output_tokens - 2
                            : 0U,
                        capacity - prompt_tokens > 0 ? capacity - prompt_tokens - 1 : 0U})
            : 0U;
    request.lifecycle       = Lifecycle::Empty;
    sequence.adapter        = request_plan.adapter;
    sequence.retained       = false;
    sequence.decision_state = false;
    sequence.captured_continuations.clear();
    try {
        if (checkpoint_ring_capacity != 0) {
            // Settle the staged checkpoint and the ring against this request's divergence
            // point: entries above the reuse base describe a history this prompt rewrites.
            if (request_plan.reuse == ReusePath::FullReset) {
                discard_checkpoint_staging(sequence);
                sequence.checkpoint_ring.clear();
            } else {
                if (checkpoint_staging[lane].pending &&
                    checkpoint_staging[lane].frontier <= base) {
                    drain_checkpoint_staging(sequence);
                } else {
                    discard_checkpoint_staging(sequence);
                }
                invalidate_checkpoint_ring(sequence, base);
            }
        }
        if (request_plan.reuse == ReusePath::FullReset) {
            sequence.kv.reset();
            ordered_reset(sequence);
            sequence.ledger.clear();
            sequence.text_kv_valid = 0;
            sequence.mtp_kv_valid  = 0;
            reserve_sequence_kv(sequence, request_plan.text_kv_page_entitlement,
                                request_plan.backend_kv_page_entitlement);
        } else if (request_plan.reuse == ReusePath::AppendAtFrontier) {
            if (!sequence.kv) {
                throw std::logic_error("resident prefix has no KV allocation bundle");
            }
            if (sequence.text_kv_valid < base) {
                throw std::logic_error("resident Text KV is shorter than the append frontier");
            }
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error("resident MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (speculative_backend == SpeculativeBackend::DFlash &&
                       sequence.dflash_context_frontier != base) {
                throw std::logic_error("resident DFlash context is not at the append frontier");
            }
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            sequence.text_kv_valid = base;
            sequence.ledger.resize(base);
        } else {
            if (!sequence.kv || sequence.text_kv_valid < base) {
                throw std::logic_error("resident turn checkpoint has no complete KV allocation");
            }
            sequence.text_kv_valid = base;
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error(
                        "turn-checkpoint MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (speculative_backend == SpeculativeBackend::DFlash) {
                if (!dflash || !sequence.kv->backend || sequence.dflash_context_frontier < base) {
                    throw std::logic_error("planned DFlash turn checkpoint is unavailable");
                }
                dflash->restore_turn_checkpoint(static_cast<std::int32_t>(sequence.lane),
                                                device.stream);
                sequence.dflash_context_frontier = base;
            }
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            if (request_plan.reuse == ReusePath::RestoreUserTurnAnchor) {
                import_linear_attention_state(
                    decoder->linear_attention,
                    LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
                    sequence.user_turn_anchor.linear_state, continuation_transfer, device.stream);
                // The MTP bridge reads the previous hidden from turn_checkpoint_hidden, so the
                // anchor lands there and the two restore paths stay identical downstream.
                copy_tensor_to_device(sequence.turn_checkpoint_hidden,
                                      sequence.user_turn_anchor.tail_hidden, continuation_transfer,
                                      device.stream);
            } else {
                decoder->linear_attention.copy_slot(
                    LinearStateSlots::turn_checkpoint_state_slot(sequence.lane, max_concurrency),
                    LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
                    device.stream);
            }
            sequence.ledger.resize(base);
        }

        trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
        bind_sequence_kv(sequence);
        const std::uint32_t backend_materialized =
            speculative_backend == SpeculativeBackend::Mtp
                ? std::min(capacity,
                           prompt_tokens + (initial_mtp_extent == 0 ? 0U : initial_mtp_extent - 1U))
            : speculative_backend == SpeculativeBackend::DFlash ? prompt_tokens
                                                                : 0U;
        materialize_sequence_kv(sequence, prompt_tokens, backend_materialized);
        install_sampling(sequence, request, request_plan.sampling);
        sequence.rope_delta = prompt.rope_delta;
        set_device_i32(io.rope_delta, sequence.rope_delta);

        if (request_plan.turn_checkpoint_action != TurnCheckpointAction::KeepExisting) {
            sequence.turn_checkpoint = {};
        }
        if (!request_plan.keep_user_turn_anchor) { sequence.user_turn_anchor = {}; }
        request.timings                 = {};
        request.pending                 = {};
        request.text_kv_page_ceiling    = request_plan.text_kv_page_ceiling;
        request.backend_kv_page_ceiling = request_plan.backend_kv_page_ceiling;
        sequence.mtp_draft_count        = 0;
        sequence.tail_hidden_valid      = base == prompt_tokens && sequence.tail_hidden_valid;
        sequence.ledger.assign(prompt.token_ids.begin(), prompt.token_ids.end());
        sequence.prefix_identity.assign(prompt);

        if (speculative_backend == SpeculativeBackend::DFlash) {
            if (!dflash || !io.dflash_decode || !sequence.kv->backend) {
                throw std::logic_error("DFlash prefill state is incomplete");
            }
            *dflash_host_ingress                         = {};
            dflash_host_ingress->lanes[0]                = static_cast<std::int32_t>(sequence.lane);
            dflash_host_ingress->dflash_kv_table_rows[0] = sequence.kv->backend->bound_row();
            CUDA_CHECK(cudaMemcpyAsync(io.dflash_decode->ingress.data, dflash_host_ingress,
                                       sizeof(qwen3_8::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                       device.stream));
        }

        const bool host_input_consumed = prompt.has_media() && !request_plan.vision;
        if (host_input_consumed) { prompt.release_media_payload(); }

        RequestControl::Prefill prefill{
            .prompt                           = std::move(prompt),
            .vision_plan                      = std::move(request_plan.vision),
            .vision                           = nullptr,
            .transient                        = transient,
            .turn_checkpoint_capture_frontier = request_plan.turn_checkpoint_capture_frontier,
            .user_turn_capture_frontier       = request_plan.user_turn_capture_frontier,
            .capture_frontiers                = std::move(request_plan.capture_frontiers),
            .base                             = base,
            .cursor                           = base,
            .prompt_tokens                    = prompt_tokens,
            .initial_mtp_extent               = initial_mtp_extent,
            .elapsed_seconds                  = 0.0,
            .host_input_consumed_pending      = host_input_consumed,
            .prepare_mtp                      = request_plan.prepare_mtp,
            .reuse                            = request_plan.reuse,
            .mtp_bridge                       = request_plan.mtp_bridge,
        };
        request.prefill.emplace(std::move(prefill));
        auto& staged = *request.prefill;
        if (staged.vision_plan) {
            staged.vision = std::make_unique<schedule::VisionPrefillSession>(
                device, model, work, staged.prompt, *staged.vision_plan, staged.transient);
        }
        staged.elapsed_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        request.lifecycle      = Lifecycle::Prefilling;
        return advance_prefill(sequence, request);
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        clear_lane(sequence, request);
        throw;
    }
}

runtime::PrefillStepResult ProgramImplCore::advance_prefill_lane(std::uint32_t lane) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    if (requests[lane].decision) { return advance_decision(sequences[lane], requests[lane]); }
    return advance_prefill(sequences[lane], requests[lane]);
}

void ProgramImplCore::resolve_prefill_lane(std::uint32_t lane, bool terminal) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    if (requests[lane].pending.kind != PendingKind::Begin) {
        throw std::logic_error("resolve_prefill_lane requires a pending prefill token");
    }
    resolve_non_speculative_pending(sequences[lane], requests[lane], 1, terminal);
}

void ProgramImplCore::resolve_pending_batch(std::span<const std::uint32_t> lanes,
                                            std::span<const std::uint32_t> accepted_tokens,
                                            std::span<const std::uint8_t> terminal,
                                            std::span<const std::uint8_t> cancelled) {
    if (lanes.empty() || lanes.size() > max_concurrency || accepted_tokens.size() != lanes.size() ||
        terminal.size() != lanes.size() || cancelled.size() != lanes.size()) {
        throw std::invalid_argument("pending batch resolution has inconsistent membership");
    }

    if (speculative_backend == SpeculativeBackend::None) {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending ||
                requests[lane].pending.kind != PendingKind::Ordinary) {
                throw std::logic_error("ordinary pending batch no longer matches Program state");
            }
            if (cancelled[row]) {
                clear_lane(sequences[lane], requests[lane]);
            } else {
                resolve_non_speculative_pending(sequences[lane], requests[lane],
                                                accepted_tokens[row], terminal[row] != 0);
            }
        }
        return;
    }

    if (!replay_records) {
        throw std::logic_error("speculative pending batch has no ReplaySSM records");
    }

    std::array<ops::GdnReplayFoldRow, kMaximumConcurrency> fold_rows{};
    std::array<std::int32_t, kMaximumConcurrency> hidden_selectors{};
    bool needs_hidden_correction = false;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending ||
            requests[lane].pending.kind != PendingKind::Speculative) {
            throw RoundFault("speculative pending batch no longer matches Program state: lane " +
                             std::to_string(lane) + " of " + std::to_string(max_concurrency));
        }
        const PendingCandidate& pending = requests[lane].pending;
        const SequenceState& sequence   = sequences[lane];
        if (sequence.execution_frontier != pending.base_E ||
            sequence.ledger_frontier != pending.base_S ||
            sequence.ledger.size() != pending.base_S ||
            sequence.prefix_identity.size() != pending.base_S ||
            sequence.text_kv_valid != pending.base_E ||
            (speculative_backend == SpeculativeBackend::Mtp &&
             sequence.mtp_kv_valid != pending.base_E) ||
            (speculative_backend == SpeculativeBackend::DFlash &&
             sequence.dflash_context_frontier != pending.base_E)) {
            throw RoundFault("speculative pending row is not at its recorded base: lane " +
                             std::to_string(lane) + " base E/S " + std::to_string(pending.base_E) +
                             "/" + std::to_string(pending.base_S) + ", frontier E/S " +
                             std::to_string(sequence.execution_frontier) + "/" +
                             std::to_string(sequence.ledger_frontier) + ", ledger " +
                             std::to_string(sequence.ledger.size()) + ", identity " +
                             std::to_string(sequence.prefix_identity.size()) + ", text kv " +
                             std::to_string(sequence.text_kv_valid) + ", mtp kv " +
                             std::to_string(sequence.mtp_kv_valid) + ", dflash " +
                             std::to_string(sequence.dflash_context_frontier));
        }
        const std::uint32_t committed = cancelled[row] ? 0U : accepted_tokens[row];
        if (!runtime::row_commit_is_licensed(cancelled[row] != 0, accepted_tokens[row],
                                             pending.produced, terminal[row] != 0)) {
            throw RoundFault("speculative pending row has an invalid committed prefix: lane " +
                             std::to_string(lane) + " committed " + std::to_string(committed) +
                             " of " + std::to_string(pending.produced) + " produced, accepted " +
                             std::to_string(accepted_tokens[row]) + ", terminal " +
                             std::to_string(terminal[row]) + ", cancelled " +
                             std::to_string(cancelled[row]));
        }
        fold_rows[row] = ops::GdnReplayFoldRow{
            .linear_state_slot = LinearStateSlots::current_state_slot(lane, max_concurrency),
            .commit_columns    = static_cast<std::int32_t>(committed),
        };
        const bool partial_terminal =
            !cancelled[row] && terminal[row] && committed < pending.produced;
        hidden_selectors[row] =
            static_cast<std::int32_t>(partial_terminal ? committed - 1U : pending.produced - 1U);
        needs_hidden_correction = needs_hidden_correction || partial_terminal;
    }

    const auto tail_started = Clock::now();
    try {
        ops::gdn_replay_fold(*replay_records, decoder->linear_attention.all_layers_view(),
                             std::span<const ops::GdnReplayFoldRow>(fold_rows.data(), lanes.size()),
                             device.stream);

        if (needs_hidden_correction) {
            const auto batch = static_cast<std::int32_t>(lanes.size());
            Tensor selector_tensor;
            Tensor hidden;
            Tensor selected;
            Tensor destinations;
            if (speculative_backend == SpeculativeBackend::Mtp && io.mtp_decode) {
                qwen3_8::MtpDecodeState& frame = *io.mtp_decode;
                selector_tensor                = frame.current_extents.slice(0, 0, batch);
                hidden                         = frame.target_hidden.slice(2, 0, batch);
                selected     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations = frame.lanes.slice(0, 0, batch);
            } else if (speculative_backend == SpeculativeBackend::DFlash && io.dflash_decode) {
                qwen3_8::DFlashDecodeState& frame = *io.dflash_decode;
                selector_tensor                   = frame.proposal_extents.slice(0, 0, batch);
                hidden                            = frame.target_hidden.slice(2, 0, batch);
                selected     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations = frame.lanes.slice(0, 0, batch);
            } else {
                throw std::logic_error("partial speculative commit has no target frame");
            }
            CUDA_CHECK(cudaMemcpyAsync(selector_tensor.data, hidden_selectors.data(),
                                       lanes.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       device.stream));
            ops::speculative_select_accepted_hidden(hidden, selector_tensor, selected,
                                                    device.stream);
            ops::scatter(selected, destinations, tail_hidden_store, device.stream);
        }

        if (speculative_backend == SpeculativeBackend::DFlash) {
            std::array<std::uint32_t, kMaximumConcurrency> append_lanes{};
            std::array<std::uint32_t, kMaximumConcurrency> append_starts{};
            std::array<std::uint32_t, kMaximumConcurrency> append_counts{};
            std::size_t append_size = 0;
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                if (!cancelled[row] && terminal[row]) {
                    append_lanes[append_size]  = lanes[row];
                    append_starts[append_size] = requests[lanes[row]].pending.base_E;
                    append_counts[append_size] = accepted_tokens[row];
                    ++append_size;
                }
            }
            if (append_size != 0) {
                enqueue_dflash_context_append(
                    std::span<const std::uint32_t>(append_lanes.data(), append_size),
                    std::span<const std::uint32_t>(append_starts.data(), append_size),
                    std::span<const std::uint32_t>(append_counts.data(), append_size));
            }
        }

        device.synchronize();
        work.reset();
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        work.reset();
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }

    const double tail_seconds = std::chrono::duration<double>(Clock::now() - tail_started).count();
    const std::uint32_t width = draft_window + 1U;
    try {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence = sequences[lanes[row]];
            RequestControl& request = requests[lanes[row]];
            if (cancelled[row]) {
                clear_lane(sequence, request);
                continue;
            }

            const PendingCandidate pending = request.pending;
            const std::uint32_t committed  = accepted_tokens[row];
            const TokenId* token_base =
                speculative_backend == SpeculativeBackend::Mtp
                    ? mtp_host_egress->licensed_tokens.data() + row * width
                    : dflash_host_egress->licensed_tokens.data() + row * width;
            sequence.ledger.insert(sequence.ledger.end(), token_base, token_base + committed);
            sequence.prefix_identity.append_generated(committed, sequence.rope_delta);
            sequence.execution_frontier = pending.base_E + committed;
            sequence.ledger_frontier    = pending.base_S + committed;
            sequence.text_kv_valid      = sequence.execution_frontier;
            sequence.tail_hidden_valid  = true;

            if (speculative_backend == SpeculativeBackend::Mtp) {
                sequence.mtp_kv_valid = sequence.execution_frontier;
                if (terminal[row]) {
                    sequence.mtp_draft_count = 0;
                } else {
                    const std::int32_t next  = mtp_host_egress->next_extents[row];
                    sequence.mtp_draft_count = static_cast<std::uint32_t>(next);
                    for (std::uint32_t step = 0; step < sequence.mtp_draft_count; ++step) {
                        sequence.mtp_drafts[step] =
                            mtp_host_egress->next_drafts[step * max_concurrency + row];
                    }
                }
            } else {
                sequence.dflash_context_frontier =
                    terminal[row] ? sequence.execution_frontier : pending.base_E;
            }

            trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
            if (terminal[row]) {
                release_sequence_growth_entitlement(sequence);
                unbind_sequence_kv(sequence);
                sequence.retained = true;
                request.lifecycle = Lifecycle::Complete;
            } else {
                request.lifecycle = Lifecycle::Active;
            }
            request.pending = {};
            request.timings.decode_seconds += tail_seconds;
        }
    } catch (...) {
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

void ProgramImplCore::abort_lane(std::uint32_t lane) noexcept {
    if (lane >= max_concurrency) { return; }
    clear_lane(sequences[lane], requests[lane]);
}

bool ProgramImplCore::has_retained_lane(std::uint32_t lane) const noexcept {
    return lane < max_concurrency && sequences[lane].retained;
}

bool ProgramImplCore::kv_reservation_fits(std::uint32_t text_pages,
                                           std::uint32_t backend_pages) const noexcept {
    if (text_pages == 0) { return false; }
    if (!decoder->text_kv.pool().can_reserve(text_pages)) { return false; }
    if (backend_pages == 0) { return true; }
    const qwen3_8::PagedKVCache* backend = backend_kv_cache();
    return backend != nullptr && backend->pool().can_reserve(backend_pages);
}

bool ProgramImplCore::try_grow_decode_headroom(std::uint32_t lane) {
    if (lane >= max_concurrency) { return false; }
    SequenceState& sequence       = sequences[lane];
    const RequestControl& request = requests[lane];
    if (!sequence.kv || !sequence.kv->text.valid()) { return false; }

    // The widest extent a single round can consume. Speculative rounds append the whole draft
    // window plus the verified token; an ordinary round appends one.
    const std::uint32_t round_extent =
        speculative_backend == SpeculativeBackend::None ? 0U : draft_window;
    const auto required_main_tokens = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(capacity, static_cast<std::uint64_t>(sequence.execution_frontier) +
                                              round_extent + 1ULL));

    // Both caches are one requirement. An MTP round materializes the main extent in the text cache
    // and that extent plus the draft window in the backend cache, so the backend can exhaust its
    // entitlement a page before the text cache does. Reporting success on the text extent alone
    // would let the round materialize past the backend entitlement and abort the engine.
    const bool has_backend = sequence.kv->backend.has_value();
    const auto footprint_for = [&](std::uint32_t main_tokens) {
        runtime::KvPageFootprint out;
        out.text_pages = pages_for_tokens(main_tokens);
        if (has_backend) {
            out.backend_pages = pages_for_tokens(
                speculative_backend == SpeculativeBackend::Mtp
                    ? static_cast<std::uint32_t>(std::min<std::uint64_t>(
                          capacity, static_cast<std::uint64_t>(main_tokens) + draft_window - 1ULL))
                    : main_tokens);
        }
        return out;
    };

    const runtime::KvPageFootprint required = footprint_for(required_main_tokens);
    const std::uint32_t current_text        = sequence.kv->text.page_entitlement();
    const std::uint32_t current_backend =
        has_backend ? sequence.kv->backend->page_entitlement() : 0U;
    if (required.text_pages <= current_text && required.backend_pages <= current_backend) {
        return true;
    }
    if (required.text_pages > request.text_kv_page_ceiling ||
        required.backend_pages > request.backend_kv_page_ceiling) {
        return false;
    }

    // Prefer the chunked step, but fall back to the exact requirement so an overshoot the pool
    // cannot cover does not fail a growth the lane could still afford.
    const std::uint32_t chunked_main =
        std::min(capacity, required_main_tokens + kDecodeGrowthChunkTokens);
    for (const std::uint32_t main_tokens : {chunked_main, required_main_tokens}) {
        const runtime::KvPageFootprint step = footprint_for(main_tokens);
        const std::uint32_t text_pages =
            std::max(current_text, std::clamp(step.text_pages, required.text_pages,
                                              request.text_kv_page_ceiling));
        const std::uint32_t backend_pages =
            has_backend ? std::max(current_backend,
                                   std::clamp(step.backend_pages, required.backend_pages,
                                              request.backend_kv_page_ceiling))
                        : 0U;
        if (!decoder->text_kv.pool().can_replace_entitlement(current_text, text_pages)) {
            continue;
        }
        if (has_backend && !backend_kv_cache()->pool().can_replace_entitlement(current_backend,
                                                                              backend_pages)) {
            continue;
        }
        resize_sequence_kv_entitlement(sequence, text_pages, backend_pages);
        return true;
    }
    return false;
}

runtime::KvPageFootprint
ProgramImplCore::retained_lane_kv_footprint(std::uint32_t lane) const noexcept {
    if (lane >= max_concurrency) { return {}; }
    const SequenceState& sequence = sequences[lane];
    if (!sequence.retained || !sequence.kv) { return {}; }
    runtime::KvPageFootprint footprint;
    footprint.text_pages = sequence.kv->text.page_entitlement();
    if (sequence.kv->backend) {
        footprint.backend_pages = sequence.kv->backend->page_entitlement();
    }
    return footprint;
}

std::size_t ProgramImplCore::retained_lane_resident_bytes(std::uint32_t lane) const noexcept {
    if (lane >= max_concurrency) { return 0; }
    const SequenceState& sequence = sequences[lane];
    const RequestControl& request = requests[lane];
    // A retained chat session always holds its tail hidden state; a retained decision state holds
    // only KV and the GDN slot.
    if (!sequence.retained || request.lifecycle != Lifecycle::Complete || request.prefill ||
        request.pending.kind != PendingKind::None || !sequence.kv ||
        (!sequence.tail_hidden_valid && !sequence.decision_state) ||
        !sequence.kv->text.valid() ||
        !sequence.kv->text.belongs_to(decoder->text_kv.pool())) {
        return 0;
    }

    std::size_t total = 0;
    const auto add = [&total](std::size_t bytes) {
        if (bytes > std::numeric_limits<std::size_t>::max() - total) { return false; }
        total += bytes;
        return true;
    };
    const auto paged_bytes = [](const PagedKVAllocation& allocation,
                                const PagedKVPool& pool) -> std::optional<std::size_t> {
        std::size_t bytes = 0;
        const std::uint32_t physical_pages = pool.page_group_count();
        if (physical_pages == 0) { return std::nullopt; }
        for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
            const std::size_t plane_bytes = pool.plane(plane).bytes();
            if (plane_bytes % physical_pages != 0) { return std::nullopt; }
            const std::size_t page_bytes = plane_bytes / physical_pages;
            if (allocation.mapped_page_count() != 0 &&
                page_bytes > std::numeric_limits<std::size_t>::max() /
                                 allocation.mapped_page_count()) {
                return std::nullopt;
            }
            const std::size_t mapped = page_bytes * allocation.mapped_page_count();
            if (mapped > std::numeric_limits<std::size_t>::max() - bytes) { return std::nullopt; }
            bytes += mapped;
        }
        return bytes;
    };
    const auto linear_slot_bytes = [&]() -> std::optional<std::size_t> {
        std::size_t bytes = 0;
        const auto& linear = decoder->linear_attention;
        for (std::uint32_t layer = 0; layer < linear.layer_count(); ++layer) {
            for (const Tensor tensor : {linear.conv_slot(layer, 0), linear.recurrent_slot(layer, 0)}) {
                if (tensor.bytes() > std::numeric_limits<std::size_t>::max() - bytes) {
                    return std::nullopt;
                }
                bytes += tensor.bytes();
            }
        }
        return bytes;
    };
    const auto cyclic_lane_bytes = [](const CyclicKVCache& cache) -> std::optional<std::size_t> {
        if (cache.lane_capacity() <= 0) { return std::nullopt; }
        std::size_t bytes = 0;
        for (std::uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = cache.layer_view(layer);
            for (const Tensor tensor : {view.k, view.v}) {
                const std::size_t lanes = static_cast<std::size_t>(cache.lane_capacity());
                if (tensor.bytes() % lanes != 0 || tensor.bytes() / lanes >
                                                       std::numeric_limits<std::size_t>::max() - bytes) {
                    return std::nullopt;
                }
                bytes += tensor.bytes() / lanes;
            }
        }
        return bytes;
    };

    const auto text = paged_bytes(sequence.kv->text, decoder->text_kv.pool());
    const auto linear = linear_slot_bytes();
    if (!text || !linear || !add(*text) || !add(*linear) ||
        (sequence.tail_hidden_valid && !add(sequence.tail_hidden.bytes()))) {
        return 0;
    }
    if (sequence.turn_checkpoint.valid &&
        (!add(*linear) || !add(sequence.turn_checkpoint_hidden.bytes()))) {
        return 0;
    }
    if (sequence.kv->backend) {
        const qwen3_8::PagedKVCache* backend = backend_kv_cache();
        if (backend == nullptr || !sequence.kv->backend->valid() ||
            !sequence.kv->backend->belongs_to(backend->pool())) {
            return 0;
        }
        const auto bytes = paged_bytes(*sequence.kv->backend, backend->pool());
        if (!bytes || !add(*bytes)) { return 0; }
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        if (!dflash) { return 0; }
        const auto local = cyclic_lane_bytes(dflash->local);
        if (!local || !add(*local)) { return 0; }
        if (sequence.turn_checkpoint.valid) {
            const auto checkpoint = cyclic_lane_bytes(dflash->turn_checkpoint_local);
            if (!checkpoint || !add(*checkpoint)) { return 0; }
        }
    }
    return total;
}

std::size_t ProgramImplCore::retained_lane_reused_bytes(
    std::uint32_t lane, const RequestPlanImpl& plan) const noexcept {
    if (lane >= max_concurrency || plan.reuse == ReusePath::FullReset || plan.reuse_base == 0) {
        return 0;
    }
    const SequenceState& sequence = sequences[lane];
    if (!sequence.retained || !sequence.kv || !sequence.kv->text.valid() ||
        !sequence.kv->text.belongs_to(decoder->text_kv.pool())) {
        return 0;
    }
    const bool checkpoint = plan.reuse == ReusePath::RestoreTurnCheckpoint;
    if (checkpoint &&
        (!sequence.turn_checkpoint.valid || sequence.turn_checkpoint.frontier != plan.reuse_base)) {
        return 0;
    }

    const auto page_bytes = [](const PagedKVPool& pool) -> std::optional<std::size_t> {
        if (pool.page_group_count() == 0) return std::nullopt;
        std::size_t total = 0;
        for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
            if (pool.plane(plane).bytes() % pool.page_group_count() != 0) return std::nullopt;
            const std::size_t bytes = pool.plane(plane).bytes() / pool.page_group_count();
            if (bytes > std::numeric_limits<std::size_t>::max() - total) return std::nullopt;
            total += bytes;
        }
        return total;
    };
    const auto linear_bytes = [&]() -> std::optional<std::size_t> {
        std::size_t total = 0;
        for (std::uint32_t layer = 0; layer < decoder->linear_attention.layer_count(); ++layer) {
            for (const Tensor tensor : {decoder->linear_attention.conv_slot(layer, 0),
                                        decoder->linear_attention.recurrent_slot(layer, 0)}) {
                if (tensor.bytes() > std::numeric_limits<std::size_t>::max() - total) {
                    return std::nullopt;
                }
                total += tensor.bytes();
            }
        }
        return total;
    };
    const auto cyclic_bytes = [](const CyclicKVCache& cache) -> std::optional<std::size_t> {
        if (cache.lane_capacity() <= 0) return std::nullopt;
        std::size_t total = 0;
        for (std::uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
            const auto view = cache.layer_view(layer);
            for (const Tensor tensor : {view.k, view.v}) {
                const std::size_t lanes = static_cast<std::size_t>(cache.lane_capacity());
                if (tensor.bytes() % lanes != 0 ||
                    tensor.bytes() / lanes > std::numeric_limits<std::size_t>::max() - total) {
                    return std::nullopt;
                }
                total += tensor.bytes() / lanes;
            }
        }
        return total;
    };
    const auto add = [](std::size_t left, std::size_t right) -> std::optional<std::size_t> {
        if (right > std::numeric_limits<std::size_t>::max() - left) return std::nullopt;
        return left + right;
    };

    const auto main_page = page_bytes(decoder->text_kv.pool());
    const auto linear = linear_bytes();
    if (!main_page || !linear) return 0;
    // A decision state is the text KV and the GDN slot alone: no tail hidden or backend state.
    const bool decision = sequence.decision_state;
    auto current_state = add(*linear, decision ? std::size_t{0} : sequence.tail_hidden.bytes());
    auto checkpoint_state = add(*linear, sequence.turn_checkpoint_hidden.bytes());
    if (!current_state || !checkpoint_state) return 0;

    std::uint32_t backend_tokens = 0;
    std::size_t backend_page_bytes = 0;
    if (!decision && speculative_backend == SpeculativeBackend::Mtp) {
        backend_tokens = plan.reuse_base - 1U;
    } else if (!decision && speculative_backend == SpeculativeBackend::DFlash) {
        backend_tokens = plan.reuse_base;
        if (!dflash) return 0;
        const auto current_local = cyclic_bytes(dflash->local);
        const auto checkpoint_local = cyclic_bytes(dflash->turn_checkpoint_local);
        if (!current_local || !checkpoint_local) return 0;
        current_state = add(*current_state, *current_local);
        checkpoint_state = add(*checkpoint_state, *checkpoint_local);
        if (!current_state || !checkpoint_state) return 0;
    }
    if (backend_tokens != 0) {
        const qwen3_8::PagedKVCache* backend = backend_kv_cache();
        if (backend == nullptr || !sequence.kv->backend || !sequence.kv->backend->valid() ||
            !sequence.kv->backend->belongs_to(backend->pool())) {
            return 0;
        }
        const auto bytes = page_bytes(backend->pool());
        if (!bytes) return 0;
        backend_page_bytes = *bytes;
    }

    const std::size_t main_pages =
        (static_cast<std::size_t>(plan.reuse_base) + kPagedKVPageSize - 1U) / kPagedKVPageSize;
    const std::size_t backend_pages =
        (static_cast<std::size_t>(backend_tokens) + kPagedKVPageSize - 1U) / kPagedKVPageSize;
    if (main_pages > sequence.kv->text.mapped_page_count() ||
        (backend_pages != 0 && backend_pages > sequence.kv->backend->mapped_page_count())) {
        return 0;
    }
    return runtime::l1_reused_bytes_at_frontier(
               runtime::L1ReuseByteLayout{.main_page_bytes = *main_page,
                                          .backend_page_bytes = backend_page_bytes,
                                          .current_state_bytes = *current_state,
                                          .checkpoint_state_bytes = *checkpoint_state},
               plan.reuse_base, backend_tokens, kPagedKVPageSize, checkpoint)
        .value_or(0);
}

void ProgramImplCore::evict_retained_lane(std::uint32_t lane) noexcept {
    if (!has_retained_lane(lane)) { return; }
    clear_lane(sequences[lane], requests[lane]);
}

std::vector<PromptBoundaryAlias>
ProgramImplCore::boundary_aliases(const PreparedPromptData& prompt) const {
    try {
        return image::boundary_aliases(continuation_compatibility_key, prompt);
    } catch (...) { return {}; }
}

std::vector<CapturedContinuation>
ProgramImplCore::take_captured_continuations_lane(std::uint32_t lane) {
    if (lane >= max_concurrency) return {};
    return std::exchange(sequences[lane].captured_continuations, {});
}

cache::ContinuationImage ProgramImplCore::export_stable_continuation(
    const SequenceState& sequence, const PreparedPromptData& prompt,
    std::uint32_t frontier) const {
    if (frontier == 0 || frontier > prompt.token_ids.size() || !sequence.kv ||
        sequence.text_kv_valid < frontier ||
        (speculative_backend == SpeculativeBackend::Mtp && sequence.mtp_kv_valid < frontier) ||
        (speculative_backend == SpeculativeBackend::DFlash &&
         sequence.dflash_context_frontier < frontier)) {
        throw std::logic_error("stable prefix state is incomplete");
    }

    ResidentPrefixIdentity identity;
    identity.assign(prompt);
    identity.truncate(frontier);
    std::vector<TokenId> ledger(prompt.token_ids.begin(),
                                prompt.token_ids.begin() + static_cast<std::ptrdiff_t>(frontier));
    const bool mtp_backend    = speculative_backend == SpeculativeBackend::Mtp;
    const bool dflash_backend = speculative_backend == SpeculativeBackend::DFlash;

    cache::ContinuationImage out;
    out.format_version    = image::kTargetImageVersion;
    out.compatibility_key = continuation_compatibility_key;
    out.frontier_tokens   = frontier;
    out.frontier_prefix_digest = image::prefix_filter_digest(prompt, frontier);
    out.prefix_identity   = image::encode_prefix(ledger, identity.export_prefix(frontier));
    out.frontier_metadata = image::encode_frontier(image::FrontierMetadata{
        .execution_frontier = frontier,
        .ledger_frontier = frontier,
        .rope_delta = prompt.rope_delta,
        .text_kv_valid = frontier,
        .backend_kv_valid = mtp_backend || dflash_backend ? frontier : 0U,
    });
    out.boundary_metadata =
        image::encode_boundary(image::BoundaryMetadata{.valid = false, .frontier = 0});
    image::emit_paged(out.segments, "main.text_kv",
                      export_paged_kv_logical(sequence.kv->text, frontier, continuation_transfer,
                                              device.stream));
    out.segments.emplace(
        "main.gdn",
        image::export_linear_segment(
            decoder->linear_attention,
            LinearStateSlots::turn_checkpoint_state_slot(sequence.lane, max_concurrency),
            continuation_transfer, device.stream));
    out.segments.emplace(
        "main.tail_hidden",
        image::encode_tensor(copy_tensor_to_host(sequence.turn_checkpoint_hidden,
                                                 continuation_transfer, device.stream)));
    if (mtp_backend) {
        image::emit_paged(out.segments, "mtp.kv",
                          export_paged_kv_logical(*sequence.kv->backend, frontier,
                                                  continuation_transfer, device.stream));
    }
    if (dflash_backend) {
        if (!dflash || !sequence.kv->backend) {
            throw std::logic_error("stable DFlash state is incomplete");
        }
        image::emit_paged(out.segments, "dflash.full_kv",
                          export_paged_kv_logical(*sequence.kv->backend, frontier,
                                                  continuation_transfer, device.stream));
        out.segments.emplace(
            "dflash.local",
            image::encode_cyclic(export_cyclic_kv_lane(
                dflash->turn_checkpoint_local, static_cast<std::int32_t>(sequence.lane),
                continuation_transfer, device.stream)));
    }
    return out;
}

cache::ContinuationImage ProgramImplCore::export_continuation_lane(std::uint32_t lane) const {
    return export_lane_image(lane, continuation_transfer, device.stream);
}

void ProgramImplCore::fence_lane_for_export(std::uint32_t lane) {
    if (lane >= max_concurrency) { throw std::out_of_range("continuation lane is out of range"); }
    CUDA_CHECK(cudaEventRecord(export_fence_events[lane], device.stream));
}

cache::ContinuationImage
ProgramImplCore::export_continuation_lane_background(std::uint32_t lane) const {
    if (lane >= max_concurrency) { throw std::out_of_range("continuation lane is out of range"); }
    CUDA_CHECK(cudaStreamWaitEvent(export_stream, export_fence_events[lane], 0));
    return export_lane_image(lane, export_transfer, export_stream);
}

cache::ContinuationImage ProgramImplCore::export_lane_image(std::uint32_t lane,
                                                            PinnedTransferBuffer& transfer,
                                                            cudaStream_t stream) const {
    if (lane >= max_concurrency) { throw std::out_of_range("continuation lane is out of range"); }
    const SequenceState& sequence = sequences[lane];
    const RequestControl& request = requests[lane];
    if (!sequence.retained || request.lifecycle != Lifecycle::Complete || request.prefill ||
        request.pending.kind != PendingKind::None || !sequence.kv ||
        sequence.kv->text.bound_row() >= 0 ||
        (sequence.kv->backend && sequence.kv->backend->bound_row() >= 0) ||
        !sequence.tail_hidden_valid || sequence.execution_frontier == 0 ||
        sequence.execution_frontier > capacity ||
        sequence.ledger_frontier != sequence.execution_frontier + 1U ||
        sequence.ledger.size() != sequence.ledger_frontier ||
        sequence.prefix_identity.size() != sequence.ledger_frontier ||
        sequence.text_kv_valid != sequence.execution_frontier) {
        throw std::logic_error("lane is not a complete exportable continuation");
    }
    if (sequence.turn_checkpoint.valid &&
        (sequence.turn_checkpoint.frontier == 0 ||
         sequence.turn_checkpoint.frontier > sequence.execution_frontier)) {
        throw std::logic_error("continuation turn checkpoint is outside the execution frontier");
    }

    const bool mtp_backend    = speculative_backend == SpeculativeBackend::Mtp;
    const bool dflash_backend = speculative_backend == SpeculativeBackend::DFlash;
    const std::uint32_t backend_valid = backend_kv_valid(sequence);
    if ((!mtp_backend && !dflash_backend && (sequence.kv->backend || backend_valid != 0)) ||
        ((mtp_backend || dflash_backend) &&
         (!sequence.kv->backend || backend_valid != sequence.execution_frontier)) ||
        (!mtp_backend && sequence.mtp_draft_count != 0) ||
        sequence.mtp_draft_count > sequence.mtp_drafts.size()) {
        throw std::logic_error("continuation backend state has inconsistent valid extents");
    }

    cache::ContinuationImage out;
    out.format_version     = image::kTargetImageVersion;
    out.compatibility_key  = continuation_compatibility_key;
    out.frontier_tokens    = sequence.execution_frontier;
    out.boundary_tokens    = sequence.turn_checkpoint.valid ? sequence.turn_checkpoint.frontier : 0;
    out.prefix_identity    = image::encode_prefix(
        sequence.ledger, sequence.prefix_identity.export_prefix(sequence.ledger_frontier));
    const auto prefix_snapshot = sequence.prefix_identity.export_prefix(sequence.ledger_frontier);
    out.frontier_prefix_digest =
        image::prefix_filter_digest(sequence.ledger, prefix_snapshot, sequence.execution_frontier);
    if (sequence.turn_checkpoint.valid) {
        out.boundary_prefix_digest = image::prefix_filter_digest(
            sequence.ledger, prefix_snapshot, sequence.turn_checkpoint.frontier);
    }
    out.frontier_metadata = image::encode_frontier(image::FrontierMetadata{
        .execution_frontier = sequence.execution_frontier,
        .ledger_frontier = sequence.ledger_frontier,
        .rope_delta = sequence.rope_delta,
        .text_kv_valid = sequence.text_kv_valid,
        .backend_kv_valid = backend_valid,
        .mtp_drafts = std::vector<TokenId>(sequence.mtp_drafts.begin(),
                                           sequence.mtp_drafts.begin() + sequence.mtp_draft_count),
    });
    out.boundary_metadata = image::encode_boundary(image::BoundaryMetadata{
        .valid = sequence.turn_checkpoint.valid,
        .frontier = sequence.turn_checkpoint.frontier,
    });

    image::emit_paged(out.segments, "main.text_kv",
                      export_paged_kv_logical(sequence.kv->text, sequence.text_kv_valid, transfer,
                                              stream));
    out.segments.emplace(
        "main.gdn",
        image::export_linear_segment(
            decoder->linear_attention,
            LinearStateSlots::current_state_slot(sequence.lane, max_concurrency), transfer,
            stream));
    out.segments.emplace("main.tail_hidden",
                          image::encode_tensor(
                              copy_tensor_to_host(sequence.tail_hidden, transfer, stream)));
    if (sequence.turn_checkpoint.valid) {
        out.segments.emplace(
            "checkpoint.gdn",
            image::export_linear_segment(
                decoder->linear_attention,
                LinearStateSlots::turn_checkpoint_state_slot(sequence.lane, max_concurrency),
                transfer, stream));
        out.segments.emplace(
            "checkpoint.hidden",
            image::encode_tensor(
                copy_tensor_to_host(sequence.turn_checkpoint_hidden, transfer, stream)));
    }
    if (mtp_backend) {
        image::emit_paged(out.segments, "mtp.kv",
                          export_paged_kv_logical(*sequence.kv->backend, sequence.mtp_kv_valid,
                                                  transfer, stream));
    }
    if (dflash_backend) {
        if (!dflash) { throw std::logic_error("DFlash continuation has no persistent state"); }
        image::emit_paged(out.segments, "dflash.full_kv",
                          export_paged_kv_logical(*sequence.kv->backend,
                                                  sequence.dflash_context_frontier, transfer,
                                                  stream));
        out.segments.emplace(
            "dflash.local",
            image::encode_cyclic(export_cyclic_kv_lane(
                dflash->local, static_cast<std::int32_t>(sequence.lane), transfer, stream)));
        if (sequence.turn_checkpoint.valid) {
            out.segments.emplace(
                "dflash.checkpoint_local",
                image::encode_cyclic(export_cyclic_kv_lane(
                    dflash->turn_checkpoint_local, static_cast<std::int32_t>(sequence.lane),
                    transfer, stream)));
        }
    }
    return out;
}

std::uint32_t
ProgramImplCore::preflight_continuation_metadata(
    const cache::SessionCandidateDescriptor& candidate,
    const PreparedPromptData& prompt) const noexcept {
    try {
        if (candidate.status != cache::CacheLookupStatus::Hit ||
            candidate.image_format_version != image::kTargetImageVersion ||
            candidate.compatibility_key != continuation_compatibility_key ||
            candidate.frontier_tokens == 0 || candidate.frontier_tokens > capacity ||
            candidate.frontier_tokens > std::numeric_limits<std::uint32_t>::max() ||
            candidate.frontier_prefix_digest.size() != artifact::Sha256Digest{}.size() ||
            candidate.boundary_tokens > candidate.frontier_tokens ||
            ((candidate.boundary_tokens == 0) != candidate.boundary_prefix_digest.empty()) ||
            (!candidate.boundary_prefix_digest.empty() &&
             candidate.boundary_prefix_digest.size() != artifact::Sha256Digest{}.size())) {
            return 0;
        }
        const auto frontier = static_cast<std::uint32_t>(candidate.frontier_tokens);
        if (frontier <= prompt.token_ids.size() &&
            image::prefix_filter_digest(prompt, frontier) == candidate.frontier_prefix_digest) {
            return frontier;
        }
        if (candidate.boundary_tokens == 0 ||
            candidate.boundary_tokens > std::numeric_limits<std::uint32_t>::max()) {
            return 0;
        }
        const auto boundary = static_cast<std::uint32_t>(candidate.boundary_tokens);
        return boundary < prompt.token_ids.size() &&
                       image::prefix_filter_digest(prompt, boundary) ==
                           candidate.boundary_prefix_digest
                   ? boundary
                   : 0;
    } catch (...) { return 0; }
}

// The exported inventory is a property of the enabled state, not of a call site. The reuse-depth
// probe and the restore preflight both compare against it, and while each spelled it out a format
// change could satisfy one and silently fail the other.
std::set<std::string> ProgramImplCore::continuation_segment_inventory(bool boundary_valid,
                                                                     bool mtp_backend,
                                                                     bool dflash_backend) const {
    std::set<std::string> names{"main.gdn", "main.tail_hidden"};
    image::paged_segment_names("main.text_kv", decoder->text_kv.pool().plane_count(), names);
    if (boundary_valid) {
        names.emplace("checkpoint.gdn");
        names.emplace("checkpoint.hidden");
    }
    if (mtp_backend) {
        image::paged_segment_names("mtp.kv", backend_kv_cache()->pool().plane_count(), names);
    }
    if (dflash_backend) {
        image::paged_segment_names("dflash.full_kv", backend_kv_cache()->pool().plane_count(),
                                   names);
        names.emplace("dflash.local");
        if (boundary_valid) { names.emplace("dflash.checkpoint_local"); }
    }
    return names;
}

std::uint32_t
ProgramImplCore::preflight_continuation(const cache::ContinuationImage& candidate,
                                        const PreparedPromptData& prompt,
                                        std::uint32_t* divergence_tokens) const noexcept {
    if (divergence_tokens != nullptr) { *divergence_tokens = 0; }
    try {
        if (candidate.format_version != image::kTargetImageVersion ||
            candidate.compatibility_key != continuation_compatibility_key) {
            return 0;
        }

        const image::FrontierMetadata metadata =
            image::decode_frontier(candidate.frontier_metadata);
        const image::BoundaryMetadata boundary =
            image::decode_boundary(candidate.boundary_metadata);
        const std::uint32_t frontier = metadata.execution_frontier;
        const bool prefix_only = metadata.ledger_frontier == frontier;
        if (frontier == 0 || frontier > capacity || candidate.frontier_tokens != frontier ||
            !image::valid_ledger_frontier(frontier, metadata.ledger_frontier) ||
            metadata.text_kv_valid != frontier ||
            candidate.boundary_tokens != (boundary.valid ? boundary.frontier : 0U) ||
            (boundary.valid && (boundary.frontier == 0 || boundary.frontier > frontier))) {
            return 0;
        }

        image::PrefixData prefix =
            image::decode_prefix(candidate.prefix_identity, metadata.ledger_frontier);
        if (divergence_tokens != nullptr) {
            *divergence_tokens = qwen3_8::detail::prefix_divergence_tokens(prompt, prefix.ledger);
        }
        if (prefix.ledger.size() != metadata.ledger_frontier ||
            image::prefix_filter_digest(prefix.ledger, prefix.identity, frontier) !=
                candidate.frontier_prefix_digest ||
            (boundary.valid &&
             image::prefix_filter_digest(prefix.ledger, prefix.identity, boundary.frontier) !=
                 candidate.boundary_prefix_digest) ||
            (!boundary.valid && !candidate.boundary_prefix_digest.empty())) {
            return 0;
        }
        if (!prefix_only) {
            const std::int64_t generated_position =
                static_cast<std::int64_t>(frontier) + metadata.rope_delta;
            if (generated_position < std::numeric_limits<std::int32_t>::min() ||
                generated_position > std::numeric_limits<std::int32_t>::max() ||
                prefix.identity.token_types[frontier] != 0 ||
                std::any_of(prefix.identity.positions.begin(), prefix.identity.positions.end(),
                            [&](const auto& axis) {
                                return axis[frontier] !=
                                       static_cast<std::int32_t>(generated_position);
                            })) {
                return 0;
            }
        }
        ResidentPrefixIdentity resident_identity;
        resident_identity.restore(std::move(prefix.identity));
        const std::uint32_t reusable_depth = qwen3_8::detail::continuation_reuse_depth(
            prompt, prefix.ledger, resident_identity, frontier,
            boundary.valid ? std::optional<std::uint32_t>(boundary.frontier) : std::nullopt);
        if (reusable_depth == 0) { return 0; }

        const bool mtp_backend    = speculative_backend == SpeculativeBackend::Mtp;
        const bool dflash_backend = speculative_backend == SpeculativeBackend::DFlash;
        if ((prefix_only && !metadata.mtp_drafts.empty()) ||
            (!mtp_backend && !metadata.mtp_drafts.empty()) ||
            metadata.mtp_drafts.size() > qwen3_8::kMtpDecodeMaximumDrafts ||
            ((!mtp_backend && !dflash_backend) != (metadata.backend_kv_valid == 0)) ||
            ((mtp_backend || dflash_backend) && metadata.backend_kv_valid != frontier)) {
            return 0;
        }

        const auto expected =
            continuation_segment_inventory(boundary.valid, mtp_backend, dflash_backend);
        std::set<std::string> actual;
        for (const auto& [name, unused] : candidate.segments) {
            (void)unused;
            actual.emplace(name);
        }
        if (actual != expected) { return 0; }

        if (dflash_backend && !dflash) { return 0; }
        // Preflight deliberately stops at metadata, prefix digests, and the segment inventory. It
        // does not decode the payload: decoding materialises the whole image on the host, and
        // import_continuation_lane decodes the same segments itself before it writes anything to
        // the device, with clear_lane() on failure. Validating here as well made every restore
        // materialise the image twice - once discarded, once kept - which measured as ~270 ms of
        // duplicated host work per restore, the single largest term in a restore.
        return reusable_depth;
    } catch (...) { return 0; }
}

std::shared_ptr<DecodedContinuation>
ProgramImplCore::decode_continuation(const cache::ContinuationImage& candidate) const {
    // Startup-fixed geometry only: pool plane specs, linear/cyclic state shapes and lane-invariant
    // tensor extents. No lane state, no device memory, no CUDA call - see the class comment on
    // DecodedContinuation for why that matters.
    const image::FrontierMetadata metadata = image::decode_frontier(candidate.frontier_metadata);
    const image::BoundaryMetadata boundary = image::decode_boundary(candidate.boundary_metadata);
    const bool mtp_backend                 = speculative_backend == SpeculativeBackend::Mtp;
    const bool dflash_backend              = speculative_backend == SpeculativeBackend::DFlash;

    auto out     = std::make_shared<DecodedContinuation>();
    out->text_kv = image::decode_paged(candidate.segments, "main.text_kv", decoder->text_kv.pool(),
                                       metadata.text_kv_valid);
    out->current_gdn =
        image::decode_linear(candidate.segments.at("main.gdn"), decoder->linear_attention);
    out->tail_hidden = image::decode_tensor(candidate.segments.at("main.tail_hidden"),
                                            sequences[0].tail_hidden.bytes());
    if (boundary.valid) {
        out->checkpoint_gdn.emplace(image::decode_linear(candidate.segments.at("checkpoint.gdn"),
                                                         decoder->linear_attention));
        out->checkpoint_hidden.emplace(
            image::decode_tensor(candidate.segments.at("checkpoint.hidden"),
                                 sequences[0].turn_checkpoint_hidden.bytes()));
    }
    if (mtp_backend) {
        out->backend_kv.emplace(image::decode_paged(candidate.segments, "mtp.kv",
                                                    backend_kv_cache()->pool(),
                                                    metadata.backend_kv_valid));
    } else if (dflash_backend) {
        if (!dflash) { throw std::invalid_argument("DFlash continuation state is unavailable"); }
        out->backend_kv.emplace(image::decode_paged(candidate.segments, "dflash.full_kv",
                                                    backend_kv_cache()->pool(),
                                                    metadata.backend_kv_valid));
        out->dflash_local.emplace(
            image::decode_cyclic(candidate.segments.at("dflash.local"), dflash->local));
        if (boundary.valid) {
            out->dflash_checkpoint_local.emplace(
                image::decode_cyclic(candidate.segments.at("dflash.checkpoint_local"),
                                     dflash->turn_checkpoint_local));
        }
    }
    return out;
}

ContinuationRestoreFailure ProgramImplCore::import_continuation_lane(
    std::uint32_t lane, const cache::ContinuationImage& candidate,
    const DecodedContinuation& decoded, const PreparedPromptData& prompt, std::int32_t adapter,
    runtime::KvPageFootprint entitlement) noexcept {
    using Failure = ContinuationRestoreFailure;
    if (lane >= max_concurrency) { return Failure::LaneUnavailable; }
    SequenceState& sequence = sequences[lane];
    RequestControl& request = requests[lane];
    if (request.lifecycle != Lifecycle::Empty || request.prefill ||
        request.pending.kind != PendingKind::None || sequence.retained || sequence.kv) {
        return Failure::LaneUnavailable;
    }

    const std::uint32_t reusable_depth = preflight_continuation(candidate, prompt);
    if (reusable_depth == 0) { return Failure::VerifyDepthMismatch; }

    try {
        if (candidate.format_version != image::kTargetImageVersion ||
            candidate.compatibility_key != continuation_compatibility_key) {
            return Failure::MetadataMismatch;
        }

        const image::FrontierMetadata metadata =
            image::decode_frontier(candidate.frontier_metadata);
        const image::BoundaryMetadata boundary =
            image::decode_boundary(candidate.boundary_metadata);
        const std::uint32_t frontier = metadata.execution_frontier;
        const bool prefix_only = metadata.ledger_frontier == frontier;
        if (frontier == 0 || frontier > capacity || candidate.frontier_tokens != frontier ||
            !image::valid_ledger_frontier(frontier, metadata.ledger_frontier) ||
            metadata.text_kv_valid != frontier ||
            candidate.boundary_tokens != (boundary.valid ? boundary.frontier : 0U) ||
            (boundary.valid && (boundary.frontier == 0 || boundary.frontier > frontier))) {
            return Failure::MetadataMismatch;
        }
        image::PrefixData prefix =
            image::decode_prefix(candidate.prefix_identity, metadata.ledger_frontier);
        if (prefix.ledger.size() != metadata.ledger_frontier) { return Failure::DecodeFailed; }

        if (!prefix_only) {
            const std::int64_t generated_position =
                static_cast<std::int64_t>(frontier) + metadata.rope_delta;
            if (generated_position < std::numeric_limits<std::int32_t>::min() ||
                generated_position > std::numeric_limits<std::int32_t>::max() ||
                prefix.identity.token_types[frontier] != 0 ||
                std::any_of(prefix.identity.positions.begin(), prefix.identity.positions.end(),
                            [&](const auto& axis) {
                                return axis[frontier] !=
                                       static_cast<std::int32_t>(generated_position);
                            })) {
                return Failure::MetadataMismatch;
            }
        }

        ResidentPrefixIdentity resident_identity;
        resident_identity.restore(std::move(prefix.identity));
        const bool frontier_matches = qwen3_8::detail::prefix_matches(
            prompt, prefix.ledger, resident_identity, frontier);
        const bool boundary_matches =
            boundary.valid && boundary.frontier < prompt.token_ids.size() &&
            qwen3_8::detail::prefix_matches(
                prompt, prefix.ledger, resident_identity, boundary.frontier);
        std::uint32_t verified_depth = 0;
        if (frontier_matches) {
            verified_depth = frontier;
        } else if (boundary_matches) {
            verified_depth = boundary.frontier;
        }
        if (verified_depth != reusable_depth) {
            return Failure::VerifyDepthMismatch;
        }

        const bool mtp_backend    = speculative_backend == SpeculativeBackend::Mtp;
        const bool dflash_backend = speculative_backend == SpeculativeBackend::DFlash;
        if ((prefix_only && !metadata.mtp_drafts.empty()) ||
            (!mtp_backend && !metadata.mtp_drafts.empty()) ||
            metadata.mtp_drafts.size() > qwen3_8::kMtpDecodeMaximumDrafts ||
            ((!mtp_backend && !dflash_backend) != (metadata.backend_kv_valid == 0)) ||
            ((mtp_backend || dflash_backend) && metadata.backend_kv_valid != frontier)) {
            return Failure::MetadataMismatch;
        }

        const auto expected =
            continuation_segment_inventory(boundary.valid, mtp_backend, dflash_backend);
        std::set<std::string> actual;
        for (const auto& [name, unused] : candidate.segments) {
            (void)unused;
            actual.emplace(name);
        }
        if (actual != expected) { return Failure::SegmentInventoryMismatch; }

        // Reserve the shared paged-KV pages before decoding anything. Decoding materialises the
        // whole image on the host, so a reservation that cannot be satisfied must be discovered
        // first: paying a half-gigabyte of host copies only to fail is what turned KV pressure
        // into a silent cold prefill. The entitlement is the caller's planned allocation, not the
        // bare frontier, so the restored lane does not have to grow on its first decode round.
        const std::uint32_t frontier_pages =
            metadata.text_kv_valid == 0 ? 0U : pages_for_tokens(metadata.text_kv_valid);
        const std::uint32_t text_pages = std::max(frontier_pages, entitlement.text_pages);
        const std::uint32_t backend_pages =
            metadata.backend_kv_valid == 0
                ? 0U
                : std::max(pages_for_tokens(metadata.backend_kv_valid), entitlement.backend_pages);
        if (!kv_reservation_fits(text_pages, backend_pages)) {
            return Failure::KvReservationExhausted;
        }
        try {
            reserve_sequence_kv(sequence, text_pages, backend_pages);
        } catch (...) {
            return Failure::KvReservationExhausted;
        }

        // The payload was decoded by decode_continuation(), possibly on a preparation thread. It
        // is bound to this exact image by content identity at the caller; these checks confirm it
        // describes the state this lane is about to receive.
        const PagedKVLogicalImage& text_kv                    = decoded.text_kv;
        const LinearAttentionStateImage& current_gdn          = decoded.current_gdn;
        const cache::Bytes& tail_hidden                       = decoded.tail_hidden;
        const std::optional<LinearAttentionStateImage>& checkpoint_gdn = decoded.checkpoint_gdn;
        const std::optional<cache::Bytes>& checkpoint_hidden  = decoded.checkpoint_hidden;
        const std::optional<PagedKVLogicalImage>& backend_kv  = decoded.backend_kv;
        const std::optional<CyclicKVCacheImage>& dflash_local = decoded.dflash_local;
        const std::optional<CyclicKVCacheImage>& dflash_checkpoint_local =
            decoded.dflash_checkpoint_local;
        if (text_kv.valid_tokens != frontier || tail_hidden.size() != sequence.tail_hidden.bytes() ||
            checkpoint_gdn.has_value() != boundary.valid ||
            checkpoint_hidden.has_value() != boundary.valid ||
            backend_kv.has_value() != (mtp_backend || dflash_backend) ||
            dflash_local.has_value() != dflash_backend ||
            dflash_checkpoint_local.has_value() != (dflash_backend && boundary.valid) ||
            (checkpoint_hidden &&
             checkpoint_hidden->size() != sequence.turn_checkpoint_hidden.bytes())) {
            return Failure::MetadataMismatch;
        }
        if (backend_kv && backend_kv->valid_tokens != metadata.backend_kv_valid) {
            return Failure::MetadataMismatch;
        }

        import_paged_kv_logical(sequence.kv->text, text_kv, continuation_transfer, device.stream);
        if (backend_kv) {
            import_paged_kv_logical(*sequence.kv->backend, *backend_kv, continuation_transfer,
                                    device.stream);
        }
        import_linear_attention_state(
            decoder->linear_attention,
            LinearStateSlots::current_state_slot(sequence.lane, max_concurrency), current_gdn,
            continuation_transfer, device.stream);
        copy_tensor_to_device(sequence.tail_hidden, tail_hidden, continuation_transfer,
                              device.stream);
        // A prefix-only image is exactly the state a turn checkpoint at its frontier would hold,
        // so the lane adopts it as one. Without that, a restore landing exactly on the prompt's
        // rewrite frontier - the common case for a turn-opener boundary - would have no
        // checkpoint there and planning would reset it to a cold prefill.
        import_linear_attention_state(
            decoder->linear_attention,
            LinearStateSlots::turn_checkpoint_state_slot(sequence.lane, max_concurrency),
            boundary.valid ? *checkpoint_gdn : current_gdn, continuation_transfer, device.stream);
        copy_tensor_to_device(sequence.turn_checkpoint_hidden,
                              boundary.valid ? *checkpoint_hidden : tail_hidden,
                              continuation_transfer, device.stream);
        if (dflash_backend) {
            if (!dflash || !dflash_local) {
                throw std::invalid_argument("DFlash continuation state is incomplete");
            }
            import_cyclic_kv_lane(dflash->local, static_cast<std::int32_t>(sequence.lane),
                                   *dflash_local, continuation_transfer, device.stream);
            import_cyclic_kv_lane(dflash->turn_checkpoint_local,
                                  static_cast<std::int32_t>(sequence.lane),
                                  boundary.valid ? *dflash_checkpoint_local : *dflash_local,
                                  continuation_transfer, device.stream);
        }
        device.synchronize();

        sequence.execution_frontier      = frontier;
        sequence.ledger_frontier         = metadata.ledger_frontier;
        sequence.ledger                  = std::move(prefix.ledger);
        sequence.prefix_identity         = std::move(resident_identity);
        sequence.rope_delta              = metadata.rope_delta;
        sequence.text_kv_valid           = metadata.text_kv_valid;
        sequence.mtp_kv_valid            = mtp_backend ? metadata.backend_kv_valid : 0;
        sequence.dflash_context_frontier = dflash_backend ? metadata.backend_kv_valid : 0;
        sequence.mtp_draft_count = static_cast<std::uint32_t>(metadata.mtp_drafts.size());
        std::copy(metadata.mtp_drafts.begin(), metadata.mtp_drafts.end(),
                  sequence.mtp_drafts.begin());
        sequence.tail_hidden_valid = true;
        sequence.turn_checkpoint   = TurnCheckpoint{
              .valid = true, .frontier = boundary.valid ? boundary.frontier : frontier};
        // A continuation image carries no user-turn anchor. Leaving the previous occupant's
        // anchor in place would let prefix_matches accept it against the newly restored ledger
        // and splice in another conversation's recurrent state.
        sequence.user_turn_anchor = {};
        sequence.adapter          = adapter;
        sequence.retained         = true;
        request.lifecycle = Lifecycle::Complete;
        return Failure::None;
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        clear_lane(sequence, request);
        return Failure::DecodeFailed;
    }
}

GenerationTimings ProgramImplCore::generation_timings_lane(std::uint32_t lane) const noexcept {
    return lane < max_concurrency ? requests[lane].timings : GenerationTimings{};
}

SpeculativeStats ProgramImplCore::speculative_stats_lane(std::uint32_t lane) const noexcept {
    return lane < max_concurrency ? requests[lane].speculative_stats : SpeculativeStats{};
}

void ProgramImplCore::clear_lane(SequenceState& sequence, RequestControl& request) noexcept {
    request.prefill.reset();
    request.decision.reset();
    sequence.decision_state = false;
    sequence.kv.reset();
    request.lifecycle           = Lifecycle::Empty;
    // A cleared lane holds no adapter-dependent state, so it must stop pinning the slot it used.
    sequence.adapter            = -1;
    sequence.execution_frontier = 0;
    sequence.ledger_frontier    = 0;
    sequence.ledger.clear();
    sequence.prefix_identity.clear();
    sequence.text_kv_valid           = 0;
    sequence.mtp_kv_valid            = 0;
    sequence.dflash_context_frontier = 0;
    sequence.mtp_draft_count         = 0;
    sequence.tail_hidden_valid       = false;
    sequence.retained                = false;
    sequence.turn_checkpoint         = {};
    sequence.user_turn_anchor        = {};
    sequence.checkpoint_ring.clear();
    discard_checkpoint_staging(sequence);
    sequence.captured_continuations.clear();
    request.pending                  = {};
}

std::size_t ProgramImplCore::checkpoint_hidden_bytes() const noexcept {
    return sequences[0].turn_checkpoint_hidden.bytes();
}

std::size_t ProgramImplCore::checkpoint_conv_bytes() const noexcept {
    const LinearAttentionStatePool& states = decoder->linear_attention;
    return states.conv_slot(0, 0).bytes() * states.layer_count();
}

std::size_t ProgramImplCore::checkpoint_recurrent_bytes() const noexcept {
    const LinearAttentionStatePool& states = decoder->linear_attention;
    return states.recurrent_slot(0, 0).bytes() * states.layer_count();
}

std::size_t ProgramImplCore::checkpoint_entry_bytes() const noexcept {
    return checkpoint_hidden_bytes() + checkpoint_conv_bytes() + checkpoint_recurrent_bytes();
}

std::uint8_t* ProgramImplCore::checkpoint_staging_base(std::uint32_t lane) const noexcept {
    return static_cast<std::uint8_t*>(checkpoint_staging_store->data()) +
           checkpoint_entry_bytes() * lane;
}

// Enqueues a device-to-host copy of the freshly captured device checkpoint (checkpoint GDN
// slot + boundary hidden) into the lane's pinned staging entry. Runs right after prefill
// completes, when the stream is synchronized, so the copy overlaps the decode that follows;
// the recorded event gates the later drain into the pageable ring.
void ProgramImplCore::stage_turn_checkpoint(SequenceState& sequence) {
    if (checkpoint_ring_capacity == 0 || !sequence.turn_checkpoint.valid) { return; }
    const std::uint32_t lane     = sequence.lane;
    CheckpointStaging& staging   = checkpoint_staging[lane];
    staging.pending              = false;
    const std::uint32_t frontier = sequence.turn_checkpoint.frontier;
    if (frontier == 0 || frontier > sequence.ledger.size()) { return; }

    const LinearAttentionStatePool& states = decoder->linear_attention;
    const std::int32_t checkpoint_slot =
        LinearStateSlots::turn_checkpoint_state_slot(lane, max_concurrency);
    std::uint8_t* base       = checkpoint_staging_base(lane);
    std::uint8_t* conv       = base + checkpoint_hidden_bytes();
    std::uint8_t* recurrent  = conv + checkpoint_conv_bytes();
    CUDA_CHECK(cudaMemcpyAsync(base, sequence.turn_checkpoint_hidden.data,
                               checkpoint_hidden_bytes(), cudaMemcpyDeviceToHost, device.stream));
    for (std::uint32_t layer = 0; layer < states.layer_count(); ++layer) {
        const Tensor conv_state = states.conv_slot(layer, checkpoint_slot);
        CUDA_CHECK(cudaMemcpyAsync(conv + layer * conv_state.bytes(), conv_state.data,
                                   conv_state.bytes(), cudaMemcpyDeviceToHost, device.stream));
        const Tensor recurrent_state = states.recurrent_slot(layer, checkpoint_slot);
        CUDA_CHECK(cudaMemcpyAsync(recurrent + layer * recurrent_state.bytes(),
                                   recurrent_state.data, recurrent_state.bytes(),
                                   cudaMemcpyDeviceToHost, device.stream));
    }
    CUDA_CHECK(cudaEventRecord(checkpoint_staging_events[lane], device.stream));
    staging.frontier       = frontier;
    staging.session_digest = ledger_prefix_digest(
        std::span<const TokenId>(sequence.ledger.data(), frontier));
    staging.pending        = true;
}

// Folds the staged checkpoint into the lane's host ring once its copy has completed. Callers
// decide validity first: a staged frontier above the request's divergence point must be
// discarded, never drained.
void ProgramImplCore::drain_checkpoint_staging(SequenceState& sequence) {
    CheckpointStaging& staging = checkpoint_staging[sequence.lane];
    if (!staging.pending) { return; }
    staging.pending = false;
    CUDA_CHECK(cudaEventSynchronize(checkpoint_staging_events[sequence.lane]));

    HostTurnCheckpoint entry;
    entry.frontier       = staging.frontier;
    entry.session_digest = std::move(staging.session_digest);
    const std::uint8_t* base = checkpoint_staging_base(sequence.lane);
    entry.hidden.assign(base, base + checkpoint_hidden_bytes());
    base += checkpoint_hidden_bytes();
    entry.conv.assign(base, base + checkpoint_conv_bytes());
    base += checkpoint_conv_bytes();
    entry.recurrent.assign(base, base + checkpoint_recurrent_bytes());
    append_ring_checkpoint(sequence, std::move(entry));
}

void ProgramImplCore::discard_checkpoint_staging(SequenceState& sequence) noexcept {
    checkpoint_staging[sequence.lane].pending = false;
}

void ProgramImplCore::invalidate_checkpoint_ring(SequenceState& sequence,
                                                 std::uint32_t keep_through) noexcept {
    std::vector<HostTurnCheckpoint>& ring = sequence.checkpoint_ring;
    while (!ring.empty() && ring.back().frontier > keep_through) { ring.pop_back(); }
}

// Ring maintenance mirrors llama.cpp's context checkpoints: replace a same-frontier entry,
// fold entries that sit within the minimum step of a deeper neighbour, then drop the oldest
// until the configured capacity holds. The newest entry always survives compaction.
void ProgramImplCore::append_ring_checkpoint(SequenceState& sequence, HostTurnCheckpoint&& entry) {
    std::vector<HostTurnCheckpoint>& ring = sequence.checkpoint_ring;
    std::erase_if(ring, [&](const HostTurnCheckpoint& held) {
        return held.frontier == entry.frontier;
    });
    std::uint32_t previous_kept = 0;
    std::erase_if(ring, [&](const HostTurnCheckpoint& held) {
        if (previous_kept != 0 && held.frontier <= previous_kept + kTurnCheckpointMinStep) {
            return true;
        }
        previous_kept = held.frontier;
        return false;
    });
    while (ring.size() + 1 > checkpoint_ring_capacity) { ring.erase(ring.begin()); }
    ring.push_back(std::move(entry));
}

// Lands a host ring entry back in the lane's device checkpoint slot so the ordinary
// RestoreTurnCheckpoint path can proceed as if the checkpoint had stayed resident.
bool ProgramImplCore::upload_ring_checkpoint(SequenceState& sequence, std::uint32_t frontier) {
    if (checkpoint_ring_capacity == 0) { return false; }
    const auto entry = std::find_if(
        sequence.checkpoint_ring.begin(), sequence.checkpoint_ring.end(),
        [&](const HostTurnCheckpoint& held) { return held.frontier == frontier; });
    if (entry == sequence.checkpoint_ring.end() || entry->frontier == 0 ||
        entry->hidden.size() != checkpoint_hidden_bytes() ||
        entry->conv.size() != checkpoint_conv_bytes() ||
        entry->recurrent.size() != checkpoint_recurrent_bytes()) {
        return false;
    }

    LinearAttentionStatePool& states = decoder->linear_attention;
    const std::int32_t checkpoint_slot =
        LinearStateSlots::turn_checkpoint_state_slot(sequence.lane, max_concurrency);
    CUDA_CHECK(cudaMemcpyAsync(sequence.turn_checkpoint_hidden.data, entry->hidden.data(),
                               entry->hidden.size(), cudaMemcpyHostToDevice, device.stream));
    for (std::uint32_t layer = 0; layer < states.layer_count(); ++layer) {
        const Tensor conv_state = states.conv_slot(layer, checkpoint_slot);
        CUDA_CHECK(cudaMemcpyAsync(conv_state.data,
                                   entry->conv.data() + layer * conv_state.bytes(),
                                   conv_state.bytes(), cudaMemcpyHostToDevice, device.stream));
        const Tensor recurrent_state = states.recurrent_slot(layer, checkpoint_slot);
        CUDA_CHECK(cudaMemcpyAsync(recurrent_state.data,
                                   entry->recurrent.data() + layer * recurrent_state.bytes(),
                                   recurrent_state.bytes(), cudaMemcpyHostToDevice,
                                   device.stream));
    }
    sequence.turn_checkpoint = TurnCheckpoint{.valid = true, .frontier = frontier};
    return true;
}

qwen3_8::PagedKVCache* ProgramImplCore::backend_kv_cache() noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (speculative_backend == SpeculativeBackend::DFlash && dflash) { return &dflash->full; }
    return nullptr;
}

const qwen3_8::PagedKVCache* ProgramImplCore::backend_kv_cache() const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (speculative_backend == SpeculativeBackend::DFlash && dflash) { return &dflash->full; }
    return nullptr;
}

std::uint32_t ProgramImplCore::backend_kv_valid(const SequenceState& sequence) const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return sequence.mtp_kv_valid; }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        return sequence.dflash_context_frontier;
    }
    return 0;
}

void ProgramImplCore::reserve_sequence_kv(SequenceState& sequence, std::uint32_t text_pages,
                                          std::uint32_t backend_pages) {
    if (sequence.kv) { throw std::logic_error("sequence already owns a KV allocation bundle"); }
    // A decision lane holds Text KV only, so a backend reservation is optional; one that the
    // Program has no backend for is not.
    if (text_pages == 0 || (backend_kv_cache() == nullptr && backend_pages != 0)) {
        throw std::invalid_argument("KV allocation entitlement does not match the active backend");
    }

    std::array<PagedKVReservation, 2> reservations{};
    std::size_t count     = 0;
    reservations[count++] = PagedKVReservation{
        .pool             = &decoder->text_kv.pool(),
        .page_entitlement = text_pages,
    };
    if (qwen3_8::PagedKVCache* backend = backend_kv_cache();
        backend != nullptr && backend_pages != 0) {
        reservations[count++] = PagedKVReservation{
            .pool             = &backend->pool(),
            .page_entitlement = backend_pages,
        };
    }

    std::vector<PagedKVAllocation> allocations =
        reserve_paged_kv_bundle(std::span<const PagedKVReservation>(reservations.data(), count));
    SequenceKVBundle bundle;
    bundle.text = std::move(allocations[0]);
    if (count == 2) { bundle.backend.emplace(std::move(allocations[1])); }
    sequence.kv.emplace(std::move(bundle));
}

void ProgramImplCore::resize_sequence_kv_entitlement(SequenceState& sequence,
                                                     std::uint32_t text_pages,
                                                     std::uint32_t backend_pages) {
    if (!sequence.kv || text_pages == 0 ||
        (sequence.kv->backend.has_value() != (backend_pages != 0))) {
        throw std::invalid_argument("KV resize entitlement does not match the sequence bundle");
    }
    std::array<PagedKVResize, 2> changes{};
    std::size_t count = 0;
    changes[count++]  = PagedKVResize{
         .allocation       = &sequence.kv->text,
         .mapped_pages     = sequence.kv->text.mapped_page_count(),
         .page_entitlement = text_pages,
    };
    if (sequence.kv->backend) {
        changes[count++] = PagedKVResize{
            .allocation       = &*sequence.kv->backend,
            .mapped_pages     = sequence.kv->backend->mapped_page_count(),
            .page_entitlement = backend_pages,
        };
    }
    resize_paged_kv_bundle(std::span<PagedKVResize>(changes.data(), count));
}

void ProgramImplCore::bind_sequence_kv(SequenceState& sequence) {
    if (!sequence.kv || sequence.kv->text.bound_row() >= 0 ||
        (sequence.kv->backend && sequence.kv->backend->bound_row() >= 0)) {
        throw std::logic_error("KV allocation bundle is unavailable or already bound");
    }
    const std::int32_t row = static_cast<std::int32_t>(sequence.lane);
    sequence.kv->text.bind_row(row, device.stream);
    try {
        if (sequence.kv->backend) { sequence.kv->backend->bind_row(row, device.stream); }
        set_device_i32(io.text_kv_table_row, sequence.kv->text.bound_row());
        set_device_i32(io.backend_kv_table_row,
                       sequence.kv->backend ? sequence.kv->backend->bound_row() : 0);
    } catch (...) {
        if (sequence.kv->backend && sequence.kv->backend->bound_row() >= 0) {
            sequence.kv->backend->unbind_row();
        }
        sequence.kv->text.unbind_row();
        throw;
    }
}

void ProgramImplCore::unbind_sequence_kv(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    if (sequence.kv->backend) { sequence.kv->backend->unbind_row(); }
    sequence.kv->text.unbind_row();
}

void ProgramImplCore::select_prefill_kv_rows(const SequenceState& sequence) {
    if (!sequence.kv || sequence.kv->text.bound_row() < 0 ||
        (sequence.kv->backend && sequence.kv->backend->bound_row() < 0)) {
        throw std::logic_error("prefill lane holds no bound KV row");
    }
    set_device_i32(io.text_kv_table_row, sequence.kv->text.bound_row());
    set_device_i32(io.backend_kv_table_row,
                   sequence.kv->backend ? sequence.kv->backend->bound_row() : 0);
    if (speculative_backend == SpeculativeBackend::DFlash && sequence.kv->backend) {
        if (!io.dflash_decode) { throw std::logic_error("DFlash prefill controls are unavailable"); }
        *dflash_host_ingress                         = {};
        dflash_host_ingress->lanes[0]                = static_cast<std::int32_t>(sequence.lane);
        dflash_host_ingress->dflash_kv_table_rows[0] = sequence.kv->backend->bound_row();
        CUDA_CHECK(cudaMemcpyAsync(io.dflash_decode->ingress.data, dflash_host_ingress,
                                   sizeof(qwen3_8::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                   device.stream));
    }
}

void ProgramImplCore::materialize_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                              std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > capacity) {
        throw std::logic_error("KV materialization request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV materialization requested without an allocation");
    }
    if (main_tokens > sequence.kv->text.mapped_token_capacity()) {
        sequence.kv->text.materialize_tokens(main_tokens, device.stream);
    }
    if (backend_tokens != 0 && backend_tokens > sequence.kv->backend->mapped_token_capacity()) {
        sequence.kv->backend->materialize_tokens(backend_tokens, device.stream);
    }
}

void ProgramImplCore::trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                       std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > main_tokens) {
        throw std::logic_error("KV trim request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV trim requested without an allocation");
    }
    sequence.kv->text.trim_tokens(main_tokens);
    if (sequence.kv->backend) { sequence.kv->backend->trim_tokens(backend_tokens); }
}

void ProgramImplCore::release_sequence_growth_entitlement(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    sequence.kv->text.cancel_unmapped_entitlement();
    if (sequence.kv->backend) { sequence.kv->backend->cancel_unmapped_entitlement(); }
}

qwen3_8::PagedKVCacheView ProgramImplCore::text_kv_view(const SequenceState& sequence) const {
    if (!sequence.kv) { throw std::logic_error("sequence has no KV allocation bundle"); }
    return decoder->text_kv.execution_view(sequence.kv->text);
}

qwen3_8::PagedKVCacheView ProgramImplCore::mtp_kv_view(const SequenceState& sequence) const {
    if (speculative_backend != SpeculativeBackend::Mtp) { return {}; }
    if (decoder->mtp_cache() == nullptr || !sequence.kv || !sequence.kv->backend) {
        throw std::logic_error("sequence has no MTP KV allocation");
    }
    return decoder->mtp_cache()->execution_view(*sequence.kv->backend);
}

void ProgramImplCore::set_device_i32(Tensor& tensor, std::int32_t value) {
    CUDA_CHECK(
        cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice, device.stream));
}

void ProgramImplCore::ordered_reset(SequenceState& sequence) {
    decoder->linear_attention.zero_slot(
        LinearStateSlots::current_state_slot(sequence.lane, max_concurrency), device.stream);
    work.reset();
    set_device_i32(io.pos, 0);
    set_device_i32(io.rope_pos, 0);
    set_device_i32(io.rope_delta, 0);
    if (io.mtp) { set_device_i32(io.mtp->position, 0); }
    sequence.text_kv_valid           = 0;
    sequence.mtp_kv_valid            = 0;
    sequence.dflash_context_frontier = 0;
}

void ProgramImplCore::prepare_graphs() {
    if (!use_cuda_graph) { return; }
    SequenceState& sequence = sequences[0];

    std::vector<PagedKVAllocation> text_capture_allocations;
    std::vector<PagedKVAllocation> mtp_capture_allocations;
    std::vector<PagedKVAllocation> dflash_capture_allocations;
    const auto reserve_capture_rows = [&](qwen3_8::PagedKVCache& cache,
                                          std::vector<PagedKVAllocation>& allocations,
                                          const char* label) {
        PagedKVPool& pool = cache.pool();
        if (pool.page_group_count() < max_concurrency) {
            throw std::invalid_argument(std::string(label) +
                                        " cannot provide one Paged KV page per concurrent request");
        }
        allocations.reserve(max_concurrency);
        for (std::uint32_t row = 0; row < max_concurrency; ++row) {
            allocations.push_back(pool.reserve(1));
            PagedKVAllocation& allocation = allocations.back();
            allocation.bind_row(static_cast<std::int32_t>(row), device.stream);
            allocation.materialize_pages(1, device.stream);

            // Capture profiles exercise arbitrary context envelopes. Repeating each row's private
            // page across its temporary table keeps every dummy read/write address valid without
            // reserving C full contexts solely for graph construction.
            const std::int32_t page = allocation.page_ids().front();
            std::vector<std::int32_t> repeated(pool.logical_page_capacity(), page);
            Tensor table = pool.block_table_row(static_cast<std::int32_t>(row));
            CUDA_CHECK(cudaMemcpyAsync(table.data, repeated.data(), table.bytes(),
                                       cudaMemcpyHostToDevice, device.stream));
        }
    };
    reserve_capture_rows(decoder->text_kv, text_capture_allocations, "target KV cache");
    if (speculative_backend == SpeculativeBackend::Mtp) {
        reserve_capture_rows(*decoder->mtp_cache(), mtp_capture_allocations, "MTP KV cache");
    } else if (speculative_backend == SpeculativeBackend::DFlash) {
        reserve_capture_rows(dflash->full, dflash_capture_allocations, "DFlash Full KV cache");
    }
    device.synchronize();

    std::size_t free_before = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_bytes));

    const auto clear_stable_controls = [&] {
        std::vector<Tensor> controls{
            io.token,
            io.pos,
            io.rope_pos,
            io.rope_delta,
        };
        if (io.mtp) {
            controls.push_back(io.mtp->position);
            controls.push_back(io.mtp->draft_tokens);
            controls.push_back(io.mtp->target_input_ids);
            controls.push_back(io.mtp->target_positions);
        }
        if (io.dflash_prefill) { controls.push_back(io.dflash_prefill->produced_count); }
        for (const Tensor& tensor : controls) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
    };
    const auto zero_capture_pages = [&](qwen3_8::PagedKVCache& cache,
                                        const std::vector<PagedKVAllocation>& allocations,
                                        std::uint32_t batch_size) {
        std::vector<std::int32_t> pages;
        pages.reserve(batch_size);
        for (std::uint32_t row = 0; row < batch_size; ++row) {
            pages.push_back(allocations[row].page_ids().front());
        }
        cache.pool().zero_pages(pages, device.stream);
    };
    const auto zero_cyclic_lane = [&](CyclicKVCache& cache, std::uint32_t lane) {
        for (std::uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = cache.layer_view(layer);
            const Tensor k                    = view.k.slice(3, static_cast<std::int32_t>(lane), 1);
            const Tensor v                    = view.v.slice(3, static_cast<std::int32_t>(lane), 1);
            CUDA_CHECK(cudaMemsetAsync(k.data, 0, k.bytes(), device.stream));
            CUDA_CHECK(cudaMemsetAsync(v.data, 0, v.bytes(), device.stream));
        }
    };

    const auto prepare_representative = [&](std::uint32_t frontier, std::uint32_t batch_size) {
        if (batch_size == 0 || batch_size > max_concurrency) {
            throw std::logic_error("CUDA Graph representative batch is invalid");
        }
        work.reset();
        clear_stable_controls();
        zero_capture_pages(decoder->text_kv, text_capture_allocations, batch_size);
        if (decoder->mtp_cache() != nullptr) {
            zero_capture_pages(*decoder->mtp_cache(), mtp_capture_allocations, batch_size);
        }
        if (dflash) { zero_capture_pages(dflash->full, dflash_capture_allocations, batch_size); }
        for (std::uint32_t row = 0; row < batch_size; ++row) {
            decoder->linear_attention.zero_slot(
                LinearStateSlots::current_state_slot(row, max_concurrency), device.stream);
            if (dflash) {
                zero_cyclic_lane(dflash->local, row);
                const Tensor pending =
                    dflash->pending_features.slice(2, static_cast<std::int32_t>(row), 1);
                CUDA_CHECK(cudaMemsetAsync(pending.data, 0, pending.bytes(), device.stream));
            }
        }
        set_device_i32(io.pos, checked_i32(frontier, "graph representative position"));
        set_device_i32(io.rope_pos, checked_i32(frontier, "graph representative rope position"));
        if (io.mtp) {
            set_device_i32(io.mtp->position,
                           checked_i32(frontier, "graph representative MTP position"));
        }
        if (io.dflash_decode) {
            *dflash_host_ingress       = {};
            *dflash_host_egress        = {};
            const std::uint32_t extent = std::min(draft_window, capacity - frontier - 1U);
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                dflash_host_ingress->anchors[row] = 0;
                dflash_host_ingress->execution_frontiers[row] =
                    checked_i32(frontier, "graph representative DFlash frontier");
                dflash_host_ingress->context_frontiers[row] =
                    checked_i32(frontier, "graph representative DFlash context frontier");
                dflash_host_ingress->proposal_extents[row] = static_cast<std::int32_t>(extent);
                dflash_host_ingress->target_valid_columns[row] =
                    static_cast<std::int32_t>(extent + 1U);
                dflash_host_ingress->text_kv_table_rows[row]   = static_cast<std::int32_t>(row);
                dflash_host_ingress->dflash_kv_table_rows[row] = static_cast<std::int32_t>(row);
                dflash_host_ingress->lanes[row]                = static_cast<std::int32_t>(row);
                dflash_host_ingress->adapters[row]             = -1;
                dflash_host_ingress->sampling[row]             = {};
            }
        }
        if (io.mtp_decode) {
            *mtp_host_ingress          = {};
            *mtp_host_egress           = {};
            const std::uint32_t extent = std::min(draft_window, capacity - frontier - 1U);
            const std::uint32_t width  = draft_window + 1U;
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                mtp_host_ingress->anchors[row] = 0;
                mtp_host_ingress->base_frontiers[row] =
                    checked_i32(frontier, "graph representative MTP frontier");
                mtp_host_ingress->remaining_budgets[row] =
                    checked_i32(capacity, "graph representative MTP budget");
                mtp_host_ingress->current_extents[row] = static_cast<std::int32_t>(extent);
                mtp_host_ingress->target_valid_columns[row] =
                    static_cast<std::int32_t>(extent + 1U);
                for (std::uint32_t step = 0; step < draft_window; ++step) {
                    mtp_host_ingress->current_drafts[row * draft_window + step] = 0;
                }
                for (std::uint32_t column = 0; column < width; ++column) {
                    mtp_host_ingress->target_rope_positions[row * width + column] =
                        checked_i32(frontier + std::min(column, extent),
                                    "graph representative MTP RoPE position");
                }
                mtp_host_ingress->text_kv_table_rows[row] = static_cast<std::int32_t>(row);
                mtp_host_ingress->mtp_kv_table_rows[row]  = static_cast<std::int32_t>(row);
                mtp_host_ingress->lanes[row]              = static_cast<std::int32_t>(row);
                mtp_host_ingress->adapters[row]           = -1;
                mtp_host_ingress->rope_deltas[row]        = 0;
                mtp_host_ingress->sampling[row]           = {};
            }
        }
        if (io.ordinary) {
            *ordinary_host_ingress = {};
            *ordinary_host_egress  = {};
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                ordinary_host_ingress->tokens[row] = 0;
                ordinary_host_ingress->cache_positions[row] =
                    checked_i32(frontier, "graph representative ordinary position");
                ordinary_host_ingress->rope_positions[row] =
                    checked_i32(frontier, "graph representative ordinary RoPE position");
                ordinary_host_ingress->text_kv_table_rows[row] = static_cast<std::int32_t>(row);
                ordinary_host_ingress->lanes[row]              = static_cast<std::int32_t>(row);
                ordinary_host_ingress->adapters[row]           = -1;
                ordinary_host_ingress->sampling[row]           = {};
            }
        }
    };
    const auto execution_core = [&] {
        return schedule::ExecutionCore{device,
                                       model,
                                       work,
                                       decoder->linear_attention,
                                       replay_records ? &*replay_records : nullptr,
                                       io,
                                       prefill_hidden,
                                       prefill_chunk,
                                       proposal_head};
    };

    if (speculative_backend == SpeculativeBackend::None) {
        const auto ordinary_profiles = ordinary_graph_profiles(capacity);
        validate_graph_profiles(ordinary_profiles, capacity - 1, "ordinary");
        const std::uint32_t ordinary_batch_limit = max_concurrency;
        schedule::OrdinaryBatchContext ordinary_state{execution_core(),      decoder->text_kv,
                                                      *io.ordinary,          *ordinary_host_ingress,
                                                      *ordinary_host_egress, tail_hidden_store};
        const GraphExecutionProfile code_warm = ordinary_profiles.front();
        prepare_representative(code_warm.min, 1);
        device.synchronize();
        schedule::ordinary_decode_batch(ordinary_state, 1, {code_warm.min + 1, code_warm.max + 1},
                                        nullptr);
        device.synchronize();

        ordinary_graphs.profiles.reserve(ordinary_profiles.size() * ordinary_batch_limit);
        for (std::uint32_t batch_size = 1; batch_size <= ordinary_batch_limit; ++batch_size) {
            for (const GraphExecutionProfile planned : ordinary_profiles) {
                ordinary_graphs.profiles.emplace_back();
                DecodeGraphProfile& profile    = ordinary_graphs.profiles.back();
                profile.batch_size             = batch_size;
                profile.min_execution_frontier = planned.min;
                profile.max_execution_frontier = planned.max;
                profile.topology_class =
                    planned.topology_class * ordinary_batch_limit + (batch_size - 1U);
                const ops::GqaExecutionEnvelope envelope{planned.min + 1, planned.max + 1};
                schedule::capture_ordinary_decode_batch(ordinary_state,
                                                        static_cast<std::int32_t>(batch_size),
                                                        envelope, profile.definition);
            }
        }
    }

    if (speculative_backend == SpeculativeBackend::Mtp) {
        const auto planned_profiles = mtp_graph_profiles(capacity, draft_window);
        validate_graph_profiles(planned_profiles, capacity - 1, "MTP");
        schedule::MtpBatchContext mtp_state{
            execution_core(),  decoder->text_kv, *decoder->mtp_cache(), *io.mtp_decode,
            *mtp_host_ingress, *mtp_host_egress, tail_hidden_store};
        const GraphExecutionProfile code_warm = planned_profiles.front();
        prepare_representative(code_warm.min, 1);
        device.synchronize();
        schedule::mtp_decode_batch(mtp_state, 1, draft_window,
                                   mtp_gqa_envelopes(code_warm.max, draft_window, capacity),
                                   nullptr);
        device.synchronize();

        mtp_graphs.profiles.reserve(planned_profiles.size() * max_concurrency);
        for (std::uint32_t batch_size = 1; batch_size <= max_concurrency; ++batch_size) {
            for (const GraphExecutionProfile planned : planned_profiles) {
                mtp_graphs.profiles.emplace_back();
                DecodeGraphProfile& profile    = mtp_graphs.profiles.back();
                profile.batch_size             = batch_size;
                profile.min_execution_frontier = planned.min;
                profile.max_execution_frontier = planned.max;
                profile.topology_class =
                    planned.topology_class * max_concurrency + (batch_size - 1U);
                schedule::capture_mtp_decode_batch(
                    mtp_state, static_cast<std::int32_t>(batch_size), draft_window,
                    mtp_gqa_envelopes(planned.max, draft_window, capacity), profile.definition);
            }
        }
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        const auto batch_one_profiles = dflash_graph_profiles(capacity, draft_window, 1);
        validate_graph_profiles(batch_one_profiles, capacity - 1, "DFlash");
        schedule::DFlashBatchContext dflash_state{
            execution_core(),     decoder->text_kv,    *dflash,          *io.dflash_decode,
            *dflash_host_ingress, *dflash_host_egress, tail_hidden_store};
        const GraphExecutionProfile code_warm = batch_one_profiles.front();
        const ops::GqaExecutionEnvelope code_warm_target{
            1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                   capacity, static_cast<std::uint64_t>(code_warm.max) + draft_window + 1ULL))};
        prepare_representative(code_warm.min, 1);
        device.synchronize();
        schedule::dflash_decode_batch(dflash_state, 1, draft_window,
                                      dflash_envelopes(code_warm.min, code_warm.max, draft_window),
                                      code_warm_target, nullptr);
        device.synchronize();

        dflash_graphs.profiles.reserve(batch_one_profiles.size() * max_concurrency);
        for (std::uint32_t batch_size = 1; batch_size <= max_concurrency; ++batch_size) {
            const auto planned_profiles =
                batch_size == 1 ? batch_one_profiles
                                : dflash_graph_profiles(capacity, draft_window, batch_size);
            validate_graph_profiles(planned_profiles, capacity - 1, "DFlash");
            for (const GraphExecutionProfile planned : planned_profiles) {
                dflash_graphs.profiles.emplace_back();
                DecodeGraphProfile& profile    = dflash_graphs.profiles.back();
                profile.batch_size             = batch_size;
                profile.min_execution_frontier = planned.min;
                profile.max_execution_frontier = planned.max;
                profile.topology_class =
                    planned.topology_class * max_concurrency + (batch_size - 1U);
                const ops::GqaExecutionEnvelope target_envelope{
                    1,
                    static_cast<std::uint32_t>(std::min<std::uint64_t>(
                        capacity, static_cast<std::uint64_t>(planned.max) + draft_window + 1ULL))};

                schedule::capture_dflash_decode_batch(
                    dflash_state, static_cast<std::int32_t>(batch_size), draft_window,
                    dflash_envelopes(planned.min, planned.max, draft_window), target_envelope,
                    profile.definition);
            }
        }
    }

    if (!ordinary_graphs.profiles.empty()) {
        instantiate_graph_family(ordinary_graphs, "ordinary", device, prepare_representative);
    }
    if (speculative_backend == SpeculativeBackend::Mtp) {
        instantiate_graph_family(mtp_graphs, "MTP", device, prepare_representative);
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        instantiate_graph_family(dflash_graphs, "DFlash", device, prepare_representative);
    }

    ordered_reset(sequence);
    clear_stable_controls();
    for (Tensor& tensor : decoder->linear_attention.conv) {
        CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
    }
    for (Tensor& tensor : decoder->linear_attention.recurrent) {
        CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
    }
    if (dflash) {
        const auto zero_cyclic_cache = [&](CyclicKVCache& cache) {
            for (std::uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
                const CyclicKVCacheLayerView view = cache.layer_view(layer);
                CUDA_CHECK(cudaMemsetAsync(view.k.data, 0, view.k.bytes(), device.stream));
                CUDA_CHECK(cudaMemsetAsync(view.v.data, 0, view.v.bytes(), device.stream));
            }
        };
        zero_cyclic_cache(dflash->local);
        zero_cyclic_cache(dflash->turn_checkpoint_local);
        CUDA_CHECK(cudaMemsetAsync(dflash->prefill_features.data, 0,
                                   dflash->prefill_features.bytes(), device.stream));
        CUDA_CHECK(cudaMemsetAsync(dflash->prefill_positions.data, 0,
                                   dflash->prefill_positions.bytes(), device.stream));
        CUDA_CHECK(cudaMemsetAsync(dflash->pending_features.data, 0,
                                   dflash->pending_features.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    device.synchronize();

    std::size_t free_after = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_bytes));
    const std::size_t consumed = free_before > free_after ? free_before - free_after : 0;
    graph_observed_bytes       = consumed;
    if (consumed > graph_allowance_bytes) {
        throw std::runtime_error("CUDA Graph preparation consumed " + std::to_string(consumed) +
                                 " bytes, exceeding the planned allowance of " +
                                 std::to_string(graph_allowance_bytes) + " bytes");
    }
    for (PagedKVAllocation& allocation : dflash_capture_allocations) { allocation.unbind_row(); }
    dflash_capture_allocations.clear();
    for (PagedKVAllocation& allocation : mtp_capture_allocations) { allocation.unbind_row(); }
    mtp_capture_allocations.clear();
    for (PagedKVAllocation& allocation : text_capture_allocations) { allocation.unbind_row(); }
    text_capture_allocations.clear();
}

void ProgramImplCore::install_sampling(SequenceState& sequence, RequestControl& request,
                                       const ops::SamplingConfig& config) {
    Tensor counts = token_counts.slice(1, static_cast<std::int32_t>(sequence.lane), 1)
                        .view({TextConfig::token_domain});
    CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), device.stream));
    request.sampling_host     = config;
    request.speculative_stats = SpeculativeStats{
        .backend               = speculative_backend,
        .enabled               = speculative_backend != SpeculativeBackend::None,
        .draft_window          = draft_window,
        .accepted_per_position = std::vector<std::uint64_t>(draft_window, 0),
    };
    const bool penalties = request.sampling_host.presence_penalty != 0.0F ||
                           request.sampling_host.frequency_penalty != 0.0F;
    request.sampling_host.token_counts =
        penalties ? static_cast<std::int32_t*>(counts.data) : nullptr;
    Tensor config_lane = sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1);
    CUDA_CHECK(cudaMemcpyAsync(config_lane.data, &request.sampling_host,
                               sizeof(request.sampling_host), cudaMemcpyHostToDevice,
                               device.stream));
}

void ProgramImplCore::copy_tail(SequenceState& sequence, const Tensor& source) {
    if (source.dtype != DType::BF16 || source.ne[0] != TextConfig::hidden || source.ne[1] != 1) {
        throw std::logic_error("target tail hidden has an invalid shape");
    }
    CUDA_CHECK(cudaMemcpyAsync(sequence.tail_hidden.data, source.data, sequence.tail_hidden.bytes(),
                               cudaMemcpyDeviceToDevice, device.stream));
    sequence.tail_hidden_valid = true;
}

void ProgramImplCore::copy_round_token() {
    CUDA_CHECK(cudaMemcpyAsync(host_tokens, io.token.data, sizeof(TokenId), cudaMemcpyDeviceToHost,
                               device.stream));
}

void ProgramImplCore::mark_workspace_usage(std::size_t phase_bytes) noexcept {
    workspace_logical_peak_bytes = std::max(workspace_logical_peak_bytes, phase_bytes);
}

void ProgramImplCore::enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                                    std::span<const std::uint32_t> starts,
                                                    std::span<const std::uint32_t> counts) {
    if (speculative_backend != SpeculativeBackend::DFlash || !dflash || !io.dflash_decode ||
        lanes.empty() || lanes.size() > max_concurrency || starts.size() != lanes.size() ||
        counts.size() != lanes.size()) {
        throw std::logic_error("DFlash context append has invalid membership");
    }

    std::uint32_t minimum_count = draft_window + 1U;
    std::uint32_t maximum_count = 0;
    *dflash_host_ingress        = {};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || counts[row] == 0 || counts[row] > draft_window + 1U ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::logic_error("DFlash context append contains an invalid row");
        }
        SequenceState& sequence   = sequences[lane];
        const std::uint32_t start = starts[row];
        const std::uint64_t end64 = static_cast<std::uint64_t>(start) + counts[row];
        const std::uint32_t end   = static_cast<std::uint32_t>(end64);
        if (!sequence.kv || !sequence.kv->backend || sequence.kv->text.bound_row() < 0 ||
            sequence.kv->backend->bound_row() < 0 || end64 > capacity) {
            throw std::logic_error("DFlash context append is outside retained target storage");
        }
        dflash_host_ingress->context_frontiers[row] =
            checked_i32(start, "DFlash append context frontier");
        dflash_host_ingress->execution_frontiers[row] =
            checked_i32(end, "DFlash append target frontier");
        dflash_host_ingress->dflash_kv_table_rows[row] = sequence.kv->backend->bound_row();
        dflash_host_ingress->lanes[row]                = static_cast<std::int32_t>(lane);
        materialize_sequence_kv(sequence, std::max(sequence.text_kv_valid, end), end);
        minimum_count = std::min(minimum_count, counts[row]);
        maximum_count = std::max(maximum_count, counts[row]);
    }

    qwen3_8::DFlashDecodeState& frame = *io.dflash_decode;
    CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, dflash_host_ingress,
                               sizeof(qwen3_8::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                               device.stream));
    const auto batch     = static_cast<std::int32_t>(lanes.size());
    Tensor lane_tensor   = frame.lanes.slice(0, 0, batch);
    Tensor device_starts = frame.context_frontiers.slice(0, 0, batch);
    Tensor device_ends   = frame.execution_frontiers.slice(0, 0, batch);
    Tensor table_rows    = frame.dflash_kv_table_rows.slice(0, 0, batch);
    Tensor positions     = frame.append_positions.slice(1, 0, batch);
    Tensor device_counts = frame.append_counts.slice(0, 0, batch);

    work.reset();
    Tensor features =
        work.alloc(DType::BF16, {DFlashConfig::feature_rows,
                                 static_cast<std::int32_t>(draft_window + 1U), batch});
    ops::prepare_ragged_prefix(dflash->pending_features, lane_tensor, device_starts, device_ends,
                               features, positions, device_counts, device.stream);

    schedule::DFlashAppendContext state{{device, model, work, decoder->linear_attention,
                                         replay_records ? &*replay_records : nullptr, io,
                                         prefill_hidden, prefill_chunk, proposal_head},
                                        *dflash};
    mark_workspace_usage(workspace_plan.dflash_context);
    schedule::dflash_append_context(state, features, positions, device_counts, lane_tensor,
                                    table_rows, {minimum_count, maximum_count});
}

void ProgramImplCore::validate_licensed_tokens(std::span<const TokenId> tokens) const {
    for (const TokenId token : tokens) {
        if (token < 0 || token >= TextConfig::token_domain) {
            throw std::runtime_error("target returned a token outside the 248077-token domain");
        }
    }
}

runtime::PrefillStepResult ProgramImplCore::advance_prefill(SequenceState& sequence,
                                                            RequestControl& request) {
    if (request.lifecycle != Lifecycle::Prefilling || !request.prefill) {
        throw std::logic_error("staged prefill step requires an active concurrent request");
    }

    RequestControl::Prefill& staged = *request.prefill;
    const runtime::BeginSummary summary{.prompt_tokens        = staged.prompt_tokens,
                                        .reused_prompt_tokens = staged.base,
                                        .prefix_reuse_path    = staged.reuse};
    bool host_input_consumed              = staged.host_input_consumed_pending;
    staged.host_input_consumed_pending    = false;
    std::uint32_t processed_prompt_tokens = 0;
    const auto started                    = Clock::now();
    try {
        select_prefill_kv_rows(sequence);
        schedule::PrefillContext schedule_state{
            {device, model, work, decoder->linear_attention,
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head},
            text_kv_view(sequence),
            mtp_kv_view(sequence),
            decoder->text_kv,
            decoder->mtp_cache(),
            dflash ? &*dflash : nullptr,
            staged.cursor,
            static_cast<const ops::SamplingConfig*>(
                sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1).data),
            &sequence.turn_checkpoint_hidden,
            LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
            LinearStateSlots::turn_checkpoint_state_slot(sequence.lane, max_concurrency),
            staged.initial_mtp_extent,
            dflash_host_ingress,
            lora_slot(sequence.adapter)};

        if (staged.mtp_bridge == MtpBridgeMode::BeforeSuffix) {
            if (staged.cursor != staged.base || staged.base == 0 ||
                staged.cursor >= staged.prompt_tokens) {
                throw std::logic_error("staged MTP bridge is outside the reusable suffix");
            }
            mark_workspace_usage(workspace_plan.mtp_prefill);
            const Tensor& previous_hidden =
                staged.reuse == ReusePath::RestoreTurnCheckpoint ||
                        staged.reuse == ReusePath::RestoreUserTurnAnchor
                    ? sequence.turn_checkpoint_hidden
                    : sequence.tail_hidden;
            const schedule::MtpBridgeInput bridge{
                .previous_hidden = &previous_hidden,
                .position        = checked_i32(staged.base - 1, "MTP bridge position"),
                .rope_position   = prompt_rope_position(staged.prompt, staged.base - 1),
            };
            if (staged.vision) {
                schedule::mtp_bridge_multimodal(schedule_state, staged.prompt, *staged.vision,
                                                bridge);
            } else {
                Tensor bridge_token = io.mtp->target_input_ids.slice(0, 0, 1);
                const TokenId token = staged.prompt.token_ids[staged.base];
                CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token),
                                           cudaMemcpyHostToDevice, device.stream));
                schedule::mtp_bridge_and_propose(schedule_state, bridge_token, previous_hidden,
                                                 bridge.position, bridge.rope_position, false);
            }
            sequence.mtp_kv_valid = staged.base;
            staged.mtp_bridge     = MtpBridgeMode::None;
        }

        if (staged.cursor < staged.prompt_tokens) {
            const std::uint32_t nominal =
                std::min(prefill_chunk, staged.prompt_tokens - staged.cursor);
            const bool final_candidate = staged.cursor + nominal == staged.prompt_tokens;
            mark_workspace_usage(staged.prepare_mtp ? workspace_plan.mtp_prefill
                                                    : workspace_plan.text_prefill);
            if (speculative_backend == SpeculativeBackend::DFlash) {
                mark_workspace_usage(workspace_plan.dflash_context);
            }
            schedule::PrefillChunkResult result;
            const std::optional<std::uint32_t> capture_frontier =
                image::next_prefill_checkpoint(staged.cursor, staged.capture_frontiers,
                                               staged.turn_checkpoint_capture_frontier,
                                               staged.user_turn_capture_frontier);
            if (staged.vision) {
                mark_workspace_usage(workspace_plan.vision_encode);
                result = schedule::prefill_multimodal_chunk(
                    schedule_state, staged.prompt, *staged.vision, nominal,
                    capture_frontier, final_candidate);
            } else {
                result = schedule::prefill_text_chunk(
                    schedule_state, std::span<const TokenId>(staged.prompt.token_ids), nominal,
                    capture_frontier, final_candidate);
            }
            if (result.processed_tokens == 0 || result.processed_tokens > nominal) {
                throw std::logic_error("ordinary prefill chunk made invalid progress");
            }
            processed_prompt_tokens = result.processed_tokens;
            if (staged.vision && staged.vision->release_consumed_media_payload()) {
                host_input_consumed = true;
            }
            staged.cursor += result.processed_tokens;
            sequence.text_kv_valid = staged.cursor;
            if (staged.prepare_mtp) { sequence.mtp_kv_valid = staged.cursor; }
            if (speculative_backend == SpeculativeBackend::DFlash) {
                sequence.dflash_context_frontier = staged.cursor;
            }

            if (staged.user_turn_capture_frontier &&
                staged.cursor == *staged.user_turn_capture_frontier) {
                UserTurnAnchor anchor;
                anchor.linear_state = export_linear_attention_state(
                    decoder->linear_attention,
                    LinearStateSlots::turn_checkpoint_state_slot(sequence.lane, max_concurrency),
                    continuation_transfer, device.stream);
                anchor.tail_hidden = copy_tensor_to_host(sequence.turn_checkpoint_hidden,
                                                         continuation_transfer, device.stream);
                anchor.frontier          = staged.cursor;
                anchor.valid             = true;
                sequence.user_turn_anchor = std::move(anchor);
                staged.user_turn_capture_frontier.reset();
            }

            if (!staged.capture_frontiers.empty() &&
                staged.cursor == staged.capture_frontiers.front()) {
                sequence.captured_continuations.push_back(CapturedContinuation{
                    .depth = staged.cursor,
                    .image = export_stable_continuation(sequence, staged.prompt, staged.cursor)});
                staged.capture_frontiers.erase(staged.capture_frontiers.begin());
            }

            if (!result.finalized) {
                if (staged.cursor == staged.prompt_tokens) {
                    throw std::logic_error("staged prefill reached the prompt without sampling");
                }
                staged.elapsed_seconds +=
                    std::chrono::duration<double>(Clock::now() - started).count();
                return runtime::PrefillStepResult{.summary = summary,
                                                  .processed_prompt_tokens =
                                                      processed_prompt_tokens,
                                                  .host_input_consumed = host_input_consumed};
            }
            if (staged.cursor != staged.prompt_tokens) {
                throw std::logic_error("staged prefill sampled before the prompt frontier");
            }
            copy_tail(sequence, prefill_hidden.slice(
                                    1, static_cast<std::int32_t>(result.processed_tokens) - 1, 1));
        } else {
            mark_workspace_usage(workspace_plan.ordinary_round);
            if (!sequence.tail_hidden_valid) {
                throw std::logic_error("zero-suffix reuse has no target tail hidden");
            }
            schedule::sample_from_hidden(schedule_state, sequence.tail_hidden,
                                         checked_i32(staged.prompt_tokens, "sample position"),
                                         ops::kSamplePurposePrefill);
            set_device_i32(io.rope_pos, checked_i32(staged.prompt_tokens, "rope position") +
                                            sequence.rope_delta);
            if (staged.prepare_mtp) {
                if (staged.mtp_bridge != MtpBridgeMode::AfterExactHit) {
                    throw std::logic_error("zero-suffix MTP reuse has no exact-hit bridge");
                }
                mark_workspace_usage(workspace_plan.mtp_prefill);
                const auto bridge_rope =
                    prompt_rope_position(staged.prompt, staged.prompt_tokens - 1);
                schedule::mtp_bridge_and_propose(
                    schedule_state, io.token, sequence.tail_hidden,
                    checked_i32(staged.prompt_tokens - 1, "MTP full-prefix bridge position"),
                    bridge_rope, staged.initial_mtp_extent != 0);
                sequence.mtp_kv_valid = staged.prompt_tokens;
                staged.mtp_bridge     = MtpBridgeMode::None;
            }
        }

        copy_round_token();
        std::array<TokenId, qwen3_8::kMtpDecodeMaximumDrafts> initial_drafts{};
        if (staged.prepare_mtp && staged.initial_mtp_extent != 0) {
            CUDA_CHECK(cudaMemcpyAsync(initial_drafts.data(), io.mtp->draft_tokens.data,
                                       staged.initial_mtp_extent * sizeof(TokenId),
                                       cudaMemcpyDeviceToHost, device.stream));
        }
        device.synchronize();
        staged.elapsed_seconds += std::chrono::duration<double>(Clock::now() - started).count();
        const double vision_seconds = staged.vision ? staged.vision->elapsed_seconds() : 0.0;
        const std::optional<std::uint32_t> turn_checkpoint_capture_frontier =
            staged.turn_checkpoint_capture_frontier;
        const std::uint32_t prompt_tokens = staged.prompt_tokens;

        validate_licensed_tokens(std::span<const TokenId>(host_tokens, 1));
        if (sequence.ledger.size() != prompt_tokens) {
            throw std::logic_error("candidate token ledger does not match prompt length");
        }
        sequence.ledger.push_back(host_tokens[0]);
        sequence.prefix_identity.append_generated(1, sequence.rope_delta);
        sequence.text_kv_valid = prompt_tokens;
        if (staged.prepare_mtp) {
            if (sequence.mtp_kv_valid != prompt_tokens) {
                throw std::logic_error("staged MTP prefill did not reach the prompt frontier");
            }
            sequence.mtp_draft_count = staged.initial_mtp_extent;
            std::copy_n(initial_drafts.begin(), staged.initial_mtp_extent,
                        sequence.mtp_drafts.begin());
        } else if (speculative_backend == SpeculativeBackend::DFlash &&
                   sequence.dflash_context_frontier != prompt_tokens) {
            throw std::logic_error("staged DFlash prefill did not reach the prompt frontier");
        }
        sequence.tail_hidden_valid      = true;
        request.timings.vision_seconds  = vision_seconds;
        request.timings.prefill_seconds = std::max(0.0, staged.elapsed_seconds - vision_seconds);
        if (turn_checkpoint_capture_frontier) {
            const std::uint32_t frontier = *turn_checkpoint_capture_frontier;
            if (frontier == 0 || frontier >= prompt_tokens || sequence.text_kv_valid < frontier) {
                throw std::logic_error("turn checkpoint was not materialized by Text prefill");
            }
            if (speculative_backend == SpeculativeBackend::Mtp &&
                (!staged.prepare_mtp || sequence.mtp_kv_valid < frontier - 1)) {
                throw std::logic_error("turn checkpoint has no complete MTP prefix");
            }
            if (speculative_backend == SpeculativeBackend::DFlash &&
                (!dflash || !sequence.kv || !sequence.kv->backend ||
                 sequence.dflash_context_frontier < frontier)) {
                throw std::logic_error("turn checkpoint has no complete DFlash prefix");
            }
            sequence.turn_checkpoint = TurnCheckpoint{.valid = true, .frontier = frontier};
            // The stream was synchronized above, so this device-to-host staging copy starts on
            // an idle stream and overlaps the decode that follows.
            stage_turn_checkpoint(sequence);
        }

        if (!staged.prompt.patches.empty()) {
            staged.prompt.release_media_payload();
            host_input_consumed = true;
        }

        request.prefill.reset();
        request.pending   = PendingCandidate{.kind          = PendingKind::Begin,
                                             .base_E        = 0,
                                             .base_S        = 0,
                                             .prompt_tokens = prompt_tokens,
                                             .produced      = 1};
        request.lifecycle = Lifecycle::Pending;
        return runtime::PrefillStepResult{
            .summary = summary,
            .round   = runtime::GeneratedRound{.tokens = std::span<const TokenId>(host_tokens, 1)},
            .processed_prompt_tokens = processed_prompt_tokens,
            .complete                = true,
            .host_input_consumed     = host_input_consumed,
        };
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        clear_lane(sequence, request);
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                                       std::span<const runtime::RoundBudget> budgets) {
    if (speculative_backend != SpeculativeBackend::None) {
        throw std::logic_error("ordinary batch execution requires the ordinary backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("ordinary batch membership is invalid");
    }

    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("ordinary batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = sequences[lane];
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            sequence.kv->text.bound_row() < 0 || sequence.execution_frontier >= capacity ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier) {
            throw std::logic_error("ordinary batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    const auto start = Clock::now();
    try {
        DecodeGraphExecutable* executable = nullptr;
        ops::GqaExecutionEnvelope envelope{maximum_frontier + 1, maximum_frontier + 1};
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(ordinary_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "ordinary batch");
            executable = &install_graph_profile(ordinary_graphs, profile, "ordinary batch");
            envelope   = {profile.min_execution_frontier + 1, profile.max_execution_frontier + 1};
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence            = sequences[lanes[row]];
            const RequestControl& request      = requests[lanes[row]];
            const std::uint32_t frontier       = sequence.execution_frontier;
            ordinary_host_ingress->tokens[row] = sequence.ledger.back();
            ordinary_host_ingress->cache_positions[row] =
                checked_i32(frontier, "ordinary batch position");
            ordinary_host_ingress->rope_positions[row] =
                checked_i32(frontier, "ordinary batch RoPE position") + sequence.rope_delta;
            ordinary_host_ingress->text_kv_table_rows[row] = sequence.kv->text.bound_row();
            ordinary_host_ingress->lanes[row]    = static_cast<std::int32_t>(sequence.lane);
            ordinary_host_ingress->adapters[row] = lora_slot(sequence.adapter);
            ordinary_host_ingress->sampling[row] = request.sampling_host;
            materialize_sequence_kv(sequence, frontier + 1, 0);
        }

        schedule::OrdinaryBatchContext schedule_state{
            {device, model, work, decoder->linear_attention,
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head},
            decoder->text_kv,
            *io.ordinary,
            *ordinary_host_ingress,
            *ordinary_host_egress,
            tail_hidden_store};

        mark_workspace_usage(workspace_plan.ordinary_round);
        schedule::ordinary_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                        envelope, executable);
        device.synchronize();

        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence    = sequences[lanes[row]];
            RequestControl& request    = requests[lanes[row]];
            const std::uint32_t base_E = sequence.execution_frontier;
            const std::uint32_t base_S = sequence.ledger_frontier;
            const TokenId token        = ordinary_host_egress->sampled_tokens[row];
            validate_licensed_tokens(std::span<const TokenId>(&token, 1));
            sequence.text_kv_valid     = base_E + 1;
            sequence.tail_hidden_valid = true;
            sequence.ledger.push_back(token);
            sequence.prefix_identity.append_generated(1, sequence.rope_delta);
            request.pending   = PendingCandidate{.kind          = PendingKind::Ordinary,
                                                 .base_E        = base_E,
                                                 .base_S        = base_S,
                                                 .prompt_tokens = 0,
                                                 .produced      = 1};
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens = std::span<const TokenId>(ordinary_host_egress->sampled_tokens.data(),
                                               lanes.size())};
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_mtp_batch(std::span<const std::uint32_t> lanes,
                                  std::span<const runtime::RoundBudget> budgets) {
    if (speculative_backend != SpeculativeBackend::Mtp || !io.mtp_decode ||
        decoder->mtp_cache() == nullptr) {
        throw std::logic_error("MTP batch execution requires the MTP backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("MTP batch membership is invalid");
    }

    const std::uint32_t width      = draft_window + 1;
    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("MTP batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = sequences[lane];
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv || !sequence.kv->backend ||
            sequence.kv->text.bound_row() < 0 || sequence.kv->backend->bound_row() < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.mtp_kv_valid != sequence.execution_frontier ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.mtp_draft_count > draft_window) {
            throw std::logic_error("MTP batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    const auto started = Clock::now();
    try {
        DecodeGraphExecutable* executable = nullptr;
        schedule::MtpGqaEnvelopes envelopes =
            mtp_gqa_envelopes(maximum_frontier, draft_window, capacity);
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(mtp_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "MTP batch");
            executable = &install_graph_profile(mtp_graphs, profile, "MTP batch");
            envelopes  = mtp_gqa_envelopes(profile.max_execution_frontier, draft_window, capacity);
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence           = sequences[lanes[row]];
            const RequestControl& request     = requests[lanes[row]];
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1
                                                    : 0;
            const std::uint32_t extent =
                std::min({sequence.mtp_draft_count, draft_window, max_by_budget,
                          capacity - sequence.execution_frontier - 1});
            mtp_host_ingress->anchors[row]        = sequence.ledger.back();
            mtp_host_ingress->base_frontiers[row] = checked_i32(frontier, "MTP batch frontier");
            mtp_host_ingress->remaining_budgets[row] =
                checked_i32(budgets[row].generated_tokens_remaining, "MTP batch remaining budget");
            mtp_host_ingress->current_extents[row]      = static_cast<std::int32_t>(extent);
            mtp_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1);
            for (std::uint32_t j = 0; j < draft_window; ++j) {
                mtp_host_ingress->current_drafts[row * draft_window + j] =
                    j < extent ? sequence.mtp_drafts[j] : sequence.ledger.back();
            }
            for (std::uint32_t j = 0; j < width; ++j) {
                const std::uint32_t position = frontier + std::min(j, extent);
                mtp_host_ingress->target_rope_positions[row * width + j] =
                    checked_i32(position, "MTP batch RoPE position") + sequence.rope_delta;
            }
            mtp_host_ingress->text_kv_table_rows[row] = sequence.kv->text.bound_row();
            mtp_host_ingress->mtp_kv_table_rows[row]  = sequence.kv->backend->bound_row();
            mtp_host_ingress->lanes[row]              = static_cast<std::int32_t>(sequence.lane);
            mtp_host_ingress->adapters[row]           = lora_slot(sequence.adapter);
            mtp_host_ingress->rope_deltas[row]        = sequence.rope_delta;
            mtp_host_ingress->sampling[row]           = request.sampling_host;
            materialize_sequence_kv(sequence, frontier + extent + 1,
                                    std::min(capacity, frontier + extent + draft_window));
        }

        schedule::MtpBatchContext schedule_state{{device, model, work, decoder->linear_attention,
                                                  replay_records ? &*replay_records : nullptr, io,
                                                  prefill_hidden, prefill_chunk, proposal_head},
                                                 decoder->text_kv,
                                                 *decoder->mtp_cache(),
                                                 *io.mtp_decode,
                                                 *mtp_host_ingress,
                                                 *mtp_host_egress,
                                                 tail_hidden_store};

        mark_workspace_usage(workspace_plan.mtp_round);
        schedule::mtp_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                   draft_window, envelopes, executable);
        device.synchronize();

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = sequences[lanes[row]];
            RequestControl& request       = requests[lanes[row]];
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
            const std::int32_t count_i    = mtp_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = mtp_host_egress->accepted_drafts[row];
            const std::int32_t next_i     = mtp_host_egress->next_extents[row];
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || next_i < 0 ||
                next_i > static_cast<std::int32_t>(draft_window) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("MTP batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(mtp_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            const std::uint32_t pcur =
                static_cast<std::uint32_t>(mtp_host_ingress->current_extents[row]);
            if (pcur == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += pcur;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
            }
            request.pending = PendingCandidate{
                .kind          = PendingKind::Speculative,
                .base_E        = base_E,
                .base_S        = base_S,
                .prompt_tokens = 0,
                .produced      = static_cast<std::uint32_t>(count_i),
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(mtp_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .row_counts = std::span<const std::int32_t>(mtp_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .row_stride = width};
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_dflash_batch(std::span<const std::uint32_t> lanes,
                                     std::span<const runtime::RoundBudget> budgets) {
    if (speculative_backend != SpeculativeBackend::DFlash || !io.dflash_decode || !dflash) {
        throw std::logic_error("DFlash batch execution requires the DFlash backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("DFlash batch membership is invalid");
    }

    const std::uint32_t width           = draft_window + 1U;
    std::uint32_t maximum_frontier      = 0;
    std::uint32_t maximum_target_tokens = 1;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("DFlash batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = sequences[lane];
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv || !sequence.kv->backend ||
            sequence.kv->text.bound_row() < 0 || sequence.kv->backend->bound_row() < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.text_kv_valid != sequence.execution_frontier ||
            sequence.dflash_context_frontier > sequence.execution_frontier ||
            sequence.execution_frontier - sequence.dflash_context_frontier > width ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier) {
            throw std::logic_error("DFlash batch row is not decode-ready");
        }
        const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                ? budgets[row].generated_tokens_remaining - 1U
                                                : 0U;
        const std::uint32_t extent =
            std::min({draft_window, max_by_budget, capacity - sequence.execution_frontier - 1U});
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
        maximum_target_tokens =
            std::max(maximum_target_tokens, sequence.execution_frontier + extent + 1U);
    }

    const auto started = Clock::now();
    try {
        DecodeGraphExecutable* executable   = nullptr;
        schedule::DFlashEnvelopes envelopes = dflash_envelopes(0, maximum_frontier, draft_window);
        ops::GqaExecutionEnvelope target_envelope{1, maximum_target_tokens};
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(dflash_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "DFlash batch");
            executable      = &install_graph_profile(dflash_graphs, profile, "DFlash batch");
            envelopes       = dflash_envelopes(profile.min_execution_frontier,
                                               profile.max_execution_frontier, draft_window);
            target_envelope = {
                1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                       capacity, static_cast<std::uint64_t>(profile.max_execution_frontier) +
                                     draft_window + 1ULL))};
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence           = sequences[lanes[row]];
            const RequestControl& request     = requests[lanes[row]];
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1U
                                                    : 0U;
            const std::uint32_t extent =
                std::min({draft_window, max_by_budget, capacity - frontier - 1U});
            dflash_host_ingress->anchors[row] = sequence.ledger.back();
            dflash_host_ingress->execution_frontiers[row] =
                checked_i32(frontier, "DFlash batch frontier");
            dflash_host_ingress->context_frontiers[row] =
                checked_i32(sequence.dflash_context_frontier, "DFlash context frontier");
            dflash_host_ingress->proposal_extents[row]     = static_cast<std::int32_t>(extent);
            dflash_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1U);
            dflash_host_ingress->text_kv_table_rows[row]   = sequence.kv->text.bound_row();
            dflash_host_ingress->dflash_kv_table_rows[row] = sequence.kv->backend->bound_row();
            dflash_host_ingress->lanes[row]    = static_cast<std::int32_t>(sequence.lane);
            dflash_host_ingress->adapters[row] = lora_slot(sequence.adapter);
            dflash_host_ingress->sampling[row] = request.sampling_host;
            materialize_sequence_kv(sequence, frontier + extent + 1U, frontier);
        }

        schedule::DFlashBatchContext schedule_state{{device, model, work, decoder->linear_attention,
                                                     replay_records ? &*replay_records : nullptr,
                                                     io, prefill_hidden, prefill_chunk,
                                                     proposal_head},
                                                    decoder->text_kv,
                                                    *dflash,
                                                    *io.dflash_decode,
                                                    *dflash_host_ingress,
                                                    *dflash_host_egress,
                                                    tail_hidden_store};

        mark_workspace_usage(workspace_plan.dflash_round);
        schedule::dflash_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                      draft_window, envelopes, target_envelope, executable);
        device.synchronize();

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = sequences[lanes[row]];
            RequestControl& request       = requests[lanes[row]];
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
            const std::int32_t count_i    = dflash_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = dflash_host_egress->accepted_drafts[row];
            const std::uint32_t extent =
                static_cast<std::uint32_t>(dflash_host_ingress->proposal_extents[row]);
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || accepted_i > static_cast<std::int32_t>(extent) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("DFlash batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(dflash_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            if (extent == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += extent;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
            }
            sequence.dflash_context_frontier = base_E;
            request.pending                  = PendingCandidate{
                                 .kind          = PendingKind::Speculative,
                                 .base_E        = base_E,
                                 .base_S        = base_S,
                                 .prompt_tokens = 0,
                                 .produced      = static_cast<std::uint32_t>(count_i),
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(dflash_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .row_counts = std::span<const std::int32_t>(dflash_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .row_stride = width};
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_batch(std::span<const std::uint32_t> lanes,
                              std::span<const runtime::RoundBudget> budgets) {
    if (speculative_backend == SpeculativeBackend::None) {
        return decode_ordinary_batch(lanes, budgets);
    }
    if (speculative_backend == SpeculativeBackend::Mtp) { return decode_mtp_batch(lanes, budgets); }
    return decode_dflash_batch(lanes, budgets);
}

void ProgramImplCore::resolve_non_speculative_pending(SequenceState& sequence,
                                                      RequestControl& request,
                                                      std::uint32_t accepted_tokens,
                                                      bool terminal) {
    if (request.lifecycle != Lifecycle::Pending) {
        throw std::logic_error("pending resolution requires a pending generated round");
    }
    if ((request.pending.kind != PendingKind::Begin &&
         request.pending.kind != PendingKind::Ordinary) ||
        request.pending.produced != 1 || accepted_tokens != 1) {
        throw std::logic_error("non-speculative pending round must commit its single token");
    }

    switch (request.pending.kind) {
    case PendingKind::Begin:
        sequence.execution_frontier = request.pending.prompt_tokens;
        sequence.ledger_frontier    = request.pending.prompt_tokens + 1;
        break;
    case PendingKind::Ordinary:
        sequence.execution_frontier = request.pending.base_E + request.pending.produced;
        sequence.ledger_frontier    = request.pending.base_S + request.pending.produced;
        break;
    case PendingKind::Speculative:
    case PendingKind::None:
        throw std::logic_error("non-speculative pending round has an invalid kind");
    }
    if (sequence.ledger_frontier != sequence.execution_frontier + 1 ||
        sequence.ledger.size() != sequence.ledger_frontier ||
        sequence.prefix_identity.size() != sequence.ledger_frontier) {
        throw std::logic_error("resolved round did not establish a valid frontier");
    }
    trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
    if (terminal) {
        sequence.mtp_draft_count = 0;
        release_sequence_growth_entitlement(sequence);
        unbind_sequence_kv(sequence);
        sequence.retained = true;
    }
    request.lifecycle = terminal ? Lifecycle::Complete : Lifecycle::Active;
    request.pending   = {};
}

void ProgramImplCore::retire_lane(std::uint32_t lane) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    SequenceState& sequence = sequences[lane];
    RequestControl& request = requests[lane];
    if (request.lifecycle != Lifecycle::Active || request.pending.kind != PendingKind::None ||
        request.prefill) {
        throw std::logic_error("retire_lane requires an active lane at a round boundary");
    }
    // The same terminal transition a resolved final round performs, without a round: the lane
    // gives back the entitlement it never mapped, releases its KV table row, and keeps its state
    // as a retained session the next turn can reuse.
    sequence.mtp_draft_count = 0;
    release_sequence_growth_entitlement(sequence);
    unbind_sequence_kv(sequence);
    sequence.retained = true;
    request.lifecycle = Lifecycle::Complete;
}

MemorySummary ProgramImplCore::memory_summary() const noexcept {
    MemorySummary out;
    out.device      = device.device;
    out.max_context = capacity;
    out.kv_capacity = kv_capacity;
    out.kv_cache = kv_dtype == DType::BF16
                       ? KvCacheStorage::BFloat16
                       : (kv_e8_root
                              ? KvCacheStorage::RK2V4E8
                              : (kv_packed_k
                                     ? KvCacheStorage::RotatedInt4KeyInt4ValueGroup64
                                     : (kv_rotate_v ? KvCacheStorage::RotatedInt8KeyInt4ValueGroup64
                                                    : KvCacheStorage::Int8Group64)));
    DeviceArena& weights = *model.weights_arena;
    out.weights = ArenaMemorySummary{weights.capacity(), weights.used(), weights.peak_used()};
    out.sequence =
        ArenaMemorySummary{persistent.capacity(), persistent.used(), persistent.peak_used()};
    out.workspace = ArenaMemorySummary{workspace_storage.capacity(), work.used(), work.peak_used()};
    out.workspace_logical_peak_bytes = workspace_logical_peak_bytes;
    out.cuda_graph_allowance_bytes   = graph_allowance_bytes;
    out.cuda_graph_observed_bytes    = graph_observed_bytes;
    out.kv_payload_bytes             = kv_payload_bytes;
    out.text_kv_bytes                = text_kv_bytes;
    out.mtp_kv_bytes                 = mtp_kv_bytes;
    out.gdn_state_bytes              = gdn_state_bytes;
    out.dflash_kv_bytes              = dflash_kv_bytes;
    out.replay_records_bytes         = replay_records_bytes;
    // The adapter bank lives in a package-owned arena outside `weights`, so it is reported from
    // the model view rather than from an arena this Program owns. Zero when no adapter is
    // registered.
    out.lora_bank_bytes =
        model.lora ? static_cast<std::size_t>(model.lora->device_bytes) : std::size_t{0};
    return out;
}

void ProgramImplCore::reset_memory_peaks() noexcept {
    model.weights_arena->reset_peak();
    persistent.reset_peak();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
}

} // namespace ninfer::targets::qwen3_8::detail::NINFER_QWEN38_RUNTIME_NS
