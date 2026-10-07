#include "core/device.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/pinned_transfer.h"
#include <ninfer/targets/qwen3_8/decoder_state.h>
#include <ninfer/targets/qwen3_8/hybrid_topology.h>
#include <ninfer/targets/qwen3_8/mtp_alignment.h>
#include <ninfer/targets/qwen3_8/round_state.h>
#include <ninfer/targets/qwen3_8/vision_control.h>

#include "targets/qwen3_8/impl/runtime/prefix_identity.h"
#include "targets/qwen3_8/impl/runtime/continuation_image.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace q36 = ninfer::targets::qwen3_8;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

void test_topology() {
    static_assert(q36::kHybridAttentionInterval == 4);
    static_assert(q36::full_attention_layers(64) == 16);
    static_assert(q36::gdn_layers(64) == 48);
    for (std::int32_t layer = 0; layer < 64; ++layer) {
        expect(q36::is_full_attention_layer(layer) == ((layer + 1) % 4 == 0), "hybrid layer kind");
        if (q36::is_full_attention_layer(layer)) {
            expect(q36::full_attention_index(layer) == layer / 4, "full-attention index");
        } else {
            expect(q36::gdn_index(layer) == layer - layer / 4, "GDN index");
        }
    }
}

q36::DecoderStateSpec decoder_spec(ninfer::DType dtype, bool mtp) {
    return q36::DecoderStateSpec{
        .full_attention_layers     = 2,
        .mtp_layers                = 1,
        .capacity                  = 129,
        .kv_heads                  = 2,
        .attention_head_dim        = 64,
        .kv_dtype                  = dtype,
        .kv_quant_group            = dtype == ninfer::DType::I8 ? q36::kKvQuantGroup : 0,
        .enable_mtp                = mtp,
        .text_physical_page_groups = 5,
        .mtp_physical_page_groups  = mtp ? 4U : 0U,
        .linear_attention =
            {
                .layers         = 3,
                .conv_channels  = 10,
                .conv_width     = 3,
                .value_heads    = 4,
                .value_head_dim = 5,
                .key_head_dim   = 6,
                .slot_count     = 4,
                .conv_dtype     = ninfer::DType::BF16,
            },
    };
}

void test_decoder_layout() {
    ninfer::LayoutBuilder bf16_builder;
    const q36::DecoderStateLayout bf16 =
        q36::plan_decoder_state(bf16_builder, decoder_spec(ninfer::DType::BF16, false));
    (void)bf16_builder.finish(256);
    expect(bf16.text_kv.pool.planes.size() == 4, "BF16 Text KV has K/V planes per layer");
    expect(bf16.text_kv.pool.spec.page_group_count == 5 &&
               bf16.text_kv.pool.spec.logical_page_capacity == 3 &&
               bf16.text_kv.pool.spec.table_rows == 1,
           "Text KV separates five physical pages from three logical pages");
    expect(std::all_of(bf16.text_kv.pool.planes.begin(), bf16.text_kv.pool.planes.end(),
                       [](const ninfer::PagedKVPlaneLayout& plane) {
                           return plane.spec.dtype == ninfer::DType::BF16;
                       }),
           "BF16 KV has no scale planes");
    expect(!bf16.mtp_kv.has_value(), "disabled MTP omits KV storage");
    expect(bf16.linear_attention.conv.size() == 3 && bf16.linear_attention.recurrent.size() == 3,
           "Linear Attention layer storage");
    expect(bf16.linear_attention.spec.slot_count == 4, "Linear Attention slot geometry");
    expect(bf16.kv_payload_bytes() == bf16.text_kv.payload_bytes(), "BF16 KV payload accounting");

    ninfer::LayoutBuilder int8_builder;
    const q36::DecoderStateLayout int8 =
        q36::plan_decoder_state(int8_builder, decoder_spec(ninfer::DType::I8, true));
    (void)int8_builder.finish(256);
    expect(int8.text_kv.pool.planes.size() == 8 &&
               int8.text_kv.pool.planes[2].spec.dtype == ninfer::DType::FP16 &&
               int8.text_kv.pool.planes[3].spec.dtype == ninfer::DType::FP16,
           "INT8 Text KV has code and scale planes per layer");
    expect(int8.mtp_kv.has_value() && int8.mtp_kv->layers == 1 &&
               int8.mtp_kv->pool.planes.size() == 4 &&
               int8.mtp_kv->pool.spec.page_group_count == 4 &&
               int8.mtp_kv->pool.spec.logical_page_capacity == 3,
           "enabled MTP has one paged KV layer");
    expect(int8.mtp_kv && int8.mtp_kv->pool.planes[2].spec.dtype == ninfer::DType::FP16 &&
               int8.mtp_kv->pool.planes[3].spec.dtype == ninfer::DType::FP16,
           "INT8 MTP KV has scale planes");
    expect(int8.kv_payload_bytes() == int8.text_kv.payload_bytes() + int8.mtp_kv->payload_bytes(),
           "INT8 Text/MTP KV payload accounting");
}

void test_round_layout() {
    ninfer::LayoutBuilder builder;
    q36::RoundStateLayout round = q36::begin_round_state_layout(
        builder, q36::RoundStateSpec{
                     .hidden = 32, .output_rows = 128, .draft_window = 5, .enable_mtp = true});
    const ninfer::TensorRegion exact_prefill =
        builder.add_tensor(ninfer::DType::BF16, {32, 16}, 256, "exact prefill hidden");
    q36::complete_round_state_layout(builder, round);
    (void)builder.finish(256);
    expect(round.complete, "round layout completes");
    expect(round.logits.shape[0] == 128 && round.logits.shape[1] == 1, "round logits shape");
    expect(round.mtp.has_value() && round.mtp->draft_tokens.shape[0] == 5 &&
               round.mtp->target_input_ids.shape[0] == 6,
           "MTP prefill scratch shapes");
    expect(round.logits.region.offset < exact_prefill.region.offset &&
               exact_prefill.region.offset < round.mtp->draft_tokens.region.offset,
           "exact prefill extension retains established round-region order");
    expect(round.mtp.has_value() && round.mtp->position.shape[0] == 1,
           "MTP prefill scratch is explicit");
    expect(round.mtp_decode.has_value() && round.mtp_decode->alignment_ids.shape[0] == 6 &&
               round.mtp_decode->alignment_ids.shape[1] == 1,
           "MTP decode frame is explicit");

    ninfer::LayoutBuilder speculative_builder;
    q36::RoundStateLayout dflash = q36::begin_round_state_layout(
        speculative_builder,
        q36::RoundStateSpec{
            .hidden = 32, .output_rows = 128, .draft_window = 15, .enable_dflash = true});
    q36::complete_round_state_layout(speculative_builder, dflash);
    (void)speculative_builder.finish(256);
    expect(dflash.logits.shape[1] == 1 && dflash.dflash_prefill.has_value() &&
               dflash.dflash_prefill->produced_count.shape[0] == 1 &&
               dflash.dflash_decode.has_value() &&
               dflash.dflash_decode->draft_tokens.shape[0] == 15,
           "K=15 DFlash storage is backend-owned");
    expect(!dflash.mtp.has_value() && !dflash.mtp_decode.has_value(),
           "DFlash layout does not allocate MTP storage");
}

void test_mtp_alignment() {
    const std::vector<std::int32_t> scatter{2, 4, 7};
    const q36::MtpAlignmentWindow first = q36::plan_mtp_alignment_window(8, 0, 4);
    expect(first.hidden_begin == 0 && first.position_begin == 0 &&
               first.shifted_embedding_begin == 1 && first.columns == 4 &&
               !first.final_column_uses_generated_token,
           "non-final MTP alignment window");
    const q36::MtpVisualOverlap first_visual = q36::shifted_visual_overlap(scatter, 8, first);
    expect(first_visual.source_begin == 0 &&
               first_visual.destination_columns == std::vector<std::int32_t>({1, 3}),
           "non-final shifted visual overlap");

    const q36::MtpAlignmentWindow final = q36::plan_mtp_alignment_window(8, 4, 4);
    expect(final.shifted_embedding_begin == 5 && final.final_column_uses_generated_token,
           "final MTP alignment window");
    const q36::MtpVisualOverlap final_visual = q36::shifted_visual_overlap(scatter, 8, final);
    expect(final_visual.source_begin == 2 &&
               final_visual.destination_columns == std::vector<std::int32_t>({2}),
           "final shifted visual overlap excludes generated-token column");
}

void test_vision_control() {
    q36::PreparedPromptData prompt;
    prompt.token_ids.resize(7);
    prompt.token_types           = {0, static_cast<std::uint8_t>(q36::PromptModality::Image),
                                    0, static_cast<std::uint8_t>(q36::PromptModality::Video),
                                    0, static_cast<std::uint8_t>(q36::PromptModality::Video),
                                    0};
    prompt.prepare.media_items   = 2;
    prompt.prepare.raw_patches   = 12;
    prompt.prepare.vision_tokens = 3;
    prompt.vision_items          = {
        q36::VisionItem{.modality    = q36::PromptModality::Image,
                                 .grid        = {.temporal = 1, .height = 2, .width = 2},
                                 .patch_begin = 0,
                                 .patch_count = 4,
                                 .token_spans = {{.begin = 1, .count = 1}}},
        q36::VisionItem{.modality    = q36::PromptModality::Video,
                                 .grid        = {.temporal = 2, .height = 2, .width = 2},
                                 .patch_begin = 4,
                                 .patch_count = 8,
                                 .token_spans = {{.begin = 3, .count = 1}, {.begin = 5, .count = 1}}},
    };

    const q36::VisionControl control = q36::build_vision_control(prompt);
    expect(control.items.size() == 2, "Vision per-item control count");
    expect(control.items[0].patch_begin == 0 && control.items[0].patch_count == 4 &&
               control.items[0].merged_count == 1 && control.items[0].segment_length == 4 &&
               control.items[0].segment_count == 1 &&
               control.items[0].cu_seqlens == std::vector<std::int32_t>({0, 4}) &&
               control.items[0].scatter_indices == std::vector<std::int32_t>({1}) &&
               control.items[0].position_ids.size() == 8 &&
               control.items[0].position_table_indices.size() == 16 &&
               control.items[0].position_table_weights.size() == 16,
           "image item control offsets");
    expect(control.items[1].patch_begin == 4 && control.items[1].patch_count == 8 &&
               control.items[1].merged_count == 2 && control.items[1].segment_length == 4 &&
               control.items[1].segment_count == 2 &&
               control.items[1].cu_seqlens == std::vector<std::int32_t>({0, 4, 8}) &&
               control.items[1].scatter_indices == std::vector<std::int32_t>({3, 5}) &&
               control.items[1].position_ids.size() == 16 &&
               control.items[1].position_table_indices.size() == 32 &&
               control.items[1].position_table_weights.size() == 32,
           "video item control offsets");
}

q36::PreparedPromptData identity_prompt(std::uint8_t digest_byte = 1) {
    q36::PreparedPromptData prompt;
    prompt.token_ids   = {10, 248056, 248056, 11};
    prompt.token_types = {0, static_cast<std::uint8_t>(q36::PromptModality::Image),
                          static_cast<std::uint8_t>(q36::PromptModality::Image), 0};
    prompt.positions   = {0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 2, 3};
    prompt.rope_delta  = 0;
    q36::VisionItem item{.modality    = q36::PromptModality::Image,
                         .grid        = {.temporal = 1, .height = 2, .width = 4},
                         .patch_begin = 0,
                         .patch_count = 8,
                         .timestamps  = {0.25, 0.5},
                         .token_spans = {{.begin = 1, .count = 2}}};
    item.content_digest.fill(digest_byte);
    item.preprocessing_digest.fill(17);
    prompt.vision_items.push_back(std::move(item));
    return prompt;
}

void append_text_token(q36::PreparedPromptData& prompt, ninfer::TokenId token,
                       std::int32_t position) {
    const std::size_t old_tokens = prompt.token_ids.size();
    std::vector<std::int32_t> positions;
    positions.reserve(3 * (old_tokens + 1));
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto begin =
            prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * old_tokens);
        positions.insert(positions.end(), begin, begin + static_cast<std::ptrdiff_t>(old_tokens));
        positions.push_back(position);
    }
    prompt.token_ids.push_back(token);
    prompt.token_types.push_back(0);
    prompt.positions = std::move(positions);
}

void test_prefix_identity() {
    q36::PreparedPromptData original    = identity_prompt();
    std::vector<ninfer::TokenId> ledger = original.token_ids;
    q36::detail::ResidentPrefixIdentity resident;
    resident.reserve(16);
    resident.assign(original);

    expect(q36::detail::prefix_matches(original, ledger, resident, original.token_ids.size()),
           "identical multimodal prefix identity");

    q36::PreparedPromptData changed_media = identity_prompt(2);
    expect(!q36::detail::prefix_matches(changed_media, ledger, resident,
                                        changed_media.token_ids.size()),
           "different media content must not reuse placeholder tokens");
    expect(q36::detail::prefix_matches(changed_media, ledger, resident, 1),
           "media wholly after the frontier does not affect prefix identity");
    expect(!q36::detail::prefix_matches(original, ledger, resident, 2),
           "frontier must not divide one Vision item");

    q36::PreparedPromptData changed_position = identity_prompt();
    changed_position.positions[0] += 1;
    expect(!q36::detail::prefix_matches(changed_position, ledger, resident,
                                        changed_position.token_ids.size()),
           "different MRoPE positions must not reuse resident state");

    resident.append_generated(1, original.rope_delta);
    ledger.push_back(12);
    append_text_token(original, 12, 4);
    expect(q36::detail::prefix_matches(original, ledger, resident, ledger.size()),
           "generated multimodal continuation identity");

    const q36::PreparedPromptData prompt_only = identity_prompt();
    resident.truncate(prompt_only.token_ids.size());
    ledger.resize(prompt_only.token_ids.size());
    expect(q36::detail::prefix_matches(prompt_only, ledger, resident, ledger.size()),
           "truncated multimodal continuation identity");
}

void test_continuation_reuse_depth() {
    q36::PreparedPromptData saved;
    saved.token_ids   = {10, 11, 12, 13};
    saved.token_types = {0, 0, 0, 0};
    saved.positions   = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    q36::detail::ResidentPrefixIdentity resident;
    resident.assign(saved);
    std::vector<ninfer::TokenId> ledger = saved.token_ids;
    resident.append_generated(1, 0);
    ledger.push_back(14);

    q36::PreparedPromptData incoming = saved;
    incoming.token_ids[2]            = 99;
    append_text_token(incoming, 100, 4);
    expect(q36::detail::continuation_reuse_depth(incoming, ledger, resident, 5, 2) == 2,
           "a divergent completion reuses its exact saved turn checkpoint");

    incoming.identity.turn_rewrite_boundary = 2;
    expect(q36::detail::continuation_reuse_depth(incoming, ledger, resident, 5, 2) == 2,
           "the planner can keep the matching saved turn checkpoint");
    incoming.identity.turn_rewrite_boundary = 1;
    expect(q36::detail::continuation_reuse_depth(incoming, ledger, resident, 5, 2) == 0,
           "preflight rejects a checkpoint the planner would replace before reuse");
    incoming.identity.turn_rewrite_boundary = 3;
    expect(q36::detail::continuation_reuse_depth(incoming, ledger, resident, 5, 2) == 2,
           "a later checkpoint can be captured while prefilling the divergent suffix");

    q36::PreparedPromptData exact = saved;
    append_text_token(exact, 14, 4);
    exact.identity.turn_rewrite_boundary = 1;
    expect(q36::detail::continuation_reuse_depth(exact, ledger, resident, 5, 2) == 0,
           "an exact frontier is rejected when planner checkpoint policy forces full reset");

    incoming.token_ids.resize(2);
    incoming.token_types.resize(2);
    incoming.positions = {0, 1, 0, 1, 0, 1};
    incoming.identity.turn_rewrite_boundary.reset();
    expect(q36::detail::continuation_reuse_depth(incoming, ledger, resident, 5, 2) == 0,
           "checkpoint fallback requires a nonempty suffix");
}

void test_prefix_identity_snapshot() {
    const q36::PreparedPromptData original = identity_prompt();
    q36::detail::ResidentPrefixIdentity resident;
    resident.assign(original);

    q36::detail::ResidentPrefixIdentity restored;
    restored.restore(resident.export_prefix(3));
    std::vector<ninfer::TokenId> ledger(original.token_ids.begin(), original.token_ids.begin() + 3);
    expect(restored.size() == 3 && q36::detail::prefix_matches(original, ledger, restored, 3),
           "owning snapshot restores an exact multimodal prefix");

    q36::PreparedPromptData changed = original;
    changed.token_types[1]          = 0;
    expect(!q36::detail::prefix_matches(changed, ledger, restored, 3),
           "snapshot retains token types");
    for (std::size_t axis = 0; axis < 3; ++axis) {
        changed = original;
        changed.positions[axis * original.token_ids.size() + 1] += 1;
        expect(!q36::detail::prefix_matches(changed, ledger, restored, 3),
               "snapshot retains every MRoPE axis");
    }

    const auto media_change_rejected = [&](auto change, std::string_view message) {
        q36::PreparedPromptData candidate = original;
        change(candidate.vision_items[0]);
        expect(!q36::detail::prefix_matches(candidate, ledger, restored, 3), message);
    };
    media_change_rejected([](q36::VisionItem& item) { item.modality = q36::PromptModality::Video; },
                          "snapshot retains media modality");
    media_change_rejected([](q36::VisionItem& item) { ++item.grid.temporal; },
                          "snapshot retains media grid");
    media_change_rejected([](q36::VisionItem& item) { ++item.patch_begin; },
                          "snapshot retains media patch offset");
    media_change_rejected([](q36::VisionItem& item) { ++item.patch_count; },
                          "snapshot retains media patch count");
    media_change_rejected([](q36::VisionItem& item) { ++item.content_digest[0]; },
                          "snapshot retains media digest");
    media_change_rejected([](q36::VisionItem& item) { ++item.preprocessing_digest[0]; },
                          "snapshot retains preprocessing digest at the same grid");
    media_change_rejected([](q36::VisionItem& item) { item.timestamps[0] += 0.25; },
                          "snapshot retains media timestamps");
    media_change_rejected([](q36::VisionItem& item) { ++item.token_spans[0].count; },
                          "snapshot retains media token spans");

    restored.restore(resident.export_prefix(2));
    ledger.resize(2);
    expect(restored.size() == 2 &&
               !q36::detail::prefix_matches(original, ledger, restored, ledger.size()),
           "snapshot preserves split-media frontier rejection");

    q36::detail::ResidentPrefixIdentitySnapshot malformed = resident.export_prefix(1);
    malformed.positions[2].clear();
    bool rejected = false;
    try {
        restored.restore(std::move(malformed));
    } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected && restored.size() == 2, "snapshot restore validates shape atomically");
}

void test_continuation_image_codec() {
    namespace image = q36::detail::continuation;
    const q36::PreparedPromptData prompt = identity_prompt();
    q36::detail::ResidentPrefixIdentity resident;
    resident.assign(prompt);
    const auto encoded = image::encode_prefix(prompt.token_ids,
                                               resident.export_prefix(prompt.token_ids.size()));
    image::PrefixData decoded = image::decode_prefix(encoded);
    q36::detail::ResidentPrefixIdentity restored;
    restored.restore(std::move(decoded.identity));
    expect(decoded.ledger == prompt.token_ids &&
               q36::detail::prefix_matches(prompt, decoded.ledger, restored,
                                            prompt.token_ids.size()),
           "continuation prefix codec preserves exact authorization identity");
    auto old_prefix                                        = encoded;
    old_prefix[8 + std::string_view("qwen-prefix").size()] = image::kTargetImageVersion - 1;
    bool old_rejected                                      = false;
    try {
        (void)image::decode_prefix(old_prefix);
    } catch (const std::invalid_argument&) { old_rejected = true; }
    expect(old_rejected,
           "continuation prefixes from before preprocessing identity are invalidated");

    const image::FrontierMetadata metadata{.execution_frontier = 3,
                                            .ledger_frontier = 4,
                                            .rope_delta = -7,
                                            .text_kv_valid = 3,
                                            .backend_kv_valid = 3,
                                            .mtp_drafts = {8, 9}};
    const image::FrontierMetadata metadata_roundtrip =
        image::decode_frontier(image::encode_frontier(metadata));
    expect(metadata_roundtrip.execution_frontier == 3 &&
               metadata_roundtrip.ledger_frontier == 4 &&
               metadata_roundtrip.rope_delta == -7 &&
               metadata_roundtrip.mtp_drafts == std::vector<ninfer::TokenId>({8, 9}),
            "continuation frontier metadata uses stable little-endian encoding");

    const image::FrontierMetadata prefix_only{.execution_frontier = 3,
                                               .ledger_frontier = 3,
                                               .rope_delta = -7,
                                               .text_kv_valid = 3,
                                               .backend_kv_valid = 3};
    const auto prefix_only_roundtrip =
        image::decode_frontier(image::encode_frontier(prefix_only));
    expect(prefix_only_roundtrip.execution_frontier ==
                   prefix_only_roundtrip.ledger_frontier &&
               prefix_only_roundtrip.mtp_drafts.empty(),
           "prefix-only frontier encodes without a fabricated bonus token");
    expect(image::valid_ledger_frontier(3, 3) && image::valid_ledger_frontier(3, 4) &&
               !image::valid_ledger_frontier(3, 5),
           "preflight accepts prefix-only and completed ledgers but rejects other shapes");

    const auto rejects = [](const auto& operation) {
        try {
            operation();
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    auto truncated = encoded;
    truncated.pop_back();
    expect(rejects([&] { (void)image::decode_prefix(truncated); }),
           "continuation codec rejects truncation");
    auto trailing = image::encode_boundary({.valid = true, .frontier = 2});
    trailing.push_back(0);
    expect(rejects([&] { (void)image::decode_boundary(trailing); }),
           "continuation codec rejects trailing bytes");
    expect(rejects([&] {
               (void)image::decode_boundary(
                   image::encode_boundary({.valid = false, .frontier = 2}));
           }),
           "continuation codec rejects invalid boundary invariants");

    image::Writer oversized;
    image::write_header(oversized, "tensor");
    oversized.u64(std::numeric_limits<std::uint64_t>::max());
    const auto malformed_size = std::move(oversized).finish();
    expect(rejects([&] { (void)image::decode_tensor(malformed_size, 0); }),
            "continuation codec rejects oversized lengths before allocation");

    image::Writer oversized_prefix;
    image::write_header(oversized_prefix, "qwen-prefix");
    oversized_prefix.u64(std::numeric_limits<std::uint64_t>::max());
    const auto malformed_prefix_count = std::move(oversized_prefix).finish();
    expect(rejects([&] { (void)image::decode_prefix(malformed_prefix_count, 1024); }),
           "continuation codec rejects huge prefix counts before allocation");

    image::Writer too_many_drafts;
    image::write_header(too_many_drafts, "qwen-frontier");
    too_many_drafts.u32(1);
    too_many_drafts.u32(2);
    too_many_drafts.i32(0);
    too_many_drafts.u32(1);
    too_many_drafts.u32(1);
    too_many_drafts.u64(q36::kMtpDecodeMaximumDrafts + 1U);
    for (std::uint32_t i = 0; i <= q36::kMtpDecodeMaximumDrafts; ++i) {
        too_many_drafts.i32(static_cast<std::int32_t>(i));
    }
    const auto malformed_drafts = std::move(too_many_drafts).finish();
    expect(rejects([&] { (void)image::decode_frontier(malformed_drafts); }),
           "continuation codec rejects MTP draft inventories above runtime geometry");

    expect(rejects([&] { (void)image::decode_prefix(encoded, prompt.token_ids.size() - 1); }),
           "continuation codec rejects prefixes above runtime capacity before allocation");
}

void test_decision_state_identity() {
    namespace image  = q36::detail::continuation;
    auto state       = identity_prompt();
    state.rope_delta = -2;
    const ninfer::cache::Bytes key{1, 2, 3};
    const auto prefix  = image::decision_state_prefix(state);
    const auto alias   = image::decision_state_alias(key, state);
    const auto digest  = image::decision_state_digest(prefix, state.rope_delta);
    const auto decoded = image::decode_prefix(prefix);
    q36::detail::ResidentPrefixIdentity resident;
    resident.restore(decoded.identity);
    expect(q36::detail::prefix_matches(state, decoded.ledger, resident, state.token_ids.size()),
           "decision state roundtrip retains image geometry and content identity");
    const auto rejects_change = [&](auto change) {
        auto changed = state;
        change(changed);
        expect(image::decision_state_alias(key, changed) != alias,
               "decision state alias rejects changed state semantics");
        expect(image::decision_state_digest(image::decision_state_prefix(changed),
                                            changed.rope_delta) != digest,
               "decision descriptor digest rejects changed state semantics");
    };
    rejects_change([](auto& value) { ++value.vision_items[0].content_digest[0]; });
    rejects_change([](auto& value) { ++value.vision_items[0].preprocessing_digest[0]; });
    rejects_change([](auto& value) { ++value.vision_items[0].grid.height; });
    rejects_change([](auto& value) { --value.vision_items[0].token_spans[0].count; });
    rejects_change([](auto& value) { value.token_types[1] = 0; });
    for (std::size_t axis = 0; axis < 3; ++axis) {
        rejects_change([&](auto& value) { ++value.positions[axis * value.token_ids.size() + 1]; });
    }
    rejects_change([](auto& value) { ++value.rope_delta; });
    rejects_change([](auto& value) { append_text_token(value, 99, 4); });
    auto payload_changed    = state;
    payload_changed.patches = {1.0F, 2.0F};
    expect(image::decision_state_prefix(payload_changed) == prefix,
           "decision state persistence contains no pixel or embedding payload");
    const image::DecisionStateMetadata metadata{
        .state_tokens = 4, .rope_delta = -2, .adapter = {7}};
    const auto restored = image::decode_decision_state(image::encode_decision_state(metadata));
    expect(restored.state_tokens == metadata.state_tokens &&
               restored.rope_delta == metadata.rope_delta && restored.adapter == metadata.adapter,
           "decision state metadata preserves its RoPE continuation and adapter");
}

void test_continuation_prefix_filter_digest() {
    namespace image = q36::detail::continuation;
    const q36::PreparedPromptData prompt = identity_prompt();
    q36::detail::ResidentPrefixIdentity resident;
    resident.assign(prompt);
    const auto snapshot = resident.export_prefix(prompt.token_ids.size());
    const auto expected = image::prefix_filter_digest(prompt, prompt.token_ids.size());
    expect(expected == image::prefix_filter_digest(prompt.token_ids, snapshot,
                                                    prompt.token_ids.size()),
           "prompt and resident paths produce the same canonical prefix filter digest");

    q36::PreparedPromptData changed = prompt;
    changed.token_ids.back() += 1;
    expect(image::prefix_filter_digest(changed, changed.token_ids.size()) != expected,
           "prefix filter digest binds token IDs");
    changed = prompt;
    changed.positions.back() += 1;
    expect(image::prefix_filter_digest(changed, changed.token_ids.size()) != expected,
           "prefix filter digest binds every position axis");
    changed = identity_prompt(2);
    expect(image::prefix_filter_digest(changed, changed.token_ids.size()) != expected,
           "prefix filter digest binds media content identity");

    bool split_rejected = false;
    try {
        (void)image::prefix_filter_digest(prompt, 2);
    } catch (const std::logic_error&) { split_rejected = true; }
    expect(split_rejected, "prefix filter digest rejects a frontier that divides a media item");
}

void test_stable_alias_identity() {
    namespace image = q36::detail::continuation;
    q36::PreparedPromptData first = identity_prompt();
    q36::PreparedPromptData leaf = first;
    leaf.token_ids[3] += 7;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        leaf.positions[axis * leaf.token_ids.size() + 3] += 9;
    }
    const ninfer::cache::Bytes domain{1, 2, 3, 4};
    const auto key = image::boundary_alias(domain, first, 3);
    expect(key && key == image::boundary_alias(domain, leaf, 3),
           "boundary alias ignores content after its depth");

    const auto differs = [&](auto mutate, std::string_view message) {
        q36::PreparedPromptData changed = first;
        mutate(changed);
        expect(image::boundary_alias(domain, changed, 3) != key, message);
    };
    differs([](auto& prompt) { ++prompt.token_ids[0]; }, "boundary alias binds prefix token IDs");
    differs([](auto& prompt) { ++prompt.token_types[1]; }, "boundary alias binds token types");
    for (std::size_t axis = 0; axis < 3; ++axis) {
        differs([axis](auto& prompt) { ++prompt.positions[axis * prompt.token_ids.size() + 1]; },
                "boundary alias binds every position axis");
    }
    differs([](auto& prompt) { ++prompt.vision_items[0].content_digest[0]; },
            "boundary alias binds media identity through the boundary");
    expect(image::boundary_alias(domain, first, 4) != key, "boundary alias binds the depth");
    expect(image::boundary_alias(ninfer::cache::Bytes{1, 2, 3, 5}, first, 3) != key,
           "boundary alias binds the Program compatibility domain");
    expect(!image::boundary_alias(domain, first, 0) && !image::boundary_alias(domain, first, 5),
           "boundary alias rejects an empty or overlong prefix");
    q36::PreparedPromptData unreusable = first;
    unreusable.identity.reusable       = false;
    expect(!image::boundary_alias(domain, unreusable, 3),
           "boundary alias is not derived for a non-reusable prompt");

    // Every boundary names its own alias; only the newest turn openers are looked up.
    q36::PreparedPromptData windowed = first;
    for (std::uint32_t depth = 4; depth < 4 + image::kTurnOpenerLookupWindow + 2; ++depth) {
        append_text_token(windowed, static_cast<ninfer::TokenId>(100 + depth),
                          static_cast<std::int32_t>(depth));
    }
    windowed.identity.boundaries.push_back(
        {.depth = 1, .kind = q36::PromptBoundaryKind::SystemTools, .publish = true});
    for (std::uint32_t depth = 3; depth < 3 + image::kTurnOpenerLookupWindow + 2; ++depth) {
        windowed.identity.boundaries.push_back(
            {.depth = depth, .kind = q36::PromptBoundaryKind::TurnOpener, .publish = false});
    }
    const std::uint32_t deepest = 3 + static_cast<std::uint32_t>(image::kTurnOpenerLookupWindow) + 2;
    windowed.identity.boundaries.push_back(
        {.depth = deepest, .kind = q36::PromptBoundaryKind::Explicit, .publish = true});
    windowed.identity.turn_rewrite_boundary = deepest;
    const std::vector<q36::PromptBoundaryAlias> aliases = image::boundary_aliases(domain, windowed);
    expect(aliases.size() == image::kTurnOpenerLookupWindow + 2,
           "boundary aliases did not window turn openers while keeping other kinds");
    expect(aliases.front().kind == q36::PromptBoundaryKind::SystemTools &&
               aliases.front().depth == 1 && aliases.front().publish &&
               aliases.back().kind == q36::PromptBoundaryKind::Explicit && aliases.back().publish &&
               aliases[1].kind == q36::PromptBoundaryKind::TurnOpener && aliases[1].depth == 5,
           "boundary aliases dropped the oldest turn openers rather than the newest");
    expect(aliases.back().rewrite_frontier &&
               std::none_of(aliases.begin(), aliases.end() - 1,
                            [](const auto& alias) { return alias.rewrite_frontier; }),
           "only the boundary at the turn-rewrite frontier is marked as such");
    expect(std::is_sorted(aliases.begin(), aliases.end(),
                          [](const auto& lhs, const auto& rhs) { return lhs.depth < rhs.depth; }) &&
               aliases.front().alias == *image::boundary_alias(domain, windowed, 1),
           "boundary aliases are not the per-depth aliases in ascending order");
}

void test_multiple_checkpoint_planning() {
    namespace image = q36::detail::continuation;
    const std::vector<std::uint32_t> none;
    const std::vector<std::uint32_t> one{3};
    const std::vector<std::uint32_t> two{3, 6};
    expect(image::next_prefill_checkpoint(0, one, 7, std::nullopt) == 3,
           "prefill plans the capture boundary before a later turn checkpoint");
    expect(image::next_prefill_checkpoint(3, none, 7, std::nullopt) == 7,
           "prefill plans the later turn checkpoint after the capture");
    expect(!image::next_prefill_checkpoint(7, none, 7, std::nullopt),
           "prefill does not recapture a completed checkpoint");
    expect(image::next_prefill_checkpoint(0, one, 3, std::nullopt) == 3,
           "coincident capture and turn boundaries use one device snapshot");
    expect(image::next_prefill_checkpoint(0, two, 7, std::nullopt) == 3 &&
               image::next_prefill_checkpoint(3, two, 7, std::nullopt) == 6 &&
               image::next_prefill_checkpoint(6, two, 7, std::nullopt) == 7,
           "prefill visits every capture boundary in order");
    // The user-turn anchor always sits between the stable prefix and the rewrite frontier, so a
    // cold prefill must land on all three in ascending order.
    expect(image::next_prefill_checkpoint(0, one, 7, 5) == 3,
           "capture boundary precedes the user turn anchor");
    expect(image::next_prefill_checkpoint(3, none, 7, 5) == 5,
           "user turn anchor precedes the turn checkpoint");
    expect(image::next_prefill_checkpoint(5, none, 7, std::nullopt) == 7,
           "turn checkpoint follows a captured user turn anchor");
    expect(image::next_prefill_checkpoint(0, none, 7, 5) == 5,
           "user turn anchor is planned without a capture boundary");
}

// The direct linear-state segment export is what the continuation image persists, so it must be
// byte-identical to serializing the per-layer state image, which is what `decode_linear` and
// every image already in L2/L3 assume.
void test_linear_segment_export() {
    int devices                 = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&devices);
    if (count_err != cudaSuccess || devices == 0) {
        (void)cudaGetLastError();
        std::cout << "note: linear segment export check skipped without a CUDA device\n";
        return;
    }
    namespace image = q36::detail::continuation;
    ninfer::DeviceContext ctx(0);
    ninfer::LayoutBuilder builder;
    const auto layout = ninfer::plan_linear_attention_state_pool(
        builder, ninfer::LinearAttentionStatePoolSpec{.layers         = 3,
                                                      .conv_channels  = 10,
                                                      .conv_width     = 3,
                                                      .value_heads    = 4,
                                                      .value_head_dim = 5,
                                                      .key_head_dim   = 6,
                                                      .slot_count     = 2,
                                                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena arena(builder.finish(256));
    ninfer::LinearAttentionStatePool pool({arena.base(), arena.capacity()}, layout);
    for (std::uint32_t layer = 0; layer < pool.layer_count(); ++layer) {
        for (std::int32_t slot = 0; slot < pool.slot_count(); ++slot) {
            const ninfer::Tensor conv      = pool.conv_slot(layer, slot);
            const ninfer::Tensor recurrent = pool.recurrent_slot(layer, slot);
            std::vector<std::uint8_t> conv_bytes(conv.bytes());
            std::vector<std::uint8_t> recurrent_bytes(recurrent.bytes());
            for (std::size_t i = 0; i < conv_bytes.size(); ++i) {
                conv_bytes[i] = static_cast<std::uint8_t>(17 * layer + 5 * slot + i);
            }
            for (std::size_t i = 0; i < recurrent_bytes.size(); ++i) {
                recurrent_bytes[i] = static_cast<std::uint8_t>(31 * layer + 7 * slot + 3 * i);
            }
            CUDA_CHECK(cudaMemcpyAsync(conv.data, conv_bytes.data(), conv_bytes.size(),
                                       cudaMemcpyHostToDevice, ctx.stream));
            CUDA_CHECK(cudaMemcpyAsync(recurrent.data, recurrent_bytes.data(),
                                       recurrent_bytes.size(), cudaMemcpyHostToDevice,
                                       ctx.stream));
        }
    }
    ctx.synchronize();
    // A ring far smaller than one layer forces the segment through many slot rotations.
    ninfer::PinnedTransferBuffer transfer(64);
    for (std::int32_t slot = 0; slot < pool.slot_count(); ++slot) {
        const auto reference = image::encode_linear(
            ninfer::export_linear_attention_state(pool, slot, transfer, ctx.stream));
        const auto direct = image::export_linear_segment(pool, slot, transfer, ctx.stream);
        expect(direct == reference, "direct linear-state segment matches the serialized image");
        const auto decoded = image::decode_linear(direct, pool);
        expect(decoded.layers == pool.layer_count() &&
                   decoded.conv.size() == pool.layer_count() &&
                   decoded.recurrent.size() == pool.layer_count(),
               "direct linear-state segment decodes");
    }
}

} // namespace

int main() {
    test_topology();
    test_decoder_layout();
    test_round_layout();
    test_mtp_alignment();
    test_vision_control();
    test_prefix_identity();
    test_continuation_reuse_depth();
    test_prefix_identity_snapshot();
    test_continuation_image_codec();
    test_decision_state_identity();
    test_continuation_prefix_filter_digest();
    test_stable_alias_identity();
    test_multiple_checkpoint_planning();
    test_linear_segment_export();
    if (failures != 0) {
        std::cerr << failures << " Qwen3.8 runtime mechanism checks failed\n";
        return 1;
    }
    std::cout << "Qwen3.8 runtime mechanism checks passed\n";
    return 0;
}
