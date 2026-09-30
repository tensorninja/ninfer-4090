// Public-Op benchmark for A4 shared-prefix segmented attention.
//
// Each case populates [0, prefix) of one block-table row through A2, then times one A4 call
// (segment-causal attention over the cached shared prefix plus each segment's own BF16 K/V)
// against the per-segment A1 loop, which serves the same segments one A1 call at a time behind
// the same prefix, appending each segment to the cache first. Both entries use only the public Op
// contracts and their capacity queries; split counts and kernel routes are private and never
// enter the dispatch or the output schema.

#include "ninfer/ops/gqa_attention.h"

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHeadDim     = 256;
constexpr std::int32_t kQueryHeads  = 24;
constexpr std::int32_t kKvHeads     = 4;
constexpr std::int32_t kQuantGroup  = 64;
constexpr std::int32_t kQuantGroups = kHeadDim / kQuantGroup;
constexpr float kScale              = 0.0625F;
constexpr std::int32_t kFillChunk   = 8192;
constexpr std::size_t kFlushBytes   = std::size_t{256} << 20;

struct Codec {
    const char* name;
    DType dtype;
    bool packed_k;
    bool packed_v;
    bool rotated;
    bool e8_root;

    [[nodiscard]] bool quantized() const { return dtype == DType::I8; }

    [[nodiscard]] std::int32_t k_row_bytes() const {
        if (!quantized()) return kHeadDim * 2;
        return e8_root ? kHeadDim / 4 : (packed_k ? kHeadDim / 2 : kHeadDim);
    }

    [[nodiscard]] std::int32_t v_row_bytes() const {
        if (!quantized()) return kHeadDim * 2;
        return packed_v ? kHeadDim / 2 : kHeadDim;
    }
};

constexpr Codec kCodecs[] = {
    {"bf16", DType::BF16, false, false, false, false},
    {"int8", DType::I8, false, false, false, false},
    {"rk8v4", DType::I8, false, true, true, false},
    {"rk4v4", DType::I8, true, true, true, false},
    {"rk2v4-e8", DType::I8, false, true, true, true},
};

struct Mix {
    std::int32_t segments;
    std::int32_t length;
};

enum class Entry : std::uint8_t { Segmented, PerSegment, Both };
enum class CacheMode : std::uint8_t { Cold, Warm };

struct Options {
    std::vector<std::int32_t> prefixes{8192, 65535};
    std::vector<Mix> mixes{{1, 40}, {6, 40}, {255, 3}};
    std::vector<const Codec*> codecs{&kCodecs[0], &kCodecs[3]};
    Entry entry     = Entry::Both;
    CacheMode cache = CacheMode::Cold;
    int warmup      = 3;
    int repeat      = 20;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: ninfer_gqa_attention_segmented_bench [--prefix L,...] [--mix SxT,...] "
                 "[--kv bf16|int8|rk8v4|rk4v4|rk2v4-e8|all[,...]] "
                 "[--entry segmented|per-segment|both] [--cache cold|warm] [--warmup N] "
                 "[--repeat N]\n",
                 message);
    std::exit(2);
}

std::int32_t parse_i32(std::string_view text, std::int32_t minimum, std::int32_t maximum,
                       const char* flag) {
    const std::string value(text);
    errno       = 0;
    char* end   = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed < minimum ||
        parsed > maximum) {
        usage(flag);
    }
    return static_cast<std::int32_t>(parsed);
}

std::vector<std::string_view> split_list(std::string_view text, const char* flag) {
    std::vector<std::string_view> items;
    while (true) {
        const std::size_t comma     = text.find(',');
        const std::string_view item = text.substr(0, comma);
        if (item.empty()) { usage(flag); }
        items.push_back(item);
        if (comma == std::string_view::npos) { break; }
        text.remove_prefix(comma + 1);
    }
    return items;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* flag) -> std::string_view {
            if (++index == argc) { usage(flag); }
            return argv[index];
        };
        if (argument == "--prefix") {
            options.prefixes.clear();
            for (const std::string_view item : split_list(next("--prefix"), "--prefix")) {
                options.prefixes.push_back(parse_i32(
                    item, 0, static_cast<std::int32_t>(ops::kGqaAttentionMaximumVisibleKeys),
                    "--prefix"));
            }
        } else if (argument == "--mix") {
            options.mixes.clear();
            for (const std::string_view item : split_list(next("--mix"), "--mix")) {
                const std::size_t x = item.find('x');
                if (x == std::string_view::npos) { usage("--mix expects SxT"); }
                options.mixes.push_back({parse_i32(item.substr(0, x), 1, 4096, "--mix"),
                                         parse_i32(item.substr(x + 1), 1, 65536, "--mix")});
            }
        } else if (argument == "--kv") {
            options.codecs.clear();
            for (const std::string_view item : split_list(next("--kv"), "--kv")) {
                if (item == "all") {
                    for (const Codec& codec : kCodecs) { options.codecs.push_back(&codec); }
                    continue;
                }
                const auto found = std::find_if(std::begin(kCodecs), std::end(kCodecs),
                                                [&](const Codec& codec) { return item == codec.name; });
                if (found == std::end(kCodecs)) { usage("--kv names an unknown codec"); }
                options.codecs.push_back(found);
            }
        } else if (argument == "--entry") {
            const std::string_view value = next("--entry");
            if (value == "segmented")
                options.entry = Entry::Segmented;
            else if (value == "per-segment")
                options.entry = Entry::PerSegment;
            else if (value == "both")
                options.entry = Entry::Both;
            else
                usage("--entry expects segmented, per-segment, or both");
        } else if (argument == "--cache") {
            const std::string_view value = next("--cache");
            if (value == "cold")
                options.cache = CacheMode::Cold;
            else if (value == "warm")
                options.cache = CacheMode::Warm;
            else
                usage("--cache expects cold or warm");
        } else if (argument == "--warmup") {
            options.warmup = parse_i32(next("--warmup"), 0, 10000, "--warmup");
        } else if (argument == "--repeat") {
            options.repeat = parse_i32(next("--repeat"), 1, 10000, "--repeat");
        } else if (argument == "--help" || argument == "-h") {
            usage("help");
        } else {
            usage("unknown argument");
        }
    }
    for (const std::int32_t prefix : options.prefixes) {
        for (const Mix& mix : options.mixes) {
            if (static_cast<std::int64_t>(prefix) + std::int64_t{mix.segments} * mix.length >
                ops::kGqaAttentionMaximumVisibleKeys) {
                usage("prefix + S x T exceeds the public GQA domain");
            }
        }
    }
    return options;
}

class Case {
public:
    Case(const Codec& codec, std::int32_t prefix, Mix mix)
        : codec_(codec), prefix_(prefix), mix_(mix), columns_(mix.segments * mix.length),
          logical_pages_((prefix + columns_ + kPagedKVPageSize - 1) / kPagedKVPageSize),
          q_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kQueryHeads * columns_)),
          k_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kKvHeads * columns_)),
          v_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kKvHeads * columns_)),
          out_(bench::make_zeros(static_cast<std::size_t>(kHeadDim) * kQueryHeads * columns_ * 2)),
          cache_k_(bench::make_zeros(plane_bytes(codec.k_row_bytes()))),
          cache_v_(bench::make_zeros(plane_bytes(codec.v_row_bytes()))),
          cache_k_scale_(bench::make_zeros(codec.quantized() ? plane_bytes(kQuantGroups * 2) : 1)),
          cache_v_scale_(bench::make_zeros(codec.quantized() ? plane_bytes(kQuantGroups * 2) : 1)),
          block_table_(static_cast<std::size_t>(logical_pages_) * sizeof(std::int32_t)),
          table_row_(bench::make_zeros(sizeof(std::int32_t))),
          segments_(static_cast<std::size_t>(2 * mix.segments) * sizeof(std::int32_t)),
          positions_(static_cast<std::size_t>(mix.length) * sizeof(std::int32_t)),
          segmented_workspace_bytes_(ops::gqa_attention_segmented_workspace_capacity_bytes(columns_)),
          per_segment_workspace_bytes_(ops::gqa_attention_workspace_capacity_bytes(
              kQueryHeads, codec.dtype, envelope(), 1, mix.length, mix.length)),
          segmented_workspace_(std::max<std::size_t>(segmented_workspace_bytes_, 1)),
          per_segment_workspace_(std::max<std::size_t>(per_segment_workspace_bytes_, 1)) {
        std::vector<std::int32_t> table(static_cast<std::size_t>(logical_pages_));
        for (std::int32_t page = 0; page < logical_pages_; ++page) {
            table[static_cast<std::size_t>(page)] = page;
        }
        block_table_.copy_from_host(table.data(), block_table_.bytes);
        std::vector<std::int32_t> segments(static_cast<std::size_t>(2 * mix.segments));
        for (std::int32_t segment = 0; segment < mix.segments; ++segment) {
            segments[static_cast<std::size_t>(2 * segment)]     = segment * mix.length;
            segments[static_cast<std::size_t>(2 * segment + 1)] = mix.length;
        }
        segments_.copy_from_host(segments.data(), segments_.bytes);
        std::vector<std::int32_t> positions(static_cast<std::size_t>(mix.length));
        for (std::int32_t token = 0; token < mix.length; ++token) {
            positions[static_cast<std::size_t>(token)] = prefix + token;
        }
        positions_.copy_from_host(positions.data(), positions_.bytes);
        populate_prefix();
    }

    void launch_segmented(cudaStream_t stream) {
        Tensor out = tensor(out_, kQueryHeads, 0, columns_);
        ops::gqa_attention_segmented(tensor(q_, kQueryHeads, 0, columns_),
                                     tensor(k_, kKvHeads, 0, columns_),
                                     tensor(v_, kKvHeads, 0, columns_),
                                     Tensor(segments_.p, DType::I32, {2, mix_.segments}),
                                     Tensor(table_row_.p, DType::I32, {1}), prefix_, kScale,
                                     batch_view(), segmented_workspace_, out, stream);
    }

    void launch_per_segment(cudaStream_t stream) {
        const Tensor positions(positions_.p, DType::I32, {mix_.length});
        const Tensor table_row(table_row_.p, DType::I32, {1});
        for (std::int32_t segment = 0; segment < mix_.segments; ++segment) {
            const std::int32_t begin = segment * mix_.length;
            Tensor out               = tensor(out_, kQueryHeads, begin, mix_.length);
            ops::gqa_attention(tensor(q_, kQueryHeads, begin, mix_.length),
                               tensor(k_, kKvHeads, begin, mix_.length),
                               tensor(v_, kKvHeads, begin, mix_.length), positions, Tensor{},
                               table_row, kScale, batch_view(), envelope(), per_segment_workspace_,
                               out, stream);
        }
    }

    [[nodiscard]] std::int32_t columns() const { return columns_; }
    [[nodiscard]] std::size_t segmented_workspace_bytes() const { return segmented_workspace_bytes_; }
    [[nodiscard]] std::size_t per_segment_workspace_bytes() const {
        return per_segment_workspace_bytes_;
    }

    // 4 * D * Hq * sum over columns of the visible key count.
    [[nodiscard]] double useful_flops() const {
        const double t        = mix_.length;
        const double visible  = mix_.segments * (t * prefix_ + t * (t + 1.0) * 0.5);
        return 4.0 * kHeadDim * kQueryHeads * visible;
    }

private:
    [[nodiscard]] ops::GqaExecutionEnvelope envelope() const {
        const auto visible = static_cast<std::uint32_t>(prefix_ + mix_.length);
        return {visible, visible};
    }

    [[nodiscard]] std::size_t plane_bytes(std::int32_t row_bytes) const {
        return static_cast<std::size_t>(row_bytes) * kPagedKVPageSize * kKvHeads *
               static_cast<std::size_t>(logical_pages_);
    }

    static Tensor tensor(DeviceBuffer& buffer, std::int32_t heads, std::int32_t begin,
                         std::int32_t columns) {
        auto* base = static_cast<std::uint16_t*>(buffer.p) +
                     static_cast<std::ptrdiff_t>(kHeadDim) * heads * begin;
        return Tensor(base, DType::BF16, {kHeadDim, heads, columns});
    }

    template <typename View>
    void bind_planes(View& view) {
        const DType k_dtype = !codec_.quantized()                    ? DType::BF16
                              : (codec_.packed_k || codec_.e8_root) ? DType::U8
                                                                     : DType::I8;
        const DType v_dtype = !codec_.quantized() ? DType::BF16
                              : codec_.packed_v   ? DType::U8
                                                  : DType::I8;
        const std::int32_t k_elements = codec_.quantized() ? codec_.k_row_bytes() : kHeadDim;
        const std::int32_t v_elements = codec_.quantized() ? codec_.v_row_bytes() : kHeadDim;
        view.k_pages = Tensor(cache_k_.p, k_dtype, {k_elements, kPagedKVPageSize, kKvHeads, logical_pages_});
        view.v_pages = Tensor(cache_v_.p, v_dtype, {v_elements, kPagedKVPageSize, kKvHeads, logical_pages_});
        if (codec_.quantized()) {
            view.k_scale_pages = Tensor(cache_k_scale_.p, DType::FP16,
                                        {kQuantGroups, kPagedKVPageSize, kKvHeads, logical_pages_});
            view.v_scale_pages = Tensor(cache_v_scale_.p, DType::FP16,
                                        {kQuantGroups, kPagedKVPageSize, kKvHeads, logical_pages_});
            view.quant_group = kQuantGroup;
        }
        view.head_dim     = kHeadDim;
        view.num_kv_heads = kKvHeads;
        view.dtype        = codec_.dtype;
        view.packed_v     = codec_.packed_v;
        view.rotate_k     = codec_.rotated;
        view.rotate_v     = codec_.rotated;
        view.packed_k     = codec_.packed_k;
        view.e8_root      = codec_.e8_root;
    }

    PagedKVBatchLayerView batch_view() {
        PagedKVBatchLayerView view;
        bind_planes(view);
        view.block_tables = Tensor(block_table_.p, DType::I32, {logical_pages_, 1});
        return view;
    }

    // Representative history through A2; attention cost does not depend on the values.
    void populate_prefix() {
        PagedKVLayerView view;
        bind_planes(view);
        view.block_table = Tensor(block_table_.p, DType::I32, {logical_pages_});
        for (std::int32_t begin = 0; begin < prefix_; begin += kFillChunk) {
            const std::int32_t tokens = std::min(kFillChunk, prefix_ - begin);
            const std::size_t elements =
                static_cast<std::size_t>(kHeadDim) * kKvHeads * static_cast<std::size_t>(tokens);
            const DeviceBuffer k = bench::make_bf16(elements);
            const DeviceBuffer v = bench::make_bf16(elements);
            std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
            for (std::int32_t token = 0; token < tokens; ++token) {
                positions[static_cast<std::size_t>(token)] = begin + token;
            }
            DeviceBuffer device_positions(positions.size() * sizeof(std::int32_t));
            device_positions.copy_from_host(positions.data(), device_positions.bytes);
            ops::gqa_kv_append(Tensor(k.p, DType::BF16, {kHeadDim, kKvHeads, tokens}),
                               Tensor(v.p, DType::BF16, {kHeadDim, kKvHeads, tokens}),
                               Tensor(device_positions.p, DType::I32, {tokens}), view, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    const Codec& codec_;
    std::int32_t prefix_;
    Mix mix_;
    std::int32_t columns_;
    std::int32_t logical_pages_;
    DeviceBuffer q_;
    DeviceBuffer k_;
    DeviceBuffer v_;
    DeviceBuffer out_;
    DeviceBuffer cache_k_;
    DeviceBuffer cache_v_;
    DeviceBuffer cache_k_scale_;
    DeviceBuffer cache_v_scale_;
    DeviceBuffer block_table_;
    DeviceBuffer table_row_;
    DeviceBuffer segments_;
    DeviceBuffer positions_;
    std::size_t segmented_workspace_bytes_;
    std::size_t per_segment_workspace_bytes_;
    WorkspaceArena segmented_workspace_;
    WorkspaceArena per_segment_workspace_;
};

template <typename Launch>
bench::ColdTiming measure(Launch&& launch, CacheMode cache, DeviceBuffer& flush,
                          cudaStream_t stream, const Options& options) {
    launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    bench::TimedGraph graph;
    graph.capture(stream, launch);
    return cache == CacheMode::Cold
               ? bench::measure_cold_graph(graph, flush, stream, options.warmup, options.repeat)
               : bench::measure_graph(graph, stream, options.warmup, options.repeat);
}

void report(const char* entry, const Codec& codec, std::int32_t prefix, const Mix& mix,
            const Case& data, std::size_t workspace_bytes, CacheMode cache,
            const bench::ColdTiming& timing) {
    const double tflops = data.useful_flops() / (timing.median_us * 1.0e-6) / 1.0e12;
    std::printf("entry=%-11s kv=%-8s prefix=%6d mix=%4dx%-3d N=%4d cache=%s workspace=%9zu "
                "median=%8.3f ms min=%8.3f ms p95=%8.3f ms math=%6.1f TFLOP/s\n",
                entry, codec.name, prefix, mix.segments, mix.length, data.columns(),
                cache == CacheMode::Cold ? "cold" : "warm", workspace_bytes,
                timing.median_us * 1.0e-3, timing.min_us * 1.0e-3, timing.p95_us * 1.0e-3, tflops);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        const Options options = parse_options(argc, argv);
        cudaStream_t stream   = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(options.cache == CacheMode::Cold ? kFlushBytes : 1);
        for (const Codec* codec : options.codecs) {
            for (const std::int32_t prefix : options.prefixes) {
                for (const Mix& mix : options.mixes) {
                    Case data(*codec, prefix, mix);
                    double segmented = 0.0;
                    if (options.entry != Entry::PerSegment) {
                        const bench::ColdTiming timing = measure(
                            [&](cudaStream_t launch_stream) { data.launch_segmented(launch_stream); },
                            options.cache, flush, stream, options);
                        report("segmented", *codec, prefix, mix, data,
                               data.segmented_workspace_bytes(), options.cache, timing);
                        segmented = timing.median_us;
                    }
                    if (options.entry != Entry::Segmented) {
                        const bench::ColdTiming timing = measure(
                            [&](cudaStream_t launch_stream) { data.launch_per_segment(launch_stream); },
                            options.cache, flush, stream, options);
                        report("per-segment", *codec, prefix, mix, data,
                               data.per_segment_workspace_bytes(), options.cache, timing);
                        if (segmented > 0.0) {
                            std::printf("speedup kv=%s prefix=%d mix=%dx%d "
                                        "per-segment/segmented=%.2fx\n",
                                        codec->name, prefix, mix.segments, mix.length,
                                        timing.median_us / segmented);
                        }
                    }
                }
            }
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_gqa_attention_segmented_bench: %s\n", error.what());
        return 1;
    }
}
