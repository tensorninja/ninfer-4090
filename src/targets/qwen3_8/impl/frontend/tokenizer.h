#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ninfer::targets::qwen3_8::frontend_internal {

struct EncodeOptions {
    bool parse_added_tokens = true;
};

struct DecodeOptions {
    bool skip_special_tokens = false;
    std::vector<int> stop_token_ids;
};

struct AddedToken {
    int id = -1;
    std::string content;
    bool single_word = false;
    bool lstrip      = false;
    bool rstrip      = false;
    bool normalized  = false;
    bool special     = false;
};

struct TokenizerResources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view generation_config_json;
};

class Tokenizer;

// The complete encoding of one text together with the added-token segmentation that produced
// it. Added tokens cut the text into independently encoded segments, so the encoding of any byte
// prefix shares every segment that ends at or before the cut and can differ from the complete
// encoding only inside the segment holding the cut. `prefix` relates encode(text[0, cut)) to the
// complete encoding by re-encoding that one partial segment instead of the whole prefix. The
// text must outlive the encoding.
class SegmentedEncoding {
public:
    struct PrefixRelation {
        // Longest common token prefix of encode(text[0, cut)) and the complete encoding.
        std::size_t common_tokens = 0;
        // encode(text[0, cut)) is exactly its first `common_tokens` tokens.
        bool exact = false;
    };

    [[nodiscard]] const std::vector<int>& ids() const noexcept { return ids_; }
    [[nodiscard]] std::vector<int> release_ids() noexcept { return std::move(ids_); }
    // Throws std::out_of_range past the text and std::invalid_argument when the partial
    // segment cannot be encoded on its own (a cut inside a multi-byte code point).
    [[nodiscard]] PrefixRelation prefix(std::size_t cut) const;

private:
    friend class Tokenizer;
    struct Segment {
        std::size_t byte_begin  = 0;
        std::size_t byte_end    = 0;
        std::size_t token_begin = 0;
        std::size_t token_end   = 0;
    };

    const Tokenizer* tokenizer_ = nullptr;
    std::string_view text_;
    std::vector<int> ids_;
    std::vector<Segment> segments_;
};

class Tokenizer {
public:
    explicit Tokenizer(TokenizerResources resources);

    std::vector<int> encode(std::string_view text, EncodeOptions options = {}) const;
    [[nodiscard]] SegmentedEncoding encode_segmented(std::string_view text) const;
    std::string decode(std::span<const int> ids, DecodeOptions options = {}) const;
    std::string decode_token_bytes(int id, bool skip_special_tokens = false) const;

    [[nodiscard]] const std::vector<int>& default_stop_token_ids() const noexcept {
        return default_stop_token_ids_;
    }

    // The id of a token named by its exact content: an added token first, then the base
    // vocabulary (HF convert_tokens_to_ids). Throws std::out_of_range when absent.
    [[nodiscard]] int token_id(std::string_view content) const;
    [[nodiscard]] bool is_special_token(int id) const noexcept;
    [[nodiscard]] bool is_valid_token(int id) const noexcept;
    [[nodiscard]] bool has_exact_token_domain(std::size_t size) const noexcept;

private:
    void encode_added_token_segments(std::string_view text, std::vector<int>& ids,
                                     std::vector<SegmentedEncoding::Segment>* segments) const;

    std::vector<std::string> id_to_token_;
    std::vector<bool> valid_token_ids_;
    std::unordered_map<std::string, int> vocab_token_to_id_;
    std::unordered_map<std::string, int> bpe_merge_ranks_;
    bool has_bpe_merges_ = true;
    std::vector<AddedToken> added_tokens_;
    std::vector<int> default_stop_token_ids_;
};

} // namespace ninfer::targets::qwen3_8::frontend_internal
