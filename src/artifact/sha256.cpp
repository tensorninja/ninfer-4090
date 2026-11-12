#include "artifact/sha256.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(_MSC_VER)
#    define NINFER_SHA256_X86 1
#    include <immintrin.h>
#endif

namespace ninfer::artifact {
namespace {

constexpr std::array<std::uint32_t, 64> kRound{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
    0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
    0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
    0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
    0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
    0xc67178f2U,
};

std::uint32_t load_be32(const std::uint8_t* bytes) noexcept {
    return (static_cast<std::uint32_t>(bytes[0]) << 24U) |
           (static_cast<std::uint32_t>(bytes[1]) << 16U) |
           (static_cast<std::uint32_t>(bytes[2]) << 8U) | static_cast<std::uint32_t>(bytes[3]);
}

void store_be32(std::uint32_t value, std::uint8_t* bytes) noexcept {
    bytes[0] = static_cast<std::uint8_t>(value >> 24U);
    bytes[1] = static_cast<std::uint8_t>(value >> 16U);
    bytes[2] = static_cast<std::uint8_t>(value >> 8U);
    bytes[3] = static_cast<std::uint8_t>(value);
}

} // namespace

void Sha256::process_block(const std::uint8_t* block) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t i = 0; i < 16; ++i) { words[i] = load_be32(block + 4 * i); }
    for (std::size_t i = 16; i < words.size(); ++i) {
        const std::uint32_t s0 =
            std::rotr(words[i - 15], 7) ^ std::rotr(words[i - 15], 18) ^ (words[i - 15] >> 3U);
        const std::uint32_t s1 =
            std::rotr(words[i - 2], 17) ^ std::rotr(words[i - 2], 19) ^ (words[i - 2] >> 10U);
        words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t i = 0; i < words.size(); ++i) {
        const std::uint32_t sum1     = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
        const std::uint32_t choose   = (e & f) ^ (~e & g);
        const std::uint32_t t1       = h + sum1 + choose + kRound[i] + words[i];
        const std::uint32_t sum0     = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t t2       = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}

#if NINFER_SHA256_X86
namespace {

// Intel SHA extensions. The compression function is unchanged; only its message schedule and
// round arithmetic move into sha256rnds2/msg1/msg2, so the digest is bit-identical to the
// portable path above. The qualification test asserts that directly, on NIST vectors and on
// randomly split updates.
#    define NINFER_SHA_TARGET __attribute__((target("sha,sse4.1")))

// One quartet of rounds: sha256rnds2 over the low then the high half of the message.
NINFER_SHA_TARGET inline void sha_quartet(__m128i& state0, __m128i& state1, __m128i message) {
    state1  = _mm_sha256rnds2_epu32(state1, state0, message);
    message = _mm_shuffle_epi32(message, 0x0E);
    state0  = _mm_sha256rnds2_epu32(state0, state1, message);
}

// Rounds 12-59 share one shape: extend the schedule for the quartet after next, run this
// quartet, and optionally advance the register that feeds the one after it.
NINFER_SHA_TARGET inline void sha_scheduled_quartet(__m128i& state0, __m128i& state1,
                                                    __m128i current, __m128i previous,
                                                    __m128i& next, __m128i* ahead,
                                                    long long high, long long low) {
    __m128i message = _mm_add_epi32(current, _mm_set_epi64x(high, low));
    state1          = _mm_sha256rnds2_epu32(state1, state0, message);
    next            = _mm_add_epi32(next, _mm_alignr_epi8(current, previous, 4));
    next            = _mm_sha256msg2_epu32(next, current);
    message         = _mm_shuffle_epi32(message, 0x0E);
    state0          = _mm_sha256rnds2_epu32(state0, state1, message);
    if (ahead != nullptr) { *ahead = _mm_sha256msg1_epu32(*ahead, current); }
}

NINFER_SHA_TARGET void process_blocks_shani(std::array<std::uint32_t, 8>& state,
                                            const std::uint8_t* data, std::size_t blocks) {
    const __m128i mask = _mm_set_epi64x(0x0c0d0e0f08090a0bLL, 0x0405060700010203LL);
    __m128i tmp        = _mm_loadu_si128(reinterpret_cast<const __m128i*>(state.data()));
    __m128i state1     = _mm_loadu_si128(reinterpret_cast<const __m128i*>(state.data() + 4));
    tmp                = _mm_shuffle_epi32(tmp, 0xB1);
    state1             = _mm_shuffle_epi32(state1, 0x1B);
    __m128i state0     = _mm_alignr_epi8(tmp, state1, 8);
    state1             = _mm_blend_epi16(state1, tmp, 0xF0);

    for (std::size_t block = 0; block < blocks; ++block, data += 64) {
        const __m128i abef_save = state0;
        const __m128i cdgh_save = state1;

        __m128i msg0 =
            _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data)), mask);
        __m128i msg1 =
            _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 16)), mask);
        __m128i msg2 =
            _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 32)), mask);
        __m128i msg3 =
            _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 48)), mask);

        sha_quartet(state0, state1,
                    _mm_add_epi32(msg0, _mm_set_epi64x(0xE9B5DBA5B5C0FBCFLL,
                                                       0x71374491428A2F98LL)));
        sha_quartet(state0, state1,
                    _mm_add_epi32(msg1, _mm_set_epi64x(0xAB1C5ED5923F82A4LL,
                                                       0x59F111F13956C25BLL)));
        msg0 = _mm_sha256msg1_epu32(msg0, msg1);
        sha_quartet(state0, state1,
                    _mm_add_epi32(msg2, _mm_set_epi64x(0x550C7DC3243185BELL,
                                                       0x12835B01D807AA98LL)));
        msg1 = _mm_sha256msg1_epu32(msg1, msg2);

        sha_scheduled_quartet(state0, state1, msg3, msg2, msg0, &msg2,
                              0xC19BF1749BDC06A7LL, 0x80DEB1FE72BE5D74LL);
        sha_scheduled_quartet(state0, state1, msg0, msg3, msg1, &msg3,
                              0x240CA1CC0FC19DC6LL, 0xEFBE4786E49B69C1LL);
        sha_scheduled_quartet(state0, state1, msg1, msg0, msg2, &msg0,
                              0x76F988DA5CB0A9DCLL, 0x4A7484AA2DE92C6FLL);
        sha_scheduled_quartet(state0, state1, msg2, msg1, msg3, &msg1,
                              0xBF597FC7B00327C8LL, 0xA831C66D983E5152LL);
        sha_scheduled_quartet(state0, state1, msg3, msg2, msg0, &msg2,
                              0x1429296706CA6351LL, 0xD5A79147C6E00BF3LL);
        sha_scheduled_quartet(state0, state1, msg0, msg3, msg1, &msg3,
                              0x53380D134D2C6DFCLL, 0x2E1B213827B70A85LL);
        sha_scheduled_quartet(state0, state1, msg1, msg0, msg2, &msg0,
                              0x92722C8581C2C92ELL, 0x766A0ABB650A7354LL);
        sha_scheduled_quartet(state0, state1, msg2, msg1, msg3, &msg1,
                              0xC76C51A3C24B8B70LL, 0xA81A664BA2BFE8A1LL);
        sha_scheduled_quartet(state0, state1, msg3, msg2, msg0, &msg2,
                              0x106AA070F40E3585LL, 0xD6990624D192E819LL);
        sha_scheduled_quartet(state0, state1, msg0, msg3, msg1, &msg3,
                              0x34B0BCB52748774CLL, 0x1E376C0819A4C116LL);
        sha_scheduled_quartet(state0, state1, msg1, msg0, msg2, nullptr,
                              0x682E6FF35B9CCA4FLL, 0x4ED8AA4A391C0CB3LL);
        sha_scheduled_quartet(state0, state1, msg2, msg1, msg3, nullptr,
                              0x8CC7020884C87814LL, 0x78A5636F748F82EELL);
        sha_quartet(state0, state1,
                    _mm_add_epi32(msg3, _mm_set_epi64x(0xC67178F2BEF9A3F7LL,
                                                       0xA4506CEB90BEFFFALL)));

        state0 = _mm_add_epi32(state0, abef_save);
        state1 = _mm_add_epi32(state1, cdgh_save);
    }

    tmp    = _mm_shuffle_epi32(state0, 0x1B);
    state1 = _mm_shuffle_epi32(state1, 0xB1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(state.data()), _mm_blend_epi16(tmp, state1, 0xF0));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(state.data() + 4), _mm_alignr_epi8(state1, tmp, 8));
}

bool have_sha_extensions() noexcept {
    static const bool supported = __builtin_cpu_supports("sha") != 0;
    return supported;
}

#    undef NINFER_SHA_TARGET

} // namespace
#endif

void Sha256::process_blocks(const std::uint8_t* data, std::size_t blocks) {
#if NINFER_SHA256_X86
    if (have_sha_extensions()) {
        process_blocks_shani(state_, data, blocks);
        return;
    }
#endif
    for (std::size_t i = 0; i < blocks; ++i) { process_block(data + i * 64); }
}

void Sha256::update(std::span<const std::byte> bytes) {
    if (finished_) { throw std::logic_error("SHA-256 digest is already finalized"); }
    if (bytes.size() > std::numeric_limits<std::uint64_t>::max() - total_bytes_ ||
        total_bytes_ + bytes.size() > std::numeric_limits<std::uint64_t>::max() / 8U) {
        throw std::invalid_argument("payload is too large to fingerprint");
    }
    total_bytes_ += bytes.size();
    const auto* input = reinterpret_cast<const std::uint8_t*>(bytes.data());
    std::size_t cursor = 0;
    if (pending_size_ != 0) {
        const std::size_t copied = std::min(bytes.size(), pending_.size() - pending_size_);
        std::copy_n(input, copied, pending_.data() + pending_size_);
        pending_size_ += copied;
        cursor += copied;
        if (pending_size_ == pending_.size()) {
            process_block(pending_.data());
            pending_size_ = 0;
        } else {
            return;
        }
    }
    const std::size_t blocks = (bytes.size() - cursor) / pending_.size();
    if (blocks != 0) {
        process_blocks(input + cursor, blocks);
        cursor += blocks * pending_.size();
    }
    pending_size_ = bytes.size() - cursor;
    std::copy_n(input + cursor, pending_size_, pending_.data());
}

Sha256Digest Sha256::finish() {
    if (finished_) { throw std::logic_error("SHA-256 digest is already finalized"); }
    finished_ = true;
    const std::uint64_t bit_count = total_bytes_ * 8U;
    pending_[pending_size_++] = 0x80U;
    if (pending_size_ > 56) {
        std::fill(pending_.begin() + static_cast<std::ptrdiff_t>(pending_size_), pending_.end(), 0);
        process_block(pending_.data());
        pending_size_ = 0;
    }
    std::fill(pending_.begin() + static_cast<std::ptrdiff_t>(pending_size_), pending_.begin() + 56,
              0);
    for (std::size_t i = 0; i < 8; ++i) {
        pending_[63 - i] = static_cast<std::uint8_t>(bit_count >> (8U * i));
    }
    process_block(pending_.data());

    Sha256Digest digest{};
    for (std::size_t i = 0; i < state_.size(); ++i) {
        store_be32(state_[i], digest.data() + 4 * i);
    }
    return digest;
}

} // namespace ninfer::artifact
