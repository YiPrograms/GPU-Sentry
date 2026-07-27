#pragma once
#include <cstdint>
#include "rotate.cuh"
#include "endian.cuh"

__device__ __forceinline__ uint32_t avalanche32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}

__device__ __forceinline__ uint32_t header_word(const uint8_t* header, uint32_t len, uint32_t idx) {
    uint32_t off = (idx * 4u) % len;
    return load_le32(header + off);
}

__device__ __forceinline__ void seed_digest(const uint8_t* header, uint32_t len, uint64_t nonce, uint32_t tag, uint32_t out[8]) {
    uint32_t lo = uint32_t(nonce);
    uint32_t hi = uint32_t(nonce >> 32);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t w = header_word(header, len, i + tag);
        out[i] = avalanche32(w ^ lo ^ rotl32(hi + tag * 0x9e3779b9u, i + 1) ^ (0x6a09e667u + i * 0x3c6ef372u));
    }
}

__device__ __forceinline__ void mix_digest_round(uint32_t s[8], uint32_t c) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t a = s[i];
        uint32_t b = s[(i + 1) & 7];
        uint32_t d = s[(i + 5) & 7];
        s[i] = rotl32((a + (b ^ c)) ^ ((~d) + 0x9e3779b9u + i), (i * 5 + 7) & 31);
    }
}

__device__ __forceinline__ void digest_to_result(uint32_t dst[8], const uint32_t src[8]) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) dst[i] = src[i];
}
