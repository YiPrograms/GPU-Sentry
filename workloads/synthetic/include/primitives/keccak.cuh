#pragma once
#include <cstdint>
#include "../common/hash_utils.cuh"

// Reduced representative implementation for SASS dataset generation.
// Not intended for real cryptocurrency mining.
__device__ __forceinline__ void keccak_digest(const uint8_t* header, uint32_t len, uint64_t nonce, uint32_t out[8]) {
    seed_digest(header, len, nonce, 0x4f46u, out);
    #pragma unroll
    for (int r = 0; r < 6; ++r) {
        mix_digest_round(out, 0x7277af9cu + r * 0x9e3779b9u);
        out[r & 7] ^= rotl32(out[(r + 3) & 7] + uint32_t(nonce >> (r & 15)), (r * 3 + 11) & 31);
    }
}
