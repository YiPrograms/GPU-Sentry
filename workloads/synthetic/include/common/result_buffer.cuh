#pragma once
#include "mining_types.cuh"

#ifndef MAX_RESULTS
#define MAX_RESULTS 1024u
#endif

__device__ __forceinline__ void store_result(MiningResult* results, unsigned int* result_count, uint64_t nonce, const uint32_t digest[8]) {
    unsigned int slot = atomicAdd(result_count, 1u);
    if (slot < MAX_RESULTS) {
        results[slot].nonce = nonce;
        #pragma unroll
        for (int i = 0; i < 8; ++i) results[slot].digest[i] = digest[i];
        results[slot].found = 1u;
    }
}
