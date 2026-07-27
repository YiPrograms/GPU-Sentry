#pragma once
#include <cstdint>

struct MiningJob {
    uint8_t header[128];
    uint32_t header_len;
    uint64_t start_nonce;
    uint64_t nonce_count;
    uint32_t target_words[8];
};

struct MiningResult {
    uint64_t nonce;
    uint32_t digest[8];
    uint32_t found;
};

struct KernelConfig {
    uint32_t num_blocks;
    uint32_t threads_per_block;
    uint32_t nonces_per_thread;
    uint32_t iterations;
};
