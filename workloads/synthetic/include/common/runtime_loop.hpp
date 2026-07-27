#pragma once
#include <chrono>
#include <cstdio>
#include <cstdint>

inline void print_standard_summary(const char* algorithm, const char* variant, int runtime_seconds, int threads_per_block, unsigned long long total_launches, unsigned long long total_nonces, unsigned int result_count, uint32_t checksum) {
    std::printf("algorithm=%s\n", algorithm);
    std::printf("variant=%s\n", variant);
    std::printf("runtime_seconds=%d\n", runtime_seconds);
    std::printf("threads_per_block=%d\n", threads_per_block);
    std::printf("total_launches=%llu\n", total_launches);
    std::printf("total_nonces=%llu\n", total_nonces);
    std::printf("result_count=%u\n", result_count);
    std::printf("checksum=0x%08x\n", checksum);
    std::printf("status=ok\n");
}
