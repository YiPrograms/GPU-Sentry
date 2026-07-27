#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <climits>
#include <cerrno>
#include <cstddef>
#include <vector>
#include <cuda_runtime.h>

#include "cuda_check.cuh"
#include "cli_args.hpp"

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

#ifndef NUM_BLOCKS
#define NUM_BLOCKS 4096
#endif

struct BenignOptions {
    CliArgs cli;
    size_t size_mb = 256;
    size_t size = 1024;
    size_t iterations = 1;
    size_t nodes = 65536;
    size_t edges = 262144;
};

inline bool benign_parse_size(const char* s, size_t& out) {
    if (s == nullptr || *s == '\0' || *s == '-') return false;
    errno = 0;
    char* end = nullptr;
    unsigned long long v = std::strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v == 0 || v > static_cast<unsigned long long>(SIZE_MAX)) return false;
    out = static_cast<size_t>(v);
    return true;
}

inline bool parse_benign_args(int argc, char** argv, BenignOptions& options) {
    if (argc < 2) return false;
    long long runtime = 0;
    if (!parse_positive_i64(argv[1], runtime)) return false;
    options.cli.runtime_seconds = static_cast<int>(runtime);
    for (int i = 2; i < argc; ++i) {
        if (i + 1 >= argc) return false;
        size_t value = 0;
        if (!benign_parse_size(argv[i + 1], value)) return false;
        if (std::strcmp(argv[i], "--blocks") == 0) options.cli.blocks = static_cast<int>(value);
        else if (std::strcmp(argv[i], "--threads") == 0) options.cli.threads = static_cast<int>(value);
        else if (std::strcmp(argv[i], "--sync-every") == 0) options.cli.sync_every = static_cast<int>(value);
        else if (std::strcmp(argv[i], "--seed") == 0) options.cli.seed = static_cast<unsigned int>(value);
        else if (std::strcmp(argv[i], "--size-mb") == 0) options.size_mb = value;
        else if (std::strcmp(argv[i], "--size") == 0) options.size = value;
        else if (std::strcmp(argv[i], "--iterations") == 0) options.iterations = value;
        else if (std::strcmp(argv[i], "--nodes") == 0) options.nodes = value;
        else if (std::strcmp(argv[i], "--edges") == 0) options.edges = value;
        else return false;
        ++i;
    }
    return options.cli.blocks > 0 && options.cli.threads > 0 && options.cli.sync_every > 0;
}

inline void print_benign_usage(const char* program) {
    std::fprintf(stderr, "usage: %s <runtime_seconds> [--blocks N] [--threads N] [--sync-every N] [--seed N] [--size-mb N] [--size N] [--iterations N] [--nodes N] [--edges N]\n", program);
}

inline uint32_t benign_mix_u32(size_t tid, uint32_t seed) {
    uint32_t x = seed ^ static_cast<uint32_t>(tid * 747796405u + 2891336453u);
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    x *= 3266489917u;
    x ^= x >> 16;
    return x;
}

inline void benign_fill_u32(uint32_t* data, size_t n, uint32_t seed) {
    for (size_t tid = 0; tid < n; ++tid) {
        data[tid] = benign_mix_u32(tid, seed);
    }
}

inline void benign_fill_float(float* data, size_t n, uint32_t seed) {
    for (size_t tid = 0; tid < n; ++tid) {
        uint32_t x = seed ^ static_cast<uint32_t>(tid * 1664525u + 1013904223u);
        data[tid] = static_cast<float>((x & 0xffffu) - 32768) / 32768.0f;
    }
}

inline void benign_fill_double(double* data, size_t n, uint32_t seed) {
    for (size_t tid = 0; tid < n; ++tid) {
        uint32_t x = seed ^ static_cast<uint32_t>(tid * 22695477u + 1u);
        data[tid] = static_cast<double>((x & 0xffffu) - 32768) / 32768.0;
    }
}

inline void print_benign_summary(const char* category, const char* workload, const BenignOptions& options,
                                 uint64_t total_launches, uint64_t total_elements, uint32_t checksum) {
    std::printf("label=benign\n");
    std::printf("category=%s\n", category);
    std::printf("workload=%s\n", workload);
    std::printf("runtime_seconds=%d\n", options.cli.runtime_seconds);
    std::printf("threads_per_block=%d\n", options.cli.threads);
    std::printf("total_launches=%llu\n", (unsigned long long)total_launches);
    std::printf("total_elements=%llu\n", (unsigned long long)total_elements);
    std::printf("checksum=0x%08x\n", checksum);
    std::printf("status=ok\n");
}
