#pragma once
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

struct CliArgs {
    int runtime_seconds = 0;
    int blocks = 4096;
    int threads = 256;
    int nonces_per_thread = 1;
    int dataset_mb = 64;
    int scratchpad_mb = 64;
    unsigned int seed = 1;
    int sync_every = 1;
    int sleep_between_launches_ms = 0;
    int sleep_between_launches_us = 0;
};

inline bool parse_positive_i64(const char* s, long long& out) {
    if (s == nullptr || *s == '\0' || *s == '-') return false;
    errno = 0;
    char* end = nullptr;
    long long v = std::strtoll(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v <= 0 || v > INT_MAX) return false;
    out = v;
    return true;
}

inline bool parse_nonnegative_i64(const char* s, long long& out) {
    if (s == nullptr || *s == '\0' || *s == '-') return false;
    errno = 0;
    char* end = nullptr;
    long long v = std::strtoll(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < 0 || v > INT_MAX) return false;
    out = v;
    return true;
}

inline bool parse_cli_args(int argc, char** argv, CliArgs& args) {
    if (argc < 2) return false;
    long long runtime = 0;
    if (!parse_positive_i64(argv[1], runtime)) return false;
    args.runtime_seconds = static_cast<int>(runtime);
    for (int i = 2; i < argc; ++i) {
        if (i + 1 >= argc) return false;
        long long value = 0;
        if (std::strcmp(argv[i], "--sleep-between-launches-ms") == 0 || std::strcmp(argv[i], "--sleep-between-launches-us") == 0) {
            if (!parse_nonnegative_i64(argv[i + 1], value)) return false;
        } else if (!parse_positive_i64(argv[i + 1], value)) {
            return false;
        }
        if (std::strcmp(argv[i], "--blocks") == 0) args.blocks = static_cast<int>(value);
        else if (std::strcmp(argv[i], "--threads") == 0) args.threads = static_cast<int>(value);
        else if (std::strcmp(argv[i], "--nonces-per-thread") == 0) args.nonces_per_thread = static_cast<int>(value);
        else if (std::strcmp(argv[i], "--dataset-mb") == 0) args.dataset_mb = static_cast<int>(value);
        else if (std::strcmp(argv[i], "--scratchpad-mb") == 0) args.scratchpad_mb = static_cast<int>(value);
        else if (std::strcmp(argv[i], "--seed") == 0) args.seed = static_cast<unsigned int>(value);
        else if (std::strcmp(argv[i], "--sync-every") == 0) args.sync_every = static_cast<int>(value);
        else if (std::strcmp(argv[i], "--sleep-between-launches-ms") == 0) args.sleep_between_launches_ms = static_cast<int>(value);
        else if (std::strcmp(argv[i], "--sleep-between-launches-us") == 0) args.sleep_between_launches_us = static_cast<int>(value);
        else return false;
        ++i;
    }
    return args.blocks > 0 && args.threads > 0 && args.nonces_per_thread > 0 && args.dataset_mb > 0 && args.scratchpad_mb > 0;
}

inline void sleep_after_launch_if_requested(const CliArgs& args) {
    const int total_sleep_us = args.sleep_between_launches_ms * 1000 + args.sleep_between_launches_us;
    if (total_sleep_us > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(total_sleep_us));
    }
}

inline void print_usage(const char* program) {
    std::fprintf(stderr, "usage: %s <runtime_seconds> [--blocks N] [--threads N] [--nonces-per-thread N] [--dataset-mb N] [--scratchpad-mb N] [--seed N] [--sync-every N] [--sleep-between-launches-ms N] [--sleep-between-launches-us N]\n", program);
}
