#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

#include "../../../include/common/mining_types.cuh"
#include "../../../include/common/cuda_check.cuh"
#include "../../../include/common/cli_args.hpp"
#include "../../../include/common/runtime_loop.hpp"
#include "../../../include/common/result_buffer.cuh"
#include "../../../include/common/hash_utils.cuh"
#include "../../../include/primitives/sha256.cuh"
#include "../../../include/primitives/keccak.cuh"
#include "../../../include/primitives/blake3.cuh"
#include "../../../include/primitives/groestl.cuh"
#include "../../../include/primitives/skein.cuh"
#include "../../../include/primitives/cubehash.cuh"
#include "../../../include/primitives/jh.cuh"
#include "../../../include/primitives/aes_like.cuh"
#include "../../../include/primitives/memory_mix.cuh"
#include "../../../include/primitives/progpow_common.cuh"
#include "../../../include/primitives/heavyhash_common.cuh"
#include "../../../include/primitives/equihash_common.cuh"
#include "../../../include/primitives/cuckoo_common.cuh"
#include "../../../include/primitives/lyra2.cuh"
#include "../../../include/primitives/scrypt_like.cuh"
#include "../../../include/primitives/randomx_lite_vm.cuh"

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif
#ifndef NUM_BLOCKS
#define NUM_BLOCKS 4096
#endif
#ifndef NONCES_PER_THREAD
#define NONCES_PER_THREAD 1
#endif

// Representative synthetic implementation for SASS dataset generation.
// Not intended to be a complete or profitable miner.

__global__ void pyrinhash_v2_init_matrix_kernel(uint32_t* workspace, size_t words, uint32_t seed) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint64_t i = tid; i < words; i += uint64_t(blockDim.x) * gridDim.x) {
        uint32_t x = avalanche32(uint32_t(i) ^ seed ^ 0x18a6759eu);
        workspace[i] = rotl32(x + uint32_t(i * 2654435761ull), int(i & 31));
    }
}

__global__ void pyrinhash_v2_search_kernel(const MiningJob* job, MiningResult* results, unsigned int* result_count, uint32_t* checksum) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = job->start_nonce + tid * uint64_t(job->nonce_count);
    uint32_t digest[8];
    sha256_digest(job->header, job->header_len, nonce, digest); heavyhash_common_mix(job->header, job->header_len, nonce ^ digest[0], digest);
    uint32_t local = digest[0] ^ digest[1] ^ digest[2] ^ digest[3] ^ digest[4] ^ digest[5] ^ digest[6] ^ digest[7];
    if ((digest[0] & 0x00ffffffu) == 0) { store_result(results, result_count, nonce, digest); }
    atomicAdd(checksum, local);
}

__global__ void pyrinhash_v2_final_hash_kernel(uint32_t* checksum, const uint32_t* workspace, size_t words) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (words > 0) atomicAdd(checksum, workspace[tid % words] ^ uint32_t(tid * 0x9e3779b9u));
}

int main(int argc, char** argv) {
    CliArgs args;
    if (!parse_cli_args(argc, argv, args)) { print_usage(argv[0]); return 1; }
    CUDA_CHECK(cudaFree(0));
    MiningJob h_job = {};
    h_job.header_len = 80;
    h_job.start_nonce = 0;
    h_job.nonce_count = static_cast<uint64_t>(args.nonces_per_thread);
    for (int i = 0; i < 128; i++) h_job.header[i] = static_cast<uint8_t>((i * 131 + args.seed) & 0xff);
    for (int i = 0; i < 8; i++) h_job.target_words[i] = 0x00ffffffu >> (i & 3);

    MiningJob* d_job = nullptr;
    MiningResult* d_results = nullptr;
    unsigned int* d_result_count = nullptr;
    uint32_t* d_checksum = nullptr;
    uint32_t* d_workspace = nullptr;
    size_t workspace_words = 0;
    workspace_words = 1u << 20;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_workspace, 0, workspace_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_job, sizeof(MiningJob)));
    CUDA_CHECK(cudaMalloc(&d_results, sizeof(MiningResult) * MAX_RESULTS));
    CUDA_CHECK(cudaMalloc(&d_result_count, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_checksum, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_job, &h_job, sizeof(MiningJob), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_result_count, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_checksum, 0, sizeof(uint32_t)));
    
    pyrinhash_v2_init_matrix_kernel<<<args.blocks, args.threads>>>(d_workspace, workspace_words, args.seed + 0x5adbu);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto start = std::chrono::steady_clock::now();
    auto end_time = start + std::chrono::seconds(args.runtime_seconds);
    uint64_t total_launches = 0;
    uint64_t total_nonces = 0;
    while (std::chrono::steady_clock::now() < end_time) {
        h_job.start_nonce = total_nonces;
        CUDA_CHECK(cudaMemcpy(d_job, &h_job, sizeof(MiningJob), cudaMemcpyHostToDevice));
        pyrinhash_v2_search_kernel<<<args.blocks, args.threads>>>(d_job, d_results, d_result_count, d_checksum);
        CUDA_CHECK(cudaGetLastError());
        pyrinhash_v2_final_hash_kernel<<<args.blocks, args.threads>>>(d_checksum, d_workspace, workspace_words);
        CUDA_CHECK(cudaGetLastError());
        total_launches++;
        total_nonces += static_cast<uint64_t>(args.blocks) * static_cast<uint64_t>(args.threads) * static_cast<uint64_t>(args.nonces_per_thread);
        if (args.sync_every > 0 && total_launches % static_cast<uint64_t>(args.sync_every) == 0) CUDA_CHECK(cudaDeviceSynchronize());
        sleep_after_launch_if_requested(args);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned int h_result_count = 0;
    uint32_t h_checksum = 0;
    CUDA_CHECK(cudaMemcpy(&h_result_count, d_result_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_checksum, d_checksum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    print_standard_summary("pyrinhash_v2", "split", args.runtime_seconds, args.threads, (unsigned long long)total_launches, (unsigned long long)total_nonces, h_result_count, h_checksum);
    cudaFree(d_job); cudaFree(d_results); cudaFree(d_result_count); cudaFree(d_checksum);
    cudaFree(d_workspace);
    return 0;
}
