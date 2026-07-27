#include <algorithm>
#include <cmath>
#include "../../../include/common/benign_runtime.cuh"

// Benign standalone CUDA workload with fixed-size deterministic work only.

__global__ void gather_scatter_kernel(const uint32_t* input, uint32_t* output, uint32_t* checksum, size_t n, size_t span) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint32_t x = input[tid];
    uint32_t y = input[(tid * 29u + 243u) % span];
    uint32_t z = static_cast<uint32_t>(tid) ^ 0x9e3779b9u;
    size_t idx=(x ^ z) % span; uint32_t v=input[idx]; output[(tid*13u)%n]=v; x=v^idx;
    output[tid] = x;
    atomicAdd(checksum, x ^ z ^ static_cast<uint32_t>(2802362286u));
}

int main(int argc, char** argv) {
    BenignOptions options;
    if (!parse_benign_args(argc, argv, options)) {
        print_benign_usage(argv[0]);
        return 1;
    }

    size_t n = (options.size_mb * 1024ull * 1024ull / sizeof(uint32_t));
    n = std::max<size_t>(n, static_cast<size_t>(options.cli.blocks) * static_cast<size_t>(options.cli.threads));
    size_t span = std::max<size_t>(n, 1024);

    uint32_t* d_input = nullptr;
    uint32_t* d_output = nullptr;
    uint32_t* d_checksum = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(uint32_t) * span));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(uint32_t) * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_checksum), sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_checksum, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(uint32_t) * n));

    std::vector<uint32_t> h_input(span);
    benign_fill_u32(h_input.data(), span, options.cli.seed + 2802362286u);
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), sizeof(uint32_t) * span, cudaMemcpyHostToDevice));

    auto start = std::chrono::steady_clock::now();
    auto end_time = start + std::chrono::seconds(options.cli.runtime_seconds);
    uint64_t total_launches = 0;
    uint64_t total_elements = 0;

    while (std::chrono::steady_clock::now() < end_time) {
        gather_scatter_kernel<<<options.cli.blocks, options.cli.threads>>>(d_input, d_output, d_checksum, n, span);
        CUDA_CHECK(cudaGetLastError());
        total_launches++;
        total_elements += n;
        if (options.cli.sync_every > 0 && total_launches % options.cli.sync_every == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_checksum = 0;
    CUDA_CHECK(cudaMemcpy(&h_checksum, d_checksum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    print_benign_summary("memory", "gather_scatter", options, total_launches, total_elements, h_checksum);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_checksum));
    return 0;
}
