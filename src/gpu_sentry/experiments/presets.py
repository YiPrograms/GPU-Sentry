WINDOW_BUDGETS = (512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8190)

GPU_SENTRY_MINERS = (
    "ethash_split",
    "kawpow_split",
    "randomx_gpu_lite_mono",
    "sha256d_mono",
)
GPU_SENTRY_RATES = (5, 10, 50, 75, 100)

BASELINE_MINERS = GPU_SENTRY_MINERS + (
    "autolykos2_split",
    "cryptonight_gpu_split",
    "cuckoo_cycle_split",
    "equihash144_5_split",
)
BASELINE_RATES = (5, 10, 25, 50, 75, 100)

MIXED_RUNS = (
    ("cublas_gemm", "randomx_gpu_lite_mono", 10),
    ("cublas_gemm", "randomx_gpu_lite_mono", 10),
    ("pytorch_training", "ethash_split", 10),
    ("cudnn_convolution", "kawpow_split", 10),
    ("hpl", "sha256d_mono", 10),
    ("vllm_inference_fallback", "randomx_gpu_lite_mono", 50),
    ("aes", "ethash_split", 50),
)

FAMILIES = {
    "autolykos2_split": "memory_hard_table_hash",
    "cryptonight_gpu_split": "cryptonight_randomx_scratchpad",
    "cuckoo_cycle_split": "cuckoo_graph_cycle",
    "equihash144_5_split": "equihash_solver",
    "ethash_split": "ethash_dag_keccak",
    "kawpow_split": "progpow_kawpow_random_math",
    "randomx_gpu_lite_mono": "cryptonight_randomx_scratchpad",
    "sha256d_mono": "pure_hash_nonce_search",
}
