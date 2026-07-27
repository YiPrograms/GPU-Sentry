#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do {     cudaError_t _cuda_err = (expr);     if (_cuda_err != cudaSuccess) {         std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_cuda_err));         std::exit(2);     } } while (0)
