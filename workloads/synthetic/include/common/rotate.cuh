#pragma once
#include <cstdint>

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int r) { r &= 31; return r ? ((x << r) | (x >> (32 - r))) : x; }
__device__ __forceinline__ uint32_t rotr32(uint32_t x, int r) { r &= 31; return r ? ((x >> r) | (x << (32 - r))) : x; }
__device__ __forceinline__ uint64_t rotl64(uint64_t x, int r) { r &= 63; return r ? ((x << r) | (x >> (64 - r))) : x; }
__device__ __forceinline__ uint64_t rotr64(uint64_t x, int r) { r &= 63; return r ? ((x >> r) | (x << (64 - r))) : x; }
