#pragma once
#include <cstdint>

__device__ __forceinline__ uint32_t load_le32(const uint8_t* p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24); }
__device__ __forceinline__ uint64_t load_le64(const uint8_t* p) { return uint64_t(load_le32(p)) | (uint64_t(load_le32(p + 4)) << 32); }
__device__ __forceinline__ void store_le32(uint8_t* p, uint32_t v) { p[0]=uint8_t(v); p[1]=uint8_t(v>>8); p[2]=uint8_t(v>>16); p[3]=uint8_t(v>>24); }
__device__ __forceinline__ uint32_t bswap32(uint32_t x) { return __byte_perm(x, 0, 0x0123); }
