
#ifndef CUHOOK_DEBUG_H
#define CUHOOK_DEBUG_H

#ifdef __cplusplus
#include <cstdio>
#else
#include <stdio.h>
#endif

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)

#ifndef CUHOOK_NO_INFO
#define INFO(fmt, ...) fprintf(stderr, "[CUHook] " fmt "\n", ##__VA_ARGS__)
#else
#define INFO(fmt, ...)
#endif

#ifdef CUHOOK_DEBUG
#define DEBUG(fmt, ...) fprintf(stderr, "[CUHook DBG] " fmt "\n", ##__VA_ARGS__)
#else
#define DEBUG(fmt, ...)
#endif

#endif  // CUHOOK_DEBUG_H