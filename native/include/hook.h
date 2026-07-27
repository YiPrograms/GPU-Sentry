#include <cuda.h>

// Real function pointers from loader.c
#define CUDA_SYMBOL(name) \
    extern void *real_##name __attribute__((visibility("hidden")));
#include "cuda_symbols.def"
#undef CUDA_SYMBOL

static void *get_hooked_function(const char *symbol);
