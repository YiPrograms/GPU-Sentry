#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "debug.h"

extern char _binary_libcuda_so_1_real_start[];
extern char _binary_libcuda_so_1_real_end[];

static void *dl_handle = NULL;

// Declare a global pointer for each CUDA symbol we want to resolve.
#define CUDA_SYMBOL(name) \
    void *real_##name __attribute__((visibility("hidden"))) = NULL;
#include "cuda_symbols.def"
#undef CUDA_SYMBOL

static void resolve_symbol(const char *symbol, void **slot) {
	dlerror();

	void *p = dlsym(dl_handle, symbol);
	const char *err = dlerror();

	if (err != NULL || p == NULL) {
		INFO("Failed to resolve CUDA symbol %s: %s",
			symbol,
			err ? err : "NULL");
		exit(EXIT_FAILURE);
	}

	*slot = p;
}

static void load_libcuda(void) {
	DEBUG("Loading embedded CUDA library...");

	char libcuda_path[] = "/tmp/cuhook-libcuda-XXXXXX";

	int fd = mkstemp(libcuda_path);
	if (fd < 0) {
		INFO("Failed to create temporary file for CUDA library: %s",
			strerror(errno));
		exit(EXIT_FAILURE);
	}

	FILE *fp = fdopen(fd, "wb");
	if (!fp) {
		INFO("Failed to open temporary file for writing: %s", strerror(errno));
		close(fd);
		unlink(libcuda_path);
		exit(EXIT_FAILURE);
	}

	size_t size =
		(size_t)(_binary_libcuda_so_1_real_end -
			_binary_libcuda_so_1_real_start);

	if (fwrite(_binary_libcuda_so_1_real_start,
		1,
		size,
		fp) != size) {
		INFO("Failed to write embedded CUDA library: %s", strerror(errno));
		fclose(fp);
		unlink(libcuda_path);
		exit(EXIT_FAILURE);
	}

	if (fflush(fp) != 0) {
		INFO("Failed to flush embedded CUDA library: %s", strerror(errno));
		fclose(fp);
		unlink(libcuda_path);
		exit(EXIT_FAILURE);
	}

	fclose(fp);

	DEBUG("Temporary CUDA library created at %s", libcuda_path);

	dl_handle = dlopen(libcuda_path, RTLD_NOW | RTLD_LOCAL);
	if (!dl_handle) {
		INFO("Failed to open embedded CUDA library: %s", dlerror());
		unlink(libcuda_path);
		exit(EXIT_FAILURE);
	}

	DEBUG("Opened embedded CUDA library successfully.");

	// Resolve each CUDA symbol and store the pointer in the corresponding global variable.
#define CUDA_SYMBOL(name) \
		resolve_symbol(#name, &real_##name);
#include "cuda_symbols.def"
#undef CUDA_SYMBOL

	DEBUG("Resolved all CUDA symbols successfully.");

	unlink(libcuda_path);
	DEBUG("Temporary CUDA library file removed.");
}


__attribute__((constructor))
static void constructor(void) {
	load_libcuda();
}
