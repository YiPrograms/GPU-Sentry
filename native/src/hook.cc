#include <cuda.h>
#include <cudaTypedefs.h>
#include <cstring>

#include "code_registry.h"
#include "debug.h"
#include "hook.h"

static CUdevice get_context_device() {
  CUdevice device;
	PFN_cuCtxGetDevice_v2000 ctxGetDevice = (PFN_cuCtxGetDevice_v2000)real_cuCtxGetDevice;
	ctxGetDevice(&device);
  return device;
}

static void get_device_pci_bus_id(char *pci_bus_id, size_t pci_bus_id_size) {
  CUdevice device = get_context_device();

	PFN_cuDeviceGetPCIBusId_v4010 deviceGetPCIBusId = (PFN_cuDeviceGetPCIBusId_v4010)real_cuDeviceGetPCIBusId;
  deviceGetPCIBusId(pci_bus_id, pci_bus_id_size, device);
}

#ifdef __cplusplus
extern "C" {
#endif

#undef cuGetProcAddress
CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) {
	PFN_cuGetProcAddress_v11030 real = (PFN_cuGetProcAddress_v11030)real_cuGetProcAddress;

	// DEBUG("cuGetProcAddress() searching for symbol %s", symbol);

  // Look for our hooked function
  void *func = get_hooked_function(symbol);

  // Not being hooked
  if (func == NULL) {
    return real(symbol, pfn, cudaVersion, flags);
  }

	DEBUG("cuGetProcAddress() hooked symbol %s", symbol);

	*pfn = func;
	return CUDA_SUCCESS;
}

#undef cuGetProcAddress_v2
CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
														 cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus) {
	PFN_cuGetProcAddress_v12000 real = (PFN_cuGetProcAddress_v12000)real_cuGetProcAddress_v2;

	// DEBUG("cuGetProcAddress_v2() searching for symbol %s", symbol);

  // Look for our hooked function
  void *func = get_hooked_function(symbol);

  // Not being hooked
  if (func == NULL) {
    return real(symbol, pfn, cudaVersion, flags, symbolStatus);
  }

	DEBUG("cuGetProcAddress_v2() hooked symbol %s", symbol);

	*pfn = func;
  if (symbolStatus != NULL)
    *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;

	return CUDA_SUCCESS;
}

#undef cuLaunchKernel
CUresult cuLaunchKernel(CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
	PFN_cuLaunchKernel_v4000 real = (PFN_cuLaunchKernel_v4000)real_cuLaunchKernel;

  char pci_bus_id[13];
  get_device_pci_bus_id(pci_bus_id, sizeof(pci_bus_id));
  send_kernel_launch(KernelLaunch{f, true, gridDimX, gridDimY, gridDimZ,
                                  blockDimX, blockDimY, blockDimZ,
                                  sharedMemBytes, hStream, pci_bus_id});

	return real(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
		sharedMemBytes, hStream, kernelParams, extra);
}

#undef cuLaunchKernelEx
CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f,
                          void **kernelParams, void **extra) {
  PFN_cuLaunchKernelEx_v11060 real =
      (PFN_cuLaunchKernelEx_v11060)real_cuLaunchKernelEx;

  char pci_bus_id[13];
  get_device_pci_bus_id(pci_bus_id, sizeof(pci_bus_id));

  if (config != NULL) {
    send_kernel_launch(KernelLaunch{f, true, config->gridDimX, config->gridDimY,
                                    config->gridDimZ, config->blockDimX,
                                    config->blockDimY, config->blockDimZ,
                                    config->sharedMemBytes, config->hStream,
                                    pci_bus_id});
  } else {
    send_kernel_launch(KernelLaunch{f, false, 0, 0, 0, 0, 0, 0, 0, NULL,
                                    pci_bus_id});
  }

  return real(config, f, kernelParams, extra);
}

#undef cuLaunchCooperativeKernel
CUresult cuLaunchCooperativeKernel(CUfunction f,
  unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
  unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
  unsigned int sharedMemBytes, CUstream hStream, void **kernelParams) {
  PFN_cuLaunchCooperativeKernel_v9000 real =
      (PFN_cuLaunchCooperativeKernel_v9000)real_cuLaunchCooperativeKernel;

  char pci_bus_id[13];
  get_device_pci_bus_id(pci_bus_id, sizeof(pci_bus_id));
  send_kernel_launch(KernelLaunch{f, true, gridDimX, gridDimY, gridDimZ,
                                  blockDimX,
                                  blockDimY, blockDimZ, sharedMemBytes,
                                  hStream, pci_bus_id});

  return real(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
              sharedMemBytes, hStream, kernelParams);
}

#undef cuModuleLoad
CUresult cuModuleLoad(CUmodule *module, const char *fname) {
  PFN_cuModuleLoad_v2000 real = (PFN_cuModuleLoad_v2000)real_cuModuleLoad;
  CUresult res = real(module, fname);

  if (res == CUDA_SUCCESS && module != NULL) {
    DEBUG("cuModuleLoad() loaded module %s, returns handle %p", fname, *module);
    load_code(fname, *module, true);
  }

  return res;
}

#undef cuModuleLoadData
CUresult cuModuleLoadData(CUmodule *module, const void *image) {
  PFN_cuModuleLoadData_v2000 real =
      (PFN_cuModuleLoadData_v2000)real_cuModuleLoadData;
  CUresult res = real(module, image);

  if (res == CUDA_SUCCESS && module != NULL) {
    DEBUG("cuModuleLoadData() loaded module from data at %p, returns handle %p",
          image, *module);
    load_code(image, *module);
  }

  return res;
}

#undef cuModuleLoadDataEx
CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                            unsigned int numOptions, CUjit_option *options,
                            void **optionValues) {
  PFN_cuModuleLoadDataEx_v2010 real =
      (PFN_cuModuleLoadDataEx_v2010)real_cuModuleLoadDataEx;
  CUresult res = real(module, image, numOptions, options, optionValues);

  if (res == CUDA_SUCCESS && module != NULL) {
    DEBUG(
        "cuModuleLoadDataEx() loaded module from data at %p, returns handle %p",
        image, *module);
    load_code(image, *module);
  }

  return res;
}

#undef cuModuleLoadFatBinary
CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
  PFN_cuModuleLoadFatBinary_v2000 real =
      (PFN_cuModuleLoadFatBinary_v2000)real_cuModuleLoadFatBinary;
  CUresult res = real(module, fatCubin);

  if (res == CUDA_SUCCESS && module != NULL) {
    DEBUG(
        "cuModuleLoadFatBinary() loaded module from fat binary at %p, returns "
        "handle %p",
        fatCubin, *module);
    load_code(fatCubin, *module);
  }

  return res;
}

#undef cuLibraryLoadData
CUresult cuLibraryLoadData(CUlibrary *library, const void *code,
                           CUjit_option *jitOptions, void **jitOptionsValues,
                           unsigned int numJitOptions,
                           CUlibraryOption *libraryOptions,
                           void **libraryOptionValues,
                           unsigned int numLibraryOptions) {
  PFN_cuLibraryLoadData_v12000 real =
      (PFN_cuLibraryLoadData_v12000)real_cuLibraryLoadData;
  CUresult res = real(library, code, jitOptions, jitOptionsValues,
                      numJitOptions, libraryOptions, libraryOptionValues,
                      numLibraryOptions);

  if (res == CUDA_SUCCESS && library != NULL) {
    DEBUG(
        "cuLibraryLoadData() loaded library from data at %p, returns handle %p",
        code, *library);
    load_code(code, *library);
  }

  return res;
}

#undef cuLibraryLoadFromFile
CUresult cuLibraryLoadFromFile(CUlibrary *library, const char *fileName,
                               CUjit_option *jitOptions,
                               void **jitOptionsValues,
                               unsigned int numJitOptions,
                               CUlibraryOption *libraryOptions,
                               void **libraryOptionValues,
                               unsigned int numLibraryOptions) {
  PFN_cuLibraryLoadFromFile_v12000 real =
      (PFN_cuLibraryLoadFromFile_v12000)real_cuLibraryLoadFromFile;
  CUresult res = real(library, fileName, jitOptions, jitOptionsValues,
                      numJitOptions, libraryOptions, libraryOptionValues,
                      numLibraryOptions);

  if (res == CUDA_SUCCESS && library != NULL) {
    DEBUG(
        "cuLibraryLoadFromFile() loaded library from file %s, returns handle %p",
        fileName, *library);
    load_code(fileName, *library, true);
  }

  return res;
}

#undef cuModuleGetFunction
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                             const char *name) {
  PFN_cuModuleGetFunction_v2000 real =
      (PFN_cuModuleGetFunction_v2000)real_cuModuleGetFunction;
  CUresult res = real(hfunc, hmod, name);

  if (res == CUDA_SUCCESS && hfunc != NULL) {
    DEBUG(
        "cuModuleGetFunction() called for module %p, name %s, returns handle %p",
        hmod, name, *hfunc);
    register_kernel(*hfunc, hmod, name);
  }

  return res;
}

#undef cuLibraryGetKernel
CUresult cuLibraryGetKernel(CUkernel *pKernel, CUlibrary library,
                            const char *name) {
  PFN_cuLibraryGetKernel_v12000 real =
      (PFN_cuLibraryGetKernel_v12000)real_cuLibraryGetKernel;
  CUresult res = real(pKernel, library, name);

  if (res == CUDA_SUCCESS && pKernel != NULL) {
    DEBUG(
        "cuLibraryGetKernel() called for library %p, name %s, returns handle %p",
        library, name, *pKernel);
    register_kernel(*pKernel, library, name);
  }

  return res;
}

#undef cuLibraryGetModule
CUresult cuLibraryGetModule(CUmodule *pMod, CUlibrary library) {
  PFN_cuLibraryGetModule_v12000 real =
      (PFN_cuLibraryGetModule_v12000)real_cuLibraryGetModule;
  CUresult res = real(pMod, library);

  if (res == CUDA_SUCCESS && pMod != NULL) {
    DEBUG("cuLibraryGetModule() called for library %p, returns module handle %p",
          library, *pMod);
    map_module_to_library(*pMod, library);
  }

  return res;
}

#ifdef __cplusplus
}
#endif

static void *get_hooked_function(const char *symbol) {
  #define HOOK_FUNCTION(fn) \
    if (strcmp(symbol, #fn) == 0) return (void *)(&fn);

  HOOK_FUNCTION(cuLaunchKernel)
  HOOK_FUNCTION(cuLaunchKernelEx)
  HOOK_FUNCTION(cuLaunchCooperativeKernel)
  HOOK_FUNCTION(cuGetProcAddress)
  HOOK_FUNCTION(cuGetProcAddress_v2)
  HOOK_FUNCTION(cuModuleLoad)
  HOOK_FUNCTION(cuModuleLoadData)
  HOOK_FUNCTION(cuModuleLoadDataEx)
  HOOK_FUNCTION(cuModuleLoadFatBinary)
  HOOK_FUNCTION(cuLibraryLoadData)
  HOOK_FUNCTION(cuLibraryLoadFromFile)
  HOOK_FUNCTION(cuModuleGetFunction)
  HOOK_FUNCTION(cuLibraryGetKernel)
  HOOK_FUNCTION(cuLibraryGetModule)

  return NULL;
}
