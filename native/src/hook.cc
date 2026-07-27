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

static void record_kernel_launch(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream) {
  char pci_bus_id[13];
  get_device_pci_bus_id(pci_bus_id, sizeof(pci_bus_id));
  send_kernel_launch(KernelLaunch{f, true, gridDimX, gridDimY, gridDimZ,
                                  blockDimX, blockDimY, blockDimZ,
                                  sharedMemBytes, hStream, pci_bus_id});
}

static void record_kernel_launch_ex(const CUlaunchConfig *config,
                                    CUfunction f) {
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
}

#ifdef __cplusplus
extern "C" {
#endif

#undef cuGetProcAddress
CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) {
	PFN_cuGetProcAddress_v11030 real = (PFN_cuGetProcAddress_v11030)real_cuGetProcAddress;
  CUresult res = real(symbol, pfn, cudaVersion, flags);
  if (res != CUDA_SUCCESS) return res;

  void *func = get_hooked_function(symbol, flags);
  if (func == NULL) return res;

	DEBUG("cuGetProcAddress() hooked symbol %s", symbol);

	*pfn = func;
	return res;
}

#undef cuGetProcAddress_v2
CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
														 cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus) {
	PFN_cuGetProcAddress_v12000 real = (PFN_cuGetProcAddress_v12000)real_cuGetProcAddress_v2;
  CUresult res = real(symbol, pfn, cudaVersion, flags, symbolStatus);
  if (res != CUDA_SUCCESS) return res;

  void *func = get_hooked_function(symbol, flags);
  if (func == NULL) return res;

	DEBUG("cuGetProcAddress_v2() hooked symbol %s", symbol);

	*pfn = func;
	return res;
}

#undef cuLaunchKernel
CUresult cuLaunchKernel(CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
	PFN_cuLaunchKernel_v4000 real = (PFN_cuLaunchKernel_v4000)real_cuLaunchKernel;

  record_kernel_launch(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                       blockDimZ, sharedMemBytes, hStream);

	return real(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
		sharedMemBytes, hStream, kernelParams, extra);
}

#undef cuLaunchKernel_ptsz
CUresult cuLaunchKernel_ptsz(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra) {
  PFN_cuLaunchKernel_v7000_ptsz real =
      (PFN_cuLaunchKernel_v7000_ptsz)real_cuLaunchKernel_ptsz;

  record_kernel_launch(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                       blockDimZ, sharedMemBytes, hStream);

  return real(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
              sharedMemBytes, hStream, kernelParams, extra);
}

#undef cuLaunchKernelEx
CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f,
                          void **kernelParams, void **extra) {
  PFN_cuLaunchKernelEx_v11060 real =
      (PFN_cuLaunchKernelEx_v11060)real_cuLaunchKernelEx;

  record_kernel_launch_ex(config, f);

  return real(config, f, kernelParams, extra);
}

#undef cuLaunchKernelEx_ptsz
CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig *config, CUfunction f,
                               void **kernelParams, void **extra) {
  PFN_cuLaunchKernelEx_v11060_ptsz real =
      (PFN_cuLaunchKernelEx_v11060_ptsz)real_cuLaunchKernelEx_ptsz;

  record_kernel_launch_ex(config, f);

  return real(config, f, kernelParams, extra);
}

#undef cuLaunchCooperativeKernel
CUresult cuLaunchCooperativeKernel(CUfunction f,
  unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
  unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
  unsigned int sharedMemBytes, CUstream hStream, void **kernelParams) {
  PFN_cuLaunchCooperativeKernel_v9000 real =
      (PFN_cuLaunchCooperativeKernel_v9000)real_cuLaunchCooperativeKernel;

  record_kernel_launch(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                       blockDimZ, sharedMemBytes, hStream);

  return real(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
              sharedMemBytes, hStream, kernelParams);
}

#undef cuLaunchCooperativeKernel_ptsz
CUresult cuLaunchCooperativeKernel_ptsz(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams) {
  PFN_cuLaunchCooperativeKernel_v9000_ptsz real =
      (PFN_cuLaunchCooperativeKernel_v9000_ptsz)
          real_cuLaunchCooperativeKernel_ptsz;

  record_kernel_launch(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                       blockDimZ, sharedMemBytes, hStream);

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

#undef cuLibraryEnumerateKernels
CUresult cuLibraryEnumerateKernels(CUkernel *kernels, unsigned int numKernels,
                                   CUlibrary library) {
  PFN_cuLibraryEnumerateKernels_v12040 real =
      (PFN_cuLibraryEnumerateKernels_v12040)real_cuLibraryEnumerateKernels;
  CUresult res = real(kernels, numKernels, library);
  if (res != CUDA_SUCCESS || kernels == NULL) return res;

  unsigned int count = 0;
  PFN_cuLibraryGetKernelCount_v12040 get_count =
      (PFN_cuLibraryGetKernelCount_v12040)real_cuLibraryGetKernelCount;
  if (get_count(&count, library) != CUDA_SUCCESS) return res;
  if (count > numKernels) count = numKernels;

  PFN_cuKernelGetName_v12030 get_name =
      (PFN_cuKernelGetName_v12030)real_cuKernelGetName;
  for (unsigned int i = 0; i < count; ++i) {
    const char *name = NULL;
    if (get_name(&name, kernels[i]) == CUDA_SUCCESS && name != NULL) {
      register_kernel(kernels[i], library, name);
    }
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

#undef cuKernelGetFunction
CUresult cuKernelGetFunction(CUfunction *pFunc, CUkernel kernel) {
  PFN_cuKernelGetFunction_v12000 real =
      (PFN_cuKernelGetFunction_v12000)real_cuKernelGetFunction;
  CUresult res = real(pFunc, kernel);

  if (res == CUDA_SUCCESS && pFunc != NULL) {
    DEBUG("cuKernelGetFunction() called for kernel %p, returns function %p",
          kernel, *pFunc);
    copy_kernel_info(*pFunc, kernel);
  }

  return res;
}

#ifdef __cplusplus
}
#endif

static void *get_hooked_function(const char *symbol, cuuint64_t flags) {
  #define HOOK_FUNCTION(fn) \
    if (strcmp(symbol, #fn) == 0) return (void *)(&fn);

  const bool per_thread =
      (flags & CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) != 0;
  if (strcmp(symbol, "cuLaunchKernel") == 0) {
    return per_thread ? (void *)(&cuLaunchKernel_ptsz)
                      : (void *)(&cuLaunchKernel);
  }
  if (strcmp(symbol, "cuLaunchKernelEx") == 0) {
    return per_thread ? (void *)(&cuLaunchKernelEx_ptsz)
                      : (void *)(&cuLaunchKernelEx);
  }
  if (strcmp(symbol, "cuLaunchCooperativeKernel") == 0) {
    return per_thread ? (void *)(&cuLaunchCooperativeKernel_ptsz)
                      : (void *)(&cuLaunchCooperativeKernel);
  }
  HOOK_FUNCTION(cuLaunchKernel_ptsz)
  HOOK_FUNCTION(cuLaunchKernelEx_ptsz)
  HOOK_FUNCTION(cuLaunchCooperativeKernel_ptsz)
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
  HOOK_FUNCTION(cuLibraryEnumerateKernels)
  HOOK_FUNCTION(cuLibraryGetModule)
  HOOK_FUNCTION(cuKernelGetFunction)

  return NULL;
}
