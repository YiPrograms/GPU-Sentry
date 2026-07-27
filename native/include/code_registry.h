#ifndef CUHOOK_CODE_REGISTRY_H
#define CUHOOK_CODE_REGISTRY_H

#include <cstdint>
#include <string>

struct KernelInfo {
  std::string name;
  uint32_t code_id;
  bool found;
};

struct KernelLaunch {
  void *kernel_handle;
  bool has_dimensions;
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  unsigned int sharedMemBytes;
  void *stream_handle;
  const char *device_pci_bus_id;
};

void load_code(const void *code, void *owner_handle, bool is_path = false);
void map_module_to_library(void *module_handle, void *library_handle);
void register_kernel(void *kernel_handle, void *owner_handle, const char *name);
KernelInfo get_kernel_info(void *kernel_handle);
void send_kernel_launch(const KernelLaunch &launch);

#endif  // CUHOOK_CODE_REGISTRY_H
