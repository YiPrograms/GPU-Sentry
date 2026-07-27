#include "code_registry.h"

#include <elf.h>
#include <sys/mman.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "debug.h"
#include "telemetry.h"

namespace {

constexpr uint32_t CUDA_FATBIN_WRAPPER_MAGIC = 0x466243b1;
constexpr uint32_t CUDA_FATBIN_HEADER_MAGIC = 0xba55ed50;
constexpr uint32_t CODE_TYPE_FATBIN = 0x1;
constexpr uint32_t CODE_TYPE_CUBIN = 0x2;
constexpr uint32_t CODE_TYPE_PTX = 0x3;

struct FatbinWrapper {
  uint32_t magic;
  uint32_t version;
  const void *data;
  void *filename_or_fatbins;
};

struct FatbinHeader {
  uint32_t magic;
  uint16_t version;
  uint16_t header_size;
  uint64_t size;
};

struct KernelRecord {
  std::string name;
  uint32_t code_id;
};

struct CodeImage {
  const void *code;
  size_t size;
  uint32_t type;
};

std::mutex registry_mutex;
std::vector<void *> code_handles;
std::unordered_map<void *, uint32_t> code_id_by_handle;
std::unordered_map<void *, void *> module_to_library_map;
std::unordered_map<void *, KernelRecord> kernel_map;

void send_code(const void *code, size_t size, uint32_t code_type,
               uint32_t code_id) {
  if (code == nullptr || size == 0) return;

  DEBUG("Captured code (type %u, id %u, size %zu bytes)",
        code_type, code_id, size);
  sg_enqueue_code(code_id, code_type, code, size);
}

CodeImage load_fatbin_header(const FatbinHeader *fatbin) {
  if (fatbin == nullptr || fatbin->magic != CUDA_FATBIN_HEADER_MAGIC) {
    DEBUG("Invalid fatbin header");
    return CodeImage{nullptr, 0, CODE_TYPE_FATBIN};
  }

  const size_t size = fatbin->header_size + fatbin->size;
  return CodeImage{fatbin, size, CODE_TYPE_FATBIN};
}

CodeImage load_fatbin_wrapper(const void *code) {
  if (code == nullptr) return CodeImage{nullptr, 0, CODE_TYPE_FATBIN};

  const FatbinWrapper *wrapper = static_cast<const FatbinWrapper *>(code);
  if (wrapper->magic != CUDA_FATBIN_WRAPPER_MAGIC) {
    DEBUG("Invalid fatbin wrapper magic number: 0x%x", wrapper->magic);
    return CodeImage{nullptr, 0, CODE_TYPE_FATBIN};
  }

  return load_fatbin_header(static_cast<const FatbinHeader *>(wrapper->data));
}

CodeImage load_cubin(const void *code) {
  if (code == nullptr) return CodeImage{nullptr, 0, CODE_TYPE_CUBIN};

  const Elf64_Ehdr *ehdr = static_cast<const Elf64_Ehdr *>(code);
  if (std::memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0) {
    DEBUG("Invalid ELF magic number");
    return CodeImage{nullptr, 0, CODE_TYPE_CUBIN};
  }

  const size_t section_end =
      ehdr->e_shoff + (ehdr->e_shentsize * ehdr->e_shnum);
  const size_t program_end =
      ehdr->e_phoff + (ehdr->e_phentsize * ehdr->e_phnum);
  const size_t size = std::max(section_end, program_end);

  return CodeImage{code, size, CODE_TYPE_CUBIN};
}

CodeImage load_ptx(const void *code) {
  if (code == nullptr) return CodeImage{nullptr, 0, CODE_TYPE_PTX};

  const char *ptx_code = static_cast<const char *>(code);
  const size_t size = std::strlen(ptx_code) + 1;

  return CodeImage{ptx_code, size, CODE_TYPE_PTX};
}

uint32_t register_code_handle(void *owner_handle) {
  std::lock_guard<std::mutex> lock(registry_mutex);

  auto it = code_id_by_handle.find(owner_handle);
  if (it != code_id_by_handle.end()) {
    DEBUG("code owner %p -> code ID %u already registered",
          owner_handle, it->second);
    return it->second;
  }

  code_handles.push_back(owner_handle);
  const uint32_t code_id = static_cast<uint32_t>(code_handles.size() - 1);
  code_id_by_handle[owner_handle] = code_id;

  DEBUG("code owner %p -> code ID %u", owner_handle, code_id);

  return code_id;
}

uint32_t get_code_id_locked(void *owner_handle) {
  if (owner_handle == nullptr) return UINT32_MAX;

  auto lib_it = module_to_library_map.find(owner_handle);
  if (lib_it != module_to_library_map.end()) {
    owner_handle = lib_it->second;
  }

  auto id_it = code_id_by_handle.find(owner_handle);
  if (id_it == code_id_by_handle.end()) return UINT32_MAX;

  return id_it->second;
}

}  // namespace

void load_code(const void *code, void *owner_handle, bool is_path) {
  if (code == nullptr || owner_handle == nullptr) return;

  const uint32_t code_id = register_code_handle(owner_handle);
  const void *mapped_code = code;
  size_t file_size = 0;

  if (is_path) {
    FILE *file = std::fopen(static_cast<const char *>(code), "rb");
    if (file == nullptr) {
      DEBUG("Failed to open code file %s: %s", static_cast<const char *>(code),
            std::strerror(errno));
      return;
    }

    if (std::fseek(file, 0, SEEK_END) != 0) {
      DEBUG("Failed to seek code file %s", static_cast<const char *>(code));
      std::fclose(file);
      return;
    }

    const long size = std::ftell(file);
    if (size <= 0) {
      DEBUG("Invalid code file size for %s", static_cast<const char *>(code));
      std::fclose(file);
      return;
    }

    file_size = static_cast<size_t>(size);
    std::rewind(file);

    mapped_code = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE,
                       fileno(file), 0);
    std::fclose(file);

    if (mapped_code == MAP_FAILED) {
      DEBUG("Failed to mmap code file %s: %s", static_cast<const char *>(code),
            std::strerror(errno));
      return;
    }
  }

  CodeImage image = {nullptr, 0, 0};
  const uint32_t magic = *static_cast<const uint32_t *>(mapped_code);
  if (magic == CUDA_FATBIN_WRAPPER_MAGIC) {
    image = load_fatbin_wrapper(mapped_code);
  } else if (magic == CUDA_FATBIN_HEADER_MAGIC) {
    image = load_fatbin_header(static_cast<const FatbinHeader *>(mapped_code));
  } else if (std::memcmp(mapped_code, ELFMAG, SELFMAG) == 0) {
    image = load_cubin(mapped_code);
  } else {
    image = load_ptx(mapped_code);
  }

  DEBUG("Loaded code image of type %u, size %zu bytes", image.type, image.size);

  send_code(image.code, image.size, image.type, code_id);

  if (is_path) {
    munmap(const_cast<void *>(mapped_code), file_size);
  }
}

void map_module_to_library(void *module_handle, void *library_handle) {
  if (module_handle == nullptr || library_handle == nullptr) return;

  std::lock_guard<std::mutex> lock(registry_mutex);
  module_to_library_map[module_handle] = library_handle;

  DEBUG("module %p -> library %p", module_handle, library_handle);
}

void register_kernel(void *kernel_handle, void *owner_handle,
                     const char *name) {
  if (kernel_handle == nullptr || name == nullptr) return;

  std::lock_guard<std::mutex> lock(registry_mutex);
  const uint32_t code_id = get_code_id_locked(owner_handle);

  if (code_id == UINT32_MAX) {
    DEBUG("owner %p -> code ID <not found> for kernel %s", owner_handle, name);
  }

  kernel_map[kernel_handle] = KernelRecord{std::string(name), code_id};

  DEBUG("kernel %p -> name %s, code ID %u", kernel_handle, name, code_id);
}

KernelInfo get_kernel_info(void *kernel_handle) {
  if (kernel_handle == nullptr) return KernelInfo{"", 0, false};

  std::lock_guard<std::mutex> lock(registry_mutex);
  auto it = kernel_map.find(kernel_handle);
  if (it == kernel_map.end()) return KernelInfo{"", 0, false};

  return KernelInfo{it->second.name, it->second.code_id, true};
}

void send_kernel_launch(const KernelLaunch &launch) {
  KernelInfo kernel = get_kernel_info(launch.kernel_handle);
  SGKernelLaunchEvent event{};

  snprintf(event.kernel_name, sizeof(event.kernel_name), "%s",
           kernel.found ? kernel.name.c_str() : "");
  event.kernel_name_found = kernel.found ? 1 : 0;
  event.code_id = kernel.found ? kernel.code_id : SG_CODE_ID_UNKNOWN;
  event.code_id_found =
      kernel.found && kernel.code_id != SG_CODE_ID_UNKNOWN ? 1 : 0;
  event.kernel_handle =
      reinterpret_cast<uint64_t>(launch.kernel_handle);
  event.grid_dim_x = launch.gridDimX;
  event.grid_dim_y = launch.gridDimY;
  event.grid_dim_z = launch.gridDimZ;
  event.block_dim_x = launch.blockDimX;
  event.block_dim_y = launch.blockDimY;
  event.block_dim_z = launch.blockDimZ;
  event.shared_mem_bytes = launch.sharedMemBytes;
  event.stream_handle = reinterpret_cast<uint64_t>(launch.stream_handle);
  snprintf(event.device_pci_bus_id, sizeof(event.device_pci_bus_id), "%s",
           launch.device_pci_bus_id != nullptr ? launch.device_pci_bus_id : "");

  if (launch.has_dimensions) {
    DEBUG("Captured kernel launch: kernel %p -> name %s, code ID %u, gridDim (%u, %u, %u), blockDim (%u, %u, %u), sharedMemBytes %u, stream %p, device %s",
          launch.kernel_handle,
          kernel.found ? kernel.name.c_str() : "<unknown>",
          kernel.found ? kernel.code_id : UINT32_MAX, launch.gridDimX,
          launch.gridDimY, launch.gridDimZ, launch.blockDimX,
          launch.blockDimY, launch.blockDimZ, launch.sharedMemBytes,
          launch.stream_handle, launch.device_pci_bus_id);
  } else {
    DEBUG("Captured kernel launch: kernel %p -> name %s, code ID %u, config <null>, device %s",
          launch.kernel_handle,
          kernel.found ? kernel.name.c_str() : "<unknown>",
          kernel.found ? kernel.code_id : UINT32_MAX,
          launch.device_pci_bus_id);
  }

  sg_enqueue_kernel_launch(&event);
}
