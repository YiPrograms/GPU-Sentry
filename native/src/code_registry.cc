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
#include <unordered_set>

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
  uint64_t code_id;
  bool code_id_found;
};

struct CodeImage {
  const void *code;
  size_t size;
  uint32_t type;
};

std::mutex registry_mutex;
std::unordered_map<void *, uint64_t> code_id_by_handle;
std::unordered_map<void *, void *> module_to_library_map;
std::unordered_map<void *, KernelRecord> kernel_map;
std::unordered_map<std::string, uint64_t> code_id_by_kernel_name;
std::unordered_set<uint64_t> captured_code_ids;

uint64_t capture_code(const void *code, size_t size, uint32_t code_type) {
  if (code == nullptr || size == 0) return 0;

  const uint64_t code_id = sg_code_id(const_cast<void *>(code), size);
  bool first_capture;
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    first_capture = captured_code_ids.insert(code_id).second;
  }

  if (first_capture) {
    DEBUG("Captured code (type %u, id %016llx, size %zu bytes)", code_type,
          static_cast<unsigned long long>(code_id), size);
    sg_enqueue_code(code_id, code_type, code, size);
  }
  return code_id;
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

void register_code_handle(void *owner_handle, uint64_t code_id) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  code_id_by_handle[owner_handle] = code_id;
  DEBUG("code owner %p -> code ID %016llx", owner_handle,
        static_cast<unsigned long long>(code_id));
}

bool range_fits(size_t offset, size_t count, size_t item_size, size_t size) {
  return offset <= size && item_size != 0 && count <= (size - offset) / item_size;
}

void index_cubin_symbols(const void *code, size_t size, uint64_t code_id) {
  if (size < sizeof(Elf64_Ehdr)) return;

  const auto *bytes = static_cast<const unsigned char *>(code);
  const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(bytes);
  if (std::memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0 ||
      ehdr->e_ident[EI_CLASS] != ELFCLASS64 ||
      ehdr->e_shentsize != sizeof(Elf64_Shdr) ||
      !range_fits(ehdr->e_shoff, ehdr->e_shnum, sizeof(Elf64_Shdr), size)) {
    return;
  }

  const auto *sections =
      reinterpret_cast<const Elf64_Shdr *>(bytes + ehdr->e_shoff);
  for (uint16_t i = 0; i < ehdr->e_shnum; ++i) {
    const Elf64_Shdr &symbols = sections[i];
    if ((symbols.sh_type != SHT_SYMTAB && symbols.sh_type != SHT_DYNSYM) ||
        symbols.sh_entsize != sizeof(Elf64_Sym) ||
        symbols.sh_link >= ehdr->e_shnum ||
        !range_fits(symbols.sh_offset, symbols.sh_size / symbols.sh_entsize,
                    symbols.sh_entsize, size)) {
      continue;
    }

    const Elf64_Shdr &strings = sections[symbols.sh_link];
    if (strings.sh_offset > size || strings.sh_size > size - strings.sh_offset) {
      continue;
    }

    const auto *entries =
        reinterpret_cast<const Elf64_Sym *>(bytes + symbols.sh_offset);
    const char *names = reinterpret_cast<const char *>(bytes + strings.sh_offset);
    const size_t count = symbols.sh_size / symbols.sh_entsize;
    for (size_t j = 0; j < count; ++j) {
      const Elf64_Sym &symbol = entries[j];
      if (ELF64_ST_TYPE(symbol.st_info) != STT_FUNC ||
          symbol.st_shndx == SHN_UNDEF || symbol.st_name >= strings.sh_size) {
        continue;
      }
      const char *name = names + symbol.st_name;
      const size_t remaining = strings.sh_size - symbol.st_name;
      if (name[0] != '\0' && std::memchr(name, '\0', remaining) != nullptr) {
        code_id_by_kernel_name[name] = code_id;
      }
    }
  }
}

bool get_code_id_locked(void *owner_handle, uint64_t *code_id) {
  if (owner_handle == nullptr || code_id == nullptr) return false;

  auto lib_it = module_to_library_map.find(owner_handle);
  if (lib_it != module_to_library_map.end()) {
    owner_handle = lib_it->second;
  }

  auto id_it = code_id_by_handle.find(owner_handle);
  if (id_it == code_id_by_handle.end()) return false;

  *code_id = id_it->second;
  return true;
}

}  // namespace

void load_code(const void *code, void *owner_handle, bool is_path) {
  if (code == nullptr || owner_handle == nullptr) return;

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

  if (image.code != nullptr && image.size != 0) {
    const uint64_t code_id = capture_code(image.code, image.size, image.type);
    register_code_handle(owner_handle, code_id);
  }

  if (is_path) {
    munmap(const_cast<void *>(mapped_code), file_size);
  }
}

void capture_cubin(const void *code, size_t size) {
  if (code == nullptr || size == 0) return;

  const uint64_t code_id = capture_code(code, size, CODE_TYPE_CUBIN);
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    index_cubin_symbols(code, size, code_id);
  }

  DEBUG("CUPTI captured cubin (id %016llx, size %zu bytes)",
        static_cast<unsigned long long>(code_id), size);
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
  uint64_t code_id = 0;
  bool code_id_found = get_code_id_locked(owner_handle, &code_id);

  if (!code_id_found) {
    auto it = code_id_by_kernel_name.find(name);
    if (it != code_id_by_kernel_name.end()) {
      code_id = it->second;
      code_id_found = true;
    }
  }

  if (!code_id_found) {
    DEBUG("owner %p -> code ID <not found> for kernel %s", owner_handle, name);
  }

  kernel_map[kernel_handle] =
      KernelRecord{std::string(name), code_id, code_id_found};

  DEBUG("kernel %p -> name %s, code ID %016llx", kernel_handle, name,
        static_cast<unsigned long long>(code_id));
}

void copy_kernel_info(void *kernel_handle, void *source_handle) {
  if (kernel_handle == nullptr || source_handle == nullptr) return;

  std::lock_guard<std::mutex> lock(registry_mutex);
  auto it = kernel_map.find(source_handle);
  if (it == kernel_map.end()) {
    DEBUG("kernel %p has no registered metadata", source_handle);
    return;
  }

  const KernelRecord record = it->second;
  kernel_map[kernel_handle] = record;
  DEBUG("kernel %p -> kernel %p, name %s, code ID %016llx", kernel_handle,
        source_handle, record.name.c_str(),
        static_cast<unsigned long long>(record.code_id));
}

KernelInfo get_kernel_info(void *kernel_handle) {
  if (kernel_handle == nullptr) return KernelInfo{"", 0, false, false};

  std::lock_guard<std::mutex> lock(registry_mutex);
  auto it = kernel_map.find(kernel_handle);
  if (it == kernel_map.end()) return KernelInfo{"", 0, false, false};

  return KernelInfo{it->second.name, it->second.code_id, true,
                    it->second.code_id_found};
}

void send_kernel_launch(const KernelLaunch &launch) {
  KernelInfo kernel = get_kernel_info(launch.kernel_handle);
  SGKernelLaunchEvent event{};

  snprintf(event.kernel_name, sizeof(event.kernel_name), "%s",
           kernel.found ? kernel.name.c_str() : "");
  event.kernel_name_found = kernel.found ? 1 : 0;
  event.code_id = kernel.code_id;
  event.code_id_found = kernel.code_id_found ? 1 : 0;
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
    DEBUG("Captured kernel launch: kernel %p -> name %s, code ID %016llx, gridDim (%u, %u, %u), blockDim (%u, %u, %u), sharedMemBytes %u, stream %p, device %s",
          launch.kernel_handle,
          kernel.found ? kernel.name.c_str() : "<unknown>",
          static_cast<unsigned long long>(kernel.code_id), launch.gridDimX,
          launch.gridDimY, launch.gridDimZ, launch.blockDimX, launch.blockDimY,
          launch.blockDimZ, launch.sharedMemBytes, launch.stream_handle,
          launch.device_pci_bus_id);
  } else {
    DEBUG("Captured kernel launch: kernel %p -> name %s, code ID %016llx, config <null>, device %s",
          launch.kernel_handle,
          kernel.found ? kernel.name.c_str() : "<unknown>",
          static_cast<unsigned long long>(kernel.code_id),
          launch.device_pci_bus_id);
  }

  sg_enqueue_kernel_launch(&event);
}
