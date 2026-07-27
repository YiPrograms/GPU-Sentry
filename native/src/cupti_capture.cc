#include "cupti_capture.h"

#include <cuda.h>
#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "code_registry.h"
#include "debug.h"

namespace {

using CuptiResult = int;
using CuptiSubscriber = void *;
using CuptiCallback = void (*)(void *, uint32_t, uint32_t, const void *);
using CuptiSubscribe = CuptiResult (*)(CuptiSubscriber *, CuptiCallback, void *);
using CuptiEnableCallback =
    CuptiResult (*)(uint32_t, CuptiSubscriber, uint32_t, uint32_t);
using CuptiUnsubscribe = CuptiResult (*)(CuptiSubscriber);

constexpr CuptiResult CUPTI_SUCCESS = 0;
constexpr uint32_t CUPTI_CB_DOMAIN_RESOURCE = 3;
constexpr uint32_t CUPTI_CBID_RESOURCE_MODULE_LOADED = 6;

struct ResourceData {
  CUcontext context;
  union {
    CUstream stream;
  } resource_handle;
  void *resource_descriptor;
};

struct ModuleResourceData {
  uint32_t module_id;
  size_t cubin_size;
  const char *cubin;
};

void *cupti_handle = nullptr;
CuptiSubscriber subscriber = nullptr;
CuptiUnsubscribe unsubscribe_callback = nullptr;

void resource_callback(void *, uint32_t domain, uint32_t callback_id,
                       const void *callback_data) {
  if (domain != CUPTI_CB_DOMAIN_RESOURCE ||
      callback_id != CUPTI_CBID_RESOURCE_MODULE_LOADED ||
      callback_data == nullptr) {
    return;
  }

  const auto *resource = static_cast<const ResourceData *>(callback_data);
  const auto *module =
      static_cast<const ModuleResourceData *>(resource->resource_descriptor);
  if (module != nullptr) {
    capture_cubin(module->cubin, module->cubin_size);
  }
}

void *open_cupti() {
  const char *configured = std::getenv("GPU_SENTRY_CUPTI_PATH");
  if (configured != nullptr && configured[0] != '\0') {
    return dlopen(configured, RTLD_NOW | RTLD_LOCAL);
  }

  void *handle = dlopen("libcupti.so", RTLD_NOW | RTLD_LOCAL);
  if (handle != nullptr) return handle;

  return dlopen("/usr/local/cuda/extras/CUPTI/lib64/libcupti.so",
                RTLD_NOW | RTLD_LOCAL);
}

}  // namespace

void start_cupti_capture(void) {
  cupti_handle = open_cupti();
  if (cupti_handle == nullptr) {
    INFO("CUPTI unavailable; private CUDA library cubins will not be captured: %s",
         dlerror());
    return;
  }

  auto subscribe =
      reinterpret_cast<CuptiSubscribe>(dlsym(cupti_handle, "cuptiSubscribe"));
  auto enable = reinterpret_cast<CuptiEnableCallback>(
      dlsym(cupti_handle, "cuptiEnableCallback"));
  unsubscribe_callback = reinterpret_cast<CuptiUnsubscribe>(
      dlsym(cupti_handle, "cuptiUnsubscribe"));
  if (subscribe == nullptr || enable == nullptr ||
      unsubscribe_callback == nullptr) {
    INFO("CUPTI callback API unavailable; private CUDA library cubins will not be captured");
    dlclose(cupti_handle);
    cupti_handle = nullptr;
    return;
  }

  CuptiResult result = subscribe(&subscriber, resource_callback, nullptr);
  if (result == CUPTI_SUCCESS) {
    result = enable(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE,
                    CUPTI_CBID_RESOURCE_MODULE_LOADED);
  }
  if (result != CUPTI_SUCCESS) {
    INFO("CUPTI module callback setup failed (%d); private CUDA library cubins will not be captured",
         result);
    if (subscriber != nullptr) {
      unsubscribe_callback(subscriber);
      subscriber = nullptr;
    }
    dlclose(cupti_handle);
    cupti_handle = nullptr;
    return;
  }

  DEBUG("CUPTI module capture enabled");
}

void stop_cupti_capture(void) {
  if (subscriber != nullptr && unsubscribe_callback != nullptr) {
    unsubscribe_callback(subscriber);
    subscriber = nullptr;
  }
  if (cupti_handle != nullptr) {
    dlclose(cupti_handle);
    cupti_handle = nullptr;
  }
}
