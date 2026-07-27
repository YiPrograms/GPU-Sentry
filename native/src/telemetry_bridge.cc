#include "config.h"
#include "telemetry.h"

#include <pthread.h>
#include <cstdlib>
#include <cstring>
#include <strings.h>

#include "debug.h"

namespace {

SGClientConfig telemetry_config{};
pthread_t telemetry_thread;
bool telemetry_initialized = false;
bool telemetry_started = false;

const char *server_address() {
  const char *value = getenv("GPU_SENTRY_SERVER_ADDR");
  return value != nullptr && value[0] != '\0'
             ? value
             : GPU_SENTRY_DEFAULT_SERVER_ADDR;
}

bool env_capture_disabled() {
  const char *value = getenv(GPU_SENTRY_DISABLE_ENV);
  if (value == nullptr) return false;
  return strcmp(value, "1") == 0 || strcasecmp(value, "true") == 0 ||
         strcasecmp(value, "yes") == 0 || strcasecmp(value, "on") == 0;
}

void *start_go_client(void *) {
  sg_go_start(&telemetry_config);
  DEBUG("telemetry client started, server %s", telemetry_config.server_addr);
  return nullptr;
}

}  // namespace

__attribute__((constructor))
static void telemetry_constructor(void) {
  if (env_capture_disabled()) {
    DEBUG("telemetry disabled by %s", GPU_SENTRY_DISABLE_ENV);
    return;
  }

  telemetry_config.server_addr = server_address();
  telemetry_config.hook_version = GPU_SENTRY_HOOK_VERSION;

  telemetry_initialized =
      sg_telemetry_init(GPU_SENTRY_DEFAULT_RING_CAPACITY) != 0;
  if (!telemetry_initialized) {
    DEBUG("failed to initialize telemetry ring");
    return;
  }

  int err = pthread_create(&telemetry_thread, nullptr, start_go_client, nullptr);
  if (err == 0) {
    pthread_detach(telemetry_thread);
    telemetry_started = true;
  } else {
    DEBUG("failed to start telemetry client thread: %d", err);
  }
}

__attribute__((destructor))
static void telemetry_destructor(void) {
  if (telemetry_started) {
    sg_go_stop();
  }
  if (telemetry_initialized) {
    sg_telemetry_shutdown();
  }
}
