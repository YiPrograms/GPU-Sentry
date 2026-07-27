#ifndef GPU_SENTRY_TELEMETRY_H
#define GPU_SENTRY_TELEMETRY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SG_KERNEL_NAME_MAX 256
#define SG_DEVICE_ID_MAX 32
#define SG_CODE_ID_UNKNOWN UINT32_MAX

enum SGEventKind {
  SG_EVENT_CODE = 1,
  SG_EVENT_KERNEL_LAUNCH = 2,
};

typedef struct SGClientConfig {
  const char *server_addr;
  const char *hook_version;
} SGClientConfig;

typedef struct SGCodeEvent {
  uint32_t code_id;
  uint32_t code_type;
  const void *data;
  uint64_t data_size;
} SGCodeEvent;

typedef struct SGKernelLaunchEvent {
  char kernel_name[SG_KERNEL_NAME_MAX];
  uint8_t kernel_name_found;
  uint32_t code_id;
  uint8_t code_id_found;
  uint64_t kernel_handle;
  uint32_t grid_dim_x;
  uint32_t grid_dim_y;
  uint32_t grid_dim_z;
  uint32_t block_dim_x;
  uint32_t block_dim_y;
  uint32_t block_dim_z;
  uint32_t shared_mem_bytes;
  uint64_t stream_handle;
  char device_pci_bus_id[SG_DEVICE_ID_MAX];
} SGKernelLaunchEvent;

typedef struct SGEvent {
  uint32_t kind;
  uint64_t sequence;
  uint64_t timestamp_ns;
  uint32_t pid;
  uint32_t tid;
  SGCodeEvent code;
  SGKernelLaunchEvent launch;
} SGEvent;

typedef struct SGTelemetryStats {
  uint64_t dropped_events;
  uint64_t dropped_bytes;
  uint64_t queued_events;
} SGTelemetryStats;

int sg_telemetry_init(uint32_t capacity);
void sg_telemetry_shutdown(void);
int sg_enqueue_code(uint32_t code_id, uint32_t code_type, const void *data,
                    uint64_t data_size);
int sg_enqueue_kernel_launch(const SGKernelLaunchEvent *launch);
size_t sg_ring_pop_batch(SGEvent *out, size_t max_events);
void sg_ring_release_batch(SGEvent *events, size_t count);
void sg_ring_stats(SGTelemetryStats *stats);

void sg_go_start(SGClientConfig *config);
void sg_go_stop(void);

#ifdef __cplusplus
}
#endif

#endif  // GPU_SENTRY_TELEMETRY_H
