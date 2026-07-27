#include "telemetry.h"

#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <pthread.h>

#include "debug.h"

namespace {

pthread_mutex_t ring_mutex = PTHREAD_MUTEX_INITIALIZER;
SGEvent *ring = nullptr;
size_t ring_capacity = 0;
size_t ring_head = 0;
size_t ring_tail = 0;
size_t ring_count = 0;
std::atomic<uint64_t> next_sequence{1};
std::atomic<uint64_t> dropped_events{0};
std::atomic<uint64_t> dropped_bytes{0};
bool ring_initialized = false;

uint64_t now_ns() {
  timespec ts;
  if (clock_gettime(CLOCK_REALTIME, &ts) != 0) return 0;
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
         static_cast<uint64_t>(ts.tv_nsec);
}

uint32_t current_tid() {
  return static_cast<uint32_t>(syscall(SYS_gettid));
}

void fill_common(SGEvent *event, uint32_t kind) {
  event->kind = kind;
  event->sequence = next_sequence.fetch_add(1, std::memory_order_relaxed);
  event->timestamp_ns = now_ns();
  event->pid = static_cast<uint32_t>(getpid());
  event->tid = current_tid();
}

void count_drop(uint64_t bytes) {
  dropped_events.fetch_add(1, std::memory_order_relaxed);
  dropped_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

int push_event(const SGEvent *event, uint64_t bytes_if_dropped) {
  if (pthread_mutex_trylock(&ring_mutex) != 0) {
    count_drop(bytes_if_dropped);
    return 0;
  }

  if (!ring_initialized || ring == nullptr || ring_capacity == 0) {
    pthread_mutex_unlock(&ring_mutex);
    count_drop(bytes_if_dropped);
    return 0;
  }

  if (ring_count == ring_capacity) {
    pthread_mutex_unlock(&ring_mutex);
    count_drop(bytes_if_dropped);
    return 0;
  }

  ring[ring_tail] = *event;
  ring_tail = (ring_tail + 1) % ring_capacity;
  ring_count++;
  pthread_mutex_unlock(&ring_mutex);
  return 1;
}

void release_event(SGEvent *event) {
  if (event->kind == SG_EVENT_CODE && event->code.data != nullptr) {
    free(const_cast<void *>(event->code.data));
    event->code.data = nullptr;
    event->code.data_size = 0;
  }
}

}  // namespace

extern "C" {

int sg_telemetry_init(uint32_t capacity) {
  pthread_mutex_lock(&ring_mutex);
  if (ring_initialized) {
    pthread_mutex_unlock(&ring_mutex);
    return 1;
  }
  if (capacity == 0) capacity = 1;

  ring = static_cast<SGEvent *>(calloc(capacity, sizeof(SGEvent)));
  if (ring == nullptr) {
    pthread_mutex_unlock(&ring_mutex);
    return 0;
  }
  ring_capacity = capacity;
  ring_head = 0;
  ring_tail = 0;
  ring_count = 0;
  ring_initialized = true;
  pthread_mutex_unlock(&ring_mutex);

  DEBUG("telemetry ring initialized with capacity %u", capacity);
  return 1;
}

void sg_telemetry_shutdown(void) {
  pthread_mutex_lock(&ring_mutex);
  if (!ring_initialized) {
    pthread_mutex_unlock(&ring_mutex);
    return;
  }

  for (size_t i = 0; i < ring_count; i++) {
    size_t idx = (ring_head + i) % ring_capacity;
    release_event(&ring[idx]);
  }

  ring_head = 0;
  ring_tail = 0;
  ring_count = 0;
  ring_initialized = false;
  pthread_mutex_unlock(&ring_mutex);
}

int sg_enqueue_code(uint32_t code_id, uint32_t code_type, const void *data,
                    uint64_t data_size) {
  if (data == nullptr || data_size == 0) return 0;

  void *copy = malloc(static_cast<size_t>(data_size));
  if (copy == nullptr) {
    count_drop(data_size);
    return 0;
  }
  memcpy(copy, data, static_cast<size_t>(data_size));

  SGEvent event{};
  fill_common(&event, SG_EVENT_CODE);
  event.code.code_id = code_id;
  event.code.code_type = code_type;
  event.code.data = copy;
  event.code.data_size = data_size;

  if (!push_event(&event, data_size)) {
    free(copy);
    return 0;
  }
  return 1;
}

int sg_enqueue_kernel_launch(const SGKernelLaunchEvent *launch) {
  if (launch == nullptr) return 0;

  SGEvent event{};
  fill_common(&event, SG_EVENT_KERNEL_LAUNCH);
  event.launch = *launch;

  return push_event(&event, 0);
}

size_t sg_ring_pop_batch(SGEvent *out, size_t max_events) {
  if (out == nullptr || max_events == 0) return 0;

  pthread_mutex_lock(&ring_mutex);
  if (!ring_initialized || ring == nullptr || ring_capacity == 0 ||
      ring_count == 0) {
    pthread_mutex_unlock(&ring_mutex);
    return 0;
  }

  size_t n = ring_count < max_events ? ring_count : max_events;
  for (size_t i = 0; i < n; i++) {
    out[i] = ring[ring_head];
    ring[ring_head] = SGEvent{};
    ring_head = (ring_head + 1) % ring_capacity;
  }
  ring_count -= n;
  pthread_mutex_unlock(&ring_mutex);
  return n;
}

void sg_ring_release_batch(SGEvent *events, size_t count) {
  if (events == nullptr) return;
  for (size_t i = 0; i < count; i++) {
    release_event(&events[i]);
  }
}

void sg_ring_stats(SGTelemetryStats *stats) {
  if (stats == nullptr) return;

  stats->dropped_events = dropped_events.load(std::memory_order_relaxed);
  stats->dropped_bytes = dropped_bytes.load(std::memory_order_relaxed);

  pthread_mutex_lock(&ring_mutex);
  stats->queued_events = ring_count;
  pthread_mutex_unlock(&ring_mutex);
}

}  // extern "C"
