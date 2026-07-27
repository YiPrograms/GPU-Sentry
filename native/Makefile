CC       := gcc
CXX      := g++
LD       := ld
PYTHON   := python3
GO       := go
SUDO     ?= sudo

BUILD ?= build
DOCKER_BUILD ?= build-ubuntu20
GPU_SENTRY_SERVER_ADDR ?= 127.0.0.1:59400
LIBCUDA_LINK ?= /usr/lib/x86_64-linux-gnu/libcuda.so.1
LIBCUDA ?= $(shell readlink -f $(LIBCUDA_LINK))
BACKUP_DIR ?= .driver-backups
DRIVER_BACKUP_DIR := $(BACKUP_DIR)/$(notdir $(LIBCUDA))
REAL_LIBCUDA := $(DRIVER_BACKUP_DIR)/libcuda.so.1.real
INSTALL_TMP := $(LIBCUDA).gpu-sentry.tmp

TARGET := $(BUILD)/libcuda.so.1
DOCKER_TARGET := $(DOCKER_BUILD)/libcuda.so.1
GO_DIR := go
GO_ARCHIVE := $(BUILD)/libgpu_sentry_go.a
GO_SERVER := $(BUILD)/gpu-sentry-collector
COLLECTOR_CONFIG := $(BUILD)/collector_config.h
GO_ENV := GOCACHE=/tmp/gpu-sentry-go-cache GOMODCACHE=/tmp/gpu-sentry-go-modcache PATH=$(HOME)/go/bin:$(PATH)
INCLUDES := -Iinclude -Iinclude/vendor
CFLAGS := -O2 -Wall -Wextra -g -fPIC $(INCLUDES)

OBJS := $(BUILD)/loader.o $(BUILD)/trampolines.o $(BUILD)/libcuda_blob.o $(BUILD)/hook.o $(BUILD)/code_registry.o $(BUILD)/cupti_capture.o $(BUILD)/telemetry.o $(BUILD)/telemetry_bridge.o

all: $(TARGET) $(GO_SERVER)

$(BUILD):
	mkdir -p $(BUILD)

$(COLLECTOR_CONFIG): FORCE | $(BUILD)
	@printf '%s\n' '#define GPU_SENTRY_DEFAULT_SERVER_ADDR "$(GPU_SENTRY_SERVER_ADDR)"' > $@.tmp
	@cmp -s $@.tmp $@ || mv $@.tmp $@
	@rm -f $@.tmp

include/cuda_symbols.def: $(REAL_LIBCUDA) hooked_symbols scripts/gen_trampoline_symbols.py
	$(PYTHON) scripts/gen_trampoline_symbols.py $(REAL_LIBCUDA)

include/cuda_trampolines.inc: include/cuda_symbols.def

$(REAL_LIBCUDA):
	@test -n "$(LIBCUDA)" && test -f "$(LIBCUDA)" || { \
		echo "CUDA driver library not found through $(LIBCUDA_LINK)"; \
		exit 1; \
	}
	mkdir -p "$(dir $(REAL_LIBCUDA))"
	cp --preserve=mode,timestamps "$(LIBCUDA)" "$@"

$(BUILD)/libcuda_blob.o: $(REAL_LIBCUDA) | $(BUILD)
	cd "$(dir $(REAL_LIBCUDA))" && \
		$(LD) -r -b binary "$(notdir $(REAL_LIBCUDA))" -o "$(abspath $@)"

$(BUILD)/loader.o: src/loader.c include/cuda_symbols.def | $(BUILD)
	$(CC) $(CFLAGS) -c src/loader.c -o $@

$(BUILD)/trampolines.o: src/trampolines.S include/cuda_trampolines.inc | $(BUILD)
	$(CC) $(CFLAGS) -c src/trampolines.S -o $@

$(BUILD)/hook.o: src/hook.cc include/cuda_symbols.def include/code_registry.h | $(BUILD)
	$(CXX) $(CFLAGS) -c src/hook.cc -o $@

$(BUILD)/code_registry.o: src/code_registry.cc include/code_registry.h include/telemetry.h | $(BUILD)
	$(CXX) $(CFLAGS) -c src/code_registry.cc -o $@

$(BUILD)/cupti_capture.o: src/cupti_capture.cc include/cupti_capture.h include/code_registry.h | $(BUILD)
	$(CXX) $(CFLAGS) -c src/cupti_capture.cc -o $@

$(BUILD)/telemetry.o: src/telemetry.cc include/telemetry.h | $(BUILD)
	$(CXX) $(CFLAGS) -c src/telemetry.cc -o $@

$(BUILD)/telemetry_bridge.o: src/telemetry_bridge.cc include/telemetry.h include/config.h include/cupti_capture.h $(COLLECTOR_CONFIG) | $(BUILD)
	$(CXX) $(CFLAGS) -include $(COLLECTOR_CONFIG) -c src/telemetry_bridge.cc -o $@

$(GO_ARCHIVE): $(GO_DIR)/client/main.go $(GO_DIR)/internal/codeid/*.go $(GO_DIR)/go.mod $(GO_DIR)/go.sum $(GO_DIR)/protocol/telemetry.pb.go include/telemetry.h | $(BUILD)
	cd $(GO_DIR) && $(GO_ENV) $(GO) build -buildmode=c-archive -o ../$(GO_ARCHIVE) ./client

$(GO_SERVER): $(GO_DIR)/cmd/gpu-sentry-collector/*.go $(GO_DIR)/go.mod $(GO_DIR)/go.sum $(GO_DIR)/protocol/telemetry.pb.go | $(BUILD)
	cd $(GO_DIR) && $(GO_ENV) $(GO) build -o ../$(GO_SERVER) ./cmd/gpu-sentry-collector

$(TARGET): $(OBJS) $(GO_ARCHIVE)
	$(CXX) \
	  -shared \
	  -Wl,-soname,libcuda.so.1 \
	  -Wl,--version-script=exports.map \
		-Wl,-z,noexecstack \
		-static-libgcc -static-libstdc++ \
	  $(OBJS) \
	  $(GO_ARCHIVE) \
	  -ldl -lpthread -lm \
	  -o $(TARGET)

clean:
	rm -rf $(BUILD)
	rm -f include/cuda_symbols.def include/cuda_trampolines.inc

docker-build: $(REAL_LIBCUDA)
	docker build \
		-f Dockerfile.ubuntu20 \
		--build-arg "REAL_LIBCUDA=$(REAL_LIBCUDA)" \
		--build-arg "GPU_SENTRY_SERVER_ADDR=$(GPU_SENTRY_SERVER_ADDR)" \
		--target artifacts \
		--output "type=local,dest=$(DOCKER_BUILD)" \
		.

install: $(TARGET)
	@test -f "$(REAL_LIBCUDA)" || { \
		echo "Refusing to install without backup $(REAL_LIBCUDA)"; \
		exit 1; \
	}
	$(SUDO) install -m 0755 "$(TARGET)" "$(INSTALL_TMP)"
	$(SUDO) mv -f "$(INSTALL_TMP)" "$(LIBCUDA)"
	@echo "GPU-Sentry hook installed at $(LIBCUDA)"
	@echo "Run 'make restore' from the repository root to restore the NVIDIA library."

install-docker:
	@test -f "$(DOCKER_TARGET)" || { \
		echo "$(DOCKER_TARGET) does not exist."; \
		echo "Build it with: make docker-build"; \
		exit 1; \
	}
	@test -f "$(REAL_LIBCUDA)" || { \
		echo "Refusing to install without backup $(REAL_LIBCUDA)"; \
		exit 1; \
	}
	$(SUDO) install -m 0755 "$(DOCKER_TARGET)" "$(INSTALL_TMP)"
	$(SUDO) mv -f "$(INSTALL_TMP)" "$(LIBCUDA)"
	@echo "GPU-Sentry Ubuntu 20.04 hook installed at $(LIBCUDA)"
	@echo "Run 'make restore' from the repository root to restore the NVIDIA library."

restore:
	@test -f "$(REAL_LIBCUDA)" || { \
		echo "No backup for the current driver: $(REAL_LIBCUDA)"; \
		exit 1; \
	}
	$(SUDO) install -m 0755 "$(REAL_LIBCUDA)" "$(INSTALL_TMP)"
	$(SUDO) mv -f "$(INSTALL_TMP)" "$(LIBCUDA)"
	@echo "NVIDIA CUDA driver library restored at $(LIBCUDA)"

FORCE:

.PHONY: all clean docker-build install install-docker restore FORCE
