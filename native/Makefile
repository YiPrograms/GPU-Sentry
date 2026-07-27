CC       := gcc
CXX      := g++
LD       := ld
OBJCOPY  := objcopy
PYTHON   := python3
GO       := go
SUDO     ?= sudo

DEBUG ?= 1

BUILD ?= build
DOCKER_BUILD ?= build-ubuntu20
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
GO_ENV := GOCACHE=/tmp/gpu-sentry-go-cache GOMODCACHE=/tmp/gpu-sentry-go-modcache PATH=$(HOME)/go/bin:$(PATH)
CONFIG_STAMP := $(BUILD)/.config-debug-$(DEBUG)

ifeq ($(DEBUG),1)
CFLAGS += -Wall -Wextra -g -fPIC -DCUHOOK_DEBUG
else
CFLAGS += -O2 -Wall -Wextra -g -fPIC
endif

INCLUDES := -Iinclude -Iinclude/vendor
CFLAGS += $(INCLUDES)

CXXFLAGS = $(CFLAGS) -std=c++17

LDFLAGS := -shared -ldl -lpthread

OBJS := $(BUILD)/loader.o $(BUILD)/trampolines.o $(BUILD)/libcuda_blob.o $(BUILD)/hook.o $(BUILD)/code_registry.o $(BUILD)/telemetry.o $(BUILD)/telemetry_bridge.o

all: $(TARGET) $(GO_SERVER)

$(BUILD):
	mkdir -p $(BUILD)

$(CONFIG_STAMP): | $(BUILD)
	rm -f $(BUILD)/.config-debug-*
	touch $@

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

$(BUILD)/loader.o: src/loader.c include/cuda_symbols.def $(CONFIG_STAMP) | $(BUILD)
	$(CC) $(CFLAGS) -c src/loader.c -o $@

$(BUILD)/trampolines.o: src/trampolines.S include/cuda_trampolines.inc $(CONFIG_STAMP) | $(BUILD)
	$(CC) $(CFLAGS) -c src/trampolines.S -o $@

$(BUILD)/hook.o: src/hook.cc include/cuda_symbols.def include/code_registry.h $(CONFIG_STAMP) | $(BUILD)
	$(CXX) $(CFLAGS) -c src/hook.cc -o $@

$(BUILD)/code_registry.o: src/code_registry.cc include/code_registry.h include/telemetry.h $(CONFIG_STAMP) | $(BUILD)
	$(CXX) $(CFLAGS) -c src/code_registry.cc -o $@

$(BUILD)/telemetry.o: src/telemetry.cc include/telemetry.h $(CONFIG_STAMP) | $(BUILD)
	$(CXX) $(CFLAGS) -c src/telemetry.cc -o $@

$(BUILD)/telemetry_bridge.o: src/telemetry_bridge.cc include/telemetry.h include/config.h $(CONFIG_STAMP) | $(BUILD)
	$(CXX) $(CFLAGS) -c src/telemetry_bridge.cc -o $@

$(GO_ARCHIVE): $(GO_DIR)/client/main.go $(GO_DIR)/go.mod $(GO_DIR)/go.sum $(GO_DIR)/protocol/telemetry.pb.go include/telemetry.h | $(BUILD)
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

.PHONY: all clean docker-build install install-docker restore
