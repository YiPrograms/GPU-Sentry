NVCC ?= /usr/local/cuda/bin/nvcc

CUDA_ARCH ?= sm_70
OPT_LEVEL ?= -O2

BUILD_DIR ?= build

CPPFLAGS ?= -Iinclude
NVCCFLAGS ?= -std=c++17 $(OPT_LEVEL) -arch=$(CUDA_ARCH) -Wno-deprecated-gpu-targets

MINING_SRC_DIR := src/mining
BENIGN_SRC_DIR := src/benign
MINING_BUILD_DIR := $(BUILD_DIR)/mining
BENIGN_BUILD_DIR := $(BUILD_DIR)/benign

MINING_SOURCES := $(shell find $(MINING_SRC_DIR) -type f -name '*.cu' | sort)
BENIGN_SOURCES := $(shell find $(BENIGN_SRC_DIR) -type f -name '*.cu' | sort)

MINING_KERNELS := $(patsubst $(MINING_SRC_DIR)/%.cu,$(MINING_BUILD_DIR)/%,$(MINING_SOURCES))
BENIGN_KERNELS := $(patsubst $(BENIGN_SRC_DIR)/%.cu,$(BENIGN_BUILD_DIR)/%,$(BENIGN_SOURCES))
KERNELS := $(MINING_KERNELS) $(BENIGN_KERNELS)

.PHONY: all kernels mining benign help clean list list-mining list-benign

all: kernels

kernels: mining benign

mining: $(MINING_KERNELS)

benign: $(BENIGN_KERNELS)

$(MINING_BUILD_DIR)/%: $(MINING_SRC_DIR)/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $< -o $@

$(BENIGN_BUILD_DIR)/%: $(BENIGN_SRC_DIR)/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $< -o $@

help:
	@echo "Available targets:"
	@echo "  all, kernels  Build all CUDA mining and benign kernel executables"
	@echo "  mining        Build CUDA mining kernel executables"
	@echo "  benign        Build CUDA benign kernel executables"
	@echo "  list          Print all discovered CUDA kernel sources"
	@echo "  list-mining   Print discovered CUDA mining kernel sources"
	@echo "  list-benign   Print discovered CUDA benign kernel sources"
	@echo "  clean         Remove build outputs and temporary files"
	@echo
	@echo "Variables:"
	@echo "  NVCC=$(NVCC)"
	@echo "  CUDA_ARCH=$(CUDA_ARCH)"
	@echo "  BUILD_DIR=$(BUILD_DIR)"

list:
	@printf '%s\n' $(MINING_SOURCES) $(BENIGN_SOURCES)

list-mining:
	@printf '%s\n' $(MINING_SOURCES)

list-benign:
	@printf '%s\n' $(BENIGN_SOURCES)

clean:
	rm -rf $(BUILD_DIR)
