# GPU-Sentry

GPU-Sentry detects mining-like CUDA workloads from their executed SASS
instructions. It combines a CUDA driver hook, a capture collector, an online
SASS pipeline, and a ModernBERT classifier.

## Architecture

```text
CUDA application
      |
      v
CUDA driver hook
      |
      v
Collector
      |
      v
SASS processing
      |
      v
Classifier
      |
      v
Decision policy
  continues or terminates the CUDA process
```

## Repository Layout

```text
config.json                  shared project configuration
scripts/                     deployment, training, and experiment scripts
native/                      CUDA hook and Go collector
src/gpu_sentry/online/       online processing and decision policies
src/gpu_sentry/sass/         capture-to-SASS dataset pipeline
src/gpu_sentry/model/        tokenizer, training, and evaluation
src/gpu_sentry/baseline/     behavioral baseline
src/gpu_sentry/experiments/  experiment presets and runners
workloads/                   benign and synthetic CUDA workloads
artifacts/                   generated captures, models, and reports
```

## Requirements

- Linux x86-64
- NVIDIA GPU and driver
- A CUDA toolkit compatible with the GPU and driver, including `nvcc`,
  `nvdisasm`, `cuobjdump`, and CUPTI
- Python 3.11 or newer
- Go, GCC, G++, GNU ld, and make
- Docker for the Ubuntu 20.04 build
- `sudo` access for hook installation and restoration

## Installation

Create the Python environment:

```bash
git clone git@github.com:YiPrograms/GPU-Sentry.git
cd GPU-Sentry

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Build the hook and collector on the deployment host. Set the address where the
hook can reach the collector:

```bash
GPU_SENTRY_SERVER_ADDR=127.0.0.1:59400 make -j
```

The build creates:

```text
native/build/libcuda.so.1
native/build/gpu-sentry-collector
```

For an Ubuntu 20.04-compatible build:

```bash
GPU_SENTRY_SERVER_ADDR=127.0.0.1:59400 make docker-build
```

The Docker artifacts are written to `native/build-ubuntu20/`. The build records
the required glibc versions in
`native/build-ubuntu20/glibc-versions.txt`.

Use the collector host's reachable IP when the CUDA application and collector
run on different machines. Set `collector.listen_address` in `config.json` to
the matching address and port. For example, a collector accepting remote
connections can listen on `0.0.0.0:59400` while the hook uses the collector
host's IP.

## Model

Authenticate with Hugging Face and download the released model:

```bash
hf auth login
hf download kuokuoyiyi/gpu-sentry-modernbert \
  --revision 99bee47277a38694f07e97f26c4a98bae81e369c \
  --local-dir artifacts/model
```

GPU-Sentry loads the model from `artifacts/model`.

Upload a trained model with the Hugging Face CLI:

```bash
scripts/upload_model.sh \
  artifacts/model \
  kuokuoyiyi/gpu-sentry-modernbert
```

See [MODEL.md](MODEL.md) for model release details.

## Deployment

The build stores a driver-specific copy of the active NVIDIA library under
`native/.driver-backups/`.

Install the native hook:

```bash
GPU_SENTRY_SERVER_ADDR=127.0.0.1:59400 make install
```

Install the Ubuntu 20.04 hook:

```bash
make install-docker
```

Start the collector and online processor:

```bash
source .venv/bin/activate
python scripts/deploy.py \
  --model artifacts/model \
  --captures artifacts/captures \
  --collector native/build/gpu-sentry-collector
```

Use `native/build-ubuntu20/gpu-sentry-collector` with the Docker build.
CUDA applications can now run normally:

```bash
your-cuda-program [arguments...]
```

### Verify Detection

In another shell, check the GPU compute capability:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

Build the SHA-256d test workload for that capability. For example, use
`CUDA_ARCH=sm_70` for compute capability 7.0:

```bash
make -C workloads/synthetic \
  NVCC=/usr/local/cuda/bin/nvcc \
  CUDA_ARCH=sm_70 \
  build/mining/pure_hash_nonce_search/sha256d_mono
```

Run the workload:

```bash
workloads/synthetic/build/mining/pure_hash_nonce_search/sha256d_mono 60
```

The default policy reports cryptomining and terminates the process with exit
code `101`. Captures are written under `artifacts/captures/`.

The deployment script sets `GPU_SENTRY_DISABLE=1` for the processor and
collector so the processor's PyTorch inference is not captured recursively.
The hook uses the collector address compiled into the library.
`GPU_SENTRY_SERVER_ADDR=host:port` overrides it for one process.
`GPU_SENTRY_CUPTI_PATH` selects a nonstandard `libcupti.so` location.
`GPU_SENTRY_DISABLE=1` disables telemetry for one process.

Stop the deployment and restore the NVIDIA library:

```bash
make restore
```

Rebuild and reinstall after an NVIDIA driver upgrade.

## Configuration

[`config.json`](config.json) contains model, windowing, decision, collector, and
online-processing settings. The default decision policy evaluates the latest
16 windows and reports mining when:

- mean mining probability is at least `0.50`
- maximum mining probability is at least `0.90`

### Custom Decision Policy

The `decision.policy` setting uses `module:ClassName` syntax:

```json
"decision": {
  "policy": "gpu_sentry.online.policy:RollingMeanMaxPolicy",
  "history_windows": 16,
  "mean_mining_probability": 0.5,
  "max_mining_probability": 0.9
}
```

A policy accepts the decision settings and implements `decide(observation)`.
Each observation contains the model output, window features, process
information, signature-cache metadata, and the kernel launches in the
classified window. Kernel launches include the kernel name, code ID,
grid/block dimensions, stream, timestamps, and L0 kernel metadata.

```python
from typing import Any, Mapping

from gpu_sentry.online.policy import Decision, PolicyInput


class BitwisePolicy:
    def __init__(self, settings: Mapping[str, Any]):
        self.probability = float(settings["mining_probability"])
        self.bitwise_ratio = float(settings["bitwise_integer_ratio"])

    def decide(self, observation: PolicyInput) -> Decision:
        ratio = float(
            observation.window_features.get("max_bitwise_integer_ratio", 0.0)
        )
        kernel_names = {
            str(launch.get("kernel_name"))
            for launch in observation.kernel_launches
        }
        suspicious = (
            observation.mining_probability >= self.probability
            and ratio >= self.bitwise_ratio
        )
        return Decision(
            suspicious=suspicious,
            reason="mining_and_bitwise_thresholds" if suspicious else "below_threshold",
            details={
                "policy": type(self).__name__,
                "mining_probability": observation.mining_probability,
                "max_bitwise_integer_ratio": ratio,
                "kernel_names": sorted(kernel_names),
            },
        )
```

Save the class in an importable module and reference it in `config.json`:

```json
"decision": {
  "policy": "gpu_sentry.online.custom_policy:BitwisePolicy",
  "mining_probability": 0.9,
  "bitwise_integer_ratio": 0.7
}
```

GPU-Sentry creates one policy instance for each CUDA process and delivers
observations in window order. A policy may be stateless or retain any state it
needs; GPU-Sentry does not impose a history strategy.

## Capturing Workloads

Install the hook and start the collector:

```bash
native/build/gpu-sentry-collector \
  --config config.json \
  --captures artifacts/captures
```

Run CUDA workloads in another shell. Each process creates a capture directory
under `artifacts/captures/`.

## Reproducing Results

See [REPRODUCING.md](REPRODUCING.md) for dataset preparation, model training,
evaluation, and the experiment commands used in the paper.
