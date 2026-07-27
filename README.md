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
  captures modules, functions, and kernel launches
      |
      v  TCP COLLECTOR_IP:COLLECTOR_PORT
Go collector
  resolves launch metadata and stores captures
      |
      v  Unix socket /tmp/gpu-sentry.sock
Python processor
  disassembles cubins and builds normalized SASS windows
      |
      v
ModernBERT classifier
      |
      v
Decision policy
  continues or terminates the CUDA process
```

The hook embeds the active NVIDIA `libcuda` implementation and forwards CUDA
calls to it. Applications continue running when the analysis services are
unavailable. CUPTI module callbacks capture cubins loaded internally by CUDA
libraries such as cuBLAS. The telemetry client deduplicates identical code
objects by type and SHA-256 and remaps launches to the first matching code ID.
PTX, fatbins, and generated cubins remain separate artifacts.

## Repository Layout

```text
config.json                  shared project configuration
scripts/                     deployment, training, and experiment scripts
native/                      CUDA hook and Go collector
src/gpu_sentry/online/       online processing and decision policies
src/gpu_sentry/sass/         capture-to-SASS dataset pipeline
src/gpu_sentry/model/        tokenizer, training, and evaluation
src/gpu_sentry/baseline/     behavioral baseline
src/gpu_sentry/experiments/  paper experiment presets and runners
workloads/                   benign and synthetic CUDA workloads
artifacts/                   generated captures, models, and reports
```

## Requirements

- Linux x86-64
- NVIDIA GPU and driver
- CUDA 12.6 with `nvcc`, `nvdisasm`, and `cuobjdump`
- CUPTI from the CUDA toolkit
- Python 3.11 or newer
- Go, GCC, G++, GNU ld, and make
- Docker for the Ubuntu 20.04 build
- `sudo` access for hook installation and restoration

## Installation

Create the Python environment:

```bash
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
make -j COLLECTOR_IP=127.0.0.1 COLLECTOR_PORT=59400
```

The build creates:

```text
native/build/libcuda.so.1
native/build/gpu-sentry-collector
```

For an Ubuntu 20.04-compatible build:

```bash
make docker-build COLLECTOR_IP=127.0.0.1 COLLECTOR_PORT=59400
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
make install COLLECTOR_IP=127.0.0.1 COLLECTOR_PORT=59400
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

A policy accepts the decision settings and implements
`update(mining_probability)`:

```python
from typing import Any, Mapping

from gpu_sentry.online.policy import Decision


class ConsecutivePolicy:
    def __init__(self, settings: Mapping[str, Any]):
        self.required = int(settings["consecutive_windows"])
        self.threshold = float(settings["mining_probability"])
        self.count = 0

    def update(self, mining_probability: float) -> Decision:
        self.count = self.count + 1 if mining_probability >= self.threshold else 0
        suspicious = self.count >= self.required
        return Decision(
            suspicious=suspicious,
            reason="consecutive_windows" if suspicious else "below_threshold",
            details={"consecutive_windows": self.count},
        )
```

Save the class in an importable module and reference it in `config.json`:

```json
"decision": {
  "policy": "gpu_sentry.online.custom_policy:ConsecutivePolicy",
  "consecutive_windows": 3,
  "mining_probability": 0.9
}
```

GPU-Sentry creates one policy instance for each CUDA process.

## Capturing Workloads

Install the hook and start the collector:

```bash
native/build/gpu-sentry-collector \
  --config config.json \
  --captures artifacts/captures
```

Run CUDA workloads in another shell. Each process creates a capture directory
under `artifacts/captures/`.

## Training

```text
CUDA workloads -> captures -> normalized SASS windows -> grouped splits
-> tokenizer -> MLM pretraining -> classifier fine-tuning -> model
```

Build the synthetic workloads for the target GPU:

```bash
python workloads/synthetic/scripts/build_binaries.py --cuda-arch sm_86 -j 8
```

Create a capture manifest:

```bash
python -m gpu_sentry.sass.capture_manifest \
  --captures-dir artifacts/captures \
  --binary-manifest workloads/synthetic/binaries/manifest.jsonl \
  --output artifacts/captures/manifest.jsonl
```

Build the training dataset:

```bash
python scripts/build_dataset.py \
  --purpose training \
  --manifest artifacts/captures/manifest.jsonl \
  --capture-root . \
  --output artifacts/data/training \
  --replace
```

Train the tokenizer, masked-language model, and classifier:

```bash
python scripts/train.py \
  --data artifacts/data/training \
  --output artifacts/model \
  --gpus 1
```

`--gpus N` starts distributed training through `torchrun`.

## Evaluation

Build an evaluation dataset from held-out captures:

```bash
python scripts/build_dataset.py \
  --purpose evaluation \
  --manifest artifacts/evaluation/manifest.jsonl \
  --capture-root . \
  --output artifacts/data/evaluation \
  --replace
```

Run evaluation:

```bash
python scripts/evaluate.py \
  --dataset-dir artifacts/data/evaluation \
  --checkpoint artifacts/model \
  --split all \
  --reports artifacts/reports/evaluation
```

## Paper Experiments

### Window-Budget Selection

```bash
python scripts/window_budget.py \
  --manifest artifacts/evaluation/manifest.jsonl \
  --capture-root . \
  --model artifacts/model \
  --output artifacts/experiments/window-budget \
  --jobs 8
```

### Behavioral Baseline Training

```bash
python scripts/baseline_train.py \
  --manifest artifacts/baseline/train-manifest.jsonl \
  --output artifacts/experiments/baseline \
  --gpu 0
```

### Throttled Mining With GPU-Sentry

Start `python scripts/deploy.py --model artifacts/model`, then run:

```bash
python scripts/throttle_gpu_sentry.py \
  --output artifacts/experiments/throttle-gpu-sentry \
  --gpu 0 \
  --runtime 180 \
  --repeats 1
```

### Throttled Mining With the Baseline

```bash
python scripts/throttle_baseline.py \
  --model artifacts/experiments/baseline/model/model.joblib \
  --output artifacts/experiments/throttle-baseline \
  --gpu 0 \
  --runtime 60 \
  --repeats 1
```

### Mixed Benign and Mining Workloads

```bash
python scripts/mixed_baseline.py \
  --model artifacts/experiments/baseline/model/model.joblib \
  --output artifacts/experiments/mixed-baseline \
  --gpu 0 \
  --runtime 180
```

Each experiment script supports `--dry-run`. See
[REPRODUCING.md](REPRODUCING.md) for the workload matrices and reporting
requirements.
