# Reproducing the Experiments

Run every command from the repository root with the virtual environment
active. Follow the installation and build instructions in
[README.md](README.md) first.

Results are written under `artifacts/` and are intentionally not versioned.
Use the same physical GPU, CUDA and driver stack, clocks, power limit, and
idle-system policy across compared runs. Record those details, the Git commit,
`config.json`, and command output with the resulting artifacts. Unless stated
otherwise, select the GPU with `--gpu 0`.

## Workloads

Build the synthetic workloads for the target GPU. Replace `sm_86` with the
compute capability of the evaluation GPU:

```bash
python workloads/synthetic/scripts/build_binaries.py --cuda-arch sm_86 -j 8
```

Install the hook and run the collector as described in the capturing section
of [README.md](README.md). Run the training and evaluation workloads to create
captures under `artifacts/captures/`.

## Dataset Preparation

Create a manifest from the training captures:

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

## Model Training

The training workflow is:

```text
captures -> normalized SASS windows -> grouped splits -> tokenizer
-> masked-language-model pretraining -> classifier fine-tuning -> model
```

Train the tokenizer, masked-language model, and classifier:

```bash
python scripts/train.py \
  --data artifacts/data/training \
  --output artifacts/model \
  --gpus 1
```

`--gpus N` starts distributed training through `torchrun`.

## Model Evaluation

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

### 1. Window-Budget Selection

Provide a held-out capture manifest containing both benign and mining workloads and a trained 8,192-token model:

```bash
python scripts/window_budget.py \
  --manifest artifacts/evaluation/manifest.jsonl \
  --capture-root . \
  --model artifacts/model \
  --output artifacts/experiments/window-budget \
  --jobs 8
```

The runner evaluates content-token budgets 512, 1,024, 2,048, 3,072, 4,096, 5,120, 6,144, 7,168, and 8,190. For each budget it rebuilds windows from the same captures and evaluates the same model. It computes the fifth percentile of mining scores and the ninety-fifth percentile of benign scores, then maximizes their separation from the 0.50 decision boundary; a tie selects the larger budget. The selection is written to `selection.json`.

Use `--dry-run` to print the full matrix without executing it. The normal training and deployment budgets remain those in `config.json`.

### 2. Behavioral Baseline Training

Prepare one or more JSONL workload manifests. Each row must contain `argv`, `workload`, `binary_label`, and may contain `label`, `family`, `program`, `variant`, `cwd`, and `runtime_sec`. Commands should cover the benign and mining training workloads used for the detector comparison.

```bash
python scripts/baseline_train.py \
  --manifest artifacts/baseline/train-manifest.jsonl \
  --output artifacts/experiments/baseline \
  --gpu 0
```

The collector samples GPU utilization, memory utilization, power, and temperature. Training uses a grouped split with seed 1,337, a 200-tree point-level random forest, and a validation threshold selected at a maximum run-level false-positive rate of 0.01. The model is written to `artifacts/experiments/baseline/model/model.joblib`.

### 3. Throttled Mining With GPU-Sentry

Start online deployment in one shell:

```bash
python scripts/deploy.py --model artifacts/model
```

Run the preset in another shell:

```bash
python scripts/throttle_gpu_sentry.py \
  --output artifacts/experiments/throttle-gpu-sentry \
  --gpu 0 \
  --runtime 180 \
  --repeats 1
```

The matrix contains `ethash_split`, `kawpow_split`, `randomx_gpu_lite_mono`, and `sha256d_mono`, each at target launch rates 5%, 10%, 50%, 75%, and 100%. A 15-second unthrottled pilot estimates launch rate; the runner converts each target percentage to an inter-launch sleep. Captures are labeled with miner, family, target percentage, and repeat.

### 4. Throttled Mining With the Behavioral Baseline

```bash
python scripts/throttle_baseline.py \
  --model artifacts/experiments/baseline/model/model.joblib \
  --output artifacts/experiments/throttle-baseline \
  --gpu 0 \
  --runtime 60 \
  --repeats 1
```

This matrix contains `autolykos2_split`, `cryptonight_gpu_split`, `cuckoo_cycle_split`, `equihash144_5_split`, `ethash_split`, `kawpow_split`, `randomx_gpu_lite_mono`, and `sha256d_mono`, each at 5%, 10%, 25%, 50%, 75%, and 100%. It uses the same pilot calibration method, records counter samples, builds point features, and scores them with the trained baseline.

### 5. Mixed Workloads With the Behavioral Baseline

```bash
python scripts/mixed_baseline.py \
  --model artifacts/experiments/baseline/model/model.joblib \
  --output artifacts/experiments/mixed-baseline \
  --gpu 0 \
  --runtime 180
```

The seven fixed runs are:

| Benign workload | Mining workload | Target rate |
| --- | --- | ---: |
| cuBLAS GEMM | RandomX GPU Lite | 10% |
| cuBLAS GEMM | RandomX GPU Lite | 10% (repeat) |
| PyTorch training | Ethash | 10% |
| cuDNN convolution | KawPow | 10% |
| HPL-like solve | SHA-256d | 10% |
| vLLM inference fallback | RandomX GPU Lite | 50% |
| AES | Ethash | 50% |

Each pair runs concurrently on the selected GPU. The runner collects the same four baseline features and scores the resulting points with the model from Section 2.

## Overrides and audit trail

The preset matrix is defined in `src/gpu_sentry/experiments/presets.py`. Each script documents its path, runtime, repeat, GPU, and dry-run arguments through `--help`. Keep every override in the experiment log; do not edit `config.json` between methods in a comparison.
