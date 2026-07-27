#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-artifacts/model}"
REPO_ID="${2:-kuokuoyiyi/gpu-sentry-modernbert}"

hf auth whoami
hf upload "$REPO_ID" "$MODEL_DIR" . \
  --repo-type model \
  --private \
  --commit-message "Publish GPU-Sentry model" \
  --include README.md \
  --include SHA256SUMS \
  --include config.json \
  --include "model*.safetensors" \
  --include "pytorch_model*.bin" \
  --include special_tokens_map.json \
  --include tokenizer.json \
  --include tokenizer_config.json
