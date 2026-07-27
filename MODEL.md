# Model Release

The GPU-Sentry deployment model is a binary ModernBERT classifier trained on
normalized GPU SASS.

## Published Model

- Repository: `kuokuoyiyi/gpu-sentry-modernbert`
- Revision: `99bee47277a38694f07e97f26c4a98bae81e369c`
- Labels: `benign`, `mining`
- Maximum sequence length: 8,192 tokens

Download the pinned release:

```bash
hf auth login
hf download kuokuoyiyi/gpu-sentry-modernbert \
  --revision 99bee47277a38694f07e97f26c4a98bae81e369c \
  --local-dir artifacts/model
```

Verify the model files:

```bash
test -f artifacts/model/config.json
test -f artifacts/model/pytorch_model.bin
test -f artifacts/model/tokenizer.json
test -f artifacts/model/tokenizer_config.json
test -f artifacts/model/special_tokens_map.json
```

## Publishing a New Revision

Train or place the export in `artifacts/model`, authenticate, and run the
upload script:

```bash
hf auth login
scripts/upload_model.sh \
  artifacts/model \
  kuokuoyiyi/gpu-sentry-modernbert
```

The script uploads the model weights, tokenizer, model configuration, checksum
file, and model card.

The underlying Hugging Face CLI command is:

```bash
hf upload kuokuoyiyi/gpu-sentry-modernbert artifacts/model . \
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
```
