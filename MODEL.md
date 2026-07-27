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

Train or place the export in `artifacts/model`, then upload the inference files:

```bash
python - <<'PY'
from huggingface_hub import HfApi

repo_id = "kuokuoyiyi/gpu-sentry-modernbert"
api = HfApi()
api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path="artifacts/model",
    allow_patterns=[
        "README.md",
        "SHA256SUMS",
        "config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
    commit_message="Publish GPU-Sentry model",
)
PY
```

Record the returned commit hash in `README.md` for reproducible deployments.
