# Synthetic CUDA workloads

This corpus contains benign and mining-like CUDA programs used to generate GPU-Sentry captures. Mining-like programs are reduced offline benchmarks: they have no wallet, pool, network, or share-submission code.

Build both `-O2` and `-O3` variants from the repository root:

```bash
python workloads/synthetic/scripts/build_binaries.py --cuda-arch sm_86 -j 8
```

The command writes executables and `manifest.jsonl` to `workloads/synthetic/binaries/`. Change `sm_86` to the compute capability of the target GPU.

The programs are offline benchmarks only: there is no pool mining, no wallet handling, no network code, and no real mining submission behavior.

Every generated executable source accepts a mandatory first positional argument:

```bash
./<program> <runtime_seconds> [optional args...]
```

Examples:

```bash
./ethash 60
./kawpow_split 120
./verthash 300 --dataset-mb 128
```

## Common options

All programs share the same parser and support:

* `<runtime_seconds>` (mandatory; must be a positive integer)
* `--blocks <N>`
* `--threads <N>`
* `--nonces-per-thread <N>`
* `--dataset-mb <N>`
* `--scratchpad-mb <N>`
* `--seed <N>`
* `--sync-every <N>`

## Standard output

Each benchmark prints:

```text
algorithm=<name>
variant=<mono|split>
runtime_seconds=<N>
threads_per_block=<N>
total_launches=<N>
total_nonces=<N>
result_count=<N>
checksum=0x...
status=ok
```

See the root `README.md` for capture, training, evaluation, and deployment commands.
