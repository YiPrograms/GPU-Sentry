#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the GPU-Sentry detector")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--data", type=Path, default=Path("artifacts/data/training"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/training"))
    parser.add_argument("--reports", type=Path, default=Path("artifacts/reports/training"))
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    common = [
        "--config", str(args.config),
        "--data", str(args.data),
        "--model", str(args.output),
        "--work-dir", str(args.work_dir),
        "--reports", str(args.reports),
    ]
    run([sys.executable, "-m", "gpu_sentry.model.train_tokenizer", *common])
    launcher = ["torchrun", f"--nproc_per_node={args.gpus}"] if args.gpus > 1 else [sys.executable]
    run([*launcher, "-m", "gpu_sentry.model.pretrain_mlm", *common])
    run([*launcher, "-m", "gpu_sentry.model.train_classifier", *common])
    return 0


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
