#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gpu_sentry.experiments.runner import throttle_gpu_sentry


def main() -> int:
    parser = argparse.ArgumentParser(description="Run throttled mining against GPU-Sentry")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments/throttle-gpu-sentry"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--runtime", type=int, default=180)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return throttle_gpu_sentry(args, args.config.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
