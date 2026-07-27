#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the GPU-Sentry processor and collector")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--captures", type=Path, default=Path("artifacts/captures"))
    parser.add_argument(
        "--collector",
        type=Path,
        default=Path("native/build/gpu-sentry-collector"),
    )
    args = parser.parse_args()
    if not args.collector.is_file():
        parser.error(f"collector not built: {args.collector}")

    env = os.environ.copy()
    env["GPU_SENTRY_DISABLE"] = "1"
    children: list[subprocess.Popen[bytes]] = []
    try:
        children.append(subprocess.Popen(
            [
                sys.executable,
                "-m",
                "gpu_sentry.online.processor",
                "--config",
                str(args.config),
                "--model",
                str(args.model.resolve()),
            ],
            env=env,
        ))
        children.append(subprocess.Popen(
            [
                str(args.collector),
                "--config",
                str(args.config),
                "--captures",
                str(args.captures.resolve()),
            ],
            env=env,
        ))
        while all(child.poll() is None for child in children):
            time.sleep(0.25)
        return next((child.returncode for child in children if child.returncode is not None), 0)
    except KeyboardInterrupt:
        return 130
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
