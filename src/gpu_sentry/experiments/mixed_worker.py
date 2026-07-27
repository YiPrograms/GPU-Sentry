from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benign", required=True)
    parser.add_argument("--miner", type=Path, required=True)
    parser.add_argument("--sleep-us", type=int, required=True)
    parser.add_argument("--runtime", type=int, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    benign = root / "workloads/benign/benign_gpu_workloads.py"
    env = os.environ.copy()
    benign_process = subprocess.Popen([sys.executable, str(benign), args.benign, "--seconds", str(args.runtime)], env=env)
    miner_process = subprocess.Popen([
        str(args.miner), str(args.runtime), "--sleep-between-launches-us", str(args.sleep_us)
    ], env=env)
    benign_status = benign_process.wait()
    miner_status = miner_process.wait()
    return benign_status or miner_status


if __name__ == "__main__":
    raise SystemExit(main())

