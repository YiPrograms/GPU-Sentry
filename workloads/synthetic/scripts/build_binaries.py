#!/usr/bin/env python3
"""Build synthetic CUDA kernels at O2/O3 and emit a manifest."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

PRIMARY_LABEL_BY_SUBFAMILY = {
    # Benign: AI / HPC / graph / signal compute
    "ai": "benign_compute_like",
    "hpc": "benign_compute_like",
    "graph": "benign_compute_like",
    "image_signal": "benign_compute_like",
    "cuda_samples": "benign_compute_like",
    # Benign: crypto/hash/compression hard negatives
    "crypto_benign": "benign_crypto_hash_like",
    "compression": "benign_crypto_hash_like",
    # Benign: memory operations
    "memory": "benign_memory_like",
    # Mining subfamilies
    "cryptonight_randomx_scratchpad": "mining_like",
    "cuckoo_graph_cycle": "mining_like",
    "equihash_solver": "mining_like",
    "ethash_dag_keccak": "mining_like",
    "heavyhash_matrix_like": "mining_like",
    "memory_hard_table_hash": "mining_like",
    "modern_alt_hash_pow": "mining_like",
    "multi_hash_chain": "mining_like",
    "progpow_kawpow_random_math": "mining_like",
    "pure_hash_nonce_search": "mining_like",
}

OPT_LEVELS = {
    "O2": "-O2",
    "O3": "-O3",
}

@dataclass(frozen=True)
class KernelSource:
    source_path: Path
    kind: str
    family: str
    name: str

def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Build all synthetic CUDA kernels at O2/O3 into binaries/."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=project_dir,
        help="Synthetic kernels project directory containing the Makefile.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "binaries",
        help="Directory where renamed binaries and manifest.jsonl are written.",
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=project_dir / "build" / "binaries",
        help="Temporary Makefile BUILD_DIR root used for O2/O3 builds.",
    )
    parser.add_argument(
        "--cuda-arch",
        default=None,
        help="Optional CUDA_ARCH value passed through to make, for example sm_86.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Optional parallel make job count.",
    )
    return parser.parse_args()


def discover_sources(project_dir: Path) -> list[KernelSource]:
    sources: list[KernelSource] = []
    src_dir = project_dir / "src"

    for kind in ("mining", "benign"):
        kind_dir = src_dir / kind
        for source_path in sorted(kind_dir.glob("*/*.cu")):
            family = source_path.parent.name
            if family not in PRIMARY_LABEL_BY_SUBFAMILY:
                raise ValueError(
                    f"No label mapping for subfamily {family!r} from {source_path}"
                )
            sources.append(
                KernelSource(
                    source_path=source_path,
                    kind=kind,
                    family=family,
                    name=source_path.stem,
                )
            )

    if not sources:
        raise RuntimeError(f"No CUDA sources found under {src_dir}")

    return sources


def run_make(
    project_dir: Path,
    build_dir: Path,
    opt_flag: str,
    jobs: int | None,
    cuda_arch: str | None,
) -> None:
    cmd = [
        "make",
        "-C",
        str(project_dir),
    ]
    if jobs:
        cmd.append(f"-j{jobs}")
    cmd.extend(
        [
            "kernels",
            f"BUILD_DIR={build_dir}",
            f"OPT_LEVEL={opt_flag}",
        ]
    )
    if cuda_arch:
        cmd.append(f"CUDA_ARCH={cuda_arch}")

    subprocess.run(cmd, check=True)


def copy_binaries(
    sources: list[KernelSource],
    build_dir: Path,
    output_dir: Path,
    opt_level: str,
) -> list[dict[str, str]]:
    suffix = opt_level.lower()
    records: list[dict[str, str]] = []
    seen_binary_names: set[str] = set()

    for source in sources:
        built_binary = build_dir / source.kind / source.family / source.name
        binary_name = f"{source.name}_{suffix}"
        output_binary = output_dir / binary_name

        if binary_name in seen_binary_names:
            raise RuntimeError(f"Duplicate binary output name: {binary_name}")
        seen_binary_names.add(binary_name)

        if not built_binary.is_file():
            raise RuntimeError(f"Expected built binary is missing: {built_binary}")

        shutil.copy2(built_binary, output_binary)
        output_binary.chmod(output_binary.stat().st_mode | 0o111)

        records.append(
            {
                "binary_name": binary_name,
                "label": PRIMARY_LABEL_BY_SUBFAMILY[source.family],
                "family": source.family,
                "opt_level": opt_level,
            }
        )

    return records


def write_manifest(output_dir: Path, records: list[dict[str, str]]) -> None:
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for record in records:
            manifest.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    output_dir = args.output_dir.resolve()
    build_root = args.build_root.resolve()

    if not (project_dir / "Makefile").is_file():
        raise FileNotFoundError(f"Makefile not found in {project_dir}")

    sources = discover_sources(project_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    all_records: list[dict[str, str]] = []
    for opt_level, opt_flag in OPT_LEVELS.items():
        build_dir = build_root / opt_level.lower()
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)

        run_make(project_dir, build_dir, opt_flag, args.jobs, args.cuda_arch)
        all_records.extend(copy_binaries(sources, build_dir, output_dir, opt_level))

    write_manifest(output_dir, all_records)
    print(f"Built {len(all_records)} binaries into {output_dir}")
    print(f"Wrote manifest to {output_dir / 'manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
