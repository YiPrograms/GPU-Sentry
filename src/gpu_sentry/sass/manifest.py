"""Binary-label manifest helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ManifestError(RuntimeError):
    """Raised when a binary manifest cannot be used."""


def load_binary_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Load the synthetic binary manifest."""
    if not path.exists():
        raise ManifestError(f"missing binary manifest: {path}")

    by_name: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{line_no}: invalid JSON: {exc}") from exc

            binary_name = obj.get("binary_name")
            if not binary_name:
                raise ManifestError(f"{path}:{line_no}: missing binary_name")
            if binary_name in by_name:
                raise ManifestError(f"{path}:{line_no}: duplicate binary_name {binary_name}")

            missing = [key for key in ("family", "label", "opt_level") if key not in obj]
            if missing:
                raise ManifestError(
                    f"{path}:{line_no}: {binary_name} missing fields: {', '.join(missing)}"
                )
            by_name[binary_name] = {
                "family": str(obj["family"]),
                "label": str(obj["label"]),
                "opt_level": str(obj["opt_level"]),
            }
    return by_name


def workload_manifest(workload_name: str, entry: dict[str, str]) -> dict[str, str]:
    """Return a dataset workload manifest with optional capture provenance."""
    manifest = {
        "workload": workload_name,
        "family": entry["family"],
        "label": entry["label"],
        "opt_level": entry["opt_level"],
    }
    for key in (
        "program",
        "variant",
        "capture_id",
        "source_capture_path",
        "binary_label",
        "parent_workload",
        "window_id",
        "window_type",
        "l0_group_key",
    ):
        if key in entry:
            manifest[key] = entry[key]
    return manifest


def write_workload_manifest(path: Path, workload_name: str, entry: dict[str, str]) -> None:
    write_json(path, workload_manifest(workload_name, entry))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")
