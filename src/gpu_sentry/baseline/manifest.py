from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST_PATTERNS = ["artifacts/captures/manifest.jsonl"]


def expand_manifest_patterns(patterns: list[str] | None) -> list[Path]:
    chosen = patterns or DEFAULT_MANIFEST_PATTERNS
    paths: list[Path] = []
    for pattern in chosen:
        matches = sorted(Path(path) for path in glob.glob(pattern))
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(pattern))
    return paths


def load_manifest_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_manifest_path"] = str(path)
                row["_manifest_line"] = line_no
                rows.append(row)
    return rows


def row_matches(row: dict[str, Any], pattern: str | None) -> bool:
    if not pattern:
        return True
    needle = pattern.lower()
    fields = [
        row.get("workload"),
        row.get("program"),
        row.get("variant"),
        row.get("family"),
        row.get("label"),
        row.get("binary_label"),
    ]
    return any(needle in str(field).lower() for field in fields if field is not None)
