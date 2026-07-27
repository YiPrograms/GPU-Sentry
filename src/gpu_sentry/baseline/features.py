from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .io import read_jsonl
from .model import FEATURES


def build_point_features(raw_dir: Path, output_csv: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    labels: Counter[str] = Counter()
    run_dirs = sorted(path for path in raw_dir.iterdir() if path.is_dir())
    for run_dir in run_dirs:
        metadata_file = run_dir / "metadata.json"
        samples_file = run_dir / "device_metrics.jsonl"
        if not metadata_file.exists() or not samples_file.exists():
            continue
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        label = str(metadata.get("binary_label"))
        labels[label] += 1
        for index, sample in enumerate(read_jsonl(samples_file)):
            row = {
                "run_id": metadata.get("run_id"),
                "workload": metadata.get("workload"),
                "program": metadata.get("program"),
                "target_percent": metadata.get("target_percent"),
                "binary_label": label,
                "sample_index": index,
                "timestamp_ns": sample.get("timestamp_ns"),
            }
            row.update({name: sample.get(name) for name in FEATURES})
            rows.append(row)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fields = ["run_id", "workload", "program", "target_percent", "binary_label", "sample_index", "timestamp_ns", *FEATURES]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"runs": len(run_dirs), "points": len(rows), "labels": dict(labels)}
