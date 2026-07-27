"""Metrics for the single GPU-Sentry binary decision policy."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .data import ChunkRecord, WorkloadRecord, workload_group_id


def softmax_rows(logits: Any) -> list[list[float]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to compute probabilities") from exc
    values = np.asarray(logits, dtype="float64")
    values -= values.max(axis=1, keepdims=True)
    exponentials = np.exp(values)
    return (exponentials / exponentials.sum(axis=1, keepdims=True)).tolist()


def aggregate_chunk_probabilities(
    chunks: list[ChunkRecord], probabilities: list[list[float]]
) -> dict[str, dict[str, Any]]:
    if len(chunks) != len(probabilities):
        raise ValueError("chunks and probabilities must have the same length")
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    first: dict[str, ChunkRecord] = {}
    for chunk, row in zip(chunks, probabilities):
        grouped[chunk.workload].append([float(value) for value in row])
        first.setdefault(chunk.workload, chunk)

    output: dict[str, dict[str, Any]] = {}
    for workload, rows in grouped.items():
        means = [sum(row[index] for row in rows) / len(rows) for index in range(len(rows[0]))]
        source = first[workload]
        output[workload] = {
            "workload": workload,
            "label": source.label,
            "label_id": source.label_id,
            "pred_id": max(range(len(means)), key=means.__getitem__),
            "probabilities": means,
            "mining_probability": means[1],
            "num_chunks": len(rows),
            "source_path": source.source_path,
        }
    return output


def add_no_window_predictions(
    records: list[WorkloadRecord], predictions: dict[str, dict[str, Any]]
) -> None:
    for record in records:
        if not record.row.get("no_l0_window") or record.workload in predictions:
            continue
        predictions[record.workload] = {
            "workload": record.workload,
            "label": record.label,
            "label_id": record.label_id,
            "pred_id": 0,
            "probabilities": [1.0, 0.0],
            "mining_probability": 0.0,
            "num_chunks": 0,
            "source_path": str(record.source_path),
            "no_l0_window": True,
        }


def workload_metrics(
    records: list[WorkloadRecord],
    chunks: list[ChunkRecord],
    probabilities: list[list[float]],
    id2label: dict[int, str],
    *,
    history_windows: int,
    mean_threshold: float,
    max_threshold: float,
) -> dict[str, Any]:
    predictions = aggregate_chunk_probabilities(chunks, probabilities)
    add_no_window_predictions(records, predictions)
    return grouped_workload_metrics(
        records,
        predictions,
        id2label,
        history_windows=history_windows,
        mean_threshold=mean_threshold,
        max_threshold=max_threshold,
    )


def grouped_workload_metrics(
    records: list[WorkloadRecord],
    predictions: dict[str, dict[str, Any]],
    id2label: dict[int, str],
    *,
    history_windows: int,
    mean_threshold: float,
    max_threshold: float,
) -> dict[str, Any]:
    windows: dict[str, list[tuple[WorkloadRecord, dict[str, Any]]]] = defaultdict(list)
    for record in records:
        windows[workload_group_id(record)].append((record, predictions[record.workload]))

    rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for group_id, group in sorted(windows.items()):
        group.sort(key=lambda item: (int(item[0].row.get("window_index", 0)), item[0].workload))
        scores = [float(item[1]["mining_probability"]) for item in group]
        triggered = False
        largest_mean = 0.0
        largest_max = 0.0
        for index, ((record, prediction), score) in enumerate(zip(group, scores)):
            history = scores[max(0, index + 1 - history_windows) : index + 1]
            rolling_mean = sum(history) / len(history)
            rolling_max = max(history)
            suspicious = rolling_mean >= mean_threshold and rolling_max >= max_threshold
            triggered = triggered or suspicious
            largest_mean = max(largest_mean, rolling_mean)
            largest_max = max(largest_max, rolling_max)
            window_rows.append({
                **prediction,
                "group_id": group_id,
                "window_index": int(record.row.get("window_index", index)),
                "rolling_mean_mining_probability": rolling_mean,
                "rolling_max_mining_probability": rolling_max,
                "suspicious": suspicious,
            })
        first_record = group[0][0]
        rows.append({
            "workload": group_id,
            "label": first_record.label,
            "label_id": first_record.label_id,
            "pred_id": 1 if triggered else 0,
            "pred_label": id2label[1 if triggered else 0],
            "window_count": len(group),
            "rolling_mean_mining_probability_max": largest_mean,
            "rolling_max_mining_probability_max": largest_max,
            "suspicious": triggered,
        })

    return _classification_report(rows, id2label) | {
        "policy": {
            "history_windows": history_windows,
            "mean_mining_probability": mean_threshold,
            "max_mining_probability": max_threshold,
        },
        "predictions": rows,
        "window_predictions": window_rows,
    }


def _classification_report(rows: list[dict[str, Any]], id2label: dict[int, str]) -> dict[str, Any]:
    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required to compute metrics") from exc
    truth = [int(row["label_id"]) for row in rows]
    predicted = [int(row["pred_id"]) for row in rows]
    labels = sorted(id2label)
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(truth, predicted, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        "classification_report": classification_report(
            truth,
            predicted,
            labels=labels,
            target_names=[id2label[label] for label in labels],
            output_dict=True,
            zero_division=0,
        ),
    }


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
