"""Train and score the behavioral random-forest baseline."""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FEATURES = (
    "gpu_utilization_pct",
    "memory_utilization_pct",
    "power_usage_watts",
    "temperature_celsius",
)


def train_point_random_forest(
    input_csv: Path,
    output_dir: Path,
    seed: int = 1337,
    max_run_fpr: float = 0.01,
) -> dict[str, Any]:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    rows = [row for row in read_csv(input_csv) if row.get("binary_label") in {"benign", "mining"}]
    rows = [row for row in rows if all(_number(row.get(name)) for name in FEATURES)]
    if {row["binary_label"] for row in rows} != {"benign", "mining"}:
        raise ValueError("the baseline requires complete benign and mining samples")
    splits = grouped_split(rows, seed)
    model = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced")
    model.fit(matrix(splits["train"]), [row["binary_label"] for row in splits["train"]])

    point_reports: dict[str, Any] = {}
    point_predictions: list[dict[str, Any]] = []
    runs_by_split: dict[str, list[dict[str, Any]]] = {}
    for name, split_rows in splits.items():
        probabilities = mining_probabilities(model, matrix(split_rows))
        predictions = list(model.predict(matrix(split_rows)))
        truth = [row["binary_label"] for row in split_rows]
        point_reports[name] = report_metrics(truth, predictions, accuracy_score, classification_report, confusion_matrix)
        detailed = []
        for row, prediction, probability in zip(split_rows, predictions, probabilities):
            item = {
                **row,
                "point_prediction": prediction,
                "mining_probability": probability,
                "split": name,
            }
            detailed.append(item)
            point_predictions.append(item)
        runs_by_split[name] = aggregate_runs(detailed)

    threshold = tune_threshold(runs_by_split["val"], max_run_fpr)
    run_reports: dict[str, Any] = {}
    run_predictions: list[dict[str, Any]] = []
    for name, runs in runs_by_split.items():
        for row in runs:
            row["split"] = name
            row["threshold"] = threshold
            row["run_prediction"] = "mining" if row["mean_mining_probability"] >= threshold else "benign"
        run_reports[name] = report_metrics(
            [row["binary_label"] for row in runs],
            [row["run_prediction"] for row in runs],
            accuracy_score,
            classification_report,
            confusion_matrix,
        )
        run_predictions.extend(runs)

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "features": list(FEATURES),
        "seed": seed,
        "max_validation_fpr": max_run_fpr,
        "mean_probability_threshold": threshold,
        "point": point_reports,
        "run": run_reports,
    }
    write_csv(output_dir / "point_predictions.csv", point_predictions)
    write_csv(output_dir / "run_predictions.csv", run_predictions)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    joblib.dump({"model": model, "feature_columns": list(FEATURES), "threshold": threshold}, output_dir / "model.joblib")
    return report


def score_points(input_csv: Path, model_path: Path, output_dir: Path, truth: str = "mining") -> dict[str, Any]:
    import joblib
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    bundle = joblib.load(model_path)
    model = bundle["model"]
    threshold = float(bundle["threshold"])
    rows = [row for row in read_csv(input_csv) if all(_number(row.get(name)) for name in FEATURES)]
    probabilities = mining_probabilities(model, matrix(rows))
    predictions = list(model.predict(matrix(rows)))
    detailed = []
    for row, prediction, probability in zip(rows, predictions, probabilities):
        detailed.append({**row, "binary_label": truth, "point_prediction": prediction, "mining_probability": probability})
    runs = aggregate_runs(detailed)
    for row in runs:
        row["threshold"] = threshold
        row["run_prediction"] = "mining" if row["mean_mining_probability"] >= threshold else "benign"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "point_predictions.csv", detailed)
    write_csv(output_dir / "run_predictions.csv", runs)
    report = report_metrics(
        [truth] * len(runs),
        [row["run_prediction"] for row in runs],
        accuracy_score,
        classification_report,
        confusion_matrix,
    )
    report.update({"runs": len(runs), "points": len(rows), "threshold": threshold})
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def aggregate_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("run_id") or row.get("workload"))].append(row)
    result = []
    for run_id, samples in sorted(grouped.items()):
        probabilities = [float(row["mining_probability"]) for row in samples]
        label = Counter(row.get("binary_label") for row in samples).most_common(1)[0][0]
        first = samples[0]
        result.append({
            "run_id": run_id,
            "workload": first.get("workload"),
            "program": first.get("program"),
            "target_percent": first.get("target_percent"),
            "binary_label": label,
            "samples": len(samples),
            "mean_mining_probability": sum(probabilities) / len(probabilities),
            "max_mining_probability": max(probabilities),
        })
    return result


def tune_threshold(runs: list[dict[str, Any]], max_fpr: float) -> float:
    candidates = sorted({0.0, 1.0, *[float(row["mean_mining_probability"]) for row in runs]})
    best: tuple[float, float] | None = None
    for threshold in candidates:
        benign = [row for row in runs if row["binary_label"] == "benign"]
        mining = [row for row in runs if row["binary_label"] == "mining"]
        fpr = sum(row["mean_mining_probability"] >= threshold for row in benign) / max(1, len(benign))
        recall = sum(row["mean_mining_probability"] >= threshold for row in mining) / max(1, len(mining))
        if fpr <= max_fpr and (best is None or (recall, threshold) > best):
            best = (recall, threshold)
    return best[1] if best else 1.0


def grouped_split(rows: list[dict[str, str]], seed: int) -> dict[str, list[dict[str, str]]]:
    groups = sorted({_group(row) for row in rows})
    random.Random(seed).shuffle(groups)
    train_end = int(len(groups) * 0.70)
    val_end = train_end + int(len(groups) * 0.15)
    side = {group: "train" for group in groups[:train_end]}
    side.update({group: "val" for group in groups[train_end:val_end]})
    side.update({group: "test" for group in groups[val_end:]})
    result = {"train": [], "val": [], "test": []}
    for row in rows:
        result[side[_group(row)]].append(row)
    return result


def report_metrics(truth, prediction, accuracy_score, classification_report, confusion_matrix) -> dict[str, Any]:
    if not truth:
        return {"accuracy": None, "classification_report": {}, "confusion_matrix": []}
    return {
        "accuracy": accuracy_score(truth, prediction),
        "classification_report": classification_report(truth, prediction, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=["benign", "mining"]).tolist(),
    }


def mining_probabilities(model, rows: list[list[float]]) -> list[float]:
    index = list(model.classes_).index("mining")
    return [float(row[index]) for row in model.predict_proba(rows)]


def matrix(rows: list[dict[str, str]]) -> list[list[float]]:
    return [[float(row[name]) for name in FEATURES] for row in rows]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def _group(row: dict[str, str]) -> str:
    return re.sub(r"_(o2|o3)$", "", row.get("workload") or row.get("run_id") or "")


def _number(value: Any) -> bool:
    try:
        float(value)
        return value not in (None, "")
    except (TypeError, ValueError):
        return False
