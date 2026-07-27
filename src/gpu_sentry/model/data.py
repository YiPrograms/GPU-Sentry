"""Dataset loading and chunking utilities for SASS ModernBERT training."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol


SPLIT_NAMES = ("train", "val", "test")


class TokenizerLike(Protocol):
    cls_token_id: int
    sep_token_id: int

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ...


@dataclass(frozen=True)
class WorkloadRecord:
    split: str
    workload: str
    source_path: Path
    label: str
    label_id: int
    row: dict[str, Any]


@dataclass(frozen=True)
class ChunkRecord:
    workload: str
    split: str
    source_path: str
    label: str
    label_id: int
    chunk_index: int
    num_chunks: int
    input_ids: list[int]
    attention_mask: list[int]


def load_split_records(
    splits_dir: Path,
    split: str,
    repo_root: Path,
    label_column: str,
    label2id: dict[str, int],
) -> list[WorkloadRecord]:
    split_path = splits_dir / f"{split}.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(f"missing split file: {split_path}")

    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(split_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [field for field in ("workload", "path", label_column) if field not in row]
        if missing:
            raise ValueError(f"{split_path}:{line_no} missing fields: {', '.join(missing)}")
        label = str(row[label_column])
        if label not in label2id:
            raise ValueError(f"{split_path}:{line_no} unknown label {label!r}")
        rows.append(row)
    return records_from_rows(rows, split, repo_root, label_column, label2id, source_name=str(split_path))


def records_from_rows(
    rows: list[dict[str, Any]],
    split: str,
    repo_root: Path,
    label_column: str,
    label2id: dict[str, int],
    *,
    source_name: str = "<rows>",
) -> list[WorkloadRecord]:
    records: list[WorkloadRecord] = []
    source_path_exists: dict[Path, bool] = {}
    for line_no, row in enumerate(rows, start=1):
        missing = [field for field in ("workload", "path", label_column) if field not in row]
        if missing:
            raise ValueError(f"{source_name}:{line_no} missing fields: {', '.join(missing)}")
        label = str(row[label_column])
        if label not in label2id:
            raise ValueError(f"{source_name}:{line_no} unknown label {label!r}")
        source_path = resolve_source_path(repo_root, row["path"])
        exists = source_path_exists.get(source_path)
        if exists is None:
            exists = source_path.exists()
            source_path_exists[source_path] = exists
        if not exists:
            raise FileNotFoundError(f"{source_name}:{line_no} missing SASS path: {source_path}")
        records.append(
            WorkloadRecord(
                split=split,
                workload=str(row["workload"]),
                source_path=source_path,
                label=label,
                label_id=label2id[label],
                row=row,
            )
        )
    return records


def load_all_splits(
    splits_dir: Path,
    repo_root: Path,
    label_column: str,
    label2id: dict[str, int],
) -> dict[str, list[WorkloadRecord]]:
    return {
        split: load_split_records(splits_dir, split, repo_root, label_column, label2id)
        for split in SPLIT_NAMES
    }


def resolve_source_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def read_sass(record: WorkloadRecord) -> str:
    return record.source_path.read_text(encoding="utf-8")


def iter_sass_texts(records: Iterable[WorkloadRecord]) -> Iterator[str]:
    for record in records:
        if record.row.get("no_l0_window"):
            continue
        yield read_sass(record)


def make_chunks_for_record(
    record: WorkloadRecord,
    tokenizer: TokenizerLike,
    max_seq_length: int,
    stride: int,
) -> list[ChunkRecord]:
    if record.row.get("no_l0_window"):
        return []
    content_window = max_seq_length - 2
    if content_window <= 0:
        raise ValueError("max_seq_length must leave room for [CLS] and [SEP]")
    if stride >= content_window:
        raise ValueError("stride must be smaller than max_seq_length - 2")

    content_ids = encode_content_ids(tokenizer, read_sass(record))
    windows = chunk_token_ids(content_ids, content_window=content_window, stride=stride)
    chunks: list[ChunkRecord] = []
    for idx, window in enumerate(windows):
        input_ids = [int(tokenizer.cls_token_id), *window, int(tokenizer.sep_token_id)]
        chunks.append(
            ChunkRecord(
                workload=record.workload,
                split=record.split,
                source_path=str(record.source_path),
                label=record.label,
                label_id=record.label_id,
                chunk_index=idx,
                num_chunks=len(windows),
                input_ids=input_ids,
                attention_mask=[1] * len(input_ids),
            )
        )
    return chunks


def make_chunks(
    records: Iterable[WorkloadRecord],
    tokenizer: TokenizerLike,
    max_seq_length: int,
    stride: int,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for record in records:
        chunks.extend(make_chunks_for_record(record, tokenizer, max_seq_length, stride))
    return chunks


def chunk_token_ids(content_ids: list[int], content_window: int, stride: int) -> list[list[int]]:
    if content_window <= 0:
        raise ValueError("content_window must be positive")
    if stride >= content_window:
        raise ValueError("stride must be smaller than content_window")
    if not content_ids:
        return [[]]

    step = content_window - stride
    windows: list[list[int]] = []
    start = 0
    while start < len(content_ids):
        windows.append(content_ids[start : start + content_window])
        if start + content_window >= len(content_ids):
            break
        start += step
    return windows


def encode_content_ids(tokenizer: TokenizerLike, text: str) -> list[int]:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        return list(backend.encode(text, add_special_tokens=False).ids)
    return tokenizer.encode(text, add_special_tokens=False)


def class_weights(records: Iterable[WorkloadRecord], label2id: dict[str, int]) -> list[float]:
    group_labels: dict[str, int] = {}
    for record in records:
        if record.row.get("no_l0_window"):
            continue
        group_id = workload_group_id(record)
        existing = group_labels.get(group_id)
        if existing is not None and existing != record.label_id:
            raise ValueError(f"conflicting labels for workload group {group_id!r}")
        group_labels[group_id] = record.label_id

    counts = Counter(group_labels.values())
    total = sum(counts.values())
    num_labels = len(label2id)
    weights: list[float] = []
    for label_id in range(num_labels):
        count = counts.get(label_id, 0)
        if count == 0:
            raise ValueError(f"cannot compute class weight for missing label id {label_id}")
        weights.append(total / (num_labels * count))
    return weights


def workload_group_id(record: WorkloadRecord) -> str:
    return str(record.row.get("group_id") or record.row.get("parent_workload") or record.workload)


class ChunkDataset:
    """Small torch-compatible dataset over already chunked SASS inputs."""

    def __init__(self, chunks: list[ChunkRecord], include_labels: bool = True):
        self.chunks = chunks
        self.include_labels = include_labels

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        chunk = self.chunks[idx]
        item: dict[str, Any] = {
            "input_ids": chunk.input_ids,
            "attention_mask": chunk.attention_mask,
        }
        if self.include_labels:
            item["labels"] = chunk.label_id
        return item


def split_counts(records_by_split: dict[str, list[WorkloadRecord]]) -> dict[str, int]:
    return {split: len(records) for split, records in records_by_split.items()}
