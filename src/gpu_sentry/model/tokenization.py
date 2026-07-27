"""Tokenizer training and validation for normalized SASS."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import WorkloadRecord, iter_sass_texts, read_sass
def train_wordlevel_tokenizer(
    records: list[WorkloadRecord],
    tokenizer_config: dict[str, Any],
    max_seq_length: int,
    output_dir: Path,
):
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Sequence, Split, WhitespaceSplit
        from tokenizers.trainers import WordLevelTrainer
    except ImportError as exc:
        raise RuntimeError("tokenizers is required to train the SASS tokenizer") from exc

    specials = special_tokens(tokenizer_config)
    raw_tokenizer = Tokenizer(WordLevel(unk_token=tokenizer_config["unk_token"]))
    raw_tokenizer.pre_tokenizer = Sequence([WhitespaceSplit(), Split(",", behavior="isolated")])
    trainer = WordLevelTrainer(
        special_tokens=specials,
        min_frequency=1,
        show_progress=True,
    )
    raw_tokenizer.train_from_iterator(iter_sass_texts(records), trainer=trainer)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_tokenizer.save(str(output_dir / "tokenizer.json"))
    _write_hf_tokenizer_metadata(output_dir, tokenizer_config, max_seq_length)
    return RawTokenizerAdapter(raw_tokenizer, tokenizer_config)


def load_sass_tokenizer(tokenizer_dir: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to load the SASS tokenizer") from exc
    tokenizer_file = tokenizer_dir / "tokenizer.json"
    if not tokenizer_file.exists():
        raise FileNotFoundError(
            f"missing tokenizer at {tokenizer_file}; run `python scripts/train.py` first"
        )
    return AutoTokenizer.from_pretrained(str(tokenizer_dir))


def special_tokens(tokenizer_config: dict[str, Any]) -> list[str]:
    return [
        str(tokenizer_config["pad_token"]),
        str(tokenizer_config["unk_token"]),
        str(tokenizer_config["cls_token"]),
        str(tokenizer_config["sep_token"]),
        str(tokenizer_config["mask_token"]),
    ]


@dataclass
class RawTokenizerAdapter:
    tokenizer: Any
    tokenizer_config: dict[str, Any]

    @property
    def pad_token_id(self) -> int:
        return self._token_id("pad_token")

    @property
    def unk_token_id(self) -> int:
        return self._token_id("unk_token")

    @property
    def cls_token_id(self) -> int:
        return self._token_id("cls_token")

    @property
    def sep_token_id(self) -> int:
        return self._token_id("sep_token")

    @property
    def mask_token_id(self) -> int:
        return self._token_id("mask_token")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def __len__(self) -> int:
        return self.tokenizer.get_vocab_size()

    def _token_id(self, key: str) -> int:
        token_id = self.tokenizer.token_to_id(str(self.tokenizer_config[key]))
        if token_id is None:
            raise ValueError(f"missing tokenizer special token: {self.tokenizer_config[key]}")
        return int(token_id)


def _write_hf_tokenizer_metadata(
    output_dir: Path,
    tokenizer_config: dict[str, Any],
    max_seq_length: int,
) -> None:
    metadata = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": max_seq_length,
        "clean_up_tokenization_spaces": False,
        "pad_token": tokenizer_config["pad_token"],
        "unk_token": tokenizer_config["unk_token"],
        "cls_token": tokenizer_config["cls_token"],
        "sep_token": tokenizer_config["sep_token"],
        "mask_token": tokenizer_config["mask_token"],
    }
    special_map = {
        "pad_token": tokenizer_config["pad_token"],
        "unk_token": tokenizer_config["unk_token"],
        "cls_token": tokenizer_config["cls_token"],
        "sep_token": tokenizer_config["sep_token"],
        "mask_token": tokenizer_config["mask_token"],
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(special_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def unk_rate(tokenizer: Any, records: list[WorkloadRecord]) -> dict[str, Any]:
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is None:
        raise ValueError("tokenizer has no unk_token_id")
    total = 0
    unknown = 0
    unknown_tokens: Counter[str] = Counter()
    for record in records:
        if record.row.get("no_l0_window"):
            continue
        text = read_sass(record)
        ids = tokenizer.encode(text, add_special_tokens=False)
        total += len(ids)
        record_unknown = sum(1 for token_id in ids if token_id == unk_id)
        unknown += record_unknown
        if record_unknown:
            # This path is only for diagnostics, so a simple whitespace view is enough.
            tokens = text.replace(",", " , ").split()
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            for token, token_id in zip(tokens, token_ids):
                if token_id == unk_id:
                    unknown_tokens[token] += 1
    return {
        "tokens": total,
        "unknown_tokens": unknown,
        "unknown_rate": unknown / total if total else 0.0,
        "top_unknowns": unknown_tokens.most_common(25),
    }


def tokenizer_report(tokenizer: Any, records_by_split: dict[str, list[WorkloadRecord]]) -> dict[str, Any]:
    return {
        "vocab_size": len(tokenizer),
        "special_token_ids": {
            "pad_token_id": tokenizer.pad_token_id,
            "unk_token_id": tokenizer.unk_token_id,
            "cls_token_id": tokenizer.cls_token_id,
            "sep_token_id": tokenizer.sep_token_id,
            "mask_token_id": tokenizer.mask_token_id,
        },
        "unk_rates": {
            split: unk_rate(tokenizer, records)
            for split, records in records_by_split.items()
        },
    }
