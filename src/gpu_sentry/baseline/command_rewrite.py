from __future__ import annotations

from dataclasses import dataclass


ZERO_BASED_FLAGS = {
    "--devices",
    "--device",
    "--cuda-devices",
    "--gpu",
    "--gpu-id",
    "-d",
    "--gpus",
}

# Claymore's -gpus flag is traditionally a one-based GPU selector/bitmask in its
# benchmark mode, so with CUDA_VISIBLE_DEVICES pinned to one device, "1" selects
# the single visible adapter.
ONE_BASED_FLAGS = {"-gpus"}


@dataclass(frozen=True)
class RewriteResult:
    argv: list[str]
    changes: list[dict[str, str]]


def rewrite_gpu_args(argv: list[str]) -> RewriteResult:
    rewritten = list(argv)
    changes: list[dict[str, str]] = []
    i = 0
    while i < len(rewritten):
        token = rewritten[i]
        if token in ZERO_BASED_FLAGS and i + 1 < len(rewritten):
            changes.append({"flag": token, "old": rewritten[i + 1], "new": "0"})
            rewritten[i + 1] = "0"
            i += 2
            continue
        if token in ONE_BASED_FLAGS and i + 1 < len(rewritten):
            changes.append({"flag": token, "old": rewritten[i + 1], "new": "1"})
            rewritten[i + 1] = "1"
            i += 2
            continue
        split = split_equals(token)
        if split and split[0] in ZERO_BASED_FLAGS:
            flag, old = split
            new = f"{flag}=0"
            changes.append({"flag": flag, "old": old, "new": "0"})
            rewritten[i] = new
        elif split and split[0] in ONE_BASED_FLAGS:
            flag, old = split
            new = f"{flag}=1"
            changes.append({"flag": flag, "old": old, "new": "1"})
            rewritten[i] = new
        i += 1
    return RewriteResult(argv=rewritten, changes=changes)


def split_equals(token: str) -> tuple[str, str] | None:
    if "=" not in token:
        return None
    flag, value = token.split("=", 1)
    if not flag:
        return None
    return flag, value
