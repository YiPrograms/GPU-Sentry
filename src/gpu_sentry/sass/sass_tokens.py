"""Token-budget helpers for normalized SASS text."""

from __future__ import annotations

from collections.abc import Iterable


MODEL_MAX_SEQ_LENGTH = 8192
SPECIAL_TOKEN_RESERVE = 2
CONTENT_TOKEN_BUDGET = MODEL_MAX_SEQ_LENGTH - SPECIAL_TOKEN_RESERVE
KERNEL_BOUNDARY = "KERNEL_BOUNDARY"
KERNEL_BOUNDARY_TOKEN_COST = 1
KERNEL_BODY_TOKEN_BUDGET = CONTENT_TOKEN_BUDGET - KERNEL_BOUNDARY_TOKEN_COST


def sass_token_count(lines_or_text: Iterable[str] | str) -> int:
    """Count tokens with the same whitespace/comma splitting used by the SASS tokenizer."""
    if isinstance(lines_or_text, str):
        text = lines_or_text
    else:
        text = "\n".join(str(line) for line in lines_or_text)
    return len(text.replace(",", " , ").split())


def truncate_lines_to_token_budget(lines: Iterable[str], token_budget: int) -> list[str]:
    """Keep whole normalized-SASS instructions while staying within a token budget."""
    if token_budget <= 0:
        return []
    selected: list[str] = []
    cost = 0
    for line in lines:
        stripped = str(line).strip()
        if not stripped:
            continue
        line_cost = sass_token_count(stripped)
        if selected and cost + line_cost > token_budget:
            break
        if not selected and line_cost > token_budget:
            break
        selected.append(stripped)
        cost += line_cost
    return selected


def truncate_lines_from_front_to_token_budget(lines: Iterable[str], token_budget: int) -> list[str]:
    """Keep the newest suffix of normalized-SASS lines while staying within a token budget."""
    if token_budget <= 0:
        return []
    selected_reversed: list[str] = []
    cost = 0
    for line in reversed([str(item).strip() for item in lines]):
        if not line:
            continue
        line_cost = sass_token_count(line)
        if selected_reversed and cost + line_cost > token_budget:
            break
        if not selected_reversed and line_cost > token_budget:
            break
        selected_reversed.append(line)
        cost += line_cost
    return list(reversed(selected_reversed))
