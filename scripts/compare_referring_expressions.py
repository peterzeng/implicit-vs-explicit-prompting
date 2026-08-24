#!/usr/bin/env python3
"""Compare two referring-expression CSV exports from analyze_metrics.py."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


WORD_RE = re.compile(r"\b\w+\b")
KEY_FIELDS = [
    "experiment",
    "session_prefix",
    "prompt_strategy",
    "director_model",
    "matcher_model",
    "session_id",
    "round",
    "object_num",
    "basket_id",
]


def words(text: str) -> list[str]:
    return WORD_RE.findall((text or "").lower())


def multiset_jaccard(left: str, right: str) -> float:
    left_words = words(left)
    right_words = words(right)
    if not left_words and not right_words:
        return 1.0
    if not left_words or not right_words:
        return 0.0

    left_counts = Counter(left_words)
    right_counts = Counter(right_words)
    intersection = sum(
        min(left_counts.get(w, 0), right_counts.get(w, 0))
        for w in left_counts.keys() | right_counts.keys()
    )
    union = len(left_words) + len(right_words) - intersection
    return intersection / union if union else 0.0


def read_rows(path: Path) -> dict[tuple[str, ...], dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {tuple(row[field] for field in KEY_FIELDS): row for row in rows}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("left_csv", type=Path)
    parser.add_argument("right_csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/metrics/re_extraction_comparison.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left = read_rows(args.left_csv)
    right = read_rows(args.right_csv)
    keys = sorted(left.keys() | right.keys())

    rows: list[dict[str, Any]] = []
    for key in keys:
        left_row = left.get(key, {})
        right_row = right.get(key, {})
        left_re = left_row.get("referring_expression", "")
        right_re = right_row.get("referring_expression", "")
        left_words = len(words(left_re))
        right_words = len(words(right_re))
        out = dict(zip(KEY_FIELDS, key))
        out.update(
            {
                "left_extractor": left_row.get("extractor", ""),
                "left_gpt_model": left_row.get("gpt_model", ""),
                "right_extractor": right_row.get("extractor", ""),
                "right_gpt_model": right_row.get("gpt_model", ""),
                "left_re_words": left_words,
                "right_re_words": right_words,
                "word_count_delta_right_minus_left": right_words - left_words,
                "multiset_jaccard": multiset_jaccard(left_re, right_re),
                "left_referring_expression": left_re,
                "right_referring_expression": right_re,
            }
        )
        rows.append(out)

    write_csv(args.output, rows)
    scores = [float(row["multiset_jaccard"]) for row in rows]
    deltas = [int(row["word_count_delta_right_minus_left"]) for row in rows]
    print(f"Compared {len(rows)} object-level referring expressions.")
    print(f"Mean multiset Jaccard: {mean(scores):.3f}")
    print(f"Mean word-count delta (right-left): {mean(deltas):.1f}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
