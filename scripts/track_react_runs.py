#!/usr/bin/env python3
"""Append ReAct-run stats to the run-tracking CSV (one row per session-round).

Reads exported session traces (`*_data.json` from data/experiments/...) and
upserts rows keyed by (session_id, round), so re-running on the same session
updates its rows instead of duplicating them. Paper metrics such as referring-
expression length and lexical overlap come from `scripts/analyze_metrics.py`;
this script adds ReAct step counts and token usage alongside accuracy, turns,
words, and the compression metrics from `analyze_metrics.analyze_trace` (free
deterministic extractor), so they match the paper's definitions.

Usage:
    python scripts/track_react_runs.py data/experiments/ACL/2026-10-08/react-smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent))
from analyze_metrics import (  # noqa: E402
    ExtractionConfig,
    analyze_trace,
    find_trace_files,
    get_content_word_count,
    load_json_trace,
)

DEFAULT_CSV = Path("docs/react_runs.csv")

FIELDS = [
    "session_id",
    "session_prefix",
    "round",
    "date",
    "prompt_strategy",
    "director_model",
    "matcher_model",
    "reasoning_effort",
    "cross_round_history",
    "react",
    "react_roles",
    "react_step_model",
    "react_max_steps",
    "accuracy",
    "turns",
    "words",
    "director_words",
    "re_words",
    "mean_re_length",
    "relative_lexical_overlap",
    "jaccard_lexical_overlap",
    "extracted_objects",
    "matcher_turns",
    "react_turns",
    "react_steps",
    "react_steps_per_react_turn",
    "zoom_steps",
    "search_steps",
    "finish_steps",
    "unparseable_steps",
    "step_input_tokens",
    "step_cached_input_tokens",
    "step_output_tokens",
    "final_input_tokens",
    "final_cached_input_tokens",
    "final_output_tokens",
    "total_tokens",
    "calls",
    "calls_missing_usage",
]


def _tok(usage: dict[str, Any] | None, key: str) -> int:
    return int((usage or {}).get(key) or 0)


def round_row(record: dict[str, Any]) -> dict[str, Any]:
    config = record.get("config") or {}
    status = record.get("status") or {}
    messages = record.get("ai_messages") or []
    log = record.get("ai_reasoning_log") or []

    actions: Counter[str] = Counter()
    sums: Counter[str] = Counter()
    react_turns = 0
    calls = 0
    missing_usage = 0
    for entry in log:
        if not isinstance(entry, dict):
            continue
        calls += 1
        final_usage = entry.get("usage")
        if final_usage is None:
            missing_usage += 1
        for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
            sums[f"final_{key}"] += _tok(final_usage, key)
        sums["total_tokens"] += _tok(final_usage, "total_tokens")

        trace = entry.get("react_trace") or []
        if trace:
            react_turns += 1
        for step in trace:
            calls += 1
            actions[step.get("action") or "unparseable"] += 1
            if step.get("usage") is None:
                missing_usage += 1
            for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
                sums[f"step_{key}"] += _tok(step.get("usage"), key)
            sums["total_tokens"] += _tok(step.get("usage"), "total_tokens")

    react_steps = sum(actions.values())
    roles = config.get("react_roles") or []
    return {
        "session_id": record.get("session_id"),
        "session_prefix": config.get("session_prefix"),
        "round": record.get("round_number"),
        "date": (record.get("updated_at") or "")[:10],
        "prompt_strategy": config.get("prompt_strategy"),
        "director_model": config.get("ai_director_model"),
        "matcher_model": config.get("ai_matcher_model"),
        "reasoning_effort": config.get("ai_reasoning_effort"),
        "cross_round_history": config.get("cross_round_history"),
        "react": bool(config.get("react", False)),
        "react_roles": ";".join(roles) if config.get("react") else "",
        "react_step_model": (config.get("react_step_model") or "") if config.get("react") else "",
        "react_max_steps": config.get("react_max_steps") if config.get("react") else "",
        "accuracy": float(status.get("accuracy") or 0.0),
        "turns": int(status.get("turn_count", len(messages)) or 0),
        "words": sum(get_content_word_count(m.get("text", "")) for m in messages),
        "director_words": sum(
            get_content_word_count(m.get("text", ""))
            for m in messages
            if m.get("sender_role") == "director"
        ),
        "matcher_turns": sum(1 for e in log if isinstance(e, dict) and e.get("ai_role") == "matcher"),
        "react_turns": react_turns,
        "react_steps": react_steps,
        "react_steps_per_react_turn": round(react_steps / react_turns, 2) if react_turns else 0,
        "zoom_steps": actions["zoom"],
        "search_steps": actions["search_history"],
        "finish_steps": actions["finish"],
        "unparseable_steps": actions["unparseable"],
        **{key: sums[key] for key in FIELDS if key.startswith(("step_", "final_"))},
        "total_tokens": sums["total_tokens"],
        "calls": calls,
        "calls_missing_usage": missing_usage,
    }


COMPRESSION_FIELDS = (
    "re_words",
    "mean_re_length",
    "relative_lexical_overlap",
    "jaccard_lexical_overlap",
    "extracted_objects",
)


def compression_by_round(path: Path) -> dict[int, dict[str, Any]]:
    """Paper compression metrics per round, via the deterministic extractor."""
    config = ExtractionConfig(
        extractor="deterministic", gpt_model="", cache_dir=Path("."), refresh_cache=False
    )
    rows, _ = analyze_trace(path, [path.parent], config)
    return {int(r["round"]): {k: r[k] for k in COMPRESSION_FIELDS} for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="Export JSON files or folders.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    if args.csv.exists():
        with args.csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[(row["session_id"], str(row["round"]))] = row

    added = 0
    for path in find_trace_files(args.paths):
        compression = compression_by_round(path)
        for record in load_json_trace(path):
            if not isinstance(record, dict) or record.get("round_number") is None:
                continue
            row = round_row(record)
            row.update(compression.get(int(record["round_number"]), {}))
            rows[(str(row["session_id"]), str(row["round"]))] = row
            added += 1

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda r: (str(r["session_id"]), int(r["round"])))
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    print(f"Upserted {added} session-round rows into {args.csv} ({len(ordered)} total).")


if __name__ == "__main__":
    main()
