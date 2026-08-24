#!/usr/bin/env python3
"""Compute referential-game metrics for one trace or whole experiment folders.

The experiment logs already contain enough structure to compute the core
metrics without an LLM extraction pass. Director utterances consistently mark
objects as "Basket N ..." or "basket N ..."; this script extracts those spans,
tracks object identity through the stable image path in `shared_grid`, and
summarizes runs separately by prompt and model setting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


BASKET_MARKER_RE = re.compile(
    r"(?i)(?:\bbasket\s*(\d{1,2})\b|\b(\d{1,2})(?=\s*:))"
    r"(?:\s*(?:is|=|:|-|--|—)\s*)?"
)
WORD_RE = re.compile(r"\b\w+\b")


@dataclass(frozen=True)
class Setting:
    experiment: str
    session_prefix: str
    prompt_strategy: str
    director_model: str
    matcher_model: str


@dataclass(frozen=True)
class ExtractionConfig:
    extractor: str
    gpt_model: str
    cache_dir: Path
    refresh_cache: bool


def content_words(text: str) -> list[str]:
    return WORD_RE.findall((text or "").lower())


def get_content_word_count(text: str) -> int:
    return len(content_words(text))


def multiset_intersection_size(words_a: list[str], words_b: list[str]) -> int:
    counts_a = Counter(words_a)
    counts_b = Counter(words_b)
    return sum(
        min(counts_a.get(w, 0), counts_b.get(w, 0))
        for w in counts_a.keys() | counts_b.keys()
    )


def calculate_relative_lexical_overlap(re_i: str, re_prev: str) -> float:
    """Paper Section 4.5 RLO: multiset overlap divided by current RE length."""

    words_i = content_words(re_i)
    words_prev = content_words(re_prev)
    if not words_i:
        return 0.0

    intersection_size = multiset_intersection_size(words_prev, words_i)
    return intersection_size / len(words_i)


def calculate_jaccard_overlap(re_i: str, re_prev: str) -> float:
    """Alternative metric from Appendix C.2, kept for diagnostics."""

    words_i = content_words(re_i)
    words_prev = content_words(re_prev)
    if not words_i or not words_prev:
        return 0.0

    intersection_size = multiset_intersection_size(words_i, words_prev)
    union_size = len(words_i) + len(words_prev) - intersection_size
    return intersection_size / union_size if union_size else 0.0


def load_json_trace(path: Path) -> list[dict[str, Any]]:
    content = path.read_text(encoding="utf-8")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Some live traces can be interrupted after a complete array plus junk.
        if "]" not in content:
            raise
        return json.loads(content[: content.rindex("]") + 1])


def normalize_phrase(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[\s:;,.!?—-]+", "", text)
    return text.strip()


def extract_referring_expressions(messages: list[dict[str, Any]]) -> dict[int, str]:
    """Extract director descriptions keyed by positional basket number.

    A director message can contain one object ("Basket 1: ...") or several
    objects in one clarification ("Basket 11 ... Basket 12 ..."). We split on
    each marker and append repeated mentions for the same basket.
    """

    expressions: dict[int, list[str]] = {}
    next_unmarked_basket = 1
    for message in messages:
        if message.get("sender_role") != "director":
            continue

        text = message.get("text", "")
        matches = list(BASKET_MARKER_RE.finditer(text))
        if not matches:
            phrase = normalize_phrase(text)
            if phrase and next_unmarked_basket <= 12:
                expressions.setdefault(next_unmarked_basket, []).append(phrase)
                next_unmarked_basket += 1
            continue

        for idx, match in enumerate(matches):
            basket_num = int(match.group(1) or match.group(2))
            if basket_num < 1 or basket_num > 12:
                continue
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            phrase = normalize_phrase(text[start:end])
            if phrase:
                expressions.setdefault(basket_num, []).append(phrase)
            next_unmarked_basket = max(next_unmarked_basket, basket_num + 1)

    return {basket_num: " ".join(parts) for basket_num, parts in expressions.items()}


def transcript_text(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = "describer" if message.get("sender_role") == "director" else "matcher"
        lines.append(f"{role}: {message.get('text', '')}")
    return "\n".join(lines)


def extraction_prompt(transcript: str, num_objects: int = 12) -> str:
    keys = "\n".join(
        f'    "object_#{i}": "descriptive phrases for object {i}",'
        for i in range(1, num_objects + 1)
    ).rstrip(",")
    return f"""This is an extractive task.

You will be given a transcript of a conversation between two participants engaged in a collaborative object-matching task. There are exactly {num_objects} target objects. One participant (the describer) describes each target object, and the other participant (the matcher) attempts to identify them.

Your task is to extract the descriptive phrases used by the describer for each target object.
- Extract phrases verbatim from the transcript.
- Do not extract the whole utterance, only the descriptive phrases.
- Exclude disfluencies, fillers, and false starts (e.g., "um", "uh", "like").
- Do not paraphrase or infer missing information.
- Each object may have one or multiple descriptive phrases.

Return the results in the following JSON format:

{{
{keys}
}}

Example description phrases:

- doesn't have handle, tip of it is thicker than rest of body, brownish color, weaves are in squares if you look at it directly
- half circle, no handles, top tip of it is a little bit thicker than rest of body
- tip which is a little bit thicker than rest of body
- tip that is a little bit larger than body, looks a little bit thicker

Transcript:

{transcript}

Output only the JSON object. Do not include any additional text or explanations."""


def gpt_cache_path(
    cache_dir: Path, trace_file: Path, session_id: str, round_num: int, model: str
) -> Path:
    key = f"{trace_file.resolve()}::{session_id}::{round_num}::{model}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    safe_session = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id)
    exact_path = cache_dir / model / f"{safe_session}_round_{round_num}_{digest}.json"
    if exact_path.exists():
        return exact_path

    # Keep caches usable when experiment folders are reorganized. Session ids are
    # unique for these runs, while the full path is not stable during cleanup.
    matches = sorted((cache_dir / model).glob(f"{safe_session}_round_{round_num}_*.json"))
    if matches:
        return matches[0]
    return exact_path


def parse_extraction_json(raw_text: str) -> dict[int, str]:
    parsed = json.loads(raw_text)
    expressions = {}
    for i in range(1, 13):
        value = parsed.get(f"object_#{i}", "")
        if isinstance(value, list):
            value = " ".join(str(item) for item in value)
        expressions[i] = normalize_phrase(str(value))
    return expressions


def extract_referring_expressions_with_gpt(
    messages: list[dict[str, Any]],
    trace_file: Path,
    session_id: str,
    round_num: int,
    config: ExtractionConfig,
) -> dict[int, str]:
    cache_path = gpt_cache_path(
        config.cache_dir, trace_file, session_id, round_num, config.gpt_model
    )
    if cache_path.exists() and not config.refresh_cache:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return parse_extraction_json(cached["response"])

    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "GPT extraction requires `openai` and `python-dotenv`; install requirements.txt."
        ) from exc

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("GPT extraction requires OPENAI_API_KEY.")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = extraction_prompt(transcript_text(messages))
    response = client.chat.completions.create(
        model=config.gpt_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    response_text = response.choices[0].message.content or "{}"
    expressions = parse_extraction_json(response_text)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "model": config.gpt_model,
                "trace_file": str(trace_file),
                "session_id": session_id,
                "round": round_num,
                "response": response_text,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return expressions


def extract_referring_expressions_for_round(
    messages: list[dict[str, Any]],
    trace_file: Path,
    session_id: str,
    round_num: int,
    config: ExtractionConfig,
) -> dict[int, str]:
    if config.extractor == "deterministic":
        return extract_referring_expressions(messages)
    if config.extractor == "gpt":
        return extract_referring_expressions_with_gpt(
            messages, trace_file, session_id, round_num, config
        )
    raise ValueError(f"Unknown extractor: {config.extractor}")


def target_map_for_round(round_data: dict[str, Any]) -> dict[int, str]:
    grid = round_data.get("shared_grid") or []
    grid_sorted = sorted(grid, key=lambda cell: (cell.get("row", 0), cell.get("col", 0)))
    return {
        index + 1: str(cell.get("image") or cell.get("basket_id", index + 1))
        for index, cell in enumerate(grid_sorted)
    }


def experiment_name(path: Path, input_roots: list[Path]) -> str:
    for root in input_roots:
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if relative.parts:
            if root.name == "experiments":
                return relative.parts[0]
            return root.name
    parts = path.parts
    if "experiments" in parts:
        idx = parts.index("experiments")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return path.parent.name


def analyze_trace(
    path: Path,
    input_roots: list[Path],
    extraction_config: ExtractionConfig,
    max_round: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = load_json_trace(path)
    if not data:
        return [], []

    first_config = data[0].get("config", {})
    setting = Setting(
        experiment=experiment_name(path, input_roots),
        session_prefix=first_config.get("session_prefix", path.parent.parent.name),
        prompt_strategy=first_config.get("prompt_strategy", "unknown"),
        director_model=first_config.get("ai_director_model", "unknown"),
        matcher_model=first_config.get("ai_matcher_model", "unknown"),
    )

    re_history: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    expression_rows: list[dict[str, Any]] = []

    for round_idx, round_data in enumerate(data):
        round_num = round_data.get("round_number", round_idx + 1)
        if max_round is not None and round_num > max_round:
            continue
        status = round_data.get("status", {})
        messages = status.get("messages", [])
        target_map = target_map_for_round(round_data)
        session_id = round_data.get("session_id", path.parent.name)
        expressions = extract_referring_expressions_for_round(
            messages,
            path,
            session_id,
            round_num,
            extraction_config,
        )

        total_re_len = 0
        total_rlo = 0.0
        total_jaccard = 0.0
        overlap_count = 0
        extracted_count = 0

        for object_num in range(1, 13):
            re_text = expressions.get(object_num, "")
            if re_text:
                extracted_count += 1
            basket_id = target_map.get(object_num, str(object_num))
            re_word_count = get_content_word_count(re_text)
            total_re_len += re_word_count

            if round_num > 1 and basket_id in re_history:
                previous_re_text = re_history[basket_id]
                object_rlo = calculate_relative_lexical_overlap(re_text, previous_re_text)
                object_jaccard = calculate_jaccard_overlap(re_text, previous_re_text)
                total_rlo += object_rlo
                total_jaccard += object_jaccard
                overlap_count += 1
            else:
                previous_re_text = ""
                object_rlo = 1.0
                object_jaccard = 0.0

            re_history[basket_id] = re_text
            expression_rows.append(
                {
                    "extractor": extraction_config.extractor,
                    "gpt_model": (
                        extraction_config.gpt_model
                        if extraction_config.extractor == "gpt"
                        else ""
                    ),
                    "experiment": setting.experiment,
                    "session_prefix": setting.session_prefix,
                    "prompt_strategy": setting.prompt_strategy,
                    "director_model": setting.director_model,
                    "matcher_model": setting.matcher_model,
                    "session_id": session_id,
                    "trace_file": str(path),
                    "round": round_num,
                    "object_num": object_num,
                    "basket_id": basket_id,
                    "referring_expression": re_text,
                    "previous_referring_expression": previous_re_text,
                    "re_words": re_word_count,
                    "relative_lexical_overlap": object_rlo,
                    "jaccard_lexical_overlap": object_jaccard,
                }
            )

        mean_rlo = total_rlo / overlap_count if overlap_count > 0 else 1.0
        mean_jaccard = total_jaccard / overlap_count if overlap_count > 0 else 0.0

        rows.append(
            {
                "extractor": extraction_config.extractor,
                "gpt_model": (
                    extraction_config.gpt_model if extraction_config.extractor == "gpt" else ""
                ),
                "experiment": setting.experiment,
                "session_prefix": setting.session_prefix,
                "prompt_strategy": setting.prompt_strategy,
                "director_model": setting.director_model,
                "matcher_model": setting.matcher_model,
                "session_id": session_id,
                "trace_file": str(path),
                "round": round_num,
                "accuracy": float(status.get("accuracy", 0.0) or 0.0),
                "turns": int(status.get("turn_count", len(messages)) or 0),
                "words": sum(get_content_word_count(m.get("text", "")) for m in messages),
                "re_words": total_re_len,
                "mean_re_length": total_re_len / 12,
                "relative_lexical_overlap": mean_rlo,
                "jaccard_lexical_overlap": mean_jaccard,
                "extracted_objects": extracted_count,
            }
        )

    return rows, expression_rows


def find_trace_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(path.rglob("*_data.json"))
    return sorted(set(files))


def summarize(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    metric_fields = [
        "accuracy",
        "turns",
        "words",
        "re_words",
        "mean_re_length",
        "relative_lexical_overlap",
        "jaccard_lexical_overlap",
        "extracted_objects",
    ]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[field] for field in group_fields), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        out = dict(zip(group_fields, key))
        out["n_rows"] = len(group)
        out["n_sessions"] = len({row["session_id"] for row in group})
        for metric in metric_fields:
            values = [float(row[metric]) for row in group]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(out)
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isclose(value, round(value)):
            return f"{value:.1f}"
        return f"{value:.3f}"
    return str(value)


def print_setting_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No trace rows found.")
        return

    summary = summarize(
        rows,
        [
            "extractor",
            "gpt_model",
            "experiment",
            "session_prefix",
            "prompt_strategy",
            "director_model",
            "matcher_model",
        ],
    )
    columns = [
        "extractor",
        "gpt_model",
        "experiment",
        "session_prefix",
        "director_model",
        "matcher_model",
        "n_sessions",
        "accuracy_mean",
        "turns_mean",
        "words_mean",
        "re_words_mean",
        "mean_re_length_mean",
        "relative_lexical_overlap_mean",
    ]
    widths = {col: max(len(col), *(len(fmt(row[col])) for row in summary)) for col in columns}
    print("=== SUMMARY BY PROMPT/MODEL SETTING ===")
    print("  ".join(col.ljust(widths[col]) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in summary:
        print("  ".join(fmt(row[col]).ljust(widths[col]) for col in columns))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Experiment folders or *_data.json traces to analyze.",
    )
    parser.add_argument(
        "--filename",
        type=Path,
        help="Backward-compatible alias for a single JSON trace.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/metrics"),
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--max-round",
        type=int,
        help="Only include rounds up to this number, e.g. 4 for paper-style tables.",
    )
    parser.add_argument(
        "--extractor",
        choices=["deterministic", "gpt"],
        default="deterministic",
        help="How to extract referring expressions.",
    )
    parser.add_argument(
        "--gpt-model",
        default="gpt-5.5",
        help="Model used when --extractor gpt.",
    )
    parser.add_argument(
        "--gpt-cache-dir",
        type=Path,
        default=None,
        help="Cache directory for GPT extraction responses.",
    )
    parser.add_argument(
        "--refresh-gpt-cache",
        action="store_true",
        help="Re-run GPT extraction even when a cached response exists.",
    )
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=[],
        help="Skip trace files whose path contains this substring. Repeatable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = list(args.paths)
    if args.filename:
        paths.append(args.filename)
    if not paths:
        paths = [Path("data/experiments/ACL"), Path("data/experiments/Cameron-style")]

    trace_files = find_trace_files(paths)
    if args.exclude_pattern:
        trace_files = [
            path
            for path in trace_files
            if not any(pattern in str(path) for pattern in args.exclude_pattern)
        ]
    extraction_config = ExtractionConfig(
        extractor=args.extractor,
        gpt_model=args.gpt_model,
        cache_dir=args.gpt_cache_dir or (args.output_dir / "gpt_re_cache"),
        refresh_cache=args.refresh_gpt_cache,
    )
    rows: list[dict[str, Any]] = []
    expression_rows: list[dict[str, Any]] = []
    for trace_file in trace_files:
        trace_rows, trace_expression_rows = analyze_trace(
            trace_file,
            paths,
            extraction_config=extraction_config,
            max_round=args.max_round,
        )
        rows.extend(trace_rows)
        expression_rows.extend(trace_expression_rows)

    round_summary = summarize(
        rows,
        [
            "extractor",
            "gpt_model",
            "experiment",
            "session_prefix",
            "prompt_strategy",
            "director_model",
            "matcher_model",
            "round",
        ],
    )
    setting_summary = summarize(
        rows,
        [
            "extractor",
            "gpt_model",
            "experiment",
            "session_prefix",
            "prompt_strategy",
            "director_model",
            "matcher_model",
        ],
    )
    collapsed_round_summary = summarize(
        rows,
        [
            "extractor",
            "gpt_model",
            "experiment",
            "prompt_strategy",
            "director_model",
            "matcher_model",
            "round",
        ],
    )
    collapsed_setting_summary = summarize(
        rows,
        [
            "extractor",
            "gpt_model",
            "experiment",
            "prompt_strategy",
            "director_model",
            "matcher_model",
        ],
    )

    write_csv(args.output_dir / "referring_expressions.csv", expression_rows)
    write_csv(args.output_dir / "metrics_by_session_round.csv", rows)
    write_csv(args.output_dir / "metrics_by_setting_round.csv", round_summary)
    write_csv(args.output_dir / "metrics_by_setting.csv", setting_summary)
    write_csv(
        args.output_dir / "metrics_by_setting_round_collapsed.csv",
        collapsed_round_summary,
    )
    write_csv(
        args.output_dir / "metrics_by_setting_collapsed.csv",
        collapsed_setting_summary,
    )

    print(f"Analyzed {len(trace_files)} trace files and {len(rows)} rounds.")
    print_setting_summary(rows)
    print(f"\nCSV outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
