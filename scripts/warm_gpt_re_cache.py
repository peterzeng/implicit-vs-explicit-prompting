#!/usr/bin/env python3
"""Fill GPT referring-expression cache files in parallel."""

from __future__ import annotations

import argparse
import concurrent.futures
import time
from pathlib import Path
from typing import Any

from analyze_metrics import (
    ExtractionConfig,
    extract_referring_expressions_with_gpt,
    find_trace_files,
    gpt_cache_path,
    load_json_trace,
)


def round_jobs(
    paths: list[Path],
    cache_dir: Path,
    model: str,
    max_round: int | None,
    exclude_patterns: list[str],
) -> list[tuple[Path, str, int, list[dict[str, Any]]]]:
    trace_files = find_trace_files(paths)
    if exclude_patterns:
        trace_files = [
            path
            for path in trace_files
            if not any(pattern in str(path) for pattern in exclude_patterns)
        ]

    jobs = []
    for trace_file in trace_files:
        data = load_json_trace(trace_file)
        for round_idx, round_data in enumerate(data):
            round_num = round_data.get("round_number", round_idx + 1)
            if max_round is not None and round_num > max_round:
                continue
            session_id = round_data.get("session_id", trace_file.parent.name)
            cache_path = gpt_cache_path(cache_dir, trace_file, session_id, round_num, model)
            if cache_path.exists():
                continue
            messages = round_data.get("status", {}).get("messages", [])
            jobs.append((trace_file, session_id, round_num, messages))
    return jobs


def run_job(
    job: tuple[Path, str, int, list[dict[str, Any]]],
    config: ExtractionConfig,
    retries: int,
) -> str:
    trace_file, session_id, round_num, messages = job
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            extract_referring_expressions_with_gpt(
                messages, trace_file, session_id, round_num, config
            )
            return f"ok {session_id} round {round_num}"
        except Exception as exc:  # noqa: BLE001 - report and retry API failures.
            last_error = exc
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    return f"failed {session_id} round {round_num}: {last_error}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--gpt-model", default="gpt-5.5")
    parser.add_argument("--max-round", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--exclude-pattern", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = round_jobs(
        args.paths,
        args.cache_dir,
        args.gpt_model,
        args.max_round,
        args.exclude_pattern,
    )
    print(f"Missing cache files: {len(jobs)}")
    if not jobs:
        return

    config = ExtractionConfig(
        extractor="gpt",
        gpt_model=args.gpt_model,
        cache_dir=args.cache_dir,
        refresh_cache=False,
    )
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_job, job, config, args.retries) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            result = future.result()
            print(f"[{completed}/{len(jobs)}] {result}", flush=True)


if __name__ == "__main__":
    main()
