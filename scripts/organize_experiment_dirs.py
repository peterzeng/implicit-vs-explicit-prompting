#!/usr/bin/env python3
"""Normalize experiment session folders into one directory per setting."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SETTING_DIRS = {
    ("ACL", "gpt-5.2"): "gpt5.2-acl",
    ("ACL", "gpt-5.5"): "gpt5.5-acl",
    ("Cameron", "gpt-5.2"): "gpt5.2-cameron",
    ("Cameron", "gpt-5.5"): "gpt5.5-cameron",
}


def load_config(trace_file: Path) -> dict:
    with trace_file.open(encoding="utf-8") as f:
        data = json.load(f)
    return data[0].get("config", {}) if data else {}


def experiment_name(trace_file: Path) -> str:
    parts = trace_file.parts
    idx = parts.index("experiments")
    return parts[idx + 1]


def date_dir(trace_file: Path) -> Path:
    parts = trace_file.parts
    idx = parts.index("experiments")
    return Path(*parts[: idx + 3])


def session_dir_for_trace(trace_file: Path) -> Path:
    return trace_file.parent


def unique_destination(dest: Path) -> Path:
    if not dest.exists():
        return dest
    suffix = 2
    while True:
        candidate = dest.with_name(f"{dest.name}-{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def prune_empty_dirs(root: Path, stop_at: Path, dry_run: bool) -> None:
    current = root
    while current != stop_at and current.is_dir():
        try:
            next(current.iterdir())
            break
        except StopIteration:
            print(f"rmdir {current}")
            if not dry_run:
                current.rmdir()
            current = current.parent


def organize(root: Path, dry_run: bool) -> None:
    trace_files = sorted(root.rglob("*_data.json"))
    moves: list[tuple[Path, Path]] = []
    for trace_file in trace_files:
        exp = experiment_name(trace_file)
        cfg = load_config(trace_file)
        model = cfg.get("ai_director_model")
        setting_dir_name = SETTING_DIRS.get((exp, model))
        if not setting_dir_name:
            print(f"skip unknown setting: {trace_file}")
            continue

        day_dir = date_dir(trace_file)
        dest_parent = day_dir / setting_dir_name
        session_dir = session_dir_for_trace(trace_file)
        desired_dest = dest_parent / session_dir.name
        if session_dir == desired_dest:
            continue

        moves.append((session_dir, unique_destination(desired_dest)))

    seen_sources: set[Path] = set()
    old_parents: set[Path] = set()
    for source, dest in moves:
        if source in seen_sources:
            continue
        seen_sources.add(source)
        old_parents.add(source.parent)
        print(f"move {source} -> {dest}")
        if dry_run:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest))

    for old_parent in sorted(old_parents, reverse=True):
        day_dir = old_parent.parent
        if old_parent.exists() and old_parent.is_dir():
            prune_empty_dirs(old_parent, day_dir, dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, default=Path("data/experiments"), nargs="?")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    organize(args.root, args.dry_run)


if __name__ == "__main__":
    main()
