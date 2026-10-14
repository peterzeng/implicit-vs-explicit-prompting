#!/usr/bin/env python3
"""Re-run the paper's AI-AI experiment (implicit vs. explicit prompt) with a new model.

Uses the paper's settings: basket set 5, 5 rounds, cross-round history on,
reasoning effort none, the same model for director and matcher, and 10
sessions per prompt condition.

The script is resumable. It counts sessions that already have a complete
export under data/experiments/ and only runs the missing ones, so you can
re-run the same command after a crash, a sleep, or a stalled session.
Sessions run in parallel worker processes, each with its own SQLite file.

    python scripts/run_replication.py --model gpt-6-luna --dry-run
    nohup caffeinate -i python scripts/run_replication.py --model gpt-6-luna \\
        > logs/gpt6-luna.log 2>&1 &

When all sessions are complete it runs analyze_metrics.py (GPT-5.5 referring-
expression extraction, as in the paper) and plot_acl_style_metrics.py. Pass
--baseline-metrics to plot the paper's GPT-5.2/5.5 results alongside.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "data" / "experiments"
PAPER_MODELS = ["gpt-5.2", "gpt-5.5"]

# (prompt strategy, export family folder, session-prefix suffix)
CONDITIONS = {
    "ACL_prompt": ("ACL", "acl"),
    "cameron-prompt": ("Cameron", "cameron"),
}


def model_slug(model: str) -> str:
    """gpt-5.5 -> gpt5.5, gpt-6-luna -> gpt6-luna (matches the paper's prefixes)."""
    return model.replace("gpt-", "gpt", 1)


def session_prefix(model: str, prompt: str, tag: str | None) -> str:
    prefix = f"{model_slug(model)}-{CONDITIONS[prompt][1]}"
    return f"{prefix}-{tag}" if tag else prefix


def is_complete_trace(
    path: Path, model: str, prompt: str, effort: str, num_rounds: int, harness_version: int
) -> bool:
    try:
        rounds = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(rounds, list) or len(rounds) < num_rounds:
        return False
    config = rounds[0].get("config") or {}
    last_status = rounds[-1].get("status") or {}
    return (
        config.get("prompt_strategy") == prompt
        and config.get("ai_director_model") == model
        and config.get("ai_matcher_model") == model
        and (config.get("ai_reasoning_effort") or "none") == effort
        and config.get("harness_version") == harness_version
        and not config.get("react")
        and not config.get("stop_after_round")
        and isinstance(last_status, dict)
        and last_status.get("is_complete") is True
    )


def completed_traces(prefix: str, model: str, prompt: str, effort: str, num_rounds: int) -> list[Path]:
    """Complete sessions for this condition, run under the current game rules."""
    from referential_task.ai_vs_ai import HARNESS_VERSION

    family = CONDITIONS[prompt][0]
    traces = sorted((EXPERIMENTS_DIR / family).glob(f"*/{prefix}/*/*_data.json"))
    return [p for p in traces if is_complete_trace(p, model, prompt, effort, num_rounds, HARNESS_VERSION)]


def split(total: int, workers: int) -> list[int]:
    workers = max(1, min(workers, total))
    return [total // workers + (1 if i < total % workers else 0) for i in range(workers)]


def launch_workers(args: argparse.Namespace, prompt: str, prefix: str, missing: int) -> list[subprocess.Popen]:
    run_dir = REPO_ROOT / "data" / "runs" / prefix
    run_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for idx, count in enumerate(split(missing, args.workers), start=1):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        cmd = [
            sys.executable,
            "scripts/run_batch_experiment.py",
            "--sessions", str(count),
            "--prompt-strategy", prompt,
            "--model", args.model,
            "--reasoning-effort", args.reasoning_effort,
            "--basket-set", "5",
            "--max-turns-per-round", str(args.max_turns_per_round),
            "--session-prefix", prefix,
            "--db", str(run_dir / f"worker{idx}_{stamp}.sqlite"),
        ]
        log_path = run_dir / f"worker{idx}_{stamp}.log"
        print(f"  worker {idx}: {count} session(s), log {log_path.relative_to(REPO_ROOT)}")
        log = open(log_path, "w", encoding="utf-8")
        procs.append(subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT))
    return procs


def usage_totals(traces: list[Path]) -> dict[str, int]:
    """Sum the per-call token usage logged in each round's reasoning log."""
    totals = {"calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    for path in traces:
        for round_data in json.loads(path.read_text(encoding="utf-8")):
            log = round_data.get("ai_reasoning_log") or []
            for entry in log if isinstance(log, list) else []:
                usage = entry.get("usage") if isinstance(entry, dict) else None
                if not usage:
                    continue
                totals["calls"] += 1
                for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
                    totals[key] += usage.get(key) or 0
    return totals


def run_analysis(args: argparse.Namespace, traces: list[Path]) -> None:
    import pandas as pd

    slug = model_slug(args.model) + (f"-{args.tag}" if args.tag else "")
    metrics_dir = REPO_ROOT / "data" / f"metrics_{slug}"
    figures_dir = REPO_ROOT / "figures" / f"replication_{slug}"

    # Pass trace files explicitly so only this replication is analyzed, not
    # every session in data/experiments/ that shares a model name. The paper
    # extracted referring expressions with GPT-5.5; keep that fixed.
    analyze = [
        sys.executable, "scripts/analyze_metrics.py", *map(str, traces),
        "--extractor", args.extractor, "--gpt-model", args.extractor_model,
        "--output-dir", str(metrics_dir),
    ]
    print(f"\nAnalyzing {len(traces)} traces -> {metrics_dir.relative_to(REPO_ROOT)}")
    subprocess.run(analyze, cwd=REPO_ROOT, check=True)

    plot_dir, models = metrics_dir, [args.model]
    if args.baseline_metrics:
        # Reuse the paper's computed rows instead of re-extracting, since GPT
        # extraction is not deterministic and would shift the published numbers.
        paper = pd.read_csv(Path(args.baseline_metrics).expanduser() / "metrics_by_session_round.csv")
        paper = paper[paper["director_model"].isin(PAPER_MODELS)]
        ours = pd.read_csv(metrics_dir / "metrics_by_session_round.csv")
        plot_dir = metrics_dir / "with_paper_baseline"
        plot_dir.mkdir(exist_ok=True)
        pd.concat([paper, ours], ignore_index=True).to_csv(plot_dir / "metrics_by_session_round.csv", index=False)
        models = PAPER_MODELS + models

    plot = [
        sys.executable, "scripts/plot_acl_style_metrics.py",
        "--metrics-dir", str(plot_dir),
        "--output-dir", str(figures_dir),
        "--models", *models,
    ]
    print("\n$ " + " ".join(plot))
    subprocess.run(plot, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model for both director and matcher, e.g. gpt-6-luna.")
    parser.add_argument("--sessions", type=int, default=10, help="Target complete sessions per prompt (paper: 10).")
    parser.add_argument(
        "--prompts", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS),
        help="Prompt conditions to run (default: both).",
    )
    parser.add_argument(
        "--reasoning-effort", default="none",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort (paper: none).",
    )
    parser.add_argument("--workers", type=int, default=2, help="Parallel sessions per prompt condition.")
    parser.add_argument(
        "--max-turns-per-round", type=int, default=120,
        help="End a round after this many turns, scoring the board as-is (0 disables).",
    )
    parser.add_argument(
        "--max-passes", type=int, default=3,
        help="Top-up passes to replace sessions that stalled or crashed.",
    )
    parser.add_argument("--tag", help="Suffix for session prefixes, to keep a variant run separate.")
    parser.add_argument(
        "--baseline-metrics",
        help="Folder with the paper's metrics_by_session_round.csv, to plot GPT-5.2/5.5 alongside, e.g. "
             "'~/Documents/dev/lvlms-referential-game/data/(ACTUAL LATEST EMNLP)"
             "metrics_gpt55_latest_all_rounds_reorganized_fixed_identity'",
    )
    parser.add_argument(
        "--extractor", default="gpt", choices=["gpt", "deterministic"],
        help="Referring-expression extractor (paper: gpt).",
    )
    parser.add_argument("--extractor-model", default="gpt-5.5", help="Model for GPT extraction (paper: gpt-5.5).")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run and exit.")
    parser.add_argument("--no-analyze", action="store_true", help="Skip analysis and plotting at the end.")
    return parser.parse_args()


def main() -> None:
    # Show progress immediately when stdout is redirected to a log file.
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    from referential_task.state import Constants

    num_rounds = Constants.num_rounds
    prefixes = {prompt: session_prefix(args.model, prompt, args.tag) for prompt in args.prompts}

    for pass_num in range(1, args.max_passes + 1):
        missing = {}
        print(f"\n== Pass {pass_num}: {args.model}, reasoning effort {args.reasoning_effort} ==")
        for prompt, prefix in prefixes.items():
            done = len(completed_traces(prefix, args.model, prompt, args.reasoning_effort, num_rounds))
            missing[prompt] = max(0, args.sessions - done)
            print(f"{prompt:15} prefix {prefix:22} complete {done}/{args.sessions}")
        if not any(missing.values()):
            break
        if args.dry_run:
            for prompt, count in missing.items():
                if count:
                    print(f"Would run {count} session(s) of {prompt} in {len(split(count, args.workers))} worker(s)")
            break

        procs = []
        for prompt, count in missing.items():
            if count:
                print(f"Launching {count} session(s) for {prompt}:")
                procs += launch_workers(args, prompt, prefixes[prompt], count)
        failed = sum(proc.wait() != 0 for proc in procs)
        if failed:
            print(f"{failed} worker(s) exited with an error; see the worker logs.")
    else:
        print(f"\nStill missing sessions after {args.max_passes} passes; check data/runs/*/worker*.log.")

    if args.dry_run:
        return

    print("\n== Token usage (complete sessions) ==")
    all_traces = []
    for prompt, prefix in prefixes.items():
        traces = completed_traces(prefix, args.model, prompt, args.reasoning_effort, num_rounds)
        all_traces += traces
        totals = usage_totals(traces)
        print(f"{prompt:15} {len(traces)} sessions, {totals['calls']} calls, "
              f"{totals['input_tokens']:,} input ({totals['cached_input_tokens']:,} cached), "
              f"{totals['output_tokens']:,} output tokens")

    if not args.no_analyze:
        run_analysis(args, all_traces)


if __name__ == "__main__":
    main()
