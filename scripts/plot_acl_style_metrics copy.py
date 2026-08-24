#!/usr/bin/env python3
"""Plot ACL-paper-style round trends from analyze_metrics.py outputs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path
from statistics import mean, stdev
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lvlms-referential-game-matplotlib")
)

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    ("accuracy", "Accuracy (%)"),
    ("words", "# Words"),
    ("turns", "# Turns"),
    ("re_words", "# RE Words"),
    ("relative_lexical_overlap", "Lexical Overlap"),
]

CONDITION_ORDER = [
    ("ACL", "ACL_prompt", "gpt-5.2", "ACL prompt, GPT-5.2"),
    ("ACL", "ACL_prompt", "gpt-5.5", "ACL prompt, GPT-5.5"),
    ("Cameron", "cameron-prompt", "gpt-5.2", "Cameron prompt, GPT-5.2"),
    ("Cameron", "cameron-prompt", "gpt-5.5", "Cameron prompt, GPT-5.5"),
]

COLORS = {
    "ACL prompt, GPT-5.2": "#4C78A8",
    "ACL prompt, GPT-5.5": "#F58518",
    "Cameron prompt, GPT-5.2": "#54A24B",
    "Cameron prompt, GPT-5.5": "#B279A2",
}


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def condition_label(row: pd.Series) -> str | None:
    for experiment, prompt, model, label in CONDITION_ORDER:
        if (
            row["experiment"] == experiment
            and row["prompt_strategy"] == prompt
            and row["director_model"] == model
            and row["matcher_model"] == model
        ):
            return label
    return None


def aggregate_rounds(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (condition, round_num), group in df.groupby(
        ["condition", "round"], sort=False, observed=True
    ):
        out: dict[str, Any] = {
            "condition": condition,
            "round": int(round_num),
            "n_sessions": group["session_id"].nunique(),
        }
        for metric, _ in METRICS:
            values = [float(value) for value in group[metric].dropna()]
            out[f"{metric}_mean"] = mean(values) if values else math.nan
            out[f"{metric}_ci95"] = ci95(values)
        rows.append(out)
    return pd.DataFrame(rows)


def write_summary_csv(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)


def write_slope_csv(path: Path, df: pd.DataFrame) -> None:
    rows: list[dict[str, Any]] = []
    for condition, group in df.groupby("condition", sort=False, observed=True):
        group = group.sort_values("round")
        for metric, _ in METRICS:
            metric_group = group.dropna(subset=[metric])
            if len(metric_group) < 2:
                slope = math.nan
            else:
                slope = float(
                    pd.Series(metric_group[metric]).cov(pd.Series(metric_group["round"]))
                    / pd.Series(metric_group["round"]).var()
                )
            rows.append({"condition": condition, "metric": metric, "slope": slope})

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "metric", "slope"])
        writer.writeheader()
        writer.writerows(rows)


def plot_figure(summary: pd.DataFrame, output_base: Path, title: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, len(METRICS), figsize=(13.5, 2.8), constrained_layout=True)
    conditions = [label for *_, label in CONDITION_ORDER]

    for ax, (metric, ylabel) in zip(axes, METRICS):
        for condition in conditions:
            group = summary[summary["condition"] == condition].sort_values("round")
            if group.empty:
                continue
            y = group[f"{metric}_mean"]
            yerr = group[f"{metric}_ci95"]
            ax.errorbar(
                group["round"],
                y,
                yerr=yerr,
                marker="o",
                markersize=4,
                linewidth=1.8,
                capsize=2.5,
                color=COLORS[condition],
                label=condition,
            )

        ax.set_title(ylabel)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(summary["round"].unique()))
        ax.grid(True, axis="y", color="#D8D8D8", linewidth=0.7)
        ax.grid(False, axis="x")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        if metric == "relative_lexical_overlap":
            ax.set_ylim(0, 1.05)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(title, y=1.05, fontsize=11)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path(
            "data/(ACTUAL LATEST EMNLP)metrics_gpt55_latest_all_rounds_reorganized_fixed_identity"
        ),
        help="Directory containing metrics_by_session_round.csv from analyze_metrics.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/emnlp_fixed_identity"),
        help="Directory for generated figures and summary CSVs.",
    )
    parser.add_argument(
        "--max-round",
        type=int,
        default=None,
        help="Only include rounds up to this value.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output basename. Defaults to acl_style_metrics_rounds_1_N.",
    )
    parser.add_argument(
        "--exclude-session-prefix",
        action="append",
        default=[],
        help="Exclude rows whose session_prefix contains this substring. Repeatable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_csv = args.metrics_dir / "metrics_by_session_round.csv"
    df = pd.read_csv(metrics_csv)
    for pattern in args.exclude_session_prefix:
        df = df[~df["session_prefix"].astype(str).str.contains(pattern, regex=False)].copy()
    df["condition"] = df.apply(condition_label, axis=1)
    df = df[df["condition"].notna()].copy()
    if args.max_round is not None:
        df = df[df["round"] <= args.max_round].copy()

    # Round 1 has no prior round. The analysis CSV stores 1.0 as a placeholder,
    # but the paper figure is about overlap with prior rounds, so leave it blank.
    df.loc[df["round"] == 1, "relative_lexical_overlap"] = math.nan

    condition_order = [label for *_, label in CONDITION_ORDER]
    df["condition"] = pd.Categorical(df["condition"], categories=condition_order, ordered=True)
    df = df.sort_values(["condition", "round", "session_id"])

    summary = aggregate_rounds(df)
    max_round = int(df["round"].max())
    output_name = args.name or f"acl_style_metrics_rounds_1_{max_round}"
    output_base = args.output_dir / output_name

    write_summary_csv(output_base.with_name(f"{output_base.name}_summary.csv"), summary)
    write_slope_csv(output_base.with_name(f"{output_base.name}_slopes.csv"), df)
    plot_figure(
        summary,
        output_base,
        title=f"Round-by-round trends (Rounds 1-{max_round})",
    )
    print(f"Wrote {output_base.with_suffix('.png')}")
    print(f"Wrote {output_base.with_suffix('.pdf')}")
    print(f"Wrote {output_base.with_name(f'{output_base.name}_summary.csv')}")
    print(f"Wrote {output_base.with_name(f'{output_base.name}_slopes.csv')}")


if __name__ == "__main__":
    main()
