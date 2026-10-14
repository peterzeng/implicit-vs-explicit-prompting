#!/usr/bin/env python3
"""Audit how matchers moved already-placed baskets, by replaying their selections.

Rejected selections were not logged before harness version 2, so this replays
every matcher selection in each round through the placement code and reports:

- move attempts: selecting a basket already placed in a different position
- how many were applied or blocked, and whether the move targeted the
  correct slot
- for blocked moves, the round's accuracy if the moves had been allowed
  (a rough counterfactual: the dialogue after the move would have differed)

Each round is replayed under the rules it ran with: version 1 (the EMNLP
paper) allowed a move only when the utterance contained a move keyword;
version 2+ uses `selection.move`. Rounds whose replayed board does not match
the recorded one are counted, so a mismatch means the audit cannot be trusted
for that data.

    python scripts/audit_move_attempts.py ~/Documents/dev/lvlms-referential-game/data/experiments
    python scripts/audit_move_attempts.py data/runs/gpt6-luna-cameron/worker1_20261009-153700.sqlite
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from referential_task.prompt_context import _build_matcher_current_sequence_state_for_prompt
from referential_task.sequence import _update_ai_partial_sequence
from referential_task.state import Group, Player, Session

# The paper's rule (harness version 1, removed in version 2): a move was
# allowed only if the matcher's utterance contained one of these.
V1_MOVE_MARKERS = (
    "move", "moving", "moved", "swap", "swapping", "replace", "replacing",
    "correction", "correcting", "wrong position", "from position", "from slot",
    "instead of position", "instead of slot",
)


def v1_allows_move(utterance: str | None) -> bool:
    text = (utterance or "").lower()
    return any(marker in text for marker in V1_MOVE_MARKERS)


def make_player(round_data: dict) -> Player:
    group = Group()
    group.shared_grid = json.dumps(round_data["shared_grid"])
    group.ai_partial_sequence = "[]"
    return Player(
        role="observer", group=group, session=Session(dict(round_data["config"])),
        round_number=round_data["round_number"],
    )


def board(player: Player) -> dict[int, str | None]:
    return {
        int(item["position"]): item.get("image")
        for item in json.loads(player.group.ai_partial_sequence or "[]")
    }


def correct_count(player: Player, shared_grid: list[dict]) -> int:
    placed = board(player)
    return sum(placed.get(i + 1) == slot["image"] for i, slot in enumerate(shared_grid))


def replay_round(round_data: dict, force_allow_moves: bool = False) -> tuple[Counter, Player]:
    version = round_data["config"].get("harness_version", 1)
    player = make_player(round_data)
    counts: Counter = Counter()
    for entry in round_data.get("ai_reasoning_log") or []:
        selection = entry.get("selection") if isinstance(entry, dict) else None
        if not isinstance(selection, dict) or selection.get("candidate_index") is None:
            continue
        if entry.get("ai_role") not in (None, "matcher"):
            continue
        allowed = bool(selection.get("move")) if version >= 2 else v1_allows_move(entry.get("utterance"))

        slots = _build_matcher_current_sequence_state_for_prompt(player)["sequence_slots"]
        placed_at = {s["candidate_index"]: s["position"] for s in slots if s.get("candidate_index")}
        candidate, position = int(selection["candidate_index"]), selection.get("position")
        if candidate in placed_at and position is not None and placed_at[candidate] != int(position):
            counts["move attempts"] += 1
            image = next(s["image"] for s in slots if s.get("candidate_index") == candidate)
            to_correct_slot = image == round_data["shared_grid"][int(position) - 1]["image"]
            if allowed:
                counts["moves applied"] += 1
            else:
                counts["moves blocked"] += 1
                counts["blocked moves to the correct slot"] += int(to_correct_slot)
        _update_ai_partial_sequence(player, selection, allow_move=allowed or force_allow_moves)
    return counts, player


def load_rounds(paths: list[Path]):
    for root in paths:
        files = [root] if root.is_file() else sorted(root.rglob("*_data.json")) + sorted(root.rglob("*.sqlite"))
        for path in files:
            if path.suffix == ".sqlite":
                rounds = load_sqlite_rounds(path)
            else:
                rounds = json.loads(path.read_text(encoding="utf-8"))
            for round_data in rounds:
                if isinstance(round_data.get("config"), str):
                    round_data["config"] = json.loads(round_data["config"])
                yield path, round_data


def load_sqlite_rounds(path: Path) -> list[dict]:
    """Rounds from a batch worker's live-state file, including unexported sessions."""
    import sqlite3

    from referential_task.exporting import load_session_export_data

    with sqlite3.connect(path) as conn:
        session_ids = [row[0] for row in conn.execute("SELECT DISTINCT session_id FROM game_sessions")]
    return [r for sid in session_ids for r in load_session_export_data(str(path), sid)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "paths", nargs="+", type=Path,
        help="Experiment folders, *_data.json traces, or batch-worker *.sqlite files (for unexported sessions).",
    )
    parser.add_argument("--rounds", action="store_true", help="List every round with a blocked move.")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)

    totals: dict[tuple, Counter] = defaultdict(Counter)
    blocked_rounds = []
    for path, round_data in load_rounds([p.expanduser() for p in args.paths]):
        config = round_data["config"]
        key = (config.get("prompt_strategy"), config.get("ai_matcher_model"), config.get("harness_version", 1))
        counts, player = replay_round(round_data)
        recorded = {
            int(item["position"]): item.get("image") for item in round_data.get("ai_partial_sequence") or []
        }
        counts["rounds"] += 1
        counts["replay matches recorded board"] += int(board(player) == recorded)
        totals[key] += counts

        if counts["moves blocked"]:
            _, allowed_player = replay_round(round_data, force_allow_moves=True)
            empty = [p for p in range(1, 13) if not board(allowed_player).get(p)]
            blocked_rounds.append((
                path.name, *key, round_data["round_number"], counts["moves blocked"],
                correct_count(player, round_data["shared_grid"]),
                correct_count(allowed_player, round_data["shared_grid"]), empty,
            ))

    print("prompt, matcher model, harness version: counts")
    for key in sorted(totals, key=str):
        print(f"{key}: {dict(totals[key])}")

    if blocked_rounds:
        changed = sum(r[-2] > r[-3] for r in blocked_rounds)
        print(f"\nRounds with blocked moves: {len(blocked_rounds)}; "
              f"higher accuracy with the moves allowed: {changed}")
    if args.rounds:
        print("\ntrace, prompt, model, version, round, blocked, correct as recorded, "
              "correct with moves allowed, empty slots with moves allowed")
        for row in blocked_rounds:
            print(", ".join(str(v) for v in row))


if __name__ == "__main__":
    main()
