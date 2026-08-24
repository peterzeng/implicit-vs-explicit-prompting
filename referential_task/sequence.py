from __future__ import annotations

import json
from typing import Any

from .state import Player
from .visual_context import _load_matcher_pool_image_urls

def _update_ai_partial_sequence(
    player: Player,
    selection: dict[str, Any] | None,
    *,
    allow_move: bool = True,
):
    """
    Update the group's incremental AI matcher sequence based on a single
    basket choice for the current turn.

    Instead of asking the model for the full 12‑basket sequence on every
    turn, we maintain an incremental sequence in `ai_partial_sequence`.

    Returns a tuple of (updated_partial_sequence, vacated_position).
    vacated_position is an int if a move caused a previously filled position
    to become empty, or None otherwise.

    When allow_move is False, selecting a candidate that is already placed in
    another position is treated as an accidental duplicate and ignored. This
    prevents late-round loops where the model says it is filling the last slot
    but reuses a candidate already sitting elsewhere, causing positions to
    alternate between filled and empty.
    """
    # `selection` must be a dict with fields
    #   {"candidate_index": int|None, "position": int|None, ...}
    selected_candidate_index = None
    target_position = None
    if isinstance(selection, dict):
        try:
            if selection.get("candidate_index") is not None:
                selected_candidate_index = int(selection.get("candidate_index"))
        except Exception:
            selected_candidate_index = None
        try:
            if selection.get("position") is not None:
                target_position = int(selection.get("position"))
        except Exception:
            target_position = None

    # If there is no concrete basket choice (clarification-only turn), do not
    # modify the partial sequence; just return the existing state.
    if selected_candidate_index is None:
        try:
            return json.loads(
                getattr(player.group, "ai_partial_sequence", "") or "[]"
            ), None
        except Exception:
            return None, None

    group = player.group
    try:
        partial = json.loads(getattr(group, "ai_partial_sequence", "") or "[]")
    except Exception:
        partial = []

    # Determine which logical position to update.
    # IMPORTANT: positions that were "cleared" by a move are stored with
    # {"position": k, "image": None}. Those should be considered EMPTY, so they
    # must NOT count as "used" when selecting the next available slot.
    used_positions: set[int] = set()
    for item in partial:
        if not isinstance(item, dict):
            continue
        pos_raw = item.get("position")
        img = item.get("image")
        # Only treat a position as used if it currently has a non-empty image.
        if not img:
            continue
        try:
            pos_int = int(pos_raw)
        except Exception:
            continue
        if 1 <= pos_int <= 12:
            used_positions.add(pos_int)
    if isinstance(target_position, int) and 1 <= target_position <= 12:
        pos = target_position
    else:
        # Legacy behaviour: append to the next unused slot.
        pos = 1
        while pos in used_positions and pos <= 12:
            pos += 1
    if pos < 1 or pos > 12:
        # Already have 12 positions filled; nothing to update.
        return partial, None

    # Map the selected candidate index onto a slot in the matcher's pool.
    try:
        pool_items = _load_matcher_pool_image_urls(player)
    except Exception:
        pool_items = []
    try:
        idx_zero_based = int(selected_candidate_index) - 1
    except Exception:
        idx_zero_based = -1
    candidate_slot = None
    if 0 <= idx_zero_based < len(pool_items):
        candidate_slot = (pool_items[idx_zero_based] or {}).get("slot") or None
    if not candidate_slot:
        # If we cannot map this candidate index back to a known slot, bail out.
        return partial, None

    selected_image = candidate_slot.get("image")
    selected_original_position = candidate_slot.get("position")

    # If this physical basket is already present anywhere in the partial
    # sequence, treat the new selection as a move.
    previous_pos = None
    for item in partial:
        if not isinstance(item, dict):
            continue
        try:
            item_pos = int(item.get("position"))
        except Exception:
            item_pos = None
        same_image = selected_image is not None and item.get("image") == selected_image
        same_orig = (
            selected_original_position is not None
            and item.get("originalPosition") == selected_original_position
        )
        if same_image or same_orig:
            previous_pos = item_pos
            break

    if previous_pos is not None and previous_pos != pos and not allow_move:
        return partial, None

    # Remove any previous entry for this logical position or previous_pos.
    # NOTE: Stored JSON may contain positions as strings, ints, or floats.
    # Always compare using int-normalized positions to avoid duplicates like:
    #   {"position": "3", "image": None} and {"position": 3, "image": "..."}.
    cleaned: list[dict[str, Any]] = []
    for item in partial:
        if not isinstance(item, dict):
            continue
        try:
            item_pos_int = int(item.get("position"))
        except Exception:
            # Keep malformed entries out of the debug state.
            continue
        if item_pos_int == pos or (previous_pos is not None and item_pos_int == previous_pos):
            continue
        cleaned.append(item)
    partial = cleaned
    # If the basket was moved from another position, mark that position as empty
    if previous_pos is not None:
        partial.append(
            {
                "position": previous_pos,
                "image": None,
                "originalPosition": None,
            }
        )
    partial.append(
        {
            "position": pos,
            "image": candidate_slot.get("image"),
            "originalPosition": candidate_slot.get("position"),
        }
    )
    # Final pass: de-duplicate by logical position (last write wins).
    # This prevents rendering/logic issues if earlier state included duplicates.
    by_pos: dict[int, dict[str, Any]] = {}
    for item in partial:
        if not isinstance(item, dict):
            continue
        try:
            p_int = int(item.get("position"))
        except Exception:
            continue
        if 1 <= p_int <= 12:
            by_pos[p_int] = item
    partial_sorted = [by_pos[p] for p in sorted(by_pos.keys())]
    try:
        group.ai_partial_sequence = json.dumps(partial_sorted)
    except Exception:
        # Do not let debug state break the main flow.
        pass
    # Return the vacated position (if a basket was moved) so the caller
    # can notify the Matcher that it needs to ask for a re-description.
    vacated = previous_pos if previous_pos is not None and previous_pos != pos else None
    return partial_sorted, vacated


# ---------------------------------------------------------------------------
# AI vs AI Orchestration
# ---------------------------------------------------------------------------
# These functions allow running both Director and Matcher as AI agents,
