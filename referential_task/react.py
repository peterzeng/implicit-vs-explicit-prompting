from __future__ import annotations

"""
ReAct baseline for the basket referential task.

ReAct is layered on top of an instruction strategy (ACL_prompt or
cameron-prompt) rather than replacing it. Before the agent's normal reply, it
may take a few private thought -> action -> observation steps:

- zoom: view enlarged images of 1-4 baskets side by side
  (matcher: candidate numbers 1-18; director: target positions 1-12)
- search_history: search earlier dialogue for a word or phrase
- recall_basket (director only): for baskets in the current grid, what the
  director said about the same physical basket in earlier rounds and whether
  it was placed correctly (end-of-round feedback the agents already saw)
- finish: stop investigating

The resulting scratchpad is appended to the context for one final call that
uses the base strategy's response schema, so the turn still produces the usual
{reasoning, utterance, selection} output. Investigation actions never change
the board and never reveal correctness; feedback still arrives only at the end
of a round.
"""

import base64
import io
import json
import logging
import re
from typing import Any, Callable, List, Literal

from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from .ai_providers import call_ai_api, detect_provider_from_model, pop_last_usage
from .prompt_context import _extract_director_basket_number
from .state import Player
from .visual_context import (
    _draw_label_badge,
    _load_font,
    _load_matcher_pool_image_urls,
    _resolve_static_image_path,
)

DEFAULT_MAX_STEPS = 4
MAX_ZOOM_BASKETS = 4
MAX_SEARCH_RESULTS = 8
MAX_SEARCH_SNIPPET_CHARS = 300
MAX_RECALL_MESSAGES_PER_ROUND = 3
MAX_RECALL_SNIPPET_CHARS = 250
MATCHER_POOL_SIZE = 18
NUM_POSITIONS = 12


class ReActStep(BaseModel):
    model_config = {"extra": "forbid"}

    thought: str = Field(
        description="Brief reasoning about what you still need to know before replying.",
    )
    action: Literal["zoom", "search_history", "finish"]
    indices: List[int] = Field(
        description="Baskets to zoom in on (1-4 numbers); empty list for other actions.",
    )
    query: str = Field(
        description="Word or phrase to search for with search_history; empty string otherwise.",
    )


class DirectorReActStep(ReActStep):
    action: Literal["zoom", "search_history", "recall_basket", "finish"]


def step_schema(role: str) -> type[ReActStep]:
    return DirectorReActStep if role == "director" else ReActStep


def react_enabled(player: Player, role: str) -> bool:
    """True when ReAct is on for this session and `role` is in `react_roles`."""
    try:
        cfg = player.session.config
        if not cfg.get("react", False):
            return False
        roles = cfg.get("react_roles") or ("director", "matcher")
        return role in roles
    except Exception:
        return False


def react_max_steps(player: Player) -> int:
    try:
        value = int(player.session.config.get("react_max_steps") or DEFAULT_MAX_STEPS)
    except Exception:
        value = DEFAULT_MAX_STEPS
    return max(0, value)


def react_step_model_config(player: Player, role_model_config: dict[str, Any]) -> dict[str, Any]:
    """Model config for investigation steps.

    Defaults to the role's own model. `react_step_model` in the session config
    swaps in a different (e.g., smaller) model for the steps only; the final
    reply always uses the role's model.
    """
    config = dict(role_model_config)
    try:
        step_model = player.session.config.get("react_step_model")
    except Exception:
        step_model = None
    if step_model:
        config["model"] = step_model
        config["provider"] = detect_provider_from_model(step_model)
    return config


def build_react_instructions(role: str, max_steps: int) -> str:
    if role == "matcher":
        zoom_target = "candidate baskets by their candidate numbers (1-18)"
        action_options = '"zoom" | "search_history" | "finish"'
        indices_hint = "numbers for zoom"
        recall_line = ""
    else:
        zoom_target = "baskets in your target grid by their positions (1-12)"
        action_options = '"zoom" | "search_history" | "recall_basket" | "finish"'
        indices_hint = "positions for zoom or recall_basket"
        recall_line = (
            f"- recall_basket: for 1-{MAX_ZOOM_BASKETS} baskets in your current grid (positions 1-12), "
            "see what you said about the same physical basket in earlier rounds and whether it was "
            "placed correctly in those rounds.\n"
        )
    return (
        "INVESTIGATION PHASE (private):\n"
        f"Before you reply to your partner, you may take up to {max_steps} investigation steps. "
        "Your partner does not see these steps, and they do not change the board.\n\n"
        "Each step, respond with a single JSON object:\n"
        "{\n"
        '  "thought": "brief reasoning about what you still need to know",\n'
        f'  "action": {action_options},\n'
        f'  "indices": [{indices_hint}, otherwise []],\n'
        '  "query": "search text for search_history, otherwise empty"\n'
        "}\n\n"
        "Actions:\n"
        f"- zoom: view enlarged images of 1-{MAX_ZOOM_BASKETS} {zoom_target}, side by side.\n"
        "- search_history: search the earlier dialogue (this round and any previous rounds you can see) "
        "for a word or phrase, such as a name your partner used for a basket.\n"
        f"{recall_line}"
        "- finish: stop investigating and write your reply.\n\n"
        "You will NOT learn whether any placement in the CURRENT round is correct until the round ends. "
        "Choose finish as soon as you have enough information to reply."
    )


def _encode_png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _render_zoom_image(items: list[tuple[str, str]], tile: int = 320) -> str | None:
    """Render (label, image_rel_path) pairs side by side with label badges."""
    pad = 16
    label_h = 44
    canvas = Image.new(
        "RGB",
        (len(items) * tile + (len(items) + 1) * pad, tile + label_h + 2 * pad),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = _load_font(22)
    rendered = 0
    for i, (label, rel_path) in enumerate(items):
        x = pad + i * (tile + pad)
        fs_path = _resolve_static_image_path(rel_path)
        if not fs_path:
            continue
        try:
            basket = Image.open(fs_path).convert("RGBA")
        except Exception:
            continue
        basket.thumbnail((tile, tile))
        offset_x = x + (tile - basket.width) // 2
        offset_y = pad + label_h + (tile - basket.height) // 2
        canvas.paste(basket, (offset_x, offset_y), basket)
        _draw_label_badge(
            draw,
            canvas,
            label,
            center_x=x + tile // 2,
            center_y=pad + label_h // 2,
            font=font,
            bg_color=(30, 64, 175),
            text_color=(255, 255, 255),
        )
        rendered += 1
    if not rendered:
        return None
    return _encode_png_data_url(canvas)


def _zoom_items(player: Player, role: str, indices: list[int]) -> tuple[list[tuple[str, str]], list[int]]:
    """Map requested indices to (label, image path); return invalid indices too."""
    if role == "matcher":
        pool = _load_matcher_pool_image_urls(player)[:MATCHER_POOL_SIZE]
        paths = [((item.get("slot") or {}).get("image") or "") for item in pool]
        label_prefix = "Candidate"
    else:
        try:
            grid = json.loads(player.group.shared_grid or "[]")
        except Exception:
            grid = []
        paths = [(slot.get("image") or "") for slot in grid[:NUM_POSITIONS]]
        label_prefix = "Basket"

    items: list[tuple[str, str]] = []
    invalid: list[int] = []
    for idx in indices:
        if 1 <= idx <= len(paths) and paths[idx - 1]:
            items.append((f"{label_prefix} {idx}", paths[idx - 1].lstrip("/ ")))
        else:
            invalid.append(idx)
    return items, invalid


def run_zoom(player: Player, role: str, indices: list[int]) -> tuple[str, str | None]:
    """Return (observation text, image data URL or None)."""
    unique: list[int] = []
    for idx in indices:
        if idx not in unique:
            unique.append(idx)
    requested = unique[:MAX_ZOOM_BASKETS]
    items, invalid = _zoom_items(player, role, requested)

    notes = []
    if len(unique) > MAX_ZOOM_BASKETS:
        notes.append(f"Only the first {MAX_ZOOM_BASKETS} baskets are shown.")
    if invalid:
        valid_range = "1-18" if role == "matcher" else "1-12"
        notes.append(f"Ignored invalid numbers {invalid}; valid range is {valid_range}.")
    if not items:
        return "OBSERVATION: Nothing to show. " + " ".join(notes), None

    image_url = _render_zoom_image(items)
    if image_url is None:
        return "OBSERVATION: The basket images could not be loaded.", None
    labels = ", ".join(label for label, _ in items)
    return " ".join([f"OBSERVATION: Enlarged view of {labels}."] + notes), image_url


def run_search_history(
    history: list[dict[str, Any]], query: str, current_round: int
) -> str:
    """Keyword search over the dialogue history the agent is allowed to see."""
    tokens = [t for t in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(t) > 2]
    if not tokens:
        return "OBSERVATION: Empty or too short search query; nothing searched."

    scored = []
    for order, msg in enumerate(history):
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        lowered = text.lower()
        score = sum(1 for t in tokens if t in lowered)
        if score:
            scored.append((score, order, msg))

    if not scored:
        return f"OBSERVATION: No earlier messages mention {query!r}."

    top = sorted(scored, key=lambda s: (-s[0], s[1]))[:MAX_SEARCH_RESULTS]
    top.sort(key=lambda s: s[1])
    lines = [f"OBSERVATION: {len(top)} of {len(scored)} matching messages for {query!r}:"]
    for _, _, msg in top:
        round_num = msg.get("round_number") or current_round
        speaker = "feedback" if msg.get("is_feedback") else (msg.get("sender_role") or "unknown")
        text = msg["text"].strip()
        if len(text) > MAX_SEARCH_SNIPPET_CHARS:
            text = text[:MAX_SEARCH_SNIPPET_CHARS] + "..."
        lines.append(f"[Round {round_num}] {speaker}: {text}")
    return "\n".join(lines)


def _cross_round_history_enabled(player: Player) -> bool:
    try:
        return bool(player.session.config.get("cross_round_history", False))
    except Exception:
        return False


def _director_messages_by_basket(messages: list[dict[str, Any]]) -> dict[int, list[str]]:
    """Group director messages by the basket they describe.

    A message starting with "Basket N" opens basket N; later director messages
    without a marker (clarifications) are attributed to the last opened basket.
    """
    by_basket: dict[int, list[str]] = {}
    current = None
    for msg in messages:
        if msg.get("sender_role") != "director":
            continue
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        current = _extract_director_basket_number(text) or current
        if current is not None:
            by_basket.setdefault(current, []).append(text)
    return by_basket


def run_recall_basket(player: Player, positions: list[int]) -> str:
    """Director-only: earlier-round descriptions and outcomes for current baskets."""
    if not _cross_round_history_enabled(player):
        return "OBSERVATION: Earlier rounds are not available in this session."
    current_round = getattr(player, "round_number", 1) or 1
    try:
        prior_rounds = sorted(
            (p for p in player.in_all_rounds() if (p.round_number or 0) < current_round),
            key=lambda p: p.round_number,
        )
    except Exception:
        prior_rounds = []
    if not prior_rounds:
        return "OBSERVATION: This is the first round; there are no earlier rounds to recall."

    try:
        grid = json.loads(player.group.shared_grid or "[]")
    except Exception:
        grid = []

    unique: list[int] = []
    for pos in positions:
        if pos not in unique:
            unique.append(pos)
    valid = [p for p in unique[:MAX_ZOOM_BASKETS] if 1 <= p <= len(grid) and grid[p - 1].get("image")]
    invalid = [p for p in unique[:MAX_ZOOM_BASKETS] if p not in valid]
    if not valid:
        return f"OBSERVATION: No valid positions given (valid range is 1-{NUM_POSITIONS})."

    lines = ["OBSERVATION:"]
    for pos in valid:
        image = grid[pos - 1]["image"]
        lines.append(f"Basket {pos} in the current round:")
        for prev in prior_rounds:
            try:
                prev_grid = json.loads(prev.group.shared_grid or "[]")
                prev_messages = json.loads(prev.group.ai_messages or "[]")
                submitted = json.loads(prev.group.matcher_sequence or "[]")
            except Exception:
                continue
            prev_pos = next(
                (i + 1 for i, slot in enumerate(prev_grid) if slot.get("image") == image), None
            )
            if prev_pos is None:
                lines.append(f"  Round {prev.round_number}: not in the target grid.")
                continue
            placed = next(
                (s.get("image") for s in submitted if str(s.get("position")) == str(prev_pos)), None
            )
            if placed is None:
                outcome = "no basket was submitted for it"
            elif placed == image:
                outcome = "placed correctly"
            else:
                outcome = "placed INCORRECTLY (a different basket was put there)"
            said = _director_messages_by_basket(prev_messages).get(prev_pos, [])
            quoted = " | ".join(
                f'"{t[:MAX_RECALL_SNIPPET_CHARS]}{"..." if len(t) > MAX_RECALL_SNIPPET_CHARS else ""}"'
                for t in said[:MAX_RECALL_MESSAGES_PER_ROUND]
            ) or "(no description found)"
            lines.append(f"  Round {prev.round_number} (was basket {prev_pos}), {outcome}. You said: {quoted}")
    if invalid:
        lines.append(f"Ignored invalid positions {invalid}; valid range is 1-{NUM_POSITIONS}.")
    return "\n".join(lines)


def _parse_step(text: str | None, schema: type[ReActStep] = ReActStep) -> ReActStep | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    try:
        return schema.model_validate(json.loads(text[start:end]))
    except Exception:
        return None


def run_react_loop(
    player: Player,
    role: str,
    base_messages: list[dict[str, Any]],
    history: list[dict[str, Any]],
    step_model_config: dict[str, Any],
    max_steps: int,
    call_fn: Callable[[Callable[[], str | None]], str | None] = lambda f: f(),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the investigation phase.

    Returns (scratchpad messages to append before the final call, trace for
    logging). `call_fn` wraps each API call (e.g., with retries).
    """
    current_round = getattr(player, "round_number", 1) or 1
    instructions = {"role": "developer", "content": build_react_instructions(role, max_steps)}
    scratchpad: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []

    for step_num in range(1, max_steps + 1):
        text = call_fn(
            lambda: call_ai_api(
                messages=base_messages + [instructions] + scratchpad,
                model_config=step_model_config,
                temperature=0,
                response_format={"type": "json_schema", "json_schema": step_schema(role)},
            )
        )
        step = _parse_step(text, step_schema(role))
        entry: dict[str, Any] = {
            "step": step_num,
            "model": step_model_config.get("model"),
            "usage": pop_last_usage(),
        }
        if step is None:
            entry["error"] = "unparseable step"
            entry["raw_text"] = text
            trace.append(entry)
            logging.warning("[REACT] %s step %d unparseable; ending investigation", role.upper(), step_num)
            break

        entry.update(step.model_dump())
        scratchpad.append({"role": "assistant", "content": step.model_dump_json()})
        if step.action == "finish":
            trace.append(entry)
            break

        if step.action == "zoom":
            obs_text, image_url = run_zoom(player, role, step.indices)
        elif step.action == "recall_basket":
            obs_text, image_url = run_recall_basket(player, step.indices), None
        else:
            obs_text, image_url = run_search_history(history, step.query, current_round), None
        entry["observation"] = obs_text
        trace.append(entry)
        logging.info("[REACT] %s step %d: %s %s", role.upper(), step_num, step.action, step.indices or step.query)

        if image_url:
            scratchpad.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": obs_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            })
        else:
            scratchpad.append({"role": "user", "content": obs_text})

    if scratchpad:
        scratchpad.append({
            "role": "user",
            "content": (
                "INVESTIGATION COMPLETE. Now write your reply to your partner, "
                "following your original response format exactly."
            ),
        })
    return scratchpad, trace
