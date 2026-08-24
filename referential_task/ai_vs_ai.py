from __future__ import annotations

import base64
import datetime
import json
import os
import random
from typing import Any

from openai import OpenAI

from .state import Constants, Player
from .prompt_context import (
    _build_cross_round_matcher_feedback_memory,
    _build_matcher_current_sequence_state_for_prompt,
    _compute_round_correct_count,
    _get_pending_refill_positions,
    _get_prompt_strategy_name,
    _inject_task_background,
    _is_acl_prompt_strategy,
    _is_instruction_message,
)
from .sequence import _update_ai_partial_sequence
from .visual_context import _inject_round_feedback_context, _inject_visual_grid_context


# Models that support the reasoning_effort parameter.
GPT_5_2_MODELS = frozenset({
    "gpt-5",
    "gpt-5.2",
    "gpt-5.2-mini",
    "gpt-5.2-chat-latest",
    "gpt-5.2-pro",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.5",
})


def _is_gpt_5_2_model(model: str) -> bool:
    """Check if a model supports GPT-5.x API features (reasoning_effort).
    
    Returns True for GPT-5, GPT-5.2, GPT-5.4, GPT-5.5, and GPT-5 snapshots that
    support the reasoning_effort parameter.
    """
    if not model:
        return False
    # Exact match in known set
    if model in GPT_5_2_MODELS:
        return True
    # Pattern match for GPT-5 snapshots plus supported GPT-5.x variants.
    model_lower = model.lower()
    return (
        model_lower == "gpt-5"
        or model_lower.startswith("gpt-5-")
        or model_lower.startswith("gpt-5.2")
        or model_lower.startswith("gpt-5.4")
        or model_lower.startswith("gpt-5.5")
    )


def _reasoning_effort_for_api(model: str, reasoning_effort: str | None) -> str | None:
    """Return a reasoning_effort value that is valid to send for a model."""
    if not _is_gpt_5_2_model(model):
        return None
    effort = reasoning_effort or "none"
    model_lower = model.lower() if model else ""
    if effort == "none" and (model_lower == "gpt-5" or model_lower.startswith("gpt-5-")):
        return None
    return effort


def _uses_max_completion_tokens(model: str) -> bool:
    """Check if a model requires max_completion_tokens instead of max_tokens.
    
    OpenAI's o1, o3, and gpt-5.x series models use max_completion_tokens.
    """
    if not model:
        return False
    model_lower = model.lower()
    # o1, o3 series models
    if model_lower.startswith(("o1", "o3")):
        return True
    # GPT-5.x series models
    if model_lower.startswith("gpt-5"):
        return True
    return False


def _coerce_ready_to_submit(value: Any) -> bool:
    """Parse matcher submit intent without treating arbitrary strings as truthy."""
    if value is True:
        return True
    if value is False or value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1"}
    if isinstance(value, (int, float)):
        return value == 1
    return False


def _filled_sequence_by_position(sequence: Any) -> dict[int, dict[str, Any]]:
    """Return unique filled matcher slots keyed by 1-indexed position."""
    by_pos: dict[int, dict[str, Any]] = {}
    if not isinstance(sequence, list):
        return by_pos
    for item in sequence:
        if not isinstance(item, dict):
            continue
        try:
            p = int(item.get("position"))
        except Exception:
            continue
        if 1 <= p <= 12 and item.get("image"):
            by_pos[p] = item
    return by_pos


def _clean_ai_utterance(value: Any) -> str | None:
    """Return a non-empty chat utterance, rejecting JSON null placeholders."""
    if not isinstance(value, str):
        return None
    utterance = value.strip()
    if not utterance or utterance.lower() in {"null", "none"}:
        return None
    return utterance


def _utterance_explicitly_moves_selection(utterance: str | None) -> bool:
    """Return True when the matcher explicitly says it is correcting by moving."""
    if not utterance:
        return False
    text = utterance.lower()
    move_markers = (
        "move",
        "moving",
        "moved",
        "swap",
        "swapping",
        "replace",
        "replacing",
        "correction",
        "correcting",
        "wrong position",
        "from position",
        "from slot",
        "instead of position",
        "instead of slot",
    )
    return any(marker in text for marker in move_markers)


def _get_ai_model(player: Player | None = None, ai_role: str | None = None) -> str:
    """Get the role-specific AI model from session config or environment.
    
    Priority:
    1. Session config ai_director_model / ai_matcher_model
    2. Environment variable OPENAI_MODEL
    3. Default: 'gpt-5.2'
    """
    if player is not None:
        try:
            if hasattr(player, "session") and player.session:
                cfg = player.session.config or {}
                role = ai_role
                if role not in ("director", "matcher"):
                    human_role = (
                        player.field_maybe_none("player_role")
                        or player.participant.vars.get("role")
                    )
                    role = "matcher" if human_role == "director" else "director"
                model = cfg.get(f"ai_{role}_model")
                if model:
                    return model
        except Exception:
            pass
    return os.environ.get("OPENAI_MODEL", "gpt-5.2")


def _get_reasoning_effort(player: Player | None = None) -> str:
    """Get the reasoning effort level for GPT-5.2+ models.
    
    Priority:
    1. Session config 'ai_reasoning_effort' (if player provided)
    2. Default: 'none'
    """
    if player is not None:
        try:
            if hasattr(player, "session") and player.session:
                cfg = player.session.config or {}
                effort = cfg.get("ai_reasoning_effort")
                if effort:
                    return effort
        except Exception:
            pass
    return "none"


def _build_api_call_kwargs(
    model: str,
    messages: list,
    player: Player | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict | None = None,
) -> dict:
    """Build the kwargs dict for an OpenAI API call, handling GPT-5.2+ specifics.
    
    Automatically adds reasoning_effort for GPT-5.2+ models.
    Uses max_completion_tokens for o1/o3/gpt-5.x models instead of max_tokens.
    Skips temperature for reasoning models (o1/o3 or GPT-5.2+ with reasoning enabled).
    """
    kwargs = {
        "model": model,
        "messages": messages,
    }
    
    # Determine if this is a reasoning model that doesn't support custom temperature
    model_lower = model.lower() if model else ""
    is_o_series = model_lower.startswith(("o1", "o3"))
    reasoning_effort = _get_reasoning_effort(player) if _is_gpt_5_2_model(model) else "none"
    is_reasoning_mode = is_o_series or _is_gpt_5_2_model(model) or (reasoning_effort != "none")
    
    # Only add temperature for non-reasoning models (reasoning models only support temperature=1)
    if temperature is not None and not is_reasoning_mode:
        kwargs["temperature"] = temperature
    
    if max_tokens is not None:
        # Use max_completion_tokens for models that require it (o1, o3, gpt-5.x)
        if _uses_max_completion_tokens(model):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
    
    if response_format is not None:
        if response_format.get("type") == "json_schema":
            schema = response_format.get("json_schema", {})
            if hasattr(schema, "model_json_schema"):
                kwargs["response_format"] = schema
            else:
                kwargs["response_format"] = response_format
        else:
            kwargs["response_format"] = response_format
    
    api_reasoning_effort = _reasoning_effort_for_api(model, reasoning_effort)
    if api_reasoning_effort is not None:
        kwargs["reasoning_effort"] = api_reasoning_effort
    
    return kwargs


def _get_ai_client():
    """Return an OpenAI client if configured, otherwise None.

    We fail gracefully when the library or API key is missing so the app
    can still run (the chat will simply not have an AI partner reply).
    
    Supports local VLM servers (vLLM, SGLang, Ollama) via OPENAI_API_BASE env var.
    When OPENAI_API_BASE is set, points to that endpoint instead of OpenAI.
    """
    if OpenAI is None:
        return None
    
    # Check for local/custom API endpoint (vLLM, SGLang, Ollama, etc.)
    base_url = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if base_url:
        # Local endpoint - API key may be optional (use dummy if not provided)
        return OpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
        )
    
    # Remote OpenAI - requires real API key
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _api_call_with_retry(api_func, max_retries: int = 3, base_delay: float = 2.0):
    """Execute an API call with exponential backoff retry on rate limit errors.

    Args:
        api_func: A callable that makes the API request (no arguments)
        max_retries: Maximum number of retry attempts (default 3)
        base_delay: Initial delay in seconds (doubles each retry)

    Returns:
        The result of api_func() if successful

    Raises:
        The last exception if all retries are exhausted
    """
    import logging
    import time

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return api_func()
        except Exception as e:
            last_exception = e
            err_name = type(e).__name__

            # Check if it's a rate limit or server error worth retrying
            is_retryable = any(x in err_name.lower() for x in ['ratelimit', 'rate_limit', 'timeout', 'connection', 'server'])
            # Also check error message for rate limit indicators
            err_msg = str(e).lower()
            if 'rate' in err_msg or '429' in err_msg or '503' in err_msg or '502' in err_msg:
                is_retryable = True

            if not is_retryable or attempt >= max_retries:
                # Non-retryable error or exhausted retries
                if attempt > 0:
                    logging.warning("[API_RETRY] All %d retries exhausted for %s: %s", max_retries, err_name, e)
                raise

            # Calculate exponential backoff delay
            delay = base_delay * (2 ** attempt)
            # Add jitter (±20%)
            jitter = delay * 0.2 * (2 * random.random() - 1)
            delay = delay + jitter

            logging.warning(
                "[API_RETRY] %s on attempt %d/%d, retrying in %.1fs: %s",
                err_name, attempt + 1, max_retries + 1, delay, str(e)[:100]
            )
            time.sleep(delay)

    # Should not reach here, but just in case
    raise last_exception

def is_ai_vs_ai_session(player: Player) -> bool:
    """Check if the current session is configured for AI vs AI mode."""
    try:
        if hasattr(player, "session") and player.session:
            return bool(player.session.config.get("ai_vs_ai_mode", False))
    except Exception:
        pass
    return False


def _cross_round_history_enabled(player: Player) -> bool:
    """Return whether prior rounds should be included for both AI roles."""
    try:
        if hasattr(player, "session") and player.session:
            return bool(player.session.config.get("cross_round_history", False))
    except Exception:
        pass
    return False


def _prompt_context_debug_enabled(player: Player) -> bool:
    """Return whether to dump full prompt context bundles before model calls."""
    try:
        if hasattr(player, "session") and player.session:
            return bool(player.session.config.get("debug_prompt_context", False))
    except Exception:
        pass
    return False


def _safe_debug_name(value: Any, fallback: str = "debug") -> str:
    text = str(value or "").strip() or fallback
    safe = []
    for ch in text:
        safe.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    return "".join(safe).strip("._") or fallback


def _decode_data_url(data_url: str) -> tuple[bytes | None, str]:
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        return None, "bin"
    try:
        header, b64_data = data_url.split(",", 1)
        mime = header.split(";", 1)[0].replace("data:", "")
        ext = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/jpg": "jpg",
            "image/webp": "webp",
        }.get(mime, "bin")
        return base64.b64decode(b64_data), ext
    except Exception:
        return None, "bin"


def _dump_prompt_context(player: Player, ai_role: str, messages: list[dict[str, Any]]) -> None:
    """Write the exact text/image prompt bundle sent to the AI provider."""
    try:
        app_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(app_dir)
        session_cfg = getattr(getattr(player, "session", None), "config", {}) or {}
        session_name = _safe_debug_name(session_cfg.get("session_prefix"), "session")
        round_num = int(getattr(player, "round_number", 1) or 1)

        try:
            current_messages = json.loads(player.group.ai_messages or "[]")
            turn_num = len(current_messages) + 1
        except Exception:
            turn_num = 0

        dump_dir = os.path.join(
            project_root,
            "_static",
            "ai_debug",
            "prompt_context",
            session_name,
            f"round_{round_num:02d}",
            f"turn_{turn_num:02d}_{_safe_debug_name(ai_role)}",
        )
        os.makedirs(dump_dir, exist_ok=True)

        sanitized: list[dict[str, Any]] = []
        text_lines = [
            f"Prompt Context Debug",
            f"session_prefix: {session_cfg.get('session_prefix')}",
            f"round: {round_num}",
            f"turn: {turn_num}",
            f"ai_role: {ai_role}",
            f"message_count: {len(messages)}",
            "",
            "=" * 80,
        ]
        image_counter = 0

        for idx, msg in enumerate(messages, start=1):
            role = msg.get("role", "unknown")
            content = msg.get("content")
            out_msg: dict[str, Any] = {"role": role}
            text_lines += ["", f"MESSAGE {idx}: role={role}", "-" * 40]

            if isinstance(content, str):
                out_msg["content"] = content
                text_lines.append(content)
            elif isinstance(content, list):
                out_parts: list[dict[str, Any]] = []
                for part_idx, part in enumerate(content, start=1):
                    if not isinstance(part, dict):
                        out_parts.append({"type": "unknown", "value": repr(part)})
                        text_lines.append(f"[part {part_idx}] {repr(part)}")
                        continue

                    part_type = part.get("type")
                    if part_type == "text":
                        text = part.get("text", "")
                        out_parts.append({"type": "text", "text": text})
                        text_lines.append(f"[text part {part_idx}]")
                        text_lines.append(str(text))
                    elif part_type == "image_url":
                        image_counter += 1
                        url = (part.get("image_url") or {}).get("url", "")
                        image_bytes, ext = _decode_data_url(url)
                        image_name = f"message_{idx:02d}_part_{part_idx:02d}_image_{image_counter:02d}.{ext}"
                        image_path = os.path.join(dump_dir, image_name)
                        if image_bytes is not None:
                            with open(image_path, "wb") as f:
                                f.write(image_bytes)
                            image_ref = image_name
                        else:
                            image_ref = "<non-data image url>"
                        out_parts.append({
                            "type": "image_url",
                            "saved_as": image_ref,
                            "source_length": len(url) if isinstance(url, str) else None,
                        })
                        text_lines.append(f"[image part {part_idx}] {image_ref}")
                    else:
                        out_parts.append(part)
                        text_lines.append(f"[part {part_idx}: {part_type}] {json.dumps(part, ensure_ascii=False)}")
                out_msg["content"] = out_parts
            else:
                out_msg["content"] = repr(content)
                text_lines.append(repr(content))

            sanitized.append(out_msg)

        with open(os.path.join(dump_dir, "context.json"), "w", encoding="utf-8") as f:
            json.dump({
                "session_prefix": session_cfg.get("session_prefix"),
                "round_number": round_num,
                "turn_number": turn_num,
                "ai_role": ai_role,
                "message_count": len(messages),
                "messages": sanitized,
            }, f, indent=2, ensure_ascii=False)
        with open(os.path.join(dump_dir, "context.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(text_lines))
    except Exception:
        # Debug dumps must never affect experiment execution.
        pass


def generate_ai_vs_ai_reply(
    player: Player,
    role: str,
    latest_message: str | None,
) -> dict[str, Any] | None:
    """Generate an AI reply for a specific role in AI vs AI mode.

    Prompt construction receives the generated AI role explicitly; no player
    role mutation is required for AI-vs-AI sessions.
    
    Supports multiple AI providers (OpenAI, Gemini) per role via session config:
    - ai_director_model / ai_director_provider: Director's model config
    - ai_matcher_model / ai_matcher_provider: Matcher's model config

    Args:
        player: Lightweight Player object used for group/session context
        role: Either "director" or "matcher" - the AI role to generate for
        latest_message: The most recent message from the other AI, or None for start

    Returns:
        A dict with "text" (utterance) and "selection" (for matcher) fields,
        or None on failure.
    """
    import logging
    from .ai_providers import (
        get_model_config_for_role,
        call_ai_api,
        has_ai_client_for_role,
    )

    if role not in ("director", "matcher"):
        logging.error("[AI_VS_AI] Invalid role: %s", role)
        return None

    # Get role-specific model configuration
    model_config = get_model_config_for_role(player, role)
    
    if not has_ai_client_for_role(player, role):
        logging.warning("[AI_VS_AI] No AI client available for %s (provider=%s)", 
                       role, model_config.get("provider"))
        return None

    try:
        # Build chat history from group's AI messages
        human_msgs = []
        ai_msgs = []

        # In AI vs AI mode, both sets of messages are in ai_messages
        # We need to load messages and attribute them to the correct role
        try:
            all_ai_msgs = json.loads(player.group.ai_messages or "[]")
        except Exception:
            all_ai_msgs = []

        # Get current round info
        current_round = getattr(player, "round_number", 1) or 1

        # Cross-round history is a universal session-level switch. When enabled,
        # both Director and Matcher receive the same prior text history and
        # submitted-grid feedback images. When disabled, both roles are scoped
        # to the current round only.
        use_cross_round_history = _cross_round_history_enabled(player)

        feedback_msgs = []
        if use_cross_round_history:
            try:
                all_round_players = player.in_all_rounds()
            except Exception:
                all_round_players = [player]

            for p_round in all_round_players:
                round_num = getattr(p_round, "round_number", None)
                try:
                    round_msgs = json.loads(p_round.group.ai_messages or "[]")
                except Exception:
                    round_msgs = []

                for m in round_msgs:
                    if isinstance(m, dict):
                        m_copy = dict(m)
                        if round_num is not None and "round_number" not in m_copy:
                            m_copy["round_number"] = round_num
                        ai_msgs.append(m_copy)

                # Add lightweight text feedback for completed rounds. Submitted-grid
                # feedback images are injected later next to each prior round's
                # transcript, behind the same cross_round_history switch.
                if round_num is not None and round_num < current_round:
                    correct_count = _compute_round_correct_count(p_round)
                    if correct_count is not None:
                        feedback_msgs.append({
                            "text": (
                                f"[ROUND {round_num} COMPLETE: {correct_count}/12 correct. "
                                f"NOTE: The same physical baskets will appear again, but they have been RESHUFFLED for the next round. "
                                f"Old position numbers and candidate numbers no longer correspond to the same places. "
                                f"Carry forward basket labels, correct matches, and wrong-match feedback, but describe/select using the current image.]"
                            ),
                            "sender_role": "system",
                            "round_number": round_num,
                            "is_feedback": True,
                        })
        else:
            ai_msgs = all_ai_msgs

        # Merge and sort all history
        all_history = ai_msgs + feedback_msgs
        all_history.sort(
            key=lambda m: (
                m.get("round_number") or 0,
                1 if m.get("is_feedback") else 0,
                m.get("server_ts") or "",
                m.get("timestamp") or "",
            )
        )

        # Determine prompt strategy
        strategy_name = _get_prompt_strategy_name(player)

        original_role = player.field_maybe_none("player_role")

        from .prompts import acl as ACL_prompt, cameron as cameron_module

        if _is_acl_prompt_strategy(strategy_name):
            chat_messages = ACL_prompt.build_acl_prompt_messages(
                player, latest_message, all_history, ai_role=role
            )
            use_structured_json = True
        elif strategy_name == "cameron-prompt":
            chat_messages = cameron_module.build_cameron_prompt_messages(
                player, latest_message, all_history, ai_role=role
            )
            use_structured_json = False
        else:
            raise ValueError(
                f"Unknown prompt_strategy {strategy_name!r}; expected 'ACL_prompt' or 'cameron-prompt'."
            )

        # Inject task background
        chat_messages = _inject_task_background(chat_messages)

        if use_cross_round_history:
            chat_messages = _inject_round_feedback_context(
                player, chat_messages, ai_role=role
            )

        # Inject current-round visual context after historical context so the
        # active grid is the freshest visual frame before current dialogue.
        chat_messages = _inject_visual_grid_context(player, chat_messages, ai_role=role)

        # For Director at round start, add explicit start message
        if role == "director" and latest_message is None:
            chat_messages.append({
                "role": "user",
                "content": (
                    f"═══ START OF ROUND {current_round} ═══\n"
                    f"This is a NEW round with the same basket set in a COMPLETELY DIFFERENT ORDER.\n"
                    f"IMPORTANT: The basket at position 1 in Round {current_round} is not necessarily the same basket "
                    f"that was at position 1 in previous rounds. All positions have been reshuffled.\n"
                    f"You may reuse successful shared names for the same physical baskets, but describe the current positions from the new image.\n"
                    f"Look at the image labeled 'ROUND {current_round} TARGET SEQUENCE' to see the ACTUAL baskets for this round.\n"
                    f"Please describe ONLY Basket 1 (top-left in the grid) for now. "
                    f"Do NOT describe multiple baskets - just Basket 1. Wait for a response before moving to Basket 2."
                )
            })

        # For Matcher, inject a single human-readable sequence state block.
        # The old approach dumped raw JSON (opaque image paths) that the model
        # couldn't act on, causing it to ignore filled positions and re-ask for
        # descriptions of already-placed baskets. This unified block is plain
        # English and explicitly lists what is done vs. what still needs work.
        if role == "matcher":
            try:
                seq_state = _build_matcher_current_sequence_state_for_prompt(player)
                slots = seq_state.get("sequence_slots", [])
                filled_positions = sorted(
                    int(s["position"])
                    for s in slots
                    if isinstance(s, dict) and s.get("image") and s.get("position") is not None
                )
                empty_positions = sorted(
                    int(s["position"])
                    for s in slots
                    if isinstance(s, dict) and not s.get("image") and s.get("position") is not None
                )

                # Pending refills: positions the dialogue completed but a basket
                # move left empty on the board.
                try:
                    pending_refills = _get_pending_refill_positions(player)
                except Exception:
                    pending_refills = []

                filled_str = ", ".join(str(p) for p in filled_positions) if filled_positions else "none yet"
                empty_str  = ", ".join(str(p) for p in empty_positions)  if empty_positions  else "NONE — all filled!"

                lines = [
                    "=== CURRENT BOARD STATE ===",
                    f"ALREADY FILLED ({len(filled_positions)}/12): {filled_str}",
                    f"  → Do NOT select a candidate that is already filled in a different position unless you explicitly say you are correcting/moving it.",
                    # f"  → Do NOT say 'Go to Basket X' for any position in the ALREADY FILLED list.",
                    f"STILL EMPTY ({len(empty_positions)}/12 remaining): {empty_str}",
                ]

                filled_details = []
                used_candidates = []
                for slot in slots:
                    if not isinstance(slot, dict):
                        continue
                    candidate_index = slot.get("candidate_index")
                    image = slot.get("image")
                    position = slot.get("position")
                    if candidate_index is None or not image or position is None:
                        continue
                    try:
                        pos_int = int(position)
                        cand_int = int(candidate_index)
                    except Exception:
                        continue
                    filled_details.append(
                        f"  - position {pos_int}: candidate {cand_int} ({os.path.basename(str(image))})"
                    )
                    used_candidates.append(cand_int)
                if filled_details:
                    lines.append("FILLED SLOT DETAILS:")
                    lines.extend(filled_details)
                if used_candidates:
                    used_str = ", ".join(str(c) for c in sorted(set(used_candidates)))
                    lines.append(f"USED CANDIDATES: {used_str}")
                    if empty_positions:
                        lines.append(
                            "For the next empty position, pick a candidate NOT in USED CANDIDATES unless you are intentionally correcting a previous slot."
                        )

                if pending_refills:
                    refill_str = ", ".join(str(p) for p in pending_refills)
                    lines += [
                        f"PENDING REFILL (basket was moved, now needs re-description): {refill_str}",
                        f"  → After placing the CURRENT basket, ask the Director to re-describe"
                        f" the LOWEST-NUMBERED pending refill position in natural language.",
                        f"  → Example: 'Placed it. Before we move on, can you remind me of Basket {pending_refills[0]}?'",
                        f"  → Do not mention 'hidden state', 'system notices', or internal bookkeeping.",
                    ]

                if not empty_positions:
                    lines.append("ALL 12 POSITIONS ARE FILLED — set ready_to_submit to true now.")

                lines.append("===========================")
                seq_state_text = "\n".join(lines)

                insert_idx = 0
                while (
                    insert_idx < len(chat_messages)
                    and isinstance(chat_messages[insert_idx], dict)
                    and _is_instruction_message(chat_messages[insert_idx])
                ):
                    insert_idx += 1
                chat_messages.insert(insert_idx, {"role": "developer", "content": seq_state_text})
            except Exception:
                pass

            if use_cross_round_history:
                try:
                    feedback_memory = _build_cross_round_matcher_feedback_memory(player)
                except Exception:
                    feedback_memory = None
                if feedback_memory:
                    insert_idx = 0
                    while (
                        insert_idx < len(chat_messages)
                        and isinstance(chat_messages[insert_idx], dict)
                        and _is_instruction_message(chat_messages[insert_idx])
                    ):
                        insert_idx += 1
                    chat_messages.insert(
                        insert_idx,
                        {"role": "developer", "content": feedback_memory},
                    )

            # Add JSON format instruction for matcher
            if not use_structured_json:
                # Get current sequence state to show which positions are empty
                try:
                    seq = json.loads(player.group.ai_partial_sequence or "[]")
                    filled_positions = set()
                    for item in seq:
                        if isinstance(item, dict):
                            try:
                                p = int(item.get("position"))
                                if 1 <= p <= 12 and item.get("image"):
                                    filled_positions.add(p)
                            except Exception:
                                pass
                    empty_positions = [p for p in range(1, 13) if p not in filled_positions]
                    filled_count = 12 - len(empty_positions)
                    empty_str = ", ".join(str(p) for p in empty_positions) if empty_positions else "NONE - all filled!"
                except Exception:
                    filled_count = 0
                    empty_str = "unknown"

                matcher_instr = (
                    f"CRITICAL: Empty positions that still need baskets: [{empty_str}]\n"
                    f"You have {filled_count}/12 positions filled.\n\n"
                    "If the Director just answered your clarification for the LOWEST empty position, "
                    "choose the matching candidate now and set `selection.position` to that empty position. "
                    "Do not ask for a later basket until the current lowest empty position has a concrete selection.\n\n"
                    "You MUST respond with valid JSON:\n"
                    "{\n"
                    '  "utterance": "<your response to show in chat>",\n'
                    '  "selection": {\n'
                    '    "candidate_index": <1-18 or null if no selection this turn>,\n'
                    '    "position": <1-12 or null if no selection this turn>,\n'
                    '    "ready_to_submit": <true ONLY when ALL 12 positions are filled, false otherwise>\n'
                    "  }\n"
                    "}\n\n"
                    "IMPORTANT: If there are empty positions, work on the LOWEST-NUMBERED empty position first. "
                    "Only ask the Director about a later basket after you have selected the current one. "
                    "Do NOT set ready_to_submit to true until all 12 positions have baskets."
                )
                insert_idx = 0
                while (
                    insert_idx < len(chat_messages)
                    and _is_instruction_message(chat_messages[insert_idx])
                ):
                    insert_idx += 1
                chat_messages.insert(insert_idx, {"role": "developer", "content": matcher_instr})

        if _prompt_context_debug_enabled(player):
            _dump_prompt_context(player, role, chat_messages)

        # Make API call using multi-provider system
        base_temperature = 0
        response_format = None
        # Use JSON format for matcher always, and for director when using ACL_prompt
        if _is_acl_prompt_strategy(strategy_name):
            from .prompts.acl import get_acl_response_schema
            response_format = {
                "type": "json_schema",
                "json_schema": get_acl_response_schema(role),
            }
        elif strategy_name == "cameron-prompt":
            from .prompts.cameron import DirectorResponse, MatcherResponse
            schema_class = MatcherResponse if role == "matcher" else DirectorResponse
            response_format = {
                "type": "json_schema",
                "json_schema": schema_class
            }
        elif role == "matcher" or use_structured_json:
            response_format = {"type": "json_object"}

        provider = model_config.get("provider", "openai")
        model = model_config.get("model", "gpt-5.2")
        
        # logging.info("[AI_VS_AI] Generating %s reply using %s/%s, %d messages in context", 
                    # role.upper(), provider.upper(), model, len(chat_messages))

        # Use the multi-provider call_ai_api function
        text = _api_call_with_retry(
            lambda: call_ai_api(
                messages=chat_messages,
                model_config=model_config,
                temperature=base_temperature,
                response_format=response_format,
            )
        )
        
        if text is None:
            logging.warning("[AI_VS_AI] API call returned None for %s", role)
            return None
        
        text = text.strip()

        # Parse response
        utterance = None
        selection = None

        if role == "matcher":
            # Parse JSON response
            try:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end > start:
                    json_str = text[start:end]
                    try:
                        data = json.loads(json_str)
                    except Exception:
                        import re
                        first_json_match = re.match(r"^(\{.*?\})\s*\{", json_str, re.DOTALL)
                        if first_json_match:
                            data = json.loads(first_json_match.group(1))
                        else:
                            raise
                    utterance = _clean_ai_utterance(data.get("utterance"))

                    if data.get("reasoning") is not None:
                        import logging
                        reasoning_json = json.dumps(data.get("reasoning"), indent=2)
                        logging.info(f"[AI_REASONING] Structured reasoning for {role.upper()}:\n{reasoning_json}")

                    sel = data.get("selection")
                    if isinstance(sel, dict):
                        cand_raw = sel.get("candidate_index")
                        pos = sel.get("position")
                        try:
                            cand_int = int(cand_raw) if cand_raw is not None else None
                        except Exception:
                            cand_int = None
                        try:
                            pos_int = int(pos) if pos is not None else None
                        except Exception:
                            pos_int = None
                        ready = _coerce_ready_to_submit(sel.get("ready_to_submit"))
                        selection = {
                            "candidate_index": cand_int,
                            "position": pos_int,
                            "ready_to_submit": ready,
                        }
            except Exception as e:
                logging.warning("[AI_VS_AI] Failed to parse matcher JSON: %s", e)
                utterance = _clean_ai_utterance(text)
        else:
            # Director response - parse JSON if using a structured prompt, otherwise plain text
            if use_structured_json or strategy_name == "cameron-prompt":
                try:
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    if start != -1 and end > start:
                        json_str = text[start:end]
                        try:
                            data = json.loads(json_str)
                        except Exception:
                            import re
                            first_json_match = re.match(r"^(\{.*?\})\s*\{", json_str, re.DOTALL)
                            if first_json_match:
                                data = json.loads(first_json_match.group(1))
                            else:
                                raise
                        utterance = _clean_ai_utterance(data.get("utterance"))
                        
                        if data.get("reasoning") is not None:
                            import logging
                            reasoning_json = json.dumps(data.get("reasoning"), indent=2)
                            logging.info(f"[AI_REASONING] Structured reasoning for {role.upper()}:\n{reasoning_json}")
                                
                        if not utterance:
                            logging.warning(
                                "[AI_VS_AI] Director structured response had empty/null utterance; dropping response: %s",
                                text[:200],
                            )
                    else:
                        utterance = _clean_ai_utterance(text)
                except Exception as e:
                    logging.warning("[AI_VS_AI] Failed to parse director structured JSON: %s", e)
                    utterance = _clean_ai_utterance(text)
            else:
                utterance = _clean_ai_utterance(text)

        if not utterance:
            logging.warning("[AI_VS_AI] %s response had no usable utterance; not sending a chat message", role.upper())
            return None

        logging.info("[AI_VS_AI] %s response: %s", role.upper(), utterance[:100])
        
        # Always log the AI's intermediate response, regardless of strategy
        if hasattr(player, "group"):
            try:
                try:
                    existing = json.loads(getattr(player.group, "ai_reasoning_log", "[]") or "[]")
                except Exception:
                    existing = []
                if not isinstance(existing, list):
                    existing = []
                
                # Fetch reasoning if it was successfully parsed
                reasoning_obj = locals().get("data", {}).get("reasoning") if isinstance(locals().get("data"), dict) else None
                # Fetch CG extractor output stored during prompt building (if available).
                # This captures the actual CG agent's structured output rather than
                # trying to read from the conversational LLM's response (which never
                # includes a "common_ground" field).
                common_ground_obj = getattr(player, "_last_cg_extractor_output", None)
                
                existing.append({
                    "round_number": getattr(player, "round_number", None),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "strategy_name": strategy_name,
                    "human_role": original_role or "observer",
                    "ai_role": role,
                    "reasoning": reasoning_obj,
                    "common_ground": common_ground_obj,
                    "utterance": utterance,
                    "raw_text": text,
                    "selection": selection
                })
                player.group.ai_reasoning_log = json.dumps(existing, ensure_ascii=False)
            except Exception:
                pass
        return {"text": utterance, "selection": selection}

    except Exception as e:
        import traceback
        logging.error(
            "[AI_VS_AI] Error generating %s reply: %s: %s\n%s",
            role, type(e).__name__, e, traceback.format_exc()
        )
        return None


def run_ai_vs_ai_turn(player: Player) -> dict[str, Any]:
    """Execute one turn of AI vs AI dialogue.

    This function determines whose turn it is (Director or Matcher),
    generates the appropriate response, stores it, and updates game state.

    Returns a dict with:
        - turn_number: The turn number (1-indexed)
        - speaker: "director" or "matcher"
        - message: The generated message text
        - selection: Matcher's selection (if applicable)
        - is_complete: Whether the round is complete (all 12 matched)
        - error: Error message if something went wrong
    """
    import logging

    if not is_ai_vs_ai_session(player):
        return {"error": "Not an AI vs AI session"}

    try:
        # Load current messages to determine turn order
        try:
            messages = json.loads(player.group.ai_messages or "[]")
        except Exception:
            messages = []

        turn_number = len(messages) + 1

        # Determine speaker: Director starts, then they alternate
        # But in practice: Director describes, Matcher responds (confirms or asks)
        # If last message was from Director, it's Matcher's turn (and vice versa)
        if not messages:
            speaker = "director"
            latest_message = None
        else:
            last_msg = messages[-1]
            last_speaker = last_msg.get("sender_role", "director")
            speaker = "matcher" if last_speaker == "director" else "director"
            latest_message = last_msg.get("text")

        # Generate the AI response
        reply = generate_ai_vs_ai_reply(player, speaker, latest_message)

        if reply is None:
            return {
                "turn_number": turn_number,
                "speaker": speaker,
                "error": "Failed to generate AI response",
            }

        utterance = reply.get("text")
        selection = reply.get("selection")

        if not utterance:
            return {
                "turn_number": turn_number,
                "speaker": speaker,
                "error": "AI generated empty response",
            }

        # Store the message
        now_iso = datetime.datetime.now().isoformat()
        new_message = {
            "text": utterance,
            "timestamp": now_iso,
            "sender_role": speaker,
            "server_ts": now_iso,
        }
        messages.append(new_message)
        player.group.ai_messages = json.dumps(messages)

        # If Matcher made a selection, update the partial sequence
        is_complete = False
        if speaker == "matcher" and selection:
            allow_move = _utterance_explicitly_moves_selection(utterance)
            _, vacated_position = _update_ai_partial_sequence(
                player,
                selection,
                allow_move=allow_move,
            )
            if vacated_position is not None:
                logging.info(
                    "[AI_VS_AI] Matcher moved basket — position %d is now vacant",
                    vacated_position,
                )

        def _submit_complete_sequence() -> bool:
            nonlocal is_complete
            try:
                sequence = json.loads(player.group.ai_partial_sequence or "[]")
                by_pos = _filled_sequence_by_position(sequence)
                filled_count = len(by_pos)
                missing_positions = [p for p in range(1, 13) if p not in by_pos]

                if missing_positions:
                    logging.info(
                        "[AI_VS_AI] Ignoring submit intent: %d/12 filled, missing positions: %s",
                        filled_count,
                        missing_positions,
                    )
                    return False

                is_complete = True
                player.group.matcher_sequence = json.dumps([by_pos[p] for p in range(1, 13)])

                try:
                    shared_grid = json.loads(player.group.shared_grid or "[]")
                    correct_count = 0
                    for pos in range(1, 13):
                        correct_img = shared_grid[pos - 1].get("image") if pos - 1 < len(shared_grid) else None
                        submitted_img = by_pos.get(pos, {}).get("image")
                        if correct_img and submitted_img and correct_img == submitted_img:
                            correct_count += 1
                    player.sequence_accuracy = (correct_count / 12) * 100
                    player.task_completed = True
                    player.completion_time = now_iso
                    logging.info(
                        "[AI_VS_AI] Submitted complete sequence: 12/12 filled, Accuracy: %.1f%%",
                        player.sequence_accuracy,
                    )
                except Exception as e:
                    logging.warning("[AI_VS_AI] Failed to calculate accuracy: %s", e)
                return True
            except Exception as e:
                logging.warning("[AI_VS_AI] Failed to submit sequence: %s", e)
                return False

        # Check if matcher signaled ready to submit via JSON
        if speaker == "matcher" and selection and selection.get("ready_to_submit"):
            _submit_complete_sequence()

        # Regex fallback: detect "submit" intent in matcher's utterance even if JSON didn't have ready_to_submit
        # This catches cases where the matcher says "I'm ready to submit" but didn't set the flag
        if not is_complete and speaker == "matcher" and utterance:
            import re
            submit_patterns = [
                r"ready to submit",
                r"submitting (the |my )?sequence",
                r"submit (the |my )?(final |complete )?sequence",
                r"i('ll| will) submit",
                r"going to submit",
                r"let me submit",
                r"all \d+ (positions?|baskets?) (are |have been )?(filled|placed|complete)",
                r"sequence is complete",
                r"completed? (the |my )?sequence",
            ]
            combined_pattern = "|".join(f"({p})" for p in submit_patterns)
            if re.search(combined_pattern, utterance.lower()):
                logging.info("[AI_VS_AI] Detected submit intent in utterance: %s", utterance[:80])
                _submit_complete_sequence()

        return {
            "turn_number": turn_number,
            "speaker": speaker,
            "message": utterance,
            "selection": selection,
            "is_complete": is_complete,
        }

    except Exception as e:
        import traceback
        logging.error(
            "[AI_VS_AI] Error in turn: %s: %s\n%s",
            type(e).__name__, e, traceback.format_exc()
        )
        return {"error": str(e)}


def get_ai_vs_ai_status(player: Player) -> dict[str, Any]:
    """Get the current status of an AI vs AI game.

    Returns a dict with:
        - round_number: Current round
        - turn_count: Number of turns so far
        - messages: List of all messages
        - partial_sequence: Current matcher sequence state
        - is_complete: Whether the round is finished
        - accuracy: Accuracy if complete
    """
    try:
        messages = json.loads(player.group.ai_messages or "[]")
    except Exception:
        messages = []

    try:
        partial_sequence = json.loads(player.group.ai_partial_sequence or "[]")
    except Exception:
        partial_sequence = []

    filled_count = len(_filled_sequence_by_position(partial_sequence))

    # Use field_maybe_none() for fields that might be None
    task_completed = player.field_maybe_none("task_completed") or False
    sequence_accuracy = player.field_maybe_none("sequence_accuracy")
    try:
        total_rounds = int(
            player.session.config.get("num_rounds") or Constants.num_rounds
        )
    except Exception:
        total_rounds = Constants.num_rounds

    return {
        "round_number": getattr(player, "round_number", 1),
        "total_rounds": total_rounds,
        "turn_count": len(messages),
        "messages": messages,
        "partial_sequence": partial_sequence,
        "filled_count": filled_count,
        "is_complete": task_completed,
        "accuracy": sequence_accuracy,
    }
