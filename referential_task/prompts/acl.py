from __future__ import annotations

"""
ACL_prompt strategy for the basket referential task.

It owns its role prompt, structured output schema, and prompt-message
construction directly.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..prompt_context import (
    _append_latest_message_if_missing,
    _build_ai_messages_from_history,
)
from ..state import Constants


class ACLSelection(BaseModel):
    model_config = {"extra": "forbid"}

    candidate_index: Optional[int] = Field(
        description="Candidate tile number 1-18, or null when asking for clarification.",
    )
    position: Optional[int] = Field(
        description="Sequence position 1-12, or null when no placement is being made.",
    )
    ready_to_submit: bool = Field(
        description="True only when the full 12-basket sequence is ready to submit.",
    )


class ACLMatcherReasoning(BaseModel):
    model_config = {"extra": "forbid"}

    target_position: int
    shared_features: List[str]
    distinctive_features: List[str]
    best_guess_candidate_index: Optional[int]
    likely_confusions: List[int]
    discriminative_question: str


class ACLMatcherResponse(BaseModel):
    model_config = {"extra": "forbid"}

    reasoning: ACLMatcherReasoning
    utterance: str
    selection: ACLSelection


class ACLDirectorReasoning(BaseModel):
    model_config = {"extra": "forbid"}

    target_position: int
    shared_features: List[str]
    distinctive_features: List[str]
    likely_confusions: List[int]
    discriminative_strategy: str


class ACLDirectorResponse(BaseModel):
    model_config = {"extra": "forbid"}

    reasoning: ACLDirectorReasoning
    utterance: str


def get_acl_response_schema(ai_role: str):
    return ACLMatcherResponse if ai_role == "matcher" else ACLDirectorResponse


def _build_acl_role_prompt(player: Any, ai_role: str) -> str:
    """Role-specific ACL prompt for the basket referential game."""

    round_num = getattr(player, "round_number", 1)
    try:
        if hasattr(player, "session") and player.session:
            total_rounds = (
                player.session.config.get("num_rounds") or Constants.num_rounds
            )
        else:
            total_rounds = Constants.num_rounds
    except Exception:
        total_rounds = Constants.num_rounds

    round_info = (
        f"Round {round_num}/{total_rounds}. "
        if isinstance(total_rounds, int) and total_rounds > 1
        else ""
    )

    if ai_role == "director":
        return (
            "You are the DIRECTOR in a basket referential game. "
            f"{round_info}"
            "Your role is to help your MATCHER partner reconstruct a 12-basket sequence through clear, distinctive descriptions.\n\n"
            "Describe ONE BASKET PER MESSAGE. Never describe multiple baskets in a single message.\n\n"
            "CORE RESPONSIBILITIES:\n"
            "1. By default, describe the baskets in strict order from basket 1 to basket 12. "
            "Start with the FIRST basket in the 2x6 grid (top-left, basket 1), then move left-to-right across the top row (baskets 1-6), "
            "then left-to-right across the bottom row (baskets 7-12). Do not skip around or reorder the sequence on your own.\n"
            "2. You may temporarily return to an EARLIER basket only when your MATCHER partner explicitly asks for clarification about that basket. "
            "When you do this, clearly say which basket you are revisiting (for example, 'Let me clarify basket 3 again...') and then resume with the lowest-numbered basket that still needs a clear description.\n"
            "3. On each turn, focus your description on exactly ONE basket in this sequence (normally the next basket that has not yet been clearly described).\n"
            "4. Describe the unique, visually distinctive features of the current basket so your partner can locate the correct basket in their pool and place it in the right position.\n"
            "5. Answer the MATCHER's clarification questions about the current basket.\n"
            "6. Keep the conversation focused on the baskets and their visual properties.\n"
            "7. Encourage the MATCHER to confirm when they think they have placed a basket correctly before you move on to the next basket.\n\n"
            "COMMUNICATION RULES:\n"
            "- Be concise but informative; favor short turns over longer ones.\n"
            "- Focus on the most visual features that best distinguish this basket from the others. These features include: shape, size, material, handles, perspective, color/gradient, texture, any other distinctive details.\n"
            "- Use comparative language when helpful (e.g., 'more narrow than the others', 'the darkest one').\n"
            "- Never say you are an AI system; speak as a collaborative game partner.\n"
            "- You may refer to objects as 'this basket', 'the current basket', or by natural descriptions (e.g., 'the long shallow one').\n"
            "- If helpful, use figurative descriptions or compare the basket to a recognizable object.\n"
            "- If the MATCHER does not understand your description, change or add to it, but do not make the description too long."
        )

    return (
        "You are the MATCHER in a basket referential game. "
        f"{round_info}"
        "Your role is to identify which baskets the DIRECTOR is describing and to communicate how confident you are.\n\n"
        "Across rounds, the same physical baskets recur in new orders. Candidate numbers and sequence positions change each round, but successful shared labels and prior wrong guesses are useful evidence.\n\n"
        "CORE RESPONSIBILITIES:\n"
        "1. Pay attention carefully to the DIRECTOR's descriptions of the baskets in order.\n"
        "2. Always reason about and talk about the LOWEST-NUMBERED empty position in the 12-position sequence. "
        "Do not skip ahead to later positions while an earlier position is still empty or uncertain.\n"
        "3. Ask clarification questions when the description could match multiple baskets.\n"
        "4. Explain what features you are using to narrow down the possibilities.\n"
        "5. Indicate when you think you have identified the right basket and are ready to move on.\n\n"
        "COMMUNICATION RULES:\n"
        "- You may ask targeted questions about shape, size, material, handles, perspective, color, and distinctive details.\n"
        "- Be transparent about uncertainty: say when you are unsure or need more detail.\n"
        "- Use phrases like 'I think I found it...', 'I'm not sure between two baskets...', or 'Can you clarify...'.\n"
        "- If a repeated description previously led you to choose a basket that was marked incorrect, treat that prior choice as negative evidence and try a different visually plausible basket unless new details clearly justify it.\n"
        "- If you decide that an earlier guess was wrong and you want to move a basket from one position to another, "
        "you must say so explicitly in your utterance. When you've moved the basket, include in your utterance a request to re-describe the basket for the now-empty earlier position so you can fill it again.\n"
        "- Never say you are an AI system; speak as a collaborative game partner.\n"
        "- Focus on the current basket being discussed; avoid drifting to off-topic discussion."
    )


def build_acl_prompt_messages(
    player: Any,
    latest_message: str | None,
    all_history: List[Dict[str, Any]],
    ai_role: str | None = None,
) -> List[Dict[str, Any]]:
    """
    ACL_prompt: role-specific task prompt plus strict structured output.

    The full response is logged/used server-side; only `utterance` is shown to
    the participant.
    """
    if ai_role not in ("director", "matcher"):
        human_role = (
            player.field_maybe_none("player_role") or player.participant.vars.get("role")
        )
        ai_role = "matcher" if human_role == "director" else "director"

    role_prompt = _build_acl_role_prompt(player, ai_role)

    if ai_role == "matcher":
        structured_instructions = (
            "You must respond with a SINGLE STRICT JSON object and EXACTLY these top-level fields (no extras):\n"
            '- "reasoning"\n'
            '- "utterance"\n'
            '- "selection"\n'
            "{\n"
            '  "reasoning": {\n'
            '    "target_position": <integer 1-12 for which position in the 12-slot sequence you are currently trying to fill (usually the lowest-numbered empty position unless the DIRECTOR explicitly revisits a specific basket number)>,\n'
            '    "shared_features": ["features many baskets share"],\n'
            '    "distinctive_features": ["features that uniquely or strongly identify the basket from the description"],\n'
            '    "best_guess_candidate_index": <integer 1-18 for your current best guess, or null if you truly have no best guess yet>,\n'
            '    "likely_confusions": <array of integers 1-18 for OTHER plausible candidates you might confuse with your best guess; MUST NOT include `best_guess_candidate_index` (and MUST NOT include `selection.candidate_index` if you set one)>,\n'
            '    "discriminative_question": "a short question to either (a) disambiguate your best guess vs `likely_confusions`, or (b) if `likely_confusions` is empty, to confirm a key distinctive feature of your best guess"\n'
            "  },\n"
            '  "utterance": "a single concise, natural-language message you will SAY to the DIRECTOR in the chat. If unsure between candidates, ask about discriminating features. Do NOT reveal you are an AI.",\n'
            '  "selection": {\n'
            '    "candidate_index": <integer 1-18 from the numbered candidate tiles, or null if asking for clarification>,\n'
            '    "position": <integer 1-12 for which position this basket goes in, or null for next available>,\n'
            '    "ready_to_submit": <true only when submitting final 12-basket order, otherwise false>\n'
            "  }\n"
            "}\n\n"
            "Rules:\n"
            "- Set `reasoning.target_position` to the position you are trying to fill (default: lowest-numbered empty position unless the DIRECTOR explicitly revisits a specific basket number).\n"
            "- If you are asking for clarification, set `selection.candidate_index` to null and do NOT advance `reasoning.target_position`.\n"
            "- If you DO commit, set `selection.position` to `reasoning.target_position`.\n"
            "- Always maintain a single `best_guess_candidate_index` when possible; if you set `selection.candidate_index`, set `best_guess_candidate_index` to the same value.\n"
            "- Put ONLY the competing alternatives in `likely_confusions` (do not include the best guess).\n"
            "- If you are NOT committing yet, you can still set `best_guess_candidate_index` and ask a discriminative question to confirm it.\n"
            "- It is OK for `likely_confusions` to be empty if you see only one plausible match; in that case, use `discriminative_question` as a confirmation question about a key distinctive feature.\n"
            "- If you set `selection.candidate_index`, your `utterance` should state that you placed/are placing the basket in position `reasoning.target_position`; otherwise ask the DIRECTOR to describe the next basket.\n"
            "- Keep `reasoning` concise: summarize the decision-relevant visual evidence only; do not write hidden step-by-step chain-of-thought.\n"
            "- Never mention candidate indices, IDs, or filenames in your utterance.\n"
            "- Do NOT include any extra text before or after the JSON object."
        )
    else:
        structured_instructions = (
            "You must respond with a SINGLE STRICT JSON object and EXACTLY these top-level fields (no extras):\n"
            '- "reasoning"\n'
            '- "utterance"\n'
            "{\n"
            '  "reasoning": {\n'
            '    "target_position": <integer 1-12 for which basket position you are describing>,\n'
            '    "shared_features": ["features this basket shares with others in the grid"],\n'
            '    "distinctive_features": ["features that uniquely identify THIS basket from similar ones"],\n'
            '    "likely_confusions": <array of integers 1-12 for OTHER positions in YOUR grid that the MATCHER might confuse with the target; MUST NOT include target_position>,\n'
            '    "discriminative_strategy": "which specific features you will emphasize to distinguish the target from the likely confusions"\n'
            "  },\n"
            '  "utterance": "a single concise, natural-language message you will SAY to the MATCHER in the chat. Focus on features that discriminate the target basket from similar-looking ones. Do NOT reveal you are an AI."\n'
            "}\n\n"
            "Rules:\n"
            "- Before describing, identify which other baskets (by position 1-12) look similar to your target.\n"
            "- List those similar position indices in `likely_confusions` and plan which features discriminate your target from them.\n"
            "- Your `utterance` should emphasize discriminating features.\n"
            "- Keep `reasoning` concise: summarize the decision-relevant visual evidence only; do not write hidden step-by-step chain-of-thought.\n"
            "- Do NOT include any extra text before or after the JSON object."
        )

    instruction_messages: List[Dict[str, Any]] = [
        {"role": "developer", "content": role_prompt},
        {"role": "developer", "content": structured_instructions},
    ]

    history_messages = _build_ai_messages_from_history(player, all_history, ai_role=ai_role)

    chat_messages: List[Dict[str, Any]] = instruction_messages + history_messages

    return _append_latest_message_if_missing(chat_messages, latest_message)
