import json

import referential_task.ai_vs_ai as ai_vs_ai
from referential_task.ai_vs_ai import (
    _apply_matcher_selection,
    _action_feedback_notice,
    _record_matcher_outcome,
    get_ai_vs_ai_status,
    run_ai_vs_ai_turn,
)
from tests.test_sequence import candidate_index_for, make_round2_player


def player_with_one_basket():
    """009 is placed in position 11, its correct slot in this grid."""
    player = make_round2_player()
    player.session.config["ai_vs_ai_mode"] = True
    player.group.ai_partial_sequence = json.dumps([
        {"position": 11, "image": "images/009.png", "originalPosition": "33"},
    ])
    player.group.ai_reasoning_log = json.dumps([{"ai_role": "matcher", "utterance": "placing it"}])
    return player


def test_already_placed_candidate_without_move_flag_is_rejected_and_reported():
    player = player_with_one_basket()
    candidate = candidate_index_for(player, "images/009.png")

    # The utterance says "move", but only the structured flag counts now.
    result = _apply_matcher_selection(
        player, {"candidate_index": candidate, "position": 12, "ready_to_submit": False, "move": False}
    )
    _record_matcher_outcome(player, "selection_result", result)

    assert result["applied"] is False
    assert result["reason"] == "already_placed"
    assert result["already_at"] == 11
    notice = "\n".join(_action_feedback_notice(player))
    assert "NOT APPLIED" in notice
    assert f"candidate {candidate} is already in position 11" in notice
    assert "set `move` to true" in notice


def test_move_flag_moves_basket_and_clears_notice():
    player = player_with_one_basket()
    candidate = candidate_index_for(player, "images/009.png")

    result = _apply_matcher_selection(
        player, {"candidate_index": candidate, "position": 12, "ready_to_submit": False, "move": True}
    )
    _record_matcher_outcome(player, "selection_result", result)

    assert result["applied"] is True
    assert result["vacated_position"] == 11
    board = {item["position"]: item["image"] for item in json.loads(player.group.ai_partial_sequence)}
    assert board[11] is None and board[12] == "images/009.png"
    assert _action_feedback_notice(player) == []


def test_clarification_turn_has_no_selection_result():
    player = player_with_one_basket()
    assert _apply_matcher_selection(player, {"candidate_index": None, "position": None}) is None


def test_turn_cap_ends_round_and_scores_board_as_is(monkeypatch):
    player = player_with_one_basket()
    player.session.config["max_turns_per_round"] = 2
    player.group.ai_messages = json.dumps([{"text": "basket 1 is...", "sender_role": "director"}])
    monkeypatch.setattr(
        ai_vs_ai, "generate_ai_vs_ai_reply", lambda *_: {"text": "all set, thanks!", "selection": None}
    )

    result = run_ai_vs_ai_turn(player)
    status = get_ai_vs_ai_status(player)

    assert result["is_complete"] is True
    assert status["turn_cap_reached"] is True
    assert round(status["accuracy"], 6) == round(100 / 12, 6)  # only position 11 filled, and correctly
    assert json.loads(player.group.matcher_sequence) == [
        {"position": 11, "image": "images/009.png", "originalPosition": "33"},
    ]


def test_turn_cap_can_be_disabled(monkeypatch):
    player = player_with_one_basket()
    player.session.config["max_turns_per_round"] = 0
    player.group.ai_messages = json.dumps([{"text": "basket 1 is...", "sender_role": "director"}] * 300)
    monkeypatch.setattr(
        ai_vs_ai, "generate_ai_vs_ai_reply", lambda *_: {"text": "ok", "selection": None}
    )

    assert run_ai_vs_ai_turn(player)["is_complete"] is False


def test_refused_submit_is_reported_with_empty_positions(monkeypatch):
    player = player_with_one_basket()
    player.group.ai_messages = json.dumps([{"text": "that's all 12", "sender_role": "director"}])

    def fake_reply(player, role, latest):
        log = json.loads(player.group.ai_reasoning_log)
        log.append({"ai_role": "matcher", "utterance": "all set, submitting!"})
        player.group.ai_reasoning_log = json.dumps(log)
        return {"text": "all set, submitting!", "selection": {"candidate_index": None, "ready_to_submit": True}}

    monkeypatch.setattr(ai_vs_ai, "generate_ai_vs_ai_reply", fake_reply)

    assert run_ai_vs_ai_turn(player)["is_complete"] is False
    notice = "\n".join(_action_feedback_notice(player))
    assert "YOUR SUBMIT WAS NOT ACCEPTED" in notice
    assert "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12" in notice


def test_board_state_is_last_message_in_matcher_context(monkeypatch):
    import referential_task.ai_providers as providers

    player = player_with_one_basket()
    player.session.config.update(prompt_strategy="cameron-prompt", ai_matcher_model="gpt-6-luna")
    player.group.ai_messages = json.dumps([{"text": "a round brown basket", "sender_role": "director"}])
    seen = {}

    def fake_call(messages, **_):
        seen["messages"] = messages
        return json.dumps({"utterance": "which one?", "reasoning": None, "selection": None})

    monkeypatch.setattr(providers, "call_ai_api", fake_call)
    monkeypatch.setattr(providers, "has_ai_client_for_role", lambda *_: True)
    ai_vs_ai.generate_ai_vs_ai_reply(player, "matcher", "a round brown basket")

    last = seen["messages"][-1]
    assert last["role"] == "developer"
    assert last["content"].startswith("=== CURRENT BOARD STATE ===")
