import json

from referential_task.prompt_context import (
    _build_director_conversation_state,
    _build_cross_round_matcher_feedback_memory,
    _get_pending_refill_positions,
)
from referential_task.state import Group, Player, Session
from referential_task.visual_context import _prepare_round_feedback_slots


def make_player(messages, partial_sequence=None):
    group = Group()
    group.ai_messages = json.dumps(messages)
    group.ai_partial_sequence = json.dumps(partial_sequence or [])
    return Player(group=group)


def test_director_conversation_state_accepts_basket_shorthand():
    player = make_player(
        [
            {"sender_role": "director", "text": f"b{pos}: already placed"}
            if idx % 2 == 0
            else {"sender_role": "matcher", "text": f"Got it, placing it in position {pos}."}
            for pos in range(1, 6)
            for idx in range(2)
        ]
        + [
            {"sender_role": "director", "text": "b6: dark brown bucket basket"},
            {"sender_role": "matcher", "text": "Ok cool, placing it in position 6."},
            {"sender_role": "director", "text": "b7: tall cylindrical hamper"},
        ]
    )

    assert _build_director_conversation_state(player) == {
        "completed_positions": [1, 2, 3, 4, 5, 6],
        "next_target": 7,
    }


def test_pending_refill_detects_confirmed_shorthand_basket_left_empty():
    player = make_player(
        [
            {"sender_role": "director", "text": "b6: dark brown bucket basket"},
            {"sender_role": "matcher", "text": "Ok cool, placing it in position 6."},
        ],
        partial_sequence=[
            {"position": 1, "image": "images/001.png", "originalPosition": "11"},
        ],
    )

    assert _get_pending_refill_positions(player) == [6]


def test_cross_round_matcher_memory_surfaces_prior_wrong_current_candidate():
    session = Session({"basket_set": 5})

    round1 = Player(
        group=Group(
            shared_grid=json.dumps([
                {"image": "images/001.png", "basket_id": 1},
                {"image": "images/015.png", "basket_id": 2},
            ]),
        ),
        session=session,
        round_number=1,
    )
    round1.group.matcher_sequence = json.dumps([
        {"position": 1, "image": "images/076.png", "originalPosition": 6},
        {"position": 2, "image": "images/015.png", "originalPosition": 2},
    ])

    round2 = Player(
        group=Group(
            shared_grid=json.dumps([
                {"image": "images/015.png", "basket_id": 1},
                {"image": "images/076.png", "basket_id": 2},
                {"image": "images/001.png", "basket_id": 3},
            ]),
        ),
        session=session,
        round_number=2,
    )

    memory = _build_cross_round_matcher_feedback_memory(round2)

    assert memory is not None
    assert "The same basket identities recur across rounds" in memory
    assert "076.png" in memory
    assert "was a WRONG match before" in memory
    assert "should have been 001.png" in memory


def test_director_feedback_slots_hide_matcher_wrong_selection():
    shared_grid = [
        {"image": "images/001.png", "basket_id": 1},
        {"image": "images/015.png", "basket_id": 2},
    ]
    matcher_sequence = [
        {"position": 1, "image": "images/076.png", "originalPosition": 6},
        {"position": 2, "image": "images/015.png", "originalPosition": 2},
    ]

    correct_count, director_slots = _prepare_round_feedback_slots(
        shared_grid, matcher_sequence, viewer_role="director"
    )
    _, matcher_slots = _prepare_round_feedback_slots(
        shared_grid, matcher_sequence, viewer_role="matcher"
    )

    assert correct_count == 1
    assert director_slots[0] == {
        "position": 1,
        "image": "images/001.png",
        "is_correct": False,
    }
    assert matcher_slots[0] == {
        "position": 1,
        "image": "images/076.png",
        "is_correct": False,
    }
