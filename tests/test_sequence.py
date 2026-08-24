import json

from referential_task.sequence import _update_ai_partial_sequence
from referential_task.state import Group, Player, Session
from referential_task.visual_context import _load_matcher_pool_image_urls


def make_round2_player():
    session = Session({"basket_set": 5})
    group = Group(
        shared_grid=json.dumps([
            {"position": "11", "image": "images/015.png", "basket_id": 1},
            {"position": "12", "image": "images/037.png", "basket_id": 2},
            {"position": "13", "image": "images/001.png", "basket_id": 3},
            {"position": "14", "image": "images/022.png", "basket_id": 4},
            {"position": "21", "image": "images/040.png", "basket_id": 5},
            {"position": "22", "image": "images/076.png", "basket_id": 6},
            {"position": "23", "image": "images/033.png", "basket_id": 7},
            {"position": "24", "image": "images/019.png", "basket_id": 8},
            {"position": "31", "image": "images/012.png", "basket_id": 9},
            {"position": "32", "image": "images/065.png", "basket_id": 10},
            {"position": "33", "image": "images/009.png", "basket_id": 11},
            {"position": "34", "image": "images/017.png", "basket_id": 12},
        ])
    )
    return Player(group=group, session=session, round_number=2)


def candidate_index_for(player, image_path):
    for idx, item in enumerate(_load_matcher_pool_image_urls(player), start=1):
        slot = item.get("slot") or {}
        if slot.get("image") == image_path:
            return idx
    raise AssertionError(f"Missing candidate for {image_path}")


def test_duplicate_candidate_does_not_vacate_existing_slot_without_explicit_move():
    player = make_round2_player()
    candidate = candidate_index_for(player, "images/009.png")
    player.group.ai_partial_sequence = json.dumps([
        {"position": 11, "image": "images/009.png", "originalPosition": "33"},
        {"position": 12, "image": None, "originalPosition": None},
    ])

    updated, vacated = _update_ai_partial_sequence(
        player,
        {"candidate_index": candidate, "position": 12, "ready_to_submit": True},
        allow_move=False,
    )

    assert vacated is None
    assert updated == [
        {"position": 11, "image": "images/009.png", "originalPosition": "33"},
        {"position": 12, "image": None, "originalPosition": None},
    ]


def test_duplicate_candidate_can_move_when_explicitly_allowed():
    player = make_round2_player()
    candidate = candidate_index_for(player, "images/009.png")
    player.group.ai_partial_sequence = json.dumps([
        {"position": 11, "image": "images/009.png", "originalPosition": "33"},
        {"position": 12, "image": None, "originalPosition": None},
    ])

    updated, vacated = _update_ai_partial_sequence(
        player,
        {"candidate_index": candidate, "position": 12, "ready_to_submit": False},
        allow_move=True,
    )

    assert vacated == 11
    assert updated == [
        {"position": 11, "image": None, "originalPosition": None},
        {"position": 12, "image": "images/009.png", "originalPosition": "33"},
    ]
