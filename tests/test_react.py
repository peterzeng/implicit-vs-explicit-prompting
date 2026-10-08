import json

from referential_task import react
from referential_task.react import (
    react_enabled,
    react_step_model_config,
    run_react_loop,
    run_search_history,
    run_zoom,
)
from referential_task.state import Group, Player, Session


def make_player(config=None):
    grid = [{"image": f"images/{i:03d}.png", "basket_id": i} for i in range(1, 13)]
    group = Group(shared_grid=json.dumps(grid))
    return Player(group=group, session=Session({"basket_set": 5, **(config or {})}))


def step(action, indices=(), query=""):
    return json.dumps({"thought": "t", "action": action, "indices": list(indices), "query": query})


def test_react_enabled_respects_roles():
    assert not react_enabled(make_player(), "matcher")
    player = make_player({"react": True, "react_roles": ["matcher"]})
    assert react_enabled(player, "matcher")
    assert not react_enabled(player, "director")


def test_step_model_override_only_changes_model_and_provider():
    role_cfg = {"provider": "openai", "model": "gpt-5.5", "reasoning_effort": "low"}
    assert react_step_model_config(make_player(), role_cfg) == role_cfg
    cfg = react_step_model_config(make_player({"react_step_model": "gpt-5-nano"}), role_cfg)
    assert cfg["model"] == "gpt-5-nano"
    assert cfg["reasoning_effort"] == "low"


def test_zoom_director_renders_image_and_reports_invalid():
    text, image_url = run_zoom(make_player(), "director", [3, 3, 13])
    assert image_url.startswith("data:image/png;base64,")
    assert "Basket 3" in text
    assert "[13]" in text


def test_zoom_matcher_uses_candidate_pool_order():
    player = make_player()
    pool = react._load_matcher_pool_image_urls(player)
    items, invalid = react._zoom_items(player, "matcher", [1, 18, 19])
    assert items[0] == ("Candidate 1", pool[0]["slot"]["image"])
    assert items[1] == ("Candidate 18", pool[17]["slot"]["image"])
    assert invalid == [19]


def test_search_history_finds_prior_round_names():
    history = [
        {"sender_role": "director", "text": "The tall wicker hamper with a lid", "round_number": 1},
        {"sender_role": "matcher", "text": "Got it", "round_number": 1},
        {"sender_role": "director", "text": "Next: the hamper again", "round_number": 2},
    ]
    obs = run_search_history(history, "wicker hamper", current_round=2)
    assert "[Round 1] director: The tall wicker hamper" in obs
    assert "[Round 2] director: Next: the hamper again" in obs
    assert "Got it" not in obs
    assert "No earlier messages" in run_search_history(history, "teapot", 2)


def test_loop_runs_tools_then_stops_on_finish(monkeypatch):
    replies = iter([step("zoom", [1, 2]), step("search_history", query="hamper"), step("finish")])
    calls = []

    def fake_call(messages, model_config, temperature, response_format):
        calls.append(messages)
        return next(replies)

    monkeypatch.setattr(react, "call_ai_api", fake_call)
    base = [{"role": "developer", "content": "base"}]
    history = [{"sender_role": "director", "text": "a hamper", "round_number": 1}]
    scratchpad, trace = run_react_loop(
        make_player(), "director", base, history, {"model": "tiny"}, max_steps=4
    )

    assert [t["action"] for t in trace] == ["zoom", "search_history", "finish"]
    assert all(t["model"] == "tiny" for t in trace)
    # zoom observation carries an image; final message hands off to the reply.
    assert scratchpad[1]["content"][1]["type"] == "image_url"
    assert scratchpad[-1]["content"].startswith("INVESTIGATION COMPLETE")
    # Each step sees the base context plus all earlier steps.
    assert len(calls[2]) == len(base) + 1 + 4


def test_loop_respects_max_steps_and_survives_bad_output(monkeypatch):
    monkeypatch.setattr(react, "call_ai_api", lambda **kw: step("zoom", [1]))
    _, trace = run_react_loop(make_player(), "director", [], [], {}, max_steps=2)
    assert len(trace) == 2

    monkeypatch.setattr(react, "call_ai_api", lambda **kw: "not json")
    scratchpad, trace = run_react_loop(make_player(), "director", [], [], {}, max_steps=3)
    assert trace[0]["error"] == "unparseable step"
    assert scratchpad == []


def test_tracker_round_row_sums_steps_and_tokens():
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
    from track_react_runs import round_row

    usage = {"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 10, "total_tokens": 110}
    record = {
        "session_id": "s1",
        "round_number": 2,
        "config": {"react": True, "react_roles": ["matcher"], "react_step_model": "tiny"},
        "status": {"accuracy": 91.5, "turn_count": 2},
        "ai_messages": [{"text": "big round basket"}, {"text": "got it"}],
        "ai_reasoning_log": [
            {"ai_role": "director", "usage": usage, "react_trace": None},
            {
                "ai_role": "matcher",
                "usage": usage,
                "react_trace": [
                    {"action": "zoom", "usage": usage},
                    {"action": "finish", "usage": None},
                ],
            },
        ],
    }
    row = round_row(record)
    assert (row["react_turns"], row["react_steps"], row["zoom_steps"], row["finish_steps"]) == (1, 2, 1, 1)
    assert row["final_input_tokens"] == 200 and row["step_input_tokens"] == 100
    assert row["total_tokens"] == 330
    assert (row["calls"], row["calls_missing_usage"]) == (4, 1)
    assert row["react_roles"] == "matcher" and row["accuracy"] == 91.5
