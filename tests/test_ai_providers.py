from referential_task.ai_providers import reasoning_effort_for_api, uses_max_completion_tokens
from referential_task.ai_vs_ai import _build_api_call_kwargs

MESSAGES = [{"role": "user", "content": "hi"}]


def test_paper_models_send_reasoning_effort_and_no_temperature():
    for model in ("gpt-5.2", "gpt-5.5"):
        kwargs = _build_api_call_kwargs(model, MESSAGES, temperature=0, max_tokens=600)
        assert kwargs["reasoning_effort"] == "none"
        assert "temperature" not in kwargs
        assert kwargs["max_completion_tokens"] == 600


def test_gpt6_models_use_gpt5_api_features():
    # gpt-6-luna rejects temperature=0 and max_tokens, and must receive
    # reasoning_effort=none to match the paper's setting.
    for model in ("gpt-6-luna", "gpt-6-sol", "gpt-5.6-luna"):
        kwargs = _build_api_call_kwargs(model, MESSAGES, temperature=0, max_tokens=600)
        assert kwargs["reasoning_effort"] == "none"
        assert "temperature" not in kwargs
        assert "max_tokens" not in kwargs
        assert kwargs["max_completion_tokens"] == 600


def test_original_gpt5_omits_none_effort():
    assert reasoning_effort_for_api("gpt-5", "none") is None
    assert reasoning_effort_for_api("gpt-5-nano", "low") == "low"


def test_non_reasoning_models_unchanged():
    kwargs = _build_api_call_kwargs("gpt-4o", MESSAGES, temperature=0, max_tokens=600)
    assert kwargs["temperature"] == 0
    assert kwargs["max_tokens"] == 600
    assert "reasoning_effort" not in kwargs
    assert not uses_max_completion_tokens("gpt-4o")
