"""Tests for the REPL's weak-model + error-UX guards.

Two failure modes observed live (NVIDIA free-tier models) locked down:

* phi-4-mini emitted ``{ "name": "read", "parameters": {...} }`` as its
  final ANSWER — a tool call leaked as text. The guard must catch that
  shape (and the fenced / OpenAI-nested variants) without false-firing
  on prose or ordinary JSON data.
* A model 404 surfaced as a raw ``PermanentModelError`` dump with no
  guidance. ``friendly_error_hint`` must map the classified error
  families to actionable next steps, and stay silent on unknowns.
"""

from __future__ import annotations

from loom_code.repl import (
    _looks_like_leaked_tool_call,
    friendly_error_hint,
)

# ---- _looks_like_leaked_tool_call ----------------------------------------


def test_detects_live_phi_leak_shape() -> None:
    # The exact output observed from phi-4-mini for a bare "hi".
    assert _looks_like_leaked_tool_call(
        '{ "name": "read", "parameters": {"path": "FileA.py"} }'
    )


def test_detects_openai_nested_function_shape() -> None:
    assert _looks_like_leaked_tool_call(
        '{"function": {"name": "bash"}, "arguments": {"command": "ls"}}'
    )


def test_detects_fenced_code_block() -> None:
    assert _looks_like_leaked_tool_call(
        '```json\n{"name": "edit", "args": {}}\n```'
    )


def test_ignores_prose() -> None:
    assert not _looks_like_leaked_tool_call(
        "Hello! How can I help you today?"
    )


def test_ignores_ordinary_json_data() -> None:
    # JSON answers about data must not trip the guard.
    assert not _looks_like_leaked_tool_call(
        '{"result": 42, "status": "ok"}'
    )


def test_ignores_unknown_tool_names() -> None:
    # Tool-call shape but not a tool the agents expose.
    assert not _looks_like_leaked_tool_call(
        '{"name": "launch_rocket", "parameters": {}}'
    )
    # A person's name in a data object is not a tool.
    assert not _looks_like_leaked_tool_call(
        '{"name": "John Smith", "parameters": {"age": 30}}'
    )


def test_ignores_json_embedded_in_prose() -> None:
    assert not _looks_like_leaked_tool_call(
        'The config is {"name": "read"} which means it reads.'
    )


# ---- friendly_error_hint --------------------------------------------------


def test_hint_for_model_not_found() -> None:
    # The exact error family from the live NVIDIA 404.
    exc = Exception(
        "OpenAI error: litellm.NotFoundError: NotFoundError: "
        "Nvidia_nimException - 404 page not found"
    )
    hint = friendly_error_hint(exc)
    assert hint is not None and "/set_model" in hint


def test_hint_for_auth_rejection() -> None:
    hint = friendly_error_hint(
        Exception("AuthenticationError: invalid api key")
    )
    assert hint is not None and "key" in hint.lower()


def test_hint_for_rate_limit() -> None:
    hint = friendly_error_hint(Exception("RateLimitError: 429"))
    assert hint is not None and "rate-limited" in hint


def test_no_hint_for_unknown_errors() -> None:
    assert friendly_error_hint(Exception("something novel")) is None
