"""Tests for the auto-compaction layer.

These cover the deterministic, no-API parts: the model →
context-window lookup, the default-threshold math, and the
compactor's contract on empty input. The end-to-end "trigger fires
and writes a working block" path is integration-shaped and skipped
here — it'd need an Agent.run() against a real or scripted model
inside the loom-code Agent shell, which is heavier than what these
unit tests are aimed at.
"""

from __future__ import annotations

import pytest

from loom_code.compact import (
    Compactor,
    context_window_for,
    default_compact_threshold,
)

pytestmark = pytest.mark.anyio


def test_context_window_known_openai_family() -> None:
    # gpt-4.1-mini / gpt-4.1-nano / gpt-4.1 all share the "gpt-4.1"
    # substring → 1M context.
    assert context_window_for("gpt-4.1-mini") == 1_000_000
    assert context_window_for("gpt-4.1") == 1_000_000


def test_context_window_known_anthropic_family() -> None:
    # claude-{opus,sonnet,haiku}-* → 200k.
    assert context_window_for("claude-sonnet-4-6") == 200_000
    assert context_window_for("claude-opus-4-7") == 200_000
    assert context_window_for("claude-haiku-4-5") == 200_000


def test_context_window_falls_back_for_unknown() -> None:
    # Local Ollama / unknown LiteLLM target — conservative fallback
    # so the user can override upward rather than being burned by
    # an over-generous default.
    assert context_window_for("ollama/llama3") == 32_000
    assert context_window_for("litellm/groq/some-new-model") == 32_000
    assert context_window_for("totally-made-up-model-v9") == 32_000


def test_default_threshold_is_80_percent() -> None:
    # The 0.8 fraction is the contract — leaves headroom for the
    # next turn's own prompt + tool I/O.
    assert default_compact_threshold("gpt-4.1-mini") == 800_000
    assert default_compact_threshold("claude-sonnet-4-6") == 160_000
    assert default_compact_threshold("ollama/llama3") == 25_600


def test_context_window_case_insensitive() -> None:
    # Substring match is lowercased, so case noise doesn't break
    # detection.
    assert context_window_for("Claude-Sonnet-4-6") == 200_000
    assert context_window_for("GPT-4.1-MINI") == 1_000_000


async def test_compactor_empty_exchanges_returns_empty() -> None:
    # No exchanges → no model call, returns ''. This guards the
    # cheapest no-op path so we don't fire a model API on an
    # empty REPL.
    c = Compactor(model="echo")
    assert await c.compact([]) == ""
