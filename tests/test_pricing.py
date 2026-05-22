"""Guard: loom-code requires a loomflow whose pricing table is correct.

loom-code used to monkeypatch ``PRICING_PER_MTOKEN`` at agent-build
time because the framework snapshot billed the GPT-5.x family at $0
and Claude Opus 4.5+ at the retired $15/$75. That fix now lives in
loomflow itself (>=0.10.21), so the override is gone.

These tests pin the framework rates loom-code depends on. If loom-code
is ever installed against an older/stale loomflow, they fail loudly —
the cue to bump the ``loomflow>=`` floor — instead of silently
reporting wrong costs in result.cost_usd / the desktop cost badge.
"""

from __future__ import annotations

from loomflow.model._pricing import estimate_cost


def _cost(model: str) -> float:
    return estimate_cost(model, 1_000_000, 100_000)


def test_opus_4_5plus_is_5_25_not_retired_15_75() -> None:
    # $5/$25 → 1M*5 + 100k*25 = $7.50 (the retired rate gave $22.50).
    for m in ("claude-opus-4-7", "claude-opus-4-6", "claude-opus-4-5"):
        assert _cost(m) == 7.5, m
    # Opus 4.1 / 4.0 keep the old $15/$75 → $22.50.
    assert _cost("claude-opus-4-1") == 22.5


def test_gpt5_family_is_priced_not_zero() -> None:
    assert _cost("gpt-5.5") == 8.0  # $5/$30
    assert round(_cost("gpt-5.4-mini"), 4) == 1.2  # $0.75/$4.50
    for m in (
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.3-codex",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
    ):
        assert _cost(m) > 0, m


def test_o3_family_repriced() -> None:
    assert _cost("o3") == 2.8  # $2/$8 (was $10/$40)
    assert _cost("o3-pro") == 28.0  # $20/$80


def test_gpt5_cache_read_is_ten_percent() -> None:
    # GPT-5.x cached input is 10% of input, not OpenAI's old uniform 50%.
    # gpt-5.4 input $2.50/MTok × 0.1 → $0.00025 for 1000 cached tokens.
    assert estimate_cost(
        "gpt-5.4", 0, 0, cached_input_tokens=1000
    ) == 0.00025
