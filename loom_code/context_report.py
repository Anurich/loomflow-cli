"""Context observability — the renderers behind ``/context`` and the
per-turn ``ctx%`` readout.

The loudest 2026 harness complaint (pi's author's core critique,
Anthropic's own Claude Code postmortems, Kilo's whole positioning) is
INVISIBLE context: harnesses inject content the user never sees,
compact silently, and change defaults without telling anyone.
loom-code's answer is to show everything — on demand (``/context``,
``/prompt``) and ambiently (a ``N% ctx`` figure on every turn's
summary line).

Pure functions only (no console, no agent) so the rendering is
trivially testable; ``repl.py`` gathers the live numbers and prints.
"""

from __future__ import annotations

# Mirrors loomflow's DEFAULT_CHARS_PER_TOKEN — the conservative
# cross-content estimate (English prose ≈4 chars/token, code ≈3;
# under-estimating context left would overflow, so estimate high).
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Char-based token estimate, floored at 0 for empty text.

    Used for the working-block sizes in ``/context`` — these blocks
    never pass through a provider tokenizer on their own, so an
    estimate is the honest label (and it's marked ``~`` in the UI).
    """
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def context_percent(used_tokens: int, window: int) -> int:
    """Whole-percent context occupancy, clamped to [0, 100]."""
    if window <= 0:
        return 0
    return max(0, min(100, round(used_tokens * 100 / window)))


def context_report(
    *,
    model: str,
    window: int,
    used_tokens: int,
    threshold: int,
    blocks: list[tuple[str, str]],
    n_exchanges: int,
) -> str:
    """Render the ``/context`` report as plain text.

    ``blocks`` is ``[(name, content), …]`` — every working block
    loomflow will inject into the next system prompt. ``used_tokens``
    is the context high-water mark of the last turn's input (the same
    figure the auto-compactor keys on), so the user sees the number
    the harness itself acts on — not a synthetic one.
    """
    pct = context_percent(used_tokens, window)
    bar_w = 24
    filled = round(bar_w * pct / 100)
    bar = "█" * filled + "░" * (bar_w - filled)

    lines = [
        f"context — {model}",
        f"  window     {window:>10,} tokens",
        f"  used       {used_tokens:>10,} tokens "
        f"[{bar}] {pct}%",
    ]
    if threshold > 0:
        lines.append(
            f"  compaction {threshold:>10,} tokens "
            f"(auto-compacts at this point)"
        )
    else:
        lines.append("  compaction        off")
    lines.append(
        f"  history    {n_exchanges:>10,} exchange"
        f"{'s' if n_exchanges != 1 else ''} this thread"
    )
    lines.append("")
    if blocks:
        lines.append(
            "injected working blocks (folded into every system prompt):"
        )
        total = 0
        for name, content in sorted(blocks, key=lambda b: b[0]):
            t = estimate_tokens(content)
            total += t
            lines.append(f"  {name:<18} ~{t:>7,} tokens")
        lines.append(f"  {'total':<18} ~{total:>7,} tokens")
    else:
        lines.append("injected working blocks: none")
    lines.append("")
    lines.append(
        "nothing else is injected — what you see here plus the "
        "conversation is the model's entire context. /prompt shows "
        "the full text."
    )
    return "\n".join(lines)


def prompt_dump(
    *,
    instructions: str | None,
    blocks: list[tuple[str, str]],
) -> str:
    """Render the ``/prompt`` dump: the coordinator's static
    instructions plus every working block body, clearly delimited.
    No paraphrasing, no elision — the point is that this IS what the
    model receives."""
    parts: list[str] = []
    if instructions:
        parts.append("═══ system instructions (static) ═══")
        parts.append(instructions.rstrip())
    else:
        parts.append(
            "═══ system instructions (static) ═══\n"
            "(not exposed by this agent build — working blocks below "
            "are still exact)"
        )
    for name, content in sorted(blocks, key=lambda b: b[0]):
        parts.append(
            f"═══ working block: {name} "
            f"(~{estimate_tokens(content):,} tokens) ═══"
        )
        parts.append(content.rstrip() or "(empty)")
    return "\n\n".join(parts)
