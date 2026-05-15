"""Bracketed-paste collapsing for the REPL.

Pasting a large block (a stack trace, a file, etc.) into the input
line used to dump the whole thing into the visible prompt — noisy,
hard to keep editing alongside it, and not what Claude Code does.
This module gives prompt_toolkit a binding that:

1. Stashes the full paste in a module-level list.
2. Inserts a short placeholder — ``[paste-N: <lines>, <chars>]`` —
   into the input line in its place.
3. On submit, :func:`expand_pastes` rewrites those placeholders
   back to the full text BEFORE the line goes to the agent. So
   the agent still sees everything; only the visible input line
   is collapsed.

The stash lives for the whole REPL session so pastes survive
multiple turns; ``/clear`` (in the REPL) drops them.
"""

from __future__ import annotations

import re

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys

# Triggers: paste is "big" if it's longer than this many chars OR
# contains more than this many newlines. The OR catches both
# "huge single-line log" and "30-line code snippet" — both are
# noisy in the prompt.
_PASTE_CHAR_THRESHOLD = 500
_PASTE_LINE_THRESHOLD = 4

# (paste_id, full_text) in submission order. Module-level so the
# binding closure and ``expand_pastes`` share the same state
# without the REPL having to thread the stash through.
_pastes: list[str] = []

_PLACEHOLDER_RE = re.compile(r"\[paste-(\d+):[^\]]*\]")


def build_paste_keybindings() -> KeyBindings:
    """Return a ``KeyBindings`` registered with a handler for
    ``Keys.BracketedPaste`` — short pastes pass through, long
    pastes are stashed + replaced with a placeholder."""
    kb = KeyBindings()

    @kb.add(Keys.BracketedPaste)
    def _on_paste(event: KeyPressEvent) -> None:
        text = event.data
        lines = text.count("\n") + 1 if text else 0
        is_big = (
            len(text) > _PASTE_CHAR_THRESHOLD
            or text.count("\n") > _PASTE_LINE_THRESHOLD
        )
        if not is_big:
            # Short paste — insert verbatim, no collapsing.
            event.current_buffer.insert_text(text)
            return
        # Stash + insert a placeholder. The placeholder syntax is
        # readable to the user AND grep-able by ``expand_pastes``.
        _pastes.append(text)
        idx = len(_pastes)
        placeholder = (
            f"[paste-{idx}: {lines} lines, {len(text)} chars]"
        )
        event.current_buffer.insert_text(placeholder)

    return kb


def expand_pastes(line: str) -> str:
    """Replace any ``[paste-N: ...]`` placeholder in ``line`` with
    the full text we stashed for that paste. Unknown indices
    (e.g. user typed one by hand) are left as-is."""

    def _replace(match: re.Match[str]) -> str:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(_pastes):
            return _pastes[idx]
        return match.group(0)

    return _PLACEHOLDER_RE.sub(_replace, line)


def reset_paste_stash() -> None:
    """Drop every stashed paste — called by ``/clear`` so the user
    can start a fresh conversation thread without yesterday's
    pastes silently lurking under old placeholder indices."""
    _pastes.clear()


def stash_size() -> int:
    """How many pastes are currently stashed. Used by /pastes or
    similar diagnostic commands (none yet)."""
    return len(_pastes)
