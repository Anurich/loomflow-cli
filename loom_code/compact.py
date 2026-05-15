"""Conversation compaction — keeps long REPL sessions from blowing
through the model's context window.

The pattern:

1. After each REPL turn, the REPL accumulates the cumulative token
   count and a list of ``(user_prompt, agent_output)`` exchanges.
2. When the cumulative count crosses a threshold (default: 80% of
   the active model's context window; configurable per session via
   the ``/compress_token_length`` slash command) the REPL hands
   the exchanges to a :class:`Compactor`.
3. The compactor is a separate, single-shot loomflow ``Agent`` —
   no tools, dedicated prompt — that produces a dense prose
   summary preserving what was attempted, what worked, what
   failed, and any constraints the user expressed.
4. The summary lands in ``agent.memory.update_block(
   "session_summary", text)`` — a working block, which loomflow
   auto-injects into every subsequent system prompt.
5. The REPL resets ``session_id`` so the next turn starts with a
   fresh conversation thread but immediately "remembers" the
   session-so-far through the working block.

Pure loomflow primitives — no framework changes.
"""

from __future__ import annotations

from loomflow import Agent

# Best-effort context-window lookup. Substring match against known
# model families; fallback for anything we don't recognise (local
# Ollama models, niche LiteLLM providers, future model names) —
# the user can always override via /compress_token_length.
#
# Keep this list short and obvious. We're not trying to be a
# tokenizer service — just give a sensible default that adapts to
# the model the user picked.
_KNOWN_CONTEXT_WINDOWS: dict[str, int] = {
    # OpenAI 4.1 family — 1M context.
    "gpt-4.1": 1_000_000,
    # OpenAI 4o family — 128k.
    "gpt-4o": 128_000,
    # OpenAI o-series reasoning models — 200k.
    "o4": 200_000,
    "o3": 200_000,
    # Anthropic 4.x family — 200k.
    "claude-opus": 200_000,
    "claude-sonnet": 200_000,
    "claude-haiku": 200_000,
}

# Conservative default for unknown models — typical of small open
# local models (llama3 8B = 8k, qwen2.5 7B = 32k, etc.). Users on
# bigger local models or unusual providers can /compress_token_length
# upward.
_FALLBACK_CONTEXT_WINDOW = 32_000

# Trigger when cumulative usage crosses this fraction of the model's
# context window. 0.8 leaves headroom for the current turn's own
# prompt + tool I/O without bumping the actual limit.
_DEFAULT_TRIGGER_FRACTION = 0.8


def context_window_for(model: str) -> int:
    """Best-effort context-window estimate for ``model``.

    Substring match — ``"gpt-4.1-mini"`` and ``"gpt-4.1-nano"`` both
    pick up the ``"gpt-4.1"`` entry. Returns
    :data:`_FALLBACK_CONTEXT_WINDOW` when nothing matches; the user
    can override via ``/compress_token_length`` if the fallback is
    wrong for their model.
    """
    lower = model.lower()
    for key, ctx in _KNOWN_CONTEXT_WINDOWS.items():
        if key in lower:
            return ctx
    return _FALLBACK_CONTEXT_WINDOW


def default_compact_threshold(model: str) -> int:
    """Default token threshold at which the REPL should compact —
    80% of the model's context window."""
    return int(context_window_for(model) * _DEFAULT_TRIGGER_FRACTION)


_COMPACTOR_PROMPT = """\
You are the CONVERSATION COMPACTOR. A user has been in a long REPL
session with a coding agent and the session is approaching its
context limit. Your job: compress the history into a dense running
summary the next turn will use as its ONLY memory of everything
before it.

You will receive a list of (USER, AGENT) exchanges. Write ONE
dense paragraph (300-500 words). Preserve, in roughly this order:

1. What the user is trying to accomplish — the overarching goal.
2. Each concrete change attempted and the outcome — succeeded /
   failed / partially done.
3. Specific identifiers worth remembering: file paths, function
   names, branch names, command outputs, error messages.
4. Decisions or constraints the user expressed ("don't touch X",
   "we agreed on Y", "the API is Z").
5. Open questions or in-progress threads.

Do NOT:
- Include casual back-and-forth that didn't lead anywhere.
- Repeat boilerplate ("I'll help you with that...").
- Use markdown headers, bullet lists, or code blocks — just prose.
- Editorialise. Be factual.

Make it load-bearing. Future-you will reconstruct context from
this and nothing else.
"""


class Compactor:
    """A small loomflow ``Agent`` whose only job is to summarise a
    conversation history. Builds once, reuses across compactions
    in the same session."""

    def __init__(self, *, model: str) -> None:
        self._agent = Agent(
            _COMPACTOR_PROMPT,
            model=model,
            # No tools — single-shot summarisation. prompt_caching
            # helps if the agent runs multiple times in a session
            # (the system prompt is stable).
            prompt_caching=True,
        )

    async def compact(
        self, exchanges: list[tuple[str, str]]
    ) -> str:
        """Run the compactor on a list of ``(user_prompt,
        agent_output)`` pairs. Returns the summary as plain text."""
        if not exchanges:
            return ""
        rendered = "\n\n".join(
            f"USER:\n{user.strip()}\n\nAGENT:\n{out.strip()}"
            for user, out in exchanges
        )
        result = await self._agent.run(rendered, user_id="loom-code")
        return result.output.strip()
