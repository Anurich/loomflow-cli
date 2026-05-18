"""Small utility helpers that wrap loomflow Agent for one-shot
text generation. Today: ``generate_commit_message``.

These live in loom-code (not in loomflow framework) because they
encode a specific opinionated voice — conventional commits, no
preamble, terse imperative. The framework stays neutral; loom-code
is where the opinions live.

Anyone calling these gets a Pythonic ``await fn(input) -> str``
shape — no streaming, no event subscription, no agent loop.
Internally we still go through ``loomflow.Agent`` because that
gives us model-string resolution + retry policy + adapter
handling for free, but the caller doesn't see any of it.
"""

from __future__ import annotations

from loomflow import Agent

# Default model: small + fast + cheap. Commit-message generation
# is a pure-function-of-diff task; no need for a top-tier model.
# Caller can override via ``model=`` if they want different
# voice (e.g. claude-sonnet-4-6 for longer-body messages).
_DEFAULT_MODEL = "claude-haiku-4-5"


_COMMIT_SYSTEM_PROMPT = """You write conventional-commit messages from git diffs.

Format:
- First line: under 50 chars, imperative mood, ``type(scope): subject``
- Types: feat, fix, refactor, docs, test, chore, perf, style, ci, build
- Subject in imperative mood: "add" not "added" or "adds"
- Lowercase subject, no trailing period
- Optional body separated by a blank line, wrapped at 72 chars, explains WHY (not what)
- No marketing fluff, no AI-disclosure trailers, no markdown fences in the output

Examples:
  feat(auth): add JWT validator with RS256 support

  fix(memory): close fact-store connection on agent teardown

  docs: explain the prompt-caching opt-in shape

  refactor(sidecar): extract git ops into _git_run helper

Reply with ONLY the commit message — no preamble, no quoting, no
markdown fences, no "Here is your commit message" lines."""


async def generate_commit_message(
    diff: str,
    *,
    model: str | None = None,
) -> str:
    """Generate a conventional-commit message from a git diff.

    One-shot call to a small/fast model. No tools, no memory, no
    workspace — deliberately stateless because commit-message
    generation is a pure function of the diff.

    Args:
        diff: The git diff text. Typically the output of
            ``git diff --cached`` (the staged changes).
        model: Override the default model. Pass any string
            loomflow's model resolver understands (e.g.
            ``"gpt-4.1-nano"``, ``"claude-sonnet-4-6"``).

    Returns:
        The suggested commit message as plain text — already
        stripped of leading/trailing whitespace. Empty diff in →
        an empty string out (caller should guard the empty case).

    Raises:
        Whatever the underlying loomflow model adapter raises on
        provider errors (missing API key, network, etc.). The
        caller is responsible for surfacing those.
    """
    if not diff or not diff.strip():
        return ""
    scribe = Agent(
        _COMMIT_SYSTEM_PROMPT,
        model=model or _DEFAULT_MODEL,
    )
    result = await scribe.run(diff)
    return (result.output or "").strip()
