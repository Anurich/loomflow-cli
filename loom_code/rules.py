"""Project rules file (``AGENTS.md``) — bootstrap + safe rule appending.

loom-code already *reads* a context file (``CLAUDE.md`` / ``AGENTS.md`` /
``.loom/context.md`` — see :mod:`loom_code.project`) into the system
prompt every session. This module *writes* to it:

* :func:`init_agents_md` — create a starter ``AGENTS.md`` when the repo
  has no context file yet (the ``/init-loom`` command).
* :func:`add_rule` — append a durable rule the user stated in chat into a
  clearly-delimited **managed block**, with dedup + supersession so the
  block stays small and contradiction-free (the ``remember_rule`` tool).

The guardrail (matches Claude Code's split): the agent only ever curates
the **managed block** between the markers; everything the human writes
*outside* it is never touched. And the block is kept tidy at write-time —
duplicates are skipped, a superseding rule removes the one it replaces —
so it doesn't bloat the way an append-only log would.
"""

from __future__ import annotations

from pathlib import Path

from loomflow import tool
from loomflow.tools.registry import Tool

# Same order loom_code.project reads, so the file we WRITE is the file
# that gets READ back into the prompt.
_CONTEXT_FILENAMES = ("CLAUDE.md", "AGENTS.md", ".loom/context.md")
# When no context file exists yet, this is the one we create — the
# cross-tool open standard (read by loom-code and every AGENTS.md-aware
# agent), not a loom-only file.
_DEFAULT_RULES_FILE = "AGENTS.md"

# Markers delimiting the agent-curated block. The human's content lives
# OUTSIDE these; the agent only edits BETWEEN them.
BLOCK_START = "<!-- loom:rules (auto-added from chat — edit or delete freely) -->"
BLOCK_END = "<!-- /loom:rules -->"

# Soft cap on managed rules: past this we nudge (never auto-drop).
SOFT_CAP = 50

_STARTER_TEMPLATE = """\
# AGENTS.md

Instructions for AI coding agents working on this project (loom-code and
any AGENTS.md-aware tool). This file is read into the agent's context
each session — keep it concise (aim under ~200 lines).

## Overview
<what this project is — a line or two>

## Conventions
- <coding standards, e.g. indentation / naming>

## Commands
- install: <...>
- test: <...>
- lint: <...>

## Rules
{block_start}
{block_end}
"""


def detect_rules_file(root: Path | str) -> Path | None:
    """Return the existing context file (first by priority), or None."""
    base = Path(root)
    for name in _CONTEXT_FILENAMES:
        fp = base / name
        if fp.is_file():
            return fp
    return None


def target_rules_file(root: Path | str) -> Path:
    """The file rules are written to: an existing context file if one is
    present, else ``AGENTS.md`` at the repo root."""
    return detect_rules_file(root) or (Path(root) / _DEFAULT_RULES_FILE)


def current_rules_text(root: Path | str) -> str:
    """Full text of the active rules file, or '' if none exists."""
    fp = detect_rules_file(root)
    if fp is None:
        return ""
    try:
        return fp.read_text(encoding="utf-8")
    except OSError:
        return ""


def project_rules_block(root: Path | str) -> str:
    """Framed body for the per-turn ``project_rules`` working block.

    The active rules file, re-read FRESH each turn so a mid-session edit
    applies on the next turn (no restart) — unlike the static
    startup-baked path. Empty string when there's no rules file."""
    fp = detect_rules_file(root)
    if fp is None:
        return ""
    try:
        text = fp.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if not text:
        return ""
    return (
        f"# Project rules ({fp.name})\n"
        "House rules for this project — conventions, and things to do or "
        "AVOID. Follow them in everything you do and delegate.\n\n"
        f"{text}"
    )


def init_agents_md(root: Path | str) -> tuple[Path, bool]:
    """Create a starter ``AGENTS.md`` if the repo has no context file.

    Returns ``(path, created)`` — ``created=False`` means a context file
    already existed and was left untouched."""
    existing = detect_rules_file(root)
    if existing is not None:
        return existing, False
    fp = Path(root) / _DEFAULT_RULES_FILE
    fp.write_text(
        _STARTER_TEMPLATE.format(block_start=BLOCK_START, block_end=BLOCK_END),
        encoding="utf-8",
    )
    return fp, True


def _normalize(rule: str) -> str:
    """Compare-key for dedup/supersede: lowercased, whitespace-collapsed,
    trailing punctuation dropped. Two rules that say the same thing in
    slightly different wording collapse to the same key."""
    return " ".join(rule.lower().split()).rstrip(".!").strip()


def _ensure_block(text: str) -> str:
    """Guarantee the managed block markers exist, appending an empty one
    (under a ## Rules heading) when absent."""
    if BLOCK_START in text and BLOCK_END in text:
        return text
    suffix = "" if text.endswith("\n") or text == "" else "\n"
    return f"{text}{suffix}\n## Rules\n{BLOCK_START}\n{BLOCK_END}\n"


def _read_managed_rules(text: str) -> list[str]:
    """The ``- `` bullet lines currently inside the managed block."""
    start = text.find(BLOCK_START)
    end = text.find(BLOCK_END)
    if start == -1 or end == -1 or end < start:
        return []
    inner = text[start + len(BLOCK_START):end]
    out: list[str] = []
    for line in inner.splitlines():
        s = line.strip()
        if s.startswith("- "):
            out.append(s[2:].strip())
    return out


def _write_managed_rules(text: str, rules: list[str]) -> str:
    """Replace the managed block's body with ``rules`` (preserving
    everything outside the markers verbatim)."""
    start = text.find(BLOCK_START)
    end = text.find(BLOCK_END)
    body = "".join(f"- {r}\n" for r in rules)
    new_block = f"{BLOCK_START}\n{body}{BLOCK_END}"
    return text[:start] + new_block + text[end + len(BLOCK_END):]


def add_rule(
    root: Path | str, rule: str, *, supersedes: str | None = None
) -> str:
    """Append a durable rule to the managed block. Dedup (skip if already
    present) + supersede (drop the rule ``supersedes`` matches before
    adding). Creates the file if absent. Returns a human-readable status
    (the tool relays it). Best-effort write; the human's content outside
    the managed block is never modified."""
    rule = " ".join((rule or "").split()).strip()
    if not rule:
        return "remember_rule: empty rule — nothing recorded."

    fp = target_rules_file(root)
    try:
        text = fp.read_text(encoding="utf-8") if fp.is_file() else (
            _STARTER_TEMPLATE.format(block_start=BLOCK_START, block_end=BLOCK_END)
        )
        text = _ensure_block(text)
        rules = _read_managed_rules(text)

        norm = _normalize(rule)

        # Supersede: drop any managed rule the new one replaces.
        superseded: str | None = None
        if supersedes:
            sup_norm = _normalize(supersedes)
            for existing in list(rules):
                en = _normalize(existing)
                if en == sup_norm or sup_norm in en or en in sup_norm:
                    rules.remove(existing)
                    superseded = existing
                    break

        # Dedup: already present (and not a supersede) → no-op.
        if any(_normalize(r) == norm for r in rules):
            if superseded is not None:
                text = _write_managed_rules(text, rules)
                fp.write_text(text, encoding="utf-8")
                return (
                    f"Removed superseded rule ({superseded!r}); "
                    f"the new rule was already recorded."
                )
            return f"Already recorded in {fp.name}: {rule!r}."

        rules.append(rule)
        text = _write_managed_rules(text, rules)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(text, encoding="utf-8")
    except OSError as exc:
        return f"remember_rule: could not write {fp.name}: {exc}"

    msg = (
        f"Saved to {fp.name}: {rule!r}"
        + (f" (replaced {superseded!r})" if superseded else "")
        + f". It applies from the next session; I'll keep following it now too."
    )
    if len(rules) > SOFT_CAP:
        msg += (
            f" Note: {len(rules)} rules are tracked — consider pruning "
            f"{fp.name} (the long file weakens adherence)."
        )
    return msg


def remember_rule_tool(root: Path | str) -> Tool:
    """Build the ``remember_rule`` tool for the coordinator. The model
    calls it when the user states a DURABLE project rule, so the rule is
    persisted to ``AGENTS.md`` (always-in-prompt next session) instead of
    relying on probabilistic memory recall."""
    base = Path(root)

    async def remember_rule(rule: str, supersedes: str = "") -> str:
        """Persist a durable, user-stated rule to AGENTS.md."""
        return add_rule(base, rule, supersedes=supersedes or None)

    return tool(
        name="remember_rule",
        description=(
            "Persist a DURABLE project rule the user explicitly stated "
            "(e.g. 'never edit X', 'always run Y before commit', "
            "'don't use Z') to AGENTS.md, so it survives future sessions "
            "instead of relying on memory recall. Args: rule (the rule, "
            "as a short imperative); supersedes (optional — the text of "
            "an earlier rule this one reverses/updates; pass it so the "
            "old rule is removed instead of leaving a contradiction). "
            "Only call this for standing rules the user states, NOT for "
            "one-off task requests. Duplicates are skipped automatically."
        ),
    )(remember_rule)
