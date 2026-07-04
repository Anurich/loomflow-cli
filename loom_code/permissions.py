"""Permission rules + modes for the approval gate.

Claude Code's safety model, adapted: instead of asking the user y/n
for every destructive call, a layered policy decides allow / ask / deny
up front, and only genuine "ask" calls reach the interactive prompt.

Two knobs:

* **Rules** — ``allow`` / ``ask`` / ``deny`` glob patterns declared in
  ``settings.toml`` (user + project scopes), matched against a
  ``tool(target)`` string like ``bash(pytest -q)`` or
  ``edit(src/.env)``. ``deny`` wins over ``ask`` wins over ``allow``,
  and deny is absolute (not even 'allow all' or accept-edits overrides
  it). This lets a user say ``allow "bash(pytest*)"`` /
  ``deny "edit(*.env)"`` once instead of confirming forever.
* **Mode** — the session's default posture for calls no rule matches:
  ``default`` (ask), ``accept-edits`` (auto-allow write/edit, still ask
  for bash), ``plan`` (deny all mutation — research only), or
  ``yolo`` (allow all; the irreversible-danger gate still fires).

The engine here is pure decision logic + parsing; :class:`ApprovalGate`
owns the interactive prompt and calls :func:`decide`.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# The mutation tools the gate arbitrates. Read-only tools never reach
# it (loomflow's StandardPermissions only flags these).
_MUTATION_TOOLS = frozenset({"write", "edit", "multi_edit", "bash"})
_EDIT_TOOLS = frozenset({"write", "edit", "multi_edit"})


class Decision(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class Mode(StrEnum):
    """Session default posture for calls no rule matches."""

    DEFAULT = "default"          # ask for every mutation
    ACCEPT_EDITS = "accept-edits"  # auto-allow edits, ask for bash
    PLAN = "plan"                # deny all mutation (research only)
    YOLO = "yolo"                # allow all (danger gate still fires)


@dataclass
class Rules:
    """Ordered allow/ask/deny glob patterns. Patterns match a
    ``tool(target)`` string, case-sensitively, via fnmatch."""

    allow: list[str] = field(default_factory=list)
    ask: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)

    def merged_with(self, other: Rules) -> Rules:
        """Combine two scopes (project onto user). Concatenation is
        enough — :func:`decide` applies deny>ask>allow precedence, so
        order within a bucket doesn't change the outcome."""
        return Rules(
            allow=[*self.allow, *other.allow],
            ask=[*self.ask, *other.ask],
            deny=[*self.deny, *other.deny],
        )


def call_target(tool: str, args: dict[str, Any]) -> str:
    """The string a rule matches against: ``tool(target)``.

    ``target`` is the command for bash (and bash_background — same
    rules bite both), the path for edits/writes, so a user writes
    intuitive patterns: ``bash(pytest*)``, ``edit(*.env)``.
    """
    if tool in ("bash", "bash_background"):
        target = str(args.get("command", "")).strip()
    else:
        target = str(args.get("path", "")).strip()
    return f"{tool}({target})"


def _matches(patterns: list[str], tool: str, target: str) -> bool:
    """True if any pattern matches, tried against both the full
    ``tool(target)`` string and a bare ``tool`` (so ``deny "bash"``
    blanket-denies bash without needing ``bash(*)``)."""
    for pat in patterns:
        if fnmatch.fnmatch(target, pat) or fnmatch.fnmatch(tool, pat):
            return True
    return False


def decide(
    tool: str, args: dict[str, Any], rules: Rules, mode: Mode
) -> Decision:
    """Resolve a destructive call to allow / ask / deny.

    Precedence, highest first:
      1. explicit ``deny`` rule       — absolute, nothing overrides it
      2. explicit ``ask`` rule        — force the prompt even in yolo
      3. explicit ``allow`` rule      — skip the prompt
      4. mode default                 — plan denies, accept-edits allows
         edits, yolo allows, default asks
    """
    target = call_target(tool, args)
    if _matches(rules.deny, tool, target):
        return Decision.DENY
    if _matches(rules.ask, tool, target):
        return Decision.ASK
    if _matches(rules.allow, tool, target):
        return Decision.ALLOW

    if mode is Mode.PLAN:
        return Decision.DENY
    if mode is Mode.YOLO:
        return Decision.ALLOW
    if mode is Mode.ACCEPT_EDITS and tool in _EDIT_TOOLS:
        return Decision.ALLOW
    return Decision.ASK


def parse_mode(text: str) -> Mode | None:
    """Parse a ``/mode`` argument, tolerant of aliases, or None."""
    t = text.strip().lower().replace("_", "-")
    aliases = {
        "default": Mode.DEFAULT,
        "normal": Mode.DEFAULT,
        "ask": Mode.DEFAULT,
        "accept-edits": Mode.ACCEPT_EDITS,
        "acceptedits": Mode.ACCEPT_EDITS,
        "edits": Mode.ACCEPT_EDITS,
        "auto-edit": Mode.ACCEPT_EDITS,
        "plan": Mode.PLAN,
        "readonly": Mode.PLAN,
        "read-only": Mode.PLAN,
        "yolo": Mode.YOLO,
        "bypass": Mode.YOLO,
        "allow-all": Mode.YOLO,
    }
    return aliases.get(t)


def load_rules(settings_dir_paths: list[Any]) -> Rules:
    """Merge ``[permissions]`` allow/ask/deny lists from each
    ``<dir>/settings.toml`` in order (user first, project last so the
    project layers on top). Lenient — missing file / bad TOML / wrong
    types are skipped, never raised (a broken config must not brick
    startup, matching the extensions loader's posture)."""
    import tomllib
    from pathlib import Path

    merged = Rules()
    for base in settings_dir_paths:
        settings = Path(base) / "settings.toml"
        try:
            data = tomllib.loads(settings.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        perms = data.get("permissions")
        if not isinstance(perms, dict):
            continue

        def _strs(perms: dict[str, Any], key: str) -> list[str]:
            v = perms.get(key)
            return [str(x) for x in v] if isinstance(v, list) else []

        merged = merged.merged_with(
            Rules(
                allow=_strs(perms, "allow"),
                ask=_strs(perms, "ask"),
                deny=_strs(perms, "deny"),
            )
        )
    return merged
