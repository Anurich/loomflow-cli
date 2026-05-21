"""``escalate_to_team`` — the SIMPLE-route → COMPLEX-route handoff.

loom-code routes each prompt to either a single SIMPLE coder (fast,
cheap) or the full COMPLEX supervisor team (parallel investigation +
review). The router classifies ONCE per prompt — so a task that
looked simple but turns out to need the team has no built-in way up.

This tool is that escape hatch. When the SIMPLE coder genuinely
can't make progress alone (the task needs cross-file coordination,
parallel investigation, or a dedicated review pass), it calls
``escalate_to_team(reason)``. The REPL detects the call after the
SIMPLE run and RE-DISPATCHES the same prompt through the supervisor
— which, thanks to ``conversation_scope="shared"``, inherits the
SIMPLE coder's entire partial conversation. So the work SIMPLE
already did isn't thrown away; it becomes context for the team.

Gating matters: the router exists to AVOID running the expensive
team when it's not needed. An over-eager escalation turns every
hard-ish task into "cheap attempt + expensive attempt", which is
worse than routing to COMPLEX once. The prompt instructs SIMPLE to
escalate only as a last resort — the tool is a safety valve, not a
default.
"""

from __future__ import annotations

from loomflow import tool
from loomflow.tools.registry import Tool

ESCALATE_TOOL_NAME = "escalate_to_team"


def escalate_to_team_tool() -> Tool:
    """Build the ``escalate_to_team(reason)`` tool for the SIMPLE
    coder. The tool itself just acknowledges + stops the SIMPLE
    run; the actual re-dispatch to the supervisor is driven by the
    REPL, which detects this tool call via the renderer."""

    async def escalate_to_team(reason: str) -> str:
        """Hand the current task off to the full team. Call ONLY
        as a last resort — see the system prompt for when this is
        warranted.

        Args:
            reason: one sentence on WHY the team is needed
                    (e.g. "needs coordinated changes across 4
                    modules + a test pass"). Surfaced to the user.
        """
        return (
            "Escalation accepted. Stopping SIMPLE-mode work — the "
            "coordinator + worker team is taking over this task "
            "with your full context preserved. Do NOT continue or "
            "make further tool calls; the team handles it from "
            "here."
        )

    return tool(
        name=ESCALATE_TOOL_NAME,
        description=(
            "LAST-RESORT escape hatch: hand the current task to "
            "the full coordinator+worker team. Call this ONLY when "
            "the task genuinely needs parallel investigation, "
            "coordinated multi-file changes, or a dedicated review "
            "pass that you can't do alone in a few more steps — "
            "NOT for things you can finish yourself. Your partial "
            "work is preserved and handed to the team. Args: "
            "reason (one sentence on why the team is needed)."
        ),
    )(escalate_to_team)
