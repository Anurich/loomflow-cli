"""Specialist sub-agents the main loom-code agent calls as tools.

loom-code's main agent is a single ReAct loop — one coherent
context, explore → act → verify. That's the right shape for the
*sequential* heart of coding. But two kinds of subtask are better
handed to a fresh, focused agent:

* **explore** — answering "how does X work / where is Y wired"
  means reading a LOT of files. Done inline, those file dumps
  flood the main agent's context with detail it'll never need
  again. A read-only explorer burns that context in ITS OWN
  window and hands back just the answer.
* **review** — verifying a change benefits from eyes that didn't
  write the code. The reviewer re-reads the changed files, runs
  the project's tests, and reports problems with severity.

Both are loomflow ``Agent``s wrapped as loomflow ``Tool``s — the
main agent calls them like any other tool, on its own judgement.
This is the Claude-Code shape: one main loop, specialists on
demand — NOT a supervisor that fragments the conversation across
workers. The subagents run with a fresh ``session_id`` (isolated
context) but inherit ``user_id`` from the live ``RunContext`` so
the multi-tenant partition holds.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from loomflow import Agent, StandardPermissions
from loomflow.architecture import ReAct
from loomflow.core.context import get_run_context
from loomflow.tools import (
    Tool,
    bash_tool,
    find_tool,
    grep_tool,
    ls_tool,
    read_tool,
)

from .project import Project

# Sub-agents are capped tighter than the main loop — they answer a
# scoped question, they don't drive a whole task.
_SUBAGENT_MAX_TURNS = 30

_EXPLORER_PROMPT = """\
You are the EXPLORER — a read-only investigator for a coding
agent. You are handed ONE question about a codebase. You answer it
thoroughly and hand the answer back.

You have read-only tools: `read`, `grep`, `find`, `ls`. You have
NO write/edit/bash — you cannot change anything, and you must not
try.

How you work:
- Start broad (`find` / `ls` / `grep` for the relevant symbols),
  then `read` the files that matter.
- Follow the actual wiring — imports, call sites, config — don't
  guess.
- Answer concretely. Cite `path:line` for every claim. Quote the
  key code, don't paraphrase it away.
- If the question has sub-parts, answer each.
- End with a short, direct summary the caller can act on.

Be exhaustive on facts, terse on prose. You are someone else's
research step — wasted words cost them context.
"""

_REVIEWER_PROMPT = """\
You are the REVIEWER — a verification specialist for a coding
agent. You are handed a description of a change that was just
made. Your job: independently confirm it is correct, complete,
and safe.

You have `read`, `grep`, `find`, `ls`, and `bash`. Use `bash` to
run the project's OWN tests / linters / build — not improvised
checks. You have NO write/edit — you do not fix things, you
REPORT.

How you work:
- Re-read the changed files yourself. Don't trust the description.
- Run the verification command (test suite, build, type-check).
- Look for: broken callers, missing edge cases, untested paths,
  things the change said it did but didn't, regressions.
- Report findings as a list, each tagged severity:
  `[blocker]` — must fix before this is done.
  `[risk]`    — probably wrong / fragile, worth a second look.
  `[nit]`     — minor, optional.
- If everything checks out, say so plainly: `VERDICT: pass` plus
  the evidence (which tests ran, what passed). If not,
  `VERDICT: fail` and the blockers.

You are the last line before the user sees the work. Be skeptical.
"""


def build_explorer(project: Project, *, model: str) -> Agent:
    """A read-only investigator agent — `read`/`grep`/`find`/`ls`
    only, scoped to the project root. No memory / workspace /
    living-plan: it answers one question and exits, so the
    integration layers would just be overhead (and loomflow's
    fast-mode flags keep it lean)."""
    root = project.root
    return Agent(
        _EXPLORER_PROMPT,
        model=model,
        architecture=ReAct(),
        tools=[
            read_tool(root),
            grep_tool(root),
            find_tool(root),
            ls_tool(root),
        ],
        prompt_caching=True,
        max_turns=_SUBAGENT_MAX_TURNS,
    )


def build_reviewer(
    project: Project,
    *,
    model: str,
    approval_handler: Callable[..., Awaitable[bool]] | None = None,
) -> Agent:
    """A verification agent — read-only inspection tools plus
    `bash` to run the project's real test suite. `bash` is gated
    through the SAME approval handler as the main agent, so a
    reviewer-triggered command surfaces the same y/n prompt; it
    has no write/edit, so it can report but never 'fix'."""
    root = project.root
    return Agent(
        _REVIEWER_PROMPT,
        model=model,
        architecture=ReAct(),
        tools=[
            read_tool(root),
            grep_tool(root),
            find_tool(root),
            ls_tool(root),
            bash_tool(root, timeout=300.0),
        ],
        permissions=StandardPermissions(),
        approval_handler=approval_handler,
        prompt_caching=True,
        max_turns=_SUBAGENT_MAX_TURNS,
    )


def _as_subagent_tool(
    agent: Agent,
    *,
    name: str,
    description: str,
    arg_name: str,
    arg_description: str,
) -> Tool:
    """Wrap a loomflow ``Agent`` as a loomflow ``Tool``.

    The subagent runs with a FRESH session (isolated context — its
    file reads don't leak into the caller's window) but inherits
    ``user_id`` from the live ``RunContext`` so the multi-tenant
    partition is preserved. The tool returns the subagent's final
    output string straight back to the caller.
    """

    async def _call(**kwargs: Any) -> str:
        query = str(kwargs.get(arg_name, "")).strip()
        if not query:
            return f"(no {arg_name} provided — nothing to do)"
        ctx = get_run_context()
        # No session_id → the subagent gets its own fresh session.
        result = await agent.run(query, user_id=ctx.user_id)
        output = (result.output or "").strip()
        if not output:
            # A weak model can end a sub-run without a final text
            # turn. Never hand the caller a blank tool result — it
            # reads as "tool succeeded, found nothing", which is
            # wrong and misleads the main agent. Say so plainly so
            # it falls back to investigating itself.
            return (
                f"(the {name} sub-agent finished without producing "
                f"a written answer — investigate this directly "
                f"instead)"
            )
        return output

    return Tool(
        name=name,
        description=description,
        fn=_call,
        input_schema={
            "type": "object",
            "properties": {
                arg_name: {
                    "type": "string",
                    "description": arg_description,
                }
            },
            "required": [arg_name],
        },
    )


def build_subagent_tools(
    project: Project,
    *,
    model: str,
    approval_handler: Callable[..., Awaitable[bool]] | None = None,
) -> list[Tool]:
    """Build the `explore` and `review` tools for the main agent.

    Returns two ``Tool``s the main ReAct agent can call on its own
    judgement — `explore` to offload heavy read-only investigation,
    `review` to get an independent verification pass. Both run on
    the same ``model`` as the main agent (specialist behaviour
    comes from the prompt + tool scoping, not a weaker model).
    """
    explorer = build_explorer(project, model=model)
    reviewer = build_reviewer(
        project, model=model, approval_handler=approval_handler
    )
    return [
        _as_subagent_tool(
            explorer,
            name="explore",
            description=(
                "Delegate a read-only investigation of the codebase "
                "to a focused explorer agent. Use this instead of "
                "reading many files yourself when you need to "
                "understand how something works or where something "
                "is wired — it keeps that detail out of your "
                "context and returns just the answer. Ask one "
                "concrete question."
            ),
            arg_name="question",
            arg_description=(
                "A concrete question about the codebase, e.g. 'how "
                "is auth middleware wired into the request path?' "
                "or 'where are DB migrations defined and run?'"
            ),
        ),
        _as_subagent_tool(
            reviewer,
            name="review",
            description=(
                "Hand a just-completed change to an independent "
                "reviewer agent. It re-reads the changed files, "
                "runs the project's tests, and reports blockers / "
                "risks / nits with a pass/fail verdict. Use this as "
                "your VERIFY step on non-trivial changes — a second "
                "pair of eyes that didn't write the code."
            ),
            arg_name="focus",
            arg_description=(
                "What to review: which files changed and what the "
                "change was meant to do, e.g. 'reviewed calc.py — "
                "added divide(); should raise on divide-by-zero'."
            ),
        ),
    ]
