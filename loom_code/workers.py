"""The worker roster for loom-code's ``Team.supervisor``.

loom-code is a hierarchical team: a coordinator Agent (the tech
lead) delegates to these workers via loomflow's ``Supervisor``
architecture. Each worker is a full loomflow ``Agent`` with a
``ReAct`` loop — the coordinator hands it a focused task through
the ``delegate`` tool and it runs to completion.

The roster is sliced by VERB, and one invariant holds it together:

* **coder** — the ONLY writer. Full file-and-shell kernel
  (read/write/edit/grep/find/ls/bash). Every actual change to the
  codebase happens here, one delegation at a time.
* **explorer** — read-only investigation → a briefing.
* **auditor** — read-only defect hunt (security / perf /
  correctness lens) → tagged findings.
* **reviewer** — read-only inspection + ``bash`` to run the
  project's tests → a pass/fail verdict.

Because only ``coder`` writes, the coordinator can delegate the
three read-only workers in parallel with zero risk of filesystem
races (loomflow's Supervisor gets parallel delegation for free —
ReAct dispatches multiple ``delegate`` calls in one turn through
an ``anyio`` task group). The coordinator serialises ``coder``
delegations itself.

Workers inherit the shared notebook (``workspace=``) and the
coordinator's memory via loomflow's ambient propagation — they
are NOT given their own, so there's one notebook and one memory
db for the whole team.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from loomflow import Agent, StandardPermissions
from loomflow.architecture import ReAct
from loomflow.tools import (
    bash_tool,
    edit_tool,
    find_tool,
    grep_tool,
    ls_tool,
    read_tool,
    write_tool,
)

from .project import Project
from .prompts import build_coder_prompt

# The coder does real, multi-step work — it gets a generous turn
# budget. Read-only specialists answer a scoped question and exit,
# so they're capped tighter; the reviewer sits in the middle
# because running a test suite + iterating on failures legitimately
# takes more turns than answering one question.
_CODER_MAX_TURNS = 60
_SPECIALIST_MAX_TURNS = 20  # explorer + auditor — one scoped question
_REVIEWER_MAX_TURNS = 30  # tests can iterate

_EXPLORER_PROMPT = """\
You are the EXPLORER on a loom-code team — a read-only
investigator. A tech lead delegates ONE question about the
codebase to you. You answer it thoroughly and hand the answer
back.

You have read-only tools: `read`, `grep`, `find`, `ls`. You have
NO write/edit/bash — you cannot change anything, and must not try.

How you work:
- Start broad (`find` / `ls` / `grep` for the relevant symbols),
  then `read` the files that matter.
- Follow the actual wiring — imports, call sites, config — don't
  guess.
- Answer concretely. Cite `path:line` for every claim. Quote the
  key code, don't paraphrase it away.
- If the question has sub-parts, answer each.
- End with a short, direct summary the lead can act on.

**Before you finish, write a finding note.** Call
`note(kind="finding", title="<short, searchable>", content=<your
findings, including path:line citations>)`. The lead and the
next worker run in fresh sessions — your note in the notebook is
how they avoid re-investigating what you just figured out. Make
the title keyword-rich so `search_notes()` finds it.

Be exhaustive on facts, terse on prose — wasted words cost the
lead context.
"""

# Appended onto _EXPLORER_PROMPT only when the explorer was built
# with a web_tool — promising a tool the agent doesn't have wastes
# turns on failed tool calls.
_EXPLORER_WEB_HINT = """\

You also have `web_search(query=...)` for investigation that goes
*outside* the codebase — an upstream library's documented
behaviour, an external API's contract, a CVE / errata page, the
known cause of a third-party error message. Use it AFTER you've
read the relevant project code, not instead. Keyword queries beat
sentences. Cite the source URL in your finding note so the lead
can verify it.
"""


def _explorer_prompt(has_web: bool) -> str:
    """The explorer's system prompt. Web-search hint is opt-in so
    the agent isn't told about a tool it doesn't have."""
    if has_web:
        return _EXPLORER_PROMPT + _EXPLORER_WEB_HINT
    return _EXPLORER_PROMPT

_AUDITOR_PROMPT = """\
You are the AUDITOR on a loom-code team — a read-only inspector.
A tech lead delegates a focus area and a lens (security,
performance, or correctness). Your job: hunt for PROBLEMS.

You have read-only tools: `read`, `grep`, `find`, `ls`. You have
NO write/edit/bash — you find problems, you do not fix them.

How you work:
- Read the code in the focus area carefully. Trace inputs to
  where they're used.
- Through your lens, look hard for concrete defects:
  - security: injection, unsanitised input, secrets in code,
    path traversal, unsafe deserialization, missing authz.
  - performance: N+1 patterns, work in hot loops, unbounded
    growth, sync I/O on a hot path.
  - correctness: unhandled edge cases, off-by-one, swallowed
    errors, race conditions, wrong input assumptions.
- Report each finding as a list item tagged severity:
  `[blocker]` — a real bug / vulnerability, must fix.
  `[risk]`    — likely wrong or fragile, worth a closer look.
  `[nit]`     — minor, optional.
- Cite `path:line` for every finding. Quote the offending code.
- If you find nothing real, say so — do NOT invent problems to
  look thorough.

**Before you finish, write a finding note.** Call
`note(kind="finding", title="<area>: <severity gist>",
content=<your tagged findings with path:line citations>)`. Cross-
turn memory — the lead and the next worker pick this up via
`search_notes()` instead of re-auditing the same area.

End with a one-line summary: how many blockers / risks / nits.
"""

_REVIEWER_PROMPT = """\
You are the REVIEWER on a loom-code team — a verification
specialist. A tech lead delegates a description of a change that
was just made. Your job: independently confirm it is correct,
complete, and safe.

You have `read`, `grep`, `find`, `ls`, and `bash`. Use `bash` to
run the project's OWN tests / linters / build — not improvised
checks. You have NO write/edit — you do not fix things, you
REPORT.

How you work:
- Re-read the changed files yourself. Don't trust the description.
- Run the verification command (test suite, build, type-check).
- Look for: broken callers, missing edge cases, untested paths,
  things the change claimed but didn't do, regressions.
- Report findings as a list, each tagged severity:
  `[blocker]` — must fix before this is done.
  `[risk]`    — probably wrong / fragile, worth a second look.
  `[nit]`     — minor, optional.
- If everything checks out, say so plainly: `VERDICT: pass` plus
  the evidence (which tests ran, what passed). Otherwise
  `VERDICT: fail` and the blockers.

You are the last line before the user sees the work. Be skeptical.
"""


def _read_only_tools(project: Project) -> list[Any]:
    """The read-only inspection kernel — `read`/`grep`/`find`/`ls`,
    all scoped to the project root. Shared by explorer + auditor;
    the reviewer adds `bash` on top."""
    root = project.root
    return [
        read_tool(root),
        grep_tool(root),
        find_tool(root),
        ls_tool(root),
    ]


def _build_coder(
    project: Project,
    *,
    model: str,
    approval_handler: Callable[..., Awaitable[bool]] | None,
    has_web: bool = False,
) -> Agent:
    """The doer. Full file-and-shell kernel, scoped to the project
    root. `StandardPermissions` gates the destructive tools
    (write / edit / bash) through the shared approval handler.
    ``has_web`` toggles the `web_search` section in the prompt —
    keep this in lockstep with whether ``build_workers`` actually
    attaches the tool, else the prompt lies."""
    root = project.root
    return Agent(
        build_coder_prompt(project, has_web=has_web),
        model=model,
        architecture=ReAct(),
        tools=[
            read_tool(root),
            write_tool(root),
            edit_tool(root),
            grep_tool(root),
            find_tool(root),
            ls_tool(root),
            bash_tool(root, timeout=300.0),
        ],
        permissions=StandardPermissions(),
        approval_handler=approval_handler,
        prompt_caching=True,
        max_turns=_CODER_MAX_TURNS,
    )


def _build_explorer(
    project: Project, *, model: str, has_web: bool = False
) -> Agent:
    """Read-only investigator — no permissions needed (none of its
    tools are destructive). ``has_web`` toggles the `web_search`
    section in the prompt — must match ``build_workers``' wiring."""
    return Agent(
        _explorer_prompt(has_web),
        model=model,
        architecture=ReAct(),
        tools=_read_only_tools(project),
        prompt_caching=True,
        max_turns=_SPECIALIST_MAX_TURNS,
    )


def _build_auditor(project: Project, *, model: str) -> Agent:
    """Read-only defect hunter — same tool scope as the explorer,
    different objective."""
    return Agent(
        _AUDITOR_PROMPT,
        model=model,
        architecture=ReAct(),
        tools=_read_only_tools(project),
        prompt_caching=True,
        max_turns=_SPECIALIST_MAX_TURNS,
    )


def _build_reviewer(
    project: Project,
    *,
    model: str,
    approval_handler: Callable[..., Awaitable[bool]] | None,
) -> Agent:
    """Independent verifier — read-only inspection plus `bash` to
    run the project's real test suite. `bash` is gated through the
    same approval handler as the coder; it has no write/edit, so
    it reports but never fixes."""
    root = project.root
    return Agent(
        _REVIEWER_PROMPT,
        model=model,
        architecture=ReAct(),
        tools=[*_read_only_tools(project), bash_tool(root, timeout=300.0)],
        permissions=StandardPermissions(),
        approval_handler=approval_handler,
        prompt_caching=True,
        max_turns=_REVIEWER_MAX_TURNS,
    )


def build_workers(
    project: Project,
    *,
    model: str,
    approval_handler: Callable[..., Awaitable[bool]] | None = None,
    web_backend: str | None = None,
) -> dict[str, Agent]:
    """Build the worker roster for ``Team.supervisor``.

    Returns ``{"coder", "explorer", "auditor", "reviewer"}`` — the
    dict keys become each worker's delegate name AND its author
    identity in the shared notebook. All four run on the same
    ``model`` as the coordinator; the specialism is in the prompt
    + tool scoping, not a weaker model.

    Only ``coder`` and ``reviewer`` get a permissions policy +
    approval handler (they hold destructive tools); ``explorer``
    and ``auditor`` are purely read-only.

    ``web_backend``: ``"serper"`` or ``"duckduckgo"`` to enable
    ``loomflow.tools.web_tool`` on ``coder`` + ``explorer``. The
    coder needs it to look up library APIs while implementing;
    the explorer for investigation that goes beyond the codebase.
    Auditor + reviewer stay read-only-and-local (no web access)
    — keeps their cost predictable and their scope honest.
    ``None`` (default) leaves web search off entirely.
    """
    has_web = web_backend is not None
    workers: dict[str, Agent] = {
        "coder": _build_coder(
            project,
            model=model,
            approval_handler=approval_handler,
            has_web=has_web,
        ),
        "explorer": _build_explorer(
            project, model=model, has_web=has_web
        ),
        "auditor": _build_auditor(project, model=model),
        "reviewer": _build_reviewer(
            project, model=model, approval_handler=approval_handler
        ),
    }
    if has_web:
        # One shared web_tool instance — same Tool object on both
        # workers. Cheap; nothing in the tool's lifecycle is per-
        # worker. If the backend is misconfigured (e.g. serper
        # without a key) ``web_tool`` raises ConfigError here; the
        # caller (REPL's /set_web) is expected to validate first.
        # ``has_web=True`` was already threaded into the prompts —
        # the model knows the tool exists; here we actually attach
        # it.
        from loomflow.tools import web_tool
        web = web_tool(backend=web_backend)  # type: ignore[arg-type]
        workers["coder"].add_tool(web)
        workers["explorer"].add_tool(web)
    return workers
