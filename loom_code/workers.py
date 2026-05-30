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

Memory propagation became real in loomflow 0.10.15 (before that,
``Team.supervisor(memory=...)`` propagated to the coordinator
only and workers silently fell back to ephemeral
``InMemoryMemory``). Combined with ``persist_tool_transcripts=True``
on each worker (also 0.10.15+), the worker's ``read`` / ``edit`` /
``bash`` results land in the coordinator's sqlite memory keyed
by the worker's stable session_id — so the same worker delegated
to twice no longer re-reads the same file. See the per-worker
constructors below for the wiring and the ``BUILD_LOG`` for the
diagnosis that led to this.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from loomflow import Agent, StandardPermissions
from loomflow.architecture import ReAct
from loomflow.tools import (
    bash_tool,
    find_tool,
    ls_tool,
    read_tool,
    write_tool,
)

from .code_index import codebase_search_tool
from .edit_tool import multi_edit_tool
from .edit_tool import verifying_edit_tool as edit_tool
from .extensions import AgentSpec
from .grep_tool import enhanced_grep_tool as grep_tool
from .lsp_tools import lsp_tools
from .project import Project
from .prompts import build_coder_prompt
from .web_fetch import web_fetch_tool

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

You have read-only tools: `read`, `grep`, `find`, `ls` (all
scoped to the project root) and `web_fetch` (for HTTPS URLs and
raw GitHub files). You have NO write/edit/bash — you cannot
change anything, and must not try.

How you work:
- Start broad (`find` / `ls` / `grep` for the relevant symbols),
  then `read` the files that matter.
- Follow the actual wiring — imports, call sites, config — don't
  guess.
- For URLs the lead names (a GitHub link, README, doc page,
  spec), use `web_fetch(url=...)` — never substitute a local
  file for a remote source you were asked to inspect. GitHub
  blob URLs auto-rewrite to raw, so you can paste the human URL.
  If you need a full repo (not a single file), say so in your
  report — full clones need `bash git clone`, which only `coder`
  has; the lead can re-route.
- Answer concretely. Cite `path:line` for every claim. Quote the
  key code, don't paraphrase it away.
- If the question has sub-parts, answer each.
- End with a short, direct summary the lead can act on.

**When your finding is non-trivial, write ONE finding note.** If
you uncovered something a teammate would otherwise have to re-
investigate (where a subsystem lives, how a flow wires together, a
gotcha), call `note(kind="finding", title="<short, searchable>",
content=<your findings, including path:line citations>)` so the
next fresh-session worker reuses it. Make the title keyword-rich.
But for a quick lookup or a one-line answer, SKIP the note — your
report to the lead is enough; a note that just restates a simple
answer is noise that costs everyone tokens.

Be exhaustive on facts, terse on prose — wasted words cost the
lead context. No summary documents or banners; your report is the
only thing read.
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

**When you found real issues, write ONE finding note.** If the
audit turned up blockers/risks worth preserving, call
`note(kind="finding", title="<area>: <severity gist>",
content=<your tagged findings with path:line citations>)` so the
lead and next worker pick it up via `search_notes()` instead of
re-auditing. If you found nothing notable, SKIP the note and just
say so in your report — don't write a note to record "no issues".

End with a one-line summary: how many blockers / risks / nits. No
summary documents or banners — the report is the only thing read.
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


def _read_only_tools(
    project: Project,
    embedder: str | None = None,
    workspace: Any | None = None,
) -> list[Any]:
    """The read-only inspection kernel — `read`/`grep`/`find`/`ls`
    scoped to the project root, plus `web_fetch` for reaching URLs
    and GitHub raw files (read-only by construction — no disk write,
    no shell). Shared by explorer + auditor; the reviewer adds
    `bash` on top.

    ``web_fetch`` closes the URL-fetch gap that previously forced
    the read-only specialists to silently substitute local files
    for remote sources; preserves the sole-writer invariant because
    the tool literally cannot write.

    ``embedder`` (``"openai"`` / ``"hash"``) — when set, adds the
    read-only ``codebase_search`` semantic tool so explorers/auditors
    can find code by meaning, not just grep strings. ``None`` (the
    default) keeps the legacy kernel for any caller that hasn't wired
    the embedder yet."""
    root = project.root
    tools: list[Any] = [
        read_tool(root),
        grep_tool(root),
        find_tool(root),
        ls_tool(root),
        web_fetch_tool(),
        # LSP navigation (jedi) — go_to_definition / find_references /
        # hover. Read-only by construction (static analysis, no disk
        # write); given to every worker so explorers/auditors navigate
        # by symbol, not grep. Python only; no embedder needed.
        *lsp_tools(root),
    ]
    if embedder is not None:
        # Same embedder name the coordinator + memory use, so every
        # agent searches the one shared index. ``workspace`` (when
        # given) fuses learned notes into the results (Phase 1b).
        # Read-only by construction — no disk write.
        tools.insert(
            2, codebase_search_tool(root, embedder, workspace=workspace)
        )
    return tools


def _build_coder(
    project: Project,
    *,
    model: str,
    approval_handler: Callable[..., Awaitable[bool]] | None,
    has_web: bool = False,
    skills: list[Any] | None = None,
    auto_compact_at_tokens: int | None = None,
    snip_window: int = 0,
    effort: str | None = None,
    mcp_registry: Any | None = None,
    sandbox: bool = False,
    sandbox_allow_network: bool = False,
    embedder: str | None = None,
    workspace: Any | None = None,
) -> Agent:
    """The doer. Full file-and-shell kernel, scoped to the project
    root. `StandardPermissions` gates the destructive tools
    (write / edit / bash) through the shared approval handler.

    ``embedder`` adds the read-only ``codebase_search`` tool — the
    coder uses it to locate the right code to change before editing,
    not just grep for strings.

    ``sandbox=True`` swaps the plain ``bash`` for the kernel-sandboxed
    one (``sandboxed_bash_tool``): the shell command runs inside
    sandbox-exec (macOS) / bwrap (Linux) so it can only WRITE under the
    project root and has NO network (unless ``sandbox_allow_network``).
    Claude-Code-style — only ``bash`` (arbitrary code) is sandboxed;
    ``edit``/``write`` keep the approval gate. Off by default.
    ``has_web`` toggles the `web_search` section in the prompt —
    keep this in lockstep with whether ``build_workers`` actually
    attaches the tool, else the prompt lies.

    ``mcp_registry`` (an ``MCPRegistry``, typed ``Any`` to avoid a hard
    ``mcp``-extra import) adds the user's MCP-server tools to the coder
    — the ONLY worker that gets them, since it's the sole writer/executor.
    When set, the coder's static tools are wrapped in an
    ``McpAugmentedHost`` so MCP tools resolve lazily (connect-on-first-
    use) and static builtins win any name collision."""
    root = project.root
    # bash is the one tool that runs arbitrary code, so it's the one we
    # kernel-sandbox when asked. edit/write only touch where the model
    # says + go through the approval gate, so they stay as-is.
    if sandbox:
        from .sandboxed_bash import sandboxed_bash_tool

        bash = sandboxed_bash_tool(
            root, allow_network=sandbox_allow_network, timeout=300.0
        )
    else:
        bash = bash_tool(root, timeout=300.0)
    static_tools: list[Any] = [
        read_tool(root),
        write_tool(root),
        edit_tool(root),
        multi_edit_tool(root),
        grep_tool(root),
        find_tool(root),
        ls_tool(root),
        bash,
        web_fetch_tool(),
        # LSP navigation (jedi) — the writer locates the symbol to
        # change by resolution, not grep, before editing. Read-only.
        *lsp_tools(root),
    ]
    if embedder is not None:
        # Semantic search for the writer too — locate the code to
        # change by meaning before editing. Same shared index.
        static_tools.insert(
            5, codebase_search_tool(root, embedder, workspace=workspace)
        )
    # Default: pass the static list straight through (framework wraps it
    # in an InProcessToolHost). With MCP, build that host ourselves and
    # compose it with the registry as one ToolHost.
    tools: Any = static_tools
    if mcp_registry is not None:
        from loomflow.tools.registry import InProcessToolHost

        from .mcp_host import McpAugmentedHost

        tools = McpAugmentedHost(
            InProcessToolHost(static_tools), mcp_registry
        )
    return Agent(
        build_coder_prompt(project, has_web=has_web),
        model=model,
        architecture=ReAct(),
        tools=tools,
        # Bundled skills (graphify, etc.) — registered on workers
        # too, not just the coordinator. Without this, when the
        # coordinator delegates "build the graph" to coder, the
        # coder spawns with its own tool host that doesn't have
        # ``graphify__build`` — and falls back to ``bash
        # graphify__build`` which doesn't exist. Skill on worker
        # = tool actually callable wherever execution lands.
        skills=skills,
        permissions=StandardPermissions(),
        approval_handler=approval_handler,
        prompt_caching=True,
        max_turns=_CODER_MAX_TURNS,
        # Bounded-window trim of the rehydrated history before each
        # run — THE active context bound. Without it a worker
        # rehydrating a heavily-used session's accumulated tool
        # transcripts (many ≤50KB entries) overflows the model window
        # and 400s (context_length_exceeded). The coordinator has it;
        # workers must too. (``auto_compact_at_tokens`` below only
        # fires between Ralph stop-hook iterations, which loom-code
        # disables via max_stop_hook_iterations=0 — so snip is what
        # actually protects a single run.)
        snip_window=snip_window,
        auto_compact_at_tokens=auto_compact_at_tokens,
        effort=effort,
        # Persistent tool-transcripts (loomflow 0.10.15+) — without
        # this the coder forgets every file read / edit / bash
        # output between delegations, even though its session_id
        # is preserved. Re-reading the same file 5x per task is
        # the single biggest token leak in long sessions; flipping
        # this on makes session_messages() rehydrate the prior
        # tool transcript so the coder QUOTES what it read instead
        # of re-running `read`.
        persist_tool_transcripts=True,
    )


def _build_explorer(
    project: Project,
    *,
    model: str,
    has_web: bool = False,
    skills: list[Any] | None = None,
    auto_compact_at_tokens: int | None = None,
    snip_window: int = 0,
    effort: str | None = None,
    embedder: str | None = None,
    workspace: Any | None = None,
) -> Agent:
    """Read-only investigator — no permissions needed (none of its
    tools are destructive). ``has_web`` toggles the `web_search`
    section in the prompt — must match ``build_workers``' wiring.
    ``embedder`` adds the read-only ``codebase_search`` semantic tool
    (the explorer is the prime beneficiary — concept-level lookups)."""
    return Agent(
        _explorer_prompt(has_web),
        model=model,
        architecture=ReAct(),
        tools=_read_only_tools(project, embedder, workspace),
        skills=skills,
        prompt_caching=True,
        max_turns=_SPECIALIST_MAX_TURNS,
        snip_window=snip_window,
        auto_compact_at_tokens=auto_compact_at_tokens,
        effort=effort,
        # See ``_build_coder`` for the rationale. Explorer benefits
        # too: a question like "how does X work, then check Y" no
        # longer re-greps + re-reads X's files when Y comes in as
        # a follow-up via ``send_message``.
        persist_tool_transcripts=True,
    )


def _build_auditor(
    project: Project,
    *,
    model: str,
    skills: list[Any] | None = None,
    auto_compact_at_tokens: int | None = None,
    snip_window: int = 0,
    effort: str | None = None,
    embedder: str | None = None,
    workspace: Any | None = None,
) -> Agent:
    """Read-only defect hunter — same tool scope as the explorer,
    different objective."""
    return Agent(
        _AUDITOR_PROMPT,
        model=model,
        architecture=ReAct(),
        tools=_read_only_tools(project, embedder, workspace),
        skills=skills,
        prompt_caching=True,
        max_turns=_SPECIALIST_MAX_TURNS,
        snip_window=snip_window,
        auto_compact_at_tokens=auto_compact_at_tokens,
        effort=effort,
        # Same rationale as explorer — auditor accumulates context
        # about the focus area across rounds when its findings get
        # iterated on.
        persist_tool_transcripts=True,
    )


def _build_reviewer(
    project: Project,
    *,
    model: str,
    approval_handler: Callable[..., Awaitable[bool]] | None,
    skills: list[Any] | None = None,
    auto_compact_at_tokens: int | None = None,
    snip_window: int = 0,
    effort: str | None = None,
    embedder: str | None = None,
    workspace: Any | None = None,
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
        tools=[
            *_read_only_tools(project, embedder, workspace),
            bash_tool(root, timeout=300.0),
        ],
        skills=skills,
        permissions=StandardPermissions(),
        approval_handler=approval_handler,
        prompt_caching=True,
        max_turns=_REVIEWER_MAX_TURNS,
        snip_window=snip_window,
        auto_compact_at_tokens=auto_compact_at_tokens,
        effort=effort,
        # Reviewer benefits too: re-review cycles ("you flagged X,
        # the coder fixed it, recheck") no longer re-read every
        # changed file from scratch.
        persist_tool_transcripts=True,
    )


def build_workers(
    project: Project,
    *,
    model: str,
    approval_handler: Callable[..., Awaitable[bool]] | None = None,
    web_backend: str | None = None,
    skills: list[Any] | None = None,
    auto_compact_at_tokens: int | None = None,
    snip_window: int = 0,
    effort: str | None = None,
    mcp_registry: Any | None = None,
    sandbox: bool = False,
    sandbox_allow_network: bool = False,
    embedder: str | None = None,
    workspace: Any | None = None,
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
            skills=skills,
            auto_compact_at_tokens=auto_compact_at_tokens,
            snip_window=snip_window,
            effort=effort,
            mcp_registry=mcp_registry,
            sandbox=sandbox,
            sandbox_allow_network=sandbox_allow_network,
            embedder=embedder,
            workspace=workspace,
        ),
        "explorer": _build_explorer(
            project,
            model=model,
            has_web=has_web,
            skills=skills,
            auto_compact_at_tokens=auto_compact_at_tokens,
            snip_window=snip_window,
            effort=effort,
            embedder=embedder,
            workspace=workspace,
        ),
        "auditor": _build_auditor(
            project,
            model=model,
            skills=skills,
            auto_compact_at_tokens=auto_compact_at_tokens,
            snip_window=snip_window,
            effort=effort,
            embedder=embedder,
            workspace=workspace,
        ),
        "reviewer": _build_reviewer(
            project,
            model=model,
            approval_handler=approval_handler,
            skills=skills,
            auto_compact_at_tokens=auto_compact_at_tokens,
            snip_window=snip_window,
            effort=effort,
            embedder=embedder,
            workspace=workspace,
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


# Builtin worker role names — protected. A user-authored subagent that
# names itself one of these is skipped rather than allowed to shadow
# the known roster (especially ``coder``, the sole writer).
BUILTIN_WORKER_NAMES = frozenset(
    {"coder", "explorer", "auditor", "reviewer"}
)

# Tool names a custom subagent's ``tools:`` frontmatter may request,
# mapped to the same builtin factories the builtin workers use.
# ``web_search`` is intentionally absent — it needs backend wiring
# (``/set_web``); custom agents get ``web_fetch`` (always available,
# read-only, no shell/disk write) instead.
_DESTRUCTIVE_TOOL_NAMES = frozenset(
    {"write", "edit", "multi_edit", "bash"}
)

# When a spec declares no ``tools:``, this read-only kernel is the
# default — we never hand a stranger's spec write/shell access
# implicitly. Matches ``_read_only_tools``.
_DEFAULT_CUSTOM_TOOLS = ("read", "grep", "find", "ls", "web_fetch")


def _custom_tool_factories(root: Any) -> dict[str, Callable[[], Any]]:
    """Map tool name → a zero-arg factory rooted at the project.

    Unknown names a spec requests simply aren't in this map and are
    skipped by :func:`build_custom_worker` — a typo'd tool name costs
    that tool, not the whole agent."""
    return {
        "read": lambda: read_tool(root),
        "write": lambda: write_tool(root),
        "edit": lambda: edit_tool(root),
        "multi_edit": lambda: multi_edit_tool(root),
        "grep": lambda: grep_tool(root),
        "find": lambda: find_tool(root),
        "ls": lambda: ls_tool(root),
        "bash": lambda: bash_tool(root, timeout=300.0),
        "web_fetch": lambda: web_fetch_tool(),
    }


def build_custom_worker(
    project: Project,
    spec: AgentSpec,
    *,
    model: str,
    approval_handler: Callable[..., Awaitable[bool]] | None,
    skills: list[Any] | None = None,
    auto_compact_at_tokens: int | None = None,
    snip_window: int = 0,
    effort: str | None = None,
) -> Agent:
    """Build a delegate-able worker Agent from a user-authored subagent
    spec (``.loom/agents/<name>.md`` — see :mod:`loom_code.extensions`).

    The worker's ``instructions`` LEAD with the frontmatter
    ``description`` followed by the markdown body, because
    ``Supervisor`` shows the coordinator only the first ~200 chars of
    each worker's instructions (not a separate description field) — so
    the description must come first for the coordinator to route to it
    correctly.

    Tools come from the spec's ``tools:`` list mapped through
    :func:`_custom_tool_factories`; an empty list defaults to the
    read-only kernel. Permissions + the approval handler are wired ONLY
    when the requested tools include a destructive one
    (write/edit/multi_edit/bash) — a read-only subagent needs no gate.
    ``model`` falls back to the coordinator's model when the spec
    doesn't override it.
    """
    root = project.root
    factories = _custom_tool_factories(root)
    requested = spec.tools or _DEFAULT_CUSTOM_TOOLS
    tools: list[Any] = [
        factories[name]() for name in requested if name in factories
    ]
    has_destructive = any(
        name in _DESTRUCTIVE_TOOL_NAMES for name in requested
    )

    instructions = spec.description
    if spec.system_prompt:
        instructions = f"{spec.description}\n\n{spec.system_prompt}"

    return Agent(
        instructions,
        model=spec.model or model,
        architecture=ReAct(),
        tools=tools,
        skills=skills,
        permissions=StandardPermissions() if has_destructive else None,
        approval_handler=approval_handler if has_destructive else None,
        prompt_caching=True,
        # Writers get the full budget; read-only specialists answer a
        # scoped question and exit (same split as the builtin roster).
        max_turns=(
            _CODER_MAX_TURNS if has_destructive else _SPECIALIST_MAX_TURNS
        ),
        snip_window=snip_window,
        auto_compact_at_tokens=auto_compact_at_tokens,
        effort=effort,
        persist_tool_transcripts=True,
    )
