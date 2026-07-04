"""The loom-code interactive REPL.

``loom-code`` with no args drops here. You chat, it codes, it asks
before destructive changes, you keep going — the Claude-Code / Pi
loop. Conversation continuity comes free: every turn reuses the
same ``session_id``, so loomflow rehydrates prior turns as real
chat history.

The self-improvement loop (Phase 3)
-----------------------------------

Every turn, the agent READS notes from the project notebook —
past plans, past findings (``recall_past_plans``, ``search_notes``,
``read_note``). loomflow records those reads as *citations* on
``RunResult.cited_slugs``. When a turn is judged successful, we
call ``workspace.attribute_outcome(success=True, slugs=...)`` — the
cited notes' ``cited_count`` / ``success_count`` climb, and future
``search_notes(boost_relevance=True)`` ranks them higher.

How "success" is judged — the **moved-on heuristic**:

* We DON'T attribute immediately. We hold the last turn's
  ``cited_slugs`` as ``pending``.
* If you give loom-code another task without complaint, the
  previous turn must have been fine → attribute the pending as
  ``success=True``.
* ``/bad`` attributes pending as ``success=False`` (it broke
  something / wasn't useful).
* ``/good`` attributes pending as ``success=True`` immediately.
* On ``/exit``, any pending is attributed ``success=True`` — you
  left satisfied.

That matches how a developer actually signals: silence + moving
on means "worked", an explicit "no" means "didn't".

Slash commands are handled here, never sent to the agent. The full
list is defined once in :data:`_COMMAND_DEFS` (grouped) and rendered
by :func:`_render_help` for ``/help`` and the autocomplete menu — see
there rather than duplicating the catalogue in this docstring.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import anyio
from loomflow import new_id
from prompt_toolkit import HTML, PromptSession
from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
)
from prompt_toolkit.document import Document
from rich.text import Text

from . import checkpoint as _checkpoint
from . import file_history, worktree
from .agent import LOOM_DIR, build_agent, build_solo_agent
from .approval import ApprovalGate
from .compact import Compactor, default_compact_threshold
from .credentials import (
    cheap_model_for,
    ensure_key_for_model,
    save_credential,
)
from .extensions import Extensions, HookSpec
from .extensions import discover as discover_extensions
from .hooks import run_repl_hooks
from .paste import (
    build_paste_keybindings,
    expand_pastes,
    reset_paste_stash,
)
from .project import Project, detect_project
from .render import StreamRenderer, banner, console
from .trust import filter_trusted_hooks

# Provider defaults for /set_model — picking a provider switches
# to that provider's commonly-used model.
_OPENAI_DEFAULT_MODEL = "gpt-4.1-mini"
_ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-6"

_USER_ID = "loom-code"

# DEFAULT cap on auto-continue iterations per turn. This is the
# Ralph-loop / Cursor-judge-agent pattern: the model's "I'm done"
# judgement is unreliable on multi-step plans, so the REPL
# overrules it as long as the plan explicitly disagrees.
#
# Bumped from 5 → 15 after empirical observation: real scaffold
# tasks the user threw at us had 6-12 plan steps, and 5 left them
# stuck mid-stream. 15 gives headroom; stall detection still kicks
# in early on genuinely-runaway loops so the higher cap doesn't
# inflate worst-case cost. Per-session overridable via
# ``/set_continue_cap N`` for power users who want more or less.
_AUTO_CONTINUE_LIMIT_DEFAULT = 15

# Consecutive IDENTICAL tool calls (same tool, same args) before the
# stall detector aborts the turn. loomflow's own no-progress hook only
# arms under /goal; this guards the everyday interactive path. 4 is
# high enough that legitimate retry-once patterns never trip it.
_STALL_REPEATS = 4

# @-completion caches the project file list this long (seconds) and
# filters it in-memory per keystroke, instead of re-walking the tree
# on every character — a cold walk per keypress froze the prompt on a
# large repo. Bounded so a huge monorepo can't build an enormous list.
_FILE_CACHE_TTL = 4.0
_FILE_CACHE_MAX = 20_000

# Pure greetings answered locally, zero tokens. Short prompts route to
# the heavy TEAM path (the anaphora rule), so before this a bare "hi"
# cost the full coordinator context (~6.6k tokens, observed live) and
# sometimes a hallucinated delegation on weak models. EXACT matches
# only — "ok"/"thanks" are moved-on feedback signals, and anything
# with more content deserves the model.
_GREETINGS = frozenset({
    "hi", "hello", "hey", "yo", "hiya", "hola", "sup",
    "good morning", "good afternoon", "good evening",
    "hi there", "hello there", "hey there",
})


def _greeting_reply(prompt: str) -> str | None:
    """A canned local reply when ``prompt`` is a bare greeting, else
    None (run the model normally)."""
    p = re.sub(r"[!.,\s]+$", "", prompt.strip().lower())
    if p in _GREETINGS:
        return (
            "hi! give me a coding task — or [cyan]/help[/cyan] "
            "for the command list."
        )
    return None


def _context_high_water(
    prev: int, *, tokens_in: int, cached_in: int
) -> int:
    """Update the compaction trigger's context-occupancy estimate.

    The last turn's INPUT tokens (uncached ``tokens_in`` + ``cached_in``)
    already represent the *entire* conversation sent to the model that
    turn — so their sum is a direct read of how full the context window
    is right now. We track the **high-water mark**, never a running sum:
    summing per-turn inputs double-counts, because each turn's input
    re-includes all prior history, so a cumulative counter races far past
    true occupancy and trips compaction much too early — discarding live
    file/edit state into a lossy prose summary mid-task.

    ``max`` (not plain assignment) so a brief dip — a short follow-up
    turn whose input momentarily shrinks — doesn't un-arm a compaction
    the conversation has genuinely grown to need. The counter is reset to
    0 by the caller on a fresh thread (compaction / clear / model switch /
    resume), where occupancy genuinely starts over.
    """
    return max(prev, tokens_in + cached_in)


# Tool names the agents actually expose — used to recognise a tool
# call that a weak model emitted as PLAIN TEXT instead of through the
# structured tool-calling interface. (Observed live with
# phi-4-mini: final answer was literally
# ``{ "name": "read", "parameters": {"path": "FileA.py"} }``.)
_KNOWN_TOOL_NAMES = frozenset({
    "read", "write", "edit", "multi_edit", "grep", "find", "ls",
    "bash", "web_fetch", "web_search", "delegate", "codebase_search",
    "plan_write", "plan_read", "note", "search_notes", "read_note",
    "list_notes", "remember_rule", "go_to_definition",
    "find_references", "hover",
})


def _looks_like_leaked_tool_call(text: str) -> bool:
    """True when a final answer is clearly a tool CALL written as
    prose — a bare JSON object naming a known tool with an args-like
    key. The ReAct loop treats "no structured tool calls" as "model
    is done", so without this guard the user sees raw JSON as the
    answer and the tool never runs."""
    t = text.strip()
    # Unwrap a single fenced code block (```json ... ```).
    if t.startswith("```") and t.endswith("```"):
        t = t.strip("`").strip()
        first, _, rest = t.partition("\n")
        if rest and len(first) <= 10:  # language tag line
            t = rest.strip()
    if not (t.startswith("{") and t.endswith("}")):
        return False
    import json

    try:
        obj = json.loads(t)
    except (ValueError, TypeError):
        return False
    if not isinstance(obj, dict):
        return False
    name = obj.get("name") or obj.get("tool") or obj.get("function")
    if isinstance(name, dict):  # OpenAI shape: {"function": {"name": ..}}
        name = name.get("name")
    if not isinstance(name, str):
        return False
    has_args = any(
        k in obj
        for k in ("parameters", "arguments", "args", "input")
    )
    return has_args and name.rsplit(".", 1)[-1] in _KNOWN_TOOL_NAMES


# The bounded corrective prompt sent when a leaked tool call is
# detected. One nudge per turn — if the model leaks again, we show
# the raw output rather than loop.
_TOOL_LEAK_NUDGE = (
    "Your previous reply was a tool invocation written as plain "
    "text, which cannot be executed. Either call the tool through "
    "the proper tool-calling interface, or answer the user's "
    "request directly in prose. Do not print JSON."
)


def friendly_error_hint(exc: BaseException) -> str | None:
    """An actionable one-liner for a model/provider error, or None
    when we have nothing better than the raw message.

    The classified exception (loomflow's PermanentModelError /
    RateLimitError / etc., or the provider SDK error inside it) is
    matched on type name + message text rather than imported types —
    keeps this working across loomflow versions and provider SDKs
    without hard dependencies. Checked most-specific first.
    """
    blob = f"{type(exc).__name__}: {exc}".lower()
    if "notfounderror" in blob or "404" in blob or "not found" in blob:
        return (
            "the model id wasn't found at the provider — check the "
            "spelling, or /set_model to pick another"
        )
    if (
        "authentication" in blob
        or "401" in blob
        or "invalid api key" in blob
        or "unauthorized" in blob
    ):
        return (
            "the API key was rejected — /set_model to re-enter it "
            "(or fix the env var / ~/.loom-code/credentials)"
        )
    if "ratelimit" in blob or "429" in blob or "rate limit" in blob:
        return (
            "the provider rate-limited us and retries ran out — "
            "wait a moment and try again, or /model to switch"
        )
    if "timeout" in blob or "timed out" in blob:
        return (
            "the turn timed out — try again, or narrow the ask; "
            "/undo restores the tree if it half-finished"
        )
    if "connection" in blob or "connect" in blob or "network" in blob:
        return (
            "couldn't reach the provider — check your network / "
            "VPN, then try again"
        )
    if "context" in blob and ("length" in blob or "window" in blob):
        return (
            "the conversation outgrew the model's context window — "
            "/compress_token_length to lower the auto-compact "
            "threshold, or /clear for a fresh session"
        )
    return None


def _flatten_exception_group(
    eg: BaseExceptionGroup,
) -> list[BaseException]:
    """Recursively unwrap nested ``BaseExceptionGroup`` into a flat
    list of the underlying exceptions.

    anyio task groups raise an ``ExceptionGroup`` whose default
    ``str()`` is "unhandled errors in a TaskGroup (N sub-exception)"
    — useless for the user. Flatten to surface what ACTUALLY went
    wrong (the wrapper might nest more wrappers if multiple groups
    were involved)."""
    out: list[BaseException] = []
    for inner in eg.exceptions:
        if isinstance(inner, BaseExceptionGroup):
            out.extend(_flatten_exception_group(inner))
        else:
            out.append(inner)
    return out

# The single source of truth for slash commands the REPL accepts.
# The autocomplete menu (popped the moment the user types '/')
# reads off this list, so adding a new command here is enough —
# Question-shaped prompts route to the TEAM coordinator (it answers
# read-only questions directly — no delegation tax — and holds the
# repo map). The heuristic only ever short-circuits TOWARD the team,
# so a false positive ("how about you rename X" → team) costs the
# status-quo overhead, never a capability.
_QUESTION_STARTERS = (
    "what", "when", "where", "which", "who", "why", "how",
    "is ", "are ", "does ", "do ", "did ", "can ", "could ",
    "should ", "would ", "will ", "explain", "show me", "tell me",
    "describe", "summarize", "summarise", "walk me through",
)


def _looks_like_question(prompt: str) -> bool:
    """True when the prompt reads as a question / explanation request
    rather than a change request."""
    p = prompt.strip().lower()
    return p.endswith("?") or p.startswith(_QUESTION_STARTERS)


# Pronouns/markers that point at conversation the SOLO/TEAM classifier
# cannot see. "fix it" after ten turns of discussion LOOKS trivial in
# isolation but may be a multi-file task — the team coordinator (which
# holds the session history) must take those.
_ANAPHORA_WORDS = frozenset(
    {"it", "that", "this", "them", "those", "these", "above", "again"}
)


def _references_prior_context(prompt: str) -> bool:
    """True when a SHORT prompt leans on prior conversation the
    stateless classifier can't see ("fix it", "continue", "do that
    again"). Long prompts may use "it" self-referentially and still
    classify fine, so only short ones short-circuit."""
    words = prompt.strip().lower().split()
    if len(words) <= 2:
        return True
    if len(words) <= 6 and any(
        w.strip(".,!:;") in _ANAPHORA_WORDS for w in words
    ):
        return True
    return False


# The SOLO/TEAM classifier's whole system prompt. One word out; the
# cheap model handles this reliably. Biased toward TEAM — solo only
# for tasks where skipping the team round-trip is a pure win.
_ROUTER_PROMPT = """\
You route tasks for a terminal coding agent. Reply with EXACTLY one
word: SOLO or TEAM.

SOLO — one small, focused change a single capable agent should just
do: a one-file edit or bugfix, a rename confined to one place, adding
one test, tweaking a config value, running or adjusting one command,
a small mechanical change with an obvious definition of done.

TEAM — everything else: multi-file features or refactors, anything
needing investigation first ("find out why...", "figure out..."),
work that warrants independent review or running a test suite to
verify, vague or large scope, external integrations, anything
destructive or wide-reaching. Also TEAM: any task that leans on
prior conversation you cannot see ("fix it", "do that again",
"continue", "the bug we discussed") — you see only this one message.

When in doubt: TEAM.
"""


# no need to also update the autocomplete separately.
#
# Each entry is (command, description, group). The GROUP tag lets
# /help print the commands clustered by purpose instead of as one
# flat 20-item wall — and both /help AND the autocomplete menu read
# off this one list, so there's a single source of truth. Add a
# command here and it shows up in both, correctly grouped, for free.
# Group order below is the order groups appear in /help.
_COMMAND_DEFS: list[tuple[str, str, str]] = [
    # Coding — the day-to-day task loop.
    ("/plan", "show the current plan, or start one", "Coding"),
    (
        "/goal",
        "work until a condition is met — durable across restarts "
        "(/goal resume)",
        "Coding",
    ),
    ("/undo", "restore the working tree to the last checkpoint", "Coding"),
    (
        "/checkpoints",
        "list auto-checkpoints (taken before each edit)",
        "Coding",
    ),
    ("/good", "mark the last turn useful (credit notes)", "Coding"),
    ("/bad", "mark the last turn unhelpful", "Coding"),
    (
        "/verify",
        "verify-before-done gate: on | off (nudges the agent to run "
        "tests before claiming done)",
        "Coding",
    ),
    ("/init-loom", "create a starter AGENTS.md rules file", "Coding"),
    # Isolation — sandbox this session in its own git worktree.
    ("/isolate", "run this session in its own git worktree", "Isolate"),
    ("/review", "show the isolated session's diff vs base", "Isolate"),
    ("/merge", "merge the isolated session's edits into base", "Isolate"),
    ("/discard", "discard the isolated session's edits", "Isolate"),
    # Model & tools — how the agent thinks and what it can reach.
    ("/model", "switch to a specific model by name", "Model"),
    ("/effort", "reasoning effort: low | medium | high | off", "Model"),
    (
        "/mode",
        "approval mode: default | accept-edits | plan | yolo",
        "Model",
    ),
    ("/set_model", "pick a provider + model (saves API key)", "Model"),
    ("/set_web", "enable web search (Serper / DuckDuckGo / off)", "Model"),
    ("/mcp", "list connected MCP servers + their tools", "Model"),
    # Session — state, cost, and history for the whole run.
    ("/cost", "session cost + token totals", "Session"),
    (
        "/context",
        "what's in the model's context right now (tokens + %)",
        "Session",
    ),
    (
        "/prompt",
        "dump the full system prompt + injected blocks",
        "Session",
    ),
    (
        "/resume",
        "resume the last session — or `/resume pick` to choose one",
        "Session",
    ),
    (
        "/fork",
        "branch this session — explore a tangent without touching it",
        "Session",
    ),
    ("/tree", "show the session tree (branches + forks)", "Session"),
    ("/export", "save this conversation to a markdown file", "Session"),
    (
        "/set_continue_cap",
        "set auto-continue cap (current=default 15)",
        "Session",
    ),
    (
        "/compact",
        "compact the conversation NOW (fold history into a summary)",
        "Session",
    ),
    (
        "/compress_token_length",
        "auto-compact threshold: <N> | auto | off",
        "Session",
    ),
    ("/clear", "fresh conversation (new session)", "Session"),
    ("/help", "show all commands", "Session"),
    ("/exit", "leave (Ctrl-D also works)", "Session"),
    # /computer (computer-operator mode) is HIDDEN for now — kept in code
    # + dispatched if typed, but not advertised in help/autocomplete
    # until it's ready to ship. Add an entry here to surface it again.
]

# Group order for /help — groups render in this sequence; any group
# not listed falls to the end in first-seen order.
_HELP_GROUP_ORDER = ("Coding", "Isolate", "Model", "Session")


def _render_help(
    extra: list[tuple[str, str, str]] | None = None,
) -> str:
    """Build the /help text from :data:`_COMMAND_DEFS` so it can never
    drift from the commands the REPL actually accepts (the old
    hand-maintained blob had silently lost a third of them). Commands
    cluster under their group header, aligned on the description.
    ``extra`` appends per-instance entries (custom markdown commands)
    in the same (cmd, desc, group) shape."""
    from collections import defaultdict

    defs = [*_COMMAND_DEFS, *(extra or [])]
    by_group: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for cmd, desc, group in defs:
        by_group[group].append((cmd, desc))
    order = [g for g in _HELP_GROUP_ORDER if g in by_group]
    order += [g for g in by_group if g not in _HELP_GROUP_ORDER]

    # Align descriptions on the widest command across ALL groups so the
    # columns line up down the whole list, not just within a group.
    width = max(len(cmd) for cmd, _, _ in defs)
    lines = ["[bold]loom-code commands[/bold]"]
    for group in order:
        lines.append(f"\n  [dim]{group}[/dim]")
        for cmd, desc in by_group[group]:
            pad = " " * (width - len(cmd))
            lines.append(f"    [cyan]{cmd}[/cyan]{pad}  {desc}")
    lines.append(
        "\nAnything else is a task — loom-code plans, codes, and "
        "verifies it.\nLong sessions auto-compact: when tokens cross "
        "the threshold, a\ncompactor writes a dense summary to memory "
        "and the run continues."
    )
    return "\n".join(lines)


class _SlashCompleter(Completer):
    """Two completions in one:

    * ``/`` at line start → the slash-command menu (filters as you
      type, so ``/co`` narrows to /cost + /compress_token_length).
    * ``@`` anywhere → a fuzzy file-path menu rooted at the project,
      so ``@src/ma`` completes to ``@src/main.py`` — the agent then
      reads the referenced file (see ``_expand_at_mentions``).

    A normal task message with neither trigger stays clean, no popup.
    """

    def __init__(
        self,
        root: Path | None = None,
        extra_commands: list[tuple[str, str]] | None = None,
    ) -> None:
        self._root = root
        # User/project custom slash commands — [(cmd, desc)], shown
        # in the same menu as the builtins.
        self._extra_commands = list(extra_commands or [])
        # Cached (rel-path list, monotonic timestamp). complete_while_
        # typing fires this on EVERY keystroke, so we walk the tree at
        # most once per _FILE_CACHE_TTL and filter the cached list per
        # keystroke instead of re-walking (a re-walk per char froze the
        # prompt on a large repo).
        self._file_cache: list[str] | None = None
        self._file_cache_at = 0.0

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ):
        text = document.text_before_cursor
        if text.startswith("/"):
            builtin = [
                (cmd, desc) for cmd, desc, _group in _COMMAND_DEFS
            ]
            for cmd, desc in (*builtin, *self._extra_commands):
                if cmd.startswith(text):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display_meta=desc,
                    )
            return
        # @-file mention: complete the token after the last '@'.
        at = text.rfind("@")
        if at != -1 and self._root is not None:
            frag = text[at + 1:]
            if " " not in frag:  # still typing one path token
                yield from self._file_completions(frag)

    def _all_files(self) -> list[str]:
        """Project file list (rel paths), walked at most once per TTL
        and cached — the completer filters this in memory per keystroke
        rather than re-walking the tree each time."""
        now = time.monotonic()
        if (
            self._file_cache is not None
            and now - self._file_cache_at < _FILE_CACHE_TTL
        ):
            return self._file_cache
        import os

        from .loominit.repomap import _SKIP_DIRS

        root = self._root
        assert root is not None
        out: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if d not in _SKIP_DIRS and not d.endswith(".egg-info")
            ]
            for fn in filenames:
                out.append(
                    os.path.relpath(os.path.join(dirpath, fn), root)
                )
                if len(out) >= _FILE_CACHE_MAX:
                    break
            if len(out) >= _FILE_CACHE_MAX:
                break
        self._file_cache = out
        self._file_cache_at = now
        return out

    def _file_completions(self, frag: str):
        """Up to 20 project files matching ``frag`` (prefix on the
        path, or substring on the basename), filtered from the cached
        file list."""
        frag_l = frag.lower()
        count = 0
        for rel in self._all_files():
            base = rel.rsplit("/", 1)[-1].lower()
            if rel.lower().startswith(frag_l) or frag_l in base:
                yield Completion(
                    rel,
                    start_position=-len(frag),
                    display_meta="file",
                )
                count += 1
                if count >= 20:
                    return


class Repl:
    """One interactive loom-code session over one project."""

    def __init__(
        self,
        project: Project,
        model: str,
        *,
        sandbox: bool = False,
        sandbox_allow_network: bool = False,
        startup_resume: str | None = None,
    ) -> None:
        self.project = project
        self.model = model
        # "last" / "pick" / None — consumed once at the top of the
        # loop (the --continue / --resume CLI flags).
        self._startup_resume = startup_resume
        # Kernel-sandbox the coder's bash (Claude-Code-style). Stored so
        # every (re)build of the agent — initial + /model switch + worktree
        # isolate — passes them through ``_rebuild_agent``.
        self._sandbox = sandbox
        self._sandbox_allow_network = sandbox_allow_network
        # Per-turn spinner controls. The ApprovalGate must pause the
        # spinner while it prompts (its Live refresh otherwise mangles
        # the keystroke). ``_turn`` points these at the live closures;
        # the gate calls them through the stable wrapper methods.
        self._active_pause_spinner: Any = None
        self._active_resume_spinner: Any = None
        # True while the ApprovalGate is waiting on the user — the
        # idle-watchdog must not count that as a hung stream.
        self._gate_active = False
        # Monotonic time of the last Ctrl-C at the idle prompt — a
        # second press within the window exits (see _run_inner).
        self._last_ctrl_c = 0.0
        # Output of the most recent ``!cmd`` inline shell run, folded
        # into the NEXT task turn's prompt then cleared (see _run_inner).
        self._last_bash_output: str | None = None
        # Which session_id already has a sessions.jsonl record — one
        # line per session, written on its first pointer save.
        self._recorded_session_id: str | None = None
        # content-hash per working block — skip the sqlite write when a
        # block's content hasn't changed. loomflow's update_block is an
        # UPSERT that bumps ``updated_at`` on every call and its block
        # ``format()`` has no timestamp, so an identical rewrite doesn't
        # itself bust the provider prompt-cache; the win here is
        # avoiding a redundant DB write (and the churn of re-reading
        # AGENTS.md from disk) every turn. Reset on agent rebuild.
        self._block_hashes: dict[str, str] = {}
        # Idle watchdog: abort a turn when the agent stream produces
        # NO events for this many seconds (a hung provider/model would
        # otherwise burn until max_turns). 0 disables. Generous default
        # — a slow non-streaming completion can legitimately be quiet
        # for a couple of minutes.
        try:
            self._idle_timeout = float(
                os.environ.get("LOOM_IDLE_TIMEOUT", "300")
            )
        except ValueError:
            self._idle_timeout = 300.0
        # Permission rules from settings.toml (user + project) + the
        # session mode. Rules load once; /mode swaps the mode live.
        from .permissions import Mode, load_rules

        rule_dirs = [
            Path.home() / ".loom-code",
            project.root / ".loom",
        ]
        perm_rules = load_rules(rule_dirs)
        # ApprovalGate persists across turns so 'allow all' sticks
        # for the whole session.
        self._gate = ApprovalGate(
            pause_spinner=self._pause_active_spinner,
            resume_spinner=self._resume_active_spinner,
            rules=perm_rules,
            mode=Mode.DEFAULT,
            project_root=project.root,
        )
        self._auto_continue_limit = _AUTO_CONTINUE_LIMIT_DEFAULT
        # Reasoning effort (None | "low" | "medium" | "high"). None =
        # provider default. Set via /effort; threaded into build_agent
        # → every work agent. Inert on non-reasoning models.
        self._effort: str | None = None
        # Session worktree isolation (/isolate). When set, this session
        # edits in its own git worktree on ``_worktree.branch`` and the
        # agent is rebuilt rooted there (_isolated_project); /merge or
        # /discard restores the main tree.
        self._worktree: worktree.WorktreeInfo | None = None
        self._isolated_project: Project | None = None
        # User + project extensions (the ``.loom`` folder). Discovered
        # once here so the SAME bundle drives both build_agent (skills,
        # subagents, tool hooks) and the REPL-lifecycle hooks fired
        # below (SessionStart / UserPromptSubmit / SessionEnd). The
        # REPL owns discovery because it also runs the trust prompt for
        # project hooks (see _consume_trusted_extensions).
        self._extensions = self._consume_trusted_extensions(
            discover_extensions(project.root)
        )
        # Custom slash commands (markdown prompt templates from
        # ~/.loom-code/commands/ + <repo>/.loom/commands/). Keyed by
        # name for dispatch; builtins always win a clash there.
        self._custom_commands = {
            c.name: c for c in self._extensions.command_specs
        }
        # /computer browser-control mode. When on, the agent gets the
        # Playwright MCP server (browser_navigate/snapshot/click/type …)
        # + a browser-oriented prompt. Off by default — toggled by the
        # /computer command, which injects the spec + rebuilds the agent.
        self._browser_mode: bool = False
        # Model the session was on before /computer bumped to a stronger
        # one — restored if operator mode is turned off.
        self._pre_operator_model: str | None = None
        # /goal run-until-done loop spec (None = off). When /goal arms a
        # condition, this holds the run_until dict (condition + cheap
        # checker + guardrails) that build_agent forwards to the
        # framework GoalStopHook. Cleared after the goal turn completes.
        self._run_until: dict[str, Any] | None = None
        # Adaptive routing (solo fast path). Both built lazily on
        # first use and invalidated by ``_rebuild_agent`` (model /
        # web changes): the solo agent is a standalone coder kernel
        # sharing the team's memory + notebook; the router agent is
        # the cheap one-word SOLO/TEAM classifier.
        self._solo_agent: Any | None = None
        self._router_agent: Any | None = None
        # Graphify and other bundled skills are auto-registered
        # by build_agent (see _bundled_skill_paths). No per-session
        # toggle needed — the agent decides when to load skills.
        self.agent, self.workspace = build_agent(
            project,
            model=model,
            approval_handler=self._gate.handler,
            max_stop_hook_iterations=self._auto_continue_limit,
            extensions=self._extensions,
            effort=self._effort,
            sandbox=self._sandbox,
            sandbox_allow_network=self._sandbox_allow_network,
            operator=self._browser_mode,
            run_until=self._run_until,
        )
        # One session_id for the whole REPL → loomflow rehydrates
        # prior turns so the agent has real conversation memory.
        self.session_id = new_id()
        # Session accumulators. ``total_in`` is *combined* input
        # tokens (uncached + cached); ``total_cached_in`` is the
        # cached subset, tracked separately so the status line can
        # show the same split (``uncached+cached in``) that the
        # end-of-turn summary uses.
        self.total_cost = 0.0
        self.total_in = 0
        self.total_cached_in = 0
        # ``total_cache_write`` is Anthropic-only — the cache CREATION
        # tokens (1.25x base price on 5m TTL, 2x on 1h). Tracked
        # separately from cached_in (which is the cache READ — cheap)
        # so /cost can surface both directions of the cache
        # accounting. OpenAI returns 0 here (no separate billing for
        # cache writes).
        self.total_cache_write = 0
        self.total_out = 0
        # Per-turn deltas (reset each turn in _account_result) — drive
        # the end-of-turn summary line so each response is separated by
        # a rule showing THAT turn's tokens + cost.
        self._turn_in = 0
        self._turn_out = 0
        self._turn_cost = 0.0
        # Context-window cache for the ctx% readout — resolved once
        # per model string (context_window_for warns on unknown
        # models; recomputing every turn would spam that warning).
        self._ctx_window = 0
        self._ctx_window_model: str | None = None
        # Framework-event counters (loomflow 0.10.13+):
        # ``total_summaries`` ticks each time
        # ``tool_result_summarized`` fires (per-tool-result LLM
        # compression — only when ``tool_result_summarizer=`` is
        # wired). ``total_compacts`` ticks each
        # ``auto_compacted`` event (mid-Ralph-loop conversation
        # summarisation when tokens cross the budget threshold).
        # ``total_snips`` ticks each ``messages_snipped`` event
        # (free list-slicing trim of older user-anchored turn
        # groups). All three surface in ``/cost`` so the user can
        # see the token-optimisation tiers actually firing.
        self.total_summaries = 0
        self.total_compacts = 0
        self.total_snips = 0
        self.turns = 0
        self.last_plan: str | None = None
        self.last_result: dict[str, Any] | None = None
        # Self-improvement: cited slugs from the last turn, awaiting
        # a success/failure judgement (the moved-on heuristic).
        self._pending_slugs: list[str] = []
        # Files the last turn WROTE to, awaiting the same judgement.
        # Recorded immediately as "unknown" (so a crash before the
        # verdict still leaves a touch record), then revised to
        # success/fail when the moved-on / good / bad signal lands —
        # the same lifecycle as ``_pending_slugs``. Feeds the file-
        # touch history that powers proactive anticipation.
        self._pending_files: list[str] = []
        # The prompt that drove the last turn — used as the touch
        # summary ("why was this file changed").
        self._last_prompt: str = ""
        # Automatic compaction state. ``_compact_threshold = -1``
        # means "auto, recompute from model"; ``0`` means "off";
        # any positive int is an explicit user override. The
        # exchange list is what the compactor sees on trigger; the
        # cumulative-tokens counter is what fires the trigger.
        # Compaction is summarisation — low-stakes, so it runs on the
        # cheap same-provider sibling (Haiku / gpt-4.1-mini) instead
        # of burning the coding model's rates on it.
        self._compactor = Compactor(
            model=cheap_model_for(model) or model
        )
        self._compact_threshold = -1  # auto
        self._compact_tokens = 0
        self._compact_exchanges: list[tuple[str, str]] = []
        # Web-search backend: ``"serper"``, ``"duckduckgo"``, or
        # ``None`` (off — default). Toggled via /set_web. Rebuilding
        # the agent picks the new backend up by adding (or not
        # adding) a ``web_tool`` to coder + explorer.
        self._web_backend: str | None = None
        # Verify-before-done gate (on by default; /verify off): a turn
        # that edits code + claims completion without running the
        # project's tests gets one nudge to run them.
        self._verify_gate = True
        # ``self._auto_continue_limit`` is initialised earlier in
        # __init__ (before build_agent is called) so the framework
        # gets the right ``max_stop_hook_iterations`` on construction.
        # See the build_agent call above.
        # prompt_toolkit drives the input line. complete_while_typing
        # opens the autocomplete menu the moment the user types '/'
        # without any extra keystroke (Tab also still works for
        # explicit completion). History gives free up-arrow recall
        # within the session. The paste keybindings collapse large
        # pastes into `[paste-N: <lines>, <chars>]` placeholders so
        # the visible prompt stays readable; expand_pastes() restores
        # the full content before the line goes to the agent.
        self._prompt_session: PromptSession[str] = PromptSession(
            completer=_SlashCompleter(
                root=project.root,
                extra_commands=[
                    (f"/{c.name}", c.description)
                    for c in self._extensions.command_specs
                ],
            ),
            complete_while_typing=True,
            key_bindings=build_paste_keybindings(),
        )

    async def run(self) -> int:
        """The REPL loop. Returns an exit code.

        Skills (graphify and friends) are wired in at agent
        construction time via :func:`build_agent` — no per-session
        spawning, no subprocess lifecycle to manage here.

        The ``finally`` DOES tear down the MCP registry: ``build_agent``
        stashes any connected MCP servers on ``agent._mcp_registry``,
        and those hold live subprocess / HTTP sessions (stdio servers
        are child processes). Closing them on every exit path — normal
        quit, Ctrl-C, or an exception — avoids leaking processes.
        """
        try:
            return await self._run_inner()
        finally:
            await self._aclose_mcp()
            await self._aclose_browsers()
            # Background processes the agent started (dev servers,
            # watchers) die with the session — no orphans. atexit
            # backstops a hard crash; this is the friendly path.
            from . import background

            killed = background.kill_all()
            if killed:
                console.print(
                    f"  [dim]stopped {killed} background process"
                    f"{'es' if killed != 1 else ''}[/dim]"
                )

    async def _aclose_mcp(self) -> None:
        """Best-effort teardown of the MCP registry's sessions. Never
        raises — shutdown must not turn a clean exit into an error."""
        registry = getattr(self.agent, "_mcp_registry", None)
        if registry is None:
            return
        try:
            await registry.aclose()
        except Exception:  # noqa: BLE001 — teardown must not fail exit
            pass

    async def _aclose_browsers(self) -> None:
        """Close any /computer browser windows on exit so a headed
        Chromium doesn't linger after loom-code quits. Best-effort."""
        try:
            from .browse import close_all_browsers

            await close_all_browsers()
        except Exception:  # noqa: BLE001 — teardown must not fail exit
            pass

    async def _run_inner(self) -> int:
        """The REPL loop body. Held as a separate method in case a
        future feature wants to wrap it in a context manager again
        — keeps the wrapping point obvious."""
        banner(
            self.model,
            str(self.project.root),
            self.project.is_git,
            sandbox=self._sandbox,
            sandbox_allow_network=self._sandbox_allow_network,
        )
        if self.project.context_file:
            console.print(
                f"  [dim]loaded context: "
                f"{self.project.context_file.name}[/dim]"
            )
        # Brief getting-started hints right after the banner. Surfaces
        # provider/web setup AND the bare /model command so users who
        # already have an API key but want a different model name
        # don't go hunting through /help. Ordering: most-common first.
        # ``•`` bullets, NOT ``▸``/``›`` — the input prompt uses a
        # ``›`` glyph, and arrow-ish banner bullets read as prompts at
        # a glance in most terminal fonts.
        console.print(
            "  [dim]• type a task, or [cyan]/help[/cyan] "
            "for the command menu[/dim]"
        )
        console.print(
            "  [dim]• [cyan]/model <name>[/cyan]  switch to a "
            "specific model by name (e.g. gpt-4.1, claude-opus-4-8)[/dim]"
        )
        console.print(
            "  [dim]• [cyan]/set_model[/cyan]     pick a provider + "
            "model (saves your API key)[/dim]"
        )
        console.print(
            "  [dim]• [cyan]/set_web[/cyan]       enable web "
            "search (Serper / DuckDuckGo)[/dim]"
        )
        # Show the resume hint ONLY when a prior session pointer
        # exists — no point telling first-time users about a
        # feature they can't use yet.
        if self._load_session_pointer() is not None:
            console.print(
                "  [dim]• [cyan]/resume[/cyan]        pick up "
                "the last session for this project (rehydrates "
                "prior turns)[/dim]"
            )
        # An unfinished durable goal survives restarts — surface it
        # so the user knows one is waiting (codex-parity /goal).
        _goal = _load_goal_state(self.project.root / LOOM_DIR)
        if _goal is not None:
            console.print(
                f"  [dim]🎯 unfinished goal: "
                f"{_truncate_one_line(str(_goal['task']), 60)} — "
                "[cyan]/goal resume[/cyan] continues it[/dim]"
            )
        self._print_extensions_banner()
        console.print()

        # SessionStart hooks fire once, after the banner and before the
        # first prompt — for side effects (env setup, logging). Their
        # added context is surfaced as a dim note rather than injected,
        # since there's no user turn to attach it to yet.
        start_result = await self._fire_repl_hooks("SessionStart")
        if start_result.added_context:
            console.print(
                f"  [dim]{start_result.added_context}[/dim]"
            )

        # --continue / --resume: rejoin a prior session before the
        # first prompt, via the same machinery as /resume.
        if self._startup_resume == "last":
            await self._handle_resume("")
        elif self._startup_resume == "pick":
            await self._handle_resume("pick")

        while True:
            # _read_line opens each turn with a full-width rule + a dim
            # cost line, so the cost/token status is attached to the
            # prompt (no separate status print above it) and the rule
            # separates this turn from the previous output.
            try:
                line = await self._read_line()
            except EOFError:
                # Ctrl-D: leaving satisfied — credit any pending turn.
                await self._attribute_pending(success=True, quiet=True)
                await self._fire_repl_hooks("SessionEnd")
                console.print("\n[dim]bye[/dim]")
                return 0
            except KeyboardInterrupt:
                # Ctrl-C at the idle prompt: the reflex from every
                # other REPL is "clear the line", not "quit" — a
                # single press exiting the whole session was a sharp
                # edge. First press warns; a second within the window
                # exits (same cleanup as Ctrl-D).
                now = time.monotonic()
                if now - self._last_ctrl_c < 2.0:
                    await self._attribute_pending(
                        success=True, quiet=True
                    )
                    await self._fire_repl_hooks("SessionEnd")
                    console.print("\n[dim]bye[/dim]")
                    return 0
                self._last_ctrl_c = now
                console.print(
                    "  [dim]press Ctrl-C again to exit "
                    "(or /exit)[/dim]"
                )
                continue

            line = line.strip()
            if not line:
                continue

            # Expand any [paste-N: ...] placeholders to the full
            # stashed content BEFORE dispatch — slash commands
            # generally won't contain pastes, but expanding here
            # keeps a single canonical "what the user really said"
            # point of truth and matches how Claude Code does it.
            line = expand_pastes(line)

            if line.startswith("/"):
                # Only dispatch KNOWN commands. An absolute filesystem
                # path (``/Users/me/x.py``) also starts with "/" — it
                # must reach the agent as a task, not error as an
                # unknown command. Heuristic: a first token with a
                # second "/" is a path; a bare unknown token like
                # "/hlep" still errors (typo protection).
                first = line.split()[0].lower()
                known = {c for c, _d, _g in _COMMAND_DEFS}
                known |= {"/quit", "/computer"}
                if first in known:
                    should_continue = await self._handle_slash(line)
                    if not should_continue:
                        await self._attribute_pending(
                            success=True, quiet=True
                        )
                        await self._fire_repl_hooks("SessionEnd")
                        console.print("[dim]bye[/dim]")
                        return 0
                    continue
                custom = self._custom_commands.get(first[1:])
                if custom is not None:
                    # A user-authored markdown command: expand the
                    # template and FALL THROUGH — the result runs as
                    # an ordinary task turn (same approval gate).
                    from .extensions import expand_command_template

                    line = expand_command_template(
                        custom.template,
                        line[len(first):].strip(),
                    )
                    console.print(
                        f"  [dim]{first} → running its prompt "
                        f"({custom.source} command)[/dim]"
                    )
                elif "/" not in first[1:]:
                    console.print(
                        f"  unknown command {first} — /help for "
                        "the list"
                    )
                    continue
                # Falls through: a path-shaped line (or an expanded
                # custom command) is a task.

            # ``!cmd`` — run a shell command inline, right now, without
            # spending a model turn. The output is echoed AND stashed so
            # the NEXT task turn can reference it ("now fix that error").
            # Matches Claude Code's ``!`` prefix.
            if line.startswith("!"):
                try:
                    await self._run_bang(line[1:].strip())
                except Exception as exc:  # noqa: BLE001 — never exit
                    console.print(
                        Text(f"  ! error: {exc}", style="red")
                    )
                continue

            # Expand @-file mentions to inline the referenced files so
            # the model gets their content, not just the path.
            line = self._expand_at_mentions(line)

            # Fold in the last ``!cmd`` output (once) so "now fix that"
            # after a bang command has the output to work from.
            if self._last_bash_output is not None:
                line = (
                    f"{line}\n\n[output of a shell command I just ran]\n"
                    f"{self._last_bash_output}"
                )
                self._last_bash_output = None

            # Pure greeting → answer locally, zero tokens. Placed
            # BEFORE hooks/attribution/injection on purpose: a "hi"
            # is neutral chatter — it must not credit the previous
            # turn as accepted, fire task hooks, or pay the per-turn
            # context injection.
            greeting = _greeting_reply(line)
            if greeting is not None:
                console.print(f"  {greeting}")
                continue

            # UserPromptSubmit hooks see the prompt before the agent
            # does. A hook may BLOCK the turn (exit 2) — e.g. a policy
            # gate — or return additionalContext we fold into the
            # prompt (e.g. inject the current ticket / branch).
            submit = await self._fire_repl_hooks(
                "UserPromptSubmit", prompt=line
            )
            if submit.blocked:
                console.print(
                    f"  [red]⊘ blocked by hook[/red]: "
                    f"{submit.reason or '(no reason given)'}"
                )
                continue
            if submit.added_context:
                line = f"{line}\n\n[context from hook]\n{submit.added_context}"

            # A new task with no prior complaint → the previous
            # turn must have been fine. Credit it, then run.
            await self._attribute_pending(success=True, quiet=False)
            # Per-turn repo-map injection — populates the
            # ``loom_index`` working block with the deterministic repo
            # map. Loomflow auto-injects working blocks into the next
            # system prompt.
            await self._inject_loom_context(line)
            await self._inject_file_history(line)
            await self._inject_learned_notes(line)
            # Auto-checkpoint before the turn runs: snapshot the working
            # tree so the user can /undo this turn's edits even if the
            # agent goes off the rails. Silent on success (a checkpoint
            # per turn would be noise); only /undo + /checkpoints surface
            # them. Best-effort — a non-git repo / git failure no-ops.
            self._checkpoint_before_turn(line)
            route = await self._route_turn(line)
            if route == "solo":
                # Surface the routing decision — silent topology
                # switches make cost/behaviour differences look
                # random to the user.
                console.print(
                    "  [dim]→ solo fast path (small task — skipping "
                    "team delegation)[/dim]"
                )
                await self._turn(line, agent=self._get_solo_agent())
            else:
                await self._turn(line)

    # ---- input ----------------------------------------------------------

    async def _read_line(self) -> str:
        """Read one line with a clean, framed-feel prompt.

        The separation is owned by the END of each turn: a full-width
        rule + that turn's tokens/cost (``_print_turn_summary``). The
        prompt itself stays minimal — one blank line of air, then the
        bold ``›`` glyph. Cumulative session totals live in ``/cost``
        (printing them above every prompt duplicated the turn rule and
        showed a noisy all-zeros line before the first input).
        Autocomplete / history / paste keybindings come from the
        ``PromptSession``.
        """
        console.print()
        return await self._prompt_session.prompt_async(
            HTML("<ansigreen><b>›</b></ansigreen>  ")
        )

    async def _run_bang(self, cmd: str) -> None:
        """Run ``cmd`` in the project root right now (``!`` prefix) and
        echo its output. The result is stashed in ``_last_bash_output``
        so the next task turn can be told about it — Claude-Code-style
        "run this, then act on what you see"."""
        if not cmd:
            console.print("  [dim]usage: !<shell command>[/dim]")
            return
        import functools
        import subprocess

        # The command AND its output are USER/tool data — render with
        # markup DISABLED (styling via Text/style=, never inline
        # ``[dim]{x}[/dim]`` tags). A ``[`` in the command or a line
        # like ``[FAILED]`` in the output would otherwise be parsed as
        # a Rich tag → MarkupError, which — uncaught in the input loop
        # — killed the whole REPL (observed: ``!pytest`` on failing
        # tests crashed the session).
        console.print(Text(f"  $ {cmd}", style="dim"))
        try:
            # Worker thread so the blocking run can't stall the event
            # loop (and Ctrl-C at the REPL stays responsive).
            proc = await anyio.to_thread.run_sync(
                functools.partial(
                    subprocess.run,
                    cmd,
                    shell=True,
                    cwd=str(self.project.root),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            )
        except subprocess.TimeoutExpired:
            console.print(
                Text("  ! command timed out (120s)", style="yellow")
            )
            return
        except Exception as exc:  # noqa: BLE001 — never kill the REPL
            console.print(Text(f"  ! failed: {exc}", style="red"))
            return
        out = (proc.stdout or "") + (proc.stderr or "")
        out = out.rstrip()
        if out:
            for ln in out.splitlines()[:200]:
                console.print(Text(f"  {ln}", style="dim"))
        code = proc.returncode
        console.print(
            Text(
                f"  exit {code}",
                style="dim" if code == 0 else "yellow",
            )
        )
        # Stash for the next turn (bounded so a huge dump can't bloat
        # the prompt). Consumed + cleared in the input loop.
        self._last_bash_output = (
            f"$ {cmd}\n{out[:4000]}" if out else f"$ {cmd}\n(exit {code})"
        )

    def _expand_at_mentions(self, line: str) -> str:
        """Inline file references so the model gets the file CONTENT,
        not just the name. Two forms with DIFFERENT trust levels:

        * ``@path`` mention — a DELIBERATE reference. Inlines the file
          AND grants outside-project EDIT consent (see
          ``loom_code.consent``): typing ``@`` is an unambiguous "act
          on this file" gesture.
        * a bare / quoted / pasted absolute path — a convenience for
          "read this". Inlined for READING only; NO edit consent. This
          is the safety boundary: a path that merely appears in a
          pasted stack trace or log line (``…/site-packages/x.py``,
          ``~/.zshrc``) must never become editable just by being
          quoted — only an explicit ``@`` unlocks edits.

        The AGENT's own read tool stays project-scoped regardless, so a
        prompt-injected model can't roam the filesystem; this is purely
        about the USER pulling a file into the prompt.

        Each existing file is inlined once; non-files are left as
        literal text. Bounded per file so a giant file can't blow the
        context."""
        import re as _re

        # (ref, is_at_mention) — @-mentions grant edit consent, bare
        # paths don't.
        refs: list[tuple[str, bool]] = [
            (m, True) for m in _re.findall(r"@([^\s]+)", line)
        ]
        # Bare/quoted absolute paths — READ-only convenience.
        for m in _re.findall(r"['\"]((?:/|~/)[^'\"]+)['\"]", line):
            refs.append((m, False))
        for m in _re.findall(r"(?<!\S)((?:/|~/)[^\s'\"]+)", line):
            refs.append((m, False))
        # macOS drag-and-drop escapes spaces (``…/Screenshot\ 2026….png``).
        for m in _re.findall(
            r"(?<!\S)((?:/|~/)(?:[^\s'\"\\]|\\ )+)", line
        ):
            if "\\ " in m:
                refs.append((m.replace("\\ ", " "), False))
        # Last resort for a PASTED path with unescaped spaces.
        for start in [
            m.start() for m in _re.finditer(r"(?<!\S)(?=/|~/)", line)
        ]:
            tail = line[start:].strip().strip("'\"")
            words = tail.split(" ")
            for end in range(len(words), 0, -1):
                cand = " ".join(words[:end])
                if Path(cand).expanduser().is_file():
                    refs.append((cand, False))
                    break
        if not refs:
            return line
        seen: set[str] = set()
        blocks: list[str] = []
        for ref, is_mention in refs:
            ref = ref.rstrip(".,;:!?")
            if ref in seen:
                continue
            seen.add(ref)
            p = Path(ref).expanduser()
            fpath = (
                p if p.is_absolute() else self.project.root / p
            ).resolve()
            # Existence IS the filter — prose that merely looks
            # path-shaped never resolves to a real file.
            if not fpath.is_file():
                continue
            try:
                raw = fpath.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:8192]:
                # Binary (image/archive/db) — inlining bytes as text
                # is garbage. Tell the user instead of silently
                # skipping; the models here are text-only.
                console.print(
                    f"  [yellow]@ {ref} is a binary file — can't "
                    "inline it (text files only)[/yellow]"
                )
                continue
            # Grant edit consent for ANY path the user typed/pasted —
            # bare OR @-mentioned. You naming a file IS the permission
            # (Claude-Code model). This is safe because the approval
            # gate ALWAYS shows the diff + asks before an outside-
            # project edit (see ApprovalGate._is_outside_project): a
            # path incidentally embedded in a pasted stack trace can't
            # silently mutate anything — the user sees the prompt and
            # rejects it. ``is_mention`` is kept only for display
            # nuance, not for the grant decision.
            del is_mention  # no longer gates consent
            from . import consent

            consent.grant(fpath)
            body = raw.decode("utf-8", errors="replace")
            if len(body) > 8000:
                body = body[:8000] + "\n… (truncated)"
            blocks.append(f"--- {ref} ---\n{body}")
            console.print(f"  [dim]@ inlined {ref}[/dim]")
        if not blocks:
            return line
        # Bare-path turn: the user pasted ONLY a path, no instruction.
        # Weak models invent a task for it (observed live: an
        # unprompted multi_edit on the referenced file). Spell the
        # implicit contract out instead of letting the model guess.
        residue = line
        for ref in seen:
            residue = residue.replace(ref, " ")
        residue = residue.replace("@", " ").strip(" \t'\".,;:!?")
        suffix = ""
        if not residue:
            suffix = (
                "\n\n(The user pasted only this file path, with no "
                "instruction. Summarise the file in 2-3 sentences "
                "and ask what they'd like done with it. Do NOT "
                "modify anything.)"
            )
        return line + "\n\n" + "\n\n".join(blocks) + suffix

    # ---- a task turn ----------------------------------------------------

    async def _update_block_if_changed(
        self, name: str, content: str
    ) -> None:
        """Write a working block only when changed — delegates to the
        shared :func:`loom_code.turn.update_block_if_changed` so the
        REPL and the learned-notes injector use ONE dirty-check."""
        from .turn import update_block_if_changed

        await update_block_if_changed(
            self.agent.memory,
            name,
            content,
            user_id=_USER_ID,
            block_hashes=self._block_hashes,
        )

    async def _inject_loom_context(self, prompt: str) -> None:
        """Update the ``loom_index`` working block with a deterministic
        repo map — the most structurally-important symbols (signatures +
        locations) — which loomflow folds into the next system prompt.

        Built from the structural index (AST walk, no model calls), so
        it needs no ``/loominit`` and is fresh-by-construction: the
        cached builder re-walks only when the tree changed. ``prompt``
        is unused (the map is a stable global overview, which keeps the
        system prompt cache-stable across turns).

        Failures are swallowed (never let memory I/O kill a turn).
        """
        del prompt  # map is global, not prompt-ranked
        try:
            from .loominit.repomap import repo_map_for_root_cached

            # Deterministic repo map (top symbols by structural
            # importance) built from the structural index — no LLM, no
            # LOOM.md/loominit needed, and fresh-by-construction
            # (re-walked only when the tree changed). Replaces the old
            # BM25-over-LLM-narrative retrieval that drifted as the
            # agent edited code.
            body = repo_map_for_root_cached(self.project.root)
            if body:
                await self._update_block_if_changed("loom_index", body)
            # Auto-reload the project rules file (AGENTS.md): re-read it
            # FRESH each turn into the ``project_rules`` working block, so
            # a mid-session edit applies on the next turn without a
            # restart. The coordinator's static prompt no longer bakes
            # the rules file (see build_unified_coordinator_instructions).
            # The dirty-check keeps the re-read cheap: an UNCHANGED file
            # skips the write, so the cache prefix survives.
            from .rules import project_rules_block

            await self._update_block_if_changed(
                "project_rules",
                project_rules_block(self.project.root),
            )
        except Exception:  # noqa: BLE001 — injection is best-effort
            pass

    async def _inject_learned_notes(self, prompt: str) -> None:
        """ACTIVE recall: push the top success-credited notes relevant
        to this prompt into the ``learned_notes`` working block, so past
        learnings shape the next action directly.

        Before this, credited notes only ranked higher in
        ``search_notes`` — a search the agent might never run. Now the
        proven ones (cited in a turn the user accepted) arrive in the
        system prompt unprompted, CLAUDE.md-style, while the full
        notebook stays behind search. Slugs are shown so the agent can
        ``read_note(slug)`` for full detail — which also keeps the
        citation-credit chain alive for the /good /bad loop.

        Bounded: top 3 notes, snippet-length excerpts (~140 chars each)
        — a few hundred tokens, not a notebook dump. Block is cleared
        when nothing relevant is proven, so stale advice never lingers.
        Failures are swallowed (never let memory I/O kill a turn).

        Delegates to :mod:`loom_code.turn` — the SHARED per-turn
        pipeline, so the desktop sidecar runs the identical logic.
        """
        from .turn import inject_learned_notes

        await inject_learned_notes(
            self.workspace,
            self.agent.memory,
            prompt,
            user_id=_USER_ID,
            block_hashes=self._block_hashes,
        )

    async def _inject_file_history(self, prompt: str) -> None:
        """Proactive anticipation: before the agent runs, surface what
        happened last time we touched the files this prompt is about.

        THE soul of loom-code over a stateless coder — "last time you
        edited src/auth.py the change was marked bad, be careful." Two
        surfaces: a ``file_anticipation`` working block (loomflow folds
        it into the next system prompt so the AGENT heeds it) AND a dim
        ``recall:`` line so the USER sees the warning fire.

        Silent when nothing's notable (a clean, rarely-touched file
        produces no block + no line) — noise would train both the
        model and the user to ignore the section. Best-effort: any
        failure is swallowed; anticipation degrading to silence is
        correct, a crash mid-turn is not."""
        try:
            candidates = file_history.candidate_paths_from_prompt(
                self.project.root, prompt
            )
            if not candidates:
                # Clear any stale block from a prior turn so last turn's
                # warning doesn't bleed into an unrelated prompt.
                # (Dirty-checked: consecutive no-candidate turns — the
                # common case — write once, then stay cache-warm.)
                await self._update_block_if_changed(
                    "file_anticipation", ""
                )
                return
            records = file_history.history_for(
                self.project.root, candidates
            )
            block = file_history.anticipation_block(records)
            await self._update_block_if_changed(
                "file_anticipation", block
            )
            if block:
                # One concise user-visible line per warned file.
                for rec in records:
                    if rec.last_outcome == "fail" or rec.fail_count > 0:
                        console.print(
                            f"  [dim]↶ recall:[/dim] [yellow]last change "
                            f"to {rec.path} was marked bad — being "
                            f"careful[/yellow]"
                        )
                    elif rec.touch_count >= 4:
                        console.print(
                            f"  [dim]↶ recall: {rec.path} is a churn "
                            f"hotspot ({rec.touch_count} edits)[/dim]"
                        )
        except Exception:  # noqa: BLE001 — anticipation is best-effort
            pass

    def _checkpoint_before_turn(self, prompt: str) -> None:
        """Snapshot the working tree before a turn (auto-checkpoint).

        Silent on success — a per-turn confirmation line would be noise;
        the snapshots only surface via /undo + /checkpoints. Best-effort:
        a non-git repo or git failure no-ops without disturbing the run.
        Skipped when the session is already isolated in a worktree
        (the worktree IS the isolation; double-snapshotting is redundant
        and would snapshot the wrong tree)."""
        if getattr(self, "_isolated_wt", None) is not None:
            return
        try:
            _checkpoint.checkpoint(self.project.root, summary=prompt)
        except Exception:  # noqa: BLE001 — checkpointing is best-effort
            pass

    async def _consume_agent_stream(
        self,
        agent: Any,
        prompt: str,
        renderer: StreamRenderer,
        pause_status: Any,
    ) -> bool:
        """Stream one agent run into ``renderer`` + tick the token-
        optimisation counters. Returns False (caller should abort
        the turn) if the stream raised; True on clean completion.

        Extracted so the escalation path can re-run a SECOND agent
        (the supervisor) through the identical consume + error-
        handling logic without duplicating it.

        Two liveness guards run alongside the consume loop:

        * **Idle watchdog** — no events for ``_idle_timeout`` seconds
          (approval-prompt waits excluded via ``_gate_active``) means
          the stream is hung (dead provider, stuck model); cancel the
          turn instead of burning until ``max_turns``.
        * **Stall detector** — ``_STALL_REPEATS`` consecutive
          IDENTICAL tool calls means the model is looping without
          progress (loomflow's no-progress hook only arms under
          /goal); cancel early with a clear message.
        """
        # Shared mutable state between the consume body and watchdog.
        state: dict[str, Any] = {
            "last_event": time.monotonic(),
            "timed_out": False,
            "stalled_tool": None,
            "tool_in_flight": False,
        }
        repeat: dict[str, Any] = {"key": None, "count": 0}
        idle_timeout = self._idle_timeout

        try:
            async with anyio.create_task_group() as tg:

                # Poll at a fraction of the timeout (capped at 5s) so
                # small timeouts — tests, aggressive configs — still
                # trip promptly; the default 300s polls every 5s.
                poll = max(0.05, min(5.0, (idle_timeout or 5.0) / 4))

                async def _watchdog() -> None:
                    while True:
                        await anyio.sleep(poll)
                        # Don't count time the user is at an approval
                        # prompt, or a tool is legitimately RUNNING
                        # (a bash test-suite/build emits tool_call then
                        # nothing until its result — that's work, not a
                        # hang). Only genuine model-side silence counts.
                        if self._gate_active or state["tool_in_flight"]:
                            state["last_event"] = time.monotonic()
                            continue
                        idle = time.monotonic() - state["last_event"]
                        if idle_timeout and idle > idle_timeout:
                            state["timed_out"] = True
                            tg.cancel_scope.cancel()
                            return

                tg.start_soon(_watchdog)

                stream = agent.stream(
                    prompt,
                    user_id=_USER_ID,
                    session_id=self.session_id,
                )
                async for event in stream:
                    state["last_event"] = time.monotonic()
                    renderer.handle(event)
                    kind = str(getattr(event, "kind", ""))
                    payload = getattr(event, "payload", None) or {}
                    # Tick the token-optimisation counters (loomflow
                    # 0.10.13+) off architecture events.
                    if kind.endswith("architecture_event"):
                        name = payload.get("name")
                        if name == "tool_result_summarized":
                            self.total_summaries += 1
                        elif name == "auto_compacted":
                            self.total_compacts += 1
                        elif name == "messages_snipped":
                            self.total_snips += 1
                    elif kind.endswith("tool_result"):
                        # Tool finished → the model is back in control;
                        # idle time counts again.
                        state["tool_in_flight"] = False
                    elif kind.endswith("tool_call"):
                        # A tool is now RUNNING — pause the idle clock
                        # until its result arrives (see _watchdog).
                        state["tool_in_flight"] = True
                        # Stall detection: same tool + same args,
                        # over and over, is a loop — not progress.
                        call = payload.get("call") or {}
                        try:
                            args_key = json.dumps(
                                call.get("args"),
                                sort_keys=True,
                                default=str,
                            )
                        except (TypeError, ValueError):
                            args_key = repr(call.get("args"))
                        key = f"{call.get('tool')}:{args_key}"
                        if key == repeat["key"]:
                            repeat["count"] += 1
                            if repeat["count"] >= _STALL_REPEATS:
                                state["stalled_tool"] = call.get(
                                    "tool"
                                )
                                # Close the stream generator IN THIS
                                # task before cancelling — a bare
                                # ``break`` leaves it suspended for GC
                                # to finalise in another task, which
                                # trips anyio's "cancel scope in a
                                # different task". aclose() unwinds its
                                # internal task group here, cleanly.
                                await stream.aclose()
                                tg.cancel_scope.cancel()
                                break
                        else:
                            repeat["key"] = key
                            repeat["count"] = 1
                # Stream done — stop the watchdog.
                tg.cancel_scope.cancel()
        except KeyboardInterrupt:
            pause_status()
            console.print(
                "\n[yellow]interrupted — turn abandoned[/yellow]"
            )
            return False
        except BaseExceptionGroup as eg:
            # anyio's task groups raise ``ExceptionGroup`` when any
            # child task fails. Unwrap to surface the REAL cause(s)
            # instead of the opaque wrapper message. Ctrl-C inside
            # the group arrives wrapped — route it to the interrupt
            # path, not the error path.
            pause_status()
            inners = _flatten_exception_group(eg)
            interrupted = any(
                isinstance(i, KeyboardInterrupt) for i in inners
            )
            if interrupted:
                console.print(
                    "\n[yellow]interrupted — turn abandoned[/yellow]"
                )
            # Print REAL errors too, even alongside a Ctrl-C — a worker
            # that crashed (e.g. a 401) at the same moment the user hit
            # Ctrl-C must not be hidden behind "interrupted", or they
            # retry the identical prompt into the same silent failure.
            for inner in inners:
                if not isinstance(
                    inner, (KeyboardInterrupt, anyio.get_cancelled_exc_class())
                ):
                    self._print_turn_error(inner)
            return False
        except Exception as exc:  # noqa: BLE001 — REPL must survive
            pause_status()
            self._print_turn_error(exc)
            return False
        finally:
            # The gate can't still be up once the stream ends — render
            # anything that queued behind an approval prompt.
            self._gate_active = False
            renderer.flush_deferred()

        if state["timed_out"]:
            pause_status()
            console.print(
                f"\n[yellow]turn aborted — no activity for "
                f"{int(idle_timeout)}s (stream hung?). The tree is "
                f"unchanged since the pre-turn checkpoint; /undo "
                f"restores it if needed. LOOM_IDLE_TIMEOUT=0 "
                f"disables this guard.[/yellow]"
            )
            return False
        if state["stalled_tool"]:
            pause_status()
            console.print(
                f"\n[yellow]turn aborted — the model repeated the "
                f"same [cyan]{state['stalled_tool']}[/cyan] call "
                f"{_STALL_REPEATS}× with identical arguments (a "
                f"loop, not progress). Try rephrasing, or a "
                f"stronger model via /model.[/yellow]"
            )
            return False
        return True

    @staticmethod
    def _print_turn_error(exc: BaseException) -> None:
        """One error, two lines max: the real cause (dim, for bug
        reports) + an actionable hint when we recognise the class.
        The raw ExceptionGroup wrapper never reaches here — callers
        flatten first — and render's error event suppresses its own
        copy, so each failure prints exactly once."""
        console.print(
            f"\n[red]error: {type(exc).__name__}: {exc}[/red]"
        )
        hint = friendly_error_hint(exc)
        if hint:
            console.print(f"  [yellow]→ {hint}[/yellow]")

    # ---- .loom extensions: trust gate + REPL-lifecycle hooks --------

    def _consume_trusted_extensions(
        self, extensions: Extensions
    ) -> Extensions:
        """Apply the project-hook trust gate to a discovered bundle.

        User hooks, skills, and subagents pass through untouched;
        project hooks survive only if already trusted or approved at
        the prompt below. Called once from ``__init__``."""
        return filter_trusted_hooks(
            extensions,
            project_root=self.project.root,
            prompt=self._prompt_trust_project_hooks,
        )

    def _prompt_trust_project_hooks(self, specs: list[HookSpec]) -> bool:
        """Show a project's hook commands and ask whether to trust them.

        Safe default is NO: a non-tty session never auto-trusts, and at
        the prompt only an explicit ``y`` approves — we don't run a
        cloned repo's shell commands without consent."""
        from .approval import _read_single_key

        console.print()
        console.print(
            "  [bold yellow]⚠ this project defines hooks[/bold yellow] "
            "(.loom/settings.toml) that run shell commands "
            "automatically:"
        )
        for s in specs:
            tag = f" [{s.matcher}]" if s.matcher not in ("", "*") else ""
            console.print(
                f"    [cyan]{s.event}[/cyan]{tag}  →  "
                f"[dim]{s.command}[/dim]"
            )
        if not sys.stdin.isatty():
            console.print(
                "  [dim](non-interactive — skipping project hooks)[/dim]"
            )
            return False
        console.print(
            "  [bold]trust and run these hooks?[/bold] "
            "[dim](press y to trust, any other key to skip)[/dim] ",
            end="",
        )
        trusted = _read_single_key() in ("y", "Y")
        console.print(
            "[green]trusted[/green]" if trusted else "[dim]skipped[/dim]"
        )
        return trusted

    def _print_extensions_banner(self) -> None:
        """Show what got picked up from ``.loom`` so the user can
        confirm their skills / subagents / hooks loaded (and which
        project hooks the trust gate let through)."""
        ext = self._extensions
        bits: list[str] = []
        if ext.skill_paths:
            bits.append(f"{len(ext.skill_paths)} skill(s)")
        if ext.agent_specs:
            names = ", ".join(s.name for s in ext.agent_specs)
            bits.append(f"{len(ext.agent_specs)} subagent(s) ({names})")
        if ext.hook_specs:
            bits.append(f"{len(ext.hook_specs)} hook(s)")
        if bits:
            console.print(
                f"  [dim]▸ .loom extensions: {' · '.join(bits)}[/dim]"
            )

    async def _fire_repl_hooks(
        self, event: str, *, prompt: str | None = None
    ) -> Any:
        """Run every REPL-lifecycle hook registered for ``event``.

        Returns the ``ReplHookResult`` so ``UserPromptSubmit`` can act
        on a block / injected context; ``SessionStart`` / ``SessionEnd``
        callers ignore it (those hooks run for their side effects)."""
        return await run_repl_hooks(
            self._extensions.hook_specs,
            event,
            cwd=self.project.root,
            prompt=prompt,
        )

    def _pause_active_spinner(self) -> None:
        """Stable hook the ApprovalGate calls to stop the current
        turn's spinner before prompting. No-op between turns.

        Also marks the gate as active so the idle-watchdog doesn't
        count the user's thinking time at an approval prompt as
        "the stream hung"."""
        self._gate_active = True
        cb = self._active_pause_spinner
        if cb is not None:
            cb()

    def _resume_active_spinner(self) -> None:
        """Stable hook the ApprovalGate calls after the prompt to
        bring the spinner back."""
        self._gate_active = False
        cb = self._active_resume_spinner
        if cb is not None:
            cb()

    def _account_result(
        self,
        result: dict[str, Any],
        renderer: StreamRenderer,
        prompt: str,
        *,
        extend_files: bool = False,
    ) -> None:
        """Fold ONE completed stream's result into the session totals:
        cost, every token bucket (incl. cache-write), turns, this
        turn's file touches, and the compaction high-water mark.

        Shared by the main turn AND the tool-leak nudge so the two can
        never drift — a hand-copied second version had already dropped
        cache_write_tokens and the high-water update, which under-
        reported /cost and delayed auto-compaction on nudged turns.

        ``extend_files`` appends to the pending file list (nudge, whose
        touches add to the same turn) instead of replacing it."""
        cost = float(result.get("cost_usd", 0.0))
        tin = int(result.get("tokens_in", 0))
        cached_in = int(result.get("cached_tokens_in", 0))
        tout = int(result.get("tokens_out", 0))
        self.total_cost += cost
        self.total_in += tin + cached_in
        self.total_cached_in += cached_in
        self.total_cache_write += int(
            result.get("cache_write_tokens", 0)
        )
        self.total_out += tout
        self.turns += int(result.get("turns", 0))
        # Per-turn deltas for the end-of-turn summary line. ``extend``
        # (the nudge path) ADDS to the same turn's numbers so the
        # summary reflects the whole turn, main + nudge.
        if extend_files:
            self._turn_in += tin + cached_in
            self._turn_out += tout
            self._turn_cost += cost
        else:
            self._turn_in = tin + cached_in
            self._turn_out = tout
            self._turn_cost = cost
        # Record this turn's file touches immediately as "unknown".
        # The outcome is revised to success/fail in
        # ``_attribute_pending`` when the moved-on / good / bad signal
        # arrives. Recording now means a crash before judgement still
        # leaves the touch on record — better unjudged than lost.
        touched = list(renderer.files_touched)
        if extend_files:
            self._pending_files.extend(touched)
        else:
            self._pending_files = touched
        self._last_prompt = prompt
        if touched:
            file_history.record_touches(
                self.project.root,
                touched,
                outcome="unknown",
                summary=prompt,
            )
        # Context-occupancy estimate for the compaction trigger — the
        # high-water mark of the last turn's INPUT (not a running sum,
        # which double-counts resent history and compacts too early).
        self._compact_tokens = _context_high_water(
            self._compact_tokens, tokens_in=tin, cached_in=cached_in
        )

    async def _turn(self, prompt: str, *, agent: Any | None = None) -> None:
        """Stream one agent run for ``prompt``, reusing the
        session so conversation history carries forward.

        ``agent`` overrides the team coordinator for this turn —
        the solo fast path passes the coder-kernel agent here (see
        ``_route_turn``). Default ``None`` keeps the team. Both run
        under the same ``session_id`` + memory db, so history is
        continuous whichever route a turn takes.

        Spinner UX: Rich's ``console.status`` runs continuously for
        the whole turn. The renderer drives its label via two
        callbacks — ``set_status(label)`` updates the text,
        ``pause_status()`` stops it (used while assistant prose is
        streaming, since the spinner shares the cursor line). Labels
        come from the in-flight event: "delegating to coder...",
        "running: pytest -q", "searching: openpyxl write_only", or
        a generic "thinking..." between events. The point is to
        avoid the long blank stretches the old "drop on first event"
        scheme produced in Supervisor mode."""
        # Fresh doom-loop counters for this turn (the guard's post-tool
        # hook steers the model when it re-edits one file or re-runs
        # one failing command; counts are per-turn by design).
        from . import loop_guard

        loop_guard.reset()

        status = console.status(
            "[dim]loomflowing...[/dim]", spinner="dots"
        )
        status.start()
        status_running = True

        def set_status(label: str) -> None:
            """Update the spinner label, restarting it if it was
            paused for a prose burst."""
            nonlocal status_running
            if not status_running:
                status.start()
                status_running = True
            status.update(f"[dim]{label}[/dim]")

        def pause_status() -> None:
            """Stop the spinner so streamed text can use the cursor
            line cleanly. ``set_status`` restarts it later."""
            nonlocal status_running
            if status_running:
                status.stop()
                status_running = False

        # Point the ApprovalGate's spinner hooks at THIS turn's
        # closures. Resume re-labels to a neutral "thinking..." since
        # the gate has no event to name.
        self._active_pause_spinner = pause_status
        self._active_resume_spinner = lambda: set_status("thinking...")

        renderer = StreamRenderer(
            set_status=set_status,
            pause_status=pause_status,
            sandbox=self._sandbox,
            # Defer event rendering while the approval selector is on
            # screen — concurrent prints displace its in-place redraw.
            gate_active=lambda: self._gate_active,
        )

        # The Ralph loop now lives in loomflow itself (>=0.10.8) via
        # the StopHook protocol — Agent(living_plan=True) auto-
        # registers a hook that re-prompts when any plan step is
        # still `doing`/`todo` after the architecture exits. We just
        # consume the agent's stream; the framework handles
        # continuation, bounded by ``max_stop_hook_iterations``.
        ok = await self._consume_agent_stream(
            agent if agent is not None else self.agent,
            prompt,
            renderer,
            pause_status,
        )
        if not ok:
            return

        if renderer.last_plan:
            self.last_plan = renderer.last_plan
        result = renderer.last_result
        # Stash for post-turn inspection (e.g. /goal reads
        # interruption_reason to report goal-met vs guardrail-stop).
        self.last_result = result
        agent_output = ""
        if result:
            self._account_result(result, renderer, prompt)
            self._pending_slugs = list(
                result.get("cited_slugs") or []
            )
            agent_output = str(result.get("output") or "")

            # Weak-model guard: the "answer" is a tool call leaked as
            # text (structured tool_calls was empty, so the loop ended
            # the turn). Nudge ONCE — same session, so the model sees
            # its own leaked reply — then take whatever comes back.
            if _looks_like_leaked_tool_call(agent_output):
                console.print(
                    "  [yellow]model emitted a tool call as text — "
                    "nudging it to use the tool interface[/yellow]"
                )
                nudge_renderer = StreamRenderer(
                    set_status=set_status,
                    pause_status=pause_status,
                    sandbox=self._sandbox,
                    gate_active=lambda: self._gate_active,
                )
                ok = await self._consume_agent_stream(
                    agent if agent is not None else self.agent,
                    _TOOL_LEAK_NUDGE,
                    nudge_renderer,
                    pause_status,
                )
                if ok and nudge_renderer.last_result:
                    n = nudge_renderer.last_result
                    # Same accounting as the main path (cost, all token
                    # buckets incl. cache_write, turns, file touches,
                    # compaction high-water) — via the shared helper so
                    # a nudged turn can't under-count or stall compaction.
                    self._account_result(
                        n, nudge_renderer, prompt, extend_files=True
                    )
                    self.last_result = n
                    agent_output = str(n.get("output") or "")

            # Verify-before-done gate: the turn CHANGED code, CLAIMS
            # completion (prose claim or all-done plan), and never ran
            # the project's tests → one bounded nudge to run them.
            # Composes with the anti-poison gate below: instead of
            # merely deleting a false "done" episode after the fact,
            # make the claim true first. /verify off disables.
            if self._verify_gate:
                from . import verify_gate as vg

                active_root = (
                    self._isolated_project or self.project
                ).root
                claims_done = _looks_like_completion_claim(
                    agent_output
                ) or vg.plan_all_done(renderer.last_plan)
                if vg.should_verify(
                    claims_done=claims_done,
                    files_touched=renderer.files_touched,
                    bash_commands=renderer.bash_commands,
                ):
                    test_cmd = vg.detect_test_command(active_root)
                    if test_cmd is not None:
                        console.print(
                            "  [dim]verify gate: changes were made "
                            "but no tests ran — asking the agent to "
                            f"run {test_cmd}[/dim]"
                        )
                        verify_renderer = StreamRenderer(
                            set_status=set_status,
                            pause_status=pause_status,
                            sandbox=self._sandbox,
                            gate_active=lambda: self._gate_active,
                        )
                        ok = await self._consume_agent_stream(
                            agent
                            if agent is not None
                            else self.agent,
                            vg.VERIFY_NUDGE.format(cmd=test_cmd),
                            verify_renderer,
                            pause_status,
                        )
                        if ok and verify_renderer.last_result:
                            v = verify_renderer.last_result
                            self._account_result(
                                v,
                                verify_renderer,
                                prompt,
                                extend_files=True,
                            )
                            self.last_result = v
                            agent_output = str(
                                v.get("output") or ""
                            )

            # Surface framework-level stop-hook exhaustion so the
            # user knows the cap was hit (and can raise it with
            # /set_continue_cap N).
            if result.get("interrupted") and (
                result.get("interruption_reason")
                == "stop_hook_iterations_exhausted"
            ):
                console.print(
                    "\n  [yellow]plan still had work but the agent "
                    f"hit the auto-continue cap "
                    f"({self._auto_continue_limit}) — type "
                    "'continue' to push further, raise the cap "
                    "with /set_continue_cap N, or accept the "
                    "partial result[/yellow]"
                )

        pause_status()
        self._compact_exchanges.append((prompt, agent_output))

        # Anti-poison gate: if the turn made ZERO tool calls AND the
        # output is a bare completion claim ("all issues fixed"),
        # the episode loomflow just persisted is a hallucination
        # with no grounding — and a self-reinforcing one (recall
        # surfaces it → next turn parrots it → new episode → doom
        # loop). Delete it so it can't poison future recall. We
        # only nuke the no-tool-call completion-claim case;
        # legitimate no-tool answers ("what does X mean?") don't
        # match the completion-claim pattern and are kept.
        n_tool_calls = len(renderer._call_names)
        if n_tool_calls == 0 and _looks_like_completion_claim(
            agent_output
        ):
            # Under /isolate the live session writes to the WORKTREE's
            # .loom/memory.db — target that one, not the main project's.
            active_root = (self._isolated_project or self.project).root
            deleted = _delete_last_episode(
                active_root / LOOM_DIR / "memory.db",
                session_id=self.session_id,
                user_id=_USER_ID,
            )
            if deleted:
                console.print(
                    "  [dim](skipped persisting an unverified "
                    "'done' claim — no tool calls backed it)[/dim]"
                )

        # Persist the current session_id to disk so /resume on the
        # next REPL launch knows what to rehydrate. Done after EVERY
        # turn (not just on /exit) so a crash doesn't lose the
        # session pointer. Cheap — one short write to a small file.
        self._save_session_pointer()

        if self._pending_slugs:
            console.print(
                "  [dim]if that worked, just continue — or "
                "/bad if it didn't[/dim]"
            )
        # End-of-turn separator: a full-width rule closing THIS
        # response, right-labelled with the turn's own token usage +
        # cost, so every answer is cleanly delimited and you can see
        # what it cost at a glance (not just the cumulative session).
        self._print_turn_summary()

        # Maybe compact. Done AFTER the turn renders + the
        # pending-slugs hint prints so the user sees the natural
        # turn boundary before any compaction status appears.
        await self._maybe_compact()

    def _print_turn_summary(self) -> None:
        """Full-width rule + this turn's tokens / cost / context
        occupancy, right-aligned — the horizontal separator between
        responses. Zero cost renders as ``free`` (free-tier models)
        instead of a noisy ``$0.0000``. The ``ctx`` figure is the
        context high-water mark as a % of the model's window — ambient
        context observability, so the user never wonders how close
        they are to compaction."""
        from .context_report import context_percent

        cost = (
            "free" if self._turn_cost == 0 else f"${self._turn_cost:.4f}"
        )
        pct = context_percent(
            self._compact_tokens, self._context_window()
        )
        stats = (
            f"{self._turn_in:,} in · {self._turn_out:,} out · "
            f"{cost} · {pct}% ctx"
        )
        width = console.size.width
        # rule that ends with the stats: dashes + " stats" flush right.
        pad = max(0, width - len(stats) - 3)
        console.print(
            f"[bright_black]{'─' * pad}[/bright_black] "
            f"[dim]{self._turn_in:,} in · {self._turn_out:,} out · "
            f"[green]{cost}[/green] · {pct}% ctx[/dim]"
        )

    # ---- self-improvement attribution -----------------------------------

    async def _attribute_pending(
        self, *, success: bool, quiet: bool
    ) -> None:
        """Flush the pending turn's citations to the workspace,
        crediting (or debiting) the notes the agent read.

        ``quiet`` suppresses the confirmation line — used for the
        implicit 'moved-on = success' path so the REPL doesn't
        chatter on every turn."""
        # Shared pipeline (loom_code.turn) owns what crediting means;
        # the REPL only owns the pending state + console feedback.
        from .turn import attribute_turn

        files = self._pending_files
        slugs = self._pending_slugs
        self._pending_files = []
        self._pending_slugs = []
        n = await attribute_turn(
            self.workspace,
            self.project.root,
            success=success,
            slugs=slugs,
            files=files,
            user_id=_USER_ID,
        )
        if n and not quiet:
            verb = "credited" if success else "debited"
            console.print(
                f"  [dim]{verb} {n} note(s) from the last "
                f"turn[/dim]"
            )

    # ---- slash commands -------------------------------------------------

    async def _handle_slash(self, line: str) -> bool:
        """Dispatch a /command. Returns False to exit the REPL."""
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit"):
            return False
        if cmd == "/help":
            console.print(
                _render_help(
                    extra=[
                        (f"/{c.name}", c.description, "Custom")
                        for c in self._extensions.command_specs
                    ]
                )
            )
        elif cmd == "/init-loom":
            from .rules import init_agents_md

            path, created = init_agents_md(self.project.root)
            if created:
                console.print(
                    f"[green]created {path.name}[/green] — a starter "
                    "rules file loom-code reads every session. Edit it, "
                    'or just state rules in chat (e.g. "never edit X") '
                    "and loom-code will save them here."
                )
            else:
                console.print(
                    f"[dim]{path.name} already exists — loom-code "
                    "already reads it. Edit it directly, or state rules "
                    "in chat.[/dim]"
                )
        elif cmd == "/plan":
            if arg:
                # "/plan <task>" reads as "plan and do <task>" — run
                # it as a normal task. loom-code plans every task
                # anyway (living_plan=True), so the plan shows up
                # mid-stream and `/plan` with no arg replays it.
                await self._attribute_pending(
                    success=True, quiet=False
                )
                await self._inject_loom_context(arg)
                await self._turn(arg)
            elif self.last_plan:
                console.print(Text(self.last_plan, style="dim"))
            else:
                console.print(
                    "[dim]no plan yet — give loom-code a task, or "
                    "`/plan <task>` to start one[/dim]"
                )
        elif cmd == "/context":
            await self._handle_context()
        elif cmd == "/prompt":
            await self._handle_prompt()
        elif cmd == "/verify":
            self._handle_verify(arg)
        elif cmd == "/cost":
            uncached = self.total_in - self.total_cached_in
            # Cache-hit ratio over total input tokens. The ratio
            # tells the user whether their prompt-caching investment
            # is actually paying off — a low ratio means the system
            # prompt is changing turn-to-turn (cache-bust) or the
            # provider doesn't expose cache reads.
            hit_pct = (
                (self.total_cached_in / self.total_in * 100.0)
                if self.total_in > 0
                else 0.0
            )
            console.print(
                Text.assemble(
                    ("  session: ", "dim"),
                    (f"{self.turns} turns", ""),
                    ("  ·  ", "dim"),
                    (
                        f"{uncached:,}+{self.total_cached_in:,} in / "
                        f"{self.total_out:,} out",
                        "",
                    ),
                    ("  ·  ", "dim"),
                    (f"${self.total_cost:.4f}", "green"),
                )
            )
            # Second line: cache breakdown. Only render when there's
            # something to report — keeps the empty-session output
            # uncluttered. ``cache_write`` only fires on Anthropic
            # (5m TTL = +25%, 1h = +100%); on OpenAI it stays 0.
            if self.total_cached_in > 0 or self.total_cache_write > 0:
                cache_color = (
                    "green" if hit_pct >= 50 else "yellow"
                    if hit_pct >= 20 else "dim"
                )
                segments: list[tuple[str, str]] = [
                    ("  cache:   ", "dim"),
                    (f"{hit_pct:.1f}% hit", cache_color),
                ]
                if self.total_cache_write > 0:
                    segments.extend(
                        [
                            ("  ·  ", "dim"),
                            (
                                f"{self.total_cache_write:,} write",
                                "dim",
                            ),
                        ]
                    )
                console.print(Text.assemble(*segments))
            # Third line: token-optimisation tier counters. Each
            # entry only renders when its counter is non-zero —
            # opted-out features stay invisible. The three counters
            # map 1:1 to the three opt-in framework knobs in
            # build_agent (snip_window, auto_compact_at_tokens,
            # tool_result_summarizer) — seeing zeros across the
            # board means "the conversation never got large enough
            # to need any of them," which is a useful diagnostic
            # signal on its own.
            opt_segments: list[tuple[str, str]] = []
            if self.total_snips > 0:
                opt_segments.append(
                    (f"{self.total_snips} snip", "dim")
                )
            if self.total_compacts > 0:
                if opt_segments:
                    opt_segments.append(("  ·  ", "dim"))
                opt_segments.append(
                    (f"{self.total_compacts} compact", "dim")
                )
            if self.total_summaries > 0:
                if opt_segments:
                    opt_segments.append(("  ·  ", "dim"))
                opt_segments.append(
                    (f"{self.total_summaries} tool-summary", "dim")
                )
            if opt_segments:
                console.print(
                    Text.assemble(("  optim:   ", "dim"), *opt_segments)
                )
        elif cmd == "/good":
            if self._pending_slugs:
                await self._attribute_pending(
                    success=True, quiet=False
                )
            else:
                console.print(
                    "  [dim]nothing pending to rate[/dim]"
                )
        elif cmd == "/bad":
            if self._pending_slugs:
                await self._attribute_pending(
                    success=False, quiet=False
                )
            else:
                console.print(
                    "  [dim]nothing pending to rate[/dim]"
                )
        elif cmd == "/model":
            if not arg:
                console.print(
                    f"  [dim]current model: {self.model}[/dim]"
                )
            else:
                self._switch_model(arg)
        elif cmd == "/clear":
            self.session_id = new_id()
            self.last_plan = None
            self._compact_tokens = 0
            self._compact_exchanges.clear()
            reset_paste_stash()
            # Fresh start also revokes outside-file edit grants —
            # they belong to the conversation that named them.
            from . import consent

            consent.reset()
            # Move the on-disk pointer to the NEW session so a
            # later /resume doesn't rewind into the conversation
            # the user just told us to forget. /clear means "I
            # want a fresh start," and that should survive a
            # quit + relaunch.
            self._save_session_pointer()
            console.print(
                "  [dim]fresh conversation — prior turns "
                "forgotten[/dim]"
            )
        elif cmd == "/compact":
            await self._handle_compact()
        elif cmd == "/compress_token_length":
            self._handle_compress_command(arg)
        elif cmd == "/set_model":
            await self._handle_set_model()
        elif cmd == "/set_web":
            await self._handle_set_web()
        elif cmd == "/resume":
            await self._handle_resume(arg)
        elif cmd == "/fork":
            self._handle_fork()
        elif cmd == "/tree":
            self._handle_tree()
        elif cmd == "/export":
            self._handle_export()
        elif cmd == "/set_continue_cap":
            self._handle_set_continue_cap(arg)
        elif cmd == "/effort":
            self._handle_effort(arg)
        elif cmd == "/mode":
            self._handle_mode(arg)
        elif cmd == "/isolate":
            self._handle_isolate()
        elif cmd == "/review":
            self._handle_review()
        elif cmd == "/merge":
            self._handle_merge()
        elif cmd == "/discard":
            self._handle_discard()
        elif cmd == "/mcp":
            await self._handle_mcp()
        elif cmd == "/computer":
            await self._handle_computer(arg)
        elif cmd == "/goal":
            await self._handle_goal(arg)
        else:
            console.print(
                f"  [yellow]unknown command {cmd}[/yellow] — "
                "/help for the list"
            )
        return True

    async def _handle_mcp(self) -> None:
        """List the connected MCP servers + their tools.

        Reads the registry stashed on the coordinator by ``build_agent``.
        Connecting is lazy, so this is the first thing that actually
        opens the sessions — surfaces a misconfigured server here rather
        than mid-task."""
        registry = getattr(self.agent, "_mcp_registry", None)
        if registry is None:
            console.print(
                "  [dim]No MCP servers configured. Add an [[mcp]] block "
                "to .loom/settings.toml (or ~/.loom-code/settings.toml) "
                "and restart.[/dim]"
            )
            return
        names = registry.server_names
        console.print(
            f"  [cyan]MCP servers[/cyan] ({len(names)}): "
            f"{', '.join(names) if names else '—'}"
        )
        try:
            tools = await registry.list_tools()  # lazily connects
        except Exception as exc:  # noqa: BLE001 — surface, don't crash
            console.print(
                f"  [red]failed to list MCP tools:[/red] {exc}"
            )
            return
        if not tools:
            console.print("  [dim]no tools exposed yet.[/dim]")
            return
        console.print(f"  [cyan]tools[/cyan] ({len(tools)}):")
        for t in tools:
            desc = (t.description or "").strip().splitlines()
            first = desc[0] if desc else ""
            console.print(f"    [green]{t.name}[/green]  [dim]{first}[/dim]")

    def _switch_model(self, model: str) -> None:
        """Rebuild the agent on a new model. Keeps the project +
        approval gate; starts a fresh conversation since the new
        model has no history of the old one. The compactor uses
        the new model too; ``_compact_threshold`` stays as-is so a
        user override survives a model switch (auto = -1 just
        recomputes against the new model on the next check)."""
        # Expand friendly provider aliases first (``nvidia/nemotron-…``
        # → ``litellm/nvidia_nim/nvidia/nemotron-…``) so ``/model`` in
        # the REPL accepts the same short forms as the ``--model`` flag,
        # and the key prompt / resolver see the canonical string.
        from .credentials import (
            normalize_model,
            quiet_litellm_model_warnings,
        )

        model = normalize_model(model)
        quiet_litellm_model_warnings(model)
        # Ensure we have a key for the NEW model before
        # constructing — otherwise build_agent crashes inside the
        # provider SDK on a missing key. ensure_key_for_model
        # prompts inline + saves so the switch just works.
        if not ensure_key_for_model(model, console):
            console.print(
                "  [yellow]model switch cancelled — staying on "
                f"{self.model}[/yellow]"
            )
            return
        self.model = model
        self._rebuild_agent()
        # Persist so this model is the default on the next launch — the
        # user shouldn't have to re-pick it every time.
        from .credentials import save_preferred_model

        save_preferred_model(model)
        console.print(
            f"  [dim]switched to {model} — fresh conversation[/dim]"
        )

    def _handle_mode(self, arg: str) -> None:
        """``/mode [default|accept-edits|plan|yolo]`` — set the
        approval mode for calls no permission rule matches. No arg
        shows the current mode. Takes effect immediately (the gate
        object is shared by every agent) — no rebuild needed.

        * ``default`` — ask for every mutation (write/edit/bash).
        * ``accept-edits`` — auto-allow file edits, still ask for bash.
        * ``plan`` — read-only: deny all mutation; the agent can
          research and propose but not touch the tree.
        * ``yolo`` — allow everything (the irreversible-danger gate
          still fires). Same risk profile as ``--yes``.

        Explicit ``deny`` rules in settings.toml beat every mode."""
        from .permissions import Mode, parse_mode

        choice = arg.strip()
        if not choice:
            console.print(
                f"  [dim]current mode: {self._gate.mode.value}[/dim] "
                "[dim](usage: /mode default|accept-edits|plan|yolo)"
                "[/dim]"
            )
            return
        mode = parse_mode(choice)
        if mode is None:
            console.print(
                f"  [yellow]unknown mode {choice!r}[/yellow] — "
                "use default | accept-edits | plan | yolo"
            )
            return
        self._gate.mode = mode
        blurb = {
            Mode.DEFAULT: "asking before every mutation",
            Mode.ACCEPT_EDITS: (
                "auto-allowing file edits; bash still asks"
            ),
            Mode.PLAN: "read-only — all mutation denied",
            Mode.YOLO: (
                "allowing everything (danger gate still fires)"
            ),
        }[mode]
        console.print(
            f"  [dim]mode → [/dim][cyan]{mode.value}[/cyan]"
            f"[dim] — {blurb}[/dim]"
        )

    def _handle_effort(self, arg: str) -> None:
        """``/effort [low|medium|high|off]`` — set the reasoning-effort
        dial + rebuild. No arg shows the current value. ``off`` (or
        ``none``/``default``) clears it back to the provider default.
        Effort only affects reasoning-capable models (Claude extended
        thinking, OpenAI o-series); it's inert on gpt-4.1/4o."""
        choice = arg.strip().lower()
        if not choice:
            console.print(
                f"  [dim]current effort: "
                f"{self._effort or 'default'}[/dim] "
                "[dim](usage: /effort low|medium|high|off)[/dim]"
            )
            return
        if choice in ("off", "none", "default"):
            new_effort: str | None = None
        elif choice in ("low", "medium", "high"):
            new_effort = choice
        else:
            console.print(
                f"  [yellow]unknown effort {choice!r}[/yellow] — "
                "use low | medium | high | off"
            )
            return
        if new_effort == self._effort:
            console.print(
                f"  [dim]effort already {new_effort or 'default'}[/dim]"
            )
            return
        self._effort = new_effort
        self._rebuild_agent()
        console.print(
            f"  [dim]reasoning effort → {new_effort or 'default'} "
            "— fresh conversation[/dim]"
        )

    # ---- session worktree isolation -----------------------------------

    def _handle_isolate(self) -> None:
        """``/isolate`` — run this session in its own git worktree so
        its edits can't collide with another loom-code session on the
        same repo (e.g. a second terminal). Rebuilds the agent rooted
        in the worktree; /merge or /discard finishes."""
        if self._worktree is not None:
            console.print(
                f"  [dim]already isolated on "
                f"{self._worktree.branch}[/dim]"
            )
            return
        if not worktree.is_git_repo(self.project.root):
            console.print("  [yellow]/isolate needs a git repo[/yellow]")
            return
        info, err = worktree.create(self.project.root, self.session_id)
        if info is None:
            console.print(f"  [red]isolate failed:[/red] {err}")
            return
        self._worktree = info
        self._isolated_project = detect_project(info.path)
        self._rebuild_agent()
        console.print(
            f"  [dim]isolated → worktree on [cyan]{info.branch}[/cyan] "
            f"(base {info.base}). Edits stay here until "
            "/merge or /discard.[/dim]"
        )

    def _handle_review(self) -> None:
        """``/review`` — show this isolated session's diff vs its base
        branch (read-only)."""
        if self._worktree is None:
            console.print("  [dim]not isolated — /isolate first[/dim]")
            return
        text, err = worktree.diff(self._worktree)
        if err:
            console.print(f"  [red]diff failed:[/red] {err}")
            return
        if not text.strip():
            console.print("  [dim]no changes in this session yet[/dim]")
            return
        self._print_diff(text)

    def _handle_merge(self) -> None:
        """``/merge`` — review the session's diff, then commit + merge
        its branch into base and return to the main tree."""
        if self._worktree is None:
            console.print("  [dim]not isolated — nothing to merge[/dim]")
            return
        text, _ = worktree.diff(self._worktree)
        if text.strip():
            self._print_diff(text)
        else:
            console.print("  [dim](no changes to merge)[/dim]")
        info = self._worktree
        ok, err = worktree.merge(self.project.root, info)
        if not ok:
            console.print(f"  [red]merge failed:[/red] {err}")
            return
        worktree.remove(self.project.root, info)
        self._worktree = None
        self._isolated_project = None
        self._rebuild_agent()
        console.print(
            f"  [dim]merged [cyan]{info.branch}[/cyan] → {info.base} "
            "+ cleaned up — back on the main tree[/dim]"
        )

    def _handle_discard(self) -> None:
        """``/discard`` — drop this isolated session's edits + remove
        the worktree, returning to the main tree."""
        if self._worktree is None:
            console.print("  [dim]not isolated — nothing to discard[/dim]")
            return
        info = self._worktree
        worktree.remove(self.project.root, info)
        self._worktree = None
        self._isolated_project = None
        self._rebuild_agent()
        console.print(
            f"  [dim]discarded [cyan]{info.branch}[/cyan] — back on "
            "the main tree[/dim]"
        )

    def _print_diff(self, text: str) -> None:
        """Print a unified diff with green/red/hunk colours — same
        vocabulary as the desktop's review modal + edit cards."""
        for raw in text.splitlines():
            if raw.startswith(("+++", "---")):
                style = "dim"
            elif raw.startswith("@@"):
                style = "cyan"
            elif raw.startswith("diff --git") or raw.startswith("index "):
                style = "bold dim"
            elif raw.startswith("+"):
                style = "green"
            elif raw.startswith("-"):
                style = "red"
            else:
                style = "default"
            console.print(Text(raw or " ", style=style))

    async def _handle_computer(self, arg: str) -> None:
        """``/computer [task]`` — turn on COMPUTER OPERATOR mode: the agent
        gets loom-code's built-in browser engine (page_open/observe/act/
        check) + media/app tools + files/shell, under an operator prompt.
        Rebuilds the agent, then (if a task was given) runs it.

        The browser engine is Playwright-based (already installed); a
        visible Chromium window opens on the first page_open. Operator
        mode also upgrades to a STRONGER reasoning model (browser
        comprehension is hard for small models) when its key is
        available — the coding session's model is restored on exit."""
        if not self._browser_mode:
            self._browser_mode = True
            # Bump to a stronger reasoning model for browser comprehension
            # (gpt-4.1-mini struggles with dense dynamic pages). Pick the
            # first candidate whose API key is already set; else keep the
            # current model. Remember the original to restore later.
            self._pre_operator_model = self.model
            strong = self._pick_operator_model()
            if strong and strong != self.model:
                self.model = strong
            self._rebuild_agent()
            console.print(
                "  [green]✓[/green] computer operator on — driving a visible "
                f"browser + files/shell/apps, on [cyan]{self.model}[/cyan]. "
                "[dim]A Chromium window opens on the first web action.[/dim]"
            )
        if arg.strip():
            await self._turn(arg.strip())

    async def _handle_goal(self, arg: str) -> None:
        """``/goal <task>`` — run until the goal is met.

        The agent works on ``<task>`` and, after each pass, a cheap
        same-provider checker model judges whether the goal is
        satisfied; if not, the agent is re-prompted and works again —
        the run-until-done loop (framework ``run_until=`` / GoalStopHook).
        Bounded by three guardrails so it can't spin forever: a max
        re-prompt count, no-progress detection, and a cost cap.

        The task text IS the stop condition. For an explicit split, use
        ``/goal <task> :: <condition>`` — everything before ``::`` is
        what to do, everything after is what the checker tests."""
        arg = arg.strip()
        loom_dir = self.project.root / LOOM_DIR
        if not arg:
            console.print(
                "  [yellow]usage: /goal <task> — e.g. "
                "/goal make all tests pass[/yellow]\n"
                "  [dim]optional explicit condition: "
                "/goal <task> :: <condition> · resume an unfinished "
                "goal: /goal resume[/dim]"
            )
            return

        if arg.lower() == "resume":
            # Durable goals: pick up where a prior process left off.
            state = _load_goal_state(loom_dir)
            if state is None:
                console.print(
                    "  [dim]no unfinished goal recorded for this "
                    "project.[/dim]"
                )
                return
            task = str(state["task"])
            condition = str(state.get("condition") or task)
            prior_sid = str(state.get("session_id") or "")
            if prior_sid and prior_sid != self.session_id:
                # Rejoin the goal's conversation so the agent
                # remembers what it already tried (loomflow
                # rehydrates by session_id).
                self.session_id = prior_sid
                self._compact_tokens = 0
                self._compact_exchanges.clear()
                console.print(
                    f"  [dim]rejoined goal session "
                    f"{prior_sid[:8]}…[/dim]"
                )
            console.print(
                f"  [dim]resuming goal from "
                f"{str(state.get('started_at') or '')[:16]}[/dim]"
            )
        # Split the optional "task :: condition" form. Default: the
        # task is also the condition (the framework's str happy-path).
        elif "::" in arg:
            task, _, condition = arg.partition("::")
            task, condition = task.strip(), condition.strip()
            if not task or not condition:
                console.print(
                    "  [yellow]both sides of :: must be non-empty — "
                    "/goal <task> :: <condition>[/yellow]"
                )
                return
        else:
            task = condition = arg

        # Persist BEFORE running: a crash, Ctrl-C, or guardrail stop
        # leaves the goal on disk, resumable across restarts.
        _save_goal_state(
            loom_dir,
            task=task,
            condition=condition,
            session_id=self.session_id,
            model=str(self.model),
        )

        # Cheap same-provider checker (Haiku / gpt-4.1-mini); falls back
        # to the main model inside the framework when no cheap key.
        checker = self._pick_checker_model()
        # Guardrails — the research is unanimous these prevent runaway
        # cost. max_iterations doubles as the loop's hard cap; the
        # framework caps each re-prompt and bails on no-progress / cost.
        self._run_until = {
            "condition": condition,
            "max_iterations": 20,
            "max_no_progress": 3,
            "max_cost_usd": 2.0,
        }
        if checker is not None:
            self._run_until["checker"] = checker

        # The goal loop needs room to re-prompt — loom-code's default
        # auto-continue cap (2) is far too low for run-until-done. Lift
        # it for this goal turn; restore after. The GoalStopHook's own
        # max_iterations is the real bound.
        saved_cap = self._auto_continue_limit
        self._auto_continue_limit = max(saved_cap, 20)
        # keep_session: the goal must run WITH the conversation so far
        # ("/goal fix the bug we discussed") — rebuilding is only about
        # arming the run_until hook, not starting over.
        try:
            self._rebuild_agent(keep_session=True)
        except TypeError:
            # Installed loomflow predates ``run_until=``. Disarm and
            # rebuild clean so the REPL keeps working — /goal is the
            # only casualty, not the session.
            self._run_until = None
            self._auto_continue_limit = saved_cap
            self._rebuild_agent(keep_session=True)
            console.print(
                "  [yellow]/goal needs a newer loomflow than is "
                "installed (Agent(run_until=) is missing). "
                "Upgrade loomflow and retry.[/yellow]"
            )
            return

        checker_label = checker or f"{self.model} (no cheap checker key)"
        console.print(
            f"  [green]🎯 goal:[/green] {condition}\n"
            f"  [dim]checker {checker_label} · max 20 passes · "
            "no-progress 3 · cap $2.00 · Esc to stop[/dim]"
        )

        try:
            await self._inject_loom_context(task)
            await self._turn(task)
        finally:
            # Disarm the goal: clear the spec, restore the cap, rebuild
            # back to a normal coding agent for the next message —
            # keeping the session so the goal turn stays part of the
            # conversation history.
            self._run_until = None
            self._auto_continue_limit = saved_cap
            self._rebuild_agent(keep_session=True)

        # Report whether the goal was met or a guardrail stopped it. The
        # framework sets interruption_reason="run_until:<reason>" on a
        # guardrail stop; a clean condition_met leaves interrupted False.
        result = self.last_result
        reason = (result or {}).get("interruption_reason") or ""
        if reason.startswith("run_until:"):
            why = reason.split(":", 1)[1]
            pretty = {
                "max_iterations": "hit the 20-pass cap",
                "no_progress": "stopped making progress",
                "cost_cap": "hit the $2.00 cost cap",
            }.get(why, why)
            console.print(
                f"  [yellow]⚠ goal not confirmed — {pretty}.[/yellow] "
                "[dim]Review the work above; /goal resume continues "
                "it — even after a restart.[/dim]"
            )
        elif reason == "stop_hook_iterations_exhausted":
            console.print(
                "  [yellow]⚠ goal not confirmed — auto-continue cap "
                "reached.[/yellow] [dim]/goal resume continues it — "
                "even after a restart.[/dim]"
            )
        else:
            # Confirmed met — retire the durable record.
            _clear_goal_state(loom_dir)
            console.print(
                "  [green]✓ goal met[/green] — the checker confirmed the "
                "condition."
            )

    def _pick_operator_model(self) -> str | None:
        """Choose a strong reasoning model for operator mode — the first
        candidate whose API key is already configured (so we never prompt
        or fail). Returns None to keep the current model if no stronger
        one is usable."""
        from .credentials import required_env_for_model

        cur = self.model.lower()

        # If already on a capable model, keep it (don't downgrade).
        if cur in ("gpt-4.1", "claude-sonnet-4-6", "claude-opus-4-7") \
                or "opus" in cur:
            return None

        # Choose the upgrade by which PROVIDER the user is already on, so
        # we never switch to a provider whose account may be unfunded.
        # A set key only proves the key exists, NOT that it has credits
        # (Anthropic 400s "credit balance too low" otherwise) — so we
        # stay within the current provider's family.
        if "claude" in cur:
            target = "claude-sonnet-4-6"
        else:
            # OpenAI family (gpt-*, o-series) → the strong OpenAI model.
            target = "gpt-4.1"

        env = required_env_for_model(target)
        if env is None or os.environ.get(env):
            return target
        return None

    def _pick_checker_model(self) -> str | None:
        """Choose a CHEAP, fast checker for /goal's run-until loop — the
        small model in the SAME provider as the current model. The
        checker runs once per loop pass to judge DONE/NOT_DONE, so it
        should be cheap; staying in-provider avoids switching to an
        account that may be unfunded (the funding lesson from operator
        mode — a set key doesn't prove credits). Returns None to let the
        framework fall back to the main model when no cheap key is set.

        Delegates to :func:`credentials.cheap_model_for` — the same
        picker the compactor and tool-result summariser use. One
        deliberate difference from the original inline logic: local /
        litellm models now return None (fall back to the main model)
        instead of silently routing judgements to OpenAI — an
        Ollama user shouldn't leak session content to a cloud
        provider just because OPENAI_API_KEY happens to be set."""
        return cheap_model_for(self.model)

    # ---- adaptive routing (solo fast path) ------------------------------

    async def _route_turn(self, prompt: str) -> str:
        """Pick ``"solo"`` or ``"team"`` for this turn.

        The supervisor team taxes a one-line fix with a full
        delegation round-trip (coordinator reads → delegates → coder
        re-reads), so obviously-small write tasks run on a standalone
        coder kernel instead. The decision is conservative — every
        branch that isn't a confident SOLO falls back to the team:

        * /goal armed or operator mode → team (their hooks live on
          the coordinator).
        * Question-shaped prompts → team (the read-only coordinator
          answers those directly — no delegation tax to dodge — and
          it holds the repo map + notebook tools).
        * Otherwise the cheap classifier votes; no usable cheap
          model, classifier error, or anything but a clear SOLO →
          team. A misroute therefore costs at most the status-quo
          overhead, never a lost capability.
        """
        if self._run_until is not None or self._browser_mode:
            return "team"
        if _looks_like_question(prompt):
            return "team"
        if _references_prior_context(prompt):
            # "fix it" / "continue" lean on history the stateless
            # classifier can't see — the coordinator has it.
            return "team"
        return (
            "solo"
            if await self._classify_task(prompt) == "SOLO"
            else "team"
        )

    async def _classify_task(self, prompt: str) -> str:
        """One-word SOLO/TEAM vote from the cheap same-provider
        model (~a hundred tokens, fractions of a cent — repaid many
        times over when it saves one delegation round-trip).

        litellm-routed models have no cheap sibling
        (``cheap_model_for`` returns None to avoid crossing
        providers) — but disabling the classifier there forced EVERY
        turn onto the heavy TEAM path, the worst deal for exactly the
        providers with the weakest models. Classify with the model
        ITSELF instead: the call is ~100 tokens, and on the free
        tiers this targets (NVIDIA NIM) it costs nothing. Local
        ollama/echo stay disabled — an extra local call is pure
        latency with no cost to save."""
        cheap = cheap_model_for(self.model)
        if cheap is None:
            model_str = str(self.model).lower()
            if model_str.startswith("litellm/"):
                cheap = self.model  # self-classification
            else:
                return "TEAM"
        try:
            if self._router_agent is None:
                from loomflow import Agent as _Agent

                self._router_agent = _Agent(
                    _ROUTER_PROMPT, model=cheap, prompt_caching=True
                )
            result = await self._router_agent.run(
                prompt[:2000], user_id=_USER_ID
            )
            return (
                "SOLO" if "SOLO" in result.output.upper() else "TEAM"
            )
        except Exception:  # noqa: BLE001 — routing must never kill a turn
            return "TEAM"

    def _get_solo_agent(self) -> Any:
        """Lazily build the standalone coder for the fast path —
        shares the team's memory db + notebook so context stays
        continuous across routes. Invalidated on /model, /set_web,
        and isolation changes via ``_rebuild_agent``."""
        if self._solo_agent is None:
            build_project = self._isolated_project or self.project
            self._solo_agent = build_solo_agent(
                build_project,
                model=self.model,
                approval_handler=self._gate.handler,
                web_backend=self._web_backend,
                effort=self._effort,
                sandbox=self._sandbox,
                sandbox_allow_network=self._sandbox_allow_network,
                extensions=self._extensions,
            )
        return self._solo_agent

    def _rebuild_agent(self, *, keep_session: bool = False) -> None:
        """Reconstruct the supervisor + workers using the current
        ``self.model`` and ``self._web_backend``. Used by
        ``/model`` (model change) and ``/set_web`` (backend change).
        Bundled skills (graphify et al.) are auto-registered
        inside ``build_agent`` so we don't pass them explicitly
        here.

        ``keep_session=True`` preserves ``session_id`` (and the
        compaction accumulators that mirror it) across the rebuild —
        used by ``/goal``, which rebuilds only to arm/disarm the
        ``run_until`` hook on the SAME conversation: "/goal fix the
        bug we discussed" must see the discussion, and the goal
        turn must stay part of the session history afterwards. The
        default (fresh session) is right for ``/model`` and
        ``/set_web``, where history is model-specific."""
        # When isolated, build rooted at the worktree (its own working
        # copy + .loom). Extensions stay ``self._extensions`` — they're
        # the MAIN project's .loom config, which the worktree (being
        # gitignored) doesn't have a copy of, so an isolated session
        # would otherwise lose its skills/subagents/hooks.
        build_project = self._isolated_project or self.project
        self.agent, self.workspace = build_agent(
            build_project,
            model=self.model,
            approval_handler=self._gate.handler,
            web_backend=self._web_backend,
            max_stop_hook_iterations=self._auto_continue_limit,
            extensions=self._extensions,
            effort=self._effort,
            sandbox=self._sandbox,
            sandbox_allow_network=self._sandbox_allow_network,
            operator=self._browser_mode,
            run_until=self._run_until,
        )
        # Routing agents are model-derived — drop them so the next
        # solo route / classification rebuilds on the new config.
        self._solo_agent = None
        self._router_agent = None
        # New agent/memory — forget what blocks we last wrote so the
        # dirty-check can't wrongly skip the first write.
        self._block_hashes.clear()
        self._compactor = Compactor(
            model=cheap_model_for(self.model) or self.model
        )
        if not keep_session:
            self._compact_tokens = 0
            self._compact_exchanges.clear()
            self.session_id = new_id()

    # ---- automatic compaction ------------------------------------------

    def _active_threshold(self) -> int:
        """Resolve the live threshold:

        * positive int  → explicit user override (set via
          ``/compress_token_length N``)
        * 0             → user disabled compaction (``... off``)
        * -1 (sentinel) → recompute from the active model
        """
        if self._compact_threshold >= 0:
            return self._compact_threshold
        return default_compact_threshold(self.model)

    async def _maybe_compact(self) -> None:
        """If cumulative tokens have crossed the active threshold,
        run the compactor, write its summary to the agent's memory
        as a working block (auto-injected into every subsequent
        system prompt), and reset the conversation thread."""
        threshold = self._active_threshold()
        if threshold == 0:
            return  # explicitly disabled
        if self._compact_tokens < threshold:
            return
        if not self._compact_exchanges:
            return
        console.print(
            f"  [dim]compacting {self._compact_tokens:,} tokens of "
            f"history (threshold {threshold:,})...[/dim]"
        )
        await self._compact_now()

    def _context_window(self) -> int:
        """The active model's context window, cached per model string
        (``context_window_for`` warns on unknown models — resolve once,
        not every turn)."""
        if self._ctx_window_model != self.model:
            import warnings

            from loomflow.agent.auto_compact import context_window_for

            from .credentials import context_window_override

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._ctx_window = context_window_override(
                    self.model
                ) or context_window_for(self.model)
            self._ctx_window_model = self.model
        return self._ctx_window

    async def _working_blocks(self) -> list[tuple[str, str]]:
        """Every working block loomflow will inject into the next
        system prompt, as ``(name, content)``. Empty list when the
        memory backend doesn't expose ``working`` (custom backends)
        — observability degrades, never crashes."""
        try:
            blocks = await self.agent.memory.working(user_id=_USER_ID)
            return [
                (str(b.name), str(b.content or "")) for b in blocks
            ]
        except Exception:  # noqa: BLE001 — report-only path
            return []

    async def _handle_context(self) -> None:
        """``/context`` — show exactly what occupies the model's
        context: window, used tokens (the same high-water figure the
        auto-compactor keys on), every injected working block with its
        size, and where compaction will fire. The transparency answer
        to harnesses that inject invisibly."""
        from .context_report import context_report

        report = context_report(
            model=self.model,
            window=self._context_window(),
            used_tokens=self._compact_tokens,
            threshold=self._active_threshold(),
            blocks=await self._working_blocks(),
            n_exchanges=len(self._compact_exchanges),
        )
        console.print()
        for line in report.splitlines():
            # markup OFF (block names may contain [ ]), style via kwarg
            console.print(f"  {line}", markup=False, style="dim")

    async def _handle_prompt(self) -> None:
        """``/prompt`` — dump the model's ACTUAL system prompt: the
        coordinator's static instructions plus every working block,
        verbatim. No other harness shows this; loom-code's position is
        that you should be able to read every byte the model reads."""
        from .context_report import prompt_dump

        instructions = getattr(self.agent, "_instructions", None)
        dump = prompt_dump(
            instructions=(
                str(instructions) if instructions else None
            ),
            blocks=await self._working_blocks(),
        )
        console.print()
        console.print(dump, markup=False, highlight=False)
        console.print()
        console.print(
            "  [dim](this + the conversation is everything the model "
            "sees; blocks refresh each turn)[/dim]"
        )

    def _handle_verify(self, arg: str) -> None:
        """``/verify [on|off]`` — toggle the verify-before-done gate.
        Bare ``/verify`` shows the current state."""
        arg = arg.strip().lower()
        if arg == "on":
            self._verify_gate = True
        elif arg == "off":
            self._verify_gate = False
        elif arg:
            console.print(
                f"  [yellow]unknown option {arg!r} — use /verify "
                "on|off[/yellow]"
            )
            return
        state = "on" if self._verify_gate else "off"
        console.print(
            f"  [dim]verify-before-done gate: {state} — a turn that "
            "edits code and claims completion without running tests "
            f"{'gets' if self._verify_gate else 'would get'} one "
            "nudge to run them[/dim]"
        )

    async def _handle_compact(self) -> None:
        """``/compact`` — force a compaction NOW, regardless of the
        auto threshold. Useful right before a big new task: fold the
        session so far into a dense summary and start the next turn
        on a fresh, cheap thread."""
        if not self._compact_exchanges:
            console.print(
                "  [dim]nothing to compact yet — no completed "
                "turns this session[/dim]"
            )
            return
        console.print(
            f"  [dim]compacting {self._compact_tokens:,} tokens of "
            f"history (manual)...[/dim]"
        )
        await self._compact_now()

    async def _compact_now(self) -> None:
        """The shared compaction body: summarise, land the summary as
        a working block, reset the thread. Callers gate on
        ``_compact_exchanges`` being non-empty and print their own
        lead-in line.

        Fires the ``PreCompact`` / ``PostCompact`` REPL hooks around
        the fold (auto AND manual /compact both route here), so users
        can e.g. export the full transcript before it's summarised."""
        await self._fire_repl_hooks("PreCompact")
        try:
            summary = await self._compactor.compact(
                self._compact_exchanges
            )
        except Exception as exc:  # noqa: BLE001 — never fatal
            console.print(
                f"  [yellow]compaction failed: {exc} — continuing "
                "without it (use /clear if you hit context "
                "limits)[/yellow]"
            )
            return

        if not summary:
            return

        # Land the summary as a working block. loomflow auto-
        # injects working blocks into every subsequent system
        # prompt, so the next turn starts on a fresh session_id
        # but immediately "remembers" the session via this block.
        try:
            await self.agent.memory.update_block(
                "session_summary", summary, user_id=_USER_ID
            )
        except Exception as exc:  # noqa: BLE001 — never fatal
            console.print(
                f"  [yellow]could not write summary to memory: "
                f"{exc}[/yellow]"
            )
            return

        # Visible before → after: never compact silently (the silent
        # version is a top harness-trust complaint elsewhere).
        before = self._compact_tokens
        after = max(1, len(summary) // 4)
        self.session_id = new_id()
        self._compact_tokens = 0
        self._compact_exchanges.clear()
        console.print(
            f"  [dim]compacted: {before:,} tokens of history → "
            f"~{after:,}-token summary (kept in every future prompt); "
            "fresh conversation thread.[/dim]"
        )
        await self._fire_repl_hooks("PostCompact")

    def _handle_set_continue_cap(self, arg: str) -> None:
        """``/set_continue_cap [N]`` — view or set the auto-continue cap.

        Bare ``/set_continue_cap`` shows the current value. ``N=0``
        disables auto-continue entirely (turns become single-shot
        again — useful when debugging a model's behaviour and you
        want to see exactly what it does on its own). Otherwise N
        is the new cap; we clamp at 100 to prevent typos like
        ``/set_continue_cap 1000`` from costing the user real money.
        """
        arg = arg.strip()
        if not arg:
            console.print(
                f"  [dim]auto-continue cap: "
                f"[b]{self._auto_continue_limit}[/b]  "
                f"(default {_AUTO_CONTINUE_LIMIT_DEFAULT}, "
                "0 disables)[/dim]"
            )
            return
        try:
            n = int(arg)
        except ValueError:
            console.print(
                "  [yellow]usage: /set_continue_cap <N> — N is "
                "an integer ≥ 0 (0 disables)[/yellow]"
            )
            return
        if n < 0:
            console.print(
                "  [yellow]cap must be non-negative (use 0 to "
                "disable auto-continue)[/yellow]"
            )
            return
        if n > 100:
            console.print(
                "  [yellow]cap clamped to 100 to prevent runaway "
                "cost on a typo. Use /set_continue_cap 100 if you "
                "really meant that.[/yellow]"
            )
            n = 100
        old = self._auto_continue_limit
        self._auto_continue_limit = n
        # The cap is a construction-time kwarg on loomflow's Agent
        # (max_stop_hook_iterations). Rebuild so the new value
        # takes effect; this also resets the conversation, which
        # matches the rebuild semantics of /model and /set_web.
        self._rebuild_agent()
        if n == 0:
            console.print(
                f"  [dim]auto-continue [b red]disabled[/b red]  "
                f"(was {old}). Multi-step plans now stop after "
                "their first ReAct exit; type 'continue' to nudge "
                "manually.[/dim]"
            )
        else:
            console.print(
                f"  [dim]auto-continue cap: [b]{old}[/b] → "
                f"[b green]{n}[/b green][/dim]"
            )

    def _handle_compress_command(self, arg: str) -> None:
        """Dispatch ``/compress_token_length`` — view, set, auto, off."""
        arg = arg.strip().lower()
        if not arg:
            current = self._active_threshold()
            mode = (
                "off (disabled)"
                if self._compact_threshold == 0
                else (
                    f"user-set ({current:,})"
                    if self._compact_threshold > 0
                    else f"auto ({current:,}, "
                    f"80% of {self.model}'s context window)"
                )
            )
            console.print(
                f"  [dim]compaction threshold: {mode}[/dim]\n"
                f"  [dim]used this session so far: "
                f"{self._compact_tokens:,} tokens[/dim]"
            )
            return
        if arg == "auto":
            self._compact_threshold = -1
            console.print(
                f"  [dim]threshold: auto "
                f"({self._active_threshold():,})[/dim]"
            )
            return
        if arg == "off":
            self._compact_threshold = 0
            console.print(
                "  [dim]auto-compaction disabled[/dim]"
            )
            return
        try:
            n = int(arg.replace(",", "").replace("_", ""))
        except ValueError:
            console.print(
                "  [yellow]usage: /compress_token_length <N> | "
                "auto | off[/yellow]"
            )
            return
        if n <= 0:
            console.print(
                "  [yellow]threshold must be positive (use 'off' "
                "to disable)[/yellow]"
            )
            return
        self._compact_threshold = n
        console.print(
            f"  [dim]threshold set to {n:,} tokens[/dim]"
        )

    # ---- /set_model + /set_web (interactive provider setup) ----------

    async def _select_menu(
        self,
        title: str,
        options: list[tuple[str, str]],
        *,
        default: int = 0,
    ) -> str | None:
        """Arrow-key vertical menu (↑/↓ + Enter, or a number/hotkey) —
        the same selector the approval prompt uses, for the REPL's
        pick-a-thing prompts (/set_model, /set_web, model lists, …).

        ``options`` is ``[(key, label), …]``; returns the chosen key,
        or ``None`` if cancelled (Esc / Ctrl-C). On a non-TTY it falls
        back to a typed line matched against the keys, so scripted use
        and tests still work.

        The raw-mode selector runs on a worker thread (its blocking
        key reads must not stall the event loop), mirroring how the
        ApprovalGate calls ``_select_option``."""
        from .approval import _select_option

        console.print()
        if title:
            console.print(f"  [bold]{title}[/bold]")
        # ``_select_option`` OWNS the option rendering (it redraws the
        # numbered list in place on each keypress); we only print the
        # title above it. A trailing "Cancel" is the safe last option
        # the selector maps Esc/EOF to — so cancel is distinguishable
        # from a real pick.
        menu = [*options, ("\x00cancel", "Cancel")]
        try:
            choice = await anyio.to_thread.run_sync(
                lambda: _select_option(menu, default=default)
            )
        except (EOFError, KeyboardInterrupt):
            choice = "\x00cancel"
        # ERASE the whole menu (blank line + title + every option row)
        # so a subsequent menu REPLACES this one in place rather than
        # stacking below it. 1 blank + (title?1:0) + len(menu) rows.
        if sys.stdout.isatty():
            n = 1 + (1 if title else 0) + len(menu)
            sys.stdout.write(f"\x1b[{n}F\x1b[0J")
            sys.stdout.flush()
        if choice == "\x00cancel":
            return None
        return choice

    async def _prompt_line(self, message: str) -> str | None:
        """Read one line from the user with a fresh PromptSession.

        We deliberately do NOT reuse ``self._prompt_session`` here.
        prompt_toolkit's PromptSession holds state on its instance
        (``is_password``, completers, key bindings) and even though
        ``prompt_async`` is supposed to save/restore per-call,
        empirically the redact-mode leaked into the next main-loop
        prompt after the secret prompt returned. A throwaway
        session per inline question keeps the main REPL's session
        pristine.

        Returns ``None`` on EOF / Ctrl-C so callers can treat the
        cancel path uniformly."""
        try:
            return (
                await PromptSession().prompt_async(message)
            ).strip()
        except (EOFError, KeyboardInterrupt):
            return None

    async def _prompt_secret(self, message: str) -> str | None:
        """Same as ``_prompt_line`` but hides the input —
        ``is_password=True`` makes prompt_toolkit redact keystrokes
        (no terminal echo, no shell history). Same fresh-session
        rationale as ``_prompt_line`` — and ESPECIALLY important
        here, because this is the prompt whose state was leaking
        back into the main REPL."""
        try:
            return (
                await PromptSession().prompt_async(
                    message, is_password=True
                )
            ).strip()
        except (EOFError, KeyboardInterrupt):
            return None

    async def _handle_set_model(self) -> None:
        """``/set_model`` — pick a provider, ensure its API key (ask
        for it FIRST if missing), then pick a specific model from that
        provider's list. Key-before-models so you can't choose a model
        you can't authenticate.

        Two-level navigation: cancelling the MODEL menu returns to the
        PROVIDER menu (the loop's ``continue``); only cancelling the
        provider menu itself exits ``/set_model`` entirely."""
        while True:
            choice = await self._select_menu(
                "Pick a model provider:",
                [
                    ("1", "OpenAI     (gpt-4.1, gpt-5.1, o4-mini, …)"),
                    ("2", "Anthropic  (claude-opus-4-8, sonnet-4-6, …)"),
                    ("3", "NVIDIA     (Nemotron — free at build.nvidia.com)"),
                    ("4", "Other      (Groq / Together / any litellm)"),
                ],
            )
            if choice is None:
                # Only a provider-menu cancel exits the command.
                return
            if choice == "4":
                await self._set_model_other()
                return

            provider = {
                "1": ("OpenAI", "OPENAI_API_KEY", self._OPENAI_MODELS),
                "2": (
                    "Anthropic",
                    "ANTHROPIC_API_KEY",
                    self._ANTHROPIC_MODELS,
                ),
                "3": ("NVIDIA", "NVIDIA_NIM_API_KEY", self._NVIDIA_MODELS),
            }[choice]
            label, env_name, models = provider

            # KEY FIRST — ask for the provider's key before showing
            # models, so a user without a key sets it up rather than
            # picking a model that then fails to authenticate. A cancel
            # here returns to the provider menu (not a full exit).
            if not await self._ensure_provider_key(label, env_name):
                continue

            # Then pick a specific model. NVIDIA uses its own picker
            # (its custom-input routes a bare ``nvidia/x`` through
            # litellm/nim); OpenAI/Anthropic use the generic list.
            if choice == "3":
                target_model = await self._pick_nvidia_model()
            else:
                target_model = await self._pick_from_models(label, models)
            if target_model is None:
                # Back out to the provider menu instead of exiting.
                continue
            console.print(f"  [dim]switching to {target_model}[/dim]")
            self._switch_model(target_model)
            return

    async def _ensure_provider_key(
        self, label: str, env_name: str
    ) -> bool:
        """Make sure ``env_name`` is set, prompting + saving it if not.
        Returns True to proceed, False if the user cancelled. Shown
        BEFORE the model list in /set_model (key-first flow).

        Silent on the common path: if the key is already set we just
        proceed to the model list. (Printing "already set" here echoed
        once per provider re-visit when backing out of the model menu —
        the stack of "using it" lines the user saw.)"""
        if os.environ.get(env_name):
            return True
        from .credentials import signup_url_for

        console.print(
            f"  [yellow]No {env_name} set.[/yellow] "
            f"loom-code needs it to use {label}."
        )
        console.print(
            f"  Get one at [dim]{signup_url_for(env_name)}[/dim]"
        )
        key = await self._prompt_secret(f"  Paste your {env_name}: ")
        if not key:
            console.print("  [yellow]no key entered — aborting[/yellow]")
            return False
        save_credential(env_name, key)
        os.environ[env_name] = key
        console.print(
            f"  [green]✓[/green] saved {env_name} "
            "(future sessions pick it up automatically)"
        )
        return True

    async def _pick_from_models(
        self, label: str, models: list[tuple[str, str, str]]
    ) -> str | None:
        """Arrow-key menu over a provider's (name, model_id, note)
        list, plus a 'type your own' escape. Returns the chosen model
        string, or None if cancelled."""
        options = [
            (str(i), f"{name:24} {note}")
            for i, (name, _mid, note) in enumerate(models, 1)
        ]
        options.append(("custom", "Type a different model id…"))
        choice = await self._select_menu(
            f"{label} models:", options
        )
        if choice is None:
            return None
        if choice.isdigit():
            return models[int(choice) - 1][1]
        ans = await self._prompt_line("  Model id: ")
        if not ans:
            return None
        from .credentials import normalize_model

        return normalize_model(ans)

    # Curated model lists per provider — (label, model_id, note). The
    # /set_model flow shows these as a sub-menu after the key is set,
    # so the user picks a SPECIFIC model, not just a provider default.
    # A "type your own" escape covers anything not listed.
    _OPENAI_MODELS: list[tuple[str, str, str]] = [
        ("gpt-4.1-mini", "gpt-4.1-mini", "fast + cheap, solid tools"),
        ("gpt-4.1", "gpt-4.1", "stronger general coding"),
        ("gpt-5.1", "gpt-5.1", "flagship reasoning (if you have access)"),
        ("o4-mini", "o4-mini", "reasoning, cost-efficient"),
    ]
    _ANTHROPIC_MODELS: list[tuple[str, str, str]] = [
        ("claude-haiku-4-5", "claude-haiku-4-5", "fast + cheap"),
        (
            "claude-sonnet-4-6",
            "claude-sonnet-4-6",
            "best speed/intelligence balance",
        ),
        (
            "claude-opus-4-8",
            "claude-opus-4-8",
            "most capable — top tool-use + agentic",
        ),
        ("claude-opus-4-7", "claude-opus-4-7", "previous-gen Opus"),
    ]

    # NVIDIA NIM models, ordered small→large. loom-code is tool-heavy
    # (delegate/write/edit/bash), so models with solid OpenAI-format
    # function calling do far better — those are marked. IDs verified
    # against NVIDIA's live /v1/models catalog.
    _NVIDIA_MODELS: list[tuple[str, str, str]] = [
        (
            "nemotron-nano-9b-v2",
            "litellm/nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2",
            "small/fast — weak at multi-step tool use",
        ),
        (
            "nemotron-super-49b-v1.5",
            "litellm/nvidia_nim/nvidia/"
            "llama-3.3-nemotron-super-49b-v1.5",
            "stronger, function-calling — better for real tasks",
        ),
        (
            "llama-3.3-70b",
            "litellm/nvidia_nim/meta/llama-3.3-70b-instruct",
            "general 70B, function-calling",
        ),
        (
            "deepseek-v4-pro",
            "litellm/nvidia_nim/deepseek-ai/deepseek-v4-pro",
            "strong code/reasoning (MoE)",
        ),
    ]

    async def _pick_nvidia_model(self) -> str | None:
        """Arrow-key menu of common NVIDIA NIM models + a "type your
        own" escape. Returns the chosen litellm model string, or None
        if cancelled."""
        from .credentials import normalize_model

        options = [
            (str(i), f"{name:22} {note}")
            for i, (name, _model, note) in enumerate(
                self._NVIDIA_MODELS, 1
            )
        ]
        options.append(("custom", "Type a different model id…"))
        choice = await self._select_menu(
            "NVIDIA models (free at build.nvidia.com):", options
        )
        if choice is None:
            return None
        if choice.isdigit():
            return self._NVIDIA_MODELS[int(choice) - 1][1]
        # "custom" → free-type an id (bare vendor id → routed via NIM,
        # explicit litellm string kept as-is).
        ans = await self._prompt_line(
            "  Model id (e.g. nvidia/nemotron-…): "
        )
        if not ans:
            return None
        if ans.lower().startswith("litellm/"):
            return ans
        if "/" in ans:
            return f"litellm/nvidia_nim/{ans}"
        return normalize_model(f"nvidia/{ans}")

    async def _set_model_other(self) -> None:
        """``/set_model`` → Other — the fully generic path for ANY
        provider loomflow can route through LiteLLM (Groq, Together,
        DeepSeek, a custom OpenAI-compatible proxy, ...).

        The user types a model string; :func:`normalize_model` in
        ``_switch_model`` expands a known alias, and
        ``ensure_key_for_model`` prompts for the right key (with a
        signup link) when the provider is in the registry. Providers
        added via ``~/.loom-code/settings.toml`` ``[[provider]]`` blocks
        work here too, with no code change."""
        from .credentials import litellm_providers

        console.print()
        known = ", ".join(sorted(litellm_providers()))
        console.print(
            "  [dim]Enter any model string loom-code can route, e.g.:[/dim]"
        )
        console.print(
            "    [cyan]groq/llama-3.3-70b-versatile[/cyan]   "
            "[dim](short alias)[/dim]"
        )
        console.print(
            "    [cyan]litellm/deepseek/deepseek-chat[/cyan]   "
            "[dim](explicit litellm form)[/dim]"
        )
        console.print(f"  [dim]known providers: {known}[/dim]")
        console.print(
            "  [dim]add more in ~/.loom-code/settings.toml "
            "([[provider]] blocks)[/dim]"
        )
        model = await self._prompt_line("  Model: ")
        if not model:
            console.print("  [dim]cancelled[/dim]")
            return
        # _switch_model normalizes the alias + prompts for the key.
        self._switch_model(model)

    async def _handle_set_web(self) -> None:
        """``/set_web`` — pick a web-search backend (or disable).
        Serper prompts for the API key on first use; DuckDuckGo
        needs nothing. Rebuilds the agent so the new tool wiring
        takes effect on the next turn."""
        choice = await self._select_menu(
            "Web search backend:",
            [
                ("1", "Serper      (Google, best quality, needs API key)"),
                ("2", "DuckDuckGo  (free, no key, lower quality)"),
                ("3", "Off         (disable web search)"),
            ],
        )
        if choice is None:
            return

        if choice == "1":
            # Serper needs SERPER_API_KEY. Prompt if missing,
            # save it so future sessions pick it up.
            if not os.environ.get("SERPER_API_KEY"):
                console.print(
                    "  [dim]Get a key at "
                    "https://serper.dev "
                    "(2,500 lifetime free searches).[/dim]"
                )
                key = await self._prompt_secret(
                    "  Paste your SERPER_API_KEY: "
                )
                if not key:
                    console.print(
                        "  [yellow]no key entered — "
                        "aborting[/yellow]"
                    )
                    return
                save_credential("SERPER_API_KEY", key)
                os.environ["SERPER_API_KEY"] = key
                console.print(
                    "  [green]✓[/green] saved SERPER_API_KEY"
                )
            self._web_backend = "serper"
        elif choice == "2":
            self._web_backend = "duckduckgo"
        elif choice == "3":
            self._web_backend = None
        else:
            console.print(
                f"  [yellow]invalid choice {choice!r} — "
                "enter 1, 2, or 3[/yellow]"
            )
            return

        self._rebuild_agent()
        state = self._web_backend or "off"
        console.print(
            f"  [dim]web search: {state} — "
            "fresh conversation[/dim]"
        )

    # ---- /resume --------------------------------------------------------

    def _session_pointer_path(self) -> Path:
        """Where we stash the last-used session_id for this project.

        Lives under ``.loom/`` (same dir loom-code already uses for
        per-project state — notebook, memory db, repo map).
        One file per project, single line: the session_id ULID.
        """
        return self.project.root / ".loom" / "last_session.txt"

    def _handle_fork(self) -> None:
        """``/fork`` — branch the session HERE (pi-style session
        tree). The fork inherits the full history up to this point
        (episodes copied under a fresh session_id, so loomflow
        rehydrates them); the parent stays exactly as it was —
        ``/resume <its id>`` returns to it, ``/tree`` shows the
        graph. Use it to chase a tangent without polluting the main
        thread's context."""
        old = self.session_id
        new = new_id()
        db = self.project.root / LOOM_DIR / "memory.db"
        copied = _fork_episodes(db, old, new)
        self.session_id = new
        # Record the fork edge immediately (don't wait for the next
        # turn's pointer save — /tree should show it right away) and
        # mark it recorded so _save_session_pointer doesn't duplicate.
        try:
            import datetime as _dt

            record = {
                "session_id": new,
                "ts": _dt.datetime.now(_dt.UTC).isoformat(
                    timespec="seconds"
                ),
                "hint": f"fork of {old[:8]}",
                "model": str(self.model),
                "parent": old,
            }
            loom = self.project.root / LOOM_DIR
            loom.mkdir(exist_ok=True)
            with (loom / "sessions.jsonl").open(
                "a", encoding="utf-8"
            ) as fh:
                fh.write(json.dumps(record) + "\n")
            self._recorded_session_id = new
            (loom / "last_session.txt").write_text(
                new + "\n", encoding="utf-8"
            )
        except OSError:
            pass
        console.print(
            f"  [green]✓[/green] forked [cyan]{old[:8]}…[/cyan] → "
            f"[cyan]{new[:8]}…[/cyan] ({copied} turn"
            f"{'s' if copied != 1 else ''} inherited)"
        )
        console.print(
            "  [dim]you're on the fork now — the original is "
            f"untouched. /tree to see branches, /resume {old[:8]} "
            "to go back.[/dim]"
        )

    def _handle_tree(self) -> None:
        """``/tree`` — render the session graph for this project:
        every recorded session, forks indented under their parent,
        current position marked."""
        path = self.project.root / LOOM_DIR / "sessions.jsonl"
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(raw)
                except ValueError:
                    continue
                sid = rec.get("session_id")
                if sid and sid not in seen:
                    seen.add(sid)
                    records.append(rec)
        except OSError:
            pass
        # The live session may not be recorded yet (no turn taken) —
        # show it anyway so /tree is never confusingly empty.
        if self.session_id not in seen:
            records.append(
                {
                    "session_id": self.session_id,
                    "ts": "",
                    "hint": "(current — no turns yet)",
                }
            )
        lines = _render_session_tree(records, self.session_id)
        console.print()
        console.print("  [bold]session tree[/bold]")
        for line in lines:
            console.print(f"  {line}", markup=False, style="dim")
        console.print(
            "  [dim]/fork branches here · /resume <id-prefix> jumps "
            "to any node[/dim]"
        )

    def _save_session_pointer(self) -> None:
        """Write the current ``session_id`` to the project's
        ``.loom/last_session.txt``. Best-effort — a write failure
        is logged once but never blocks a turn (the file is a
        convenience; the agent's actual memory keys off
        ``session_id`` in loomflow's Memory which we don't touch
        here).

        Also appends one record per NEW session_id to
        ``.loom/sessions.jsonl`` — the history behind ``/resume pick``
        and ``--resume``. One line per session (first turn only), with
        a first-prompt hint so the picker is legible."""
        try:
            p = self._session_pointer_path()
            p.parent.mkdir(exist_ok=True)
            p.write_text(self.session_id + "\n", encoding="utf-8")
            if self.session_id != self._recorded_session_id:
                import datetime as _dt

                record = {
                    "session_id": self.session_id,
                    "ts": _dt.datetime.now(_dt.UTC).isoformat(
                        timespec="seconds"
                    ),
                    "hint": (self._last_prompt or "")[:80],
                    "model": str(self.model),
                }
                with (p.parent / "sessions.jsonl").open(
                    "a", encoding="utf-8"
                ) as fh:
                    fh.write(json.dumps(record) + "\n")
                self._recorded_session_id = self.session_id
        except OSError:
            # Silent failure: a read-only filesystem or perms
            # issue would otherwise spam the chat with the same
            # warning every turn.
            pass

    def _recent_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        """Recent sessions from ``.loom/sessions.jsonl``, newest first,
        current session excluded. Lenient on malformed lines."""
        path = self.project.root / LOOM_DIR / "sessions.jsonl"
        out: list[dict[str, Any]] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return out
        for raw in reversed(lines):
            try:
                rec = json.loads(raw)
            except ValueError:
                continue
            sid = rec.get("session_id")
            if not sid or sid == self.session_id:
                continue
            if any(s["session_id"] == sid for s in out):
                continue
            out.append(rec)
            if len(out) >= limit:
                break
        return out

    async def _pick_session(self) -> str | None:
        """Numbered menu over recent sessions; returns the chosen
        session_id or None on cancel / nothing to show."""
        sessions = self._recent_sessions()
        if not sessions:
            console.print(
                "  [yellow]no other recorded sessions for this "
                "project.[/yellow]"
            )
            return None
        console.print()
        console.print("  [bold]Recent sessions[/bold]")
        for i, s in enumerate(sessions, 1):
            ts = str(s.get("ts", ""))[:16].replace("T", " ")
            hint = s.get("hint") or "(no prompt recorded)"
            console.print(
                f"    [cyan]{i}[/cyan]. [dim]{ts}[/dim] "
                f"{s['session_id'][:8]}…  {hint}"
            )
        ans = await self._prompt_line("  Resume which? (number): ")
        if not ans or not ans.isdigit():
            console.print("  [dim]cancelled[/dim]")
            return None
        idx = int(ans) - 1
        if not 0 <= idx < len(sessions):
            console.print(f"  [yellow]no option {ans}[/yellow]")
            return None
        return str(sessions[idx]["session_id"])

    def _load_session_pointer(self) -> str | None:
        """Read the last saved session_id for this project, or
        ``None`` if no prior session has been recorded yet (first
        run on this project)."""
        try:
            p = self._session_pointer_path()
            if not p.exists():
                return None
            value = p.read_text(encoding="utf-8").strip()
            return value or None
        except OSError:
            return None

    def _handle_export(self) -> None:
        """``/export`` — write this session's turns to a markdown file
        under ``.loom/exports/`` and print the path. Uses the same
        (prompt, response) pairs the compactor sees, so it covers the
        current conversation thread."""
        if not self._compact_exchanges:
            console.print(
                "  [dim]nothing to export yet — no completed turns "
                "this session[/dim]"
            )
            return
        import datetime as _dt

        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = self.project.root / LOOM_DIR / "exports"
        out_path = out_dir / f"session-{ts}.md"
        lines = [
            "# loom-code session",
            "",
            f"- project: `{self.project.root}`",
            f"- model: `{self.model}`",
            f"- session: `{self.session_id}`",
            f"- exported: {ts}",
            "",
        ]
        for i, (prompt, reply) in enumerate(
            self._compact_exchanges, 1
        ):
            lines += [
                f"## Turn {i}",
                "",
                f"**user:** {prompt}",
                "",
                f"**loom-code:** {reply}",
                "",
            ]
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text("\n".join(lines), encoding="utf-8")
        except OSError as exc:
            console.print(f"  [red]export failed: {exc}[/red]")
            return
        console.print(
            f"  [green]✓[/green] exported "
            f"{len(self._compact_exchanges)} turn(s) → "
            f"[cyan]{out_path.relative_to(self.project.root)}[/cyan]"
        )

    async def _handle_resume(self, arg: str = "") -> None:
        """``/resume`` — pick up a prior session on this project.

        * no arg → the LAST session (the 95% case, unchanged).
        * ``pick`` / ``list`` → numbered menu of recent sessions
          (from ``.loom/sessions.jsonl``), choose one.
        * a session-id prefix → resume that session directly.

        loomflow's Memory keys episodes by ``(user_id, session_id)``;
        when the agent's next ``run()`` reuses the same session_id,
        loomflow rehydrates the prior turns into the conversation
        context for free. We don't need to do any rehydration here
        — just swap the id and let loomflow do its thing.

        Edge case: the saved session_id might be from a /clear
        boundary (i.e. the user explicitly told us to forget) or
        from a different model. We don't try to guard against
        either — /resume is a deliberate gesture the user owns.
        """
        arg = arg.strip()
        prior: str | None
        if arg in ("pick", "list"):
            prior = await self._pick_session()
            if prior is None:
                return
        elif arg:
            # A session-id prefix.
            matches = [
                s["session_id"]
                for s in self._recent_sessions()
                if s["session_id"].startswith(arg)
            ]
            if not matches:
                console.print(
                    f"  [yellow]no recorded session starts with "
                    f"{arg!r} — try /resume pick[/yellow]"
                )
                return
            prior = matches[0]
        else:
            prior = self._load_session_pointer()
        if prior is None:
            console.print(
                "  [yellow]no prior session recorded for this "
                "project — nothing to resume.[/yellow]"
            )
            console.print(
                "  [dim](sessions are saved per project after each "
                "turn — your first task here starts a fresh one.)"
                "[/dim]"
            )
            return
        if prior == self.session_id:
            console.print(
                "  [dim]you're already on the latest session "
                f"({prior[:8]}…) — nothing to resume.[/dim]"
            )
            return
        # Swap. Reset the compaction state so we don't blend the
        # newly-resumed session with whatever happened in this
        # REPL launch before /resume was called.
        old = self.session_id
        self.session_id = prior
        self._compact_tokens = 0
        self._compact_exchanges.clear()

        # Legacy data migration — loom-code pre-0.10.18 ran the
        # Router in ``per_route`` mode, so episodes were stored
        # under ``{prior}__route_simple`` / ``{prior}__route_complex``,
        # NOT under ``prior`` itself. Post-upgrade we run
        # ``conversation_scope='shared'`` which keys rehydration on
        # ``prior`` — so a /resume to a pre-upgrade session loses
        # all context unless we migrate.
        #
        # One-shot UPDATE in the sqlite db (loom-code hardcodes the
        # sqlite backend). Idempotent — a post-upgrade session has
        # nothing under the derived names. Episode_tool_transcripts
        # cascades via episode_id, so no separate migration needed.
        migrated = _migrate_legacy_per_route_episodes(
            self.project.root / LOOM_DIR / "memory.db", prior
        )
        if migrated:
            console.print(
                f"  [dim]migrated {migrated} legacy per-route "
                "episode(s) into the shared session for "
                "rehydration[/dim]"
            )
        # Repair transcripts written by loomflow < 0.10.30, whose
        # capture window leaked rehydrated prose (a prior answer +
        # the run's own prompt) into the tool transcript — that
        # prose gets spliced back by session_messages and shows up
        # as duplicated/misaligned turns in rehydration AND the
        # preview below. Idempotent; silent when nothing to fix.
        scrubbed = _scrub_prose_from_tool_transcripts(
            self.project.root / LOOM_DIR / "memory.db", prior
        )
        if scrubbed:
            console.print(
                f"  [dim]scrubbed {scrubbed} leaked prose message(s) "
                "from stored tool transcripts[/dim]"
            )

        console.print(
            f"  [green]✓[/green] resumed session [cyan]{prior[:8]}…"
            f"[/cyan] (was on {old[:8]}…)"
        )
        console.print(
            "  [dim]loomflow will rehydrate prior turns from "
            "memory on your next task.[/dim]"
        )

        # Surface the last N turns of the resumed session so the
        # user has visual context of WHAT they're resuming. Without
        # this, /resume is invisible — user has no way to confirm
        # the rehydration actually picked up real content vs an
        # empty session id, and no way to catch a wrong-session
        # mistake before they type the next prompt.
        await self._render_resumed_history_preview(prior)

    async def _render_resumed_history_preview(
        self, session_id: str
    ) -> None:
        """Fetch + render the last 5 turn groups from the resumed
        session so the user sees what they're inheriting. Silently
        no-ops when the memory backend doesn't expose
        ``session_messages`` (some custom backends don't) or the
        session is empty."""
        try:
            messages = await self.agent._memory.session_messages(
                session_id, user_id=_USER_ID, limit=100
            )
        except (AttributeError, TypeError):
            return
        if not messages:
            return
        turn_groups = _group_messages_into_turns(messages)
        if not turn_groups:
            return
        raw_count = len(turn_groups)
        # Collapse consecutive identical (user, assistant) pairs
        # into one row with a repeat count — without this, runs of
        # "user typed the same thing twice" or "stop-hook re-fired
        # the same prompt" produce visual noise in the preview.
        collapsed = _collapse_consecutive_duplicate_turns(
            turn_groups
        )
        recent = collapsed[-5:]
        skipped = raw_count - sum(r[3] for r in recent)
        console.print()
        title = (
            f"history (last {len(recent)} of {raw_count} "
            "turns — agent sees the full set)"
        )
        rule = "─" * max(0, 64 - len(title) - 4)
        console.print(f"  [dim]── {title} {rule}[/dim]")
        for user_prompt, assistant_text, n_tool_calls, repeats in recent:
            console.print()
            u = _truncate_one_line(user_prompt, 140)
            repeat_tag = f" [dim](×{repeats})[/dim]" if repeats > 1 else ""
            console.print(
                f"  [bold]user:[/bold] {u}{repeat_tag}"
            )
            a = _truncate_one_line(assistant_text, 200)
            if a:
                console.print(f"  [dim]loom:[/dim] {a}")
            else:
                console.print(
                    "  [dim]loom: (no text response)[/dim]"
                )
            if n_tool_calls:
                console.print(
                    f"        [dim]({n_tool_calls} tool call"
                    f"{'s' if n_tool_calls != 1 else ''})[/dim]"
                )
        console.print(f"  [dim]{'─' * 68}[/dim]")
        if skipped > 0:
            console.print(
                f"  [dim]+ {skipped} earlier turn(s) recovered "
                "(visible to the agent, not shown here)[/dim]"
            )


def _truncate_one_line(text: str, max_chars: int) -> str:
    """Collapse to one line + cap length. For the /resume history
    preview where multi-line messages would blow the layout."""
    if not text:
        return ""
    first = text.replace("\r", " ").strip()
    # Collapse all whitespace runs to a single space so multi-line
    # responses fit on one line cleanly.
    first = " ".join(first.split())
    if len(first) <= max_chars:
        return first
    return first[: max_chars - 1].rstrip() + "…"


def _collapse_consecutive_duplicate_turns(
    groups: list[tuple[str, str, int]],
) -> list[tuple[str, str, int, int]]:
    """Collapse runs of consecutive identical
    ``(user_prompt, assistant_text)`` turn groups into one entry
    annotated with a repeat count.

    Used by the /resume history preview to dedupe the visual when
    the user (or a prior framework version's stop-hook re-prompt)
    persisted the same exchange multiple times in a row. Three
    consecutive identical groups collapse to one ``(user, asst,
    n_tool, repeats=3)`` row; non-consecutive duplicates are kept
    as separate rows (different points in the conversation should
    show separately even if identical).

    ``n_tool`` from the FIRST occurrence is preserved — the
    assumption being that all collapsed copies had the same
    tool-call shape (they had identical assistant text, so
    almost certainly identical tools).
    """
    if not groups:
        return []
    out: list[tuple[str, str, int, int]] = []
    cur_user, cur_asst, cur_tools = groups[0]
    repeats = 1
    for user, asst, tools in groups[1:]:
        if user == cur_user and asst == cur_asst:
            repeats += 1
        else:
            out.append((cur_user, cur_asst, cur_tools, repeats))
            cur_user, cur_asst, cur_tools = user, asst, tools
            repeats = 1
    out.append((cur_user, cur_asst, cur_tools, repeats))
    return out


def _group_messages_into_turns(
    messages: list[Any],
) -> list[tuple[str, str, int]]:
    """Walk a rehydrated message list and group it into the
    natural ``(user_prompt, assistant_text, n_tool_calls)`` shape
    used by the /resume preview.

    Each USER message starts a new turn group; ASSISTANT messages
    contribute their text content + tool_call count to the
    currently-open group; TOOL result messages are folded into the
    current group's tool-call count too (they're the other half of
    a tool_call pair). SYSTEM messages are ignored — they're
    framework context, not conversation.

    Returns groups in source order (oldest first). Empty list for
    a message stream with no USER turns.
    """
    groups: list[tuple[str, str, int]] = []
    cur_user: str | None = None
    cur_assistant: list[str] = []
    cur_tool_calls = 0
    for m in messages:
        role = getattr(m, "role", None)
        # Role enum values are lowercase strings: 'user', 'assistant',
        # 'tool', 'system'. Some custom backends may pass plain strings.
        role_s = str(role).lower().split(".")[-1]
        content = str(getattr(m, "content", "") or "")
        if role_s == "user":
            # Close the previous group if any.
            if cur_user is not None:
                groups.append((
                    cur_user,
                    " ".join(cur_assistant).strip(),
                    cur_tool_calls,
                ))
            cur_user = content
            cur_assistant = []
            cur_tool_calls = 0
        elif role_s == "assistant":
            if content:
                cur_assistant.append(content)
            tool_calls = getattr(m, "tool_calls", None) or ()
            cur_tool_calls += len(tool_calls)
        elif role_s == "tool":
            # Tool result — counts as part of the open group's
            # tool activity. We don't double-count vs the
            # assistant's tool_calls list (which counted CALLS);
            # the tool message is the RESULT of one of those.
            # Skipping it avoids 2x-ing the displayed count.
            pass
        # SYSTEM messages: drop, not user-facing.
    # Close the final group.
    if cur_user is not None:
        groups.append((
            cur_user,
            " ".join(cur_assistant).strip(),
            cur_tool_calls,
        ))
    return groups


# Phrases a hallucinated "I'm done" turn uses. Matched against the
# agent's output when the turn made ZERO tool calls. Deliberately
# narrow — we want completion CLAIMS, not legitimate no-tool
# answers ("here's what X means"). Each pattern is "verb of
# completion + object of work".
_COMPLETION_CLAIM_RE = re.compile(
    r"\b("
    r"all (the )?(detected |previously )?(issues|problems|"
    r"bugs|fixes)\b.{0,40}\b(fixed|addressed|resolved|done)"
    r"|already been fixed"
    r"|have been fixed"
    r"|were fixed"
    r"|no (remaining |outstanding )?(issues|problems|blockers)"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)


def _looks_like_completion_claim(text: str) -> bool:
    """True if ``text`` reads like "I finished the work" — used to
    detect hallucinated completion claims on zero-tool-call turns.
    Narrow on purpose: a normal answer that happens to say 'fixed'
    once shouldn't trip it, but 'all the detected issues have been
    fixed' should."""
    if not text:
        return False
    return _COMPLETION_CLAIM_RE.search(text) is not None


def _delete_last_episode(
    db_path: Path, *, session_id: str, user_id: str
) -> bool:
    """Delete the most-recently-persisted episode for
    ``(user_id, session_id)``. Used by the anti-poison gate to
    remove a just-written no-tool-call completion claim before it
    pollutes recall.

    Direct sqlite (loom-code hardcodes the sqlite backend) because
    the Memory protocol's ``forget`` is coarse (by user/session/
    time, not 'the single most-recent row'). Returns True if a row
    was deleted. Best-effort — swallows errors so a gate failure
    never breaks the turn.
    """
    if not db_path.is_file():
        return False
    try:
        import sqlite3
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.cursor()
            # Find the most-recent episode id for this scope, then
            # delete by id (episode_tool_transcripts cascades via
            # the episode_id FK).
            cur.execute(
                "SELECT id FROM episodes "
                "WHERE user_id = ? AND session_id = ? "
                "ORDER BY occurred_at DESC LIMIT 1",
                (user_id, session_id),
            )
            row = cur.fetchone()
            if row is None:
                return False
            cur.execute(
                "DELETE FROM episodes WHERE id = ?", (row[0],)
            )
            conn.commit()
            return (cur.rowcount or 0) > 0
    except (sqlite3.Error, OSError):
        return False


def _migrate_legacy_per_route_episodes(
    db_path: Path, parent_session_id: str
) -> int:
    """Re-key any legacy per-route episodes into the parent
    session_id so ``conversation_scope='shared'`` rehydration sees
    them.

    Pre-0.10.18 loom-code ran the Router in default ``per_route``
    mode, persisting episodes under ``{parent}__route_simple`` and
    ``{parent}__route_complex``. The new shared-mode lookup keys on
    ``parent`` alone, so /resume'd pre-upgrade sessions had no
    visible history. This UPDATE rewrites the session_id column for
    any matching legacy rows. Idempotent — re-running on a
    post-upgrade session is a no-op.

    Returns the number of rows migrated. Silently no-ops when the
    db file is absent or unreadable — failure here must NEVER
    block /resume.

    Why direct sqlite (not via the Memory protocol): the Memory
    protocol exposes ``remember(Episode)`` and ``session_messages``
    but no primitive for ``rekey-session``. Adding one to the
    framework just to satisfy this one-shot loom-code migration
    isn't worth the surface. We know the backend is sqlite (the
    REPL hardcodes it) and the column name is stable.
    """
    if not db_path.is_file():
        return 0
    legacy_simple = f"{parent_session_id}__route_simple"
    legacy_complex = f"{parent_session_id}__route_complex"
    try:
        import sqlite3
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE episodes SET session_id = ? "
                "WHERE session_id IN (?, ?)",
                (parent_session_id, legacy_simple, legacy_complex),
            )
            migrated = cur.rowcount or 0
            conn.commit()
            return int(migrated)
    except (sqlite3.Error, OSError):
        return 0


def _save_goal_state(
    loom_dir: Path,
    *,
    task: str,
    condition: str,
    session_id: str,
    model: str,
) -> None:
    """Persist the active goal to ``.loom/goal.json`` so it survives
    restarts (codex-parity: /goal is a durable, multi-day workflow,
    not a per-process one). Best-effort — persistence failing must
    never block the goal itself."""
    try:
        import datetime as _dt

        loom_dir.mkdir(exist_ok=True)
        (loom_dir / "goal.json").write_text(
            json.dumps(
                {
                    "task": task,
                    "condition": condition,
                    "session_id": session_id,
                    "model": model,
                    "status": "active",
                    "started_at": _dt.datetime.now(
                        _dt.UTC
                    ).isoformat(timespec="seconds"),
                }
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def _load_goal_state(loom_dir: Path) -> dict[str, Any] | None:
    """The persisted goal, or None when absent/done/corrupt."""
    try:
        data = json.loads(
            (loom_dir / "goal.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("status") != "active":
        return None
    if not str(data.get("task") or "").strip():
        return None
    return data


def _clear_goal_state(loom_dir: Path) -> None:
    """Mark the goal done (file removed). Best-effort."""
    try:
        (loom_dir / "goal.json").unlink(missing_ok=True)
    except OSError:
        pass


def _fork_episodes(
    db_path: Path, old_session_id: str, new_session_id: str
) -> int:
    """Copy every episode of ``old_session_id`` (plus tool
    transcripts) under ``new_session_id`` — the storage half of
    ``/fork``. loomflow rehydrates by session_id, so after the copy
    the fork "remembers" everything up to the fork point while the
    parent stays untouched (pi-style session branching, on the same
    sqlite the rest of loom-code already manages directly).

    Returns episodes copied; 0 on any failure — a failed fork must
    never block the REPL (the fork still works go-forward, it just
    starts blank)."""
    if not db_path.is_file():
        return 0
    try:
        import sqlite3
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.cursor()
            rows = cur.execute(
                "SELECT id, user_id, occurred_at, input, output, "
                "embedding FROM episodes WHERE session_id = ? "
                "ORDER BY occurred_at",
                (old_session_id,),
            ).fetchall()
            copied = 0
            for eid, uid, at, inp, outp, emb in rows:
                new_eid = f"{eid}_fork_{new_session_id[-8:]}"
                cur.execute(
                    "INSERT OR IGNORE INTO episodes "
                    "(id, session_id, user_id, occurred_at, input, "
                    "output, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        new_eid,
                        new_session_id,
                        uid,
                        at,
                        inp,
                        outp,
                        emb,
                    ),
                )
                copied += cur.rowcount or 0
                cur.execute(
                    "INSERT OR IGNORE INTO episode_tool_transcripts "
                    "(episode_id, sequence, message_json) "
                    "SELECT ?, sequence, message_json "
                    "FROM episode_tool_transcripts "
                    "WHERE episode_id = ?",
                    (new_eid, eid),
                )
            conn.commit()
            return copied
    except (sqlite3.Error, OSError):
        return 0


def _render_session_tree(
    records: list[dict[str, Any]], current_session_id: str
) -> list[str]:
    """ASCII tree over the session records (each ``{session_id, ts,
    hint, parent?}``). Roots are sessions without a recorded parent
    (or whose parent predates the log); children indent under their
    parent, oldest first. The current session is marked. Pure —
    testable without a REPL."""
    by_id = {r["session_id"]: r for r in records if r.get("session_id")}
    children: dict[str, list[str]] = {}
    roots: list[str] = []
    for r in records:
        sid = r.get("session_id")
        if not sid:
            continue
        parent = r.get("parent")
        if parent and parent in by_id:
            children.setdefault(parent, []).append(sid)
        else:
            roots.append(sid)

    lines: list[str] = []

    def _walk(sid: str, depth: int) -> None:
        r = by_id[sid]
        marker = "●" if sid == current_session_id else "○"
        hint = _truncate_one_line(str(r.get("hint") or ""), 60)
        ts = str(r.get("ts") or "")[:16].replace("T", " ")
        indent = "  " * depth + ("└─ " if depth else "")
        you = "  ← you are here" if sid == current_session_id else ""
        lines.append(
            f"{indent}{marker} {sid[:8]}…  {ts}  {hint}{you}"
        )
        for child in children.get(sid, []):
            _walk(child, depth + 1)

    for root in roots:
        _walk(root, 0)
    return lines


def _scrub_prose_from_tool_transcripts(
    db_path: Path, session_id: str
) -> int:
    """Delete NON-tool messages from a session's persisted tool
    transcripts.

    loomflow < 0.10.30 built the transcript by EXCLUDING {system,
    first USER, last ASSISTANT-text} and keeping the rest — on a
    resumed session the "first USER" was a prior turn's rehydrated
    prompt, so a prior answer + the run's own input leaked into the
    transcript. ``session_messages`` splices the transcript between
    input/output, so consumers (rehydration AND the /resume preview)
    saw duplicated, misaligned turns.

    A transcript row is legitimate iff it is tool work:
    ``role == "tool"`` or ``role == "assistant"`` with a non-empty
    ``tool_calls``. Everything else is leaked prose — delete it.
    Idempotent; returns rows deleted. Same direct-sqlite rationale
    as :func:`_migrate_legacy_per_route_episodes`, and failure here
    must NEVER block /resume.
    """
    if not db_path.is_file():
        return 0
    try:
        import sqlite3
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.cursor()
            rows = cur.execute(
                "SELECT t.episode_id, t.sequence, t.message_json "
                "FROM episode_tool_transcripts t "
                "JOIN episodes e ON e.id = t.episode_id "
                "WHERE e.session_id = ?",
                (session_id,),
            ).fetchall()
            doomed: list[tuple[str, int]] = []
            for episode_id, sequence, message_json in rows:
                try:
                    msg = json.loads(message_json)
                except (json.JSONDecodeError, ValueError):
                    continue  # unparseable — leave it alone
                role = str(msg.get("role", "")).lower()
                is_tool_work = role == "tool" or (
                    role == "assistant" and msg.get("tool_calls")
                )
                if not is_tool_work:
                    doomed.append((episode_id, sequence))
            for episode_id, sequence in doomed:
                cur.execute(
                    "DELETE FROM episode_tool_transcripts "
                    "WHERE episode_id = ? AND sequence = ?",
                    (episode_id, sequence),
                )
            conn.commit()
            return len(doomed)
    except (sqlite3.Error, OSError):
        return 0


async def run_repl(
    project: Project,
    model: str,
    *,
    sandbox: bool = False,
    sandbox_allow_network: bool = False,
    resume: str | None = None,
) -> int:
    """Entry point for the interactive REPL — construct the Repl and
    run its loop until the user exits.

    ``resume`` maps the CLI flags onto the /resume machinery before
    the first prompt: ``"last"`` (--continue) rejoins the most recent
    session, ``"pick"`` (--resume) shows the session picker."""
    repl = Repl(
        project,
        model,
        sandbox=sandbox,
        sandbox_allow_network=sandbox_allow_network,
        startup_resume=resume,
    )
    return await repl.run()
