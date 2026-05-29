"""User-extensibility discovery for loom-code (the ``.loom`` folder).

loom-code mirrors Claude Code's ``.claude`` folder: users drop in
their own skills, subagents, and hooks and loom-code picks them up.
There are THREE layers, lowest priority first:

    bundled   ``loom_code/skills/``                  (shipped)
    user      ``~/.loom-code/{skills,agents}/``       (your global config)
    project   ``<repo>/.loom/{skills,agents}/``       (this repo only)

The user + project layers also carry a ``settings.toml`` declaring
hooks.

:func:`discover` scans the user + project layers and returns one
:class:`Extensions` bundle. ``build_agent`` calls it once and threads
the three results into the existing wiring:

* **skills** — appended to the bundled skill list passed to every
  agent. The framework's ``SkillRegistry`` does last-source-wins by
  name, so we append user *then* project (project wins on collision).
* **agents** — parsed into :class:`AgentSpec` and merged into the
  ``Team.supervisor`` worker roster (project wins on name collision).
* **hooks** — parsed into :class:`HookSpec`. Tool-lifecycle hooks
  (PreToolUse/PostToolUse/Stop) become framework hooks on the
  tool-executing agents; REPL-lifecycle hooks (UserPromptSubmit/
  SessionStart/SessionEnd) the REPL fires itself. Hooks are
  **additive** across scopes — a project cannot disable your personal
  hooks.

Parsing is dependency-free on purpose: loom-code does not depend on
pyyaml, so frontmatter is split by a tiny in-house parser
(:func:`_parse_frontmatter`) that handles the handful of fields a
subagent declares, and ``settings.toml`` is read with stdlib
``tomllib``.

This module is pure discovery + parsing. It does NOT execute hooks,
construct Agents, or prompt for trust — those live with the code that
consumes the specs (``workers.py`` for agents, ``hooks.py`` for the
shim + trust gate).
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only import: the annotation is a string under
    # ``from __future__ import annotations``, so this never loads the
    # ``mcp`` extra at runtime. The actual spec is constructed lazily in
    # ``_discover_mcp`` where the import is genuinely needed.
    from loomflow.mcp import MCPServerSpec

# The two scope roots. ``.loom`` (project) matches ``agent.LOOM_DIR``;
# ``.loom-code`` (user, with the hyphen) matches
# ``credentials._CREDENTIALS_DIR`` — the hyphen is the deliberate
# "this is GLOBAL config, not a project" differentiator.
PROJECT_DIRNAME = ".loom"
USER_DIRNAME = ".loom-code"

# Hook events loom-code recognises. Tool-lifecycle events map onto the
# framework's HookRegistry / stop_hooks; REPL-lifecycle events the REPL
# fires directly (the framework has no hook point for them).
TOOL_HOOK_EVENTS = frozenset({"PreToolUse", "PostToolUse"})
STOP_HOOK_EVENTS = frozenset({"Stop"})
REPL_HOOK_EVENTS = frozenset(
    {"UserPromptSubmit", "SessionStart", "SessionEnd"}
)
KNOWN_HOOK_EVENTS = TOOL_HOOK_EVENTS | STOP_HOOK_EVENTS | REPL_HOOK_EVENTS


@dataclass(frozen=True)
class AgentSpec:
    """A user-authored subagent parsed from ``<scope>/agents/<name>.md``.

    The markdown body is the subagent's system prompt; the frontmatter
    carries the routing contract (``name`` + ``description`` — the
    supervisor delegates by description) and optional ``model`` /
    ``tools`` overrides.

    ``tools`` is the list of builtin tool names the subagent may use
    (``read``/``write``/``edit``/``multi_edit``/``grep``/``find``/``ls``/
    ``bash``/``web_fetch``). Empty means "unspecified" — the wiring
    applies a read-only default rather than handing a stranger's spec
    write access implicitly.
    """

    name: str
    description: str
    system_prompt: str
    model: str | None = None
    tools: tuple[str, ...] = ()
    source: str = "project"  # "user" | "project"
    path: Path | None = None


@dataclass(frozen=True)
class HookSpec:
    """A hook parsed from a ``[[hooks]]`` entry in ``settings.toml``.

    ``event`` is one of :data:`KNOWN_HOOK_EVENTS`. ``matcher`` is a
    tool-name pattern (only meaningful for tool-lifecycle events): the
    literal ``"*"`` / ``""`` matches all, a pipe-separated list like
    ``"bash|edit"`` matches any of those tools, anything else is a
    regex. ``command`` is the shell command run with the event's JSON
    on stdin. ``source`` records which scope declared it — user-scope
    hooks are trusted; project-scope hooks are trust-gated.
    """

    event: str
    command: str
    matcher: str = "*"
    timeout: float = 60.0
    source: str = "project"  # "user" | "project"


@dataclass(frozen=True)
class McpEntry:
    """An MCP server declared in a ``[[mcp]]`` block of ``settings.toml``.

    ``source`` is "user" or "project"; the trust gate keys off it exactly
    as it does for :class:`HookSpec` — a project-declared server from an
    untrusted repo is dropped, a user-scope server is your own config and
    always kept (connecting an MCP server runs external code / hits an
    external endpoint). ``spec`` is the framework's ``MCPServerSpec``,
    built lazily in :func:`_discover_mcp` so importing this module never
    requires the ``mcp`` extra.
    """

    source: str
    spec: MCPServerSpec


@dataclass
class Extensions:
    """Everything discovered across the user + project layers.

    ``skill_paths`` are individual skill *directories* (each holds a
    ``SKILL.md``), ordered user-then-project so the framework's
    last-source-wins resolution gives project skills priority. The
    caller prepends the bundled skills.
    """

    skill_paths: list[Path] = field(default_factory=list)
    agent_specs: list[AgentSpec] = field(default_factory=list)
    hook_specs: list[HookSpec] = field(default_factory=list)
    # MCP servers declared in settings.toml [[mcp]] blocks, each tagged
    # with its source ("user" | "project") so the trust gate can drop
    # project-declared servers from an untrusted repo — same posture as
    # project hooks (connecting an MCP server runs external code / hits
    # an external endpoint, so a cloned repo must not auto-connect one).
    mcp_specs: list[McpEntry] = field(default_factory=list)

    def has_any(self) -> bool:
        return bool(
            self.skill_paths
            or self.agent_specs
            or self.hook_specs
            or self.mcp_specs
        )


def safe_role_name(name: str) -> str:
    """Map a subagent's authored ``name`` to a valid worker-role id.

    loomflow's worker registry requires role names to be Python
    identifiers (no hyphens/spaces), but Claude-Code-style subagent
    names are lowercase-with-hyphens. We translate any run of
    non-identifier characters to a single underscore so
    ``security-auditor.md`` becomes the delegate role
    ``security_auditor``. A name that would start with a digit is
    prefixed; an empty result falls back to ``"subagent"``."""
    safe = re.sub(r"\W+", "_", name.strip()).strip("_")
    if not safe:
        return "subagent"
    if safe[0].isdigit():
        safe = f"a_{safe}"
    return safe


def discover(
    project_root: Path,
    *,
    user_dir: Path | None = None,
) -> Extensions:
    """Scan the user + project layers and return the merged bundle.

    ``project_root`` is the repo root (``<root>/.loom/`` is scanned).
    ``user_dir`` overrides the user scope root (``~/.loom-code/`` by
    default) — tests pass a tmp dir here.

    Merge rules differ by type:

    * skills — user dirs then project dirs (caller prepends bundled);
      collisions resolved later by the framework (project wins).
    * agents — project overrides user on duplicate ``name``.
    * hooks — additive; every scope's hooks are kept (a project must
      not be able to silently drop your personal hooks).
    """
    user_base = (
        user_dir if user_dir is not None else (Path.home() / USER_DIRNAME)
    )
    project_base = project_root / PROJECT_DIRNAME

    ext = Extensions()
    ext.skill_paths = _discover_skill_dirs(user_base) + _discover_skill_dirs(
        project_base
    )

    merged: dict[str, AgentSpec] = {}
    for spec in _discover_agents(user_base, "user"):
        merged[spec.name] = spec
    for spec in _discover_agents(project_base, "project"):
        merged[spec.name] = spec  # project wins
    ext.agent_specs = list(merged.values())

    ext.hook_specs = _discover_hooks(user_base, "user") + _discover_hooks(
        project_base, "project"
    )
    # MCP servers — additive across scopes like hooks. A project's
    # [[mcp]] servers ride the same trust gate (see loom_code.trust).
    ext.mcp_specs = _discover_mcp(user_base, "user") + _discover_mcp(
        project_base, "project"
    )
    return ext


# ---- skills ---------------------------------------------------------


def _discover_skill_dirs(base: Path) -> list[Path]:
    """Return each ``<base>/skills/<name>/`` dir that has a SKILL.md.

    Same shape as ``agent._bundled_skill_paths`` so the result drops
    straight into the ``skills=`` list every agent builder already
    accepts."""
    skills_dir = base / "skills"
    if not skills_dir.is_dir():
        return []
    out: list[Path] = []
    for entry in sorted(skills_dir.iterdir()):
        if entry.is_dir() and (entry / "SKILL.md").is_file():
            out.append(entry)
    return out


# ---- agents ---------------------------------------------------------


def _discover_agents(base: Path, source: str) -> list[AgentSpec]:
    """Parse every ``<base>/agents/*.md`` into an :class:`AgentSpec`.

    A file missing either ``name`` or ``description`` frontmatter is
    skipped — those two fields are the delegation contract (the
    supervisor can't route to an agent it can't describe). Unreadable
    or malformed files are skipped rather than aborting discovery, so
    one bad file doesn't break the whole session."""
    agents_dir = base / "agents"
    if not agents_dir.is_dir():
        return []
    out: list[AgentSpec] = []
    for path in sorted(agents_dir.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        fm, body = _parse_frontmatter(text)
        name = str(fm.get("name") or path.stem).strip()
        description = str(fm.get("description") or "").strip()
        if not name or not description:
            continue
        model_raw = fm.get("model")
        model = str(model_raw).strip() if model_raw else None
        tools = _normalize_tools(fm.get("tools", ()))
        out.append(
            AgentSpec(
                name=name,
                description=description,
                system_prompt=body,
                model=model,
                tools=tools,
                source=source,
                path=path,
            )
        )
    return out


def _normalize_tools(value: object) -> tuple[str, ...]:
    """Coerce a ``tools`` frontmatter value into a tuple of names.

    Accepts a YAML-ish list (already split by :func:`_parse_frontmatter`)
    or a single string like ``"read, edit, bash"`` / ``"read edit"``."""
    if isinstance(value, (list, tuple)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    if isinstance(value, str):
        parts = re.split(r"[,\s]+", value.strip())
        return tuple(p for p in parts if p)
    return ()


# ---- hooks ----------------------------------------------------------


def _discover_hooks(base: Path, source: str) -> list[HookSpec]:
    """Parse ``[[hooks]]`` entries from ``<base>/settings.toml``.

    Expected shape::

        [[hooks]]
        event = "PreToolUse"
        matcher = "bash"            # optional, defaults to "*"
        command = "./scripts/check.sh"
        timeout = 30                # optional, seconds

    Entries with an unknown ``event`` or a missing ``event``/``command``
    are skipped. A malformed TOML file is skipped wholesale (returns
    ``[]``) rather than aborting the session."""
    settings = base / "settings.toml"
    if not settings.is_file():
        return []
    try:
        data = tomllib.loads(settings.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []
    raw = data.get("hooks")
    if not isinstance(raw, list):
        return []
    out: list[HookSpec] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        event = str(entry.get("event", "")).strip()
        command = str(entry.get("command", "")).strip()
        if event not in KNOWN_HOOK_EVENTS or not command:
            continue
        matcher = str(entry.get("matcher", "*")).strip() or "*"
        try:
            timeout = float(entry.get("timeout", 60.0))
        except (TypeError, ValueError):
            timeout = 60.0
        out.append(
            HookSpec(
                event=event,
                command=command,
                matcher=matcher,
                timeout=timeout,
                source=source,
            )
        )
    return out


def _discover_mcp(base: Path, source: str) -> list[McpEntry]:
    """Parse ``[[mcp]]`` blocks from ``<base>/settings.toml`` into
    :class:`McpEntry` (source-tagged :class:`MCPServerSpec`).

    Each block declares one MCP server. Recognised keys::

        [[mcp]]
        name = "linear"            # required, unique
        transport = "stdio"        # "stdio" (default) or "http"
        command = "npx"            # stdio: the server binary
        args = ["-y", "linear-mcp"]
        env = { LINEAR_API_KEY = "..." }
        # or, for http:
        # transport = "http"
        # url = "https://mcp.example.com"
        # headers = { Authorization = "Bearer ..." }

    Bad entries are skipped (missing name, or stdio without a command /
    http without a url) rather than aborting the session — matches the
    lenient, never-crash posture of :func:`_discover_hooks`. Returns
    ``[]`` when the ``mcp`` extra isn't installed (the lazy import
    fails) so loom-code runs fine without it.
    """
    settings = base / "settings.toml"
    if not settings.is_file():
        return []
    try:
        data = tomllib.loads(settings.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []
    raw = data.get("mcp")
    if not isinstance(raw, list):
        return []
    try:
        from loomflow.mcp import MCPServerSpec
    except ImportError:
        # ``mcp`` extra not installed — silently skip MCP discovery.
        return []
    out: list[McpEntry] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        transport = str(entry.get("transport", "stdio")).strip() or "stdio"
        command = str(entry.get("command", "")).strip() or None
        url = str(entry.get("url", "")).strip() or None
        # Skip specs that can't possibly connect — same "bad entry is
        # dropped, not fatal" rule as hooks.
        if transport == "stdio" and not command:
            continue
        if transport == "http" and not url:
            continue
        # MCPServerSpec is a frozen dataclass with hashable (tuple)
        # fields, so coerce the TOML list/dict into tuples here.
        args_raw = entry.get("args", [])
        args = (
            tuple(str(a) for a in args_raw)
            if isinstance(args_raw, list)
            else ()
        )
        env_raw = entry.get("env", {})
        env = (
            tuple((str(k), str(v)) for k, v in env_raw.items())
            if isinstance(env_raw, dict)
            else ()
        )
        headers_raw = entry.get("headers", {})
        headers = (
            tuple((str(k), str(v)) for k, v in headers_raw.items())
            if isinstance(headers_raw, dict)
            else ()
        )
        description = str(entry.get("description", "")).strip()
        spec = MCPServerSpec(
            name=name,
            transport=transport,  # type: ignore[arg-type]
            command=command,
            args=args,
            env=env,
            url=url,
            headers=headers,
            description=description,
        )
        out.append(McpEntry(source=source, spec=spec))
    return out


# ---- frontmatter ----------------------------------------------------

_FENCE_RE = re.compile(
    r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n(.*))?\Z", re.DOTALL
)


def _parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Split ``---\\n<frontmatter>\\n---\\n<body>`` into ``(fields, body)``.

    A deliberately tiny parser — loom-code doesn't depend on pyyaml and
    subagent frontmatter only needs ``key: value`` scalars plus a list
    value for ``tools``. Supported list forms::

        tools: [read, edit]        # flow
        tools:                     # block
          - read
          - edit

    Scalars are returned as strings (surrounding quotes stripped);
    flow/block lists are returned as ``list[str]``. Comma-splitting of
    bare scalars is intentionally NOT done here — descriptions contain
    commas — so ``tools: read, edit`` arrives as the string
    ``"read, edit"`` and :func:`_normalize_tools` splits it. Returns
    ``({}, text)`` when there's no frontmatter fence."""
    m = _FENCE_RE.match(text)
    if not m:
        return {}, text.strip()
    raw = m.group(1)
    body = (m.group(2) or "").strip()

    fields: dict[str, object] = {}
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in line:
            i += 1
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if not val:
            # Possible block list on the following indented "- " lines.
            items: list[str] = []
            j = i + 1
            while j < len(lines) and lines[j].lstrip().startswith("- "):
                items.append(_strip_quotes(lines[j].lstrip()[2:].strip()))
                j += 1
            if items:
                fields[key] = items
                i = j
                continue
            fields[key] = ""
            i += 1
            continue
        fields[key] = _coerce_value(val)
        i += 1
    return fields, body


def _coerce_value(val: str) -> object:
    """Turn a frontmatter scalar into a flow list (``[a, b]``) or a
    quote-stripped string."""
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1]
        return [
            _strip_quotes(p.strip()) for p in inner.split(",") if p.strip()
        ]
    return _strip_quotes(val)


def _strip_quotes(val: str) -> str:
    if len(val) >= 2 and val[0] == val[-1] and val[0] in {'"', "'"}:
        return val[1:-1]
    return val
