"""Canned :class:`MCPServerSpec` factories for the MCP servers
loom-code knows about out of the box.

Each factory returns a spec the caller hands to
:class:`loomflow.mcp.MCPRegistry`, then to
:func:`loom_code.agent.build_agent` via the ``mcp_registry=`` kwarg.
The caller owns the registry lifecycle
(``async with registry: ...``) — these factories just produce the
declarative spec, no side effects.

Currently shipped:

* :func:`default_graphify_spec` — graphify's stdio MCP server
  (``graphify --mcp``), output dir nested under ``.loom/graphify/``
  to match loom-code's per-project state convention.

Adding a new server: write another ``<name>_spec`` factory here,
then advertise in the README. Keep the surface a flat function so
TOML configs (``.loom/config.toml``) can map onto it 1:1 when the
config layer lands.
"""

from __future__ import annotations

from pathlib import Path

from loomflow.mcp import MCPServerSpec

_GRAPHIFY_OUT_SUBDIR = ".loom/graphify"


def default_graphify_spec(project_root: Path | str) -> MCPServerSpec:
    """Spec for the graphify stdio MCP server.

    Spawns ``graphify <project_root> --mcp --out <.loom/graphify>``
    so the server reads/writes graph state inside the per-project
    ``.loom/`` directory — same convention as the sqlite memory and
    workspace notebook. Inherits the project's gitignore in one
    line (everyone already excludes ``.loom/``).

    Prerequisites: ``pip install graphifyy`` (or the equivalent
    install per graphify's README). The MCP server is only spawned
    when the registry's context manager is entered — constructing
    the spec is side-effect-free, so this function is safe to call
    even when graphify isn't installed.

    The returned spec exposes graphify's tools — typically
    ``graphify_query`` / ``graphify_path`` / ``graphify_explain`` —
    in whatever Agent's ``tools=`` the registry is wired into. In
    loom-code that's the supervisor coordinator (COMPLEX route);
    SIMPLE-mode coder doesn't see MCP tools yet (pending
    ExtendedToolHost composition).

    Example::

        from loom_code.mcp_specs import default_graphify_spec
        from loomflow.mcp import MCPRegistry

        registry = MCPRegistry([default_graphify_spec(project.root)])
        async with registry:
            coordinator, workspace = build_agent(
                project, mcp_registry=registry
            )
            await coordinator.run("what connects auth to billing?")
    """
    root = Path(project_root).resolve()
    out_dir = root / _GRAPHIFY_OUT_SUBDIR
    return MCPServerSpec.stdio(
        name="graphify",
        command="graphify",
        args=(str(root), "--mcp", "--out", str(out_dir)),
        description=(
            "Knowledge-graph queries over the project — code, docs, "
            "papers, images extracted into a NetworkX graph. Use "
            "for structural questions (what connects X to Y, paths "
            "between concepts, god nodes) that grep can't answer."
        ),
    )
