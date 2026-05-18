"""Mode B Python tools for the ``graphify`` skill.

Wraps graphify's public Python primitives (``collect_files`` /
``extract`` / ``build_from_json`` / ``cluster`` / ``to_json``) into
``@tool``-decorated functions the agent calls directly. No
subprocess, no MCP server — just in-process Python.

Why not the standalone ``graphify`` CLI: it doesn't have a
``graphify <path>`` subcommand. The CLI is a SKILL installer
(``graphify install`` copies ``SKILL.md`` to
``~/.claude/skills/graphify/`` for Claude Code to find). The
actual extraction pipeline lives in the Python modules and is
intended to be orchestrated by the host AI tool — Claude Code
runs the multi-step skill flow; loom-code does the same here
via its own skill machinery, scoped to AST-only extraction
(code files) for predictable in-process behavior.

Multi-modal extraction (docs / papers / images) needs the full
skill-driven semantic pass with parallel subagents — that's a
follow-up; AST-only covers the 90% case for loom-code's use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
from loomflow import tool

_GRAPHIFY_OUT_SUBDIR = ".loom/graphify"
_GRAPH_FILENAME = "graph.json"

# Extensions graphify has a tree-sitter extractor for. Keep in sync
# with ``graphify.extract``'s per-language dispatch table — anything
# outside this set the extractor would silently skip, so we drop it
# before paying the dict-lookup + Path stat cost. Source of truth is
# the ``extract_<lang>`` functions in ``graphify/extract.py``.
_GRAPHIFY_SUPPORTED_SUFFIXES = frozenset({
    ".astro", ".sh", ".bash", ".blade.php", ".c", ".h",
    ".cpp", ".cc", ".cxx", ".hpp", ".cs", ".dart", ".pas",
    ".dfm", ".ex", ".exs", ".f", ".f90", ".f95", ".for",
    ".go", ".groovy", ".java", ".js", ".jsx", ".mjs", ".cjs",
    ".json", ".jl", ".kt", ".kts", ".lpr", ".lpk", ".lua",
    ".md", ".markdown", ".m", ".mm", ".lpi", ".lps",
    ".php", ".ps1", ".psm1", ".py", ".pyi", ".rb",
    ".rs", ".scala", ".sc", ".sql", ".svelte", ".swift",
    ".ts", ".tsx", ".v", ".sv", ".zig",
})


async def _git_ls_files(project_root: Path) -> list[Path] | None:
    """Fast path for source-file discovery: ask git for the tracked
    file list. Returns ``None`` (caller falls back to
    ``graphify.collect_files``) on any failure — not a git repo, no
    git binary, timeout, anything. Trust git's ignore handling
    (``.gitignore`` + ``.git/info/exclude``) — it already knows
    ``.venv`` / ``node_modules`` / build artifacts are noise.

    Why this matters: ``graphify.collect_files`` walks every file
    under the root (including everything in a 17k-file ``.venv``)
    then filters to known extensions. On loomflow-ide that walk is
    6+ seconds; git ls-files returns the same 46-file set in ~10ms.
    """
    try:
        result = await anyio.run_process(
            ["git", "-C", str(project_root), "ls-files"],
            check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    out: list[Path] = []
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        if not line:
            continue
        # git ls-files emits paths relative to the repo root. Resolve
        # against project_root, not cwd, so cwd shifts can't change
        # which files we see. Skip directory entries (submodules
        # appear as bare names with no extension).
        full = project_root / line
        if (
            full.is_file()
            and full.suffix.lower() in _GRAPHIFY_SUPPORTED_SUFFIXES
        ):
            out.append(full)
    return out


def _graph_path(project_root: Path | str) -> Path:
    """Where the graph file lives for a given project root.
    Single source of truth so build + query agree."""
    return (
        Path(project_root).resolve()
        / _GRAPHIFY_OUT_SUBDIR
        / _GRAPH_FILENAME
    )


def _load_graph(project_root: Path | str) -> Any:
    """Load the persisted graph, or raise a tool-friendly error
    with the build hint baked in."""
    path = _graph_path(project_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"No graph at {path}. Run `graphify__build()` first "
            "to extract + persist the knowledge graph for this "
            "project."
        )
    from networkx.readwrite import json_graph
    data = json.loads(path.read_text())
    return json_graph.node_link_graph(data, edges="links")


@dataclass(frozen=True)
class GraphifyBuildResult:
    """Structured outcome of one ``graphify_build_impl`` run.

    Used by callers that need the numbers (``/loominit`` for the
    LOOM.md ``## Knowledge Graph`` section, ``_post_commit`` for a
    log line). The ``@tool`` wrapper formats the same fields into
    the string the agent sees."""

    graph_path: Path
    project_root: Path
    n_nodes: int
    n_edges: int
    n_files: int
    n_communities: int
    source: str  # "git ls-files" or "graphify.collect_files (no git index)"
    skipped_reason: str | None = None  # set when build was a no-op


async def graphify_build_impl(path: str | Path = ".") -> GraphifyBuildResult:
    """Build + persist the project's knowledge graph. Shared core
    that both the ``@tool`` wrapper (below) and the loom-code REPL's
    ``/loominit`` + post-commit refresh call directly.

    Steps: source-file discovery (git fast path → graphify fallback)
    → tree-sitter extraction → NetworkX graph → Leiden clustering →
    JSON persistence at ``<path>/.loom/graphify/graph.json``.
    Idempotent; incremental via per-file hash caching inside
    graphify.

    Returns a structured ``GraphifyBuildResult``. When no source
    files are discoverable, returns a result with
    ``skipped_reason`` set + zero counts — caller decides whether to
    surface that as a warning or as silent success.

    IMPORTANT — every graphify callable is imported from its OWN
    submodule, never via ``graphify.X``. graphify's ``__init__``
    uses lazy ``__getattr__`` to expose top-level callables, but
    importing any submodule (``from graphify.extract import
    extract``) cascades other submodule loads (``graphify.cluster``
    / ``graphify.build`` / ``graphify.export``), and once a
    submodule is in ``sys.modules`` it gets bound on the
    ``graphify`` namespace and SHADOWS the lazy callable of the
    same name. ``graphify.cluster(g)`` then raises "'module' object
    is not callable". The only safe form is "import the function
    from its submodule".
    """
    from graphify.build import build_from_json
    from graphify.cluster import cluster
    from graphify.export import to_json
    from graphify.extract import collect_files, extract

    root = Path(path).resolve()  # noqa: ASYNC240 — trivial fs op
    out_path = _graph_path(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240

    # Fast path: ``git ls-files`` returns the tracked source files
    # in ~10ms by reading the git index, skipping ``.venv`` /
    # ``node_modules`` / ``.pytest_cache`` etc. for free.
    # ``graphify.collect_files`` does an unconditional os.walk that
    # costs 6+ seconds on projects with a venv at the root,
    # dominating 95% of build wall time. Fall back to the walker
    # for non-git projects (or when git itself is missing).
    files = await _git_ls_files(root)
    source = "git ls-files"
    if files is None:
        files = collect_files(root)
        source = "graphify.collect_files (no git index)"
    if not files:
        return GraphifyBuildResult(
            graph_path=out_path,
            project_root=root,
            n_nodes=0,
            n_edges=0,
            n_files=0,
            n_communities=0,
            source=source,
            skipped_reason=(
                "no extractable source files (check tree-sitter "
                "language coverage — graphify supports py/ts/js/"
                "go/rs/java/c/cpp/rb/cs/kt/scala/php and more)"
            ),
        )
    # extract → dict (NOT list); build_from_json takes that dict
    # straight through. cluster returns a community map
    # (``dict[int, list[str]]``), NOT the graph; to_json wants both
    # the graph AND the community map as positional args, plus
    # ``force=True`` so re-runs can overwrite the prior graph.json.
    extraction = extract(files)
    graph_obj = build_from_json(extraction)
    communities = cluster(graph_obj)
    to_json(graph_obj, communities, str(out_path), force=True)
    return GraphifyBuildResult(
        graph_path=out_path,
        project_root=root,
        n_nodes=graph_obj.number_of_nodes(),
        n_edges=graph_obj.number_of_edges(),
        n_files=len(files),
        n_communities=len(communities),
        source=source,
    )


@tool
async def build(path: str = ".") -> str:
    """Extract + cluster + persist the project's knowledge graph.

    Walks code files under ``path``, parses them with tree-sitter
    via graphify's extractor, builds a NetworkX graph (nodes =
    symbols/files, edges = imports/calls/references), runs Leiden
    community detection, and writes ``<path>/.loom/graphify/graph.json``.

    Idempotent: incremental via file-hash gating — re-running on
    an unchanged repo is fast. Run once per project (or after
    major refactors); the post-commit hook keeps it current every
    5 commits.

    Returns a short summary the agent can quote back to the user.
    """
    result = await graphify_build_impl(path)
    if result.skipped_reason is not None:
        return (
            f"graphify__build: {result.skipped_reason} "
            f"(searched via {result.source})."
        )
    return (
        f"graphify__build: ✓ wrote "
        f"{result.graph_path.relative_to(result.project_root)} "
        f"({result.n_nodes} nodes, {result.n_edges} edges, "
        f"{result.n_files} source files via {result.source}, "
        f"{result.n_communities} communities)"
    )


@tool
async def query(question: str, path: str = ".") -> str:
    """BFS traversal from nodes matching ``question`` keywords —
    returns ranked related nodes + their edges.

    Use for "how does X work in this codebase" / "what's involved
    in Y" — questions that need to see the neighbourhood around a
    concept, not a single file's text.
    """
    graph_obj = _load_graph(path)
    terms = [t.lower() for t in question.split() if len(t) > 2]
    if not terms:
        return (
            "graphify__query: question too short (need at least "
            "one keyword > 2 chars)."
        )
    # Score nodes by keyword presence in label or source_file.
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for nid, data in graph_obj.nodes(data=True):
        label = str(data.get("label", "")).lower()
        source = str(data.get("source_file", "")).lower()
        score = sum(1.0 for t in terms if t in label)
        score += sum(0.4 for t in terms if t in source)
        if score > 0:
            scored.append((score, nid, data))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return f"graphify__query: no nodes matched {terms!r}."
    # Top 5 matches + their immediate neighbours.
    out: list[str] = []
    for _score, nid, data in scored[:5]:
        label = data.get("label", nid)
        src = data.get("source_file", "?")
        loc = data.get("source_location", "")
        neighbour_count = graph_obj.degree(nid)
        community = data.get("community", "?")
        out.append(
            f"• {label} [{src}{':' + str(loc) if loc else ''}] — "
            f"degree {neighbour_count}, community {community}"
        )
    return "graphify__query results:\n" + "\n".join(out)


@tool
async def path_between(a: str, b: str, path: str = ".") -> str:
    """Shortest path between two named concepts. The single most
    useful graph query: "how does A get to B?" / "what connects
    X and Y?" — exactly what grep can't answer.
    """
    graph_obj = _load_graph(path)
    a_match = _find_node(graph_obj, a)
    b_match = _find_node(graph_obj, b)
    if a_match is None:
        return f"graphify__path: no node matched {a!r}."
    if b_match is None:
        return f"graphify__path: no node matched {b!r}."
    import networkx as nx
    try:
        nodes = nx.shortest_path(graph_obj, a_match, b_match)
    except nx.NetworkXNoPath:
        return (
            f"graphify__path: no path from {a!r} → {b!r}. They "
            "live in disconnected components — likely separate "
            "subsystems with no static linkage."
        )
    hops: list[str] = []
    for i, nid in enumerate(nodes):
        label = graph_obj.nodes[nid].get("label", nid)
        src = graph_obj.nodes[nid].get("source_file", "?")
        hops.append(f"  {i}. {label} [{src}]")
        if i < len(nodes) - 1:
            edge_data = graph_obj.get_edge_data(nid, nodes[i + 1]) or {}
            relation = edge_data.get("relation", "→")
            hops.append(f"     —[{relation}]→")
    return (
        f"graphify__path {a!r} → {b!r} "
        f"({len(nodes) - 1} hops):\n" + "\n".join(hops)
    )


@tool
async def explain(node: str, path: str = ".") -> str:
    """Plain-language explanation of a single node: source
    location, immediate neighbours, community, edge count."""
    graph_obj = _load_graph(path)
    nid = _find_node(graph_obj, node)
    if nid is None:
        return f"graphify__explain: no node matched {node!r}."
    data = graph_obj.nodes[nid]
    label = data.get("label", nid)
    src = data.get("source_file", "?")
    loc = data.get("source_location", "")
    community = data.get("community", "?")
    in_degree = graph_obj.in_degree(nid) if graph_obj.is_directed() else None
    out_degree = (
        graph_obj.out_degree(nid) if graph_obj.is_directed() else None
    )
    total_degree = graph_obj.degree(nid)
    neighbours = list(graph_obj.neighbors(nid))[:10]
    parts = [
        f"{label}",
        f"  source: {src}{':' + str(loc) if loc else ''}",
        f"  community: {community}",
        f"  total degree: {total_degree}",
    ]
    if in_degree is not None:
        parts.append(
            f"  in-edges: {in_degree}  out-edges: {out_degree}"
        )
    if neighbours:
        parts.append("  neighbours:")
        for n in neighbours:
            n_label = graph_obj.nodes[n].get("label", n)
            parts.append(f"    • {n_label}")
        if total_degree > 10:
            parts.append(f"    ... and {total_degree - 10} more")
    return "graphify__explain:\n" + "\n".join(parts)


def _find_node(graph_obj: Any, name: str) -> str | None:
    """Resolve a user-supplied name to a node ID. Exact ID match
    wins; otherwise case-insensitive label substring match."""
    if name in graph_obj.nodes:
        return name
    needle = name.lower()
    for nid, data in graph_obj.nodes(data=True):
        if needle in str(data.get("label", "")).lower():
            return nid
    return None
