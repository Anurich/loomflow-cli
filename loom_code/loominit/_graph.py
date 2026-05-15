"""PageRank over the file-level import graph.

Hand-rolled power iteration — networkx would be a more familiar
implementation but it's a 5 MB transitive-dependency chain to do
roughly twenty lines of arithmetic. The loomflow design rule "no
SDK at module top" applies here too: cheap things should be self-
contained.

Math: standard PageRank with a damping factor (0.85). For a graph
where node ``i`` has out-edges to neighbours ``N(i)``::

    pr(j) = (1 - d) / N  +  d * sum( pr(i) / |N(i)| for i in inbound(j) )

We iterate until L1 change drops below tolerance or 100 iterations
elapse. Dangling nodes (no out-edges) distribute their score
uniformly across all nodes, the textbook fix.

The result is per-FILE — file-level centrality, which Aider also
uses. Per-symbol PageRank requires a call graph, which we don't
extract in v1 (see :mod:`_ast_walk` design note). Each symbol
inherits its file's PageRank score in :mod:`extractor`'s aggregation
step.
"""

from __future__ import annotations

from collections import defaultdict

_DAMPING = 0.85
_TOLERANCE = 1e-6
_MAX_ITERATIONS = 100


def pagerank_file_graph(
    *, files: list[str], edges: list[tuple[str, str]]
) -> dict[str, float]:
    """Compute PageRank for a directed graph of files.

    ``files`` is the full node set (every indexed file). ``edges`` is
    ``[(from_path, to_path), ...]`` — only RESOLVED imports, so
    third-party / stdlib edges don't dominate.

    Returns ``{rel_path: score}`` for every file in ``files``;
    files not appearing in any edge get the uniform 1/N base score.
    Returns ``{}`` if ``files`` is empty (degenerate case).
    """
    n = len(files)
    if n == 0:
        return {}

    file_set = set(files)
    # Inbound and outbound adjacency. Drop edges whose endpoints
    # aren't in our file set — defensive: caller should already
    # have filtered to resolved edges, but never trust that.
    out_adj: dict[str, list[str]] = defaultdict(list)
    in_adj: dict[str, list[str]] = defaultdict(list)
    for src, dst in edges:
        if src in file_set and dst in file_set and src != dst:
            out_adj[src].append(dst)
            in_adj[dst].append(src)

    # Initialise uniform.
    pr = {f: 1.0 / n for f in files}
    base = (1.0 - _DAMPING) / n

    for _ in range(_MAX_ITERATIONS):
        # Dangling mass: sum of scores at nodes with no out-edges,
        # redistributed uniformly to every node so the system stays
        # stochastic.
        dangling = sum(pr[f] for f in files if not out_adj[f])
        dangling_share = _DAMPING * dangling / n

        new_pr: dict[str, float] = {}
        for f in files:
            inbound_mass = sum(
                pr[src] / len(out_adj[src])
                for src in in_adj[f]
            )
            new_pr[f] = base + dangling_share + _DAMPING * inbound_mass

        # L1 convergence check — converges fast on typical repos.
        delta = sum(abs(new_pr[f] - pr[f]) for f in files)
        pr = new_pr
        if delta < _TOLERANCE:
            break

    return pr


def cluster_by_path_prefix(
    files: list[str], *, max_files_per_cluster: int = 50
) -> dict[str, list[str]]:
    """Group files by their top-level directory, then split oversized
    clusters by the NEXT directory level.

    Recursion is shallow (depth 3) — beyond that, clusters get too
    fine-grained to be useful. The result is a ``{cluster_id: [paths]}``
    map; ``cluster_id`` is the directory prefix or the bare filename
    for files at the repo root.

    Example for the loomflow tree::

        loomflow/agent/*.py             → cluster "loomflow/agent"
        loomflow/architecture/*.py      → cluster "loomflow/architecture"
        loomflow/memory/ (>50 files)    → split into
                                          "loomflow/memory/postgres",
                                          "loomflow/memory/chroma", ...

    This is a deliberately simple heuristic — most well-organized
    codebases already group by directory by convention. Import-graph
    community detection would do better on tangled codebases but
    adds complexity for marginal gain on the typical case.
    """
    return _cluster(files, depth=1, max_files=max_files_per_cluster)


def _cluster(
    files: list[str], depth: int, max_files: int
) -> dict[str, list[str]]:
    """Cluster by the first ``depth`` directory components. If any
    resulting cluster exceeds ``max_files``, recurse on it with
    ``depth+1``."""
    groups: dict[str, list[str]] = defaultdict(list)
    for path in files:
        parts = path.split("/")
        if len(parts) <= depth:
            # Top-level file (e.g. ``cli.py``) — it's its own cluster.
            groups[path].append(path)
        else:
            key = "/".join(parts[:depth])
            groups[key].append(path)

    out: dict[str, list[str]] = {}
    for key, paths in groups.items():
        if len(paths) <= max_files or depth >= 3:
            out[key] = sorted(paths)
            continue
        # Recurse one level deeper for the oversized cluster.
        sub = _cluster(paths, depth=depth + 1, max_files=max_files)
        out.update(sub)
    return out
