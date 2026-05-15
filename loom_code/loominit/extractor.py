"""Top-level orchestrator — ``build_index(repo_root) -> LoomIndex``.

This is the only public entry point for slice 1. It composes every
private helper in the package into one deterministic pipeline:

1. :func:`_files.discover_files` — walk the repo, hash every file,
   capture mtime + git heat + ``is_test`` flag.
2. :func:`_ast_walk.walk_python_file` — per-file AST extraction
   producing ``(_RawSymbol, _RawImport, _RawDecorator)`` triples.
3. :func:`_resolve.build_module_index` + :func:`resolve_imports` —
   turn dotted module names into rel-path edges.
4. :func:`_resolve.detect_api_surface` — flag files reachable from
   ``__init__.py`` re-exports / ``__all__``.
5. :func:`_graph.pagerank_file_graph` — file-level centrality from
   the resolved import graph.
6. :func:`_tests_map.build_test_map` — grep test files for each
   symbol's bare name, emit citations.
7. :func:`_resolve.extract_entry_points` — pyproject scripts,
   ``__main__`` blocks, landmark-decorated callables.
8. :func:`_graph.cluster_by_path_prefix` — group files into
   subsystem clusters with hash buckets for diff-aware refresh.

The result is a :class:`schema.LoomIndex` ready for
:func:`persistence.save_index` to write to disk. No LLM calls
anywhere here — slice 1 is deterministic by design.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from ._ast_walk import _RawDecorator, _RawSymbol, walk_python_file
from ._files import DiscoveredFile, discover_files, is_git_repo
from ._graph import cluster_by_path_prefix, pagerank_file_graph
from ._resolve import (
    build_module_index,
    detect_api_surface,
    extract_entry_points,
    resolve_imports,
)
from ._tests_map import build_test_map
from .schema import (
    CallEdge,
    Cluster,
    DecoratorLandmark,
    FileEntry,
    LoomIndex,
    SymbolEntry,
)


def build_index(repo_root: Path) -> LoomIndex:
    """Run the full structural extraction pass and return the index.

    Side effect: NONE. The caller persists via
    :func:`persistence.save_index`. Keeping I/O separate makes the
    pipeline trivially testable against in-memory fixtures.
    """
    files = discover_files(repo_root)
    py_files = [f for f in files if f.lang == "python"]

    # ---- 1. Walk every Python file ------------------------------------
    per_file = _walk_all(py_files)

    # ---- 2. Module-name resolution / API surface ----------------------
    module_index = build_module_index(files)
    api_surface = detect_api_surface(files, module_index)

    # ---- 3. Aggregate symbols + assign file-level PageRank ------------
    raw_imports_by_file: dict[str, list[tuple[str, int, int]]] = {
        path: [(imp.to_module, imp.line, imp.level) for imp in imps]
        for path, (_syms, imps, _decs) in per_file.items()
    }
    import_edges = resolve_imports(raw_imports_by_file, module_index)

    # File-level PageRank: edges restricted to resolved (in-repo) ones.
    edges_for_pagerank = _pagerank_edges(
        import_edges, py_files, module_index, raw_imports_by_file
    )
    file_scores = pagerank_file_graph(
        files=[f.rel_path for f in py_files],
        edges=edges_for_pagerank,
    )

    # n_callers / n_callees per file (resolved edges only).
    in_degree, out_degree = _degree_maps(edges_for_pagerank)

    # ---- 4. Test→symbol map -------------------------------------------
    all_symbol_names = {
        sym.name
        for path, (syms, _imps, _decs) in per_file.items()
        for sym in syms
        if sym.is_public
    }
    test_map = build_test_map(files=files, symbol_names=all_symbol_names)

    # ---- 5. Build schema.FileEntry list -------------------------------
    file_entries = [
        FileEntry(
            path=f.rel_path,
            lang=f.lang,
            size_bytes=f.size_bytes,
            lines=f.lines,
            sha256=f.sha256,
            mtime=f.mtime,
            git_changes_90d=f.git_changes_90d,
            is_test=f.is_test,
            in_api_surface=f.rel_path in api_surface,
        )
        for f in files
    ]

    # ---- 6. Build schema.SymbolEntry list -----------------------------
    symbol_entries: list[SymbolEntry] = []
    for path, (raw_syms, _imps, _decs) in per_file.items():
        file_pr = file_scores.get(path, 0.0)
        file_in = in_degree.get(path, 0)
        file_out = out_degree.get(path, 0)
        for raw in raw_syms:
            sym_id = f"{path}:{raw.qualified_name}"
            symbol_entries.append(
                SymbolEntry(
                    id=sym_id,
                    name=raw.name,
                    qualified_name=raw.qualified_name,
                    kind=raw.kind,
                    path=path,
                    line=raw.line,
                    end_line=raw.end_line,
                    signature=raw.signature,
                    docstring_first_line=raw.docstring_first_line,
                    decorators=list(raw.decorators),
                    is_public=raw.is_public,
                    in_api_surface=(path in api_surface) and raw.is_public,
                    # File-level for now; symbol-level requires call graph
                    pagerank=file_pr,
                    n_callers=file_in,
                    n_callees=file_out,
                    tests=test_map.get(raw.name, []) if raw.is_public else [],
                )
            )

    # ---- 7. Decorator landmarks --------------------------------------
    decorator_entries: list[DecoratorLandmark] = []
    decorator_path_lookup: dict[_RawDecorator, str] = {}
    for path, (_syms, _imps, raw_decs) in per_file.items():
        for raw in raw_decs:
            decorator_path_lookup[raw] = path
            decorator_entries.append(
                DecoratorLandmark(
                    decorator=raw.decorator,
                    target=f"{path}:{raw.target_qualname}",
                    path=path,
                    line=raw.line,
                )
            )

    # ---- 8. Entry points ---------------------------------------------
    all_decorators = [
        raw
        for _path, (_syms, _imps, raw_decs) in per_file.items()
        for raw in raw_decs
    ]
    entry_points = extract_entry_points(
        repo_root=repo_root,
        files=files,
        decorators=all_decorators,
        decorator_path_lookup=decorator_path_lookup,
    )

    # ---- 9. Clusters --------------------------------------------------
    cluster_entries = _build_clusters(file_entries, py_files)

    git_commit = _current_git_commit(repo_root)

    return LoomIndex(
        generated_at=datetime.now(UTC),
        repo_root=str(repo_root.resolve()),
        git_commit=git_commit,
        files=file_entries,
        symbols=symbol_entries,
        imports=import_edges,
        # Call graph stays empty in v1 — see _ast_walk's design note.
        # Schema accommodates future call-graph extraction so we
        # don't need a schema bump when we add it.
        calls=list[CallEdge](),
        decorators=decorator_entries,
        entry_points=entry_points,
        clusters=cluster_entries,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _walk_all(
    py_files: list[DiscoveredFile],
) -> dict[
    str,
    tuple[
        list[_RawSymbol],
        list,  # _RawImport
        list[_RawDecorator],
    ],
]:
    """Parse every Python file in parallel-friendly form (sequential
    for now; can wrap in ``anyio.to_thread.run_sync`` later if
    profiling shows it matters)."""
    out = {}
    for f in py_files:
        try:
            text = f.abs_path.read_text(encoding="utf-8")
        except OSError:
            continue
        syms, imps, decs = walk_python_file(text, f.rel_path)
        out[f.rel_path] = (syms, imps, decs)
    return out


def _pagerank_edges(
    edges: Iterable,  # ImportEdge
    py_files: list[DiscoveredFile],
    module_index,
    raw_imports_by_file: dict[str, list[tuple[str, int, int]]],
) -> list[tuple[str, str]]:
    """Convert :class:`ImportEdge` records into ``(from_path,
    to_path)`` pairs suitable for :func:`_graph.pagerank_file_graph`.

    We can't read ``to_path`` straight off :class:`ImportEdge` (the
    schema stores the display module name, not the resolved file).
    So we re-resolve here using the same machinery the edges came
    from. Cheap and keeps the schema clean.
    """
    from ._resolve import resolve_import

    py_paths = {f.rel_path for f in py_files}
    pairs: list[tuple[str, str]] = []
    for from_path, items in raw_imports_by_file.items():
        if from_path not in py_paths:
            continue
        for to_module, _line, level in items:
            target = resolve_import(
                from_file=from_path,
                to_module=to_module,
                level=level,
                module_index=module_index,
            )
            if target is not None and target in py_paths:
                pairs.append((from_path, target))
    return pairs


def _degree_maps(
    edges: list[tuple[str, str]],
) -> tuple[dict[str, int], dict[str, int]]:
    """``in_degree[file]`` = number of files importing it.
    ``out_degree[file]`` = number of files it imports.

    Both are used for the SymbolEntry's ``n_callers`` / ``n_callees``
    fields as file-level approximations until call-graph extraction
    lands."""
    in_deg: dict[str, int] = {}
    out_deg: dict[str, int] = {}
    for src, dst in edges:
        out_deg[src] = out_deg.get(src, 0) + 1
        in_deg[dst] = in_deg.get(dst, 0) + 1
    return in_deg, out_deg


def _build_clusters(
    files: list[FileEntry], py_files: list[DiscoveredFile]
) -> list[Cluster]:
    """Group Python files into subsystem clusters via path prefix,
    compute centroid symbols (highest-PageRank file → its symbols)
    and hash buckets per cluster."""
    py_paths = [f.rel_path for f in py_files]
    groups = cluster_by_path_prefix(py_paths)

    # Map files -> entries for hash lookup.
    by_path = {f.path: f for f in files}

    clusters: list[Cluster] = []
    for cluster_id, paths in groups.items():
        hashes = sorted(
            by_path[p].sha256 for p in paths if p in by_path
        )
        bucket = hashlib.sha256(
            "\n".join(hashes).encode("utf-8")
        ).hexdigest()[:16]
        title = cluster_id.replace("/", "/").rstrip("/") or "root"
        clusters.append(
            Cluster(
                id=cluster_id.replace("/", ".") or "root",
                title=title,
                paths=list(paths),
                centroid_symbols=[],  # populated in annotator pass
                centrality=0.0,  # populated alongside centroids
                hash_bucket=bucket,
            )
        )
    return sorted(clusters, key=lambda c: c.id)


def _current_git_commit(repo_root: Path) -> str | None:
    """``git rev-parse HEAD`` — short-form (12 chars) commit hash.
    Useful for diff-aware refresh: "the index was built against
    commit X; you're on commit Y; here's what files changed"."""
    if not is_git_repo(repo_root):
        return None
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out or None
