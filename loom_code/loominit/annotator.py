"""LLM annotation pass — turn a :class:`schema.LoomIndex` into
``LOOM.md``.

Two LLM stages, both parallel-safe and bounded by ``concurrency``:

1. **Project overview** — one call, builds the ``## Overview`` and
   ``## Tech Stack`` sections from package metadata + central
   files + notable imports.
2. **Per-cluster annotation** — one call per :class:`schema.Cluster`,
   producing the cluster's narrative + data flow + conventions +
   per-symbol purpose lines.

The result is a markdown body the caller (the ``/loominit`` slash
command in slice 6) writes via :func:`persistence.write_markdown`.
We deliberately keep file I/O out of this module — the orchestrator
returns a string so tests can pin the markdown without touching
disk.

Concurrency: ``anyio.create_task_group`` + ``anyio.Semaphore``.
Following loomflow's "anyio everywhere" rule so a Ctrl-C in the
REPL cancels in-flight annotation calls cleanly.

Why ``Agent`` and not the bare model: each annotation is one
LLM round-trip with no tools, so the Agent loop terminates after a
single model call. Using ``Agent`` gives us prompt-caching,
output-schema validation with retry, and consistent telemetry —
worth the millisecond of overhead.
"""

from __future__ import annotations

from typing import Any

import anyio
from loomflow import Agent
from loomflow.core import OutputValidationError

from ._prompts import (
    CLUSTER_ANNOTATION_INSTRUCTIONS,
    PROJECT_OVERVIEW_INSTRUCTIONS,
    SymbolPurpose,
    _ClusterAnnotationOutput,
    _ProjectOverviewOutput,
    render_cluster_prompt,
    render_project_overview_prompt,
)
from .schema import Cluster, LoomIndex, SymbolEntry

# How many concurrent annotation calls to dispatch. Most providers
# rate-limit on RPM (OpenAI ~3500/min, Anthropic ~1000/min), so 4
# is a safe default that still parallelises a ~30-cluster repo into
# ~8 round-trip windows. The REPL's /loominit can override.
DEFAULT_CONCURRENCY = 4

# Hard cap on how many symbols we put in front of the LLM per
# cluster — beyond this the prompt gets noisy and the model picks
# at random. We always include API-surface symbols; the rest are
# sorted by ``n_callers`` (proxy for "people care about it") and
# truncated.
_MAX_SYMBOLS_PER_CLUSTER = 60


async def annotate(
    index: LoomIndex,
    *,
    model: Any,
    concurrency: int = DEFAULT_CONCURRENCY,
    project_metadata: dict[str, str] | None = None,
) -> str:
    """Run the full annotation pipeline against ``index``.

    Returns the LOOM.md body as a string. Side-effect-free; the
    caller persists via :func:`persistence.write_markdown`.

    ``project_metadata``: optional ``{"name": ..., "description":
    ..., "requires_python": ...}`` from pyproject.toml. The REPL
    builds this dict; passing it in keeps this module out of TOML
    parsing.
    """
    if not index.files:
        # Empty repo — emit a minimal placeholder so the user sees
        # /loominit ran (and so subsequent /loominit refresh has
        # something to diff against).
        return _render_empty(index)

    metadata = project_metadata or {}

    overview = await _annotate_project(
        model=model, index=index, metadata=metadata
    )
    cluster_results = await _annotate_clusters(
        model=model, index=index, concurrency=concurrency
    )

    return _assemble_markdown(
        index=index,
        metadata=metadata,
        overview=overview,
        cluster_results=cluster_results,
    )


# ---------------------------------------------------------------------------
# Stage 1 — project overview
# ---------------------------------------------------------------------------


async def _annotate_project(
    *, model: Any, index: LoomIndex, metadata: dict[str, str]
) -> _ProjectOverviewOutput:
    """Single LLM call that produces overview + tech_stack."""
    project_name = metadata.get(
        "name", index.repo_root.rstrip("/").rsplit("/", 1)[-1] or "project"
    )
    description = metadata.get("description")
    requires_python = metadata.get("requires_python")

    # Top-level directory counts.
    dir_counts: dict[str, int] = {}
    for f in index.files:
        top = f.path.split("/", 1)[0] if "/" in f.path else "(root)"
        dir_counts[top] = dir_counts.get(top, 0) + 1
    top_dirs = sorted(dir_counts.items(), key=lambda x: -x[1])[:8]

    entry_lines = [
        _render_entry_point(ep) for ep in index.entry_points[:10]
    ]

    # Most-central files by file-level PageRank (inferred from
    # symbol.pagerank — symbols in the same file share their file's
    # PageRank in slice 1).
    file_pr: dict[str, float] = {}
    file_inbound: dict[str, int] = {}
    for s in index.symbols:
        file_pr[s.path] = max(file_pr.get(s.path, 0.0), s.pagerank)
        file_inbound[s.path] = max(
            file_inbound.get(s.path, 0), s.n_callers
        )
    central = sorted(file_pr.items(), key=lambda x: -x[1])[:10]
    central_files = [(p, file_inbound.get(p, 0)) for p, _ in central]

    notable_imports = _notable_third_party(index)[:10]

    user_msg = render_project_overview_prompt(
        project_name=project_name,
        project_description=description,
        requires_python=requires_python,
        top_dirs=top_dirs,
        entry_points=entry_lines,
        central_files=central_files,
        notable_imports=notable_imports,
    )
    # Single-call agent dedicated to project overview — gets its own
    # output_schema since Agent freezes that at construction.
    project_agent = Agent(
        instructions=PROJECT_OVERVIEW_INSTRUCTIONS,
        model=model,
        prompt_caching=True,
        output_schema=_ProjectOverviewOutput,
    )
    # Two failure modes to handle:
    #
    # 1. ``OutputValidationError`` — loomflow raises this after
    #    exhausting ``output_validation_retries`` if the model
    #    keeps returning unparseable JSON (markdown fences,
    #    prose before/after the JSON, etc.). Was originally
    #    treated as ``parsed=None`` here — but that branch never
    #    fires because the exception aborts the run first.
    #    Without this catch, /loominit crashes on a single
    #    malformed response.
    # 2. Other exceptions (network blip, key missing, adapter
    #    bug) — same fallback. Better to ship a placeholder
    #    overview than crash a 30-second pipeline.
    parsed: _ProjectOverviewOutput | None = None
    try:
        result = await project_agent.run(user_msg, user_id="loom-code")
        if isinstance(result.parsed, _ProjectOverviewOutput):
            parsed = result.parsed
    except (OutputValidationError, Exception):  # noqa: BLE001
        parsed = None
    if parsed is not None:
        return parsed
    return _ProjectOverviewOutput(
        overview=(description or f"{project_name} — see source.").strip(),
        tech_stack=[],
    )


# ---------------------------------------------------------------------------
# Stage 2 — per-cluster annotation (parallel)
# ---------------------------------------------------------------------------


async def _annotate_clusters(
    *, model: Any, index: LoomIndex, concurrency: int
) -> dict[str, _ClusterAnnotationOutput]:
    """Dispatch one annotation call per cluster, bounded by
    ``concurrency``. Returns ``{cluster_id: output}``."""
    sem = anyio.Semaphore(max(1, concurrency))
    results: dict[str, _ClusterAnnotationOutput] = {}

    async def _one(cluster: Cluster) -> None:
        async with sem:
            out = await _annotate_one_cluster(
                cluster=cluster, index=index, model=model
            )
            results[cluster.id] = out

    async with anyio.create_task_group() as tg:
        for c in index.clusters:
            tg.start_soon(_one, c)
    return results


async def _annotate_one_cluster(
    *, cluster: Cluster, index: LoomIndex, model: Any
) -> _ClusterAnnotationOutput:
    """One LLM call for one cluster — assembles the structured
    facts, dispatches, returns the parsed output."""
    syms_in_cluster = [s for s in index.symbols if s.path in cluster.paths]
    syms_in_cluster = _trim_symbols_for_cluster(syms_in_cluster)
    symbol_table = _render_symbol_table(syms_in_cluster)

    outbound_internal, inbound_internal, third_party = _classify_imports(
        cluster=cluster, index=index
    )

    entry_lines = [
        _render_entry_point(ep)
        for ep in index.entry_points
        if ep.path in cluster.paths
    ]

    prompt = render_cluster_prompt(
        cluster_title=cluster.title,
        paths=cluster.paths,
        symbol_table=symbol_table,
        outbound_internal=outbound_internal,
        inbound_internal=inbound_internal,
        third_party=third_party,
        entry_points_in_cluster=entry_lines,
    )

    cluster_agent = Agent(
        instructions=CLUSTER_ANNOTATION_INSTRUCTIONS,
        model=model,
        prompt_caching=True,
        output_schema=_ClusterAnnotationOutput,
    )
    # Same defensive shape as :func:`_annotate_project`: catch
    # OutputValidationError + everything else, fall back to a
    # placeholder annotation rather than aborting the pipeline.
    # One bad cluster shouldn't kill /loominit for the rest of
    # the codebase.
    parsed: _ClusterAnnotationOutput | None = None
    try:
        result = await cluster_agent.run(prompt, user_id="loom-code")
        if isinstance(result.parsed, _ClusterAnnotationOutput):
            parsed = result.parsed
    except (OutputValidationError, Exception):  # noqa: BLE001
        parsed = None
    if parsed is not None:
        return _filter_hallucinated_citations(parsed, syms_in_cluster)
    return _ClusterAnnotationOutput(
        narrative="(annotation unavailable)",
        data_flow=[],
        conventions=[],
        symbol_purposes=[],
    )


def _trim_symbols_for_cluster(
    symbols: list[SymbolEntry],
) -> list[SymbolEntry]:
    """Cap the per-cluster symbol list so big clusters don't blow
    the prompt budget. Strategy: keep all API-surface symbols, then
    pad up to :data:`_MAX_SYMBOLS_PER_CLUSTER` with the highest
    ``n_callers``."""
    api = [s for s in symbols if s.in_api_surface]
    non_api = sorted(
        (s for s in symbols if not s.in_api_surface),
        key=lambda s: (-s.n_callers, s.id),
    )
    remaining = max(0, _MAX_SYMBOLS_PER_CLUSTER - len(api))
    return api + non_api[:remaining]


def _filter_hallucinated_citations(
    out: _ClusterAnnotationOutput, real_symbols: list[SymbolEntry]
) -> _ClusterAnnotationOutput:
    """Drop ``symbol_purposes`` entries whose ``(path, line)``
    doesn't ground in a real symbol in this cluster.

    Models drift the citation in two distinct ways and we treat
    them differently:

    * **Off-by-one line** (same path, line within :data:`_SNAP_RADIUS`
      of a real symbol of that name) — snap to the real location.
      The purpose text is still useful.
    * **Invented path or wildly wrong line** — drop entirely. If
      the model can't get the citation roughly right, we can't
      trust the description either.

    The grounding promise is strict: the index NEVER carries a
    citation pointing at code that doesn't exist."""
    real_by_loc = {(s.path, s.line): s for s in real_symbols}
    real_by_name = {s.name: s for s in real_symbols}
    kept: list[SymbolPurpose] = []
    for p in out.symbol_purposes:
        if (p.path, p.line) in real_by_loc:
            kept.append(p)
            continue
        # Snap-window: same name + close line + same OR sibling
        # path within the cluster.
        target = real_by_name.get(p.name)
        if (
            target is not None
            and abs(target.line - p.line) <= _SNAP_RADIUS
        ):
            kept.append(
                SymbolPurpose(
                    name=p.name,
                    path=target.path,
                    line=target.line,
                    purpose=p.purpose,
                )
            )
            # else: model hallucinated path AND line — discard.
    return _ClusterAnnotationOutput(
        narrative=out.narrative,
        data_flow=out.data_flow,
        conventions=out.conventions,
        symbol_purposes=kept,
    )


# How far the model's line number can drift from the real symbol's
# line and still get snapped instead of dropped. 5 lines is wide
# enough to absorb 1-indexed/0-indexed confusion and "the model
# counted the decorator line instead of the def line" — and narrow
# enough that a totally fabricated line still gets discarded.
_SNAP_RADIUS = 5


# ---------------------------------------------------------------------------
# Helpers — formatting / classification
# ---------------------------------------------------------------------------


def _render_symbol_table(symbols: list[SymbolEntry]) -> str:
    """One line per symbol in the prompt — keeps the LLM input
    compact + scannable. The model sees this and uses ``path:line``
    + ``name`` verbatim when emitting citations."""
    if not symbols:
        return "(no symbols)"
    lines: list[str] = []
    for s in symbols:
        marker = " [api]" if s.in_api_surface else ""
        doc = f" — {s.docstring_first_line}" if s.docstring_first_line else ""
        lines.append(
            f"{s.path}:{s.line}  {s.qualified_name}  ({s.kind})  "
            f"{s.signature.rstrip(':')}{marker}{doc}"
        )
    return "\n".join(lines)


def _classify_imports(
    *, cluster: Cluster, index: LoomIndex
) -> tuple[list[str], list[str], list[str]]:
    """Split a cluster's import edges into three buckets:

    * **outbound_internal** — edges from this cluster TO other
      in-repo files (resolved=True, src in cluster, dst not in
      cluster).
    * **inbound_internal** — edges INTO this cluster from elsewhere
      in the repo.
    * **third_party** — unresolved edges originating in this
      cluster (likely PyPI deps).

    Returns short markdown-ready strings, not raw edges, since
    that's what the prompt template wants. We dedupe + cap each
    list at 15 entries to bound prompt size."""
    cluster_paths = set(cluster.paths)
    outbound: dict[str, int] = {}
    inbound: dict[str, int] = {}
    third: dict[str, int] = {}

    # Map files -> their cluster id so we can label inbound edges
    # by source cluster.
    file_to_cluster: dict[str, str] = {}
    for c in index.clusters:
        for p in c.paths:
            file_to_cluster[p] = c.id

    for edge in index.imports:
        src_in = edge.from_path in cluster_paths
        if edge.resolved:
            # Resolved edges in the schema only carry display
            # to_module, not the file. Re-derive by checking if
            # the to_module dotted name maps to any path inside
            # the cluster.
            target_path = _resolved_target(edge.to_module, index)
            dst_in = target_path in cluster_paths if target_path else False
            if src_in and not dst_in and target_path:
                outbound[target_path] = outbound.get(target_path, 0) + 1
            elif dst_in and not src_in:
                src_cluster = file_to_cluster.get(edge.from_path, "?")
                key = f"{edge.from_path} (cluster {src_cluster})"
                inbound[key] = inbound.get(key, 0) + 1
        else:
            if src_in:
                # ``edge.to_module`` for third-party is the literal
                # source form (``"click"`` / ``"loomflow.tools"``).
                # Strip submodule suffixes so we group ``loomflow.X``
                # and ``loomflow.Y`` under ``loomflow``.
                top = edge.to_module.split(".")[0]
                if top and not top.startswith("."):
                    third[top] = third.get(top, 0) + 1

    def _top(d: dict[str, int]) -> list[str]:
        return [
            f"{name} ({n})"
            for name, n in sorted(d.items(), key=lambda x: -x[1])[:15]
        ]

    return _top(outbound), _top(inbound), _top(third)


def _resolved_target(to_module: str, index: LoomIndex) -> str | None:
    """Map a resolved import's ``to_module`` (display form, may
    include leading dots for relative imports) back to a file path.

    Returns ``None`` if no file in the index matches — caller treats
    that as "edge points outside the indexed file set".
    """
    if to_module.startswith("."):
        # Relative imports in the schema display retain their dots;
        # we can't resolve them without re-running the relative-
        # import logic from :mod:`_resolve`. They're already counted
        # by the resolver into ``edge.resolved`` so for the cluster
        # classifier we'd need the original raw form. Compromise:
        # treat as unmatched here; cluster classification on
        # relative imports across clusters is rare in well-organised
        # repos.
        return None
    target_dotted = to_module
    for f in index.files:
        rel = f.path
        if rel.endswith(".py"):
            dotted = rel.removesuffix(".py").replace("/", ".")
            if dotted == target_dotted or dotted.endswith(
                "." + target_dotted
            ):
                return rel
            if rel.endswith("/__init__.py"):
                pkg = rel.removesuffix("/__init__.py").replace("/", ".")
                if pkg == target_dotted:
                    return rel
    return None


def _notable_third_party(index: LoomIndex) -> list[str]:
    """Count top-level third-party packages across the repo, sorted
    by import frequency. Used in the project-overview prompt to hint
    at the tech stack."""
    counts: dict[str, int] = {}
    for edge in index.imports:
        if edge.resolved:
            continue
        top = edge.to_module.split(".")[0]
        if not top or top.startswith("."):
            continue
        counts[top] = counts.get(top, 0) + 1
    return [
        f"{name} ({n})"
        for name, n in sorted(counts.items(), key=lambda x: -x[1])
    ]


def _render_entry_point(ep) -> str:  # noqa: ANN001 — local helper
    """Compact one-line entry-point description for prompts."""
    where = ep.callable_id or ep.path
    if ep.kind == "pyproject_script":
        return f"[script] {ep.name}  →  {where}"
    if ep.kind == "main_block":
        return f"[__main__] {where}:{ep.line}"
    return f"[{ep.name}] {where}"


# ---------------------------------------------------------------------------
# Markdown assembler
# ---------------------------------------------------------------------------


def _assemble_markdown(
    *,
    index: LoomIndex,
    metadata: dict[str, str],
    overview: _ProjectOverviewOutput,
    cluster_results: dict[str, _ClusterAnnotationOutput],
) -> str:
    """Stitch the LLM outputs + structural data into the final
    LOOM.md body. Deterministic — same inputs → byte-equal output."""
    name = metadata.get("name", "project")
    parts: list[str] = []
    parts.append(f"# {name} — LOOM.md\n")
    parts.append(
        "Generated by loom-code. Run `/loominit refresh` to update; "
        "claims marked `(stale: ...)` were derived from a file whose "
        "content has since changed.\n"
    )
    if index.git_commit:
        parts.append(f"_built at commit {index.git_commit}_\n")
    parts.append("")

    parts.append("## Overview\n")
    parts.append(overview.overview.strip() + "\n")

    if overview.tech_stack:
        parts.append("## Tech Stack\n")
        for item in overview.tech_stack:
            parts.append(f"- {item.strip()}")
        parts.append("")

    if index.entry_points:
        parts.append("## Entry Points\n")
        for ep in index.entry_points:
            parts.append(f"- {_render_entry_point(ep)}")
        parts.append("")

    parts.append("## Subsystems\n")
    for cluster in index.clusters:
        cr = cluster_results.get(cluster.id)
        if cr is None:
            continue
        parts.append(f"### {cluster.title}\n")
        parts.append(f"_files: {', '.join(cluster.paths[:6])}"
                     + ("…" if len(cluster.paths) > 6 else "") + "_\n")
        parts.append(cr.narrative.strip() + "\n")
        if cr.data_flow:
            parts.append("**Data flow:**\n")
            for b in cr.data_flow:
                parts.append(f"- {b.strip()}")
            parts.append("")
        if cr.conventions:
            parts.append("**Conventions:**\n")
            for b in cr.conventions:
                parts.append(f"- {b.strip()}")
            parts.append("")
        if cr.symbol_purposes:
            parts.append("**Symbols:**\n")
            for sp in cr.symbol_purposes:
                parts.append(
                    f"- `{sp.name}` ({sp.path}:{sp.line}) — "
                    f"{sp.purpose.strip()}"
                )
            parts.append("")

    # Pending annotations — empty on first generation; slice 4
    # populates this section as the agent adds new symbols. We
    # always emit the header so the staleness pipeline knows where
    # to append.
    parts.append("## Pending annotations\n")
    parts.append(
        "_New public symbols added since the last `/loominit refresh` "
        "will appear here. None yet._\n"
    )

    return "\n".join(parts)


def _render_empty(index: LoomIndex) -> str:
    """Placeholder LOOM.md for a project with no source files yet.

    The user might run /loominit on a freshly-cloned repo or an
    empty scaffold; emitting something (rather than crashing) keeps
    the flow consistent. Slice 4's refresh logic will detect new
    files added later."""
    return (
        "# LOOM.md\n\n"
        "_no source files indexed yet. Run `/loominit refresh` "
        "after adding code._\n"
    )
