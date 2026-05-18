"""Tests for the LLM annotation pass.

Strategy:

* **Deterministic helpers** (prompt rendering, symbol-table render,
  import classifier, citation filter, markdown assembler) are
  tested directly — no model needed.
* **The full :func:`annotate` orchestrator** is tested against
  loomflow's ``ScriptedModel``, which returns canned JSON for
  each turn. That exercises the parallel-dispatch + structured-
  output + assembler path end-to-end without touching a paid
  endpoint.

The fixture index is hand-built rather than going through
``build_index`` so the tests stay independent of slice 1's
extractor (this also makes it obvious WHICH structural fact a
given prompt assertion is checking)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from loomflow import ScriptedModel, ScriptedTurn

from loom_code.loominit._prompts import (
    SymbolPurpose,
    _ClusterAnnotationOutput,
    _ProjectOverviewOutput,
    render_cluster_prompt,
    render_project_overview_prompt,
)
from loom_code.loominit.annotator import (
    _assemble_markdown,
    _classify_imports,
    _filter_hallucinated_citations,
    _render_symbol_table,
    annotate,
)
from loom_code.loominit.schema import (
    Cluster,
    DecoratorLandmark,
    EntryPoint,
    FileEntry,
    ImportEdge,
    LoomIndex,
    SymbolEntry,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _file(path: str, *, sha: str = "h", api: bool = False) -> FileEntry:
    return FileEntry(
        path=path,
        lang="python",
        size_bytes=1,
        lines=1,
        sha256=sha,
        mtime=datetime(2026, 1, 1, tzinfo=UTC),
        git_changes_90d=0,
        is_test=False,
        in_api_surface=api,
    )


def _sym(
    path: str,
    name: str,
    *,
    line: int = 1,
    api: bool = False,
    callers: int = 0,
    pr: float = 0.1,
) -> SymbolEntry:
    return SymbolEntry(
        id=f"{path}:{name}",
        name=name,
        qualified_name=name,
        kind="class",
        path=path,
        line=line,
        end_line=line,
        signature=f"class {name}:",
        docstring_first_line=f"The {name}.",
        decorators=[],
        is_public=True,
        in_api_surface=api,
        pagerank=pr,
        n_callers=callers,
        n_callees=0,
        tests=[],
    )


# ---- prompt rendering -----------------------------------------------


def test_render_project_overview_prompt_includes_facts() -> None:
    """Project overview prompt must echo the structural facts the
    LLM is supposed to ground its narrative in. If we drop a fact
    silently, the model has to invent — which is exactly what we
    promised not to allow."""
    body = render_project_overview_prompt(
        project_name="mypkg",
        project_description="A small thing.",
        requires_python=">=3.11",
        top_dirs=[("mypkg", 5), ("tests", 2)],
        entry_points=["[script] mycli  →  mypkg/cli.py:main"],
        central_files=[("mypkg/engine.py", 3)],
        notable_imports=["click (1)"],
    )
    assert "mypkg" in body
    assert "A small thing." in body
    assert ">=3.11" in body
    assert "mypkg/  (5 files)" in body
    assert "tests/  (2 files)" in body
    assert "[script] mycli" in body
    assert "mypkg/engine.py" in body
    assert "click (1)" in body
    assert "Return a JSON object" in body


def test_render_cluster_prompt_includes_symbols_and_edges() -> None:
    body = render_cluster_prompt(
        cluster_title="loom_code/loominit",
        paths=["loom_code/loominit/extractor.py"],
        symbol_table="x.py:1  Foo  (class)  class Foo: [api]",
        outbound_internal=["other.py (2)"],
        inbound_internal=["caller.py (cluster X) (1)"],
        third_party=["pydantic (3)"],
        entry_points_in_cluster=["[script] thing  →  x.py:main"],
    )
    assert "loom_code/loominit" in body
    assert "Foo" in body
    assert "other.py (2)" in body
    assert "pydantic (3)" in body
    assert "Return a JSON object" in body


# ---- symbol-table renderer ------------------------------------------


def test_render_symbol_table_marks_api() -> None:
    table = _render_symbol_table(
        [
            _sym("a.py", "Pub", api=True),
            _sym("a.py", "Priv", api=False),
        ]
    )
    assert "[api]" in table
    pub_line = next(ln for ln in table.splitlines() if "Pub" in ln)
    priv_line = next(ln for ln in table.splitlines() if "Priv" in ln)
    assert "[api]" in pub_line
    assert "[api]" not in priv_line


def test_render_symbol_table_empty() -> None:
    """No symbols → readable placeholder, NOT empty string —
    "(no symbols)" gives the LLM a clear signal."""
    assert _render_symbol_table([]) == "(no symbols)"


# ---- import classifier ----------------------------------------------


def test_classify_imports_separates_internal_and_external() -> None:
    files = [_file("a.py"), _file("b.py"), _file("c.py")]
    cluster_a = Cluster(
        id="a",
        title="a",
        paths=["a.py"],
        centroid_symbols=[],
        centrality=0.0,
        hash_bucket="x",
    )
    cluster_b = Cluster(
        id="b",
        title="b",
        paths=["b.py", "c.py"],
        centroid_symbols=[],
        centrality=0.0,
        hash_bucket="y",
    )
    edges = [
        # a.py imports b (internal outbound for cluster A)
        ImportEdge(from_path="a.py", to_module="b", line=1, resolved=True),
        # c.py imports a (internal inbound for cluster A)
        ImportEdge(from_path="c.py", to_module="a", line=1, resolved=True),
        # a.py imports click (third-party)
        ImportEdge(
            from_path="a.py", to_module="click", line=2, resolved=False
        ),
    ]
    idx = LoomIndex(
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        repo_root="/r",
        git_commit=None,
        files=files,
        symbols=[],
        imports=edges,
        decorators=[],
        entry_points=[],
        clusters=[cluster_a, cluster_b],
    )
    outbound, inbound, third = _classify_imports(
        cluster=cluster_a, index=idx
    )
    # Outbound: b.py (count 1)
    assert any("b.py" in s for s in outbound)
    # Inbound: c.py
    assert any("c.py" in s for s in inbound)
    # Third party: click
    assert any("click" in s for s in third)


# ---- hallucinated citation filter ------------------------------------


def test_filter_drops_invented_paths() -> None:
    """Model returns a citation with a path NOT in the cluster's
    real symbols → drop it. Filter promise is strict because the
    LLM occasionally hallucinates."""
    real = [_sym("real.py", "X", line=10)]
    out = _ClusterAnnotationOutput(
        narrative="x",
        data_flow=[],
        conventions=[],
        symbol_purposes=[
            SymbolPurpose(
                name="X", path="real.py", line=10, purpose="ok"
            ),
            SymbolPurpose(
                name="X",
                path="invented.py",
                line=99,
                purpose="hallucinated",
            ),
        ],
    )
    filtered = _filter_hallucinated_citations(out, real)
    assert len(filtered.symbol_purposes) == 1
    assert filtered.symbol_purposes[0].path == "real.py"


def test_filter_corrects_off_by_one_line() -> None:
    """Same name, same cluster, wrong line by one → snap to the
    real location rather than drop. Off-by-one is the common LLM
    failure mode; dropping useful annotations is worse than
    snapping."""
    real = [_sym("real.py", "X", line=10)]
    out = _ClusterAnnotationOutput(
        narrative="x",
        data_flow=[],
        conventions=[],
        symbol_purposes=[
            SymbolPurpose(
                name="X", path="wrong.py", line=11, purpose="ok"
            ),
        ],
    )
    filtered = _filter_hallucinated_citations(out, real)
    assert len(filtered.symbol_purposes) == 1
    assert filtered.symbol_purposes[0].path == "real.py"
    assert filtered.symbol_purposes[0].line == 10


# ---- markdown assembler ---------------------------------------------


def test_assemble_markdown_includes_all_sections() -> None:
    index = LoomIndex(
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        repo_root="/r",
        git_commit="abc",
        files=[_file("x.py")],
        symbols=[_sym("x.py", "X")],
        imports=[],
        decorators=[
            DecoratorLandmark(
                decorator="click.command",
                target="x.py:main",
                path="x.py",
                line=1,
            )
        ],
        entry_points=[
            EntryPoint(
                kind="pyproject_script",
                name="mycli",
                path="pyproject.toml",
                line=None,
                callable_id="x.py:main",
            )
        ],
        clusters=[
            Cluster(
                id="root",
                title="root",
                paths=["x.py"],
                centroid_symbols=[],
                centrality=0.0,
                hash_bucket="b",
            )
        ],
    )
    overview = _ProjectOverviewOutput(
        overview="A library that does the thing.",
        tech_stack=["Python 3.11+", "click for CLI"],
    )
    cluster_results = {
        "root": _ClusterAnnotationOutput(
            narrative="The root cluster runs everything.",
            data_flow=["request → handler → response"],
            conventions=["uses async everywhere"],
            symbol_purposes=[
                SymbolPurpose(
                    name="X", path="x.py", line=1, purpose="entry"
                )
            ],
        )
    }
    md = _assemble_markdown(
        index=index,
        metadata={"name": "mypkg"},
        overview=overview,
        cluster_results=cluster_results,
    )
    assert "# mypkg — LOOM.md" in md
    assert "## Overview" in md
    assert "A library that does the thing." in md
    assert "## Tech Stack" in md
    assert "- Python 3.11+" in md
    assert "## Entry Points" in md
    assert "[script] mycli" in md
    assert "## Subsystems" in md
    assert "### root" in md
    assert "The root cluster runs everything." in md
    assert "**Data flow:**" in md
    assert "request → handler → response" in md
    assert "**Conventions:**" in md
    assert "uses async everywhere" in md
    assert "**Symbols:**" in md
    assert "## Pending annotations" in md
    assert "built at commit abc" in md


def test_assemble_emits_knowledge_graph_when_section_supplied() -> None:
    """The ``## Knowledge Graph`` section appears between Subsystems
    and Pending annotations when ``graphify_section`` is non-empty.
    Pins the contract that ``/loominit`` relies on to surface the
    pre-built graph + graphify tool usage hints to the agent on
    every turn."""
    index = LoomIndex(
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        repo_root="/r",
        git_commit=None,
        files=[_file("x.py")],
        symbols=[_sym("x.py", "X")],
        imports=[],
        decorators=[],
        entry_points=[],
        clusters=[
            Cluster(
                id="c", title="c", paths=["x.py"],
                centroid_symbols=[], centrality=0.0, hash_bucket="b",
            )
        ],
    )
    cluster_results = {
        "c": _ClusterAnnotationOutput(
            narrative="n", data_flow=[], conventions=[],
            symbol_purposes=[],
        )
    }
    section_body = (
        "Pre-built knowledge graph at `.loom/graphify/graph.json` "
        "(700 nodes, 1000 edges, 40 communities)."
    )
    md = _assemble_markdown(
        index=index,
        metadata={"name": "p"},
        overview=_ProjectOverviewOutput(overview="x", tech_stack=[]),
        cluster_results=cluster_results,
        graphify_section=section_body,
    )
    assert "## Knowledge Graph" in md
    assert section_body in md
    # Ordering: must sit between Subsystems and Pending annotations
    # — the assembler stitches it in that slot.
    assert (
        md.index("## Subsystems")
        < md.index("## Knowledge Graph")
        < md.index("## Pending annotations")
    )


def test_assemble_omits_knowledge_graph_by_default() -> None:
    """No ``graphify_section`` arg → no Knowledge Graph heading.
    Keeps LOOM.md unchanged for callers that don't have a graph
    built (or where the graphify build failed and the REPL skipped
    passing a section)."""
    index = LoomIndex(
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        repo_root="/r",
        git_commit=None,
        files=[_file("x.py")],
        symbols=[_sym("x.py", "X")],
        imports=[],
        decorators=[],
        entry_points=[],
        clusters=[
            Cluster(
                id="c", title="c", paths=["x.py"],
                centroid_symbols=[], centrality=0.0, hash_bucket="b",
            )
        ],
    )
    cluster_results = {
        "c": _ClusterAnnotationOutput(
            narrative="n", data_flow=[], conventions=[],
            symbol_purposes=[],
        )
    }
    md = _assemble_markdown(
        index=index,
        metadata={"name": "p"},
        overview=_ProjectOverviewOutput(overview="x", tech_stack=[]),
        cluster_results=cluster_results,
    )
    assert "## Knowledge Graph" not in md


def test_assemble_skips_empty_sections() -> None:
    """data_flow=[] / conventions=[] / no entry_points → the
    corresponding sections are omitted, not rendered empty. Keeps
    LOOM.md tight."""
    index = LoomIndex(
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        repo_root="/r",
        git_commit=None,
        files=[],
        symbols=[],
        imports=[],
        decorators=[],
        entry_points=[],
        clusters=[
            Cluster(
                id="c",
                title="c",
                paths=[],
                centroid_symbols=[],
                centrality=0.0,
                hash_bucket="b",
            )
        ],
    )
    cluster_results = {
        "c": _ClusterAnnotationOutput(
            narrative="narrative",
            data_flow=[],
            conventions=[],
            symbol_purposes=[],
        )
    }
    md = _assemble_markdown(
        index=index,
        metadata={"name": "p"},
        overview=_ProjectOverviewOutput(overview="x", tech_stack=[]),
        cluster_results=cluster_results,
    )
    assert "**Data flow:**" not in md
    assert "**Conventions:**" not in md
    assert "**Symbols:**" not in md
    assert "## Entry Points" not in md
    assert "## Tech Stack" not in md


# ---- empty repo placeholder -----------------------------------------


async def test_annotate_empty_repo_returns_placeholder() -> None:
    """No source files → annotator emits a minimal placeholder
    without calling the model. Important so /loominit on a fresh
    repo doesn't burn a model call."""
    empty = LoomIndex(
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        repo_root="/r",
        git_commit=None,
        files=[],
        symbols=[],
        imports=[],
        decorators=[],
        entry_points=[],
        clusters=[],
    )
    sm = ScriptedModel(turns=[])  # would error if called
    md = await annotate(empty, model=sm)
    assert "no source files indexed yet" in md


# ---- end-to-end with ScriptedModel ----------------------------------


async def test_annotate_end_to_end_with_scripted_model() -> None:
    """Drive the full pipeline against a canned ScriptedModel. We
    pre-script enough turns for: 1 project overview + 1 cluster
    annotation. The output must integrate both into one LOOM.md."""
    index = LoomIndex(
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        repo_root="/r",
        git_commit=None,
        files=[_file("pkg/a.py", api=True)],
        symbols=[
            _sym(
                "pkg/a.py",
                "Engine",
                line=10,
                api=True,
                callers=3,
                pr=0.5,
            )
        ],
        imports=[],
        decorators=[],
        entry_points=[],
        clusters=[
            Cluster(
                id="pkg",
                title="pkg",
                paths=["pkg/a.py"],
                centroid_symbols=[],
                centrality=0.0,
                hash_bucket="bucket1",
            )
        ],
    )

    project_payload = json.dumps(
        {
            "overview": "Mypkg is a small package.",
            "tech_stack": ["Python 3.11+"],
        }
    )
    cluster_payload = json.dumps(
        {
            "narrative": "Pkg runs engines.",
            "data_flow": ["start → Engine → done"],
            "conventions": ["one class per file"],
            "symbol_purposes": [
                {
                    "name": "Engine",
                    "path": "pkg/a.py",
                    "line": 10,
                    "purpose": "core engine",
                }
            ],
        }
    )
    sm = ScriptedModel(
        turns=[
            ScriptedTurn(text=project_payload),
            ScriptedTurn(text=cluster_payload),
        ]
    )

    md = await annotate(
        index,
        model=sm,
        project_metadata={"name": "mypkg"},
    )
    assert "# mypkg — LOOM.md" in md
    assert "Mypkg is a small package." in md
    assert "Python 3.11+" in md
    assert "### pkg" in md
    assert "Pkg runs engines." in md
    assert "core engine" in md
    assert "## Pending annotations" in md
