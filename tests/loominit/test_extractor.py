"""End-to-end tests for the structural extractor.

We build a tiny fixture repo on a tmp_path and run
:func:`build_index` against it. The fixture is deliberately
diverse — multiple packages, an ``__all__``-shaped API, relative
imports, an entry point in pyproject, a ``__main__`` block, a
landmark decorator, and one test file — so a single golden run
exercises every aggregation step.

The tests don't snapshot the entire LoomIndex (would be brittle);
they assert specific invariants on each section.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from loom_code.loominit.extractor import build_index
from loom_code.loominit.schema import LoomIndex


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    """A small but realistic Python project layout.

    Tree::

        tmp_path/
          pyproject.toml           [project.scripts]
          mypkg/
            __init__.py            re-exports Engine
            cli.py                 main block + @click.command
            engine.py              the Engine class
            utils/
              __init__.py          empty
              math.py              add(), MAX_ITER constant
          tests/
            test_engine.py         exercises Engine
    """
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "mypkg"
            version = "0.1.0"

            [project.scripts]
            mycli = "mypkg.cli:main"
            """
        ),
        encoding="utf-8",
    )

    (tmp_path / "mypkg").mkdir()
    (tmp_path / "mypkg" / "__init__.py").write_text(
        'from .engine import Engine\n\n__all__ = ["Engine"]\n',
        encoding="utf-8",
    )
    (tmp_path / "mypkg" / "cli.py").write_text(
        textwrap.dedent(
            '''
            """CLI entrypoint."""
            import click
            from .engine import Engine

            @click.command
            def main():
                """Run the thing."""
                Engine().go()

            if __name__ == "__main__":
                main()
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "mypkg" / "engine.py").write_text(
        textwrap.dedent(
            '''
            """The core engine."""
            from .utils.math import MAX_ITER, add


            class Engine:
                """Does engine things."""

                def go(self) -> int:
                    return add(MAX_ITER, 1)
            '''
        ).lstrip(),
        encoding="utf-8",
    )

    (tmp_path / "mypkg" / "utils").mkdir()
    (tmp_path / "mypkg" / "utils" / "__init__.py").write_text(
        "", encoding="utf-8"
    )
    (tmp_path / "mypkg" / "utils" / "math.py").write_text(
        textwrap.dedent(
            '''
            """Math helpers."""
            MAX_ITER = 100


            def add(a: int, b: int) -> int:
                """Return a + b."""
                return a + b
            '''
        ).lstrip(),
        encoding="utf-8",
    )

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_engine.py").write_text(
        textwrap.dedent(
            """
            from mypkg.engine import Engine


            def test_engine_go():
                assert Engine().go() == 101
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return tmp_path


def test_build_index_returns_loom_index(fixture_repo: Path) -> None:
    """The basic contract — :func:`build_index` returns a valid
    :class:`LoomIndex` for any readable directory."""
    idx = build_index(fixture_repo)
    assert isinstance(idx, LoomIndex)


def test_files_section_populated(fixture_repo: Path) -> None:
    """Every Python file in the fixture must appear in ``files``."""
    idx = build_index(fixture_repo)
    paths = {f.path for f in idx.files}
    assert "pyproject.toml" in paths
    assert "mypkg/__init__.py" in paths
    assert "mypkg/cli.py" in paths
    assert "mypkg/engine.py" in paths
    assert "mypkg/utils/math.py" in paths
    assert "tests/test_engine.py" in paths


def test_files_have_content_hash(fixture_repo: Path) -> None:
    """Every file entry must have a non-empty sha256 — staleness
    detection depends on it being populated, not just structural."""
    idx = build_index(fixture_repo)
    for f in idx.files:
        assert f.sha256
        assert len(f.sha256) == 64  # hex sha256


def test_test_files_flagged(fixture_repo: Path) -> None:
    """``tests/test_engine.py`` must be ``is_test=True``."""
    idx = build_index(fixture_repo)
    by_path = {f.path: f for f in idx.files}
    assert by_path["tests/test_engine.py"].is_test is True
    assert by_path["mypkg/engine.py"].is_test is False


def test_symbols_extracted(fixture_repo: Path) -> None:
    """Every top-level def / class / constant in the fixture must
    appear as a symbol."""
    idx = build_index(fixture_repo)
    by_id = {s.id: s for s in idx.symbols}
    assert "mypkg/engine.py:Engine" in by_id
    assert "mypkg/engine.py:Engine.go" in by_id
    assert "mypkg/cli.py:main" in by_id
    assert "mypkg/utils/math.py:add" in by_id
    assert "mypkg/utils/math.py:MAX_ITER" in by_id


def test_method_kind_assignment(fixture_repo: Path) -> None:
    """A function defined inside a class must be ``kind="method"``,
    NOT ``"function"``. Annotator uses this to render methods under
    their class."""
    idx = build_index(fixture_repo)
    by_id = {s.id: s for s in idx.symbols}
    assert by_id["mypkg/engine.py:Engine.go"].kind == "method"
    assert by_id["mypkg/cli.py:main"].kind == "function"


def test_api_surface_detection(fixture_repo: Path) -> None:
    """``mypkg/__init__.py`` does ``from .engine import Engine`` and
    declares ``__all__ = ["Engine"]``. So ``mypkg/engine.py`` must
    be on the API surface; ``mypkg/utils/math.py`` must NOT."""
    idx = build_index(fixture_repo)
    by_path = {f.path: f for f in idx.files}
    assert by_path["mypkg/engine.py"].in_api_surface is True
    assert by_path["mypkg/utils/math.py"].in_api_surface is False


def test_symbol_api_surface_inherits_file(fixture_repo: Path) -> None:
    """Public symbols in API-surface files inherit the flag."""
    idx = build_index(fixture_repo)
    by_id = {s.id: s for s in idx.symbols}
    assert by_id["mypkg/engine.py:Engine"].in_api_surface is True
    assert by_id["mypkg/utils/math.py:add"].in_api_surface is False


def test_imports_resolved_against_repo(fixture_repo: Path) -> None:
    """``from .engine import Engine`` in ``mypkg/__init__.py`` should
    resolve (``resolved=True``); ``import click`` should not
    (it's a third-party module not in the fixture)."""
    idx = build_index(fixture_repo)
    edges_by_pair = {(e.from_path, e.to_module): e for e in idx.imports}
    # Relative import in __init__ — to_module display form is ".engine"
    assert (
        "mypkg/__init__.py",
        ".engine",
    ) in edges_by_pair
    edge = edges_by_pair[("mypkg/__init__.py", ".engine")]
    assert edge.resolved is True

    # Third-party import — click is not in the fixture
    assert ("mypkg/cli.py", "click") in edges_by_pair
    assert edges_by_pair[("mypkg/cli.py", "click")].resolved is False


def test_decorator_landmark_captured(fixture_repo: Path) -> None:
    """``@click.command`` on ``main`` must be in :attr:`decorators`."""
    idx = build_index(fixture_repo)
    matched = [
        d
        for d in idx.decorators
        if d.decorator == "click.command"
        and d.target == "mypkg/cli.py:main"
    ]
    assert len(matched) == 1


def test_pyproject_entry_point_extracted(fixture_repo: Path) -> None:
    """``[project.scripts] mycli = "mypkg.cli:main"`` must surface."""
    idx = build_index(fixture_repo)
    by_kind = [ep for ep in idx.entry_points if ep.kind == "pyproject_script"]
    assert len(by_kind) == 1
    assert by_kind[0].name == "mycli"
    assert by_kind[0].callable_id == "mypkg/cli.py:main"


def test_main_block_entry_point_extracted(fixture_repo: Path) -> None:
    """``if __name__ == "__main__":`` in cli.py must surface."""
    idx = build_index(fixture_repo)
    mains = [ep for ep in idx.entry_points if ep.kind == "main_block"]
    assert len(mains) == 1
    assert mains[0].path == "mypkg/cli.py"


def test_pagerank_assigns_nonzero_scores(fixture_repo: Path) -> None:
    """At least one symbol should have a non-trivial PageRank score.
    All-zero PageRank means the graph code never ran or was
    catastrophically broken."""
    idx = build_index(fixture_repo)
    py_symbols = [
        s
        for s in idx.symbols
        if s.kind in ("class", "function", "method")
    ]
    assert any(s.pagerank > 0 for s in py_symbols)


def test_test_to_symbol_map_finds_engine(fixture_repo: Path) -> None:
    """``tests/test_engine.py`` references ``Engine`` — the test
    map must list it."""
    idx = build_index(fixture_repo)
    engine_sym = next(
        s for s in idx.symbols if s.id == "mypkg/engine.py:Engine"
    )
    assert any(
        "tests/test_engine.py" in citation for citation in engine_sym.tests
    )


def test_clusters_built_by_path_prefix(fixture_repo: Path) -> None:
    """``mypkg/`` files should cluster together; ``tests/`` separately."""
    idx = build_index(fixture_repo)
    cluster_paths = {c.id: c.paths for c in idx.clusters}
    # Cluster IDs are dotted ("mypkg" for "mypkg/" dir, "tests" for "tests/")
    assert "mypkg" in cluster_paths
    assert "tests" in cluster_paths
    assert "mypkg/engine.py" in cluster_paths["mypkg"]
    assert "tests/test_engine.py" in cluster_paths["tests"]


def test_clusters_have_hash_buckets(fixture_repo: Path) -> None:
    """Each cluster has a hash bucket over its files' content
    hashes — used for diff-aware refresh."""
    idx = build_index(fixture_repo)
    for c in idx.clusters:
        assert c.hash_bucket
        assert len(c.hash_bucket) == 16  # truncated sha256


def test_build_index_is_deterministic(fixture_repo: Path) -> None:
    """Running build_index twice on the same tree must produce
    equal indexes (up to the timestamp field). Critical for diff-
    aware refresh — non-deterministic ordering would corrupt
    bucket hashes."""
    a = build_index(fixture_repo)
    b = build_index(fixture_repo)
    # generated_at and git_commit can differ; compare structural lists
    assert [f.path for f in a.files] == [f.path for f in b.files]
    assert [s.id for s in a.symbols] == [s.id for s in b.symbols]
    assert [c.hash_bucket for c in a.clusters] == [
        c.hash_bucket for c in b.clusters
    ]


def test_empty_repo_returns_empty_index(tmp_path: Path) -> None:
    """Building against a directory with no source files must NOT
    crash — returns an index with empty lists."""
    idx = build_index(tmp_path)
    assert idx.files == []
    assert idx.symbols == []
    assert idx.imports == []
    assert idx.clusters == []


def test_nonexistent_root_returns_empty_index(tmp_path: Path) -> None:
    """Pointing at a nonexistent path should not raise — useful for
    the REPL when the user CDs to something gone."""
    idx = build_index(tmp_path / "does-not-exist")
    assert idx.files == []
