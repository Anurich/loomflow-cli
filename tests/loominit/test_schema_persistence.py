"""Schema round-trip + persistence smoke tests.

The schema and the on-disk format are the contract every other
loominit module trusts. Pin them with cheap deterministic tests so
any future ``model_dump`` / Pydantic-version drift surfaces here
instead of breaking the annotator at runtime.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from loom_code.loominit.persistence import (
    IndexVersionMismatch,
    index_path,
    load_index,
    loom_dir,
    markdown_path,
    read_markdown,
    save_index,
    write_markdown,
)
from loom_code.loominit.schema import (
    SCHEMA_VERSION,
    CallEdge,
    Cluster,
    DecoratorLandmark,
    EntryPoint,
    FileEntry,
    ImportEdge,
    LoomIndex,
    SymbolEntry,
)


def _minimal_index() -> LoomIndex:
    """An index with one of every entity — enough to exercise
    serialisation of every Pydantic model in the schema."""
    now = datetime(2026, 5, 15, 17, 0, tzinfo=UTC)
    return LoomIndex(
        generated_at=now,
        repo_root="/repo",
        git_commit="abc123",
        files=[
            FileEntry(
                path="src/a.py",
                lang="python",
                size_bytes=42,
                lines=3,
                sha256="deadbeef",
                mtime=now,
                git_changes_90d=2,
                is_test=False,
                in_api_surface=True,
            )
        ],
        symbols=[
            SymbolEntry(
                id="src/a.py:foo",
                name="foo",
                qualified_name="foo",
                kind="function",
                path="src/a.py",
                line=1,
                end_line=2,
                signature="def foo() -> int:",
                docstring_first_line="A function.",
                decorators=[],
                is_public=True,
                in_api_surface=True,
                pagerank=0.5,
                n_callers=1,
                n_callees=0,
                tests=["tests/test_a.py:test_foo"],
            )
        ],
        imports=[
            ImportEdge(
                from_path="src/a.py",
                to_module="src.b",
                line=1,
                resolved=True,
            )
        ],
        calls=[
            CallEdge(caller="src/a.py:foo", callee="src/b.py:bar", line=3)
        ],
        decorators=[
            DecoratorLandmark(
                decorator="@click.command",
                target="src/a.py:foo",
                path="src/a.py",
                line=1,
            )
        ],
        entry_points=[
            EntryPoint(
                kind="pyproject_script",
                name="my-cli",
                path="pyproject.toml",
                line=None,
                callable_id="src/a.py:foo",
            )
        ],
        clusters=[
            Cluster(
                id="src",
                title="src/",
                paths=["src/a.py"],
                centroid_symbols=["src/a.py:foo"],
                centrality=0.5,
                hash_bucket="bucket1",
            )
        ],
    )


def test_loom_index_round_trips_through_json() -> None:
    """``model_dump(mode='json')`` then ``model_validate`` must
    reproduce the same object. Catches any Pydantic field that
    silently drops on serialise (e.g. computed_field misuse)."""
    idx = _minimal_index()
    dumped = idx.model_dump(mode="json")
    restored = LoomIndex.model_validate(dumped)
    assert restored == idx


def test_save_and_load_index(tmp_path: Path) -> None:
    """save_index → load_index returns an equal index, and the
    file lives at the documented path."""
    idx = _minimal_index()
    save_index(tmp_path, idx)
    assert index_path(tmp_path).exists()
    loaded = load_index(tmp_path)
    assert loaded == idx


def test_load_index_returns_none_when_absent(tmp_path: Path) -> None:
    """Missing index.json is the "never ran /loominit" state. Must
    return None (not raise) so callers can branch on first-run."""
    assert load_index(tmp_path) is None


def test_load_index_raises_on_version_mismatch(tmp_path: Path) -> None:
    """An on-disk index from a different schema version must NOT be
    silently consumed — the annotation logic depends on current
    schema semantics. We raise so the REPL can nudge a rebuild."""
    p = index_path(tmp_path)
    p.write_text(
        '{"version": 999, "generated_at": "2026-05-15T00:00:00+00:00", '
        '"repo_root": "/x", "git_commit": null}',
        encoding="utf-8",
    )
    with pytest.raises(IndexVersionMismatch):
        load_index(tmp_path)


def test_save_index_is_atomic(tmp_path: Path) -> None:
    """A crash mid-write must not leave a half-written index.json.
    We can't easily simulate a crash, but we CAN assert there's
    no stale temp file after a clean save (atomic write cleans up
    on success)."""
    idx = _minimal_index()
    save_index(tmp_path, idx)
    leftover = list((tmp_path / ".loom").glob(".index-*.tmp"))
    assert leftover == []


def test_markdown_write_and_read(tmp_path: Path) -> None:
    """LOOM.md lives at the repo root, not under .loom/. Round-trip
    proves the path is right and the read/write pair is symmetric."""
    body = "# LOOM.md\n\nproject overview\n"
    write_markdown(tmp_path, body)
    assert markdown_path(tmp_path) == tmp_path / "LOOM.md"
    assert read_markdown(tmp_path) == body


def test_read_markdown_returns_none_when_absent(tmp_path: Path) -> None:
    """None (not "") so first-run vs. emptied-file are
    distinguishable upstream."""
    assert read_markdown(tmp_path) is None


def test_loom_dir_creates_on_first_call(tmp_path: Path) -> None:
    """loom_dir is the canonical way to get .loom/, including the
    side effect of mkdir. No other module should call mkdir for it
    — concentrating that behaviour here keeps the layout in one
    place."""
    d = loom_dir(tmp_path)
    assert d == tmp_path / ".loom"
    assert d.is_dir()


def test_schema_version_constant_matches_default() -> None:
    """The model default for ``version`` must equal SCHEMA_VERSION.
    A drift between these two is the kind of bug that would break
    every newly-written index without surfacing in tests."""
    idx = LoomIndex(
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        repo_root="/x",
        git_commit=None,
    )
    assert idx.version == SCHEMA_VERSION
