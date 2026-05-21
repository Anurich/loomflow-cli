"""Tests for ``_git_ls_files`` — the graphify source-file discovery.

The contract (what the user actually wants from /loominit):
  - index ALL code files: tracked AND brand-new untracked
  - NEVER index loom-code's own generated artifacts: LOOM.md,
    .loom/, graphify-out/
  - respect .gitignore (don't index .venv / node_modules / etc.)
  - return None (→ caller falls back to graphify.collect_files)
    when the directory isn't a git repo
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from loom_code.skills.graphify import tools as graphify_tools


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
    )


def _init_repo(root: Path) -> None:
    _git(root, "init")
    _git(root, "config", "user.email", "t@t.com")
    _git(root, "config", "user.name", "t")


def _discover(root: Path) -> set[str]:
    files = asyncio.run(graphify_tools._git_ls_files(root))
    assert files is not None
    return {str(p.relative_to(root)) for p in files}


def test_includes_tracked_and_untracked_code(tmp_path: Path) -> None:
    """Both committed AND brand-new (un-added) source files appear.
    The untracked case is the headline fix — new files shouldn't
    be invisible to the graph until git add."""
    _init_repo(tmp_path)
    (tmp_path / "tracked.py").write_text("x = 1\n")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-m", "init")
    # Brand-new, never added.
    (tmp_path / "untracked.py").write_text("y = 2\n")

    discovered = _discover(tmp_path)
    assert "tracked.py" in discovered
    assert "untracked.py" in discovered, (
        "untracked new source file missing — the whole point of "
        "the --others fix"
    )


def test_excludes_loom_md(tmp_path: Path) -> None:
    """LOOM.md is loominit's OWN output — indexing it is circular.
    Excluded even though it's an untracked .md (which the suffix
    filter would otherwise accept)."""
    _init_repo(tmp_path)
    (tmp_path / "real.py").write_text("x = 1\n")
    (tmp_path / "LOOM.md").write_text("# project\n## Overview\nstuff\n")

    discovered = _discover(tmp_path)
    assert "real.py" in discovered
    assert "LOOM.md" not in discovered


def test_excludes_dot_loom_dir(tmp_path: Path) -> None:
    """.loom/ is generated state (memory.db, graph.json, notebook).
    None of it belongs in the knowledge graph."""
    _init_repo(tmp_path)
    (tmp_path / "real.py").write_text("x = 1\n")
    loom = tmp_path / ".loom" / "graphify"
    loom.mkdir(parents=True)
    (loom / "graph.json").write_text("{}\n")
    (tmp_path / ".loom" / "notebook").mkdir()
    (tmp_path / ".loom" / "notebook" / "note.md").write_text("# n\n")

    discovered = _discover(tmp_path)
    assert "real.py" in discovered
    assert not any(d.startswith(".loom/") for d in discovered)


def test_excludes_graphify_out_cache(tmp_path: Path) -> None:
    """graphify-out/ is graphify's AST cache — derived JSON, not
    source."""
    _init_repo(tmp_path)
    (tmp_path / "real.py").write_text("x = 1\n")
    cache = tmp_path / "graphify-out" / "cache" / "ast"
    cache.mkdir(parents=True)
    (cache / "abc123.json").write_text("{}\n")

    discovered = _discover(tmp_path)
    assert "real.py" in discovered
    assert not any(d.startswith("graphify-out/") for d in discovered)


def test_respects_gitignore(tmp_path: Path) -> None:
    """A gitignored dir (.venv) must NOT be indexed — git's
    --exclude-standard handles this, we just need to not undo it."""
    _init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text(".venv/\n")
    (tmp_path / "real.py").write_text("x = 1\n")
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "junk.py").write_text("import junk\n")

    discovered = _discover(tmp_path)
    assert "real.py" in discovered
    assert not any(".venv" in d for d in discovered)


def test_non_supported_extensions_dropped(tmp_path: Path) -> None:
    """Files with no tree-sitter extractor (e.g. .png, .lock)
    don't appear — graphify can't parse them anyway."""
    _init_repo(tmp_path)
    (tmp_path / "real.py").write_text("x = 1\n")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "poetry.lock").write_text("[[package]]\n")

    discovered = _discover(tmp_path)
    assert "real.py" in discovered
    assert "image.png" not in discovered
    assert "poetry.lock" not in discovered


def test_returns_none_for_non_git_dir(tmp_path: Path) -> None:
    """A non-git directory yields None so graphify_build_impl
    falls back to graphify.collect_files."""
    (tmp_path / "real.py").write_text("x = 1\n")
    result = asyncio.run(graphify_tools._git_ls_files(tmp_path))
    assert result is None


def test_is_loom_own_artifact_helper() -> None:
    """Unit-pin the path classifier so the exclusion set is
    explicit + testable."""
    f = graphify_tools._is_loom_own_artifact
    assert f("LOOM.md") is True
    assert f(".loom/graphify/graph.json") is True
    assert f(".loom/notebook/note.md") is True
    assert f("graphify-out/cache/ast/x.json") is True
    # Real source must NOT be flagged.
    assert f("loomflow_ide/app.py") is False
    assert f("README.md") is False
    # A file merely CONTAINING 'loom' in the name is fine.
    assert f("loom_helpers.py") is False
