"""Guards on the repo-map walk (loom_code.loominit.repomap).

The signature walk runs at EVERY turn start — it must never stall a
turn. Regression: a Windows user launched loom-code at ``D:\\`` (drive
root, no git → cwd becomes the project root) and the first turn hung
walking the entire drive.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loom_code.loominit import repomap as rm


@pytest.fixture(autouse=True)
def _clean_cache():
    rm._REPO_MAP_CACHE.clear()
    yield
    rm._REPO_MAP_CACHE.clear()


def test_drive_root_is_unmappable() -> None:
    anchor = Path(Path.cwd().anchor)  # "/" on POSIX, "C:\\" on Windows
    assert rm._unmappable_root(anchor)
    assert rm.repo_map_for_root_cached(anchor) is None
    # ...and the refusal is CACHED (no walk on later turns)
    key = str(anchor.resolve())
    assert rm._REPO_MAP_CACHE[key][0] is rm._UNMAPPABLE


def test_home_dir_is_unmappable() -> None:
    assert rm._unmappable_root(Path.home())
    assert rm.repo_map_for_root_cached(Path.home()) is None


def test_normal_project_still_maps(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("def hello():\n    return 1\n")
    assert not rm._unmappable_root(tmp_path)
    body = rm.repo_map_for_root_cached(tmp_path)
    assert body and "hello" in body


def test_budget_breach_refuses_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tree that blows the walk budget yields no map — and the
    refusal is cached so the walk never re-runs for that root."""
    (tmp_path / "a.py").write_text("x = 1\n")
    monkeypatch.setattr(rm, "_SIG_MAX_FILES", 0)  # everything breaches
    assert rm.repo_map_for_root_cached(tmp_path) is None
    key = str(tmp_path.resolve())
    assert rm._REPO_MAP_CACHE[key][0] is rm._UNMAPPABLE
    # Second call: served from the refusal cache (walk not re-run) —
    # raise if the walk executes again.
    monkeypatch.setattr(
        rm, "_tree_signature",
        lambda _r: (_ for _ in ()).throw(AssertionError("re-walked")),
    )
    assert rm.repo_map_for_root_cached(tmp_path) is None


def test_time_budget_bounds_the_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    monkeypatch.setattr(rm, "_SIG_TIME_BUDGET_S", -1.0)  # instant breach
    assert rm._tree_signature(tmp_path) is None
