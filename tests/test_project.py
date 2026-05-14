"""Tests for project detection — repo-root walk + context-file load."""

from __future__ import annotations

from pathlib import Path

from loom_code.project import _MAX_CONTEXT_CHARS, detect_project


def test_detect_finds_git_root_from_subdir(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    proj = detect_project(sub)
    assert proj.root == tmp_path
    assert proj.is_git is True


def test_detect_no_git_falls_back_to_start(tmp_path: Path) -> None:
    proj = detect_project(tmp_path)
    assert proj.root == tmp_path
    assert proj.is_git is False
    assert proj.context_file is None
    assert proj.context_text == ""


def test_detect_reads_context_file(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "CLAUDE.md").write_text("house rules here")
    proj = detect_project(tmp_path)
    assert proj.context_file == tmp_path / "CLAUDE.md"
    assert proj.context_text == "house rules here"


def test_context_file_priority_loom_beats_claude(tmp_path: Path) -> None:
    # LOOM.md is first in the priority list — it must win even
    # when a CLAUDE.md sits right beside it.
    (tmp_path / "LOOM.md").write_text("loom wins")
    (tmp_path / "CLAUDE.md").write_text("claude loses")
    proj = detect_project(tmp_path)
    assert proj.context_file == tmp_path / "LOOM.md"
    assert proj.context_text == "loom wins"


def test_large_context_file_is_truncated(tmp_path: Path) -> None:
    big = "x" * (_MAX_CONTEXT_CHARS + 500)
    (tmp_path / "AGENTS.md").write_text(big)
    proj = detect_project(tmp_path)
    assert len(proj.context_text) < len(big)
    assert "truncated" in proj.context_text
