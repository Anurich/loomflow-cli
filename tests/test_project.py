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


def test_loom_md_does_not_short_circuit_claude_static_bake(
    tmp_path: Path,
) -> None:
    # As of loominit slice 3, LOOM.md is the codebase INDEX (large,
    # sectioned, retrieved per-turn via BM25 into a working block by
    # ``LoomRetriever``). It is deliberately NOT in the static-bake
    # candidate list — baking it here would double-ship every turn.
    # CLAUDE.md remains the house-rules static bake (small, every-
    # turn relevant). When BOTH exist, CLAUDE.md wins this path.
    (tmp_path / "LOOM.md").write_text("codebase index (per-turn)")
    (tmp_path / "CLAUDE.md").write_text("house rules (static)")
    proj = detect_project(tmp_path)
    assert proj.context_file == tmp_path / "CLAUDE.md"
    assert proj.context_text == "house rules (static)"


def test_loom_md_alone_is_not_baked_into_context_text(
    tmp_path: Path,
) -> None:
    """Only LOOM.md present → no static context, because LOOM.md
    flows through per-turn retrieval instead."""
    (tmp_path / "LOOM.md").write_text("codebase index")
    proj = detect_project(tmp_path)
    assert proj.context_file is None
    assert proj.context_text == ""


def test_large_context_file_is_truncated(tmp_path: Path) -> None:
    big = "x" * (_MAX_CONTEXT_CHARS + 500)
    (tmp_path / "AGENTS.md").write_text(big)
    proj = detect_project(tmp_path)
    assert len(proj.context_text) < len(big)
    assert "truncated" in proj.context_text
