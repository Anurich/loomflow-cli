"""Tests for prompt assembly — coordinator vs coder, project context."""

from __future__ import annotations

from pathlib import Path

from loom_code.project import Project
from loom_code.prompts import (
    build_coder_prompt,
    build_coordinator_instructions,
)


def _proj(
    tmp_path: Path, *, is_git: bool = False, context: str = ""
) -> Project:
    return Project(
        root=tmp_path,
        is_git=is_git,
        context_file=(tmp_path / "CLAUDE.md") if context else None,
        context_text=context,
    )


def test_coordinator_names_every_worker(tmp_path: Path) -> None:
    instr = build_coordinator_instructions(_proj(tmp_path))
    for worker in ("coder", "explorer", "auditor", "reviewer"):
        assert worker in instr
    # The coordinator orchestrates — it is told NOT to write code.
    assert "do NOT write code" in instr


def test_coder_prompt_is_the_doer(tmp_path: Path) -> None:
    prompt = build_coder_prompt(_proj(tmp_path))
    assert "CODER" in prompt
    # gather -> act -> verify is the coder's loop.
    assert "gather" in prompt.lower()
    assert "verify" in prompt.lower()


def test_git_hint_present_when_git(tmp_path: Path) -> None:
    assert "git repository" in build_coder_prompt(
        _proj(tmp_path, is_git=True)
    )


def test_no_git_hint_when_not_git(tmp_path: Path) -> None:
    assert "not a git repository" in build_coder_prompt(
        _proj(tmp_path, is_git=False)
    )


def test_context_file_inlined_in_both_prompts(tmp_path: Path) -> None:
    marker = "BINDING-HOUSE-RULE-MARKER"
    proj = _proj(tmp_path, context=marker)
    assert marker in build_coordinator_instructions(proj)
    assert marker in build_coder_prompt(proj)
