"""Tests for prompt assembly — coordinator vs coder, project context."""

from __future__ import annotations

from pathlib import Path

from loom_code.project import Project
from loom_code.prompts import (
    build_coder_prompt,
    build_unified_coordinator_instructions,
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
    instr = build_unified_coordinator_instructions(_proj(tmp_path))
    for worker in ("coder", "explorer", "auditor", "reviewer"):
        assert worker in instr
    # The unified coordinator does focused work ITSELF and delegates
    # multi-file / parallel work — both capabilities must be present.
    assert "delegate" in instr.lower()
    assert "yourself" in instr.lower()


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


def test_context_file_reaches_both_agents(tmp_path: Path) -> None:
    """The coder inlines the context file statically; the coordinator
    deliberately does NOT (``include_context_file=False``) — it gets
    the rules FRESH each turn via the ``project_rules`` working block
    instead, so mid-session edits apply without a restart. Pin both
    halves of that contract."""
    from loom_code.rules import project_rules_block

    marker = "BINDING-HOUSE-RULE-MARKER"
    proj = _proj(tmp_path, context=marker)
    # Coder: static bake.
    assert marker in build_coder_prompt(proj)
    # Coordinator: no static bake...
    assert marker not in build_unified_coordinator_instructions(proj)
    # ...because the per-turn working block carries it. AGENTS.md is
    # loom-code's native convention (what /init-loom creates); CLAUDE.md
    # is only read for compatibility with repos that already have one.
    (tmp_path / "AGENTS.md").write_text(marker, encoding="utf-8")
    assert marker in project_rules_block(tmp_path)


def test_coordinator_instructs_use_of_repo_map(
    tmp_path: Path,
) -> None:
    """The coordinator must be told to use the repo map (injected
    into the system prompt via the ``loom_index`` working block) when
    answering project-level questions, instead of asking the user to
    specify a file.

    Observed failure mode in production: 'what is this code about?' →
    'Please specify the file or snippet of code you want me to check'
    — despite the repo map being right there in context. Pin the
    language so a future prompt rewrite can't silently drop the
    connection between project-level prompts and the repo map."""
    proj = _proj(tmp_path)
    prompt = build_unified_coordinator_instructions(proj)
    assert "repo map" in prompt.lower()
    # Must explicitly forbid asking the user when the map exists.
    assert "DO NOT ask the user to specify a file" in prompt


def test_coordinator_forces_regrounding_on_action_prompts(
    tmp_path: Path,
) -> None:
    """The coordinator must instruct the model to re-read file
    state before claiming work is done. This is what stops the
    "parrot prior session's lie" failure mode (observed in
    production: the agent answered 'all 12 issues fixed' with
    zero tool calls because episode recall surfaced a prior
    session's hallucinated 'all done' claim).

    Without this directive being load-bearing in the system prompt,
    the bug recurs every time stale completion claims land in
    memory recall. Pin the language so a future prompt rewrite
    can't silently drop it."""
    proj = _proj(tmp_path)
    prompt = build_unified_coordinator_instructions(proj)
    # The directive heading itself.
    assert "GROUND CLAIMS IN CURRENT FILE STATE" in prompt
    # The behavioural rule: re-read on action verbs.
    assert "fix / check / verify" in prompt
    # The reason: explicit naming of the failure mode so the
    # model understands WHY (LLMs follow directives better when
    # the rationale is visible).
    assert "stale completion claims" in prompt
    assert "Trust file contents, not memory" in prompt
