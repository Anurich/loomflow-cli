"""Adaptive routing — the solo fast path.

Two halves pinned here:

* ``build_solo_agent`` — a standalone coder kernel that shares the
  team's memory db + notebook, so a solo turn and a team turn see the
  same history.
* ``Repl._route_turn`` — conservative routing: every uncertain branch
  lands on "team"; only a confident classifier SOLO takes the fast
  path. A misroute can cost the status-quo delegation overhead, never
  a capability.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from loomflow import Agent

from loom_code.agent import build_solo_agent
from loom_code.project import Project
from loom_code.repl import (
    Repl,
    _looks_like_question,
    _references_prior_context,
)

pytestmark = pytest.mark.anyio

_WRITER_KERNEL = {
    "read", "write", "edit", "multi_edit", "grep", "find", "ls", "bash",
}


def _tool_names(agent: Agent) -> set[str]:
    return set(agent._tool_host._tools.keys())


# ---------------------------------------------------------------------------
# build_solo_agent
# ---------------------------------------------------------------------------


def test_solo_agent_has_full_writer_kernel(project: Project) -> None:
    agent = build_solo_agent(project, model="echo")
    assert _WRITER_KERNEL <= _tool_names(agent)


def test_solo_agent_shares_team_memory_db(project: Project) -> None:
    """Same ``.loom/memory.db`` as the team — context is continuous
    across routes."""
    agent = build_solo_agent(project, model="echo")
    from loomflow.memory.sqlite import SqliteMemory

    assert isinstance(agent._memory, SqliteMemory)


def test_solo_agent_has_notebook_tools(project: Project) -> None:
    """Standalone runs have no parent to inherit the workspace from —
    the notebook tools must be wired explicitly."""
    agent = build_solo_agent(project, model="echo")
    names = _tool_names(agent)
    assert any("note" in n for n in names), names


# ---------------------------------------------------------------------------
# _looks_like_question — only ever short-circuits TOWARD the team
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "what does the retry decorator do?",
        "how is auth wired",
        "explain the worker roster",
        "is the cache enabled in prod?",
        "anything ending in a question mark?",
    ],
)
def test_questions_detected(prompt: str) -> None:
    assert _looks_like_question(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "fix the typo in README.md",
        "add a retry decorator to the http client",
        "rename foo to bar in utils.py",
        "bump the version to 0.2.0",
    ],
)
def test_change_requests_not_questions(prompt: str) -> None:
    assert not _looks_like_question(prompt)


# ---------------------------------------------------------------------------
# _references_prior_context — anaphora routes to the team, which
# holds the session history the stateless classifier can't see
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "fix it",
        "continue",
        "try that again",
        "do this now please",
        "ok fix them all",
    ],
)
def test_anaphoric_prompts_detected(prompt: str) -> None:
    assert _references_prior_context(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "fix the typo in README.md",
        "rename foo to bar in utils.py",
        # Long prompts may use "it" self-referentially — the
        # classifier (plus its own anaphora instruction) handles
        # those; the heuristic must not blanket-team them.
        "add a cache to the parser and make sure it invalidates"
        " on file change",
    ],
)
def test_self_contained_prompts_pass_through(prompt: str) -> None:
    assert not _references_prior_context(prompt)


async def test_anaphora_skips_the_classifier() -> None:
    fake = _fake_repl(classifier_must_not_run=True)
    assert await Repl._route_turn(fake, "fix it") == "team"


# ---------------------------------------------------------------------------
# Repl._route_turn — exercised unbound with a stub self
# ---------------------------------------------------------------------------


def _fake_repl(
    *,
    run_until=None,
    browser=False,
    classify="TEAM",
    classifier_must_not_run=False,
):
    async def _classify_task(prompt: str) -> str:
        if classifier_must_not_run:
            raise AssertionError("classifier should not have been called")
        return classify

    return SimpleNamespace(
        _run_until=run_until,
        _browser_mode=browser,
        _classify_task=_classify_task,
    )


async def test_goal_mode_always_team() -> None:
    fake = _fake_repl(
        run_until={"condition": "tests pass"},
        classifier_must_not_run=True,
    )
    assert await Repl._route_turn(fake, "fix the typo") == "team"


async def test_operator_mode_always_team() -> None:
    fake = _fake_repl(browser=True, classifier_must_not_run=True)
    assert await Repl._route_turn(fake, "fix the typo") == "team"


async def test_questions_skip_the_classifier() -> None:
    fake = _fake_repl(classifier_must_not_run=True)
    assert (
        await Repl._route_turn(fake, "how does auth work?") == "team"
    )


async def test_confident_solo_takes_fast_path() -> None:
    fake = _fake_repl(classify="SOLO")
    assert await Repl._route_turn(fake, "fix the typo") == "solo"


async def test_team_vote_stays_team() -> None:
    fake = _fake_repl(classify="TEAM")
    assert (
        await Repl._route_turn(fake, "refactor the auth module")
        == "team"
    )
