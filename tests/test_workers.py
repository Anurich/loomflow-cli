"""Tests for the worker roster — the sole-writer invariant.

The roster's whole safety story rests on one rule: only ``coder``
writes. These tests pin that down so a future edit to
``workers.py`` can't quietly hand a read-only specialist a
``write`` tool.
"""

from __future__ import annotations

from loomflow import Agent

from loom_code.project import Project
from loom_code.workers import build_workers

_WRITE_TOOLS = {"write", "edit"}
_READ_KERNEL = {"read", "grep", "find", "ls"}


def _tool_names(agent: Agent) -> set[str]:
    return set(agent._tool_host._tools.keys())


def test_roster_has_four_named_workers(project: Project) -> None:
    workers = build_workers(project, model="echo")
    assert set(workers) == {"coder", "explorer", "auditor", "reviewer"}
    for w in workers.values():
        assert isinstance(w, Agent)


def test_coder_is_the_sole_writer(project: Project) -> None:
    workers = build_workers(project, model="echo")
    # coder has write + edit ...
    assert _WRITE_TOOLS <= _tool_names(workers["coder"])
    # ... and NO other worker has either.
    for name in ("explorer", "auditor", "reviewer"):
        assert not (_WRITE_TOOLS & _tool_names(workers[name])), name


def test_bash_only_where_it_earns_its_keep(project: Project) -> None:
    workers = build_workers(project, model="echo")
    # coder runs builds; reviewer runs the test suite.
    assert "bash" in _tool_names(workers["coder"])
    assert "bash" in _tool_names(workers["reviewer"])
    # pure investigators get no shell.
    assert "bash" not in _tool_names(workers["explorer"])
    assert "bash" not in _tool_names(workers["auditor"])


def test_every_worker_has_the_read_kernel(project: Project) -> None:
    for name, agent in build_workers(project, model="echo").items():
        assert _READ_KERNEL <= _tool_names(agent), name


def test_every_worker_has_web_fetch(project: Project) -> None:
    # web_fetch closes the URL-fetch gap. It's read-only by
    # construction (no disk write, no shell), so giving it to the
    # read-only specialists doesn't weaken the sole-writer
    # invariant pinned in test_coder_is_the_sole_writer.
    for name, agent in build_workers(project, model="echo").items():
        assert "web_fetch" in _tool_names(agent), name


def test_every_worker_persists_tool_transcripts(project: Project) -> None:
    # loomflow 0.10.15+ persists the per-delegation tool transcript
    # (read/edit/bash results) on each worker's Episode rows so the
    # NEXT delegation rehydrates them via session_messages() — the
    # worker quotes prior reads instead of re-running them. We pin
    # this so future edits to ``workers.py`` don't silently drop the
    # flag and regress to the pre-0.10.15 re-read pattern that
    # leaked tokens on every long session.
    for name, agent in build_workers(project, model="echo").items():
        assert agent._persist_tool_transcripts is True, name
