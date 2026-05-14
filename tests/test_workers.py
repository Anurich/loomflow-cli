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
