"""Tests for build_agent — the Team.supervisor wiring."""

from __future__ import annotations

from loomflow import Agent
from loomflow.architecture import Supervisor
from loomflow.workspace import LocalDiskWorkspace

from loom_code.agent import LOOM_DIR, build_agent
from loom_code.project import Project


def test_build_agent_returns_supervisor_coordinator(
    project: Project,
) -> None:
    coordinator, _workspace = build_agent(project, model="echo")
    assert isinstance(coordinator, Agent)
    assert isinstance(coordinator.architecture, Supervisor)


def test_coordinator_declares_the_four_workers(
    project: Project,
) -> None:
    coordinator, _ = build_agent(project, model="echo")
    workers = coordinator.architecture.declared_workers()
    assert set(workers) == {"coder", "explorer", "auditor", "reviewer"}


def test_build_agent_creates_loom_dir(project: Project) -> None:
    assert not (project.root / LOOM_DIR).exists()
    build_agent(project, model="echo")
    assert (project.root / LOOM_DIR).is_dir()


def test_workspace_rooted_under_loom_dir(project: Project) -> None:
    _, workspace = build_agent(project, model="echo")
    assert isinstance(workspace, LocalDiskWorkspace)
    # the notebook lives at <root>/.loom/notebook
    expected = project.root / LOOM_DIR / "notebook"
    assert str(expected) in str(workspace.root)


def test_coordinator_has_memory_wired(project: Project) -> None:
    coordinator, _ = build_agent(project, model="echo")
    assert coordinator.memory is not None
