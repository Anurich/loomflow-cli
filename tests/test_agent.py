"""Tests for build_agent — the unified Team.supervisor coordinator.

``build_agent`` returns a single ``Team.supervisor`` Agent whose
coordinator holds the full coding kernel AND a ``delegate`` tool. It
decides inline whether to do focused work itself or delegate
multi-file / parallel work to the worker roster (coder / explorer /
auditor / reviewer).
"""

from __future__ import annotations

from typing import Any

from loomflow import Agent
from loomflow.architecture import Supervisor
from loomflow.workspace import LocalDiskWorkspace

from loom_code.agent import LOOM_DIR, build_agent
from loom_code.project import Project


def _tool_names(agent: Agent) -> set[str]:
    """The agent's STATIC tool surface by name. (``delegate`` is
    injected by the Supervisor architecture at run-time, so it is
    intentionally NOT here.)"""
    host: Any = agent._tool_host
    tools = getattr(host, "_tools", None)
    return set(tools.keys()) if tools is not None else set()


def test_build_agent_returns_supervisor_coordinator(
    project: Project,
) -> None:
    # build_agent returns a Team.supervisor Agent — the coordinator
    # decides inline whether to act itself or delegate.
    coordinator, _workspace = build_agent(project, model="echo")
    assert isinstance(coordinator, Agent)
    assert isinstance(coordinator.architecture, Supervisor)


def test_coordinator_has_full_worker_roster(project: Project) -> None:
    # The four delegate workers must be wired; regressing this loses
    # the team-pattern capability the coordinator delegates into.
    coordinator, _ = build_agent(project, model="echo")
    workers = coordinator.architecture.declared_workers()
    assert set(workers) == {"coder", "explorer", "auditor", "reviewer"}


def test_coordinator_is_read_only(project: Project) -> None:
    # The coordinator is a READ-ONLY supervisor: it has read/grep/ls/
    # find to understand + answer, but must NOT carry write/edit/bash —
    # so it can't grind edits itself and MUST delegate every change to
    # `coder`. Regressing this (handing it the writer kernel) brings
    # back the "coordinator does everything itself, workers idle" bug.
    coordinator, _ = build_agent(project, model="echo")
    names = _tool_names(coordinator)
    assert {"read", "grep", "ls", "find"} <= names, (
        f"coordinator missing read-only tools: "
        f"{ {'read','grep','ls','find'} - names }"
    )
    writers = {"write", "edit", "multi_edit", "bash"}
    assert not (writers & names), (
        f"coordinator must NOT hold writer/exec tools (found "
        f"{writers & names}) — those belong to the coder worker"
    )


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


def test_no_auto_extract_override_band_aid_in_source() -> None:
    """Pre-loomflow 0.10.20 we explicitly disabled auto_extract
    everywhere as a band-aid against the per-turn LLM fact-
    extraction latency (3-10s of invisible wait between visible
    response and next ``loom:`` prompt). Since 0.10.20,
    ``AutoExtractMemory`` runs extraction as a fire-and-forget
    background task — the band-aid is unnecessary and should
    not exist in our source.

    If ``auto_extract=False`` creeps back into loom-code, fact
    extraction silently turns off for the team — losing the
    "remembers user preferences across sessions" feature that
    the framework now provides for free.

    This is a SOURCE-LEVEL check rather than a runtime assertion
    because the runtime wrapping depends on the model (echo
    defaults differently than real providers); pinning the
    source guarantees we're not papering over the framework's
    correct behaviour.
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    for rel in ("loom_code/agent.py", "loom_code/workers.py"):
        text = (repo_root / rel).read_text()
        assert "auto_extract=False" not in text, (
            f"{rel} contains `auto_extract=False` — that band-aid "
            "was needed pre-loomflow-0.10.20 to mask synchronous "
            "extraction latency. Now that AutoExtractMemory is "
            "background-by-default, the override is unnecessary "
            "and silently disables fact extraction. Drop it."
        )
        assert "auto_consolidate=False" not in text, (
            f"{rel} contains `auto_consolidate=False` — same "
            "band-aid story as auto_extract. Drop it."
        )


def test_coordinator_persists_tool_transcripts(project: Project) -> None:
    # loomflow 0.10.16+ — Team.supervisor forwards
    # persist_tool_transcripts to the coordinator Agent. We pin the
    # flag at True so future edits to ``build_agent`` can't silently
    # drop the kwarg and regress the supervisor to "forgets every
    # prior delegation between Agent.run() calls" behavior. Workers
    # have their own assertion in tests/test_workers.py.
    coordinator, _ = build_agent(project, model="echo")
    assert coordinator._persist_tool_transcripts is True


def test_graphify_skill_registered_on_coordinator(
    project: Project,
) -> None:
    """The bundled graphify skill in ``loom_code/skills/graphify/``
    must be auto-discovered by ``build_agent`` and wired into the
    coordinator's skill registry. Regression-pins the skill-discovery
    path (importlib.resources lookup + package-data inclusion in
    pyproject) so a future edit can't silently drop the graphify
    integration."""
    coordinator, _ = build_agent(project, model="echo")
    skill_names = set(coordinator.skills.names())
    assert "graphify" in skill_names, (
        f"graphify skill missing from coordinator's surface "
        f"(found: {skill_names}). Check pyproject.toml's "
        "package-data section + loom_code/skills/graphify/SKILL.md."
    )
