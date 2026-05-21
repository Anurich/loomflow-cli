"""Tests for build_agent — the Team.router(simple, complex) wiring."""

from __future__ import annotations

from loomflow import Agent
from loomflow.architecture import Supervisor
from loomflow.architecture.router import Router
from loomflow.workspace import LocalDiskWorkspace

from loom_code.agent import LOOM_DIR, build_agent
from loom_code.project import Project


def test_build_agent_returns_router_coordinator(project: Project) -> None:
    # As of the SIMPLE-mode fast-lane work, ``build_agent`` returns
    # a ``Team.router`` Agent that classifies each user prompt and
    # dispatches to SIMPLE (single coder) or COMPLEX (supervisor
    # team). Test that the outer Agent is the router.
    coordinator, _workspace = build_agent(project, model="echo")
    assert isinstance(coordinator, Agent)
    assert isinstance(coordinator.architecture, Router)


def test_router_has_simple_and_complex_routes(project: Project) -> None:
    coordinator, _ = build_agent(project, model="echo")
    route_names = {r.name for r in coordinator.architecture._routes}
    assert route_names == {"simple", "complex"}


def test_complex_route_is_the_supervisor_team(project: Project) -> None:
    # The COMPLEX route must point at the supervisor (with the four
    # workers); regressing this collapses both routes into single-
    # agent mode and loses team-pattern capability.
    coordinator, _ = build_agent(project, model="echo")
    complex_route = next(
        r for r in coordinator.architecture._routes if r.name == "complex"
    )
    assert isinstance(complex_route.agent.architecture, Supervisor)
    workers = complex_route.agent.architecture.declared_workers()
    assert set(workers) == {"coder", "explorer", "auditor", "reviewer"}


def test_simple_route_is_a_plain_agent(project: Project) -> None:
    # The SIMPLE route must NOT be a Supervisor/Router/Team — it's
    # a single Agent with the full file-and-shell kernel. Regressing
    # this defeats the whole purpose of the fast-lane.
    coordinator, _ = build_agent(project, model="echo")
    simple_route = next(
        r for r in coordinator.architecture._routes if r.name == "simple"
    )
    # Plain Agent → ReAct architecture (default), not Supervisor/Router.
    assert isinstance(simple_route.agent, Agent)
    assert not isinstance(simple_route.agent.architecture, Supervisor)
    assert not isinstance(simple_route.agent.architecture, Router)


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
    """The bundled graphify skill in
    ``loom_code/skills/graphify/`` must be auto-discovered by
    ``build_agent`` and wired into the COMPLEX-route supervisor's
    skill registry. Regression-pins the skill-discovery path
    (importlib.resources lookup + package-data inclusion in
    pyproject) so a future edit can't silently drop the
    graphify integration."""
    coordinator, _ = build_agent(project, model="echo")
    from loomflow.architecture.router import Router

    assert isinstance(coordinator.architecture, Router)
    complex_route = next(
        r for r in coordinator.architecture._routes if r.name == "complex"
    )
    supervisor_agent = complex_route.agent
    # The Skill registry on the supervisor must list 'graphify'
    # — that's what the agent sees when deciding to call
    # ``load_skill('graphify')``. ``names()`` is the public listing.
    skill_names = set(supervisor_agent.skills.names())
    assert "graphify" in skill_names, (
        f"graphify skill missing from coordinator's surface "
        f"(found: {skill_names}). Check pyproject.toml's "
        "package-data section + loom_code/skills/graphify/SKILL.md."
    )
