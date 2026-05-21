"""Tests for the SIMPLE → SUPERVISOR escalation path.

When the SIMPLE coder calls ``escalate_to_team(reason)``, the REPL
detects it (via the renderer) and re-dispatches the same prompt
through the supervisor, which inherits SIMPLE's partial context via
the shared session_id. These tests pin the pieces:

  - the tool is wired onto the SIMPLE coder ONLY (not the team's
    own coder — escalating from inside the team makes no sense)
  - the renderer detects the call + captures the reason
  - build_agent stamps the supervisor on the coordinator so the
    REPL can reach it
"""

from __future__ import annotations

import asyncio

from loom_code.escalate import ESCALATE_TOOL_NAME, escalate_to_team_tool
from loom_code.render import StreamRenderer


def test_escalate_tool_name_and_returns_handoff_message() -> None:
    tool = escalate_to_team_tool()
    assert tool.name == ESCALATE_TOOL_NAME
    result = asyncio.run(tool.fn(reason="needs 4-file refactor"))
    assert "team" in result.lower()
    # Tells the model to STOP — it shouldn't keep working after
    # escalating.
    assert "do not continue" in result.lower() or "do not" in result.lower()


def test_renderer_detects_escalation_call() -> None:
    """The renderer must flag escalation + capture the reason from
    the tool_call event so the REPL can read it post-run."""
    r = StreamRenderer()
    assert r.escalation_requested is False

    # Simulate the tool_call event shape loomflow emits.
    r.handle(
        _FakeEvent(
            "tool_call",
            {
                "call": {
                    "id": "c1",
                    "tool": ESCALATE_TOOL_NAME,
                    "args": {"reason": "cross-file coordination needed"},
                }
            },
        )
    )
    assert r.escalation_requested is True
    assert r.escalation_reason == "cross-file coordination needed"


def test_renderer_ignores_normal_tools() -> None:
    """A normal tool call must NOT set the escalation flag."""
    r = StreamRenderer()
    r.handle(
        _FakeEvent(
            "tool_call",
            {"call": {"id": "c1", "tool": "read", "args": {"path": "x"}}},
        )
    )
    assert r.escalation_requested is False


def test_simple_coder_has_escalate_tool_not_team_coder(project) -> None:
    """The escalate tool is on the SIMPLE route only. The team's
    coder worker must NOT have it — escalating from inside the team
    is nonsensical (it IS the team)."""
    from loom_code.agent import build_agent

    coord, _ = build_agent(project, model="echo")
    routes = {r.name: r.agent for r in coord.architecture._routes}

    simple_tools = _tool_names(routes["simple"])
    assert ESCALATE_TOOL_NAME in simple_tools, (
        f"SIMPLE coder missing escalate tool (has: {simple_tools})"
    )

    # The complex route is the supervisor; its workers (coder etc.)
    # should NOT carry escalate. Check the coder worker.
    sup = routes["complex"]
    workers = sup.architecture.declared_workers()
    coder_tools = _tool_names(workers["coder"])
    assert ESCALATE_TOOL_NAME not in coder_tools, (
        "team's coder worker should NOT have escalate_to_team"
    )


def test_build_agent_stamps_complex_agent(project) -> None:
    """The REPL re-dispatches escalations to ``coordinator.
    _complex_agent``; build_agent must stamp it."""
    from loom_code.agent import build_agent

    coord, _ = build_agent(project, model="echo")
    sup = getattr(coord, "_complex_agent", None)
    assert sup is not None
    # It's the same object as the complex route's agent.
    complex_route = next(
        r for r in coord.architecture._routes if r.name == "complex"
    )
    assert sup is complex_route.agent


# ---- helpers --------------------------------------------------------


class _FakeEvent:
    """Minimal Event stand-in for the renderer's handle()."""

    def __init__(self, kind: str, payload: dict) -> None:
        self.kind = kind
        self.payload = payload


def _tool_names(agent) -> set[str]:
    """Read tool names off an Agent's in-process tool host."""
    host = agent._tool_host  # noqa: SLF001 — test introspection
    tools = getattr(host, "_tools", {})
    return set(tools.keys())
