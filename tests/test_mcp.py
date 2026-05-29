"""Tests for loom-code's MCP integration.

Covers the three pieces wired in:

* discovery — ``[[mcp]]`` blocks parsed from ``settings.toml`` into
  source-tagged ``McpEntry`` (lenient: bad entries dropped, not fatal).
* trust gate — project-scope MCP servers are dropped from an UNTRUSTED
  repo and load once trusted; the fingerprint changes when a server's
  identity (command/url/args) changes, re-prompting.
* ``McpAugmentedHost`` — merges the coder's static tools with the MCP
  registry, static tools winning a name collision, MCP-only names
  routing to the registry, a down server degrading gracefully.

All offline — no real MCP server is started.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from loomflow.core.types import ToolDef, ToolResult
from loomflow.mcp import MCPServerSpec
from loomflow.tools.registry import InProcessToolHost, Tool

from loom_code.extensions import McpEntry, discover
from loom_code.mcp_host import McpAugmentedHost
from loom_code.trust import (
    _fingerprint,
    discover_trusted,
    is_trusted,
    record_trust,
)

pytestmark = pytest.mark.anyio


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---- discovery ------------------------------------------------------


def test_discovers_project_and_user_mcp(tmp_path: Path) -> None:
    proj = tmp_path / "repo"
    user = tmp_path / "home"
    _write(
        proj / ".loom" / "settings.toml",
        '[[mcp]]\nname = "linear"\ncommand = "npx"\n'
        'args = ["-y", "linear-mcp"]\nenv = { K = "v" }\n',
    )
    _write(
        user / "settings.toml",
        '[[mcp]]\nname = "sentry"\ntransport = "http"\n'
        'url = "https://mcp.example/v1"\n',
    )
    ext = discover(proj, user_dir=user)
    by_name = {e.spec.name: e for e in ext.mcp_specs}
    assert set(by_name) == {"linear", "sentry"}
    assert by_name["linear"].source == "project"
    assert by_name["linear"].spec.command == "npx"
    assert by_name["linear"].spec.args == ("-y", "linear-mcp")
    assert by_name["sentry"].source == "user"
    assert by_name["sentry"].spec.transport == "http"


def test_bad_mcp_entries_are_dropped(tmp_path: Path) -> None:
    proj = tmp_path / "repo"
    _write(
        proj / ".loom" / "settings.toml",
        # no name; stdio w/o command; http w/o url — all invalid.
        '[[mcp]]\ncommand = "x"\n\n'
        '[[mcp]]\nname = "no_cmd"\ntransport = "stdio"\n\n'
        '[[mcp]]\nname = "no_url"\ntransport = "http"\n\n'
        '[[mcp]]\nname = "ok"\ncommand = "run"\n',
    )
    ext = discover(proj, user_dir=tmp_path / "none")
    assert [e.spec.name for e in ext.mcp_specs] == ["ok"]


# ---- trust gate -----------------------------------------------------


def test_project_mcp_dropped_when_untrusted(tmp_path: Path) -> None:
    proj = tmp_path / "repo"
    _write(
        proj / ".loom" / "settings.toml",
        '[[mcp]]\nname = "linear"\ncommand = "npx"\n',
    )
    store = tmp_path / "trust.json"
    ext = discover_trusted(
        proj, user_dir=tmp_path / "home", trust_store=store
    )
    assert ext.mcp_specs == []  # gated out


def test_project_mcp_loads_once_trusted(tmp_path: Path) -> None:
    proj = tmp_path / "repo"
    _write(
        proj / ".loom" / "settings.toml",
        '[[mcp]]\nname = "linear"\ncommand = "npx"\n',
    )
    user = tmp_path / "home"
    store = tmp_path / "trust.json"

    full = discover(proj, user_dir=user)
    pmcp = [m for m in full.mcp_specs if m.source == "project"]
    record_trust(proj.resolve(), [], pmcp, trust_store=store)

    ext = discover_trusted(proj, user_dir=user, trust_store=store)
    assert [e.spec.name for e in ext.mcp_specs] == ["linear"]


def test_user_mcp_always_kept_even_untrusted(tmp_path: Path) -> None:
    proj = tmp_path / "repo"
    user = tmp_path / "home"
    _write(user / "settings.toml", '[[mcp]]\nname = "mine"\ncommand = "x"\n')
    store = tmp_path / "trust.json"
    ext = discover_trusted(proj, user_dir=user, trust_store=store)
    assert [e.spec.name for e in ext.mcp_specs] == ["mine"]


def test_fingerprint_changes_when_mcp_command_changes() -> None:
    a = McpEntry(
        "project",
        MCPServerSpec(name="s", transport="stdio", command="old"),
    )
    b = McpEntry(
        "project",
        MCPServerSpec(name="s", transport="stdio", command="new"),
    )
    assert _fingerprint([], [a]) != _fingerprint([], [b])


def test_is_trusted_true_after_record(tmp_path: Path) -> None:
    store = tmp_path / "trust.json"
    mcp = [
        McpEntry(
            "project",
            MCPServerSpec(name="s", transport="stdio", command="x"),
        )
    ]
    root = tmp_path / "repo"
    assert is_trusted(root.resolve(), [], mcp, trust_store=store) is False
    record_trust(root.resolve(), [], mcp, trust_store=store)
    assert is_trusted(root.resolve(), [], mcp, trust_store=store) is True


# ---- McpAugmentedHost -----------------------------------------------


class _FakeRegistry:
    """A stand-in MCPRegistry: returns canned tools + echoes calls."""

    def __init__(self, *, down: bool = False) -> None:
        self._down = down

    async def list_tools(self, *, query: str | None = None) -> list[ToolDef]:
        if self._down:
            raise RuntimeError("server unreachable")
        return [
            ToolDef(
                name="linear_issue",
                description="create an issue",
                input_schema={"type": "object"},
            ),
            ToolDef(
                name="edit",  # collides with the static builtin
                description="mcp edit (should be shadowed)",
                input_schema={"type": "object"},
            ),
        ]

    async def call(
        self, tool: str, args: object, *, call_id: str = ""
    ) -> ToolResult:
        return ToolResult.success(call_id=call_id, output=f"mcp:{tool}")


def _static_host() -> InProcessToolHost:
    host = InProcessToolHost()
    host.register(
        Tool(
            name="edit",
            description="static edit",
            fn=lambda **_: "edited",
            input_schema={"type": "object"},
        )
    )
    return host


async def test_host_merges_tools_static_wins_collision() -> None:
    host = McpAugmentedHost(_static_host(), _FakeRegistry())
    names = [d.name for d in await host.list_tools()]
    assert names.count("edit") == 1  # no duplicate
    assert "linear_issue" in names


async def test_host_routes_static_to_base() -> None:
    host = McpAugmentedHost(_static_host(), _FakeRegistry())
    res = await host.call("edit", {}, call_id="c1")
    assert res.output == "edited"  # static, not mcp


async def test_host_routes_unknown_to_mcp() -> None:
    host = McpAugmentedHost(_static_host(), _FakeRegistry())
    res = await host.call("linear_issue", {}, call_id="c2")
    assert res.output == "mcp:linear_issue"


async def test_host_tolerates_down_server() -> None:
    host = McpAugmentedHost(_static_host(), _FakeRegistry(down=True))
    # A down MCP server must not break listing the static tools.
    names = [d.name for d in await host.list_tools()]
    assert names == ["edit"]
