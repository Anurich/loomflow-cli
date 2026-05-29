"""Compose loom-code's static coder tools with MCP servers.

loom-code's coder is built with a static ``list[Tool]`` (read/write/
edit/bash/...). MCP servers, by contrast, are a *dynamic* ``ToolHost``
(``MCPRegistry``) whose tool list is only known after connecting. The
framework's ``Agent(tools=)`` accepts either a list (wrapped in an
``InProcessToolHost``) or a ready ``ToolHost`` — but not a list *plus* a
host.

:class:`McpAugmentedHost` bridges the two: it fronts the static
``InProcessToolHost`` and the ``MCPRegistry`` as one ``ToolHost``,
resolving MCP tools **lazily** (``MCPRegistry.list_tools`` / ``call``
auto-connect on first use, so building the agent costs nothing and a
down server only surfaces when a tool is actually used). Static tools
win on a name collision — an MCP server can't shadow ``edit`` / ``bash``.

The agent loop needs ``list_tools`` + ``call`` (``watch`` optional); we
implement all three. Lifecycle: the caller owns the registry and must
``await registry.aclose()`` on shutdown (the REPL does this on exit).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

from loomflow.core.types import ToolDef, ToolEvent, ToolResult
from loomflow.tools.registry import InProcessToolHost


class McpAugmentedHost:
    """A ``ToolHost`` = static in-process tools + a lazy MCP registry."""

    def __init__(self, base: InProcessToolHost, mcp: Any) -> None:
        """``base`` holds the coder's static tools; ``mcp`` is an
        ``MCPRegistry`` (typed ``Any`` so importing this module never
        requires the ``mcp`` extra)."""
        self._base = base
        self._mcp = mcp

    @property
    def base(self) -> InProcessToolHost:
        return self._base

    async def list_tools(self, *, query: str | None = None) -> list[ToolDef]:
        base_defs = await self._base.list_tools(query=query)
        base_names = {d.name for d in base_defs}
        try:
            mcp_defs = await self._mcp.list_tools(query=query)
        except Exception:  # noqa: BLE001 — a down MCP server must not
            # break tool listing; the static tools still work.
            mcp_defs = []
        # Static tools win on collision — an MCP server can't shadow a
        # builtin like ``edit`` or ``bash``.
        merged = list(base_defs)
        merged.extend(d for d in mcp_defs if d.name not in base_names)
        return merged

    async def call(
        self,
        tool: str,
        args: Mapping[str, Any],
        *,
        call_id: str = "",
    ) -> ToolResult:
        # Static tools take precedence; only fall through to MCP for a
        # name the base host doesn't know.
        if self._base.get(tool) is not None:
            return await self._base.call(tool, args, call_id=call_id)
        # ``self._mcp`` is typed ``Any`` (lazy mcp-extra), so annotate
        # the result for mypy --strict.
        result: ToolResult = await self._mcp.call(
            tool, args, call_id=call_id
        )
        return result

    async def watch(self) -> AsyncIterator[ToolEvent]:
        async for event in self._base.watch():
            yield event
