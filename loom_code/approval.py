"""The terminal diff-approval gate.

When ``StandardPermissions`` flags a destructive tool call
(``write`` / ``edit`` / ``bash``), loomflow routes it to the
Agent's ``approval_handler``. This module is that handler: it
renders WHAT the agent wants to do — a unified diff for edits, the
full content for writes, the command for bash — and asks the user
y / n / a (allow-all-this-session).

Pure UI. The decision logic loomflow owns; we just collect the
human's answer.
"""

from __future__ import annotations

import difflib
from typing import Any

import anyio
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.text import Text

from .render import console


async def auto_approve(call: Any, user_id: str | None = None) -> bool:
    """A non-interactive approval handler that allows everything.

    For unattended runs — CI, scripted use — where there's no
    human at a TTY to answer the y/n/a prompt and the working tree
    is disposable. Wired in via ``loom-code --yes``.

    NEVER point this at a repo you care about: it lets the agent
    write / edit / run shell commands with no gate at all.
    """
    return True


class ApprovalGate:
    """Stateful approval handler — remembers an 'allow all this
    session' choice so the user isn't asked twice for the same
    kind of risk.

    Pass :meth:`handler` as the Agent's ``approval_handler``.
    """

    def __init__(self) -> None:
        # Once the user picks 'a' (allow all), every subsequent
        # destructive call this session auto-approves. Reset only
        # by restarting loom-code.
        self._allow_all = False

    async def handler(
        self, call: Any, user_id: str | None = None
    ) -> bool:
        """The ``approval_handler`` loomflow calls. ``call`` is a
        ``ToolCall``; return True to allow, False to deny."""
        if self._allow_all:
            return True

        tool = getattr(call, "tool", "?")
        args = getattr(call, "args", {}) or {}

        console.print()
        console.print(
            Text(f"  ⚠ {tool} wants to run:", style="bold yellow")
        )
        self._render_preview(tool, args)

        # The prompt itself is blocking input() — push it to a
        # worker thread so we don't stall the anyio event loop.
        choice = await anyio.to_thread.run_sync(self._ask)
        if choice == "a":
            self._allow_all = True
            console.print(
                Text(
                    "  → allowing all destructive calls this "
                    "session",
                    style="dim",
                )
            )
            return True
        if choice == "y":
            return True
        console.print(Text("  → denied", style="red"))
        return False

    # ---- internals ------------------------------------------------------

    def _ask(self) -> str:
        """Blocking y/n/a prompt. Runs on a worker thread."""
        return Prompt.ask(
            "  [bold]allow?[/bold]",
            choices=["y", "n", "a"],
            default="y",
            show_choices=True,
        ).strip().lower()

    def _render_preview(self, tool: str, args: dict[str, Any]) -> None:
        """Show the user exactly what's about to happen."""
        if tool == "edit":
            self._render_edit_diff(args)
        elif tool == "write":
            path = args.get("path", "?")
            content = str(args.get("content", ""))
            console.print(Text(f"    write {path}:", style="dim"))
            preview = content if len(content) <= 800 else (
                content[:800] + f"\n… (+{len(content) - 800} chars)"
            )
            console.print(
                Syntax(
                    preview, _lexer_for(path), theme="ansi_dark",
                    line_numbers=False,
                )
            )
        elif tool == "bash":
            cmd = str(args.get("command", ""))
            console.print(
                Syntax(cmd, "bash", theme="ansi_dark")
            )
        else:
            # Unknown destructive tool — show raw args.
            for k, v in args.items():
                sv = str(v)
                if len(sv) > 200:
                    sv = sv[:200] + "…"
                console.print(Text(f"    {k} = {sv}", style="dim"))

    def _render_edit_diff(self, args: dict[str, Any]) -> None:
        """Render a unified diff for an ``edit`` call so the user
        sees the change in context, not two opaque strings."""
        path = args.get("path", "?")
        old = str(args.get("old_string", ""))
        new = str(args.get("new_string", ""))
        diff = difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"{path} (before)",
            tofile=f"{path} (after)",
            lineterm="",
        )
        body = "".join(diff)
        if not body.strip():
            console.print(
                Text(f"    edit {path} (no textual change?)", "dim")
            )
            return
        console.print(Text(f"    edit {path}:", style="dim"))
        console.print(Syntax(body, "diff", theme="ansi_dark"))


def _lexer_for(path: str) -> str:
    """Best-effort lexer name from a file extension."""
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "tsx": "tsx",
        "jsx": "jsx",
        "rs": "rust",
        "go": "go",
        "rb": "ruby",
        "java": "java",
        "c": "c",
        "h": "c",
        "cpp": "cpp",
        "sh": "bash",
        "bash": "bash",
        "json": "json",
        "toml": "toml",
        "yaml": "yaml",
        "yml": "yaml",
        "md": "markdown",
        "html": "html",
        "css": "css",
        "sql": "sql",
    }.get(ext, "text")
