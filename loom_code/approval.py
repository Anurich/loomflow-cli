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
import sys
from collections.abc import Callable
from typing import Any

import anyio
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from .render import console

# History-/repo-destroying shell commands. These are NOT covered by the
# "allow all this session" choice — even after the user picks 'a', one
# of these still demands a fresh, explicit confirmation, because they're
# irreversible in a way ordinary edits aren't. A real incident motivated
# this: the agent ran ``rm -rf .git`` on request and silently deleted a
# repo's entire history with no extra friction. Patterns are matched
# against the normalized (lowercased, whitespace-collapsed) command.
_DANGER_PATTERNS: tuple[str, ...] = (
    "rm -rf .git",
    "rm -r .git",
    "rm -fr .git",
    "rm -rf .git/",
    "git reset --hard",
    "git clean -fd",
    "git clean -xfd",
    "git push --force",
    "git push -f",
    "git push --force-with-lease",
    "git branch -d",  # force-delete branch
    "git update-ref -d",
    "rm -rf /",
)


def _is_danger_command(tool: str, args: dict[str, Any]) -> str | None:
    """Return a human label if this call is a history-/repo-destroying
    bash command, else None. Only ``bash`` can carry these — edits and
    writes are bounded to a single file and already gated.

    The normalized match collapses ``rm   -rf    .git`` and quoting
    variants down so a stray space can't slip a destructive command
    past the check. False positives are acceptable here: an extra
    confirmation on a benign ``git reset --hard`` to a known-safe ref
    costs one keypress; a missed ``rm -rf .git`` costs the repo."""
    if tool != "bash":
        return None
    cmd = str(args.get("command", "")).lower()
    norm = " ".join(cmd.split())
    for pat in _DANGER_PATTERNS:
        if pat in norm:
            return pat
    return None


def _read_single_key() -> str:
    """Read ONE keypress without waiting for Enter.

    POSIX raw-mode read; falls back to ``msvcrt`` on Windows and to a
    line read when stdin isn't a TTY (piped input, tests). Returning a
    single character lets the approval prompt act like a button row —
    the user reported the type-then-Enter form as a real obstacle.
    """
    if not sys.stdin.isatty():
        # Non-interactive (piped/CI/tests) — degrade to a line read so
        # the gate still resolves instead of blocking forever.
        try:
            return sys.stdin.readline().strip()[:1].lower()
        except Exception:
            return ""
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        try:
            import msvcrt

            return msvcrt.getch().decode("utf-8", "ignore")
        except Exception:
            try:
                return sys.stdin.readline().strip()[:1].lower()
            except Exception:
                return ""


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

    def __init__(
        self,
        *,
        pause_spinner: Callable[[], None] | None = None,
        resume_spinner: Callable[[], None] | None = None,
    ) -> None:
        # Once the user picks 'a' (allow all), every subsequent
        # destructive call this session auto-approves. Reset only
        # by restarting loom-code.
        self._allow_all = False
        # The REPL drives a ``console.status`` spinner for the whole
        # turn. Its Live refresh shares the cursor line, so leaving it
        # running corrupts the approval prompt's keystrokes (mangled
        # input → endless "select an option" re-prompt). We pause it
        # around the prompt and resume after.
        self._pause_spinner = pause_spinner or (lambda: None)
        self._resume_spinner = resume_spinner or (lambda: None)

    async def handler(
        self, call: Any, user_id: str | None = None
    ) -> bool:
        """The ``approval_handler`` loomflow calls. ``call`` is a
        ``ToolCall``; return True to allow, False to deny."""
        tool = getattr(call, "tool", "?")
        args = getattr(call, "args", {}) or {}

        # History-/repo-destroying commands get a HARD gate that
        # 'allow all this session' does NOT bypass — they're
        # irreversible and a single careless 'a' earlier in the
        # session must not silently green-light wiping git history.
        # This check runs BEFORE the _allow_all shortcut on purpose.
        danger = _is_danger_command(tool, args)
        if danger is not None:
            return await self._confirm_danger(danger, args)

        if self._allow_all:
            return True

        # Pause the spinner BEFORE any console output — even the
        # warning lines below get garbled if the Live is still
        # repainting the cursor line.
        self._pause_spinner()
        try:
            console.print()
            console.print(
                Text(f"  ⚠ {tool} wants to run:", style="bold yellow")
            )
            self._render_preview(tool, args)

            # Single keypress, on a worker thread so the raw-mode
            # read doesn't stall the anyio event loop.
            choice = await anyio.to_thread.run_sync(self._ask)
        finally:
            self._resume_spinner()
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

    async def _confirm_danger(
        self, label: str, args: dict[str, Any]
    ) -> bool:
        """High-friction confirm for an irreversible command. Unlike the
        normal gate there is NO 'allow all', and the default (Enter /
        any non-'y' key / Esc) is DENY — the user must deliberately type
        'y' to proceed. Never auto-approves, regardless of session state
        or ``--yes``-style handlers wrapped around this gate."""
        self._pause_spinner()
        try:
            console.print()
            console.print(
                Text(
                    f"  ⛔ DESTRUCTIVE: this would run '{label}' — it is "
                    "IRREVERSIBLE",
                    style="bold red",
                )
            )
            cmd = str(args.get("command", ""))
            console.print(Syntax(cmd, "bash", theme="ansi_dark"))
            console.print(
                Text(
                    "  This permanently destroys history / data and is "
                    "NOT covered by 'allow all'.",
                    style="red",
                )
            )
            console.print(
                "  [bold]Type 'y' to confirm, anything else cancels:[/bold]"
                " ",
                end="",
                highlight=False,
            )
            choice = await anyio.to_thread.run_sync(_read_single_key)
        finally:
            self._resume_spinner()
        if choice in ("y", "Y"):
            console.print("[red]confirmed[/red]")
            return True
        console.print("[green]cancelled[/green]")
        return False

    def _ask(self) -> str:
        """Blocking single-keypress choice. Runs on a worker thread.

        Renders a bordered "button-row" panel, then reads ONE key —
        no Enter required. Y / Enter = yes, N = no, A = allow all.
        User reported the prior type-then-Enter form as a real
        obstacle: "we can simply click on it don't have to write".
        Single keypress is the closest a terminal gets to a button.

        Unknown keys are ignored (loop keeps waiting) rather than
        re-prompting noisily; Ctrl-C / Esc resolve to 'no' so a
        startled user can always back out safely.
        """
        button_row = Text.assemble(
            ("  ", ""),
            ("Y", "bold green"),
            (" yes  ", "green"),
            ("(or press ", "dim"),
            ("Enter", "bold"),
            (")    ", "dim"),
            ("N", "bold red"),
            (" no    ", "red"),
            ("A", "bold yellow"),
            (" yes to all this session", "yellow"),
        )
        console.print(
            Panel(
                button_row,
                border_style="dim",
                padding=(0, 1),
                expand=False,
            )
        )
        console.print(
            "  [dim]press a key:[/dim] ", end="", highlight=False
        )
        while True:
            ch = _read_single_key()
            if ch in ("\r", "\n", "y", "Y"):
                console.print("[green]yes[/green]")
                return "y"
            if ch in ("n", "N"):
                console.print("[red]no[/red]")
                return "n"
            if ch in ("a", "A"):
                console.print("[yellow]yes to all[/yellow]")
                return "a"
            if ch in ("\x03", "\x1b", ""):  # Ctrl-C, Esc, EOF
                console.print("[red]no[/red]")
                return "n"
            # Any other key: ignore and keep waiting.

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
