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
from pathlib import Path
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


def _read_key_raw(fd: int) -> str:
    """Read one LOGICAL key from an ALREADY-raw ``fd``: 'up' / 'down'
    / 'enter' / 'esc' / 'eof' / a single lowercased printable char.

    Arrow keys arrive as an escape sequence — ``ESC [ A/B`` (normal)
    or ``ESC O A/B`` (application-cursor mode, common over SSH/tmux).
    The bytes can also SPLIT across reads on a slow PTY, so after ESC
    we poll-and-read up to two more bytes rather than assuming they
    land in one ``os.read`` (the earlier one-shot ``os.read(fd, 2)``
    turned a split ↓ into 'esc' → an accidental deny, and left the
    trailing 'A'/'B' in the buffer to be misread as the 'a' hotkey).

    ``os.read`` on the raw fd, never ``sys.stdin.read`` — Python's
    stdin buffers ahead, hiding continuation bytes from ``select``.
    An empty read is EOF (terminal hangup) → 'eof', which callers
    treat as a safe cancel, never an approval."""
    import os
    import select

    def _more(timeout: float) -> str:
        r, _, _ = select.select([fd], [], [], timeout)
        if not r:
            return ""
        return os.read(fd, 1).decode("utf-8", "ignore")

    data = os.read(fd, 1)
    if not data:  # EOF / hangup
        return "eof"
    ch = data.decode("utf-8", "ignore")
    if ch in ("\r", "\n"):
        return "enter"
    if ch == "\x03":  # Ctrl-C
        return "esc"
    if ch == "\x1b":
        intro = _more(0.05)
        if intro in ("[", "O"):  # CSI or SS3 arrows
            final = _more(0.05)
            return {"A": "up", "B": "down"}.get(final, "esc")
        return "esc"
    return ch.lower()


def _read_key_msvcrt() -> str:
    """Windows equivalent of :func:`_read_key_raw` — one LOGICAL key
    via ``msvcrt.getwch()`` (already raw + no echo, so no mode
    setup/teardown is needed).

    Arrow keys arrive as a TWO-event sequence: a ``'\\xe0'`` (or
    ``'\\x00'`` for some layouts) prefix, then ``'H'`` (up) /
    ``'P'`` (down). Ctrl-C surfaces as ``'\\x03'`` and maps to the
    SAFE 'esc', mirroring the POSIX reader."""
    import msvcrt

    ch = msvcrt.getwch()
    if ch in ("\r", "\n"):
        return "enter"
    if ch == "\x03":  # Ctrl-C
        return "esc"
    if ch == "\x1b":
        return "esc"
    if ch in ("\xe0", "\x00"):  # extended-key prefix
        final = msvcrt.getwch()
        return {"H": "up", "P": "down"}.get(final, "esc")
    return ch.lower()


def _read_key() -> str:
    """Single-key read that manages its own raw-mode window. Prefer
    :func:`_read_key_raw` inside a selector that enters raw mode ONCE
    (no per-key termios churn); this wrapper is for one-off reads.

    Non-TTY (piped/CI/tests): a line read whose EOF maps to 'eof'
    (fail-closed), NOT 'enter'."""
    if not sys.stdin.isatty():
        try:
            line = sys.stdin.readline()
        except Exception:
            return "eof"
        if line == "":
            return "eof"
        ch = line.strip()[:1].lower()
        return ch or "enter"
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return _read_key_raw(fd)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        return _read_single_key() or "eof"


def _select_option(options: list[tuple[str, str]], default: int = 0) -> str:
    """A Claude-Code-style vertical selector. ``options`` is a list of
    ``(key, label)``; returns the chosen key.

    Navigation: ↑/↓ move, Enter confirms, 1-9 jump-select, the
    option's own hotkey (y/n/a) selects directly. Esc / Ctrl-C / EOF
    pick the LAST option — which callers make the SAFE choice ("No"),
    so a closed stdin, a hangup, or a startled Ctrl-C can never
    approve. Rendering is plain ANSI — the block redraws in place on
    every keypress, which works in any VT terminal and deliberately
    avoids nesting a Rich Live inside the REPL's spinner (a known
    recursion hazard)."""
    n = len(options)
    if not sys.stdin.isatty():
        # Non-interactive: a single letter picks by hotkey; EOF/empty
        # is the SAFE last option (never the default-yes).
        ch = _read_key()
        for key, _label in options:
            if ch == key:
                return key
        if ch == "enter":
            return options[default][0]
        return options[-1][0]  # eof / esc / unknown → safe

    idx = default

    def _draw(first: bool) -> None:
        out = sys.stdout
        if not first:
            out.write(f"\x1b[{n}A")  # cursor up n rows, to block start
        for i, (_key, label) in enumerate(options):
            # ``\r`` first: in RAW mode ``tty.setraw`` disables NL→CRNL
            # translation, so a bare ``\n`` drops a row WITHOUT
            # returning to column 0 — each line would start further
            # right than the last (the staircase). Carriage-return to
            # column 0, clear the whole line, then print.
            out.write("\r\x1b[2K")
            if i == idx:
                out.write(f"  \x1b[36;1m❯ {i + 1}. {label}\x1b[0m")
            else:
                out.write(f"    \x1b[2m{i + 1}. {label}\x1b[0m")
            out.write("\r\n")  # explicit CR+LF for raw mode
        out.flush()

    # Pick the platform's key reader + raw-mode strategy.
    #
    # POSIX: enter raw mode ONCE for the whole selector session — no
    # per-keypress termios churn, and no cooked-mode gap between keys
    # where type-ahead would echo raw escape bytes onto the prompt.
    #
    # Windows: there is NO termios (the bare import crashed /set_model
    # for pipx users with ModuleNotFoundError). msvcrt.getwch() is
    # already raw + unbuffered, so no mode management is needed at
    # all; ``os.system("")`` nudges legacy conhost into processing the
    # ANSI redraw sequences (Windows Terminal has VT on by default).
    def _restore() -> None:
        return None

    def _reader() -> str:
        return _read_key()

    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
        except Exception:
            old = None
        if old is not None:
            tty.setraw(fd)

            def _reader() -> str:
                return _read_key_raw(fd)

            def _restore() -> None:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except ImportError:
        try:
            import msvcrt  # noqa: F401 — probe: Windows console?
            import os

            os.system("")  # enable VT processing on legacy conhost
            _reader = _read_key_msvcrt
        except ImportError:
            pass  # exotic platform → keep the _read_key fallback

    try:
        _draw(first=True)
        while True:
            key = _reader()
            if key == "up":
                idx = (idx - 1) % n
            elif key == "down":
                idx = (idx + 1) % n
            elif key == "enter":
                return options[idx][0]
            elif key in ("esc", "eof"):
                return options[-1][0]
            elif key.isdigit() and 1 <= int(key) <= n:
                return options[int(key) - 1][0]
            else:
                for k, _label in options:
                    if key == k:
                        return k
                continue  # unknown key: ignore, keep waiting
            _draw(first=False)
    finally:
        _restore()


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


def _question_for(tool: str) -> str:
    """The bold question above the option list, per tool."""
    return {
        "edit": "Apply this edit?",
        "multi_edit": "Apply these edits?",
        "write": "Write this file?",
        "bash": "Run this command?",
    }.get(tool, f"Allow {tool}?")


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
        rules: Any | None = None,
        mode: Any | None = None,
        project_root: Any | None = None,
    ) -> None:
        # Once the user picks 'a' (allow all), every subsequent
        # destructive call this session auto-approves. Reset only
        # by restarting loom-code.
        self._allow_all = False
        # Permission rules (allow/ask/deny globs) + session mode
        # (default/accept-edits/plan/yolo). Imported here to keep the
        # module import-light. The REPL swaps ``mode`` via /mode.
        from .permissions import Mode, Rules

        self.rules = rules if rules is not None else Rules()
        self.mode = mode if mode is not None else Mode.DEFAULT
        # Project root — used to force a confirm on edits OUTSIDE it
        # even in auto-approve modes. None disables the check (the
        # outside-edit path just behaves like any other edit then).
        self.project_root = (
            Path(project_root).resolve() if project_root else None
        )
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
        ``ToolCall``; return True to allow, False to deny.

        Policy resolves in ONE place — :func:`permissions.decide` — so
        precedence is consistent: **deny > ask > allow > mode**. Then:

        * DENY → refuse, unconditionally (a deny rule / plan mode is
          absolute; nothing below overrides it).
        * ASK → the interactive prompt.
        * ALLOW → skip the prompt, UNLESS the command is one of the
          irreversible history-/repo-destroyers, which ALWAYS get a
          fresh high-friction confirm even under allow-all / yolo.

        Session 'allow all' is modelled as an effective yolo mode
        INSIDE ``decide`` (not a shortcut above it), so an explicit
        ``ask`` rule the user configured still forces the prompt —
        one careless 'a' can't silently defeat their own ask-rules."""
        tool = getattr(call, "tool", "?")
        args = getattr(call, "args", {}) or {}

        from .permissions import Decision, Mode, decide

        effective_mode = Mode.YOLO if self._allow_all else self.mode
        decision = decide(tool, args, self.rules, effective_mode)

        # An edit/write to a file OUTSIDE the project always shows the
        # diff and asks — even in accept-edits / yolo / allow-all /
        # --yes. Consent (an @-mention) lets the edit tool TARGET the
        # file; it does not waive the human confirmation. Without this,
        # /mode accept-edits + an @-mention of ~/.zshrc would silently
        # mutate a dotfile. Never UPGRADES a deny.
        if (
            decision is Decision.ALLOW
            and tool in ("edit", "multi_edit", "write")
            and self._is_outside_project(args.get("path"))
        ):
            decision = Decision.ASK

        if decision is Decision.DENY:
            self._pause_spinner()
            try:
                console.print(
                    Text(
                        f"  ⊘ {tool} denied by permission policy "
                        f"({self.mode.value})",
                        style="red",
                    )
                )
            finally:
                self._resume_spinner()
            return False

        # Danger gate — irreversible commands ALWAYS re-confirm, even
        # when policy said ALLOW. It never UPGRADES a deny (handled
        # above) — only forces friction on an otherwise-allowed
        # destructive command.
        danger = _is_danger_command(tool, args)
        if danger is not None:
            return await self._confirm_danger(danger, args)

        if decision is Decision.ALLOW:
            return True

        # Pause the spinner BEFORE any console output — even the
        # header lines below get garbled if the Live is still
        # repainting the cursor line.
        self._pause_spinner()
        try:
            console.print()
            self._render_header(tool, args)
            self._render_preview(tool, args)
            console.print()
            console.print(
                Text(f"  {_question_for(tool)}", style="bold")
            )
            # Selector runs on a worker thread so the raw-mode key
            # reads don't stall the anyio event loop.
            choice = await anyio.to_thread.run_sync(self._ask)
        finally:
            self._resume_spinner()
        # ``_ask`` already echoed the choice — no second line here
        # (the old "→ denied" after "→ no" read as a double refusal).
        if choice == "a":
            self._allow_all = True
            return True
        return choice == "y"

    # ---- internals ------------------------------------------------------

    def _is_outside_project(self, path: Any) -> bool:
        """True if ``path`` resolves outside the project root. False
        when no root is configured or the path is unusable (fail
        toward the normal in-project flow — the edit tool's own
        workdir guard still refuses genuinely-outside writes)."""
        if self.project_root is None or not path:
            return False
        try:
            target = (self.project_root / Path(path)).resolve()
            target.relative_to(self.project_root)
            return False
        except ValueError:
            return True
        except OSError:
            return False

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
        """Blocking option-selector. Runs on a worker thread.

        Claude-Code-style vertical menu: ↑/↓ + Enter, number keys, or
        the y/a/n hotkeys. Esc / Ctrl-C picks "No" so a startled user
        can always back out safely. Returns 'y' / 'a' / 'n'."""
        choice = _select_option(
            [
                ("y", "Yes"),
                ("a", "Yes, and don't ask again this session"),
                ("n", "No (esc)"),
            ],
            default=0,
        )
        echo = {
            "y": Text("  → yes", style="green"),
            "a": Text("  → yes, allowing all this session", "yellow"),
            "n": Text("  → no", style="red"),
        }[choice]
        console.print(echo)
        return choice

    def _render_header(self, tool: str, args: dict[str, Any]) -> None:
        """One bold title line naming the action + its target —
        ``● Edit  src/main.py`` — Claude-Code-style, replacing the
        old '⚠ tool wants to run:' warning shout."""
        target = (
            str(args.get("command", "")).strip()
            if tool == "bash"
            else str(args.get("path", "")).strip()
        )
        if len(target) > 60:
            target = target[:60] + "…"
        label = {
            "edit": "Edit",
            "multi_edit": "Edit",
            "write": "Write",
            "bash": "Run",
        }.get(tool, tool)
        console.print(
            Text.assemble(
                ("  ● ", "cyan"),
                (f"{label}", "bold"),
                ("  ", ""),
                (target, "dim"),
            )
        )

    def _render_preview(self, tool: str, args: dict[str, Any]) -> None:
        """Show the user exactly what's about to happen, inside a
        rounded panel so the preview reads as one contained artifact
        (Claude-Code-style) rather than loose lines."""
        body: Any
        if tool == "edit":
            body = self._edit_diff_renderable(args)
        elif tool == "write":
            path = args.get("path", "?")
            content = str(args.get("content", ""))
            preview = content if len(content) <= 800 else (
                content[:800] + f"\n… (+{len(content) - 800} chars)"
            )
            body = Syntax(
                preview, _lexer_for(path), theme="ansi_dark",
                line_numbers=False,
            )
        elif tool == "bash":
            cmd = str(args.get("command", ""))
            body = Syntax(cmd, "bash", theme="ansi_dark")
        else:
            # Unknown destructive tool — show raw args.
            lines = []
            for k, v in args.items():
                sv = str(v)
                if len(sv) > 200:
                    sv = sv[:200] + "…"
                lines.append(f"{k} = {sv}")
            body = Text("\n".join(lines), style="dim")
        console.print(
            Panel(
                body,
                border_style="dim",
                padding=(0, 1),
                expand=False,
            )
        )

    def _edit_diff_renderable(self, args: dict[str, Any]) -> Any:
        """A unified-diff renderable for an ``edit`` call so the user
        sees the change in context, not two opaque strings."""
        path = args.get("path", "?")
        old = str(args.get("old_string", ""))
        new = str(args.get("new_string", ""))
        # No keepends + join on "\n": with ``lineterm=""`` the header
        # lines carry no newline of their own, so keepends-content
        # mixed with them used to collapse the whole diff onto one
        # wrapped line.
        diff = difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile=f"{path} (before)",
            tofile=f"{path} (after)",
            lineterm="",
        )
        body = "\n".join(diff)
        if not body.strip():
            return Text(f"edit {path} (no textual change?)", "dim")
        return Syntax(body, "diff", theme="ansi_dark")


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
