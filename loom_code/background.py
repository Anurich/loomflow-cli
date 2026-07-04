"""Background bash — long-running processes that don't block the turn.

The capability gap vs Claude Code (background subagents/shells by
default) and opencode (non-blocking subagents): loom-code's ``bash``
blocks until the command exits, so a dev server, a watcher, or a long
test run wedges the whole turn. These tools let the agent start a
process, keep working, and check on it later:

* ``bash_background(command)`` → spawns detached, returns a handle
  (``bg1``, ``bg2``, …). Runs arbitrary code, so it is
  ``destructive=True`` and rides the SAME approval gate + allow/ask/
  deny rules as ``bash`` (``permissions.call_target`` maps it to the
  command string, and the irreversible-danger scan applies).
* ``bash_output(handle)`` → status + the output tail. Read-only.
* ``bash_kill(handle)`` → terminate the process group. Read-only
  gate-wise (it only stops what the agent itself started).

Registry is module-level (the ``consent.py`` pattern): tools are
built at agent construction, the REPL kills leftovers on exit, and
``atexit`` backstops a crash so no orphan dev-servers outlive the
session. Output goes to a spool file per process — bounded tail
reads, no pipe-buffer deadlocks.
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loomflow import tool

_TAIL_CHARS = 4_000


@dataclass
class _Proc:
    handle: str
    command: str
    popen: subprocess.Popen[bytes]
    spool_path: Path
    started_at: float = field(default_factory=time.monotonic)


_procs: dict[str, _Proc] = {}
_counter = 0


def reset() -> None:
    """Kill everything and clear the registry (tests + /clear)."""
    kill_all()
    _procs.clear()


def _spawn(command: str, cwd: Path) -> _Proc:
    global _counter
    _counter += 1
    handle = f"bg{_counter}"
    fd, spool = tempfile.mkstemp(prefix=f"loom-{handle}-", suffix=".log")
    spool_file = os.fdopen(fd, "wb")
    # New process GROUP so bash_kill can terminate the whole tree
    # (a dev server forks children; killing just the shell leaks
    # them). start_new_session works on POSIX; on Windows we fall
    # back to CREATE_NEW_PROCESS_GROUP.
    kwargs: dict[str, Any] = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True
    else:  # pragma: no cover - windows
        kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    popen = subprocess.Popen(  # noqa: S602 - the whole point
        command,
        shell=True,
        cwd=str(cwd),
        stdout=spool_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        **kwargs,
    )
    spool_file.close()  # child holds its own fd; parent reads by path
    proc = _Proc(
        handle=handle,
        command=command,
        popen=popen,
        spool_path=Path(spool),
    )
    _procs[handle] = proc
    return proc


def _tail(proc: _Proc) -> str:
    try:
        data = proc.spool_path.read_bytes()
    except OSError:
        return ""
    text = data.decode("utf-8", "replace")
    if len(text) > _TAIL_CHARS:
        text = f"…(earlier output trimmed)\n{text[-_TAIL_CHARS:]}"
    return text


def _status_line(proc: _Proc) -> str:
    rc = proc.popen.poll()
    elapsed = time.monotonic() - proc.started_at
    if rc is None:
        return (
            f"{proc.handle}: RUNNING ({elapsed:.0f}s) — {proc.command}"
        )
    return (
        f"{proc.handle}: EXITED rc={rc} after {elapsed:.0f}s — "
        f"{proc.command}"
    )


def _kill(proc: _Proc) -> str:
    rc = proc.popen.poll()
    if rc is not None:
        return f"{proc.handle} already exited (rc={rc})"
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.popen.pid), signal.SIGTERM)
        else:  # pragma: no cover - windows
            proc.popen.terminate()
    except (ProcessLookupError, PermissionError, OSError):
        proc.popen.terminate()
    try:
        proc.popen.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.popen.pid), signal.SIGKILL)
            else:  # pragma: no cover - windows
                proc.popen.kill()
        except (ProcessLookupError, PermissionError, OSError):
            proc.popen.kill()
    return f"{proc.handle} terminated"


def kill_all() -> int:
    """Terminate every live background process. Returns how many were
    still running. Called on REPL exit + atexit."""
    n = 0
    for proc in list(_procs.values()):
        if proc.popen.poll() is None:
            _kill(proc)
            n += 1
    return n


atexit.register(kill_all)


def background_tools(workdir: Path | str) -> list[Any]:
    """The three background-process tools, rooted at ``workdir``."""
    root = Path(workdir)

    async def bash_background(command: str) -> str:
        command = str(command).strip()
        if not command:
            return "ERROR: empty command"
        proc = _spawn(command, root)
        return (
            f"started {proc.handle} (pid {proc.popen.pid}): "
            f"{command}\nCheck it with bash_output(handle="
            f"'{proc.handle}'); stop it with bash_kill(handle="
            f"'{proc.handle}'). Keep working while it runs."
        )

    async def bash_output(handle: str) -> str:
        proc = _procs.get(str(handle).strip())
        if proc is None:
            live = ", ".join(sorted(_procs)) or "none"
            return f"ERROR: unknown handle {handle!r} (live: {live})"
        tail = _tail(proc)
        body = tail if tail.strip() else "(no output yet)"
        return f"{_status_line(proc)}\n---\n{body}"

    async def bash_kill(handle: str) -> str:
        proc = _procs.get(str(handle).strip())
        if proc is None:
            live = ", ".join(sorted(_procs)) or "none"
            return f"ERROR: unknown handle {handle!r} (live: {live})"
        msg = _kill(proc)
        tail = _tail(proc)
        if tail.strip():
            msg += f"\nfinal output tail:\n{tail[-800:]}"
        return msg

    return [
        tool(
            name="bash_background",
            description=(
                "Run a shell command in the BACKGROUND and return a "
                "handle immediately — for dev servers, watchers, "
                "long builds/test runs you want to keep working "
                "past. Check progress with bash_output(handle); "
                "stop with bash_kill(handle). Use plain bash for "
                "anything under ~30s."
            ),
            # Runs arbitrary code — same safety contract as bash:
            # approval gate + allow/ask/deny rules + danger scan.
            destructive=True,
        )(bash_background),
        tool(
            name="bash_output",
            description=(
                "Status + output tail of a background process "
                "started with bash_background. Args: handle "
                "(e.g. 'bg1')."
            ),
        )(bash_output),
        tool(
            name="bash_kill",
            description=(
                "Terminate a background process (whole process "
                "group) started with bash_background. Args: handle."
            ),
        )(bash_kill),
    ]
