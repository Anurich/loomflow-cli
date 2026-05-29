"""Kernel-sandboxed bash for the coder — Claude-Code-style.

The coder's ``bash`` runs arbitrary shell, the one genuinely dangerous
tool (``edit`` / ``write`` only write where the model says, gated by the
approval prompt). So — like Claude Code, and unlike the framework's
``OSSandbox`` which ships a *Python function* to a child — we sandbox at
the **command** boundary: wrap the shell command string in the
platform's isolation wrapper.

* macOS  -> ``sandbox-exec -p '<profile>' /bin/bash -lc "<cmd>"``
* Linux  -> ``bwrap <binds> /bin/bash -lc "<cmd>"``
* else   -> plain bash + a one-time warning (no kernel backend).

The policy: **deny writes outside the project root, deny network by
default.** Reads stay broad (a build/test needs to read system libs).
This is the right risk model — arbitrary shell can't exfiltrate or
tamper outside the workspace, while the structured file tools keep their
existing approval gate.

Wrapping the command (not a pickled callable) sidesteps the
picklability constraint that makes ``OSSandbox`` unsuitable for
loom-code's closure-based builtin tools.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import anyio
from loomflow import tool
from loomflow.tools.registry import Tool


def _detect_backend() -> str:
    """``"seatbelt"`` | ``"bubblewrap"`` | ``"none"`` for this host."""
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        return "seatbelt"
    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return "bubblewrap"
    return "none"


def _seatbelt_profile(root: Path, allow_network: bool) -> str:
    """Deny-by-default Seatbelt profile: read anywhere, write only under
    ``root`` + the OS temp dir, network gated by the flag.

    Mirrors loomflow's OSSandbox profile (read-broad is required so the
    shell + build tools can load system libs; the enforced properties
    are no-write-outside-root + network control)."""
    tmp = Path(tempfile.gettempdir()).resolve()
    lines = [
        "(version 1)",
        ";; loom-code sandboxed bash — deny by default.",
        "(deny default)",
        ";; Reads broad: shell + build/test tools must load system libs.",
        "(allow file-read*)",
        "(allow process-fork)",
        "(allow process-exec)",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        "(allow signal (target self))",
        f'(allow file-write* (subpath "{root}"))',
        f'(allow file-write* (subpath "{tmp}"))',
        # /dev/null and friends — writes the shell expects.
        '(allow file-write* (subpath "/dev"))',
    ]
    lines.append(
        "(allow network*)" if allow_network
        else ";; network denied (default)."
    )
    return "\n".join(lines) + "\n"


def _bubblewrap_argv(root: Path, allow_network: bool) -> list[str]:
    """``bwrap`` argv: fresh namespace, read-only system, read-write
    bind for ``root`` + a private /tmp, network dropped unless allowed."""
    argv = [
        "bwrap",
        "--die-with-parent",
        "--unshare-pid",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/bin", "/bin",
        "--ro-bind", "/lib", "/lib",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
    ]
    if Path("/lib64").exists():
        argv += ["--ro-bind", "/lib64", "/lib64"]
    if Path("/etc").exists():
        argv += ["--ro-bind", "/etc", "/etc"]
    argv += ["--bind", str(root), str(root)]
    if not allow_network:
        argv += ["--unshare-net"]
    return argv


def sandboxed_bash_tool(
    root: str | Path,
    *,
    allow_network: bool = False,
    timeout: float = 300.0,
) -> Tool:
    """A ``bash`` tool whose command runs under OS isolation.

    Drop-in replacement for ``loomflow.tools.bash_tool`` on the coder:
    same name/signature, but the command executes inside ``sandbox-exec``
    (macOS) / ``bwrap`` (Linux) with writes confined to ``root`` and
    network denied unless ``allow_network=True``. On a host with no
    backend it degrades to plain bash and warns once.
    """
    root_path = Path(root).resolve()
    backend = _detect_backend()

    @tool(name="bash")
    async def sandboxed_bash(command: str) -> str:
        """Run a shell command. It executes inside an OS sandbox:
        it may read anywhere but can only WRITE under the project
        root, and has NO network access (unless the session enabled
        it). Use it for builds, tests, git, and file inspection."""
        if backend == "seatbelt":
            profile = _seatbelt_profile(root_path, allow_network)
            fd, prof_path = tempfile.mkstemp(suffix=".sb", prefix="loomsb-")
            import os

            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(profile)
            argv = [
                "sandbox-exec", "-f", prof_path,
                "/bin/bash", "-lc", command,
            ]
        elif backend == "bubblewrap":
            prof_path = None
            argv = [
                *_bubblewrap_argv(root_path, allow_network),
                "/bin/bash", "-lc", command,
            ]
        else:
            # No kernel backend (Windows / Linux w/o bwrap). Run plain
            # bash; the output is prefixed with a not-isolated warning
            # in the result-assembly below so the model + user know.
            prof_path = None
            argv = ["/bin/bash", "-lc", command]

        try:
            with anyio.fail_after(timeout):
                proc = await anyio.run_process(
                    argv, check=False, cwd=str(root_path)
                )
        except TimeoutError:
            return f"[sandboxed bash] timed out after {timeout}s"
        finally:
            if prof_path is not None:
                import os

                try:
                    os.unlink(prof_path)
                except OSError:
                    pass

        out = (proc.stdout or b"").decode("utf-8", "replace")
        err = (proc.stderr or b"").decode("utf-8", "replace")
        body = out + (("\n" + err) if err else "")
        if backend == "none":
            body = (
                "[warning: no OS sandbox backend on this host — command "
                "ran WITHOUT kernel isolation]\n" + body
            )
        if proc.returncode != 0:
            body = f"[exit {proc.returncode}]\n{body}"
        return body

    return sandboxed_bash
