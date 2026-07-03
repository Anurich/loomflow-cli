"""Tests for the Claude-Code-style sandboxed bash tool.

Two layers:

* **Construction / wiring** — backend-agnostic. The tool is built, named
  ``bash``, and ``build_workers(sandbox=True)`` swaps the coder onto it.
  These run everywhere (CI included).
* **Kernel guarantees** — only meaningful where a real backend exists
  (Seatbelt on macOS, bwrap on Linux). Gated by ``requires_backend`` so
  they SKIP rather than give a false pass on a host with no isolation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from loom_code.project import detect_project
from loom_code.sandboxed_bash import (
    _bubblewrap_argv,
    _detect_backend,
    _seatbelt_profile,
    sandboxed_bash_tool,
)
from loom_code.workers import build_workers

pytestmark = pytest.mark.anyio

# Kernel-guarantee tests are honest only with a real backend — skip
# (don't xfail/pass) on a host without one, so a plain-bash fallback
# never reads as a green security test.
requires_backend = pytest.mark.skipif(
    _detect_backend() == "none",
    reason="no OS sandbox backend (Seatbelt/bwrap) on this host",
)


@pytest.fixture
def tmp_root() -> Path:
    root = Path(tempfile.mkdtemp(prefix="loomsbtest-"))
    (root / "in_root.txt").write_text("hello\n", encoding="utf-8")
    return root


# ── construction / wiring (backend-agnostic) ──────────────────────


def test_tool_is_named_bash(tmp_root: Path) -> None:
    # It must be a drop-in for loomflow's bash_tool — same name, or the
    # model won't recognise it and the swap silently breaks the coder.
    bash = sandboxed_bash_tool(tmp_root)
    assert bash.name == "bash"


def test_detect_backend_value_is_known() -> None:
    assert _detect_backend() in {"seatbelt", "bubblewrap", "none"}


def test_build_workers_swaps_coder_bash_when_sandboxed(
    tmp_root: Path,
) -> None:
    proj = detect_project(tmp_root)
    sandboxed = build_workers(proj, model="echo", sandbox=True)
    plain = build_workers(proj, model="echo", sandbox=False)

    coder_sb = sandboxed["coder"]._tool_host._tools["bash"]
    coder_plain = plain["coder"]._tool_host._tools["bash"]

    assert coder_sb.fn.__module__ == "loom_code.sandboxed_bash"
    assert coder_plain.fn.__module__ != "loom_code.sandboxed_bash"


def test_seatbelt_profile_confines_writes_to_root(tmp_root: Path) -> None:
    prof = _seatbelt_profile(tmp_root, allow_network=False)
    assert "(deny default)" in prof
    assert "(allow file-read*)" in prof  # reads stay broad
    assert f'(allow file-write* (subpath "{tmp_root}"))' in prof
    assert "(allow network*)" not in prof  # denied by default


def test_seatbelt_profile_network_gated_by_flag(tmp_root: Path) -> None:
    assert "(allow network*)" in _seatbelt_profile(tmp_root, True)
    assert "(allow network*)" not in _seatbelt_profile(tmp_root, False)


def test_bubblewrap_argv_binds_root_and_drops_net(tmp_root: Path) -> None:
    argv = _bubblewrap_argv(tmp_root, allow_network=False)
    assert "bwrap" in argv
    # root is bound read-write...
    assert "--bind" in argv
    i = argv.index("--bind")
    assert argv[i + 1] == str(tmp_root)
    # ...and the network namespace is unshared (no net).
    assert "--unshare-net" in argv


def test_bubblewrap_argv_keeps_net_when_allowed(tmp_root: Path) -> None:
    argv = _bubblewrap_argv(tmp_root, allow_network=True)
    assert "--unshare-net" not in argv


# ── kernel guarantees (require a real backend) ────────────────────


@requires_backend
async def test_can_write_inside_root(tmp_root: Path) -> None:
    bash = sandboxed_bash_tool(tmp_root)
    out = await bash.fn(
        command="echo sandboxed > made.txt && cat made.txt"
    )
    assert "sandboxed" in out
    assert (tmp_root / "made.txt").exists()


@requires_backend
async def test_write_outside_root_is_blocked(tmp_root: Path) -> None:
    # The whole point: arbitrary shell cannot tamper outside the
    # workspace. Target the user's HOME — neither the project root nor
    # the OS temp dir (which the policy intentionally allows so build
    # tools work), so a successful write here is a real escape.
    outside = Path.home() / ".loom_sandbox_escape_probe.txt"
    if outside.exists():  # pre-clean from a prior failed run
        outside.unlink()
    bash = sandboxed_bash_tool(tmp_root)
    out = await bash.fn(command=f"echo pwned > {outside}")
    escaped = outside.exists()
    if escaped:  # don't leave the probe behind
        outside.unlink()
    assert not escaped, f"sandbox escape: wrote {outside}\n{out}"


@requires_backend
async def test_network_is_blocked_by_default(tmp_root: Path) -> None:
    bash = sandboxed_bash_tool(tmp_root, allow_network=False)
    # A DNS/connect attempt must fail under the no-network policy. Use a
    # short timeout so a (wrongly) reachable network doesn't hang the test.
    #
    # The sentinel is CONCATENATED inside the probe ("CONN"+"ECTED"):
    # Python 3.13+ tracebacks echo the failing SOURCE LINE, so a literal
    # print("CONNECTED") put the sentinel into the output even when the
    # connect was correctly refused (observed on the CI macOS runner's
    # python 3.14). The joined string can only appear if connect()
    # genuinely succeeded and the print ran.
    out = await bash.fn(
        command=(
            "python3 -c 'import socket,sys; "
            "s=socket.create_connection((\"1.1.1.1\",53),timeout=4); "
            "print(\"CONN\"+\"ECTED\"); s.close()' 2>&1 || echo BLOCKED"
        )
    )
    assert "CONNECTED" not in out, f"network not blocked:\n{out}"
