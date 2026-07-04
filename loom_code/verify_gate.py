"""Verify-before-done — the gate that turns "I finished" into
"I finished AND the tests agree".

Terminal-Bench evidence: a pre-completion verification pass
(middleware forcing a build/test run before the agent may exit) was
one of the measured harness wins behind deepagents-cli's +13.7-point
jump, and "high performers don't make fewer mistakes, they recover
from them" is the recurring scaffold finding. loom-code already
PREVENTS false-done claims from poisoning memory (the anti-poison
gate deletes unverified "done" episodes); this module goes one step
earlier and makes the claim true: if a turn edited code, claims
completion, and never ran the project's tests, the REPL sends one
bounded nudge telling the model to run them.

Pure helpers here (detection + decision); ``repl.py`` owns the nudge
mechanics (same machinery as the tool-leak nudge) and the ``/verify``
toggle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Substrings that mark a bash command as "running the tests". Checked
# against every command the turn executed — if any matches, the turn
# already verified itself and the gate stays quiet.
_TEST_RUNNER_TOKENS = (
    "pytest",
    "npm test",
    "npm run test",
    "yarn test",
    "pnpm test",
    "bun test",
    "cargo test",
    "go test",
    "make test",
    "make check",
    "just test",
    "unittest",
    "jest",
    "vitest",
    "tox",
    "rspec",
    "mvn test",
    "gradle test",
    "./gradlew test",
)


def detect_test_command(root: Path) -> str | None:
    """The project's canonical test command, or ``None`` when we can't
    tell (the gate then stays out of the way — never guess a command
    that might not exist; 'executable not found' is the #1 agent
    failure on Terminal-Bench and the gate must not cause it)."""
    try:
        pyproject = root / "pyproject.toml"
        if (
            (root / "pytest.ini").is_file()
            or (root / "conftest.py").is_file()
            or (
                pyproject.is_file()
                and "[tool.pytest" in pyproject.read_text(
                    encoding="utf-8", errors="ignore"
                )
            )
            or (
                (root / "tests").is_dir()
                and (
                    pyproject.is_file()
                    or (root / "setup.py").is_file()
                    or (root / "setup.cfg").is_file()
                )
            )
        ):
            return "pytest -q"
        pkg = root / "package.json"
        if pkg.is_file():
            text = pkg.read_text(encoding="utf-8", errors="ignore")
            # npm init's placeholder script fails by design — a
            # project carrying it has NO tests; don't send the agent
            # to run a command that exits 1 unconditionally.
            if '"test"' in text and "no test specified" not in text:
                return "npm test"
        if (root / "Cargo.toml").is_file():
            return "cargo test"
        if (root / "go.mod").is_file():
            return "go test ./..."
        for mk in ("Makefile", "makefile"):
            f = root / mk
            if f.is_file() and "test:" in f.read_text(
                encoding="utf-8", errors="ignore"
            ):
                return "make test"
        for jf in ("justfile", "Justfile", ".justfile"):
            f = root / jf
            if f.is_file() and "test" in f.read_text(
                encoding="utf-8", errors="ignore"
            ):
                return "just test"
    except OSError:
        return None
    return None


def ran_tests(bash_commands: list[str]) -> bool:
    """True when any command this turn looks like a test run."""
    for cmd in bash_commands:
        low = cmd.lower()
        if any(tok in low for tok in _TEST_RUNNER_TOKENS):
            return True
    return False


def plan_all_done(plan: list[dict[str, Any]] | None) -> bool:
    """True when a non-empty plan has every step done/skipped —
    the structured completion signal (vs the prose claim)."""
    if not plan:
        return False
    return all(
        str(step.get("status", "")).lower() in ("done", "skipped")
        for step in plan
    )


def should_verify(
    *,
    claims_done: bool,
    files_touched: list[str],
    bash_commands: list[str],
) -> bool:
    """The gate's decision: nudge iff the turn CHANGED code, CLAIMS
    completion, and never RAN tests. Q&A turns (no edits) and turns
    that already tested themselves pass through untouched."""
    return bool(
        claims_done and files_touched and not ran_tests(bash_commands)
    )


VERIFY_NUDGE = (
    "Hold on — you made code changes this turn but never ran the "
    "project's tests, and you're presenting the work as done. Run "
    "`{cmd}` now. If everything passes, say so in one line. If "
    "anything fails, fix it and re-run until green — or state "
    "precisely which failures are pre-existing and out of scope."
)
