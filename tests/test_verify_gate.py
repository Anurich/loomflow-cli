"""Verify-before-done gate (loom_code.verify_gate).

Locks down: test-command detection never guesses (None when unsure —
'executable not found' is the #1 agent failure and the gate must not
cause it), the ran-tests recogniser, the all-done plan signal, and
the gate decision itself (edits + claim + no tests → nudge; anything
else → silent)."""

from __future__ import annotations

from pathlib import Path

from loom_code.verify_gate import (
    VERIFY_NUDGE,
    detect_test_command,
    plan_all_done,
    ran_tests,
    should_verify,
)

# ---- detect_test_command ---------------------------------------------


def test_pytest_via_pyproject_tool_section(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths=['tests']\n"
    )
    assert detect_test_command(tmp_path) == "pytest -q"


def test_pytest_via_tests_dir_with_packaging(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "tests").mkdir()
    assert detect_test_command(tmp_path) == "pytest -q"


def test_bare_tests_dir_without_packaging_is_not_enough(
    tmp_path: Path,
) -> None:
    # A random folder named tests/ in a non-Python project must not
    # send the agent to run pytest that isn't installed.
    (tmp_path / "tests").mkdir()
    assert detect_test_command(tmp_path) is None


def test_npm_test_script(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(
        '{"scripts": {"test": "vitest run"}}'
    )
    assert detect_test_command(tmp_path) == "npm test"


def test_npm_placeholder_script_rejected(tmp_path: Path) -> None:
    # npm init's default test script exits 1 unconditionally.
    (tmp_path / "package.json").write_text(
        '{"scripts": {"test": "echo \\"Error: no test specified\\" '
        '&& exit 1"}}'
    )
    assert detect_test_command(tmp_path) is None


def test_cargo_and_go(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[package]\n")
    assert detect_test_command(tmp_path) == "cargo test"


def test_makefile_test_target(tmp_path: Path) -> None:
    (tmp_path / "Makefile").write_text("test:\n\tpytest\n")
    assert detect_test_command(tmp_path) == "make test"


def test_empty_project_is_none(tmp_path: Path) -> None:
    assert detect_test_command(tmp_path) is None


# ---- ran_tests --------------------------------------------------------


def test_ran_tests_detects_pytest() -> None:
    assert ran_tests(["ls -la", ".venv/bin/pytest -q tests/"])


def test_ran_tests_detects_npm_variants() -> None:
    assert ran_tests(["npm run test -- --watch=false"])


def test_ran_tests_false_on_ordinary_commands() -> None:
    assert not ran_tests(["git status", "cat README.md"])


def test_ran_tests_empty() -> None:
    assert not ran_tests([])


# ---- plan_all_done ----------------------------------------------------


def test_plan_all_done_true() -> None:
    assert plan_all_done(
        [{"status": "done"}, {"status": "skipped"}, {"status": "DONE"}]
    )


def test_plan_with_open_step_false() -> None:
    assert not plan_all_done([{"status": "done"}, {"status": "doing"}])


def test_empty_or_missing_plan_false() -> None:
    assert not plan_all_done([])
    assert not plan_all_done(None)


# ---- should_verify -----------------------------------------------------


def test_gate_fires_on_edit_plus_claim_without_tests() -> None:
    assert should_verify(
        claims_done=True,
        files_touched=["a.py"],
        bash_commands=["git diff"],
    )


def test_gate_quiet_when_tests_ran() -> None:
    assert not should_verify(
        claims_done=True,
        files_touched=["a.py"],
        bash_commands=["pytest -q"],
    )


def test_gate_quiet_on_qa_turn_no_edits() -> None:
    assert not should_verify(
        claims_done=True, files_touched=[], bash_commands=[]
    )


def test_gate_quiet_without_completion_claim() -> None:
    assert not should_verify(
        claims_done=False,
        files_touched=["a.py"],
        bash_commands=[],
    )


def test_nudge_mentions_command() -> None:
    assert "pytest -q" in VERIFY_NUDGE.format(cmd="pytest -q")
