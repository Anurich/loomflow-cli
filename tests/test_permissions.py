"""Permission rules + modes (loom_code.permissions).

The gate used to be binary y/n/allow-all; these tests pin the layered
policy: deny > ask > allow rules, then the session mode's default —
and that deny is absolute (not even yolo bypasses it).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loom_code.permissions import (
    Decision,
    Mode,
    Rules,
    call_target,
    decide,
    load_rules,
    parse_mode,
)

# ---- decide: rule precedence ----------------------------------------------


def test_deny_beats_allow_and_yolo() -> None:
    rules = Rules(allow=["bash(*)"], deny=["bash(rm*)"])
    assert (
        decide("bash", {"command": "rm -rf build"}, rules, Mode.YOLO)
        is Decision.DENY
    )


def test_allow_rule_skips_prompt() -> None:
    rules = Rules(allow=["bash(pytest*)"])
    assert (
        decide("bash", {"command": "pytest -q"}, rules, Mode.DEFAULT)
        is Decision.ALLOW
    )
    # ...but an unmatched command still asks.
    assert (
        decide("bash", {"command": "pip install x"}, rules, Mode.DEFAULT)
        is Decision.ASK
    )


def test_ask_rule_forces_prompt_even_in_yolo() -> None:
    rules = Rules(ask=["edit(*prod*)"])
    assert (
        decide("edit", {"path": "config/prod.yaml"}, rules, Mode.YOLO)
        is Decision.ASK
    )


def test_env_file_deny_pattern() -> None:
    # The canonical "never touch .env" rule.
    rules = Rules(deny=["edit(*.env)", "write(*.env)"])
    for tool in ("edit", "write"):
        assert (
            decide(tool, {"path": ".env"}, rules, Mode.DEFAULT)
            is Decision.DENY
        )
    # Other files unaffected.
    assert (
        decide("edit", {"path": "main.py"}, Rules(), Mode.DEFAULT)
        is Decision.ASK
    )


def test_bare_tool_name_pattern_blankets_the_tool() -> None:
    rules = Rules(deny=["bash"])
    assert (
        decide("bash", {"command": "echo hi"}, rules, Mode.DEFAULT)
        is Decision.DENY
    )


# ---- decide: mode defaults --------------------------------------------------


def test_plan_mode_denies_all_mutation() -> None:
    for tool, args in (
        ("edit", {"path": "a.py"}),
        ("write", {"path": "b.py"}),
        ("bash", {"command": "ls"}),
    ):
        assert decide(tool, args, Rules(), Mode.PLAN) is Decision.DENY


def test_accept_edits_allows_edits_but_asks_for_bash() -> None:
    assert (
        decide("edit", {"path": "a.py"}, Rules(), Mode.ACCEPT_EDITS)
        is Decision.ALLOW
    )
    assert (
        decide("bash", {"command": "ls"}, Rules(), Mode.ACCEPT_EDITS)
        is Decision.ASK
    )


def test_default_mode_asks() -> None:
    assert (
        decide("edit", {"path": "a.py"}, Rules(), Mode.DEFAULT)
        is Decision.ASK
    )


# ---- helpers ----------------------------------------------------------------


def test_call_target_shapes() -> None:
    assert call_target("bash", {"command": "pytest -q"}) == "bash(pytest -q)"
    assert call_target("edit", {"path": "src/x.py"}) == "edit(src/x.py)"


def test_parse_mode_aliases() -> None:
    assert parse_mode("plan") is Mode.PLAN
    assert parse_mode("read-only") is Mode.PLAN
    assert parse_mode("accept-edits") is Mode.ACCEPT_EDITS
    assert parse_mode("YOLO") is Mode.YOLO
    assert parse_mode("nonsense") is None


# ---- load_rules -------------------------------------------------------------


def test_load_rules_merges_user_and_project(tmp_path: Path) -> None:
    user = tmp_path / "user"
    proj = tmp_path / "proj"
    user.mkdir()
    proj.mkdir()
    (user / "settings.toml").write_text(
        '[permissions]\nallow = ["bash(pytest*)"]\n'
    )
    (proj / "settings.toml").write_text(
        '[permissions]\ndeny = ["edit(*.env)"]\n'
    )
    rules = load_rules([user, proj])
    assert rules.allow == ["bash(pytest*)"]
    assert rules.deny == ["edit(*.env)"]


def test_load_rules_tolerates_garbage(tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "settings.toml").write_text("not [ valid = toml ((")
    missing = tmp_path / "missing"  # no dir at all
    rules = load_rules([bad, missing])
    assert rules.allow == [] and rules.deny == []


# ---- gate: precedence at the handler level --------------------------------


@pytest.mark.anyio
async def test_deny_rule_beats_session_allow_all() -> None:
    from types import SimpleNamespace

    from loom_code.approval import ApprovalGate

    gate = ApprovalGate(rules=Rules(deny=["bash(rm*)"]))
    gate._allow_all = True  # user pressed 'a' earlier
    call = SimpleNamespace(tool="bash", args={"command": "rm -rf build"})
    assert await gate.handler(call) is False


@pytest.mark.anyio
async def test_plan_mode_flat_denies_danger_command() -> None:
    # git push --force is a _DANGER_PATTERN; plan mode must DENY it
    # outright, not downgrade to a confirmable prompt.
    from types import SimpleNamespace

    from loom_code.approval import ApprovalGate

    gate = ApprovalGate(mode=Mode.PLAN)
    call = SimpleNamespace(
        tool="bash", args={"command": "git push --force"}
    )
    assert await gate.handler(call) is False


@pytest.mark.anyio
async def test_ask_rule_survives_allow_all() -> None:
    # An explicit ask rule must still prompt even after 'allow all';
    # non-TTY selector resolves to the safe last option → deny.
    from types import SimpleNamespace

    from loom_code.approval import ApprovalGate

    gate = ApprovalGate(rules=Rules(ask=["bash(git commit*)"]))
    gate._allow_all = True
    call = SimpleNamespace(
        tool="bash", args={"command": "git commit -m x"}
    )
    assert await gate.handler(call) is False


@pytest.mark.anyio
async def test_outside_project_edit_confirms_even_in_yolo(
    tmp_path: Path,
) -> None:
    # An edit to a file outside the project must ALWAYS confirm — even
    # in yolo — so an @-mentioned dotfile can't be silently mutated.
    # In-project edits auto-allow in accept-edits as normal.
    from types import SimpleNamespace

    from loom_code.approval import ApprovalGate

    proj = tmp_path / "proj"
    proj.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n")

    gate = ApprovalGate(mode=Mode.YOLO, project_root=proj)
    inside = SimpleNamespace(tool="edit", args={"path": "a.py"})
    assert await gate.handler(inside) is True  # in-project: allowed
    out = SimpleNamespace(tool="edit", args={"path": str(outside)})
    # non-TTY selector → safe last option → False (i.e. it PROMPTED)
    assert await gate.handler(out) is False
