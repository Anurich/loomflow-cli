"""Custom slash commands (extensions.CommandSpec + discovery +
template expansion).

Contract: a markdown file in <scope>/commands/ becomes /<stem>;
project wins a name clash with user scope; $1-$9 then $ARGUMENTS
substitute (positionals first so user args containing $N survive);
empty bodies and unreadable files are skipped."""

from __future__ import annotations

from pathlib import Path

from loom_code.extensions import (
    discover,
    expand_command_template,
)

# ---- expand_command_template -------------------------------------------


def test_arguments_substitution() -> None:
    assert (
        expand_command_template("Review PR $ARGUMENTS carefully", "123")
        == "Review PR 123 carefully"
    )


def test_positional_substitution() -> None:
    assert (
        expand_command_template("compare $1 against $2", "a.py b.py")
        == "compare a.py against b.py"
    )


def test_missing_positionals_become_empty() -> None:
    assert expand_command_template("do $1 $2", "only") == "do only "


def test_user_args_containing_dollar_survive() -> None:
    # $2 inside the ARGUMENT string must not be re-substituted.
    out = expand_command_template("run $ARGUMENTS", "echo $2")
    assert out == "run echo $2"


def test_no_placeholders_appends_nothing() -> None:
    assert expand_command_template("fixed prompt", "ignored") == (
        "fixed prompt"
    )


# ---- discovery -----------------------------------------------------------


def _write_cmd(base: Path, name: str, text: str) -> None:
    d = base / "commands"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.md").write_text(text, encoding="utf-8")


def test_discovers_user_and_project_commands(tmp_path: Path) -> None:
    user = tmp_path / "user"
    proj = tmp_path / "proj"
    (proj / ".loom").mkdir(parents=True)
    _write_cmd(
        user,
        "review-pr",
        "---\ndescription: review a PR\n---\nReview PR $1.",
    )
    _write_cmd(proj / ".loom", "ship", "Run the release checklist.")
    ext = discover(proj, user_dir=user)
    names = {c.name for c in ext.command_specs}
    assert names == {"review-pr", "ship"}
    by = {c.name: c for c in ext.command_specs}
    assert by["review-pr"].description == "review a PR"
    assert by["review-pr"].source == "user"
    assert by["ship"].source == "project"
    # description falls back to the first body line
    assert by["ship"].description.startswith("Run the release")


def test_project_wins_name_clash(tmp_path: Path) -> None:
    user = tmp_path / "user"
    proj = tmp_path / "proj"
    (proj / ".loom").mkdir(parents=True)
    _write_cmd(user, "deploy", "user version")
    _write_cmd(proj / ".loom", "deploy", "project version")
    ext = discover(proj, user_dir=user)
    (spec,) = [c for c in ext.command_specs if c.name == "deploy"]
    assert spec.template == "project version"
    assert spec.source == "project"


def test_empty_body_skipped(tmp_path: Path) -> None:
    user = tmp_path / "user"
    proj = tmp_path / "proj"
    proj.mkdir()
    _write_cmd(user, "blank", "---\ndescription: nothing\n---\n\n")
    ext = discover(proj, user_dir=user)
    assert not ext.command_specs


def test_no_commands_dir_is_fine(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    ext = discover(proj, user_dir=tmp_path / "user")
    assert ext.command_specs == []
