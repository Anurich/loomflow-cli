"""Tests for loom_code.rules — the AGENTS.md rules engine."""

from __future__ import annotations

from pathlib import Path

from loom_code import rules


def _managed(path: Path) -> list[str]:
    return rules._read_managed_rules(path.read_text(encoding="utf-8"))


def test_init_creates_agents_md_when_absent(tmp_path: Path) -> None:
    path, created = rules.init_agents_md(tmp_path)
    assert created is True
    assert path.name == "AGENTS.md"
    body = path.read_text(encoding="utf-8")
    assert rules.BLOCK_START in body
    assert rules.BLOCK_END in body


def test_init_is_noop_when_context_file_exists(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("# rules\n", encoding="utf-8")
    path, created = rules.init_agents_md(tmp_path)
    assert created is False
    assert path.name == "CLAUDE.md"


def test_add_rule_appends_into_managed_block(tmp_path: Path) -> None:
    rules.init_agents_md(tmp_path)
    rules.add_rule(tmp_path, "Never edit test.py")
    assert _managed(tmp_path / "AGENTS.md") == ["Never edit test.py"]


def test_add_rule_dedups_regardless_of_wording(tmp_path: Path) -> None:
    rules.add_rule(tmp_path, "Never edit test.py")
    # different case + trailing punctuation → same normalized key
    msg = rules.add_rule(tmp_path, "never edit test.py.")
    assert "already recorded" in msg.lower()
    assert _managed(tmp_path / "AGENTS.md") == ["Never edit test.py"]


def test_supersedes_replaces_not_stacks(tmp_path: Path) -> None:
    rules.add_rule(tmp_path, "Never edit test.py")
    rules.add_rule(tmp_path, "Always run pytest before commit")
    rules.add_rule(
        tmp_path, "test.py may now be edited", supersedes="Never edit test.py"
    )
    managed = _managed(tmp_path / "AGENTS.md")
    assert "Never edit test.py" not in managed
    assert "test.py may now be edited" in managed
    # an unrelated rule is left intact
    assert "Always run pytest before commit" in managed


def test_human_content_outside_block_untouched(tmp_path: Path) -> None:
    path = tmp_path / "AGENTS.md"
    path.write_text(
        "# AGENTS.md\n\n## Conventions\n- 2-space indent\n\n"
        f"## Rules\n{rules.BLOCK_START}\n{rules.BLOCK_END}\n",
        encoding="utf-8",
    )
    rules.add_rule(tmp_path, "Never delete the db")
    body = path.read_text(encoding="utf-8")
    assert "## Conventions" in body
    assert "- 2-space indent" in body
    assert "- Never delete the db" in body


def test_add_rule_creates_file_when_absent(tmp_path: Path) -> None:
    # no init first — add_rule should still create AGENTS.md
    rules.add_rule(tmp_path, "Use tabs")
    assert (tmp_path / "AGENTS.md").is_file()
    assert _managed(tmp_path / "AGENTS.md") == ["Use tabs"]


def test_empty_rule_is_rejected(tmp_path: Path) -> None:
    msg = rules.add_rule(tmp_path, "   ")
    assert "empty" in msg.lower()
    assert rules.detect_rules_file(tmp_path) is None


def test_writes_into_existing_context_file(tmp_path: Path) -> None:
    # An existing CLAUDE.md is the write target (priority order), not a
    # second AGENTS.md.
    (tmp_path / "CLAUDE.md").write_text("# House rules\n", encoding="utf-8")
    rules.add_rule(tmp_path, "No force pushes")
    assert not (tmp_path / "AGENTS.md").exists()
    assert _managed(tmp_path / "CLAUDE.md") == ["No force pushes"]
