"""Tests for the anti-poison gate — the fix for the "all fixed"
memory doom loop.

The loop: agent claims completion with zero tool calls → episode
persisted → semantic recall surfaces it on the next "fix X" prompt
→ model parrots it → new episode → self-reinforcing. The gate
deletes the just-persisted episode when a turn made NO tool calls
AND the output is a bare completion claim.

Two pure helpers are unit-tested here:
  - _looks_like_completion_claim — the classifier
  - _delete_last_episode — the surgical sqlite delete
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from loom_code.repl import (
    _delete_last_episode,
    _looks_like_completion_claim,
)

# ---- _looks_like_completion_claim -----------------------------------


def test_detects_all_issues_fixed_variants() -> None:
    """The exact phrasings observed poisoning production memory."""
    for text in (
        "All the detected issues have already been fixed in this "
        "codebase: Missing await ...",
        "All the previously detected issues have already been fixed.",
        "All the detected issues were fixed in the last updates.",
        "There are no remaining issues or blockers.",
    ):
        assert _looks_like_completion_claim(text), (
            f"should flag as completion claim: {text!r}"
        )


def test_does_not_flag_legitimate_answers() -> None:
    """Normal no-tool answers must NOT be flagged — only bare
    completion claims. A review that mentions 'fixed' once in
    passing is fine."""
    for text in (
        "This code is a TUI editor built on loomflow.",
        "The grep_file function uses shell=True which is a "
        "shell-injection risk.",
        "Scenario 1 (hardcoded key) was fixed by using os.getenv, "
        "but grep_file is still vulnerable to shell injection.",
        "Please specify which file you want me to check.",
        "",
    ):
        assert not _looks_like_completion_claim(text), (
            f"should NOT flag: {text!r}"
        )


def test_empty_and_none_safe() -> None:
    assert _looks_like_completion_claim("") is False


# ---- _delete_last_episode -------------------------------------------


def _seed(
    db: Path, session_id: str, input_text: str, output_text: str
) -> None:
    with sqlite3.connect(str(db)) as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                input TEXT NOT NULL,
                output TEXT NOT NULL,
                embedding BLOB
            )
        """)
        # Monotonic occurred_at via row count so "most recent" is
        # deterministic.
        cur.execute("SELECT COUNT(*) FROM episodes")
        n = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO episodes (id, session_id, user_id, "
            "occurred_at, input, output) VALUES (?, ?, ?, ?, ?, ?)",
            (
                f"ep{n}",
                session_id,
                "loom-code",
                f"2026-05-21T00:00:{n:02d}+00:00",
                input_text,
                output_text,
            ),
        )
        conn.commit()


def _count(db: Path) -> int:
    with sqlite3.connect(str(db)) as conn:
        return conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]


def test_delete_last_removes_only_most_recent(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _seed(db, "s1", "first", "good answer 1")
    _seed(db, "s1", "second", "good answer 2")
    _seed(db, "s1", "third", "all the issues have been fixed")

    assert _count(db) == 3
    deleted = _delete_last_episode(db, session_id="s1", user_id="loom-code")
    assert deleted is True
    assert _count(db) == 2
    # The two good answers survive; only the most-recent (the
    # poison) was removed.
    with sqlite3.connect(str(db)) as conn:
        outputs = [
            r[0] for r in conn.execute("SELECT output FROM episodes")
        ]
    assert "good answer 1" in outputs
    assert "good answer 2" in outputs
    assert not any("fixed" in o for o in outputs)


def test_delete_last_scoped_to_session(tmp_path: Path) -> None:
    """Deleting the last episode for session A must not touch
    session B."""
    db = tmp_path / "memory.db"
    _seed(db, "A", "a1", "answer a")
    _seed(db, "B", "b1", "answer b")  # most recent overall, but session B

    _delete_last_episode(db, session_id="A", user_id="loom-code")
    with sqlite3.connect(str(db)) as conn:
        sessions = [
            r[0] for r in conn.execute("SELECT session_id FROM episodes")
        ]
    # Session A's only episode gone; B untouched.
    assert "A" not in sessions
    assert "B" in sessions


def test_delete_last_noop_on_missing_db(tmp_path: Path) -> None:
    assert _delete_last_episode(
        tmp_path / "nope.db", session_id="s", user_id="u"
    ) is False


def test_delete_last_noop_when_no_matching_episode(
    tmp_path: Path,
) -> None:
    db = tmp_path / "memory.db"
    _seed(db, "other", "x", "y")
    assert _delete_last_episode(
        db, session_id="nonexistent", user_id="loom-code"
    ) is False
    assert _count(db) == 1  # the unrelated episode untouched
