"""Session tree — /fork's episode copy + /tree's rendering.

The contract: a fork inherits the parent's full history (episodes +
tool transcripts) under a fresh session_id while the parent's rows
stay untouched; the tree renders forks under their parents with the
current session marked."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from loom_code.repl import _fork_episodes, _render_session_tree

# ---- _fork_episodes ---------------------------------------------------


def _seed(db: Path, session_id: str, n: int = 2) -> None:
    with sqlite3.connect(str(db)) as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                user_id TEXT, occurred_at REAL NOT NULL,
                input TEXT NOT NULL, output TEXT NOT NULL,
                embedding BLOB
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS episode_tool_transcripts (
                episode_id TEXT NOT NULL, sequence INTEGER NOT NULL,
                message_json TEXT NOT NULL,
                PRIMARY KEY (episode_id, sequence)
            )
        """)
        for i in range(n):
            cur.execute(
                "INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    f"ep-{session_id}-{i}",
                    session_id,
                    "loom-code",
                    1000.0 + i,
                    f"prompt {i}",
                    f"answer {i}",
                    None,
                ),
            )
            cur.execute(
                "INSERT INTO episode_tool_transcripts VALUES (?, ?, ?)",
                (
                    f"ep-{session_id}-{i}",
                    0,
                    '{"role":"tool","content":"r"}',
                ),
            )
        conn.commit()


def _count(db: Path, table: str, session_col: str, sid: str) -> int:
    with sqlite3.connect(str(db)) as conn:
        if table == "episodes":
            return conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE session_id=?",
                (sid,),
            ).fetchone()[0]
        return conn.execute(
            "SELECT COUNT(*) FROM episode_tool_transcripts t "
            "JOIN episodes e ON e.id=t.episode_id "
            "WHERE e.session_id=?",
            (sid,),
        ).fetchone()[0]


def test_fork_copies_episodes_and_transcripts(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _seed(db, "PARENT", n=3)
    copied = _fork_episodes(db, "PARENT", "CHILD")
    assert copied == 3
    assert _count(db, "episodes", "session_id", "CHILD") == 3
    assert _count(db, "transcripts", "", "CHILD") == 3
    # parent untouched
    assert _count(db, "episodes", "session_id", "PARENT") == 3


def test_fork_of_empty_session_is_zero(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _seed(db, "OTHER", n=1)
    assert _fork_episodes(db, "NOTHING", "CHILD") == 0


def test_fork_missing_db_is_zero(tmp_path: Path) -> None:
    assert _fork_episodes(tmp_path / "nope.db", "A", "B") == 0


def test_fork_is_idempotent(tmp_path: Path) -> None:
    # Re-forking to the same child (retry after crash) must not
    # duplicate rows — INSERT OR IGNORE on derived ids.
    db = tmp_path / "memory.db"
    _seed(db, "PARENT", n=2)
    _fork_episodes(db, "PARENT", "CHILD")
    _fork_episodes(db, "PARENT", "CHILD")
    assert _count(db, "episodes", "session_id", "CHILD") == 2


# ---- _render_session_tree ----------------------------------------------


def _rec(sid: str, parent: str | None = None, hint: str = "") -> dict:
    r = {"session_id": sid, "ts": "2026-07-04T10:00:00", "hint": hint}
    if parent:
        r["parent"] = parent
    return r


def test_tree_indents_fork_under_parent() -> None:
    lines = _render_session_tree(
        [_rec("AAAA1111"), _rec("BBBB2222", parent="AAAA1111")],
        current_session_id="BBBB2222",
    )
    assert len(lines) == 2
    assert lines[0].startswith("○ AAAA1111"[0])  # root not indented
    assert "└─" in lines[1]  # child indented
    assert "you are here" in lines[1]


def test_tree_marks_current_root() -> None:
    lines = _render_session_tree(
        [_rec("AAAA1111")], current_session_id="AAAA1111"
    )
    assert "●" in lines[0] and "you are here" in lines[0]


def test_tree_orphan_parent_becomes_root() -> None:
    # A fork whose parent predates the log still renders (as root).
    lines = _render_session_tree(
        [_rec("CCCC3333", parent="GONE")], current_session_id="X"
    )
    assert len(lines) == 1
    assert "CCCC3333"[:8] in lines[0]


def test_tree_multi_level() -> None:
    lines = _render_session_tree(
        [
            _rec("R0000000"),
            _rec("C1111111", parent="R0000000"),
            _rec("G2222222", parent="C1111111"),
        ],
        current_session_id="G2222222",
    )
    assert len(lines) == 3
    # grandchild is deeper-indented than child
    assert lines[2].index("└─") > lines[1].index("└─")
