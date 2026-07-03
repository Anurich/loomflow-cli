"""Tests for ``_migrate_legacy_per_route_episodes`` — the /resume
legacy-data migration that rekeys pre-0.10.18 per-route episodes
into the parent session_id so ``conversation_scope='shared'``
rehydration finds them.

Without this, a /resume to a pre-upgrade session loses all
conversational context because the new shared-mode lookup keys
on the parent session_id alone (episodes live under
``{parent}__route_simple`` / ``{parent}__route_complex``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from loom_code.repl import (
    _migrate_legacy_per_route_episodes,
    _scrub_prose_from_tool_transcripts,
)


def _seed_episode(
    db_path: Path,
    *,
    session_id: str,
    user_id: str = "loom-code",
    input_text: str = "hi",
    output_text: str = "ok",
) -> None:
    """Insert a fake episode row matching loomflow's sqlite schema
    (the bits this test cares about — id, session_id, user_id,
    input, output, occurred_at)."""
    with sqlite3.connect(str(db_path)) as conn:
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
        cur.execute(
            "INSERT INTO episodes (id, session_id, user_id, "
            "occurred_at, input, output) VALUES (?, ?, ?, ?, ?, ?)",
            (
                f"ep-{session_id}-{input_text[:8]}",
                session_id,
                user_id,
                "2026-05-18T00:00:00+00:00",
                input_text,
                output_text,
            ),
        )
        conn.commit()


def _count_by_session(db_path: Path) -> dict[str, int]:
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT session_id, COUNT(*) FROM episodes "
            "GROUP BY session_id"
        )
        return dict(cur.fetchall())


def test_migrate_rekeys_legacy_simple_and_complex(
    tmp_path: Path,
) -> None:
    """Both ``{parent}__route_simple`` and ``{parent}__route_complex``
    rows get rewritten to ``{parent}``. The shared session_id then
    has all the episodes from both routes — that's what makes
    /resume of a pre-upgrade session work."""
    db = tmp_path / "memory.db"
    parent = "01KRY3"
    _seed_episode(
        db, session_id=f"{parent}__route_simple", input_text="t1"
    )
    _seed_episode(
        db, session_id=f"{parent}__route_simple", input_text="t2"
    )
    _seed_episode(
        db, session_id=f"{parent}__route_complex", input_text="t3"
    )

    migrated = _migrate_legacy_per_route_episodes(db, parent)

    assert migrated == 3
    counts = _count_by_session(db)
    assert counts.get(parent) == 3
    assert f"{parent}__route_simple" not in counts
    assert f"{parent}__route_complex" not in counts


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    """A second /resume call must not double-migrate or fail. After
    the first call, no per-route rows remain; the second call
    returns 0 with the data unchanged."""
    db = tmp_path / "memory.db"
    parent = "abc"
    _seed_episode(db, session_id=f"{parent}__route_simple")
    assert _migrate_legacy_per_route_episodes(db, parent) == 1
    assert _migrate_legacy_per_route_episodes(db, parent) == 0
    assert _count_by_session(db) == {parent: 1}


def test_migrate_leaves_unrelated_sessions_alone(
    tmp_path: Path,
) -> None:
    """Episodes belonging to a DIFFERENT REPL session must not be
    touched. We only migrate routes derived from the specific
    parent session_id passed in — cross-session leakage would
    silently merge unrelated conversations."""
    db = tmp_path / "memory.db"
    _seed_episode(db, session_id="OTHER__route_simple", input_text="x")
    _seed_episode(db, session_id="MYPARENT__route_simple", input_text="y")
    _seed_episode(db, session_id="POSTUPGRADE", input_text="z")

    assert _migrate_legacy_per_route_episodes(db, "MYPARENT") == 1

    counts = _count_by_session(db)
    assert counts.get("MYPARENT") == 1
    assert counts.get("OTHER__route_simple") == 1
    assert counts.get("POSTUPGRADE") == 1


def test_migrate_noop_when_db_absent(tmp_path: Path) -> None:
    """Resumes on projects with no memory.db (fresh install, no
    prior loom-code use) must succeed silently — returning 0,
    never raising. /resume calls this unconditionally and a raise
    here would block resume of perfectly valid sessions."""
    missing = tmp_path / "does-not-exist.db"
    assert _migrate_legacy_per_route_episodes(missing, "anything") == 0


def test_migrate_noop_on_post_upgrade_session(
    tmp_path: Path,
) -> None:
    """A session created AFTER the 0.10.18 upgrade already has
    episodes under the parent session_id directly. Migration is a
    no-op — no per-route rows exist to rekey."""
    db = tmp_path / "memory.db"
    _seed_episode(db, session_id="NEW", input_text="x")
    _seed_episode(db, session_id="NEW", input_text="y")
    assert _migrate_legacy_per_route_episodes(db, "NEW") == 0
    assert _count_by_session(db) == {"NEW": 2}


@pytest.mark.parametrize("bad_path", [Path("/dev/null/cant-open")])
def test_migrate_swallows_db_errors(bad_path: Path) -> None:
    """A corrupted / unreadable db must not block /resume. The
    migration is a best-effort assist; the user should still get
    their session swap even if migration fails."""
    # bad_path: parent is /dev/null/, not a dir → sqlite3.connect
    # will refuse to create the file there.
    result = _migrate_legacy_per_route_episodes(bad_path, "x")
    assert result == 0


# ---- transcript prose scrub (loomflow < 0.10.30 leak) ----------------


def _seed_transcript_row(
    db_path: Path,
    *,
    episode_id: str,
    sequence: int,
    message_json: str,
) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS episode_tool_transcripts (
                episode_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                message_json TEXT NOT NULL,
                PRIMARY KEY (episode_id, sequence)
            )
        """)
        cur.execute(
            "INSERT INTO episode_tool_transcripts VALUES (?, ?, ?)",
            (episode_id, sequence, message_json),
        )
        conn.commit()


def _transcript_count(db_path: Path) -> int:
    with sqlite3.connect(str(db_path)) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM episode_tool_transcripts"
        ).fetchone()[0]


def test_scrub_removes_leaked_prose_keeps_tool_work(
    tmp_path: Path,
) -> None:
    """The loomflow <0.10.30 capture leak: a resumed session's
    transcript contains a prior turn's assistant answer and the
    run's own user prompt. The scrub deletes exactly those and
    keeps legitimate tool work (tool results + assistant messages
    carrying tool_calls)."""
    db = tmp_path / "memory.db"
    _seed_episode(db, session_id="S", input_text="what today")
    ep_id = "ep-S-what tod"
    # leaked prose (must go)
    _seed_transcript_row(
        db, episode_id=ep_id, sequence=0,
        message_json='{"role":"assistant","content":"prior answer",'
        '"tool_calls":[]}',
    )
    _seed_transcript_row(
        db, episode_id=ep_id, sequence=1,
        message_json='{"role":"user","content":"what today"}',
    )
    # legitimate tool work (must stay)
    _seed_transcript_row(
        db, episode_id=ep_id, sequence=2,
        message_json='{"role":"assistant","content":"",'
        '"tool_calls":[{"id":"t1","tool":"grep","args":{}}]}',
    )
    _seed_transcript_row(
        db, episode_id=ep_id, sequence=3,
        message_json='{"role":"tool","content":"match",'
        '"tool_call_id":"t1"}',
    )
    assert _scrub_prose_from_tool_transcripts(db, "S") == 2
    assert _transcript_count(db) == 2
    # idempotent
    assert _scrub_prose_from_tool_transcripts(db, "S") == 0


def test_scrub_scoped_to_session(tmp_path: Path) -> None:
    """Rows belonging to OTHER sessions are untouched."""
    db = tmp_path / "memory.db"
    _seed_episode(db, session_id="A", input_text="a")
    _seed_episode(db, session_id="B", input_text="b")
    _seed_transcript_row(
        db, episode_id="ep-A-a", sequence=0,
        message_json='{"role":"user","content":"leak"}',
    )
    _seed_transcript_row(
        db, episode_id="ep-B-b", sequence=0,
        message_json='{"role":"user","content":"leak"}',
    )
    assert _scrub_prose_from_tool_transcripts(db, "A") == 1
    assert _transcript_count(db) == 1  # B's row survives


def test_scrub_swallows_db_errors(tmp_path: Path) -> None:
    """Missing db / missing table must not block /resume."""
    assert _scrub_prose_from_tool_transcripts(
        Path("/dev/null/cant-open"), "x"
    ) == 0
    # db exists but has no transcript table at all
    db = tmp_path / "memory.db"
    _seed_episode(db, session_id="S", input_text="x")
    assert _scrub_prose_from_tool_transcripts(db, "S") == 0
