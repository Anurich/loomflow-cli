"""Tests for file-touch history (loom_code.file_history).

Pure JSON-store logic — no agent, no embedder, fully offline + sync
(the module's functions are sync; only the repl wiring is async).
"""

from __future__ import annotations

import json
from pathlib import Path

from loom_code import file_history as fh


def _mk(tmp_path: Path) -> Path:
    (tmp_path / ".loom").mkdir()
    return tmp_path


def test_record_then_query(tmp_path: Path) -> None:
    root = _mk(tmp_path)
    fh.record_touches(root, ["src/a.py", "src/b.py"], outcome="unknown",
                      summary="add feature X")
    recs = fh.history_for(root, ["src/a.py", "src/missing.py"])
    assert len(recs) == 1  # missing.py has no history → omitted
    a = recs[0]
    assert a.path == "src/a.py"
    assert a.touch_count == 1
    assert a.last_outcome == "unknown"
    assert a.last_summary == "add feature X"


def test_touch_count_accumulates(tmp_path: Path) -> None:
    root = _mk(tmp_path)
    for _ in range(3):
        fh.record_touches(root, ["hot.py"], outcome="success")
    rec = fh.history_for(root, ["hot.py"])[0]
    assert rec.touch_count == 3
    assert rec.success_count == 3
    assert rec.fail_count == 0


def test_dedupes_within_a_call(tmp_path: Path) -> None:
    root = _mk(tmp_path)
    # Same file twice in one turn counts as one touch.
    fh.record_touches(root, ["x.py", "x.py"], outcome="success")
    rec = fh.history_for(root, ["x.py"])[0]
    assert rec.touch_count == 1


def test_update_last_outcome_revises_unknown(tmp_path: Path) -> None:
    root = _mk(tmp_path)
    # The real lifecycle: record unknown, then revise when judged.
    fh.record_touches(
        root, ["auth.py"], outcome="unknown", summary="fix login"
    )
    fh.update_last_outcome(root, ["auth.py"], "fail")
    rec = fh.history_for(root, ["auth.py"])[0]
    assert rec.last_outcome == "fail"
    assert rec.fail_count == 1
    assert rec.success_count == 0
    # touch_count must NOT have double-counted (record + revise = 1 touch)
    assert rec.touch_count == 1


def test_update_does_not_clobber_already_judged(tmp_path: Path) -> None:
    root = _mk(tmp_path)
    fh.record_touches(root, ["a.py"], outcome="success")  # already judged
    fh.update_last_outcome(root, ["a.py"], "fail")  # should be ignored
    rec = fh.history_for(root, ["a.py"])[0]
    assert rec.last_outcome == "success"
    assert rec.fail_count == 0


def test_corrupt_file_yields_empty_history(tmp_path: Path) -> None:
    root = _mk(tmp_path)
    (root / ".loom" / "file_history.json").write_text(
        "{not json", encoding="utf-8"
    )
    # Read degrades to empty; a subsequent write recovers cleanly.
    assert fh.history_for(root, ["x.py"]) == []
    fh.record_touches(root, ["x.py"], outcome="success")
    assert len(fh.history_for(root, ["x.py"])) == 1


def test_missing_loom_dir_is_created_on_write(tmp_path: Path) -> None:
    # No .loom dir yet — record_touches must create it (best-effort).
    fh.record_touches(tmp_path, ["x.py"], outcome="success")
    assert (tmp_path / ".loom" / "file_history.json").is_file()


def test_empty_paths_is_noop(tmp_path: Path) -> None:
    root = _mk(tmp_path)
    fh.record_touches(root, [], outcome="success")
    assert not (root / ".loom" / "file_history.json").exists()


def test_invalid_outcome_coerced_to_unknown(tmp_path: Path) -> None:
    root = _mk(tmp_path)
    fh.record_touches(root, ["x.py"], outcome="garbage")
    rec = fh.history_for(root, ["x.py"])[0]
    assert rec.last_outcome == "unknown"


def test_all_records_returns_every_tracked_file(tmp_path: Path) -> None:
    root = _mk(tmp_path)
    fh.record_touches(root, ["old.py"], outcome="success")
    fh.record_touches(root, ["new.py"], outcome="success")
    recs = fh.all_records(root)
    # Every tracked file is returned. (Ordering is by last_touched_at,
    # but the timestamp resolution is seconds — two touches in the same
    # second tie, so we don't assert a strict order here; the recency
    # sort matters across sessions/minutes, not within one test tick.)
    assert {r.path for r in recs} == {"old.py", "new.py"}


def test_candidate_paths_matches_full_path_in_prompt(
    tmp_path: Path,
) -> None:
    root = _mk(tmp_path)
    fh.record_touches(root, ["src/auth.py"], outcome="fail")
    cands = fh.candidate_paths_from_prompt(
        root, "please fix the bug in src/auth.py now"
    )
    assert cands == ["src/auth.py"]


def test_candidate_paths_matches_basename(tmp_path: Path) -> None:
    root = _mk(tmp_path)
    fh.record_touches(root, ["src/payments.py"], outcome="success")
    # User names the file without its dir.
    cands = fh.candidate_paths_from_prompt(
        root, "the payments.py logic needs work"
    )
    assert "src/payments.py" in cands


def test_candidate_paths_ignores_untracked(tmp_path: Path) -> None:
    root = _mk(tmp_path)
    fh.record_touches(root, ["a.py"], outcome="success")
    # Prompt mentions a file with NO history → no candidate.
    cands = fh.candidate_paths_from_prompt(root, "edit other_file.py")
    assert cands == []


def test_anticipation_block_warns_on_failure(tmp_path: Path) -> None:
    root = _mk(tmp_path)
    fh.record_touches(
        root, ["auth.py"], outcome="fail", summary="broke login"
    )
    recs = fh.history_for(root, ["auth.py"])
    block = fh.anticipation_block(recs)
    assert "auth.py" in block
    assert "BAD" in block
    assert "broke login" in block


def test_anticipation_block_silent_on_clean_file(tmp_path: Path) -> None:
    root = _mk(tmp_path)
    fh.record_touches(root, ["clean.py"], outcome="success")
    recs = fh.history_for(root, ["clean.py"])
    # A once-touched, successful file produces NO warning.
    assert fh.anticipation_block(recs) == ""


def test_anticipation_block_flags_churn_hotspot(tmp_path: Path) -> None:
    root = _mk(tmp_path)
    for _ in range(4):
        fh.record_touches(root, ["hot.py"], outcome="success")
    recs = fh.history_for(root, ["hot.py"])
    block = fh.anticipation_block(recs)
    assert "hot.py" in block
    assert "hotspot" in block


def test_persisted_json_is_valid_and_versioned(tmp_path: Path) -> None:
    root = _mk(tmp_path)
    fh.record_touches(root, ["x.py"], outcome="success")
    data = json.loads(
        (root / ".loom" / "file_history.json").read_text(encoding="utf-8")
    )
    assert data["version"] == 1
    assert "x.py" in data["files"]
    assert data["files"]["x.py"]["success_count"] == 1
