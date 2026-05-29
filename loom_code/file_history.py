"""File-touch history — what happened the last time we changed a file.

The data foundation for *anticipation* (the loom-code differentiator
over a stateless coder): before the agent touches ``foo.py`` again, it
can be reminded "last time you edited this, the change was marked bad"
or "you've edited this 6 times — it's a hotspot." Cursor's index knows
what the code IS; this knows what HAPPENED to it, across runs.

One JSON file per project — ``<root>/.loom/file_history.json`` — same
convention as the rest of loom-code's per-project state (notebook,
memory.db, code_index.db, last_session.txt). Schema:

    {
      "version": 1,
      "files": {
        "src/auth.py": {
          "touch_count": 6,
          "success_count": 4,
          "fail_count": 1,
          "last_touched_at": "2026-05-29T18:40:00Z",
          "last_outcome": "success" | "fail" | "unknown",
          "last_summary": "<one-line gist of the turn that touched it>"
        },
        ...
      }
    }

Outcome is the SAME signal the self-improvement loop uses — the
moved-on heuristic / ``/good`` / ``/bad`` (success: bool). We don't
yet parse test output for a finer signal; that's a future refinement
(the schema's ``last_outcome`` already has room for "fail" from a
detected test break). ``unknown`` is recorded when a turn touched
files but no success/failure was ever attributed (e.g. the user
quit mid-judgement).

Everything here is best-effort: a malformed file, a disk error, a
concurrent writer — none may ever break a turn. Reads return an empty
history; writes silently no-op. Anticipation degrading to silence is
correct; a crash is not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1
_HISTORY_FILENAME = "file_history.json"

# Cap the number of files we track — a runaway monorepo touch history
# would bloat the JSON + the recall prompt. We keep the most-recently-
# touched N; older untouched entries fall off. 500 covers any real
# working set without the file growing unbounded.
_MAX_FILES = 500


@dataclass(frozen=True)
class FileRecord:
    """One file's accumulated touch history."""

    path: str
    touch_count: int
    success_count: int
    fail_count: int
    last_touched_at: str
    last_outcome: str  # "success" | "fail" | "unknown"
    last_summary: str


def _history_path(root: Path | str) -> Path:
    return Path(root) / ".loom" / _HISTORY_FILENAME


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _load(root: Path | str) -> dict[str, Any]:
    """Load the raw history dict, or a fresh empty one. Never raises —
    a corrupt/absent file yields an empty history so the caller always
    gets a usable structure."""
    path = _history_path(root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": _SCHEMA_VERSION, "files": {}}
    if not isinstance(data, dict) or not isinstance(
        data.get("files"), dict
    ):
        return {"version": _SCHEMA_VERSION, "files": {}}
    return data


def _save(root: Path | str, data: dict[str, Any]) -> None:
    """Persist the history dict. Best-effort: a disk error silently
    no-ops (the history is a convenience, never load-bearing)."""
    path = _history_path(root)
    try:
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def _prune(files: dict[str, Any]) -> dict[str, Any]:
    """Keep the most-recently-touched ``_MAX_FILES`` entries so the
    store can't grow without bound on a huge repo."""
    if len(files) <= _MAX_FILES:
        return files
    ordered = sorted(
        files.items(),
        key=lambda kv: kv[1].get("last_touched_at", ""),
        reverse=True,
    )
    return dict(ordered[:_MAX_FILES])


def record_touches(
    root: Path | str,
    paths: list[str],
    *,
    outcome: str,
    summary: str = "",
) -> None:
    """Record that ``paths`` were edited this turn with ``outcome``
    (``"success"`` / ``"fail"`` / ``"unknown"``).

    Idempotent per call, additive across calls: each path's
    ``touch_count`` increments and the matching outcome counter bumps.
    ``summary`` is a one-line gist of the turn (the user's prompt,
    typically) so recall can say *why* the file was touched. Dedupes
    ``paths`` so one turn editing the same file twice counts once.

    Best-effort — never raises."""
    if not paths:
        return
    if outcome not in ("success", "fail", "unknown"):
        outcome = "unknown"
    summary = summary.replace("\n", " ").strip()[:200]
    now = _now_iso()

    data = _load(root)
    files: dict[str, Any] = data.get("files", {})
    for path in dict.fromkeys(paths):  # dedupe, preserve order
        rec = files.get(path) or {
            "touch_count": 0,
            "success_count": 0,
            "fail_count": 0,
            "last_touched_at": "",
            "last_outcome": "unknown",
            "last_summary": "",
        }
        rec["touch_count"] = int(rec.get("touch_count", 0)) + 1
        if outcome == "success":
            rec["success_count"] = int(rec.get("success_count", 0)) + 1
        elif outcome == "fail":
            rec["fail_count"] = int(rec.get("fail_count", 0)) + 1
        rec["last_touched_at"] = now
        rec["last_outcome"] = outcome
        if summary:
            rec["last_summary"] = summary
        files[path] = rec

    data["files"] = _prune(files)
    data["version"] = _SCHEMA_VERSION
    _save(root, data)


def update_last_outcome(
    root: Path | str, paths: list[str], outcome: str
) -> None:
    """Revise the outcome of the MOST RECENT touch of ``paths`` without
    incrementing the touch count.

    The flow that needs this: a turn records its touches as
    ``"unknown"`` immediately (so a crash before judgement still leaves
    a record), then the moved-on / ``/good`` / ``/bad`` signal arrives
    a turn later and revises it to success/fail. We move the count from
    unknown into the right bucket rather than double-counting the touch.

    Best-effort — never raises."""
    if not paths or outcome not in ("success", "fail"):
        return
    data = _load(root)
    files: dict[str, Any] = data.get("files", {})
    changed = False
    for path in dict.fromkeys(paths):
        rec = files.get(path)
        if rec is None:
            continue
        # Only revise if the last recorded outcome was unknown — don't
        # clobber an already-judged touch (idempotent re-attribution).
        if rec.get("last_outcome") != "unknown":
            continue
        if outcome == "success":
            rec["success_count"] = int(rec.get("success_count", 0)) + 1
        else:
            rec["fail_count"] = int(rec.get("fail_count", 0)) + 1
        rec["last_outcome"] = outcome
        files[path] = rec
        changed = True
    if changed:
        data["files"] = files
        _save(root, data)


def history_for(root: Path | str, paths: list[str]) -> list[FileRecord]:
    """Return the recorded history for each of ``paths`` that has any.

    Used by proactive recall (Phase 2b) to answer "what happened last
    time we touched the files this prompt is about." Order matches the
    input; paths with no history are omitted. Never raises."""
    data = _load(root)
    files: dict[str, Any] = data.get("files", {})
    out: list[FileRecord] = []
    for path in paths:
        rec = files.get(path)
        if rec is None:
            continue
        out.append(
            FileRecord(
                path=path,
                touch_count=int(rec.get("touch_count", 0)),
                success_count=int(rec.get("success_count", 0)),
                fail_count=int(rec.get("fail_count", 0)),
                last_touched_at=str(rec.get("last_touched_at", "")),
                last_outcome=str(rec.get("last_outcome", "unknown")),
                last_summary=str(rec.get("last_summary", "")),
            )
        )
    return out


def candidate_paths_from_prompt(
    root: Path | str, prompt: str
) -> list[str]:
    """Best-effort extract the tracked file paths a prompt is ABOUT.

    Matches any tracked path that appears verbatim in the prompt (the
    common case: "fix the bug in src/auth.py"), plus a basename match
    ("the auth.py change broke") so a user who names the file without
    its dir still gets the warning. Only returns paths that actually
    have history — this is the candidate set for proactive recall.
    Cheap string containment; no embedding. Never raises."""
    if not prompt:
        return []
    data = _load(root)
    tracked = list(data.get("files", {}).keys())
    if not tracked:
        return []
    low = prompt.lower()
    hits: list[str] = []
    for path in tracked:
        p_low = path.lower()
        base = path.rsplit("/", 1)[-1].lower()
        # Full path mentioned, or the basename as a whole word-ish
        # token (guard against "a.py" matching inside "data.py" by
        # requiring the basename to be reasonably specific).
        if p_low in low or (len(base) >= 5 and base in low):
            hits.append(path)
    return hits


def anticipation_block(records: list[FileRecord]) -> str:
    """Render file-touch records into a proactive-recall working block.

    Only surfaces records WORTH warning about — a prior failure, or a
    churn hotspot (touched many times). A clean, once-touched file
    produces nothing (silence is correct; noise trains the user to
    ignore the section). Returns "" when nothing's notable."""
    notable: list[FileRecord] = []
    for r in records:
        if r.last_outcome == "fail" or r.fail_count > 0:
            notable.append(r)
        elif r.touch_count >= 4:  # churn hotspot
            notable.append(r)
    if not notable:
        return ""
    lines = ["# What happened last time"]
    lines.append(
        "You're about to work on files with relevant history. "
        "Heed it before repeating a past mistake."
    )
    for r in notable:
        if r.last_outcome == "fail" or r.fail_count > 0:
            why = f" ({r.last_summary})" if r.last_summary else ""
            note = (
                f"- `{r.path}` — last change was marked BAD{why}. "
                f"Touched {r.touch_count}×, {r.fail_count} failed. "
                "Be extra careful + verify."
            )
        else:
            note = (
                f"- `{r.path}` — churn hotspot ({r.touch_count} edits). "
                "Fragile / frequently-revised; tread carefully."
            )
        lines.append(note)
    return "\n".join(lines)


def all_records(root: Path | str) -> list[FileRecord]:
    """Every tracked file, most-recently-touched first. For a future
    'hotspots' view / the desktop anticipation surface (Phase 2c)."""
    data = _load(root)
    files: dict[str, Any] = data.get("files", {})
    recs = [
        FileRecord(
            path=p,
            touch_count=int(r.get("touch_count", 0)),
            success_count=int(r.get("success_count", 0)),
            fail_count=int(r.get("fail_count", 0)),
            last_touched_at=str(r.get("last_touched_at", "")),
            last_outcome=str(r.get("last_outcome", "unknown")),
            last_summary=str(r.get("last_summary", "")),
        )
        for p, r in files.items()
    ]
    recs.sort(key=lambda r: r.last_touched_at, reverse=True)
    return recs
