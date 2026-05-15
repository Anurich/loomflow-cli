"""Read/write ``.loom/index.json`` and ``LOOM.md``.

All loominit modules go through this layer rather than touching
files directly so atomicity, schema-version checking, and the
``.loom/`` directory layout stay in one place.

Files we own::

    <repo_root>/.loom/index.json     — machine-readable, this module
    <repo_root>/.loom/index.json.tmp — atomic-write staging
    <repo_root>/LOOM.md              — human-readable, annotator's output

``LOOM.md`` lives at the repo root (not under ``.loom/``) on purpose:
it's meant to be in git, visible in file trees, and editable by
the user. ``.loom/`` is the machine state — gitignored by default.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .schema import SCHEMA_VERSION, LoomIndex

# Names used everywhere. Constants so a future move
# (e.g., ``.loom/`` → ``.loom-code/``) is a one-line change.
LOOM_DIR_NAME = ".loom"
INDEX_FILE_NAME = "index.json"
MARKDOWN_FILE_NAME = "LOOM.md"


def loom_dir(repo_root: Path) -> Path:
    """Return ``<repo_root>/.loom/`` — creating it if missing."""
    d = repo_root / LOOM_DIR_NAME
    d.mkdir(exist_ok=True)
    return d


def index_path(repo_root: Path) -> Path:
    """Absolute path of ``.loom/index.json``."""
    return loom_dir(repo_root) / INDEX_FILE_NAME


def markdown_path(repo_root: Path) -> Path:
    """Absolute path of ``LOOM.md`` (at repo root, NOT under ``.loom``).
    Kept in the repo root so it's discoverable + git-trackable."""
    return repo_root / MARKDOWN_FILE_NAME


class IndexVersionMismatch(RuntimeError):
    """Raised when an existing ``index.json`` was written by an
    incompatible schema version. The REPL catches this and nudges
    the user to ``/loominit rebuild``."""


def load_index(repo_root: Path) -> LoomIndex | None:
    """Read ``.loom/index.json`` and return the parsed
    :class:`LoomIndex`, or ``None`` if the file does not exist.

    Raises :class:`IndexVersionMismatch` if a file is present but
    its ``version`` differs from :data:`schema.SCHEMA_VERSION`. We
    do not auto-migrate — the annotation is built on the current
    schema's semantics, so a rebuild is the honest fix."""
    p = index_path(repo_root)
    if not p.exists():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    found_version = raw.get("version", 0)
    if found_version != SCHEMA_VERSION:
        raise IndexVersionMismatch(
            f"index.json at {p} is schema v{found_version}; "
            f"this loom-code is v{SCHEMA_VERSION}. "
            "Run /loominit rebuild to regenerate."
        )
    return LoomIndex.model_validate(raw)


def save_index(repo_root: Path, index: LoomIndex) -> None:
    """Write ``.loom/index.json`` atomically.

    Atomic = temp file in the same directory, fsync, rename. A
    crash mid-write leaves the previous index intact — important
    because losing the index forces a full rebuild (expensive)."""
    p = index_path(repo_root)
    # NamedTemporaryFile in the same dir so the rename is atomic
    # (cross-filesystem rename would fall back to copy+unlink, which
    # is not atomic).
    payload = index.model_dump(mode="json")
    text = json.dumps(payload, indent=2, sort_keys=False)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".index-", suffix=".json.tmp", dir=str(p.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, p)
    except Exception:
        # Best-effort cleanup; if the temp file was already moved
        # the unlink will harmlessly fail.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_markdown(repo_root: Path) -> str | None:
    """Return the current ``LOOM.md`` body or ``None`` if absent.

    Returning None (rather than empty string) lets callers
    distinguish "never run /loominit" from "/loominit ran but
    produced nothing"."""
    p = markdown_path(repo_root)
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


def write_markdown(repo_root: Path, body: str) -> None:
    """Write ``LOOM.md`` atomically. Same rationale as
    :func:`save_index` — losing the file mid-write would be a
    big regression for the user."""
    p = markdown_path(repo_root)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".LOOM-", suffix=".md.tmp", dir=str(p.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, p)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
