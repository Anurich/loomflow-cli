"""File discovery + per-file metadata (hash, lang, git heat).

The structural extractor calls :func:`discover_files` once at the
start of indexing; everything downstream uses the returned
:class:`DiscoveredFile` list as the canonical "what files exist".

Discovery strategy:

* **In a git repo** — use ``git ls-files --cached --others
  --exclude-standard``. This respects ``.gitignore`` for free, which
  is the *only* reliable way to skip a project's actual ignore set
  (venvs, build outputs, generated code). The alternative — re-
  implementing gitignore semantics — is a tar pit.
* **No git** — walk the tree, skip a hard-coded set of well-known
  noise directories (``.venv``, ``node_modules``, ``__pycache__``,
  etc.). Less accurate but covers the "loose folder" case.

Git heat (commits touching a file in the last 90 days) comes from
``git log --since=90.days --name-only`` parsed once at discovery
time. If the repo is huge, this is the most expensive step in
discovery — still O(seconds). Cached in the returned dataclass so
no other module needs to re-run git.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

# Hard-coded noise directories for the non-git walker. Add only
# things that are universally noise — when in doubt, leave it in,
# the user is in a git repo 99% of the time and these matter only
# for the fallback path.
_NOISE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "dist",
        "build",
        ".eggs",
        ".loom",  # our own output dir — never re-index ourselves
        ".idea",
        ".vscode",
    }
)

# File extensions we recognize. Anything else is skipped — the index
# is for code understanding, not asset cataloguing. Markdown is kept
# because docs often capture architecture decisions that supplement
# the LLM-generated narrative.
_LANG_BY_EXT: dict[
    str, Literal["python", "markdown", "toml", "yaml", "json"]
] = {
    ".py": "python",
    ".pyi": "python",
    ".md": "markdown",
    ".markdown": "markdown",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
}


@dataclass(frozen=True)
class DiscoveredFile:
    """One file the extractor will inspect, with everything it needs
    to know up front. ``rel_path`` is repo-relative POSIX.

    Hash is computed lazily on first read — the dataclass stores
    the absolute path; callers compute + cache via :func:`hash_file`.
    """

    rel_path: str
    abs_path: Path
    lang: Literal["python", "markdown", "toml", "yaml", "json", "other"]
    size_bytes: int
    lines: int
    sha256: str
    mtime: datetime
    git_changes_90d: int | None
    is_test: bool


def is_git_repo(root: Path) -> bool:
    """True when ``root`` (or any ancestor up to the filesystem
    boundary) contains a ``.git`` directory. Using ``git rev-parse``
    rather than just checking ``.git/`` makes us correct on
    submodules + worktrees, where ``.git`` is a file."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except FileNotFoundError:
        # No git binary installed — definitely no git repo by our
        # operational definition (we'd need git to enumerate it).
        return False


def discover_files(root: Path) -> list[DiscoveredFile]:
    """Enumerate every indexable file under ``root``.

    Order is deterministic (sorted by ``rel_path``) so the resulting
    :class:`schema.LoomIndex` is byte-stable across runs that see
    the same tree — important for diff-aware refresh.

    Returns an empty list when ``root`` doesn't exist; raises only on
    permission errors. A non-readable repo is a real problem worth
    surfacing.
    """
    if not root.exists():
        return []

    rel_paths = _list_paths(root)
    git_heat = _git_heat(root) if is_git_repo(root) else {}

    out: list[DiscoveredFile] = []
    for rel in sorted(rel_paths):
        abs_path = root / rel
        if not abs_path.is_file():
            continue
        try:
            data = abs_path.read_bytes()
        except OSError:
            continue
        ext = abs_path.suffix.lower()
        lang = _LANG_BY_EXT.get(ext, "other")
        if lang == "other":
            # We keep ``other`` files OUT of the index for now — the
            # annotator can't do anything useful with binary blobs
            # and including them just bloats files[]. Future: re-
            # enable for shell / docker / etc. with a language
            # filter in extractor.
            continue
        sha = hashlib.sha256(data).hexdigest()
        text = data.decode("utf-8", errors="replace")
        size = len(data)
        n_lines = text.count("\n") + (
            1 if text and not text.endswith("\n") else 0
        )
        mtime = datetime.fromtimestamp(abs_path.stat().st_mtime).astimezone()
        out.append(
            DiscoveredFile(
                rel_path=rel,
                abs_path=abs_path,
                lang=lang,
                size_bytes=size,
                lines=n_lines,
                sha256=sha,
                mtime=mtime,
                git_changes_90d=git_heat.get(rel),
                is_test=_is_test_path(rel),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _list_paths(root: Path) -> list[str]:
    """Enumerate POSIX-relative paths under ``root``.

    Routes through ``git ls-files`` when applicable (free .gitignore
    handling), else walks + filters noise dirs.
    """
    if is_git_repo(root):
        return _git_list(root)
    return _walk_list(root)


def _git_list(root: Path) -> list[str]:
    """``git ls-files --cached --others --exclude-standard`` —
    tracked + untracked but not ignored. Skips submodule contents
    (recurse=False by default) which is what we want; submodule
    code belongs to a different repo's index."""
    proc = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        # Fall back to walking — better partial coverage than zero.
        return _walk_list(root)
    return [
        line.strip()
        for line in proc.stdout.splitlines()
        if line.strip()
    ]


def _walk_list(root: Path) -> list[str]:
    """Manual walk: skip directories in :data:`_NOISE_DIRS`. POSIX
    paths only — Windows users get the same shape via PurePosixPath
    conversion in the caller (loom-code runs on macOS / Linux today
    but the contract should not be tripped by OS quirks)."""
    out: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        # Skip if any part is a noise dir.
        if any(part in _NOISE_DIRS for part in path.relative_to(root).parts):
            continue
        out.append(path.relative_to(root).as_posix())
    return out


def _git_heat(root: Path) -> dict[str, int]:
    """Return ``{rel_path: n_commits_in_last_90d}``.

    Uses ``git log --since=90.days --name-only --pretty=`` — outputs
    one path per line per commit, with blank lines between commits.
    Counting occurrences gives us the heat score directly.

    Returns ``{}`` on any subprocess error — heat is a hint, not a
    correctness guarantee."""
    try:
        proc = subprocess.run(
            [
                "git",
                "log",
                "--since=90.days",
                "--name-only",
                "--pretty=format:",
                "--no-merges",
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}
    if proc.returncode != 0:
        return {}
    counts: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        counts[line] = counts.get(line, 0) + 1
    return counts


def _is_test_path(rel: str) -> bool:
    """Best-effort detection: anything under ``tests/`` or named
    ``test_*.py`` / ``*_test.py``. Same heuristic pytest uses, which
    matches the vast majority of Python projects."""
    parts = rel.split("/")
    if "tests" in parts or "test" in parts:
        return True
    name = parts[-1]
    return name.startswith("test_") or name.endswith("_test.py")
