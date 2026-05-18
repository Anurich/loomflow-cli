"""Post-commit hook runner — debounced indexer refresh.

Invoked by ``.git/hooks/post-commit`` (installed by
``loom_code.git_hook.install``). Counts commits since the last
refresh per indexer; when the threshold is hit (5 by default),
runs the indexer's incremental update.

Designed to FAIL SILENT. A git hook crashing has the same UX
cost as a broken commit — we'd rather skip a refresh than make
``git commit`` look broken. All exception handling is broad and
mute; the worst case is "graph stayed stale one more commit."

Why debounce: graphify rebuilds + loominit structural rebuilds
are fast (5-15s on typical projects) but not free. Running them
on every single commit during a heavy dev session (10+ commits/
hour) is wasteful. Every-5-commits keeps the indexes "close
enough" without burning cycles.

Invoked as::

    python -m loom_code._post_commit <project_root>

Backgrounded by the shell hook so it doesn't delay the commit.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

# Refresh threshold — commits since last refresh before triggering
# the indexer's incremental rebuild. Empirically tuned: every
# commit is wasteful, every 20 is too stale to be useful, 5 sits
# in the goldilocks zone for typical dev pace.
_THRESHOLD = 5


def main() -> int:
    if len(sys.argv) < 2:
        return 0
    project_root = Path(sys.argv[1])
    loom_dir = project_root / ".loom"
    if not loom_dir.is_dir():
        return 0

    # Graphify: incremental rebuild via the package's own
    # ``--update`` flag (re-extracts only changed files, merges
    # into the existing graph). Only runs if graphify has been
    # set up at least once for this project.
    graphify_dir = loom_dir / "graphify"
    if (graphify_dir / "graph.json").is_file():
        _maybe_refresh(
            counter_file=graphify_dir / "_commits_since_refresh.txt",
            refresh_fn=lambda: _refresh_graphify(project_root, graphify_dir),
        )

    # Loominit: structural rebuild only (file walk + AST + graph
    # scoring). NO LLM annotation — that's `/loominit refresh`,
    # an explicit user-driven action. Keeps the post-commit hook
    # zero-cost and zero-credentials.
    if (loom_dir / "index.json").is_file():
        _maybe_refresh(
            counter_file=loom_dir / "_loominit_commits_since_refresh.txt",
            refresh_fn=lambda: _refresh_loominit(project_root),
        )

    return 0


def _maybe_refresh(
    *, counter_file: Path, refresh_fn: Callable[[], None]
) -> None:
    """Increment the counter; if it crosses the threshold, run
    the refresh and reset. Errors are swallowed — better to skip
    a refresh than break the commit."""
    try:
        count = (
            int(counter_file.read_text())
            if counter_file.is_file()
            else 0
        )
    except (ValueError, OSError):
        count = 0
    count += 1
    if count >= _THRESHOLD:
        try:
            refresh_fn()
            counter_file.write_text("0")
        except Exception:  # noqa: BLE001 — never break a commit
            # Leave counter at threshold; next commit will retry.
            pass
    else:
        try:
            counter_file.write_text(str(count))
        except OSError:
            pass


def _refresh_graphify(project_root: Path, graphify_dir: Path) -> None:
    """Run graphify in incremental-update mode. ``--update`` is
    much cheaper than a full rebuild — it only re-extracts files
    whose hash changed since the last build. Capped at 5 min to
    avoid runaway on giant repos."""
    subprocess.run(
        [
            "graphify",
            str(project_root),
            "--update",
            "--out",
            str(graphify_dir),
        ],
        check=False,
        capture_output=True,
        timeout=300,
    )


def _refresh_loominit(project_root: Path) -> None:
    """Re-run the structural pass and overwrite ``.loom/index.json``.
    Pure Python (no LLM, no provider key). LOOM.md stays as-is —
    its inline ``(stale: path:line)`` markers automatically appear
    where the new index disagrees with old annotated claims."""
    from .loominit.extractor import build_index
    from .loominit.persistence import save_index
    index = build_index(project_root)
    save_index(project_root, index)


if __name__ == "__main__":
    sys.exit(main())
