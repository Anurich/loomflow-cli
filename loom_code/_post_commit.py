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
    """Re-run the graphify extract → build → cluster → persist
    pipeline in-process via the shared ``graphify_build_impl``
    helper that the ``@tool`` wrapper + ``/loominit`` already use.

    Single source of truth means three things stay in sync: the
    submodule-import shim that dodges graphify's ``__getattr__``
    namespace shadowing, the git-ls-files fast path that skips
    walking ``.venv`` / ``node_modules``, and the exact tree-sitter
    / Leiden / JSON pipeline. Bypassing it here is what caused this
    function to silently fail on every commit before: it called
    ``graphify.extract(files)`` (a submodule, not a function),
    passed ``[extraction]`` (build_from_json wants a dict), and
    dropped the ``communities`` arg to ``to_json``.

    Capped via subprocess from the shell hook (5 min, see the hook
    wrapper) so a hung extraction can't block git indefinitely."""
    # graphify_dir kept in the signature for the caller's existing
    # path math; the impl writes to the same `.loom/graphify/graph.json`
    # via its own ``_graph_path`` helper, so we don't need to use it.
    _ = graphify_dir
    import anyio

    from .skills.graphify.tools import graphify_build_impl

    anyio.run(graphify_build_impl, project_root)


if __name__ == "__main__":
    sys.exit(main())
