"""Enhanced ``grep`` tool for loom-code agents.

Same role as ``loomflow.tools.grep_tool`` — find regex matches under a
working directory — but with structured output the agent can actually
USE without N follow-up reads:

* **Surrounding context** — ±N lines around each match so the agent
  sees the match in context (loomflow's default is just the matching
  line, which forces a separate ``read`` of every interesting hit).
* **Grouped by file** — all matches for one file in a single block
  with the path as a header. Easier to scan than 50 ``path:line:``
  prefixes.
* **Test-file collapsing** — hits in ``tests/`` / ``test_*.py`` /
  ``*_test.py`` collapse into a one-line "+N matches in test files"
  summary by default. Keeps prod code in focus; agent opts in to
  test matches with ``include_tests=True``.
* **Optional language filter** — ``type=("py", "ts")`` restricts to
  those extensions.

Default behaviour is the enhanced form so the agent's default grep is
the good one. Pass ``raw=True`` for the old flat-line format if a
tight one-line-per-match shape is needed.

Why this lives in loom-code (not loomflow): loom-code is opinionated
about the SHAPE of grep output for coding-agent UX. The framework's
``grep_tool`` is a sensible generic; this wrapper adds the loom-code
ergonomics without forking the framework.
"""

from __future__ import annotations

import re
from pathlib import Path

from loomflow import tool
from loomflow.tools.registry import Tool

# Directories we never search — matches loomflow's grep_tool noise
# list so behaviour around big virtualenvs / build outputs is the
# same as the framework default. Keeps walk time bounded on real
# projects with .venv / node_modules / etc.
_NOISE_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".env", ".tox", "dist", "build", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", "graphify-out", ".loom",
})

# Heuristics for "is this a test file?". Anything under a ``tests``
# folder, or named ``test_*.py`` / ``*_test.py`` / ``*.test.*``.
_TEST_DIR_NAMES = frozenset({"tests", "test", "__tests__"})


def _is_test_path(rel: Path) -> bool:
    """True if the relative path is a test file by directory or
    filename convention."""
    parts = set(rel.parts)
    if parts & _TEST_DIR_NAMES:
        return True
    name = rel.name.lower()
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    if ".test." in name or ".spec." in name:
        return True
    return False


def _walk_files(
    root: Path,
    glob: str,
    type_filter: tuple[str, ...] | None,
) -> list[Path]:
    """Walk ``root`` honouring ``_NOISE_DIRS`` + the user's glob +
    optional extension allowlist. Returns absolute paths."""
    out: list[Path] = []
    for path in root.rglob(glob):
        if not path.is_file():
            continue
        if any(p in _NOISE_DIRS for p in path.parts):
            continue
        if type_filter is not None:
            suffix = path.suffix.lstrip(".").lower()
            if suffix not in type_filter:
                continue
        out.append(path)
    return out


def _render_grouped(
    matches_by_file: dict[Path, list[tuple[int, str]]],
    file_lines: dict[Path, list[str]],
    *,
    context: int,
    root: Path,
) -> str:
    """Render the structured per-file output with context lines."""
    if not matches_by_file:
        return "no matches"
    sections: list[str] = []
    for path in sorted(matches_by_file):
        hits = matches_by_file[path]
        lines = file_lines[path]
        rel = path.relative_to(root)
        sections.append(
            f"─ {rel} ({len(hits)} match{'es' if len(hits) != 1 else ''}) "
            + "─" * max(0, 40 - len(str(rel)))
        )
        # For each hit, show context lines. If multiple hits are
        # close together their context windows merge naturally —
        # we DON'T deduplicate here because that'd hide hit
        # boundaries; instead we render each hit's window. Agent
        # gets a slight redundancy in exchange for clearer per-
        # hit framing.
        for lineno, _ in hits:
            start = max(0, lineno - 1 - context)
            end = min(len(lines), lineno + context)
            for i in range(start, end):
                marker = "▸ " if i + 1 == lineno else "  "
                sections.append(
                    f"  {marker}{i + 1:4d} │ {lines[i].rstrip()}"
                )
            sections.append("")  # blank line between hit windows
        # Remove trailing blank for tidiness.
        if sections and sections[-1] == "":
            sections.pop()
    return "\n".join(sections)


def enhanced_grep_tool(
    workdir: Path | str,
    *,
    max_files_with_matches: int = 30,
    max_matches_per_file: int = 10,
    default_context: int = 2,
) -> Tool:
    """Build the loom-code grep tool. Sees the model with:

        grep(pattern, path=".", glob="*",
             ignore_case=False, context=2,
             include_tests=False, raw=False,
             type="")

    ``pattern`` is a Python regex. ``path`` is relative to the agent's
    workdir. ``context`` is ±N lines around each match (default 2).
    ``include_tests=True`` un-collapses test-file hits. ``raw=True``
    drops the grouped/contextual rendering and falls back to flat
    ``path:lineno: line`` lines (loomflow's classic shape) for
    consumers that want one-line-per-match. ``type`` is a
    comma-separated extension filter (e.g. ``"py,ts"``) — empty
    means no filter.
    """
    root = Path(workdir).resolve()

    async def grep(
        pattern: str,
        path: str = ".",
        glob: str = "*",
        ignore_case: bool = False,
        context: int = default_context,
        include_tests: bool = False,
        raw: bool = False,
        type: str = "",  # noqa: A002 — model-facing arg name; matches CLI ergonomics
    ) -> str:
        """Find regex matches under ``path`` and return grouped,
        context-rich results. See module docstring for the full
        contract."""
        # Resolve and bounds-check ``path``. Must stay under root.
        target = (root / path).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return f"grep: refusing to search outside workdir: {target}"
        if not target.exists():
            return f"grep: path not found: {path}"

        # Pre-compile the pattern; surface regex errors to the agent
        # so it can fix the call instead of getting empty output.
        flags = re.IGNORECASE if ignore_case else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return f"grep: invalid regex {pattern!r}: {exc}"

        # Normalise the type filter.
        type_filter: tuple[str, ...] | None = None
        if type:
            type_filter = tuple(
                t.strip().lower() for t in type.split(",") if t.strip()
            )

        # Walk + collect matches per file.
        files = _walk_files(target, glob, type_filter)
        matches_by_file: dict[Path, list[tuple[int, str]]] = {}
        file_lines_cache: dict[Path, list[str]] = {}
        test_file_count = 0
        test_match_count = 0
        for fpath in files:
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = text.splitlines()
            hits: list[tuple[int, str]] = []
            for i, line in enumerate(lines, start=1):
                if regex.search(line):
                    hits.append((i, line))
                    if len(hits) >= max_matches_per_file:
                        break
            if not hits:
                continue
            rel = fpath.relative_to(root)
            if (not include_tests) and _is_test_path(rel):
                test_file_count += 1
                test_match_count += len(hits)
                continue
            matches_by_file[fpath] = hits
            file_lines_cache[fpath] = lines
            if len(matches_by_file) >= max_files_with_matches:
                break

        if raw:
            # Old flat shape — loomflow's classic output. Kept as
            # an escape hatch for one-line-per-match consumers.
            out: list[str] = []
            for fpath in sorted(matches_by_file):
                rel = fpath.relative_to(root)
                for lineno, line in matches_by_file[fpath]:
                    out.append(f"{rel}:{lineno}: {line}")
            if test_match_count and not include_tests:
                out.append(
                    f"... +{test_match_count} match(es) in "
                    f"{test_file_count} test file(s) "
                    "(pass include_tests=True to show)"
                )
            return "\n".join(out) if out else "no matches"

        # Default rendering: grouped + with context.
        body = _render_grouped(
            matches_by_file,
            file_lines_cache,
            context=context,
            root=root,
        )
        if test_match_count and not include_tests:
            body += (
                f"\n\n+ {test_match_count} match(es) in "
                f"{test_file_count} test file(s) — "
                "pass include_tests=True to show"
            )
        return body

    # Use the @tool decorator pattern by promoting the closure
    # into a Tool with a manually-built schema. We can't use the
    # bare @tool decorator on a nested function because the
    # decorator-derived description would lose the loom-code-
    # specific guidance the agent needs to pick the right args.
    return tool(
        name="grep",
        description=(
            "Search file contents for a regex. Returns grouped "
            "results: one block per file, with ±2 lines of "
            "context around each hit. Test-file matches are "
            "collapsed by default — pass include_tests=True to "
            "show. Args: pattern (regex), path='.', glob='*', "
            "ignore_case=False, context=2 (±N lines), "
            "include_tests=False, raw=False (raw=True for "
            "flat one-line-per-match output), type='' "
            "(comma-separated extensions e.g. 'py,ts')."
        ),
    )(grep)
