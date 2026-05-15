"""Map test files to the symbols they exercise.

For each non-test public symbol, scan every test file for exact-
match references to its bare name (``Agent`` not ``loomflow.Agent``).
Bare-name match works because the test will have either imported
the symbol (so it appears as a bare identifier in the body) or
called it via an attribute chain (where the bare name still
appears).

False positives are tolerated — a test that mentions ``Agent`` in
a comment but doesn't really exercise it is a minor inaccuracy
the agent can verify in seconds. False negatives (missing edges)
are also tolerated; the agent can still grep manually.

Cost: O(test_file_bytes × n_symbols) in the worst case. We bound
by building a single regex with alternation over all symbol names
and running it once per test file — linear in the test corpus.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from ._files import DiscoveredFile


def build_test_map(
    *,
    files: list[DiscoveredFile],
    symbol_names: Iterable[str],
) -> dict[str, list[str]]:
    """Return ``{symbol_name: ["test_file:line", ...]}``.

    ``symbol_names`` is the bare-name set (NOT qualified). When two
    symbols share a name (``foo`` in two modules), both attribution
    to the same test is the honest answer — we can't disambiguate
    without execution.
    """
    names = sorted({n for n in symbol_names if _is_match_candidate(n)})
    if not names:
        return {}
    # One regex per scan — Python's re engine is fast and alternation
    # over a few hundred names is fine. Word-boundary anchored so
    # ``Agent`` doesn't match ``AgentManager``.
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in names) + r")\b"
    )

    out: dict[str, list[str]] = {n: [] for n in names}
    for f in files:
        if not f.is_test or f.lang != "python":
            continue
        try:
            text = f.abs_path.read_text(encoding="utf-8")
        except OSError:
            continue
        # Track first-line hit per name in this file — we want a
        # citation, not every occurrence.
        seen_in_file: dict[str, int] = {}
        for i, line in enumerate(text.splitlines(), start=1):
            for match in pattern.finditer(line):
                name = match.group(1)
                if name not in seen_in_file:
                    seen_in_file[name] = i
        for name, line_no in seen_in_file.items():
            out[name].append(f"{f.rel_path}:{line_no}")
    return out


def _is_match_candidate(name: str) -> bool:
    """Skip names too generic / too short to bare-name-match without
    a flood of false positives.

    Heuristic: identifier must be at least 4 characters, and must
    not be a Python keyword or builtin. We don't have the full
    builtin list inline — the length filter alone catches the worst
    offenders (``run``, ``main``, ``new``, ``foo``)."""
    if len(name) < 4:
        return False
    if name in _IGNORED_NAMES:
        return False
    return True


# Common method/function names that are present in basically every
# codebase. Bare-match for these would be noise. Filter ruthlessly
# — false-negative is "agent can still grep" while false-positive
# is "agent reads N false-positive tests".
_IGNORED_NAMES: frozenset[str] = frozenset(
    {
        "main",
        "run",
        "test",
        "tests",
        "setup",
        "teardown",
        "Setup",
        "Teardown",
        "name",
        "data",
        "value",
        "item",
        "items",
        "load",
        "save",
    }
)
