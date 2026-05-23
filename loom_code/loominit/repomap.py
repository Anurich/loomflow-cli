"""Deterministic, LLM-free repo map — an Aider-style ranked symbol
overview.

Why this exists: the older LOOM.md path injected BM25-ranked *LLM
narrative* sections, which (a) drifted from the code the moment the
agent edited it and (b) was a lossy paraphrase, not the code. This
renderer instead consumes the **structural** index (``build_index`` —
AST walk, no model calls) and emits the most structurally-important
symbols — real signatures + locations — within a token budget. It is:

* **deterministic** — same index in, same map out (stable tiebreaks);
* **fresh-by-construction** — re-running ``build_index`` reflects the
  current tree, with zero LLM cost (unlike re-annotation);
* **cache-friendly** — the map is a stable global overview, so it
  doesn't churn the system prompt per turn the way BM25 retrieval did.

Ranking heuristic (Aider's core insight: "a symbol referenced by many
others is more valuable context than a private helper called once"):
``n_callers`` (how widely the symbol's file is imported — already
computed by the extractor) plus bonuses for entry points, public
surface, and classes (architecture lands first).
"""

from __future__ import annotations

from pathlib import Path

from .schema import LoomIndex, SymbolEntry

# Rough chars→tokens ratio for budgeting (English/code ≈ 4 chars/token).
_CHARS_PER_TOKEN = 4


_DEF_PREFIXES = ("class ", "def ", "async def ")


def _score(sym: SymbolEntry, entry_point_paths: set[str]) -> float:
    """Structural importance of a symbol. Higher = surfaced sooner."""
    score = float(sym.n_callers) * 2.0  # import popularity (the core signal)
    if not sym.name.startswith("_"):
        score += 1.0  # public surface beats private helpers
    if sym.signature.startswith("class "):
        score += 2.0  # classes carry architecture
    elif not sym.signature.startswith(_DEF_PREFIXES):
        score -= 3.0  # bare module constants are noise in an overview
    if sym.path in entry_point_paths:
        score += 6.0  # CLI / route / main entry points orient fast
    return score


def build_repo_map(
    index: LoomIndex,
    *,
    max_tokens: int = 1500,
    max_per_file: int = 8,
) -> str:
    """Render the top symbols by structural importance, grouped by
    file, within ``max_tokens`` (best-effort char estimate). Files are
    ordered by their most-important symbol; within a file, symbols are
    ordered by score then line. Deterministic."""
    budget = max_tokens * _CHARS_PER_TOKEN
    ep_paths = {ep.path for ep in index.entry_points if ep.path}

    by_file: dict[str, list[SymbolEntry]] = {}
    for sym in index.symbols:
        by_file.setdefault(sym.path, []).append(sym)
    for syms in by_file.values():
        syms.sort(key=lambda s: (-_score(s, ep_paths), s.line))

    def file_rank(path: str) -> float:
        return max((_score(s, ep_paths) for s in by_file[path]), default=0.0)

    file_order = sorted(by_file, key=lambda p: (-file_rank(p), p))

    head = "# Repo map — top symbols by structural importance\n"
    out: list[str] = [head]
    used = len(head)
    for path in file_order:
        header = f"\n## {path}\n"
        if used + len(header) > budget:
            break
        block: list[str] = [header]
        blen = len(header)
        for sym in by_file[path][:max_per_file]:
            doc = (
                f"  — {sym.docstring_first_line}"
                if sym.docstring_first_line
                else ""
            )
            entry = f"- `{sym.signature}` ({sym.path}:{sym.line}){doc}\n"
            if used + blen + len(entry) > budget:
                break
            block.append(entry)
            blen += len(entry)
        # Only emit the file header if at least one symbol fit under it.
        if blen > len(header):
            out.extend(block)
            used += blen
    return "".join(out)


def repo_map_for_root(
    root: Path | str, *, max_tokens: int = 1500
) -> str | None:
    """Convenience: build the structural index for ``root`` (no LLM)
    and render the repo map. Returns ``None`` when the repo has no
    indexable symbols. Caller decides caching/freshness policy."""
    from .extractor import build_index

    index = build_index(Path(root))
    if not index.symbols:
        return None
    return build_repo_map(index, max_tokens=max_tokens)


# Dirs the freshness signature ignores — build_index skips them too, so
# walking them would only add cost + false cache misses.
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".loom",
        ".loom-code",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".tox",
    }
)

# Cache: root -> (signature, rendered map). The signature is cheap to
# recompute; the expensive AST walk only re-runs when it changes.
_REPO_MAP_CACHE: dict[str, tuple[tuple[float, int], str | None]] = {}


def _tree_signature(root: Path) -> tuple[float, int]:
    """A cheap freshness key: (newest .py mtime, file count). Either
    moving means the tree changed → rebuild. Count catches add/delete
    that don't bump the newest mtime."""
    newest = 0.0
    count = 0
    for p in root.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        try:
            newest = max(newest, p.stat().st_mtime)
            count += 1
        except OSError:
            continue
    return (newest, count)


def _repo_map_cached(
    root: Path | str, max_tokens: int
) -> tuple[str | None, bool]:
    """Core cached build. Returns ``(map, rebuilt)`` — ``rebuilt`` is
    True when the source tree changed and we re-walked, False on a
    cache hit. Freshness-keyed (newest .py mtime + file count), so it
    only re-parses when something actually moved. Mirrors the map to
    ``<root>/.loom/repomap.md`` on every rebuild for inspection."""
    root_p = Path(root).resolve()
    key = str(root_p)
    sig = _tree_signature(root_p)
    cached = _REPO_MAP_CACHE.get(key)
    if cached is not None and cached[0] == sig:
        return cached[1], False
    rendered = repo_map_for_root(root_p, max_tokens=max_tokens)
    _REPO_MAP_CACHE[key] = (sig, rendered)
    _write_repomap_file(root_p, rendered)
    return rendered, True


def repo_map_for_root_cached(
    root: Path | str, *, max_tokens: int = 1500
) -> str | None:
    """Cached repo map (the "fresh-by-construction" path). Safe to call
    every turn — re-walks only when the source tree changed. The agent
    reads the map from its memory block; ``.loom/repomap.md`` mirrors it
    for inspection."""
    return _repo_map_cached(root, max_tokens)[0]


def repo_map_meta_for_root_cached(
    root: Path | str, *, max_tokens: int = 1500
) -> dict[str, object]:
    """Like :func:`repo_map_for_root_cached` but returns the map plus
    metadata for UI surfaces: ``{map, rebuilt, symbols, chars}``.
    ``rebuilt`` distinguishes a fresh re-index (tree changed) from a
    cache hit, so the desktop can show "re-indexed" vs "cached"."""
    body, rebuilt = _repo_map_cached(root, max_tokens)
    text = body or ""
    return {
        "map": text,
        "rebuilt": rebuilt,
        "symbols": text.count("\n- `"),
        "chars": len(text),
    }


def _write_repomap_file(root: Path, rendered: str | None) -> None:
    """Mirror the injected repo map to ``<root>/.loom/repomap.md`` so a
    human can see what the agent receives. Overwritten on every rebuild;
    best-effort (a write failure never affects the returned map). The
    agent reads the map from its memory block, not this file — it's
    inspection-only. ``.loom/`` is gitignored, so this isn't committed."""
    if not rendered:
        return
    try:
        loom_dir = root / ".loom"
        loom_dir.mkdir(parents=True, exist_ok=True)
        note = (
            "<!-- Auto-generated by loom-code. Rebuilt whenever the "
            "source tree changes; this is the repo map injected into the "
            "agent's system prompt each turn (the `loom_index` block). "
            "Inspection-only — do not edit, it is overwritten. -->\n\n"
        )
        (loom_dir / "repomap.md").write_text(
            note + rendered, encoding="utf-8"
        )
    except OSError:
        pass
