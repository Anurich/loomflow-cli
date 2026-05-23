"""loominit — structural codebase indexing for loom-code.

The **structural extractor** (:mod:`extractor`, via ``build_index``) is
deterministic and LLM-free: it walks the repo with Python's stdlib
``ast``, builds the import + call graphs, scores symbols by PageRank,
detects API surface from ``__init__.py`` / ``__all__``, maps tests to
symbols, and samples git heat — producing a :class:`schema.LoomIndex`.

This index feeds :mod:`repomap`, which ranks the most structurally-
important symbols and renders a compact, token-budgeted **repo map**.
That map is rebuilt fresh-by-construction (re-walked only when the
source tree changes) and injected into the agent's ``loom_index``
working block every turn — no LLM cost, no persisted artifact, never
stale. See ``repomap.repo_map_for_root_cached``.

(Historical note: this package once also ran an LLM annotator that
produced a human-readable ``LOOM.md`` + a persisted ``index.json``,
retrieved per turn via BM25. That subsystem was removed once the
deterministic repo map replaced it — the narrative drifted as the
agent edited code, and nothing read it at runtime.)
"""

from __future__ import annotations
