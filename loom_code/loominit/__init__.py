"""loominit — codebase indexing for loom-code.

The flow, end-to-end:

1. ``/loominit`` runs the **structural extractor** (:mod:`extractor`) —
   deterministic, no LLM. Walks the repo with Python's stdlib
   ``ast``, builds the import + call graphs, scores symbols by
   PageRank, detects API surface from ``__init__.py`` / ``__all__``,
   maps tests to symbols, samples git heat. Outputs the
   machine-readable :class:`schema.LoomIndex` to ``.loom/index.json``.

2. The **annotator** (:mod:`annotator`) takes that index as INPUT
   (the LLM does not re-grep). It clusters files, dispatches one
   explorer delegation per cluster via ``Team.supervisor``, and
   emits the human-readable ``LOOM.md`` — sectioned markdown with
   subsystem narratives, data flows, conventions (each verified by
   grep), and per-symbol one-line purposes. Every claim carries a
   ``path:line`` citation and a source-file hash.

3. On every subsequent agent run, :mod:`injection` retrieves the
   1-3 sections of ``LOOM.md`` relevant to the user's prompt via
   BM25, wraps stale claims (citations whose source-hash drifted)
   inline with ``(stale: path:line)``, and pins the result as a
   ``codebase_index`` working block — loomflow auto-injects it into
   the system prompt.

4. After any agent turn that wrote files, :mod:`staleness` re-runs
   the structural pass on touched files, re-hashes them, and
   updates ``.loom/index.json``. Existing claims whose source
   changed get a ``(stale)`` marker in ``LOOM.md``; new public
   symbols get appended to ``## Pending annotations``.

5. ``/loominit refresh`` (:mod:`refresh`) re-annotates only clusters
   with changed hash buckets or pending entries — surgical, not
   full-rebuild. User hand-edits in untouched sections survive.

The architectural rule: only the annotator costs LLM tokens.
Everything else (structural, staleness, retrieval) is deterministic
and cheap, so it can run automatically without user say-so.
"""

from __future__ import annotations
