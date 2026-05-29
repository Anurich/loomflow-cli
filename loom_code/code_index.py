"""Semantic codebase index — embed source symbols, search by meaning.

The differentiator loom-code ships over grep: ``grep`` finds the *string*
``authenticate``; ``codebase_search`` finds the code that *handles auth*
even when the word never appears. It mirrors Cursor's ``@Codebase`` —
but local, in the same ``.loom`` partition as memory, and (Phase 1b)
fusible with what the agent has *learned* about that code across runs.

How it works, end to end:

* **Chunk** — reuse the structural AST walk (:func:`walk_python_file`)
  to split each ``.py`` file into class/function/method chunks. Bare
  module constants are skipped (same call repomap's ``_score`` makes:
  they're noise in a semantic overview). Each chunk's embeddable text
  is ``path + qualified_name + signature + docstring + body`` — the
  body is sliced ``line:end_line`` from the file we already read.
* **Embed** — via the SAME embedder loom-code picks for memory
  (:class:`OpenAIEmbedder` for OpenAI chat models, :class:`HashEmbedder`
  otherwise — zero-key, offline, lower quality but never a cross-
  provider call). The caller passes the resolved name so the index and
  memory always embed in the same space (Phase 1b fuses them). Both
  embedders expose ``async embed_batch(texts) -> list[list[float]]``.
* **Store** — a SEPARATE sqlite db ``<root>/.loom/code_index.db``.
  NOT ``memory.db``: loomflow's memory schema is locked to Episodes /
  Facts, so a fourth data model (code chunks) gets its own file. Per-
  file ``sha256`` gates re-embedding — only changed files re-embed,
  which matters because OpenAI embedding costs real money per token.
* **Search** — cosine over the stored vectors, grouped + file:line
  cited like ``grep`` so the agent can ``read`` the exact range next.

Python-only today (the AST walk is stdlib ``ast``). The walk already
routes by language, so a future tree-sitter backend drops in here
without touching the tool or the store — the seam is the chunker, not
the index.

Failure is always graceful: a broken build, a missing embedder key, an
empty index — the tool returns a one-line explanation, never raises.
A semantic-search outage must not abort a turn.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from loomflow import tool
from loomflow.tools.registry import Tool

from .loominit._ast_walk import walk_python_file

# Directories we never index — vendored / generated / VCS noise. Same
# spirit as repomap's skip set; kept local so the two can diverge (the
# code index may later want to include tests, which the overview map
# collapses).
_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".loom",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".tox",
        "site-packages",
    }
)

# Chunk bodies are capped before embedding: a 2000-line god-function
# would blow the embedder's token limit AND dilute its own signal (the
# first ~120 lines carry the intent; the tail is detail). Slicing keeps
# embeddings cheap and focused. The agent reads the full range from the
# file:line citation anyway.
_MAX_CHUNK_LINES = 120

# Default result count — enough to surface the relevant cluster, few
# enough not to flood the model's context. Matches grep's file-cap feel.
_DEFAULT_LIMIT = 8


@dataclass(frozen=True)
class _Chunk:
    """One indexable code unit (a class / function / method)."""

    path: str  # repo-relative POSIX
    qualified_name: str
    kind: str
    start_line: int
    end_line: int
    signature: str
    text: str  # the embeddable doc (header + body)


@dataclass(frozen=True)
class CodeHit:
    """A semantic search result — enough to cite and re-read."""

    path: str
    qualified_name: str
    kind: str
    start_line: int
    end_line: int
    signature: str
    score: float


# ---------------------------------------------------------------------------
# Embedder resolution (shared with memory — see agent._is_openai_model)
# ---------------------------------------------------------------------------


def resolve_embedder(name: str) -> Any:
    """Build the embedder backend for ``name`` (``"openai"`` / ``"hash"``).

    Returns an object with ``async embed_batch(texts) -> list[list[
    float]]`` — the batch method both backends share. The caller passes
    the same name loom-code resolved for memory, so the code index and
    the note store embed in one vector space (Phase 1b reciprocal-rank-
    fuses across them). ``"hash"`` is the zero-key, offline default for
    non-OpenAI chat models; anything unrecognised also falls to hash so
    the index degrades to "works, lower quality" rather than crashing.
    """
    from loomflow.memory import HashEmbedder, OpenAIEmbedder

    if name == "openai":
        return OpenAIEmbedder()  # text-embedding-3-small, reads OPENAI_API_KEY
    return HashEmbedder()


# ---------------------------------------------------------------------------
# Chunking — AST walk -> embeddable units
# ---------------------------------------------------------------------------


def _iter_py_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        out.append(p)
    return out


def _file_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _chunks_for_file(rel_path: str, source: str) -> list[_Chunk]:
    """Split one file's source into embeddable chunks via the AST walk.

    Skips module-level constants (``kind == "constant"``) — they're a
    single line of value with no behaviour to search for, and they
    crowd out real symbols. A syntax error yields no chunks (the walk
    returns empty lists rather than raising), so one broken file never
    aborts a build.
    """
    symbols, _imports, _decorators = walk_python_file(source, rel_path)
    lines = source.splitlines()
    chunks: list[_Chunk] = []
    for sym in symbols:
        if sym.kind == "constant":
            continue
        # Slice the body we already have in memory. AST line numbers are
        # 1-based inclusive; clamp end to the cap so giant functions
        # don't blow the embedder budget (the citation still spans the
        # true range so the agent can read all of it).
        start = max(sym.line, 1)
        end = min(sym.end_line, start + _MAX_CHUNK_LINES - 1)
        body = "\n".join(lines[start - 1 : end])
        # The embeddable doc: location + identity + intent + body. The
        # path + qualname + docstring carry most of the semantic signal
        # cheaply; the body grounds it in the actual implementation.
        doc_parts = [f"{rel_path} :: {sym.qualified_name}", sym.signature]
        if sym.docstring_first_line:
            doc_parts.append(sym.docstring_first_line)
        doc_parts.append(body)
        chunks.append(
            _Chunk(
                path=rel_path,
                qualified_name=sym.qualified_name,
                kind=sym.kind,
                start_line=sym.line,
                end_line=sym.end_line,
                signature=sym.signature,
                text="\n".join(doc_parts),
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Store — sqlite, separate from memory.db
# ---------------------------------------------------------------------------

# Vectors persist as packed little-endian float32 blobs — compact and
# numpy-free (loom-code has no numpy dep; cosine is a plain loop). The
# dimension is implied by the blob length, so the store is agnostic to
# whether hash (384) or openai (1536) wrote it; switching embedders
# invalidates every file via the staleness gate, forcing a clean
# re-embed in the new dimension.


def _pack(vec: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


class CodeIndexStore:
    """The sqlite-backed code-chunk index for one project.

    Async note: sqlite calls here are synchronous and fast (local
    file). They run inside the tool's async function but are not
    offloaded to a thread — acceptable because a query is a single
    indexed read + an in-process cosine loop, well under the latency
    that would justify a thread pool. Embedding (the slow part) IS
    async and awaited.
    """

    def __init__(self, db_path: Path, embedder_name: str) -> None:
        self._db_path = db_path
        self._embedder_name = embedder_name
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id          TEXT PRIMARY KEY,   -- path::qualname
                path        TEXT NOT NULL,
                qualname    TEXT NOT NULL,
                kind        TEXT NOT NULL,
                start_line  INTEGER NOT NULL,
                end_line    INTEGER NOT NULL,
                signature   TEXT NOT NULL,
                embedding   BLOB NOT NULL
            )
            """
        )
        # Per-file content hash — re-embed only what changed. Stores the
        # embedder name too, so switching providers (hash -> openai)
        # invalidates every file (different vector space).
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                path     TEXT PRIMARY KEY,
                sha256   TEXT NOT NULL,
                embedder TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def file_is_fresh(self, rel_path: str, sha: str) -> bool:
        """True when ``rel_path`` is already indexed at ``sha`` with the
        current embedder — i.e. nothing to re-embed."""
        row = self._conn.execute(
            "SELECT sha256, embedder FROM files WHERE path = ?", (rel_path,)
        ).fetchone()
        return (
            row is not None
            and row[0] == sha
            and row[1] == self._embedder_name
        )

    def replace_file_chunks(
        self,
        rel_path: str,
        sha: str,
        chunks: list[_Chunk],
        vectors: list[Sequence[float]],
    ) -> None:
        """Atomically swap one file's chunks (delete-then-insert in a
        single transaction) so a crash mid-reindex never leaves a file
        half-indexed."""
        cur = self._conn
        cur.execute("DELETE FROM chunks WHERE path = ?", (rel_path,))
        for chunk, vec in zip(chunks, vectors):
            cur.execute(
                "INSERT OR REPLACE INTO chunks "
                "(id, path, qualname, kind, start_line, end_line, "
                "signature, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"{chunk.path}::{chunk.qualified_name}",
                    chunk.path,
                    chunk.qualified_name,
                    chunk.kind,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.signature,
                    _pack(vec),
                ),
            )
        cur.execute(
            "INSERT OR REPLACE INTO files (path, sha256, embedder) "
            "VALUES (?, ?, ?)",
            (rel_path, sha, self._embedder_name),
        )
        cur.commit()

    def prune_missing(self, live_paths: set[str]) -> None:
        """Drop chunks/files for source files that no longer exist (the
        delete half of incremental indexing)."""
        rows = self._conn.execute("SELECT path FROM files").fetchall()
        stale = [r[0] for r in rows if r[0] not in live_paths]
        for path in stale:
            self._conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
            self._conn.execute("DELETE FROM files WHERE path = ?", (path,))
        if stale:
            self._conn.commit()

    def search(self, query_vec: Sequence[float], limit: int) -> list[CodeHit]:
        """Cosine-rank every stored chunk against ``query_vec``.

        A linear scan: fine for the tens-of-thousands of symbols a
        normal repo has (each cosine is a 384–1536-float dot product).
        If a monorepo ever makes this slow, the swap-in is a vector
        index (sqlite-vec / faiss) behind this same method — callers
        don't change.
        """
        rows = self._conn.execute(
            "SELECT path, qualname, kind, start_line, end_line, signature, "
            "embedding FROM chunks"
        ).fetchall()
        qn = _norm(query_vec)
        if qn == 0.0:
            return []
        scored: list[CodeHit] = []
        for path, qual, kind, start, end, sig, blob in rows:
            vec = _unpack(blob)
            score = _cosine(query_vec, vec, qn)
            scored.append(
                CodeHit(
                    path=path,
                    qualified_name=qual,
                    kind=kind,
                    start_line=start,
                    end_line=end,
                    signature=sig,
                    score=score,
                )
            )
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:limit]

    def is_empty(self) -> bool:
        row = self._conn.execute("SELECT 1 FROM chunks LIMIT 1").fetchone()
        return row is None


def _norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def _cosine(a: Sequence[float], b: Sequence[float], a_norm: float) -> float:
    """Cosine similarity; ``a_norm`` is precomputed (the query norm is
    constant across all chunks, so we hoist it out of the scan loop).
    Mismatched dims (shouldn't happen — the staleness gate forces a
    uniform embedder) score 0 rather than raising."""
    if len(a) != len(b):
        return 0.0
    bn = _norm(b)
    if bn == 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (a_norm * bn)


# ---------------------------------------------------------------------------
# Build — incremental index over the tree
# ---------------------------------------------------------------------------


async def build_index(
    root: Path, store: CodeIndexStore, embedder: Any
) -> tuple[int, int]:
    """(Re)index ``root`` into ``store``. Returns ``(files_embedded,
    files_skipped)``.

    Incremental: a file whose sha256 + embedder match the stored row is
    skipped (no re-embed). Deleted files are pruned. Embedding is
    batched per file (one ``embed_batch()`` call covers all of a file's
    chunks) to amortise the API round-trip.
    """
    files = _iter_py_files(root)
    live: set[str] = set()
    embedded = 0
    skipped = 0
    for fpath in files:
        rel = fpath.relative_to(root).as_posix()
        live.add(rel)
        try:
            source = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        sha = _file_sha256(source)
        if store.file_is_fresh(rel, sha):
            skipped += 1
            continue
        chunks = _chunks_for_file(rel, source)
        if not chunks:
            # Empty/constant-only file: record the hash so we don't
            # re-walk it every build, but store no chunks.
            store.replace_file_chunks(rel, sha, [], [])
            continue
        vectors = await embedder.embed_batch([c.text for c in chunks])
        store.replace_file_chunks(rel, sha, chunks, vectors)
        embedded += 1
    store.prune_missing(live)
    return embedded, skipped


async def search_code(
    root: Path | str, embedder_name: str, query: str, *, limit: int = 8
) -> list[CodeHit]:
    """Structured semantic search — build/refresh the index for ``root``
    and return ranked :class:`CodeHit`s for ``query``.

    The structured-results entry point (the tool returns rendered text
    for the model; callers that need ``(path, score, line)`` — the
    desktop ``@Codebase`` RPC — use this). Builds lazily + incrementally
    like the tool, so first call on a fresh repo embeds, later calls are
    cheap. Returns ``[]`` (never raises) on an empty index so the caller
    can degrade gracefully.
    """
    root_p = Path(root).resolve()
    db_path = root_p / ".loom" / "code_index.db"
    db_path.parent.mkdir(exist_ok=True)
    embedder = resolve_embedder(embedder_name)
    store = CodeIndexStore(db_path, embedder_name)
    try:
        await build_index(root_p, store, embedder)
        if store.is_empty():
            return []
        qvecs = await embedder.embed_batch([query])
        return store.search(qvecs[0], limit)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Tool — codebase_search
# ---------------------------------------------------------------------------


def _render_hits(hits: list[CodeHit]) -> str:
    """Cite file:line so the agent can ``read`` the exact range next —
    same citation shape as grep."""
    if not hits:
        return "no semantic matches"
    out: list[str] = []
    for h in hits:
        loc = f"{h.path}:{h.start_line}-{h.end_line}"
        out.append(f"  [{h.score:.2f}] {h.kind} {h.qualified_name}  ({loc})")
        out.append(f"        {h.signature.strip()}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Phase 1b — blend code hits with learned notes (the differentiator)
# ---------------------------------------------------------------------------

# Reciprocal Rank Fusion constant. RRF score for an item at rank r (0-
# based) in a list is 1/(k + r + 1); summed across lists. k=60 is the
# canonical value (Cormack et al.) — it damps the top-rank dominance so
# a strong #2 in both lists beats a #1-in-one/absent-in-other. We fuse
# by RANK not raw score precisely because the two stores live in
# different score spaces (cosine vs the notebook's BM25/hybrid RRF) —
# ranks are comparable, raw scores are not. THIS is the call that lets
# "the code that does X" and "what we learned about X" share one list.
_RRF_K = 60


@dataclass(frozen=True)
class _BlendRow:
    """A unified result — either a code symbol or a learned note."""

    kind: str  # "code" or "note"
    label: str  # rendered one-liner
    rrf: float


def _fuse(
    code_hits: list[CodeHit], note_matches: list[Any]
) -> list[_BlendRow]:
    """Reciprocal-rank-fuse code symbols and notes into one ranked list.

    Each source contributes ``1/(k + rank)`` per item; since an item
    appears in only one source here (a code symbol is never also a
    note), the fusion is really an interleave weighted by within-source
    rank — a #1 code hit and a #1 note land adjacent, a #5 note sinks
    below a #2 code hit. Keeps the best of both surfaces visible
    instead of letting whichever store happens to score higher in its
    own units dominate.
    """
    rows: list[_BlendRow] = []
    for rank, h in enumerate(code_hits):
        loc = f"{h.path}:{h.start_line}-{h.end_line}"
        rows.append(
            _BlendRow(
                kind="code",
                label=(
                    f"  code  {h.kind} {h.qualified_name}  ({loc})\n"
                    f"        {h.signature.strip()}"
                ),
                rrf=1.0 / (_RRF_K + rank + 1),
            )
        )
    for rank, m in enumerate(note_matches):
        # NoteMatch = (summary: NoteSummary, score: float, snippet: str).
        # ``summary`` is the STRUCTURED note metadata (NOT a string) —
        # pull .title/.slug off it. ``snippet`` is the query-relevant
        # text excerpt. Show title + the excerpt so the agent sees what
        # was learned and why it matched, plus the slug to read_note.
        summary = getattr(m, "summary", None)
        title = (getattr(summary, "title", None) or "learned note").strip()
        slug = getattr(summary, "slug", "") or ""
        snippet = (getattr(m, "snippet", "") or "").strip().replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        header = f"  learned  {title}"
        if slug:
            header += f"  (note:{slug})"
        label = header
        if snippet:
            label += f"\n        {snippet}"
        rows.append(
            _BlendRow(
                kind="note",
                label=label,
                rrf=1.0 / (_RRF_K + rank + 1),
            )
        )
    rows.sort(key=lambda r: r.rrf, reverse=True)
    return rows


def _render_blend(rows: list[_BlendRow], limit: int) -> str:
    if not rows:
        return "no semantic matches"
    return "\n".join(r.label for r in rows[:limit])


def _current_user_id() -> str | None:
    """The live tenant from the run context (set by the agent loop), or
    None outside a run. Keeps note recall partitioned per user in the
    multi-tenant desktop; harmless (None) in the single-tenant CLI."""
    try:
        from loomflow.core.context import get_run_context

        ctx = get_run_context()
        return getattr(ctx, "user_id", None) if ctx is not None else None
    except Exception:
        return None


def codebase_search_tool(
    workdir: Path | str,
    embedder_name: str,
    *,
    default_limit: int = _DEFAULT_LIMIT,
    workspace: Any | None = None,
) -> Tool:
    """Build the ``codebase_search`` tool for ``workdir``.

    The model sees::

        codebase_search(query, limit=8)

    ``query`` is a natural-language description of behaviour ("where do
    we validate JWTs", "the retry/backoff logic"). Returns the most
    semantically similar code symbols with file:line citations to
    ``read`` next. Use this when ``grep`` would miss the code because
    the words don't match the concept; use ``grep`` when you know the
    literal string.

    ``embedder_name`` (``"openai"`` / ``"hash"``) MUST match the name
    loom-code resolved for memory, so the index embeds in the same
    space the notes do. The index is built lazily on first search and
    incrementally refreshed each call — only changed files re-embed, so
    steady-state search is cheap.

    ``workspace`` (Phase 1b — the differentiator): when a
    ``LocalDiskWorkspace`` is passed, every search ALSO queries the
    shared notebook (``search_notes``, hybrid + citation-boosted) and
    reciprocal-rank-fuses the learned notes INTO the code results. One
    ranked list then surfaces "the code that does X" *and* "what we
    learned about X across past runs" — the thing a stateless indexer
    (Cursor's ``@Codebase``) structurally cannot do. ``None`` falls
    back to code-only results (identical to Phase 1).
    """
    root = Path(workdir).resolve()
    db_path = root / ".loom" / "code_index.db"

    # Lazily constructed on first call so building the agent stays cheap
    # (no disk/embedder touch until the tool actually runs) and so an
    # embedder import error surfaces as a tool message, not a build
    # crash.
    state: dict[str, Any] = {"store": None, "embedder": None}

    async def codebase_search(query: str, limit: int = default_limit) -> str:
        """Semantic code search — find symbols by meaning, not string,
        blended with what we've learned about this code. Args: query
        (natural language), limit (max results, default 8). Returns a
        ranked list of code symbols (file:line to read next) and any
        relevant learned notes. Prefer grep for literal strings; use
        this for conceptual lookups."""
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = default_limit
        limit = max(1, min(limit, 50))

        try:
            db_path.parent.mkdir(exist_ok=True)
            if state["store"] is None:
                state["embedder"] = resolve_embedder(embedder_name)
                state["store"] = CodeIndexStore(db_path, embedder_name)
            store: CodeIndexStore = state["store"]
            embedder = state["embedder"]

            # Incremental refresh — cheap when nothing changed (all
            # files skip on sha match). First call on a fresh repo does
            # the full embed.
            await build_index(root, store, embedder)

            code_hits: list[CodeHit] = []
            if not store.is_empty():
                qvecs = await embedder.embed_batch([query])
                # Over-fetch so the fusion has depth to rank against the
                # notes; the blend trims to ``limit``.
                code_hits = store.search(qvecs[0], limit * 2)

            # Phase 1b: pull learned notes from the shared notebook and
            # fuse. A notebook failure (or no workspace) degrades to
            # code-only — never an error.
            note_matches: list[Any] = []
            if workspace is not None:
                try:
                    note_matches = await workspace.search_notes(
                        query,
                        user_id=_current_user_id(),
                        mode="hybrid",
                        boost_relevance=True,
                        limit=limit,
                    )
                except Exception:
                    note_matches = []

            if not code_hits and not note_matches:
                if store.is_empty():
                    return (
                        "codebase_search: index is empty (no Python "
                        "symbols found under this project) and no learned "
                        "notes match. Use grep for non-Python files."
                    )
                return "no semantic matches"

            if note_matches:
                return _render_blend(_fuse(code_hits, note_matches), limit)
            return _render_hits(code_hits[:limit])
        except Exception as exc:  # never abort a turn on a search failure
            return (
                f"codebase_search unavailable ({type(exc).__name__}: {exc}). "
                "Fall back to grep."
            )

    return tool(
        name="codebase_search",
        description=(
            "Semantic code search: find code by MEANING, not literal "
            "string — blended with what we've learned about this code "
            "across past runs. Args: query (natural-language description "
            "of the behaviour, e.g. 'where JWTs are validated'), limit=8. "
            "Returns ranked code symbols (file:line to read next) plus "
            "any relevant learned notes. Use this when grep would miss "
            "the code because the words don't match the concept; use "
            "grep when you know the exact string."
        ),
    )(codebase_search)
