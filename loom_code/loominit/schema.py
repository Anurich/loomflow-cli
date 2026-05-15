"""Schema for ``.loom/index.json`` — the machine-readable index.

This file is the CONTRACT between every other loominit module. The
extractor produces it; the annotator consumes it; the staleness
pipeline diffs against it; the refresh pass re-emits it. Lock the
shape carefully — every field is hash-keyed where possible so the
diff-aware refresh in :mod:`refresh` can identify what changed.

We use Pydantic v2 because loomflow already pins it (``>=2.6``) —
no new dep. Models are immutable (``frozen=True``) to surface
accidental mutation as a TypeError; the refresh pass replaces, not
edits.

Versioning: ``LoomIndex.version`` is an integer. Bumping it
invalidates older ``index.json`` files on read. Bump it any time
the JSON shape changes incompatibly; for additive fields, the
``model_config`` ``extra="ignore"`` policy keeps older readers
working against newer writers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Bump on incompatible schema changes. The reader (``persistence.py``)
# refuses to load an index whose ``version`` differs from this — and
# the REPL nudges the user to ``/loominit rebuild``.
SCHEMA_VERSION = 1


class _Immutable(BaseModel):
    """Shared base — every loominit model is immutable + ignores
    unknown fields so newer writers and older readers stay compatible
    as long as we only add fields."""

    model_config = ConfigDict(frozen=True, extra="ignore")


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


class FileEntry(_Immutable):
    """One file in the repo. ``path`` is repo-relative POSIX.

    ``sha256`` is the content hash the staleness pipeline diffs
    against. ``in_api_surface`` is true when the file is reachable
    from a package's ``__init__.py`` — agents over-read internals,
    so this lets the annotator default to API-level descriptions.
    """

    path: str
    lang: Literal["python", "markdown", "toml", "yaml", "json", "other"]
    size_bytes: int
    lines: int
    sha256: str
    mtime: datetime
    # Number of commits touching this file in the last 90 days. None
    # if the repo is not a git checkout. Used as a heat hint when
    # ranking which files to annotate first.
    git_changes_90d: int | None
    is_test: bool
    in_api_surface: bool


# ---------------------------------------------------------------------------
# Symbols (top-level class / function / module-level constant)
# ---------------------------------------------------------------------------


class SymbolEntry(_Immutable):
    """One symbol — class, function, method, or module-level constant.

    ``id`` is ``<path>:<qualified_name>`` (POSIX). The qualified name
    is dotted for nested classes/methods (``Repl.run``); module
    constants use just the bare name.

    ``signature`` is the verbatim Python source for the
    ``def …`` / ``class …`` line (single-line, no body). Useful in
    LOOM.md as ground-truth — the annotator can quote it.

    ``pagerank`` is the symbol's centrality in the import+call graph.
    The annotator picks the top-K by this score plus everything in
    ``in_api_surface=True``.

    ``tests`` lists test-file callers (``tests/foo.py:42``). Built by
    grepping test directories for the symbol's bare name — exact
    matches only; the agent can verify if false-positive worries it.
    """

    id: str  # f"{path}:{qualified_name}"
    name: str  # bare name ("login")
    qualified_name: str  # dotted ("AuthManager.login")
    kind: Literal["class", "function", "method", "constant"]
    path: str
    line: int
    end_line: int
    signature: str
    docstring_first_line: str | None
    decorators: list[str] = Field(default_factory=list)
    is_public: bool  # not _-prefixed; in __all__ if defined
    in_api_surface: bool  # reachable through package __init__.py
    pagerank: float
    n_callers: int
    n_callees: int
    tests: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph edges
# ---------------------------------------------------------------------------


class ImportEdge(_Immutable):
    """An ``import`` or ``from X import Y`` statement.

    ``from_path`` and ``to_module`` are dotted Python module paths
    when resolvable, otherwise the literal string from the source.
    Unresolvable imports (third-party deps, stdlib) are still
    recorded — they're a useful tech-stack signal — but never
    contribute to the call graph."""

    from_path: str
    to_module: str
    line: int
    resolved: bool  # True if to_module maps to a file in this repo


class CallEdge(_Immutable):
    """A function-call edge ``caller -> callee``.

    Both are symbol IDs (``<path>:<qualified_name>``). We only record
    edges where BOTH ends resolve to symbols we know about — calls
    into stdlib / third-party are dropped (would dominate PageRank
    noise). Line is the call-site line in ``caller``'s file."""

    caller: str
    callee: str
    line: int


# ---------------------------------------------------------------------------
# Landmarks — decorators / entry points
# ---------------------------------------------------------------------------


class DecoratorLandmark(_Immutable):
    """A "landmark" decorator the annotator treats specially —
    ``@app.route``, ``@click.command``, ``@tool``, ``@step``,
    ``@pytest.fixture``, ``@dataclass``, ``@property``. These mark
    entry points, tool definitions, fixtures, etc., and the
    annotator uses them to populate the Entry Points section without
    needing the LLM to discover them.

    ``decorator`` is the source-form name (``"@app.route"`` —
    keep the @ to disambiguate from regular calls). ``target`` is
    the symbol ID being decorated."""

    decorator: str
    target: str  # symbol id
    path: str
    line: int


class EntryPoint(_Immutable):
    """A user-facing entry to the program.

    Sources we mine:

    * pyproject.toml ``[project.scripts]`` — ``kind="pyproject_script"``
    * ``if __name__ == "__main__":`` blocks — ``kind="main_block"``
    * Decorators on the landmark allow-list (``@click.command``,
      ``@app.route``, etc.) — ``kind="decorated"``
    """

    kind: Literal["pyproject_script", "main_block", "decorated"]
    name: str  # CLI name / route path / function name
    path: str
    line: int | None
    callable_id: str | None  # symbol id, when known


# ---------------------------------------------------------------------------
# Clusters — file groups the annotator treats as one subsystem
# ---------------------------------------------------------------------------


class Cluster(_Immutable):
    """A subsystem — group of files the annotator describes together.

    Clustering signal: (1) path-prefix grouping (``loom_code/loominit/*``
    is one cluster) takes precedence; (2) within larger packages, an
    import-graph community-detection pass splits further.

    ``hash_bucket`` is a hash over the SORTED list of file sha256s
    in the cluster. When a file changes, the bucket changes; the
    surgical refresh in :mod:`refresh` re-annotates only clusters
    whose bucket moved.
    """

    id: str  # stable slug; used as the LOOM.md section anchor
    title: str  # human-readable ("Loominit indexer")
    paths: list[str]
    centroid_symbols: list[str] = Field(default_factory=list)
    centrality: float  # mean PageRank over symbols in the cluster
    hash_bucket: str


# ---------------------------------------------------------------------------
# The top-level container
# ---------------------------------------------------------------------------


class LoomIndex(_Immutable):
    """The whole ``.loom/index.json`` — produced by the structural
    extractor, consumed by everything else.

    Field ordering matches the conceptual layering: metadata first,
    then files, then symbols, then graph edges, then landmarks /
    clusters. Don't reorder — git diffs read more cleanly when the
    JSON output preserves this shape across refreshes.
    """

    version: int = SCHEMA_VERSION
    generated_at: datetime
    repo_root: str  # absolute path on the generating machine
    git_commit: str | None  # None if not a git repo
    # Number of LLM calls the annotator made (recorded so /loominit
    # status can report cost ballpark without re-reading workspace).
    annotation_calls: int = 0

    files: list[FileEntry] = Field(default_factory=list)
    symbols: list[SymbolEntry] = Field(default_factory=list)
    imports: list[ImportEdge] = Field(default_factory=list)
    calls: list[CallEdge] = Field(default_factory=list)
    decorators: list[DecoratorLandmark] = Field(default_factory=list)
    entry_points: list[EntryPoint] = Field(default_factory=list)
    clusters: list[Cluster] = Field(default_factory=list)
