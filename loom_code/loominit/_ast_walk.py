"""Python AST visitor — extracts symbols + imports from one file.

Single-file scope: ``walk_python_file(source, path)`` returns
``(symbols, imports, decorators)`` for the file. No cross-file
resolution happens here — that lives in :mod:`extractor`, which
owns the global symbol table and builds the import graph from
many per-file results.

Why stdlib ast (not tree-sitter): loom-code's first-party target
is Python. The stdlib parser is exact-grammar by definition and
zero-install. Tree-sitter (with grammar packages per language)
will come later for polyglot repos — :mod:`extractor` already
routes by language so a future `_treesitter_walk.py` can drop
in alongside this module without touching callers.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Literal

# Decorators that ALWAYS indicate a landmark (entry point / tool /
# fixture / route handler). The annotator uses these to surface
# entry points without an LLM call. Keep the list short — false
# positives are cheap (one more line in the entry-points section),
# false negatives mean the annotator has to discover them with
# grep, which costs tokens.
LANDMARK_DECORATORS: frozenset[str] = frozenset(
    {
        # CLI frameworks
        "click.command",
        "click.group",
        "typer.command",
        # Web frameworks (FastAPI, Flask, Starlette common shapes)
        "app.route",
        "app.get",
        "app.post",
        "app.put",
        "app.delete",
        "router.get",
        "router.post",
        "router.put",
        "router.delete",
        # loomflow / loom-code
        "tool",
        "step",
        # pytest / testing
        "pytest.fixture",
        "fixture",
    }
)


@dataclass(frozen=True)
class _RawSymbol:
    """Intermediate symbol record — what we extract from ast, before
    cross-file enrichment in :mod:`extractor` (PageRank, n_callers,
    test map, API-surface flag).

    Frozen so accidental mutation across the extractor pipeline
    surfaces as TypeError rather than a silent bug."""

    name: str
    qualified_name: str
    kind: Literal["class", "function", "method", "constant"]
    line: int
    end_line: int
    signature: str
    docstring_first_line: str | None
    decorators: tuple[str, ...]
    is_public: bool  # name does not start with "_"


@dataclass(frozen=True)
class _RawImport:
    """One import-edge candidate from one file.

    ``to_module`` is the dotted module path as written in source
    (``"foo.bar"`` for ``from foo.bar import x``). Relative-import
    resolution (``from .. import x``) and "does this resolve to a
    file in our repo" happen in :mod:`extractor` because they need
    the full file set.
    """

    to_module: str
    line: int
    # The literal level for ``from . import x`` style. 0 for absolute,
    # 1 for ``from .``, 2 for ``from ..``, etc. Needed downstream to
    # resolve relative imports against the package layout.
    level: int


@dataclass(frozen=True)
class _RawDecorator:
    """A decorator on a symbol that matches :data:`LANDMARK_DECORATORS`.

    ``decorator`` is normalized (no leading ``@``, no call args:
    ``app.route("/x")`` is stored as ``"app.route"``). ``target_qualname``
    is the qualified name of the decorated symbol (``Outer.method``).
    """

    decorator: str
    target_qualname: str
    line: int


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def walk_python_file(
    source: str, path: str
) -> tuple[list[_RawSymbol], list[_RawImport], list[_RawDecorator]]:
    """Parse ``source`` (the textual contents of ``path``) and return
    everything the cross-file extractor needs.

    ``path`` is repo-relative POSIX; we don't actually open the file
    here — the caller already read it. Keeping I/O out makes this
    function trivially unit-testable with inline strings.

    A syntax error returns three empty lists rather than raising:
    one broken file should NOT abort indexing a repo. The extractor
    logs the broken paths so the user sees what was skipped.
    """
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return [], [], []

    source_lines = source.splitlines()
    visitor = _Visitor(source_lines)
    visitor.visit(tree)
    return visitor.symbols, visitor.imports, visitor.decorators


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


class _Visitor(ast.NodeVisitor):
    """Walks a parsed Python file once. The traversal is class-aware
    (methods get ``Class.method`` qualified names + ``kind="method"``)
    but does NOT recurse into function bodies — symbols inside a
    function are local and not worth indexing.

    Decorators are filtered against :data:`LANDMARK_DECORATORS` here
    on the per-symbol decorator list, but ALSO collected verbatim
    onto every symbol (so the annotator can quote them in LOOM.md).
    """

    def __init__(self, source_lines: list[str]) -> None:
        self._source_lines = source_lines
        # Qualified-name prefix stack; pushed when entering ClassDef.
        self._scope: list[str] = []
        self.symbols: list[_RawSymbol] = []
        self.imports: list[_RawImport] = []
        self.decorators: list[_RawDecorator] = []

    # ---- class / function / method ----------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualified = self._qualify(node.name)
        decorators = tuple(self._decorator_names(node.decorator_list))
        self._record_landmarks(decorators, qualified, node)
        self.symbols.append(
            _RawSymbol(
                name=node.name,
                qualified_name=qualified,
                kind="class",
                line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno),
                signature=self._signature(node),
                docstring_first_line=_docstring_first_line(node),
                decorators=decorators,
                is_public=not node.name.startswith("_"),
            )
        )
        # Recurse INTO the class so methods inherit Class.qualname,
        # but use a fresh scope frame so we don't see siblings as
        # methods.
        self._scope.append(node.name)
        for child in node.body:
            self.visit(child)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_callable(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_callable(node)

    def _visit_callable(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        qualified = self._qualify(node.name)
        kind: Literal["function", "method"] = (
            "method" if self._scope else "function"
        )
        decorators = tuple(self._decorator_names(node.decorator_list))
        self._record_landmarks(decorators, qualified, node)
        self.symbols.append(
            _RawSymbol(
                name=node.name,
                qualified_name=qualified,
                kind=kind,
                line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno),
                signature=self._signature(node),
                docstring_first_line=_docstring_first_line(node),
                decorators=decorators,
                is_public=not node.name.startswith("_"),
            )
        )
        # Do NOT descend into function bodies — nested defs aren't
        # worth indexing as symbols.

    # ---- module-level constants -------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        # Only catch module-level assignments (no enclosing scope).
        if self._scope:
            return
        for target in node.targets:
            if isinstance(target, ast.Name) and _is_constant_name(target.id):
                self.symbols.append(
                    _RawSymbol(
                        name=target.id,
                        qualified_name=target.id,
                        kind="constant",
                        line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        signature=(
                            self._source_lines[node.lineno - 1].strip()
                            if 0 < node.lineno <= len(self._source_lines)
                            else f"{target.id} = ..."
                        ),
                        docstring_first_line=None,
                        decorators=(),
                        is_public=not target.id.startswith("_"),
                    )
                )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # ``X: T = value`` at module level — same handling as Assign.
        if self._scope:
            return
        if not isinstance(node.target, ast.Name):
            return
        name = node.target.id
        if not _is_constant_name(name):
            return
        self.symbols.append(
            _RawSymbol(
                name=name,
                qualified_name=name,
                kind="constant",
                line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno),
                signature=(
                    self._source_lines[node.lineno - 1].strip()
                    if 0 < node.lineno <= len(self._source_lines)
                    else f"{name}: ..."
                ),
                docstring_first_line=None,
                decorators=(),
                is_public=not name.startswith("_"),
            )
        )

    # ---- imports ----------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                _RawImport(
                    to_module=alias.name,
                    line=node.lineno,
                    level=0,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # node.module is None for ``from . import x`` — represent as
        # empty string so the resolver can still distinguish it from
        # an absent module reference.
        self.imports.append(
            _RawImport(
                to_module=node.module or "",
                line=node.lineno,
                level=node.level or 0,
            )
        )

    # ---- helpers ----------------------------------------------------

    def _qualify(self, name: str) -> str:
        """Build the qualified name relative to the current scope.
        ``["Outer", "Inner"] + "method"`` → ``"Outer.Inner.method"``."""
        if not self._scope:
            return name
        return ".".join((*self._scope, name))

    def _decorator_names(
        self, decorator_list: list[ast.expr]
    ) -> list[str]:
        """Convert decorator AST nodes to display strings.

        ``@foo`` → ``"foo"``
        ``@foo.bar`` → ``"foo.bar"``
        ``@foo(...)`` → ``"foo"`` (call args dropped)
        ``@foo.bar(x)`` → ``"foo.bar"``
        Anything else → ``ast.unparse`` (rare; lambdas, etc.).
        """
        out: list[str] = []
        for dec in decorator_list:
            out.append(_decorator_to_name(dec))
        return out

    def _record_landmarks(
        self,
        decorators: tuple[str, ...],
        target_qualname: str,
        node: ast.AST,
    ) -> None:
        """Match decorators against :data:`LANDMARK_DECORATORS` and
        record any hits. Match is exact (dotted form) — we don't try
        to handle aliased imports (``from click import command as
        cmd``); that's an explicit choice to keep the matcher cheap.
        """
        for dec in decorators:
            if dec in LANDMARK_DECORATORS:
                self.decorators.append(
                    _RawDecorator(
                        decorator=dec,
                        target_qualname=target_qualname,
                        line=node.lineno,
                    )
                )

    def _signature(
        self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
    ) -> str:
        """The verbatim source for the ``def …`` / ``class …`` line.

        Multi-line signatures (``def f(\\n  a,\\n) -> int:``) are
        collapsed onto one line by joining + whitespace-squashing,
        because the index is text — a single-line ground-truth string
        is more useful to the annotator than preserved formatting."""
        start = node.lineno - 1  # 1-indexed to 0-indexed
        # ``body`` is always non-empty for ClassDef / FunctionDef; ast
        # guarantees at least one statement (often a Pass or Expr).
        if not node.body:
            return self._source_lines[start].strip()
        body_start = node.body[0].lineno - 1
        sig_lines = self._source_lines[start:body_start]
        if not sig_lines:
            return self._source_lines[start].strip()
        joined = " ".join(line.strip() for line in sig_lines if line.strip())
        # Trim trailing ``:`` whitespace — looks nicer in LOOM.md.
        return joined.rstrip()


def _decorator_to_name(node: ast.expr) -> str:
    """Stringify a decorator node to ``foo.bar`` form.

    Handles three common cases explicitly + falls back to
    ``ast.unparse`` for anything exotic (lambdas, walrus, etc.).
    """
    if isinstance(node, ast.Call):
        return _decorator_to_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        # Walk the attribute chain right-to-left.
        parts: list[str] = []
        cur: ast.expr = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
        # Attribute on something complex (function call result, etc.)
        # — fall through to unparse.
    return ast.unparse(node)


def _is_constant_name(name: str) -> bool:
    """A module-level assignment counts as a "constant" symbol if
    it's NAMED_LIKE_THIS (uppercase + underscores) or is a
    type-alias-style name (PascalCase).

    Filter design: most module-level ``x = ...`` lines are private
    plumbing the agent shouldn't care about. CONSTANT_CASE and
    TypeAlias-style names are the ones with documentation value.
    Dunders (``__all__``, ``__version__``) are caught here too —
    they're useful index entries because they signal API surface.
    """
    if not name:
        return False
    if name.startswith("__") and name.endswith("__"):
        return True
    if name.isupper() or "_" in name and name.replace("_", "").isupper():
        return True
    # PascalCase (type aliases, sentinel singletons): first char upper,
    # contains a lowercase letter somewhere (rule out ALL_CAPS twice).
    if name[0].isupper() and any(c.islower() for c in name):
        return True
    return False


def _docstring_first_line(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> str | None:
    """Return the first non-empty line of the symbol's docstring, or
    ``None`` if absent. The annotator quotes this verbatim so the
    LOOM.md entry stays grounded in what the code actually says
    about itself."""
    raw = ast.get_docstring(node)
    if not raw:
        return None
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None
