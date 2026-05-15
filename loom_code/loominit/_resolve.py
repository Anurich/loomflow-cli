"""Cross-file resolution — imports, API surface, entry points.

Three things only resolvable once we have the full file set:

1. **Import resolution.** ``from foo.bar import x`` is a dotted
   module name in :mod:`_ast_walk`'s output; here we map it to a
   file path within the repo (or mark unresolved for third-party /
   stdlib imports). Relative imports (``from .. import x``) need
   the importing file's package depth to resolve.

2. **API surface.** A symbol is in the "API surface" when reachable
   from a package's ``__init__.py`` — either re-exported by a
   ``from .module import X`` line or named in ``__all__``. Agents
   over-read internals; flagging API symbols lets the annotator
   default to the public surface.

3. **Entry points.** ``pyproject.toml [project.scripts]`` gives us
   ``loom-code = "loom_code.cli:main"``-style entries directly. We
   also surface ``if __name__ == "__main__":`` blocks (mined from
   :mod:`_ast_walk`'s symbol output via a second AST pass) and any
   landmark decorators (``@click.command``, etc.) collected during
   the walk.

All three live here because they share machinery — they all need
to map dotted module names to repo-relative file paths.
"""

from __future__ import annotations

import ast
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ._ast_walk import _RawDecorator
from ._files import DiscoveredFile
from .schema import EntryPoint, ImportEdge


@dataclass(frozen=True)
class _ModuleIndex:
    """Maps dotted module names to repo-relative file paths. Built
    once from the file list, queried by every resolution step.

    Two kinds of mapping:

    * ``modules[dotted] = rel_path`` — e.g.
      ``"loom_code.cli" -> "loom_code/cli.py"``
    * ``packages[dotted] = init_rel_path`` — e.g.
      ``"loom_code" -> "loom_code/__init__.py"``

    Both maps are populated so ``from loom_code import cli`` resolves
    correctly: the resolver checks packages first (``loom_code``
    is a package), then attempts to find ``cli`` as a sub-module.
    """

    modules: dict[str, str]
    packages: dict[str, str]


def build_module_index(files: list[DiscoveredFile]) -> _ModuleIndex:
    """Build the dotted-module ↔ file-path map.

    ``foo/bar/baz.py`` → module ``foo.bar.baz``
    ``foo/bar/__init__.py`` → package ``foo.bar``

    We don't try to detect namespace packages (PEP 420) — those
    have no ``__init__.py`` and disambiguation requires a sys.path
    walk. Loom-code's first-party layout always has explicit
    package roots, so this is fine in practice.
    """
    modules: dict[str, str] = {}
    packages: dict[str, str] = {}
    for f in files:
        if f.lang != "python":
            continue
        parts = f.rel_path.split("/")
        if parts[-1] == "__init__.py":
            dotted = ".".join(parts[:-1])
            packages[dotted] = f.rel_path
        else:
            name = parts[-1].removesuffix(".py").removesuffix(".pyi")
            dotted = ".".join((*parts[:-1], name))
            modules[dotted] = f.rel_path
    return _ModuleIndex(modules=modules, packages=packages)


def resolve_import(
    *,
    from_file: str,
    to_module: str,
    level: int,
    module_index: _ModuleIndex,
) -> str | None:
    """Turn a raw import into a repo-relative file path.

    Returns the target file's rel_path on success, or ``None`` for
    unresolved (third-party / stdlib / typo) imports — those still
    get recorded in the :class:`schema.ImportEdge` list with
    ``resolved=False`` (useful tech-stack signal) but never feed
    into the PageRank graph.
    """
    target = _resolve_dotted(from_file, to_module, level)
    if target is None:
        return None
    # Try as module first (``foo.bar`` → ``foo/bar.py``), then as
    # package (``foo.bar`` → ``foo/bar/__init__.py``).
    if target in module_index.modules:
        return module_index.modules[target]
    if target in module_index.packages:
        return module_index.packages[target]
    # Could also be ``from foo.bar import baz`` where ``foo.bar``
    # is the module and ``baz`` is a symbol within. We treat the
    # edge as pointing to the module file — granularity is at the
    # file level, which is what PageRank needs.
    parent = ".".join(target.split(".")[:-1])
    if parent in module_index.modules:
        return module_index.modules[parent]
    if parent in module_index.packages:
        return module_index.packages[parent]
    return None


def _resolve_dotted(
    from_file: str, to_module: str, level: int
) -> str | None:
    """Apply Python's relative-import rules to produce an absolute
    dotted module path.

    ``level=0`` → ``to_module`` is already absolute.
    ``level=1`` → ``to_module`` is relative to ``from_file``'s
                  package.
    ``level=2`` → relative to the parent package, etc.

    Returns ``None`` if the level overshoots (``from .. import x``
    in a top-level package).
    """
    if level == 0:
        return to_module or None
    parts = from_file.split("/")
    # ``from_file = "a/b/c.py"`` → package parts = ["a", "b"]
    pkg_parts = parts[:-1]
    if len(pkg_parts) < level:
        return None
    base = pkg_parts[: len(pkg_parts) - level + 1]
    if to_module:
        base = [*base, *to_module.split(".")]
    return ".".join(base) if base else None


def resolve_imports(
    raw_imports_by_file: dict[str, list[tuple[str, int, int]]],
    module_index: _ModuleIndex,
) -> list[ImportEdge]:
    """Resolve every ``_RawImport`` produced by :mod:`_ast_walk`
    into the schema's :class:`ImportEdge`.

    Input shape: ``{from_path: [(to_module, line, level), ...]}``.
    We expose this rather than a list of :class:`_RawImport` so the
    caller controls how the per-file results are aggregated.
    """
    edges: list[ImportEdge] = []
    for from_path, items in raw_imports_by_file.items():
        for to_module, line, level in items:
            resolved_path = resolve_import(
                from_file=from_path,
                to_module=to_module,
                level=level,
                module_index=module_index,
            )
            # Store the dotted module name as-written when resolvable
            # to a real file, ELSE preserve the literal source form
            # — third-party imports are a useful tech-stack signal
            # the annotator can read.
            display_module = (
                to_module
                if level == 0
                else _relative_display(level, to_module)
            )
            edges.append(
                ImportEdge(
                    from_path=from_path,
                    to_module=display_module,
                    line=line,
                    resolved=resolved_path is not None,
                )
            )
    return edges


def _relative_display(level: int, to_module: str) -> str:
    """Render ``from .. import x`` style imports in a stable text
    form for the schema. ``level=2, to_module="x"`` → ``"..x"``."""
    return ("." * level) + to_module


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def detect_api_surface(
    files: list[DiscoveredFile], module_index: _ModuleIndex
) -> set[str]:
    """Return the set of ``rel_path``s reachable from any
    ``__init__.py``.

    Heuristic:

    * Every file an ``__init__.py`` does ``from .module import X``
      against → API surface.
    * Every dotted entry in an ``__all__`` literal of an
      ``__init__.py`` whose target resolves to a file → API surface.

    We do NOT chase the dependency graph past the first hop — being
    "imported by something on the API surface" is private-by-default.
    If a user pulls a helper into ``__init__.py`` to publish it,
    they're saying *that* helper is public; its callees aren't
    automatically.
    """
    api: set[str] = set()
    for f in files:
        if not f.rel_path.endswith("__init__.py"):
            continue
        try:
            tree = ast.parse(f.abs_path.read_text(encoding="utf-8"))
        except (SyntaxError, OSError):
            continue
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                target = resolve_import(
                    from_file=f.rel_path,
                    to_module=node.module or "",
                    level=node.level or 0,
                    module_index=module_index,
                )
                if target is not None:
                    api.add(target)
            elif isinstance(node, ast.Assign):
                # Find __all__ = ["a", "b"] — each name resolved as
                # a sibling module of this __init__.py.
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        api.update(
                            _resolve_all_entries(
                                node.value, f.rel_path, module_index
                            )
                        )
    return api


def _resolve_all_entries(
    value: ast.expr, init_path: str, module_index: _ModuleIndex
) -> set[str]:
    """``__all__ = [...]`` — pull out string literals and try to
    resolve each as a sibling module of ``init_path``. Anything that
    doesn't resolve gets quietly skipped (the entry might be a
    re-export symbol, not a sub-module — that's still API surface
    but covered by the ImportFrom pass)."""
    paths: set[str] = set()
    if not isinstance(value, ast.List | ast.Tuple):
        return paths
    init_pkg = ".".join(init_path.split("/")[:-1])
    for elt in value.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            dotted = f"{init_pkg}.{elt.value}" if init_pkg else elt.value
            if dotted in module_index.modules:
                paths.add(module_index.modules[dotted])
            elif dotted in module_index.packages:
                paths.add(module_index.packages[dotted])
    return paths


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def extract_entry_points(
    *,
    repo_root: Path,
    files: list[DiscoveredFile],
    decorators: list[_RawDecorator],
    decorator_path_lookup: dict[_RawDecorator, str],
) -> list[EntryPoint]:
    """Mine entry points from three sources.

    ``decorator_path_lookup`` maps every ``_RawDecorator`` back to
    the file it came from — the extractor builds it during its
    aggregation pass (decorators don't carry the source file inside
    the dataclass because :mod:`_ast_walk` is per-file already).
    """
    out: list[EntryPoint] = []

    # 1. pyproject.toml [project.scripts]
    pyproject = repo_root / "pyproject.toml"
    if pyproject.exists():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, OSError):
            data = {}
        scripts = (
            data.get("project", {}).get("scripts", {})
            if isinstance(data.get("project"), dict)
            else {}
        )
        for name, target in scripts.items():
            if not isinstance(target, str):
                continue
            out.append(
                EntryPoint(
                    kind="pyproject_script",
                    name=name,
                    path="pyproject.toml",
                    line=None,
                    callable_id=_target_to_symbol_id(target, files),
                )
            )

    # 2. ``if __name__ == "__main__":`` blocks
    for f in files:
        if f.lang != "python":
            continue
        try:
            tree = ast.parse(f.abs_path.read_text(encoding="utf-8"))
        except (SyntaxError, OSError):
            continue
        for node in tree.body:
            if _is_main_block(node):
                out.append(
                    EntryPoint(
                        kind="main_block",
                        name=f.rel_path,
                        path=f.rel_path,
                        line=node.lineno,
                        callable_id=None,
                    )
                )
                break  # one main block per file is enough

    # 3. Landmark decorators
    for dec in decorators:
        path = decorator_path_lookup.get(dec)
        if path is None:
            continue
        out.append(
            EntryPoint(
                kind="decorated",
                name=dec.decorator,
                path=path,
                line=dec.line,
                callable_id=f"{path}:{dec.target_qualname}",
            )
        )
    return out


def _is_main_block(node: ast.stmt) -> bool:
    """``if __name__ == "__main__":`` detection — exact AST shape
    match (no fuzziness; if the user wrote it weirdly that's on
    them)."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if not (
        isinstance(test.left, ast.Name) and test.left.id == "__name__"
    ):
        return False
    if len(test.comparators) != 1 or len(test.ops) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    rhs = test.comparators[0]
    return isinstance(rhs, ast.Constant) and rhs.value == "__main__"


def _target_to_symbol_id(
    target: str, files: list[DiscoveredFile]
) -> str | None:
    """``"loom_code.cli:main"`` → ``"loom_code/cli.py:main"`` when
    the module file exists. Returns ``None`` otherwise (annotator
    falls back to the literal string in LOOM.md)."""
    if ":" not in target:
        return None
    module_dotted, _, callable_name = target.partition(":")
    rel = module_dotted.replace(".", "/") + ".py"
    for f in files:
        if f.rel_path == rel:
            return f"{rel}:{callable_name}"
    return None
