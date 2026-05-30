"""LSP-backed navigation tools — go-to-definition, find-references, hover.

The agent's answer to "where is X defined / who calls X / what is X" WITHOUT
grep. grep finds the *string* ``login``; these resolve the *symbol* —
through imports, across files, respecting scope — the way an IDE's
"Go to Definition" does. Built on ``jedi`` (the pure-Python engine
behind most editors' Python intelligence): no language-server process,
no protocol, just in-process static analysis.

Three tools, all keyed by a bare or dotted symbol name (the agent
rarely knows exact line/columns, so we resolve the name across the
project first, then run jedi at that location):

* ``go_to_definition(symbol)`` — where it's defined (file:line + signature).
* ``find_references(symbol)`` — every place it's used (file:line list).
* ``hover(symbol)`` — its signature + docstring (the "what is this").

Python-only (jedi is Python-only), matching the rest of loom-code's v1
static-analysis surface (repomap, code index, AST walk). When jedi
isn't installed, or a symbol can't be resolved, the tool returns a
plain explanatory string and suggests grep — it never raises, so a
navigation miss never aborts a turn.

Why jedi over a real LSP server (pyright/pylsp): jedi is a single pure-
Python dependency, starts instantly (no server lifecycle), and delivers
>90% of the navigation value at a fraction of the weight. A real server
is a future option if multi-language support is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loomflow import tool
from loomflow.tools.registry import Tool

# Directories we never scan for symbol resolution — vendored / generated
# / VCS noise. Same spirit as the code index's skip set.
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

# Cap how many files we scan to resolve a bare symbol name to its
# definition site. A monorepo with thousands of files would make the
# resolve-by-name scan slow; the cap keeps the tool snappy. The agent
# can pass a more specific dotted name or a path hint if a symbol is
# missed in a huge tree.
_MAX_SCAN_FILES = 400

# Cap references returned — a heavily-used symbol could have hundreds of
# call sites; the agent wants the map, not the phone book.
_MAX_REFS = 40


def _iter_py_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        out.append(p)
        if len(out) >= _MAX_SCAN_FILES:
            break
    return out


def _find_definition_site(
    root: Path, symbol: str
) -> tuple[Any, Path, int, int] | None:
    """Resolve a bare/dotted ``symbol`` to its definition location by
    scanning project files with jedi's name index.

    Returns ``(jedi.Script, path, line, column)`` at the definition, or
    None if not found. We match the LAST dotted segment against jedi's
    definition names (so ``AuthManager.login`` matches a ``login``
    method), preferring an exact qualified match when available.
    """
    import jedi  # local import — optional dep, surfaced as a tool message

    bare = symbol.rsplit(".", 1)[-1]
    fallback: tuple[Any, Path, int, int] | None = None
    for fpath in _iter_py_files(root):
        try:
            source = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            script = jedi.Script(code=source, path=str(fpath))
            names = script.get_names(all_scopes=True, definitions=True)
        except Exception:
            continue
        for n in names:
            if n.name != bare:
                continue
            # Exact match on the qualified name wins immediately; a
            # bare-name match is held as a fallback (first one found).
            qual = _qualified(n)
            if qual == symbol or n.name == symbol:
                return script, fpath, n.line, n.column
            if fallback is None:
                fallback = (script, fpath, n.line, n.column)
    return fallback


def _qualified(name: Any) -> str:
    """Best-effort dotted name for a jedi Name (full_name when jedi
    resolved it, else the bare name)."""
    full = getattr(name, "full_name", None)
    return str(full) if full else str(getattr(name, "name", ""))


def _rel(root: Path, p: Path | str | None) -> str:
    if p is None:
        return "<unknown>"
    try:
        return Path(p).resolve().relative_to(root).as_posix()
    except (ValueError, OSError):
        return str(p)


# ---------------------------------------------------------------------------
# go_to_definition
# ---------------------------------------------------------------------------


def go_to_definition_tool(workdir: Path | str) -> Tool:
    """Build the ``go_to_definition`` tool for ``workdir``.

    Model-facing: ``go_to_definition(symbol)`` — ``symbol`` is a
    function/class/method name (bare ``login`` or dotted
    ``AuthManager.login``). Returns the definition's file:line +
    signature so the agent can ``read`` it directly instead of
    grepping. Python-only; falls back to a grep suggestion when jedi
    is unavailable or the symbol can't be resolved.
    """
    root = Path(workdir).resolve()

    async def go_to_definition(symbol: str) -> str:
        symbol = str(symbol).strip()
        if not symbol:
            return "go_to_definition: empty symbol"
        try:
            import jedi  # noqa: F401
        except ImportError:
            return (
                "go_to_definition unavailable (jedi not installed). "
                "Use grep to find the definition."
            )
        try:
            found = _find_definition_site(root, symbol)
            if found is None:
                return (
                    f"go_to_definition: no Python definition found for "
                    f"'{symbol}'. Try grep, or a more specific name."
                )
            script, path, line, col = found
            defs = script.goto(
                line=line,
                column=col,
                follow_imports=True,
                follow_builtin_imports=False,
            )
            if not defs:
                # The name-index hit IS the definition site.
                sig = _line_text(path, line)
                return (
                    f"{_rel(root, path)}:{line}  {sig}"
                )
            out: list[str] = []
            for d in defs:
                dpath = d.module_path
                dline = d.line or line
                sig = (d.description or "").strip()
                loc = f"{_rel(root, dpath)}:{dline}"
                out.append(f"  {d.type} {d.name}  ({loc})  {sig}")
            return "\n".join(out)
        except Exception as exc:  # never abort a turn on a nav failure
            return (
                f"go_to_definition unavailable "
                f"({type(exc).__name__}: {exc}). Use grep."
            )

    return tool(
        name="go_to_definition",
        description=(
            "Jump to where a Python symbol is DEFINED — resolves "
            "through imports + scope like an IDE's Go-to-Definition, "
            "unlike grep which only matches the string. Args: symbol "
            "(function/class/method name, bare 'login' or dotted "
            "'AuthManager.login'). Returns the definition's file:line "
            "+ signature to read next. Python only; use grep for "
            "other languages."
        ),
    )(go_to_definition)


# ---------------------------------------------------------------------------
# find_references
# ---------------------------------------------------------------------------


def find_references_tool(workdir: Path | str) -> Tool:
    """Build the ``find_references`` tool for ``workdir``.

    Model-facing: ``find_references(symbol)`` — every place ``symbol``
    is used across the project (file:line list), scope-aware via jedi.
    The answer to "what breaks if I change this?" — far more precise
    than grepping the bare name (which matches comments, strings, and
    unrelated same-named locals).
    """
    root = Path(workdir).resolve()

    async def find_references(symbol: str) -> str:
        symbol = str(symbol).strip()
        if not symbol:
            return "find_references: empty symbol"
        try:
            import jedi  # noqa: F401
        except ImportError:
            return (
                "find_references unavailable (jedi not installed). "
                "Use grep to find usages."
            )
        try:
            found = _find_definition_site(root, symbol)
            if found is None:
                return (
                    f"find_references: no Python definition found for "
                    f"'{symbol}'. Try grep."
                )
            script, path, line, col = found
            refs = script.get_references(
                line=line, column=col, scope="project"
            )
            if not refs:
                return f"find_references: no usages found for '{symbol}'"
            seen: set[tuple[str, int]] = set()
            lines: list[str] = []
            for r in refs:
                rel = _rel(root, r.module_path)
                key = (rel, r.line or 0)
                if key in seen:
                    continue
                seen.add(key)
                text = _line_text(
                    Path(r.module_path) if r.module_path else path,
                    r.line or 0,
                )
                lines.append(f"  {rel}:{r.line}  {text}")
                if len(lines) >= _MAX_REFS:
                    lines.append(
                        f"  … (+more; showing first {_MAX_REFS})"
                    )
                    break
            return "\n".join(lines)
        except Exception as exc:
            return (
                f"find_references unavailable "
                f"({type(exc).__name__}: {exc}). Use grep."
            )

    return tool(
        name="find_references",
        description=(
            "Find every place a Python symbol is USED across the "
            "project — scope-aware (resolves through imports, ignores "
            "same-named unrelated locals/strings/comments that grep "
            "would falsely match). Args: symbol (function/class/method "
            "name). Returns a file:line list of real usages — the "
            "answer to 'what breaks if I change this?'. Python only."
        ),
    )(find_references)


# ---------------------------------------------------------------------------
# hover
# ---------------------------------------------------------------------------


def hover_tool(workdir: Path | str) -> Tool:
    """Build the ``hover`` tool for ``workdir``.

    Model-facing: ``hover(symbol)`` — the symbol's signature +
    docstring (the IDE "what is this" tooltip), so the agent learns a
    function's contract without reading its whole file.
    """
    root = Path(workdir).resolve()

    async def hover(symbol: str) -> str:
        symbol = str(symbol).strip()
        if not symbol:
            return "hover: empty symbol"
        try:
            import jedi  # noqa: F401
        except ImportError:
            return (
                "hover unavailable (jedi not installed). "
                "Use read to inspect the symbol."
            )
        try:
            found = _find_definition_site(root, symbol)
            if found is None:
                return (
                    f"hover: no Python definition found for '{symbol}'. "
                    "Use grep / read."
                )
            script, path, line, col = found
            helps = script.help(line=line, column=col)
            if not helps:
                return f"hover: nothing to show for '{symbol}'"
            parts: list[str] = []
            for h in helps:
                loc = f"{_rel(root, path)}:{line}"
                header = f"{h.type} {h.name}  ({loc})"
                sigs = []
                try:
                    sigs = [s.to_string() for s in h.get_signatures()]
                except Exception:
                    sigs = []
                doc = ""
                try:
                    doc = (h.docstring() or "").strip()
                except Exception:
                    doc = ""
                block = header
                if sigs:
                    block += "\n  " + sigs[0]
                if doc:
                    # jedi's docstring() prepends the call signature as
                    # the FIRST paragraph, then the real docstring. Drop
                    # that leading signature paragraph so we show the
                    # actual contract, not "verify_token(token)" twice.
                    paras = doc.split("\n\n", 1)
                    body = (
                        paras[1] if len(paras) == 2 else paras[0]
                    ).strip()
                    if body:
                        block += "\n  " + body.replace("\n", "\n  ")
                parts.append(block)
            return "\n\n".join(parts)
        except Exception as exc:
            return (
                f"hover unavailable ({type(exc).__name__}: {exc}). "
                "Use read."
            )

    return tool(
        name="hover",
        description=(
            "Show a Python symbol's signature + docstring (the IDE "
            "'what is this' tooltip) — learn a function's contract "
            "without reading its whole file. Args: symbol "
            "(function/class/method name). Python only; use read for "
            "other languages or full bodies."
        ),
    )(hover)


def _line_text(path: Path | str, line: int) -> str:
    """The stripped source text at ``path:line`` — a one-line preview
    so reference/definition lists show what's there. Best-effort."""
    if line <= 0:
        return ""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()
    if 0 < line <= len(lines):
        return lines[line - 1].strip()[:100]
    return ""


def lsp_tools(workdir: Path | str) -> list[Tool]:
    """All three navigation tools for ``workdir`` — the convenience
    bundle the agent builder wires in one call (mirrors how the file
    tools are grouped)."""
    return [
        go_to_definition_tool(workdir),
        find_references_tool(workdir),
        hover_tool(workdir),
    ]
