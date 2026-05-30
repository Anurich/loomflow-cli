"""Tests for LSP navigation tools (loom_code.lsp_tools).

Exercises go_to_definition / find_references / hover against a real
two-file temp project using the actual jedi backend. Offline, no model.
These verify the IDE-precision win over grep: cross-file resolution,
real usage lists, and signature+docstring hover.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from loom_code.lsp_tools import lsp_tools

pytestmark = pytest.mark.anyio


async def _invoke(tool_obj: object, **kwargs: object) -> str:
    for attr in ("func", "_func", "fn", "_fn", "callable", "handler"):
        fn = getattr(tool_obj, attr, None)
        if fn and callable(fn):
            res = fn(**kwargs)
            return await res if inspect.iscoroutine(res) else res  # type: ignore[no-any-return]
    res = tool_obj(**kwargs)  # type: ignore[operator]
    return await res if inspect.iscoroutine(res) else res  # type: ignore[no-any-return]


def _project(tmp_path: Path) -> dict[str, object]:
    (tmp_path / "auth.py").write_text(
        'def verify_token(token):\n'
        '    """Validate a JWT and return the user id."""\n'
        '    return decode(token)\n'
    )
    (tmp_path / "app.py").write_text(
        "from auth import verify_token\n"
        "\n"
        "def handler(req):\n"
        "    return verify_token(req.token)\n"
        "\n"
        "def other():\n"
        '    return verify_token("x")\n'
    )
    return {t.name: t for t in lsp_tools(tmp_path)}


def test_tools_present(tmp_path: Path) -> None:
    by = _project(tmp_path)
    assert set(by) == {"go_to_definition", "find_references", "hover"}


async def test_go_to_definition_resolves_cross_file(tmp_path: Path) -> None:
    by = _project(tmp_path)
    out = await _invoke(by["go_to_definition"], symbol="verify_token")
    # Defined in auth.py, even though queried from a project that uses
    # it in app.py — resolution, not string match.
    assert "auth.py" in out
    assert "verify_token" in out


async def test_find_references_finds_all_usages(tmp_path: Path) -> None:
    by = _project(tmp_path)
    out = await _invoke(by["find_references"], symbol="verify_token")
    # Both call sites in app.py + the import line — the grep-beating
    # precision: real usages, scope-aware.
    assert out.count("app.py") >= 2


async def test_hover_shows_signature_and_docstring(tmp_path: Path) -> None:
    by = _project(tmp_path)
    out = await _invoke(by["hover"], symbol="verify_token")
    assert "verify_token" in out
    # The docstring is the contract — must surface, not just the sig.
    assert "Validate a JWT" in out


async def test_unknown_symbol_is_graceful(tmp_path: Path) -> None:
    by = _project(tmp_path)
    out = await _invoke(by["go_to_definition"], symbol="no_such_symbol_xyz")
    assert "no python definition" in out.lower()


async def test_empty_symbol_is_graceful(tmp_path: Path) -> None:
    by = _project(tmp_path)
    out = await _invoke(by["hover"], symbol="")
    assert "empty" in out.lower()


async def test_dotted_method_name(tmp_path: Path) -> None:
    (tmp_path / "svc.py").write_text(
        "class Service:\n"
        "    def start(self):\n"
        '        """Boot the service."""\n'
        "        return True\n"
    )
    by = {t.name: t for t in lsp_tools(tmp_path)}
    out = await _invoke(by["go_to_definition"], symbol="Service.start")
    assert "svc.py" in out
