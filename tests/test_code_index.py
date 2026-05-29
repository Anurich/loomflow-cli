"""Tests for the semantic code index (loom_code.code_index).

Zero-key: uses the ``hash`` embedder so the whole suite runs offline.
The hash embedder's semantic quality is weak (that's the point — it's
the no-API-key fallback), so assertions check PLUMBING (chunking,
incremental re-index, the code+note blend) and the *presence* of the
obvious hit, never its exact rank.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from loom_code.code_index import (
    CodeIndexStore,
    _chunks_for_file,
    build_index,
    codebase_search_tool,
    resolve_embedder,
)

pytestmark = pytest.mark.anyio


# A tool() wraps an async closure; find + call it regardless of the
# Tool object's internal attribute name.
async def _invoke(tool_obj: object, **kwargs: object) -> str:
    for attr in ("func", "_func", "fn", "_fn", "callable", "handler"):
        fn = getattr(tool_obj, attr, None)
        if fn and callable(fn):
            res = fn(**kwargs)
            return await res if inspect.iscoroutine(res) else res  # type: ignore[no-any-return]
    res = tool_obj(**kwargs)  # type: ignore[operator]
    return await res if inspect.iscoroutine(res) else res  # type: ignore[no-any-return]


_SRC = '''
class Auth:
    """Handles login."""
    def verify_token(self, t):
        """Validate a JWT and return the user."""
        return decode(t)

def helper():
    pass

CONST = 42
'''


def test_chunking_extracts_symbols_and_skips_constants() -> None:
    chunks = _chunks_for_file("auth.py", _SRC)
    names = {c.qualified_name for c in chunks}
    assert "Auth" in names
    assert "Auth.verify_token" in names
    assert "helper" in names
    # Module constants carry no behaviour to search for.
    assert "CONST" not in names


def test_syntax_error_yields_no_chunks() -> None:
    # A broken file must not abort indexing — the AST walk swallows it.
    assert _chunks_for_file("broken.py", "def (:\n  pass") == []


async def test_build_is_incremental(tmp_path: Path) -> None:
    (tmp_path / ".loom").mkdir()
    (tmp_path / "a.py").write_text(_SRC)
    (tmp_path / "b.py").write_text("def f():\n    pass\n")
    emb = resolve_embedder("hash")
    store = CodeIndexStore(tmp_path / ".loom" / "code_index.db", "hash")

    embedded, skipped = await build_index(tmp_path, store, emb)
    assert (embedded, skipped) == (2, 0)

    # Unchanged tree → everything skips (the sha gate).
    embedded, skipped = await build_index(tmp_path, store, emb)
    assert (embedded, skipped) == (0, 2)

    # Touch one file → only it re-embeds.
    (tmp_path / "a.py").write_text(_SRC + "\ndef g():\n    pass\n")
    embedded, skipped = await build_index(tmp_path, store, emb)
    assert (embedded, skipped) == (1, 1)
    store.close()


async def test_deleted_file_is_pruned(tmp_path: Path) -> None:
    (tmp_path / ".loom").mkdir()
    (tmp_path / "gone.py").write_text("def doomed():\n    pass\n")
    emb = resolve_embedder("hash")
    store = CodeIndexStore(tmp_path / ".loom" / "code_index.db", "hash")
    await build_index(tmp_path, store, emb)
    assert not store.is_empty()

    (tmp_path / "gone.py").unlink()
    await build_index(tmp_path, store, emb)
    assert store.is_empty()
    store.close()


async def test_search_finds_semantic_hit(tmp_path: Path) -> None:
    (tmp_path / ".loom").mkdir()
    (tmp_path / "pay.py").write_text(
        "def charge_card(amount):\n"
        "    '''Bill the customer credit card via Stripe.'''\n"
        "    return stripe.charge(amount)\n"
    )
    tool_obj = codebase_search_tool(tmp_path, "hash")
    out = await _invoke(tool_obj, query="credit card billing", limit=5)
    assert "charge_card" in out


async def test_empty_repo_is_graceful(tmp_path: Path) -> None:
    (tmp_path / ".loom").mkdir()
    tool_obj = codebase_search_tool(tmp_path, "hash")
    out = await _invoke(tool_obj, query="anything")
    assert "empty" in out.lower()


async def test_blend_fuses_code_and_learned_notes(tmp_path: Path) -> None:
    """The differentiator: one query returns BOTH the code that does X
    and what we learned about it."""
    from loomflow.memory import HashEmbedder
    from loomflow.workspace import LocalDiskWorkspace

    (tmp_path / ".loom").mkdir()
    (tmp_path / "pay.py").write_text(
        "def charge_card(amount):\n"
        "    '''Bill the customer credit card via Stripe.'''\n"
        "    return stripe.charge(amount)\n"
    )
    ws = LocalDiskWorkspace(
        str(tmp_path / ".loom" / "notebook"), embedder=HashEmbedder()
    )
    # write_note takes keyword fields directly (no wrapper type); the
    # body field is ``body`` not ``content``. user_id is None to match
    # the tenant codebase_search queries under: outside an agent run
    # there's no RunContext, so _current_user_id() resolves to None.
    # Writing under a different user_id would (correctly) hide the note
    # behind the multi-tenant partition — exactly what the production
    # path enforces.
    await ws.write_note(
        author="auditor",
        title="Stripe retries are NOT idempotent here",
        body=(
            "charge_card double-charged in prod when the network flaked "
            "— there is no idempotency key. Add one before retry logic."
        ),
        kind="finding",
        user_id=None,
    )

    # Without the workspace: code only.
    code_only = await _invoke(
        codebase_search_tool(tmp_path, "hash"),
        query="credit card billing retry safety",
    )
    assert "charge_card" in code_only
    assert "learned" not in code_only

    # With the workspace: code + the learned note, fused.
    blended = await _invoke(
        codebase_search_tool(tmp_path, "hash", workspace=ws),
        query="credit card billing retry safety",
    )
    assert "charge_card" in blended
    assert "learned" in blended
    assert "idempot" in blended.lower()


async def test_workspace_failure_degrades_to_code_only(tmp_path: Path) -> None:
    """A notebook outage must not break code search — it falls back."""

    class _BoomWorkspace:
        async def search_notes(self, *a: object, **k: object) -> list[object]:
            raise RuntimeError("notebook down")

    (tmp_path / ".loom").mkdir()
    (tmp_path / "pay.py").write_text(
        "def charge_card(amount):\n    return 1\n"
    )
    out = await _invoke(
        codebase_search_tool(tmp_path, "hash", workspace=_BoomWorkspace()),
        query="billing",
    )
    # Still returns the code hit despite the workspace raising.
    assert "charge_card" in out
