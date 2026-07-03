"""Shared per-turn pipeline pieces — ONE home for the learning loop.

loom-code has two frontends that each run their own turn loop: the
terminal REPL (``repl.py``) and loomflow-desktop's sidecar. Per-turn
behaviour written inside either frontend exists only there and the
copies drift — the desktop shipped with active recall present but the
credit signal dormant, exactly that failure. Anything that should
happen "around every turn" regardless of surface belongs HERE; the
frontends keep only their surface-specific concerns (console output,
RPC events, pending-state storage).

Current residents:

* :func:`learned_notes_block` / :func:`inject_learned_notes` — ACTIVE
  recall: surface the top success-credited notebook notes relevant to
  this prompt as the ``learned_notes`` working block.
* :func:`attribute_turn` — the credit signal: flush a finished turn's
  cited note slugs (+ file touches) to the workspace / file history.
  Frontends decide *when* (explicit /good–/bad, or the moved-on
  heuristic: a new task arriving without complaint credits the last
  turn); this owns *what crediting means*.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import file_history

# Bounded injection: top-3 proven notes, snippet-length excerpts —
# a few hundred tokens, not a notebook dump.
_RECALL_LIMIT = 8
_INJECT_TOP = 3


async def learned_notes_block(
    workspace: Any, prompt: str, *, user_id: str | None
) -> str:
    """Render the ``learned_notes`` working-block body for ``prompt``.

    Empty string when no PROVEN note matches — callers must still
    write the empty block so stale advice from a prior prompt never
    lingers. Slugs are shown so the agent can ``read_note(slug)`` for
    full detail, which also keeps the citation-credit chain alive.
    """
    matches = await workspace.search_notes(
        prompt,
        user_id=user_id,
        boost_relevance=True,
        limit=_RECALL_LIMIT,
    )
    proven = [m for m in matches if m.summary.success_count > 0][
        :_INJECT_TOP
    ]
    if not proven:
        return ""
    lines = [
        "# Learned notes (proven on past turns)",
        "Notes from earlier work on THIS project that were used in "
        "turns the user accepted. Trust but verify — "
        "`read_note(slug)` for the full note.",
        "",
    ]
    lines.extend(
        f"- [{m.summary.slug}] (worked "
        f"{m.summary.success_count}x) {m.snippet}"
        for m in proven
    )
    return "\n".join(lines)


async def update_block_if_changed(
    memory: Any,
    name: str,
    content: str,
    *,
    user_id: str | None,
    block_hashes: dict[str, str] | None,
) -> None:
    """Write a working block only when its content changed since the
    last write this session — the one dirty-check both the REPL and
    :func:`inject_learned_notes` share.

    ``block_hashes`` (caller-owned) is the per-session content-hash
    map; ``None`` disables the check (always writes). The hash is
    recorded only AFTER a successful write, so a failed write can't
    convince a later turn the block is up to date. Rationale: skips a
    redundant sqlite write + the churn of re-serialising an unchanged
    block every turn (loomflow's update_block is a plain UPSERT)."""
    if block_hashes is None:
        await memory.update_block(name, content, user_id=user_id)
        return
    import hashlib

    h = hashlib.sha1(content.encode("utf-8")).hexdigest()
    if block_hashes.get(name) == h:
        return
    await memory.update_block(name, content, user_id=user_id)
    block_hashes[name] = h


async def inject_learned_notes(
    workspace: Any,
    memory: Any,
    prompt: str,
    *,
    user_id: str | None,
    block_hashes: dict[str, str] | None = None,
) -> None:
    """Write this turn's active-recall block into ``memory``.

    Best-effort by contract: memory/workspace I/O failing must never
    kill a turn, so callers can ``await`` this bare. ``block_hashes``
    (optional) enables the shared dirty-check — see
    :func:`update_block_if_changed`."""
    try:
        body = await learned_notes_block(
            workspace, prompt, user_id=user_id
        )
        await update_block_if_changed(
            memory,
            "learned_notes",
            body,
            user_id=user_id,
            block_hashes=block_hashes,
        )
    except Exception:  # noqa: BLE001 — recall is best-effort
        pass


async def attribute_turn(
    workspace: Any,
    root: Path | str,
    *,
    success: bool,
    slugs: list[str],
    files: list[str],
    user_id: str | None,
) -> int:
    """Flush one finished turn's learning signal.

    * ``files`` — paths the turn wrote; their file-history records
      are revised from "unknown" to the now-known outcome
      (independent of the slug path: a turn can edit files without
      citing notes).
    * ``slugs`` — notes the agent read during the turn (the run
      result's ``cited_slugs``); each gets ``cited_count`` += 1 and,
      when ``success``, ``success_count`` += 1 — which is what makes
      it eligible for future active recall.

    Returns the number of notes credited/debited (0 on failure —
    best-effort by contract)."""
    if files:
        try:
            file_history.update_last_outcome(
                root, files, "success" if success else "fail"
            )
        except Exception:  # noqa: BLE001 — history is best-effort
            pass
    if not slugs:
        return 0
    try:
        n = await workspace.attribute_outcome(
            success=success, slugs=slugs, user_id=user_id
        )
        return int(n or 0)
    except Exception:  # noqa: BLE001 — crediting is best-effort
        return 0
