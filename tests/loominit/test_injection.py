"""Tests for the per-turn LOOM.md BM25 retriever (loominit slice 3).

The retriever's job is simple: parse ``LOOM.md`` into ``##``-bounded
sections, build a BM25 index, and on demand return the top-N
sections most relevant to a query — with hard caps on character
budget so the rendered block stays within token-budget.

Coverage in this file:

* ``parse_sections`` splits on ``##`` boundaries and skips empty
  sections + the empty intro case.
* ``LoomRetriever.relevant`` picks the section whose tokens overlap
  the query and ignores sections with zero overlap.
* Top-N ranking respects the requested ``top_n``.
* ``max_chars`` caps the rendered block size.
* No-overlap query returns empty string (not "use everything as a
  fallback" — that would defeat the purpose of retrieval).
* ``from_repo_root`` returns ``None`` when ``LOOM.md`` is missing
  or empty.
* Wired correctly with loomflow's ``Memory.update_block`` (smoke
  via an :class:`InMemoryMemory` round-trip).
"""

from __future__ import annotations

import pytest

from loom_code.loominit.injection import (
    LOOM_BLOCK_NAME,
    LoomRetriever,
    parse_sections,
)
from loom_code.loominit.persistence import markdown_path

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# parse_sections
# ---------------------------------------------------------------------------


def test_parse_sections_splits_on_level_2_headings() -> None:
    md = (
        "# Top-level title\n"
        "Intro paragraph.\n"
        "\n"
        "## Workspace\n"
        "Notes about the workspace subsystem.\n"
        "\n"
        "## Memory tiers\n"
        "Three tiers: working, episodes, facts.\n"
    )
    secs = parse_sections(md)
    headings = [s.heading for s in secs]
    assert headings == ["(intro)", "Workspace", "Memory tiers"]
    assert "Three tiers" in secs[2].body


def test_parse_sections_skips_empty_intro() -> None:
    """When the doc starts with a heading, no spurious intro
    section is created."""
    md = "## Only section\nHello world.\n"
    secs = parse_sections(md)
    assert [s.heading for s in secs] == ["Only section"]


def test_parse_sections_ignores_level_3_headings_as_splitters() -> None:
    """``###`` and below stay INSIDE their parent ``##`` section."""
    md = (
        "## Parent\n"
        "parent body\n"
        "### Subsection\n"
        "sub body\n"
    )
    secs = parse_sections(md)
    assert len(secs) == 1
    assert "Subsection" in secs[0].body  # carried into parent's body


def test_parse_sections_skips_heading_only_sections() -> None:
    """An empty section (heading with no body before the next
    heading) is dropped — BM25 has nothing to score against."""
    md = "## Empty\n## Real\nReal content here.\n"
    secs = parse_sections(md)
    assert [s.heading for s in secs] == ["Real"]


# ---------------------------------------------------------------------------
# Retrieval ranking
# ---------------------------------------------------------------------------


def _three_section_doc() -> str:
    """Fixture covering three distinct subsystems with non-overlapping
    vocab so BM25 ranks them unambiguously."""
    return (
        "## Workspace internals\n"
        "The shared notebook lets agents call note, list_notes, "
        "search_notes, read_note, update_note. Slugs are auto-"
        "generated from titles. Atomic writes. Multi-tenant.\n"
        "\n"
        "## Memory tiers\n"
        "Three tiers: working blocks (pinned strings), episodes "
        "(one I/O exchange per episode, hybrid recall), facts "
        "(bi-temporal triples with supersession).\n"
        "\n"
        "## Permission decision flow\n"
        "Pre-tool hook, permissions policy check, ask resolution "
        "via approval_handler, execute via tools.call, post-tool "
        "hook. Modes: default, acceptEdits, bypassPermissions.\n"
    )


def test_relevant_picks_the_matching_section() -> None:
    retr = LoomRetriever(parse_sections(_three_section_doc()))
    out = retr.relevant("how does the shared notebook work")
    assert "Workspace internals" in out
    # Other sections shouldn't sneak in just because top_n=3 — only
    # one section has token overlap with this query.
    assert "Memory tiers" not in out
    assert "Permission decision flow" not in out


def test_relevant_ranks_by_overlap() -> None:
    retr = LoomRetriever(
        parse_sections(_three_section_doc()), top_n=2
    )
    out = retr.relevant("memory facts episodes")
    # ``memory`` + ``facts`` + ``episodes`` are concentrated in the
    # memory section — it should be the top hit.
    assert out.index("Memory tiers") < (
        out.index("Workspace internals")
        if "Workspace internals" in out
        else len(out)
    )


def test_relevant_empty_string_when_no_overlap() -> None:
    """A query with zero token overlap returns "" — the fallback
    "include everything anyway" would defeat retrieval's purpose."""
    retr = LoomRetriever(parse_sections(_three_section_doc()))
    out = retr.relevant("xyzzy frobnicate quux")
    assert out == ""


def test_relevant_respects_top_n() -> None:
    retr = LoomRetriever(
        parse_sections(_three_section_doc()), top_n=1
    )
    out = retr.relevant("memory permission notebook")
    # All three sections would match, but top_n=1 caps at one.
    # Each rendered section is prefixed with "\n## " (the leading
    # "\n" separates it from the "# Relevant ..." header or from a
    # prior section).
    assert out.count("\n## ") == 1


def test_relevant_respects_max_chars_budget() -> None:
    """When ``max_chars`` is tight, fewer sections are returned —
    but at least one if any matches at all."""
    long_doc = (
        "## A\n"
        + ("alpha " * 200)
        + "\n## B\n"
        + ("alpha " * 200)
        + "\n## C\n"
        + ("alpha " * 200)
    )
    retr = LoomRetriever(
        parse_sections(long_doc), top_n=3, max_chars=500
    )
    out = retr.relevant("alpha")
    # First section fits; later ones don't. ``A`` should be present.
    assert "## A" in out
    # Either one or two sections — depends on exact chars. Not all
    # three.
    assert out.count("\n## ") <= 1


# ---------------------------------------------------------------------------
# from_repo_root + working-block plumbing
# ---------------------------------------------------------------------------


def test_from_repo_root_returns_none_when_missing(tmp_path) -> None:
    # No LOOM.md at all.
    assert LoomRetriever.from_repo_root(tmp_path) is None


def test_from_repo_root_returns_none_when_empty(tmp_path) -> None:
    markdown_path(tmp_path).write_text("", encoding="utf-8")
    assert LoomRetriever.from_repo_root(tmp_path) is None


def test_from_repo_root_loads_real_file(tmp_path) -> None:
    markdown_path(tmp_path).write_text(
        _three_section_doc(), encoding="utf-8"
    )
    retr = LoomRetriever.from_repo_root(tmp_path)
    assert retr is not None
    assert retr.is_loaded is True
    assert retr.section_count == 3


async def test_retrieved_block_round_trips_through_loomflow_memory(
    tmp_path,
) -> None:
    """End-to-end smoke: the rendered block stays intact when
    written via loomflow's ``Memory.update_block`` and read back via
    ``working()`` — confirms the wiring shape the REPL relies on."""
    from loomflow import InMemoryMemory

    markdown_path(tmp_path).write_text(
        _three_section_doc(), encoding="utf-8"
    )
    retr = LoomRetriever.from_repo_root(tmp_path)
    assert retr is not None

    body = retr.relevant("notebook list_notes search_notes")
    assert body  # non-empty — must have matched the workspace section

    mem = InMemoryMemory()
    await mem.update_block(LOOM_BLOCK_NAME, body, user_id="alice")
    blocks = await mem.working(user_id="alice")
    names = {b.name for b in blocks}
    assert LOOM_BLOCK_NAME in names
