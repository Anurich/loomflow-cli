"""Tests for the agentic LOOM.md retrieval mode.

Pins the three contracts this slice ships:

1. ``parse_sections`` emits stable kebab-case slugs (with collision
   numbering) so the agent can address sections by name.
2. ``LoomRetriever(mode='agentic')`` returns a STABLE TOC every
   turn instead of per-turn keyword-scored bodies — the property
   that unblocks prompt caching.
3. ``read_loom_section_tool`` fetches one section by slug and
   returns a helpful error (listing available slugs) on unknown
   slug.
4. ``build_agent(loom_retrieval='agentic')`` wires the tool into
   the coordinator + simple coder and stamps the mode on the
   coordinator so the REPL's LoomRetriever can pick it up.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from loom_code.loominit.injection import (
    LoomRetriever,
    parse_sections,
)

_FAKE_LOOM_MD = """\
# myproj — LOOM.md

## Overview
The project does X and Y.

## Workspace Internals
Workspace lives at .loom/notebook.
It stores notes.

## Workspace Internals
Duplicate heading on purpose — should get a -2 slug suffix.

## Tech Stack
Python 3.12 + loomflow.
"""


def test_parse_sections_assigns_stable_slugs() -> None:
    secs = parse_sections(_FAKE_LOOM_MD)
    slugs = [s.slug for s in secs]
    # Each heading gets a kebab-case slug. Collision dedup with -2.
    assert "overview" in slugs
    assert "workspace-internals" in slugs
    assert "workspace-internals-2" in slugs
    assert "tech-stack" in slugs


def test_parse_sections_intro_section_has_slug() -> None:
    """Content before the first ## becomes the (intro) section;
    it still gets a slug so it's addressable."""
    md = "Leading content with no heading yet.\n\n## First\nbody"
    secs = parse_sections(md)
    assert secs[0].slug != ""
    # Intro heading is "(intro)" — slugify collapses parens.
    assert secs[0].slug == "intro"


def test_retriever_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        LoomRetriever(parse_sections(_FAKE_LOOM_MD), mode="vector")


def test_agentic_mode_returns_stable_toc(tmp_path: Path) -> None:
    """In agentic mode, ``relevant(query)`` returns the TOC and
    is INDEPENDENT of the query. Same input → same output across
    different queries → cacheable system prompt."""
    secs = parse_sections(_FAKE_LOOM_MD)
    retriever = LoomRetriever(secs, mode="agentic")

    toc_a = retriever.relevant("how does login work?")
    toc_b = retriever.relevant("what is the tech stack?")
    toc_c = retriever.relevant("unrelated nonsense xyz")

    # All three must be identical — that's the prefix-stability
    # contract that unblocks prompt caching.
    assert toc_a == toc_b == toc_c

    # And it must actually contain the slug map the agent uses.
    assert "section map" in toc_a.lower()
    assert "[overview] Overview" in toc_a
    assert "[workspace-internals] Workspace Internals" in toc_a
    assert "[workspace-internals-2] Workspace Internals" in toc_a


def test_bm25_mode_still_changes_with_query() -> None:
    """Inverse pin: default ``bm25`` mode IS per-turn (different
    queries → different ranked outputs). Without this we couldn't
    detect a regression where shared accidentally takes over."""
    secs = parse_sections(_FAKE_LOOM_MD)
    retriever = LoomRetriever(secs, mode="bm25")

    out_tech = retriever.relevant("python loomflow tech")
    out_workspace = retriever.relevant("notebook workspace notes")
    # They should pick different top sections.
    assert out_tech != out_workspace


def test_section_body_returns_match_or_none() -> None:
    secs = parse_sections(_FAKE_LOOM_MD)
    retriever = LoomRetriever(secs, mode="agentic")
    body = retriever.section_body("tech-stack")
    assert body is not None
    assert "Python 3.12" in body
    assert retriever.section_body("nonexistent") is None


def test_available_slugs_returns_all_in_source_order() -> None:
    secs = parse_sections(_FAKE_LOOM_MD)
    retriever = LoomRetriever(secs, mode="agentic")
    slugs = retriever.available_slugs()
    # Source order, including the leading "(intro)" pseudo-
    # section (content before the first ##) and the dedup'd
    # duplicate-heading suffix.
    assert slugs == [
        "intro",
        "overview",
        "workspace-internals",
        "workspace-internals-2",
        "tech-stack",
    ]


def test_read_loom_section_tool_happy_path(tmp_path: Path) -> None:
    """Tool fetches a body by slug. Reads from disk on call so it
    stays in sync with /loominit refresh without retriever
    re-construction."""
    from loom_code.loom_section_tool import read_loom_section_tool

    (tmp_path / "LOOM.md").write_text(_FAKE_LOOM_MD)
    tool = read_loom_section_tool(tmp_path)
    result = asyncio.run(tool.fn(slug="tech-stack"))
    assert "Python 3.12" in result
    assert "## tech-stack" in result


def test_read_loom_section_tool_unknown_slug_lists_available(
    tmp_path: Path,
) -> None:
    """Wrong slug → error message lists available slugs so the
    agent can self-correct without an extra round-trip."""
    from loom_code.loom_section_tool import read_loom_section_tool

    (tmp_path / "LOOM.md").write_text(_FAKE_LOOM_MD)
    tool = read_loom_section_tool(tmp_path)
    result = asyncio.run(tool.fn(slug="does-not-exist"))
    assert "no section with slug" in result.lower()
    assert "tech-stack" in result  # at least one available slug listed


def test_read_loom_section_tool_no_loom_md(tmp_path: Path) -> None:
    """Tool surfaces a clear 'run /loominit' message when there's
    no LOOM.md at all."""
    from loom_code.loom_section_tool import read_loom_section_tool

    tool = read_loom_section_tool(tmp_path)
    result = asyncio.run(tool.fn(slug="anything"))
    assert "LOOM.md not found" in result
    assert "/loominit" in result


def test_build_agent_agentic_mode_stamps_coordinator(
    project,
) -> None:
    """``build_agent(loom_retrieval='agentic')`` must stamp the
    mode on the coordinator so the REPL's LoomRetriever build
    picks it up. Pins the cross-module wiring."""
    from loom_code.agent import build_agent

    coord, _ = build_agent(
        project, model="echo", loom_retrieval="agentic"
    )
    assert getattr(coord, "_loom_retrieval_mode", None) == "agentic"


def test_build_agent_default_mode_is_agentic(project) -> None:
    """Pin the default: agentic is the supported retrieval path.
    Reason: it both (a) lets the LLM pick relevant sections by
    intent instead of keyword overlap, AND (b) produces a stable
    system-prompt prefix so OpenAI's prompt cache actually hits
    (per-turn BM25 injection invalidates the prefix every turn,
    which we measured as $0.16+/turn on a real REPL session).
    BM25 is still reachable for callers that explicitly opt out
    via ``loom_retrieval='bm25'``."""
    from loom_code.agent import build_agent

    coord, _ = build_agent(project, model="echo")
    assert getattr(coord, "_loom_retrieval_mode", None) == "agentic"


def test_build_agent_bm25_mode_explicit_opt_out(project) -> None:
    """The opt-out path: ``loom_retrieval='bm25'`` still works
    for callers who explicitly want the per-turn keyword-scored
    section dump (testing, A/B comparisons, etc.)."""
    from loom_code.agent import build_agent

    coord, _ = build_agent(
        project, model="echo", loom_retrieval="bm25"
    )
    assert getattr(coord, "_loom_retrieval_mode", None) == "bm25"


def test_build_agent_rejects_invalid_mode(project) -> None:
    from loom_code.agent import build_agent

    with pytest.raises(ValueError, match="loom_retrieval"):
        build_agent(
            project, model="echo", loom_retrieval="vector"
        )


def test_build_agent_agentic_adds_tool_to_both_routes(
    project,
) -> None:
    """When loom_retrieval='agentic', both the SIMPLE coder AND
    the COMPLEX supervisor must have ``read_loom_section`` in
    their tool surface — otherwise the TOC injection in the
    system prompt points at a tool the agent can't call.

    NOTE: the top-level coordinator returned by ``build_agent``
    is the ``Team.router`` classifier, which has NO tools other
    than what it inherits from workspace (it just picks a route).
    Verification has to inspect the route agents themselves —
    that's where the tool actually needs to be reachable."""
    from loom_code.agent import build_agent

    coord, _ = build_agent(
        project, model="echo", loom_retrieval="agentic"
    )
    # Router's routes — pick out the simple + complex agents.
    routes = {
        r.name: r.agent for r in coord.architecture._routes
    }
    for route_name, route_agent in routes.items():
        tools = asyncio.run(route_agent.tools_list())
        tool_names = {
            getattr(t, "name", t) if not isinstance(t, str) else t
            for t in tools
        }
        assert "read_loom_section" in tool_names, (
            f"read_loom_section missing on route '{route_name}' "
            f"(found: {sorted(tool_names)})"
        )
