"""Pin the three-tier output contract of ``graphify__query``.

The tool returns DIRECT (literal label hits) + NEIGHBOR (1-hop
graph neighbours) + COMMUNITY (Leiden-cluster peers). Without
this test the next refactor could silently regress back to
literal-only matching — which would miss the whole point of
having a knowledge graph (you'd just be using a slow grep).

Tests build a tiny in-memory graph rather than relying on a
real graphify-built graph.json, so they're hermetic and fast.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from loom_code.skills.graphify import tools as graphify_tools


def _build_fake_graph_json(out: Path) -> None:
    """Create a small NetworkX-node-link-formatted graph with two
    communities and a clear DIRECT/NEIGHBOR/COMMUNITY split.

    Topology:
      community 0:  auth_handler ── validate_token ── session_store
      community 1:  ui_button ── render_panel

    Query "auth" should:
      - DIRECT hit  → auth_handler
      - NEIGHBOR    → validate_token (1-hop, no "auth" in name)
      - COMMUNITY   → session_store (community 0, no "auth", no edge)
      - exclude     → ui_button, render_panel (community 1)
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    graph = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {
                "id": "auth_handler",
                "label": "AuthHandler",
                "source_file": "auth.py",
                "source_location": "L10",
                "community": 0,
            },
            {
                "id": "validate_token",
                "label": "validate_token",
                "source_file": "tokens.py",
                "source_location": "L25",
                "community": 0,
            },
            {
                "id": "session_store",
                "label": "SessionStore",
                "source_file": "sessions.py",
                "source_location": "L5",
                "community": 0,
            },
            {
                "id": "ui_button",
                "label": "Button",
                "source_file": "ui.py",
                "source_location": "L1",
                "community": 1,
            },
            {
                "id": "render_panel",
                "label": "render_panel",
                "source_file": "ui.py",
                "source_location": "L40",
                "community": 1,
            },
        ],
        "links": [
            {"source": "auth_handler", "target": "validate_token"},
            {"source": "validate_token", "target": "session_store"},
            {"source": "ui_button", "target": "render_panel"},
        ],
        "hyperedges": [],
    }
    out.write_text(json.dumps(graph))


def test_query_returns_direct_neighbor_and_community_tiers(
    tmp_path: Path,
) -> None:
    """A query for ``auth`` against the fake graph hits exactly:
    DIRECT auth_handler, NEIGHBOR validate_token, COMMUNITY
    session_store. UI cluster (community 1) is absent."""
    graph_path = tmp_path / ".loom" / "graphify" / "graph.json"
    _build_fake_graph_json(graph_path)

    result = asyncio.run(
        graphify_tools.query.fn("auth", path=str(tmp_path))
    )

    # Tier headers all present.
    assert "DIRECT" in result, f"DIRECT section missing:\n{result}"
    assert "NEIGHBOR" in result, f"NEIGHBOR section missing:\n{result}"
    assert "COMMUNITY" in result, (
        f"COMMUNITY section missing:\n{result}"
    )

    # Right node in each tier.
    assert "AuthHandler" in result, (
        f"direct match AuthHandler missing:\n{result}"
    )
    assert "validate_token" in result, (
        "1-hop neighbour validate_token missing — query is "
        f"regressing to literal-only matching:\n{result}"
    )
    assert "SessionStore" in result, (
        "community peer SessionStore missing — Leiden cluster "
        f"info not being used at query time:\n{result}"
    )

    # Unrelated community absent (no leakage from UI cluster).
    assert "Button" not in result
    assert "render_panel" not in result


def test_query_handles_no_match(tmp_path: Path) -> None:
    """An unmatched keyword returns a helpful message, not an
    empty section dump. Important — the agent needs a clear
    'no match, try X' signal."""
    graph_path = tmp_path / ".loom" / "graphify" / "graph.json"
    _build_fake_graph_json(graph_path)

    result = asyncio.run(
        graphify_tools.query.fn(
            "nonexistent_keyword_xyzzy", path=str(tmp_path)
        )
    )
    assert "no nodes matched" in result.lower()
    # Should not include the tier headers — nothing to show.
    assert "DIRECT matches" not in result
    assert "NEIGHBOR" not in result


def test_query_neighbor_tier_works_without_community_data(
    tmp_path: Path,
) -> None:
    """Older graph.json files might lack the community field.
    Neighbor tier should still surface even when community
    extraction yields nothing — degrades gracefully."""
    graph_path = tmp_path / ".loom" / "graphify" / "graph.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {
                "id": "auth_handler",
                "label": "AuthHandler",
                "source_file": "auth.py",
                "source_location": "L1",
                # NO community field
            },
            {
                "id": "validate_token",
                "label": "validate_token",
                "source_file": "tokens.py",
                "source_location": "L1",
                # NO community field
            },
        ],
        "links": [
            {"source": "auth_handler", "target": "validate_token"},
        ],
        "hyperedges": [],
    }
    graph_path.write_text(json.dumps(graph))

    result = asyncio.run(
        graphify_tools.query.fn("auth", path=str(tmp_path))
    )
    assert "AuthHandler" in result
    # Neighbor tier still works.
    assert "validate_token" in result
    # COMMUNITY header SHOULDN'T appear since there were no
    # community ids to expand from.
    assert "COMMUNITY (" not in result
