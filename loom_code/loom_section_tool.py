"""``read_loom_section(slug)`` — the agentic-retrieval companion tool.

When loom-code's coordinator is built with
``loom_retrieval='agentic'``, the system prompt only carries the
LOOM.md TOC (heading + slug, no bodies). The agent reads the TOC,
identifies the section(s) that look relevant to the user's prompt,
and fetches their bodies on demand via this tool.

Trade vs BM25 retrieval:

* Pro — system prompt stays stable across turns → OpenAI prompt
  cache actually hits (fix for the per-turn LOOM.md injection
  invalidation we hit in production).
* Pro — LLM judgment over keyword overlap; the agent picks
  sections by intent, not by tf-idf.
* Con — adds tool-call latency (one per fetched section).
* Con — slightly larger system prompt (the TOC) but capped + stable.

The tool re-reads LOOM.md on every call (os file cache makes this
cheap). That keeps it stateless — no shared retriever instance to
keep in sync between the REPL's per-turn injection and the agent's
tool surface.
"""

from __future__ import annotations

from pathlib import Path

from loomflow import tool
from loomflow.tools.registry import Tool

from .loominit.injection import LoomRetriever


def read_loom_section_tool(project_root: Path) -> Tool:
    """Build the ``read_loom_section(slug)`` tool for one project.

    ``project_root`` is closed over so the tool always reads from
    the right LOOM.md regardless of agent cwd. The tool LAZY-loads
    LOOM.md on each call — no setup cost at tool-build time, no
    stale cache risk after ``/loominit refresh``.
    """

    async def read_loom_section(slug: str) -> str:
        """Fetch the body of one LOOM.md section by its slug.

        Slugs come from the TOC injected into the system prompt
        (``# LOOM.md section map``). Returns the section body
        verbatim. On unknown slug, returns an error listing the
        available slugs so the agent can self-correct without a
        round-trip.

        Args:
            slug: stable kebab-case identifier from the TOC, e.g.
                  ``"workspace-internals"``.
        """
        retriever = LoomRetriever.from_repo_root(
            project_root, mode="agentic"
        )
        if retriever is None:
            return (
                "read_loom_section: LOOM.md not found at "
                f"{project_root}/LOOM.md. Run /loominit first."
            )
        body = retriever.section_body(slug)
        if body is None:
            available = retriever.available_slugs()
            preview = ", ".join(available[:20])
            extra = (
                f" (and {len(available) - 20} more)"
                if len(available) > 20
                else ""
            )
            return (
                f"read_loom_section: no section with slug "
                f"{slug!r}. Available slugs: {preview}{extra}."
            )
        # Lead with the slug header so the agent can quote both the
        # heading and the body in its response. Match the original
        # heading prefix style so downstream renderers don't have
        # to special-case this tool's output.
        return f"## {slug}\n{body}"

    return tool(
        name="read_loom_section",
        description=(
            "Fetch the body of one LOOM.md section by its slug. "
            "Slugs come from the '# LOOM.md section map' in the "
            "system prompt (when agentic retrieval is enabled). "
            "Use to pull in just the relevant section(s) for the "
            "user's question rather than re-reading source files. "
            "Args: slug (e.g. 'workspace-internals')."
        ),
    )(read_loom_section)
