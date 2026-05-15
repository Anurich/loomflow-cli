"""Annotator prompts + structured-output schemas.

We use Pydantic models as output schemas (loomflow's ``Agent`` takes
``output_schema=``) so each annotation call returns validated JSON
instead of free-form markdown. Two wins:

* No parse-from-prose layer — the assembler in :mod:`annotator`
  receives typed objects directly.
* Loomflow's retry-on-invalid path (``output_validation_retries``)
  handles the occasional malformed response without polluting the
  output.

Prompts are deliberately TERSE and FACT-CONSTRAINED. The point of
loominit is grounded narrative — the model should describe what it
sees in the structural data, not embellish. Adjectives like
"powerful" / "elegant" / "robust" are explicitly forbidden.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Project-level overview
# ---------------------------------------------------------------------------


class _ProjectOverviewOutput(BaseModel):
    """Result of the project-overview LLM call.

    ``overview`` is one paragraph describing what the project is +
    what problem it solves. ``tech_stack`` is a short bulleted list
    of the technology stack (language, frameworks, test runner,
    build system) — bullets are plain strings, the assembler
    renders them with ``- ``.
    """

    overview: str = Field(
        description=(
            "One paragraph (3-5 sentences) explaining what this "
            "project IS and what problem it solves. Ground every "
            "claim in the structural facts provided. No adjectives "
            "like 'powerful', 'elegant', 'robust' — just describe."
        )
    )
    tech_stack: list[str] = Field(
        default_factory=list,
        description=(
            "Bulleted technology-stack items: language + version "
            "requirement, major frameworks visible in imports, test "
            "runner, build / package format. One bullet per item. "
            "Omit anything you cannot determine from the facts."
        ),
    )


PROJECT_OVERVIEW_INSTRUCTIONS = """\
You are documenting a Python codebase. You will NOT invent details;
every claim MUST be supported by the structural facts you are
given. If a fact is not provided, leave it out.

Your output goes into a `LOOM.md` file the engineer reads on every
session. Tight, factual, no marketing voice. Three to five
sentences in the overview is plenty.
"""


def render_project_overview_prompt(
    *,
    project_name: str,
    project_description: str | None,
    requires_python: str | None,
    top_dirs: list[tuple[str, int]],
    entry_points: list[str],
    central_files: list[tuple[str, int]],
    notable_imports: list[str],
) -> str:
    """Build the user-message body for the project-overview call.

    Arguments are pre-extracted from the :class:`schema.LoomIndex`
    by :mod:`annotator`; this function is pure formatting so the
    test suite can pin the prompt shape without spinning up an
    indexer."""
    lines: list[str] = ["# Structural facts\n"]
    lines.append(f"Project name: {project_name}")
    if project_description:
        lines.append(f"Description (from pyproject): {project_description}")
    if requires_python:
        lines.append(f"Requires Python: {requires_python}")
    lines.append("")

    if top_dirs:
        lines.append("Top-level directories:")
        for d, n in top_dirs:
            lines.append(f"  {d}/  ({n} files)")
        lines.append("")

    if entry_points:
        lines.append("Entry points:")
        for ep in entry_points:
            lines.append(f"  {ep}")
        lines.append("")

    if central_files:
        lines.append(
            "Most-central files (by import-graph PageRank, top 10):"
        )
        for path, n in central_files:
            lines.append(f"  {path}  (imported by {n} files)")
        lines.append("")

    if notable_imports:
        lines.append(
            "Notable third-party packages imported (top 10 by usage):"
        )
        for imp in notable_imports:
            lines.append(f"  {imp}")
        lines.append("")

    lines.append("# Your task\n")
    lines.append(
        "Return a JSON object with `overview` (one paragraph, 3-5 "
        "sentences) and `tech_stack` (bulleted list of strings). "
        "Ground EVERY claim in the facts above."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cluster-level annotation
# ---------------------------------------------------------------------------


class SymbolPurpose(BaseModel):
    """One symbol → one-line purpose, with the citation the model
    was given. ``path`` and ``line`` come straight from the input
    so the model can't invent a location (we discard returned
    citations that don't match)."""

    name: str
    path: str
    line: int
    purpose: str = Field(
        description=(
            "One short line (≤120 chars) describing what this "
            "symbol does — verb-first, no marketing. If you can't "
            "determine the purpose from the signature + docstring + "
            "callers, return 'undocumented' rather than guess."
        )
    )


class _ClusterAnnotationOutput(BaseModel):
    """Result of one per-cluster LLM call.

    The assembler in :mod:`annotator` stitches each section into a
    cluster's chunk of LOOM.md. Empty lists are valid — the assembler
    elides the corresponding heading rather than rendering an empty
    section.
    """

    narrative: str = Field(
        description=(
            "ONE paragraph (3-6 sentences) describing what this "
            "subsystem does, the way a new contributor would need "
            "explained. Reference specific symbols with their "
            "`path:line` when natural. No adjectives like "
            "'powerful' / 'elegant'."
        )
    )
    data_flow: list[str] = Field(
        default_factory=list,
        description=(
            "3-7 bullets describing how data / control moves "
            "through this subsystem. Each bullet should mention a "
            "specific symbol if possible. Return [] if this "
            "subsystem is a leaf utility with no flow."
        ),
    )
    conventions: list[str] = Field(
        default_factory=list,
        description=(
            "Verified conventions used here — derived from the "
            "structural data (repeated decorators, naming, file "
            "layout). One bullet per convention. Return [] if "
            "nothing notable."
        ),
    )
    symbol_purposes: list[SymbolPurpose] = Field(
        default_factory=list,
        description=(
            "API-surface symbols + the top 10 by inbound-call "
            "count. One entry per symbol. Use the `name` / `path` / "
            "`line` provided in the structural data verbatim."
        ),
    )


CLUSTER_ANNOTATION_INSTRUCTIONS = """\
You are documenting ONE subsystem of a Python codebase for a
`LOOM.md` index that an engineer reads on every session. Your
output must be terse, factual, and grounded in the structural data
provided. NO invented details, NO marketing adjectives.

Return JSON matching the requested schema. Empty lists are valid
when the corresponding aspect doesn't apply to this subsystem.
"""


def render_cluster_prompt(
    *,
    cluster_title: str,
    paths: list[str],
    symbol_table: str,
    outbound_internal: list[str],
    inbound_internal: list[str],
    third_party: list[str],
    entry_points_in_cluster: list[str],
) -> str:
    """Build the user-message body for one cluster's annotation call.

    All arguments are pre-rendered strings/lists from
    :class:`schema.LoomIndex`; this function is pure formatting."""
    lines: list[str] = ["# Subsystem\n"]
    lines.append(f"Title: {cluster_title}")
    lines.append("Files:")
    for p in paths:
        lines.append(f"  {p}")
    lines.append("")

    lines.append("# Symbols in this subsystem")
    lines.append("")
    lines.append(
        "Format: `path:line  qualified_name  (kind)  "
        "signature  — docstring_first_line`"
    )
    lines.append("")
    lines.append(symbol_table)
    lines.append("")

    if outbound_internal:
        lines.append("# This subsystem imports from (other parts of the repo)")
        for line in outbound_internal:
            lines.append(f"  {line}")
        lines.append("")
    if inbound_internal:
        lines.append("# This subsystem is imported by")
        for line in inbound_internal:
            lines.append(f"  {line}")
        lines.append("")
    if third_party:
        lines.append("# Third-party packages imported here")
        for line in third_party:
            lines.append(f"  {line}")
        lines.append("")
    if entry_points_in_cluster:
        lines.append("# Entry points within this subsystem")
        for line in entry_points_in_cluster:
            lines.append(f"  {line}")
        lines.append("")

    lines.append("# Your task\n")
    lines.append(
        "Return a JSON object with `narrative` (one paragraph), "
        "`data_flow` (bullets), `conventions` (bullets), "
        "`symbol_purposes` (list of {name, path, line, purpose}). "
        "Cite path:line where relevant. Empty lists are valid."
    )
    return "\n".join(lines)
