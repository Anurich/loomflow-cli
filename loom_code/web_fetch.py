"""HTTPS fetch tool for loom-code workers.

Closes a gap in the read-only specialists' tool surface:
``explorer`` and ``auditor`` can read the local project root
(``read``/``grep``/``find``/``ls`` are all path-scoped), but
cannot reach a URL or a GitHub repo. Without a fetch primitive
they silently substitute local files for remote sources when a
task names a URL — a hallucinated-authority failure mode that
``bash curl`` from ``coder`` only solves if the coordinator
routes the work there.

:func:`web_fetch_tool` returns a :class:`loomflow.Tool` that
takes one ``url`` arg and returns the body as text. It is
read-only by construction — it cannot write to disk, mutate
state, or run arbitrary shell — so it preserves the parallel-
delegation safety claim (only ``coder`` writes) even when wired
into the read-only workers.

Lives in loom-code (not loomflow) intentionally: the framework's
``web_tool`` covers SEARCH, this covers FETCH. The two would
naturally pair under a single ``loomflow.tools.web`` namespace,
but until that lands upstream loom-code carries it locally.

Implementation notes:

- Uses ``httpx`` which ships via ``loomflow[web]`` (pyproject
  declares the floor) — no new top-level dependency.
- ``http://`` is silently upgraded to ``https://``; other
  schemes are rejected with a clear ``ERROR: ...`` string so
  the model sees what went wrong instead of stack-tracing.
- GitHub blob URLs rewrite to ``raw.githubusercontent.com``
  before fetching. Models naturally type the human URL and would
  otherwise get a page of HTML — rewriting saves a turn.
- Response cap (5 MB default) is structural, not a soft warning:
  an accidental tarball or binary blob doesn't get to blow
  conversation context.
- Errors return as strings (``ERROR: ...``), not raises — same
  convention as ``loomflow.tools.web.web_tool``. The agent sees
  the error and decides what to do (retry, change URL, escalate).
"""

from __future__ import annotations

import re
from typing import Any

from loomflow import Tool

_TOOL_DESCRIPTION = (
    "Fetch the body of an HTTPS URL and return it as text. Use "
    "for READMEs, raw source files on GitHub, documentation "
    "pages, JSON/YAML configs. For a full repository prefer "
    "`git clone` via bash. GitHub blob URLs are auto-rewritten "
    "to raw URLs so you can paste the human URL and get file "
    "content. Responses over 5MB are rejected. Returns the body "
    "as a string prefixed with status + final URL; errors come "
    "back as `ERROR: ...` strings, not exceptions."
)

_URL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": (
                "Fully-qualified URL to fetch. http:// is "
                "upgraded to https://; other schemes are rejected."
            ),
        }
    },
    "required": ["url"],
}

# github.com/<owner>/<repo>/blob/<ref>/<path>
#   → raw.githubusercontent.com/<owner>/<repo>/<ref>/<path>
# Match the host explicitly so we don't rewrite gitlab/bitbucket/etc.
_GITHUB_BLOB_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/"
    r"(?P<owner>[^/]+)/(?P<repo>[^/]+)/blob/(?P<rest>.+)$"
)

# github.com/<owner>/<repo>/tree/<ref>(/<path>)? is a directory page.
# Fetching it returns ~700kB of React-rendered HTML — pure token
# waste — instead of the file listing the model usually wants.
# We refuse and direct the model at the GitHub contents API
# (returns JSON: file names + download URLs in one call). The
# `rest` group can be empty (tree at root), so use `.*` not `.+`.
_GITHUB_TREE_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/"
    r"(?P<owner>[^/]+)/(?P<repo>[^/]+)/tree/(?P<rest>.*)$"
)


def _normalize_url(url: str) -> tuple[str | None, str | None]:
    """Return ``(normalized_url, error)``. Exactly one is non-None.

    Pure function — extracted from the tool body so it can be
    unit-tested without monkeypatching httpx.
    """
    url = url.strip()
    if not url:
        return None, "ERROR: empty URL"
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    if not url.startswith("https://"):
        return None, (
            f"ERROR: only http(s) URLs are supported, got "
            f"{url[:60]!r}. For local files use `read`; for "
            f"shell commands use `bash`."
        )
    # /tree/ before /blob/: a tree URL would slip past /blob/
    # matching anyway, but we want the directive error, not a
    # silent fallthrough fetch of HTML.
    m_tree = _GITHUB_TREE_RE.match(url)
    if m_tree:
        owner = m_tree["owner"]
        repo = m_tree["repo"]
        # `rest` may be "" (tree at root), "<ref>", or "<ref>/<path>".
        # Split into ref + path so the suggested API URL is correct.
        parts = m_tree["rest"].split("/", 1)
        ref = parts[0] if parts and parts[0] else "main"
        path = parts[1] if len(parts) > 1 else ""
        api_url = (
            f"https://api.github.com/repos/{owner}/{repo}/contents/"
            f"{path}?ref={ref}"
        )
        return None, (
            f"ERROR: {url} is a GitHub DIRECTORY page (React HTML, "
            f"~700kB). Fetching it wastes tokens; use one of:\n"
            f"  - LIST contents (JSON): web_fetch {api_url}\n"
            f"  - LIST via gh CLI:      "
            f"`bash gh api repos/{owner}/{repo}/contents/{path}?ref={ref}`\n"
            f"  - FETCH a file:         web_fetch the /blob/ URL "
            f"(github.com/{owner}/{repo}/blob/{ref}/<path>)\n"
            f"  - FETCH README:         "
            f"web_fetch https://github.com/{owner}/{repo}/blob/{ref}/README.md"
        )
    m = _GITHUB_BLOB_RE.match(url)
    if m:
        url = (
            f"https://raw.githubusercontent.com/"
            f"{m['owner']}/{m['repo']}/{m['rest']}"
        )
    return url, None


def web_fetch_tool(
    *,
    name: str = "web_fetch",
    timeout: float = 15.0,
    max_bytes: int = 5_000_000,
) -> Tool:
    """Build a :class:`Tool` that fetches an HTTPS URL's body.

    Args:
        name: Tool name the model sees (default ``web_fetch``).
            Overridable mostly for tests / co-existence with
            other fetch tools.
        timeout: Per-request timeout in seconds (default 15).
            Applies to connect + read combined.
        max_bytes: Reject responses larger than this — keeps an
            accidental tarball or binary blob from blowing
            conversation context. Default 5 MB.

    Returns:
        A :class:`Tool` named ``web_fetch`` with one ``url: str``
        parameter, returning the response body prefixed by status
        code and final URL (after redirects + GitHub rewriting).

    Example::

        from loom_code.web_fetch import web_fetch_tool
        from loomflow import Agent

        agent = Agent("...", tools=[web_fetch_tool()])
    """

    async def _fetch(url: str) -> str:
        normalized, err = _normalize_url(url)
        if err is not None:
            return err
        # Lazy import — matches the loomflow tool convention so
        # `import loom_code.web_fetch` doesn't pay the httpx cost.
        import httpx

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=timeout,
            ) as client:
                r = await client.get(normalized)
        except httpx.HTTPError as exc:
            return f"ERROR: fetch failed: {exc}"

        # Reject oversized payloads after the fact rather than via
        # Content-Length — many CDNs don't set it correctly and
        # we'd rather download-and-reject than incorrectly block a
        # small response with a bogus header.
        if len(r.content) > max_bytes:
            return (
                f"ERROR: response exceeds {max_bytes} bytes "
                f"({len(r.content)} actual). For large repos use "
                f"`git clone` via bash; for partial reads pass a "
                f"more specific URL."
            )

        # Render with a small header so the model knows what it
        # got — the final URL (after redirects + GitHub rewriting)
        # and status are both load-bearing for follow-ups.
        return (
            f"# {r.url}\n"
            f"status: {r.status_code}\n"
            f"\n"
            f"{r.text}"
        )

    return Tool(
        name=name,
        description=_TOOL_DESCRIPTION,
        fn=_fetch,
        input_schema=_URL_SCHEMA,
    )
