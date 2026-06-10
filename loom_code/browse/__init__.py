"""loom-code's focused browser engine for ``/computer`` mode.

Exposes ``browse_tools(model)`` — a list of loom-code tools backed by ONE
shared headed-Chromium :class:`BrowserSession`. The agent drives the web
through these instead of the Playwright MCP server (which had ephemeral
refs that broke on dynamic pages). Reliability comes from stable
``data-loom-id`` handles (observe), re-resolve-fresh + overlay-safe
acting (act), and vision verification (check).

Tools:
  page_open(url)              navigate; launches the visible browser
  page_observe()             list interactive elements with stable [ids]
  page_act(id, action, text) click/type/select on an element by id
  page_check(question)       vision-verify what's on screen
  page_back()                browser back
"""

from __future__ import annotations

from loomflow import tool
from loomflow.tools.registry import Tool

from .act import act, fill_combobox, press_key, scroll, set_date
from .observe import observe, read_text
from .session import BrowserSession
from .verify import look, verify

# Live browser sessions created this process. The REPL closes them all on
# exit so the headed Chromium window doesn't linger after loom-code quits.
_LIVE_SESSIONS: list[BrowserSession] = []


async def close_all_browsers() -> None:
    """Close every browser session opened by /computer. Best-effort;
    called from the REPL's exit teardown. Never raises."""
    for s in _LIVE_SESSIONS:
        try:
            await s.close()
        except Exception:  # noqa: BLE001 — teardown must not fail exit
            pass
    _LIVE_SESSIONS.clear()


def browse_tools(model: str | None = None) -> list[Tool]:
    """Build the page_* tools over a single shared browser session. The
    session launches lazily on the first page_open and persists across
    calls so observe → act → observe walks the same evolving page."""
    session = BrowserSession()
    _LIVE_SESSIONS.append(session)  # so the REPL can close it on exit

    async def page_open(url: str) -> str:
        """Open a URL in the visible browser, then list what's on it."""
        try:
            await session.goto(url)
        except Exception as exc:  # noqa: BLE001
            return f"could not open {url}: {exc}"
        _els, rendered = await observe(session.page)
        return f"opened {url}\n\n{rendered}"

    async def page_observe() -> str:
        """Re-read the current page: list interactive elements + their
        stable ids. Call this before EVERY act (ids change as the page
        changes) and after navigation."""
        try:
            _els, rendered = await observe(session.page)
        except RuntimeError:
            return "no page open yet — call page_open(url) first."
        return rendered

    async def page_act(id: int, action: str, text: str = "") -> str:
        """Act on an element by its [id] from the latest page_observe.
        action: click | type (with text) | clear | press_enter | select
        (with text=option). After acting the page may change — call
        page_observe again to get fresh ids."""
        try:
            page = session.page
        except RuntimeError:
            return "no page open yet — call page_open(url) first."
        result = await act(page, id, action, text)
        # Auto re-observe so the agent always sees the post-action state
        # with fresh ids (this is what keeps ids from going stale).
        _els, rendered = await observe(page)
        return f"{result}\n\n{rendered}"

    async def page_fill(id: int, value: str) -> str:
        """Fill an AUTOCOMPLETE / combobox field by id (origin/destination
        on flight + map sites, address fields, search-with-suggestions).
        Types, waits for the suggestion dropdown, and selects the first
        match via keyboard — the robust way these widgets commit a value
        (plain page_act 'type' reverts on them). Use this for any field
        that shows a suggestion list. After it, page_check the value."""
        try:
            page = session.page
        except RuntimeError:
            return "no page open yet — call page_open(url) first."
        result = await fill_combobox(page, id, value)
        _els, rendered = await observe(page)
        return f"{result}\n\n{rendered}"

    async def page_set_date(id: int, date: str) -> str:
        """Pick a date in a calendar/date-picker by id (flight dates,
        booking dates). Opens the picker and clicks the day matching the
        date — pages forward through months if needed. date forms:
        '2026-06-09' or 'June 9 2026'. Use this for ANY calendar widget;
        do NOT guess day-cell ids with page_act."""
        try:
            page = session.page
        except RuntimeError:
            return "no page open yet — call page_open(url) first."
        result = await set_date(page, id, date)
        _els, rendered = await observe(page)
        return f"{result}\n\n{rendered}"

    async def page_scroll(direction: str = "down", amount: int = 1) -> str:
        """Scroll the page so lazy-loaded content (search results, flight
        listings, feeds) renders. direction: down | up | top | bottom.
        After scrolling, call page_read to capture the newly-loaded text.
        Args: direction; amount (viewport-heights, default 1)."""
        try:
            page = session.page
        except RuntimeError:
            return "no page open yet — call page_open(url) first."
        return await scroll(page, direction, amount)

    async def page_read() -> str:
        """READ the page's visible text — prices, listings, results,
        article text. Use this to extract OUTCOMES (e.g. flight prices,
        product prices, search results). page_observe lists clickable
        elements; page_read gives you the actual CONTENT to report."""
        try:
            page = session.page
        except RuntimeError:
            return "no page open yet — call page_open(url) first."
        return await read_text(page)

    async def page_look(question: str) -> str:
        """SEE the page — take a screenshot (with numbered [id] boxes on
        elements) and have the vision model answer about it. Use when DOM
        text isn't enough: reading prices/results, understanding layout,
        confirming what a complex widget shows. Costs more than page_read
        (sends an image), so use when you need to actually SEE. Arg:
        question (what to look for)."""
        try:
            page = session.page
        except RuntimeError:
            return "no page open yet — call page_open(url) first."
        return await look(page, question, model=model)

    async def page_check(question: str) -> str:
        """Visually verify the page: screenshot + ask a yes/no question
        (e.g. 'is Delhi the origin?'). Use after typing to confirm it
        stuck, or when the DOM text is ambiguous."""
        try:
            page = session.page
        except RuntimeError:
            return "no page open yet — call page_open(url) first."
        return await verify(page, question, model=model)

    async def page_back() -> str:
        """Go back one page in history, then list what's on it."""
        try:
            page = session.page
        except RuntimeError:
            return "no page open yet — call page_open(url) first."
        try:
            await page.go_back(wait_until="domcontentloaded")
        except Exception as exc:  # noqa: BLE001
            return f"could not go back: {exc}"
        _els, rendered = await observe(page)
        return rendered

    async def page_press(key: str) -> str:
        """Press a global key — most useful: Escape to close an overlay /
        dropdown that's blocking a click. Then re-observe."""
        try:
            page = session.page
        except RuntimeError:
            return "no page open yet — call page_open(url) first."
        result = await press_key(page, key)
        _els, rendered = await observe(page)
        return f"{result}\n\n{rendered}"

    tools: list[Tool] = [
        tool(
            name="page_open",
            description=(
                "Open a URL in the user's VISIBLE browser and list the "
                "interactive elements on it. Start every web task here. "
                "Arg: url."
            ),
        )(page_open),
        tool(
            name="page_observe",
            description=(
                "List the current page's interactive elements with stable "
                "[ids] + their values. Call before EVERY page_act (ids can "
                "change as the page updates) and after any navigation."
            ),
        )(page_observe),
        tool(
            name="page_act",
            description=(
                "Act on an element by its [id] from the latest "
                "page_observe. Args: id (int); action (click | type | clear "
                "| press_enter | select); text (for type/select). For an "
                "autocomplete: type a few chars, page_observe, then click "
                "the matching suggestion. Always set BOTH origin and "
                "destination explicitly for travel sites."
            ),
        )(page_act),
        tool(
            name="page_fill",
            description=(
                "Fill an AUTOCOMPLETE/combobox field (the kind that shows a "
                "suggestion dropdown — flight origin/destination, maps, "
                "address, search-with-suggestions). Types, waits for the "
                "dropdown, and selects the first match via keyboard so the "
                "value actually COMMITS (plain page_act 'type' reverts on "
                "these). Use this instead of page_act type for any field "
                "with suggestions. Args: id; value. Then page_check it."
            ),
        )(page_fill),
        tool(
            name="page_scroll",
            description=(
                "Scroll the page so lazy-loaded content (search results, "
                "flight/product listings, feeds) renders — then page_read "
                "to capture it. Args: direction (down|up|top|bottom); "
                "amount (default 1)."
            ),
        )(page_scroll),
        tool(
            name="page_look",
            description=(
                "SEE the page with vision — screenshots it (numbered [id] "
                "boxes on elements) and a vision model answers your "
                "question. Use when DOM text isn't enough: reading "
                "prices/results, understanding layout, or when page_read/"
                "page_observe seem to miss content. Costs more (sends an "
                "image) — use when you must actually SEE. Arg: question."
            ),
        )(page_look),
        tool(
            name="page_read",
            description=(
                "READ the page's visible text content — prices, search "
                "results, listings, article body. Use this to EXTRACT and "
                "report outcomes (flight prices, product prices, results). "
                "page_observe lists clickable elements; page_read gives the "
                "actual content. No args."
            ),
        )(page_read),
        tool(
            name="page_set_date",
            description=(
                "Pick a date in a calendar/date-picker by id (flight dates, "
                "booking calendars). Opens the picker + clicks the matching "
                "day, paging forward through months as needed. NEVER guess "
                "day-cell ids with page_act — use this. Args: id; date "
                "('2026-06-09' or 'June 9 2026')."
            ),
        )(page_set_date),
        tool(
            name="page_check",
            description=(
                "Verify the page: returns the REAL current values of the "
                "page's fields so you can confirm what landed where (e.g. "
                "'is Delhi in Where from?'). Use after filling. Arg: "
                "question."
            ),
        )(page_check),
        tool(
            name="page_back",
            description="Go back one page in browser history, then list it.",
        )(page_back),
        tool(
            name="page_press",
            description=(
                "Press a global key — usually Escape, to dismiss an overlay/"
                "dropdown blocking a click. Then the page is re-listed. "
                "Arg: key (e.g. 'Escape')."
            ),
        )(page_press),
    ]
    # Stash the session so the REPL can close it on exit / mode-off.
    for t in tools:
        t._loom_browser_session = session
    return tools


__all__ = ["browse_tools", "BrowserSession"]
