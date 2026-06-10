"""A single headed browser session for loom-code's ``/computer`` mode.

One ``BrowserSession`` owns one visible Chromium window for the life of a
REPL session. The agent's page_* tools all act through it. It is created
lazily on the first navigation and torn down on mode-off / exit.

Reliability design (learned from our Google-Flights failures + how
browser-use stays robust):

* **Stable element handles.** Instead of ephemeral snapshot refs (which
  die the instant the DOM changes — the ``Ref e145 not found`` error),
  every interactive element is tagged in the live DOM with a
  ``data-loom-id="N"`` attribute. Acting later re-selects by that
  attribute (``[data-loom-id="N"]``), so Playwright re-resolves the
  element FRESH on every action — no stale handles. The id rides on the
  element through re-renders unless the element itself is recreated.

* **One persistent page.** The session keeps the active page so observe →
  act → observe operates on the same evolving page, like a human.

This module is intentionally dependency-light: Playwright only (already
installed for the project). No browser-use, no extra packages.
"""

from __future__ import annotations

from typing import Any


class BrowserSession:
    """Lazily-launched headed Chromium + the active page. Single-page for
    now (the active tab); multi-tab can come later."""

    def __init__(self, *, headless: bool = False) -> None:
        self._headless = headless
        self._pw: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        # index -> True; the live truth lives in the DOM (data-loom-id),
        # this just tracks the highest id assigned this observe pass.
        self._max_id: int = 0

    async def start(self) -> None:
        """Launch the browser if not already running. Idempotent."""
        if self._page is not None:
            return
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        # Headed = the user watches it work (the whole point of /computer).
        self._browser = await self._pw.chromium.launch(
            headless=self._headless,
            args=["--start-maximized"],
        )
        # A real-ish context: viewport=None lets the window size drive it
        # (with --start-maximized), and a normal UA reduces trivial
        # bot-walls.
        self._context = await self._browser.new_context(
            viewport=None,
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        self._page = await self._context.new_page()

    @property
    def page(self) -> Any:
        if self._page is None:
            raise RuntimeError("browser not started — call start() first")
        return self._page

    async def goto(self, url: str) -> None:
        await self.start()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        await self._page.goto(
            url, wait_until="domcontentloaded", timeout=45000
        )

    async def close(self) -> None:
        """Tear everything down. Safe to call multiple times."""
        for closer in (
            getattr(self._context, "close", None),
            getattr(self._browser, "close", None),
            getattr(self._pw, "stop", None),
        ):
            if closer is not None:
                try:
                    await closer()
                except Exception:  # noqa: BLE001 — best-effort teardown
                    pass
        self._pw = self._browser = self._context = self._page = None
        self._max_id = 0
