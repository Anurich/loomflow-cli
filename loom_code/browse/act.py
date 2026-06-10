"""Act on an element by its stable ``data-loom-id``.

Robustness measures (each targets a failure we hit on Google Flights):

* Re-select FRESH every call via ``[data-loom-id="N"]`` — never a stale
  handle. (fixes ``Ref e145 not found``)
* scroll_into_view + wait before acting.
* On a click that's intercepted by an overlay/dialog (the
  ``subtree intercepts pointer events`` timeout), fall back to a direct
  JS ``.click()`` dispatch which ignores pointer-event interception.
* For typing: focus, clear, type, and for autocompletes optionally press
  Enter — then the caller re-observes + can page_check the value stuck.
"""

from __future__ import annotations

from typing import Any


def _sel(loom_id: int | str) -> str:
    return f'[data-loom-id="{loom_id}"]'


async def act(
    page: Any,
    loom_id: int | str,
    action: str,
    text: str = "",
) -> str:
    """Perform ``action`` on the element tagged ``loom_id``.

    Actions: click | type | clear | press_enter | select (text=option).
    Returns a short human-readable result string."""
    sel = _sel(loom_id)
    locator = page.locator(sel)
    try:
        count = await locator.count()
    except Exception as exc:  # noqa: BLE001
        return f"error locating [{loom_id}]: {exc}"
    if count == 0:
        return (
            f"element [{loom_id}] is no longer on the page — call "
            "page_observe to get the current elements + ids."
        )
    el = locator.first

    # Bring it into view; ignore failures (some elements report unstable).
    try:
        await el.scroll_into_view_if_needed(timeout=3000)
    except Exception:  # noqa: BLE001
        pass

    action = action.strip().lower()

    if action == "click":
        # Try a normal click first (respects real UX); on interception /
        # timeout, fall back to a JS click that bypasses overlays.
        try:
            await el.click(timeout=4000)
            return f"clicked [{loom_id}]"
        except Exception:  # noqa: BLE001 — overlay / unstable; JS fallback
            try:
                await el.evaluate("e => e.click()")
                return (
                    f"clicked [{loom_id}] "
                    "(via JS — an overlay was in the way)"
                )
            except Exception as exc:  # noqa: BLE001
                return (
                    f"could not click [{loom_id}]: {exc}. An overlay may be "
                    "blocking it — try pressing Escape (page_act on a close "
                    "button) or re-observe."
                )

    if action in ("type", "fill"):
        try:
            await el.click(timeout=3000)
        except Exception:  # noqa: BLE001
            try:
                await el.evaluate("e => e.focus()")
            except Exception:  # noqa: BLE001
                pass
        try:
            await el.fill("")  # clear
        except Exception:  # noqa: BLE001
            pass
        try:
            await el.fill(text)
            return (
                f'typed "{text}" into [{loom_id}]. If it is an '
                "autocomplete, "
                "re-observe and CLICK the matching suggestion; "
                "then page_check the field value stuck."
            )
        except Exception:  # noqa: BLE001 — fall back to keyboard typing
            try:
                await el.press_sequentially(text, delay=20)
                return f'typed "{text}" into [{loom_id}] (key-by-key)'
            except Exception as exc:  # noqa: BLE001
                return f"could not type into [{loom_id}]: {exc}"

    if action == "clear":
        try:
            await el.fill("")
            return f"cleared [{loom_id}]"
        except Exception as exc:  # noqa: BLE001
            return f"could not clear [{loom_id}]: {exc}"

    if action in ("press_enter", "enter", "submit"):
        try:
            await el.press("Enter")
            return f"pressed Enter on [{loom_id}]"
        except Exception as exc:  # noqa: BLE001
            return f"could not press Enter on [{loom_id}]: {exc}"

    if action == "select":
        try:
            await el.select_option(label=text)
            return f'selected "{text}" in [{loom_id}]'
        except Exception as exc:  # noqa: BLE001
            return f"could not select in [{loom_id}]: {exc}"

    return (
        f"unknown action '{action}'. Use: click | type (with text) | clear | "
        "press_enter | select (with text=option)."
    )


async def set_date(page: Any, loom_id: int | str, date_text: str) -> str:
    """Pick a date in a calendar/date-picker widget.

    Date pickers are a grid of day cells — guessing cell ids fails. The
    robust path: open the picker (click the date field), then find the day
    cell whose accessible name matches the target date and click it. Day
    cells almost always carry an aria-label / data-iso with the FULL date
    ("Saturday, June 9, 2026" or "2026-06-09"), so we match on that —
    navigating forward through months if the target isn't visible yet.

    ``date_text`` accepts forms like "2026-06-09", "June 9 2026",
    "9 June 2026". We derive both an ISO and a long-form to match against.
    """
    import asyncio
    import re

    # Parse the date loosely into (year, month, day).
    months = {m: i for i, m in enumerate(
        ["january","february","march","april","may","june","july","august",
         "september","october","november","december"], start=1)}
    t = date_text.strip().lower()
    y = m = d = None
    iso = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if iso:
        y, m, d = int(iso[1]), int(iso[2]), int(iso[3])
    else:
        ym = re.search(r"(20\d{2})", t)
        if ym:
            y = int(ym[1])
        for name, num in months.items():
            if name[:3] in t:
                m = num
                break
        dm = re.search(r"\b(\d{1,2})\b", t.replace(str(y or ""), ""))
        if dm:
            d = int(dm[1])
    if not (y and m and d):
        return (
            f'could not parse the date "{date_text}" — use a form like '
            '"2026-06-09" or "June 9 2026".'
        )

    iso_target = f"{y:04d}-{m:02d}-{d:02d}"
    month_name = [k for k, v in months.items() if v == m][0].capitalize()
    # Long-form fragment most aria-labels contain, e.g. "June 9, 2026".
    long_frag = f"{month_name} {d}, {y}"
    long_frag_alt = f"{month_name} {d} {y}"

    # 1. Open the picker.
    try:
        await page.locator(_sel(loom_id)).first.click(timeout=4000)
    except Exception:  # noqa: BLE001
        pass
    await asyncio.sleep(0.5)

    # JS to find + click a day cell matching the target across the open
    # calendar; returns "clicked" | "not-found". Tries data-iso / aria.
    _CLICK_DAY_JS = r"""
    (args) => {
      const {iso, longA, longB} = args;
      const cands = document.querySelectorAll(
        '[role="gridcell"], [data-iso], [aria-label], [jsname] div, button');
      for (const el of cands) {
        const al = (el.getAttribute('aria-label') || '').trim();
        const di = (el.getAttribute('data-iso') ||
                    el.getAttribute('data-date') || '').trim();
        if (di === iso ||
            (al && (al.includes(longA) || al.includes(longB)))) {
          const r = el.getBoundingClientRect();
          if (r.width > 2 && r.height > 2) { el.click(); return 'clicked'; }
        }
      }
      return 'not-found';
    }
    """

    # JS to close the open date dialog: click a "Done" (or confirm/search)
    # button that lives inside the calendar dialog. Returns "clicked" |
    # "none". We scope to a dialog/popup so we don't hit the page's main
    # Search prematurely — but a "Done" anywhere visible is fine.
    _CLOSE_CALENDAR_JS = r"""
    () => {
      const norm = (s) => (s || '').replace(/\s+/g,' ').trim().toLowerCase();
      const wants = ['done', 'apply', 'ok', 'select', 'confirm'];
      const els = document.querySelectorAll(
        'button, [role="button"], [jsname]');
      for (const el of els) {
        const t = norm(el.innerText || el.textContent);
        const al = norm(el.getAttribute('aria-label'));
        if (wants.includes(t) || wants.includes(al)) {
          const r = el.getBoundingClientRect();
          if (r.width > 2 && r.height > 2) { el.click(); return 'clicked'; }
        }
      }
      return 'none';
    }
    """

    # Try to find the day; if not visible, click a "next month" control and
    # retry a few times (the target month may be ahead).
    for _attempt in range(8):
        try:
            res = await page.evaluate(
                _CLICK_DAY_JS,
                {
                    "iso": iso_target,
                    "longA": long_frag,
                    "longB": long_frag_alt,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return f"date-pick failed: {exc}"
        if res == "clicked":
            await asyncio.sleep(0.4)
            # AUTO-CLOSE the calendar so the Search button becomes
            # reachable. The model reliably fails to click "Done" itself
            # (it clicks calendar arrows instead and gets stuck), so the
            # tool does it: click a Done/Search/confirm control if one is
            # in the open dialog, else press Escape.
            closed = "no"
            try:
                closed = await page.evaluate(_CLOSE_CALENDAR_JS)
            except Exception:  # noqa: BLE001
                pass
            if closed != "clicked":
                try:
                    await page.keyboard.press("Escape")
                except Exception:  # noqa: BLE001
                    pass
            await asyncio.sleep(0.4)
            return (
                f'selected {iso_target} ({long_frag}) and closed the date '
                "picker. page_observe → click the Search button to load "
                "results, then page_read."
            )
        # Advance to the next month and retry.
        try:
            await page.evaluate(r"""
              () => {
                const next = document.querySelector(
                  '[aria-label*="Next" i], [aria-label*="next month" i], '
                  + 'button[jsname][aria-label*="forward" i]');
                if (next) next.click();
              }
            """)
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(0.4)

    return (
        f'could not find {iso_target} in the calendar after paging forward. '
        "The date may be too far out, or the picker uses an unusual layout — "
        "page_check to see it, or proceed with flexible dates."
    )


async def press_key(page: Any, key: str) -> str:
    """Press a global key (e.g. Escape to dismiss an overlay)."""
    try:
        await page.keyboard.press(key)
        return f"pressed {key}"
    except Exception as exc:  # noqa: BLE001
        return f"could not press {key}: {exc}"


async def scroll(page: Any, direction: str = "down", amount: int = 1) -> str:
    """Scroll the page so lazy-loaded content (search results, listings,
    infinite feeds) renders + becomes readable. direction: down | up |
    top | bottom. amount = number of viewport-heights to scroll."""
    import asyncio

    direction = direction.strip().lower()
    try:
        if direction == "top":
            await page.evaluate("window.scrollTo(0, 0)")
        elif direction == "bottom":
            await page.evaluate(
                "window.scrollTo(0, document.body.scrollHeight)"
            )
        else:
            sign = -1 if direction == "up" else 1
            for _ in range(max(1, amount)):
                await page.evaluate(
                    "(s) => window.scrollBy(0, s * window.innerHeight * 0.9)",
                    sign,
                )
                await asyncio.sleep(0.4)  # let content load between scrolls
        await asyncio.sleep(0.5)
        return f"scrolled {direction}"
    except Exception as exc:  # noqa: BLE001
        return f"could not scroll: {exc}"


# JS: count visible autocomplete-suggestion options currently on screen.
# Covers the common patterns (role=option, listbox children, *li in a
# popup). Used to WAIT for the dropdown to render before selecting.
_COUNT_SUGGESTIONS_JS = r"""
() => {
  const sels = [
    '[role="option"]',
    '[role="listbox"] li',
    '[role="listbox"] [role="option"]',
    'ul[role="listbox"] *[role]',
    '.autocomplete-suggestion',
  ];
  let n = 0;
  const seen = new Set();
  for (const s of sels) {
    for (const el of document.querySelectorAll(s)) {
      if (seen.has(el)) continue;
      seen.add(el);
      const r = el.getBoundingClientRect();
      if (r.width > 2 && r.height > 2) n++;
    }
  }
  return n;
}
"""


# JS: type a value into the CURRENTLY FOCUSED element (the real input
# that appeared after we clicked the field) + dispatch input/keyboard
# events so the site's JS reacts. This is the key browser-use trick:
# operate document.activeElement, not the display box we clicked.
_FOCUSED_VALUE_JS = r"""
() => {
  const a = document.activeElement;
  if (!a) return "(no focused element)";
  return (a.value != null ? a.value : (a.innerText || ""));
}
"""


async def fill_combobox(page: Any, loom_id: int | str, value: str) -> str:
    """Fill an autocomplete/combobox the way a human does — the pattern
    that breaks naive 'type into the box' (flights, maps, address, etc.).

    The crucial fix (what browser-use does, what we were missing): the
    visible field is often a DISPLAY box that, when CLICKED, opens a
    separate dialog with the REAL <input>. So:
      1. click the field → opens the widget.
      2. WAIT for it to settle (the real input gets focus).
      3. type into the FOCUSED element (page.keyboard), not the display
         box — slowly, so suggestions load.
      4. WAIT for the suggestion dropdown to render.
      5. ArrowDown + Enter to commit the first suggestion.
      6. read back the committed value.
    """
    import asyncio

    sel = _sel(loom_id)
    locator = page.locator(sel)
    try:
        if await locator.count() == 0:
            return (
                f"combobox [{loom_id}] not found — page_observe for current "
                "ids."
            )
    except Exception as exc:  # noqa: BLE001
        return f"error locating [{loom_id}]: {exc}"
    el = locator.first

    try:
        await el.scroll_into_view_if_needed(timeout=3000)
    except Exception:  # noqa: BLE001
        pass

    # 1. CLICK to open the widget (this is what reveals the real input).
    try:
        await el.click(timeout=4000)
    except Exception:  # noqa: BLE001
        try:
            await el.evaluate("e => e.click()")
        except Exception:  # noqa: BLE001
            pass

    # 2. Wait for the widget to settle + the real input to take focus.
    await asyncio.sleep(0.6)

    # 3. Clear whatever's focused, then type into the FOCUSED element via
    #    the keyboard (so we hit the real input, not the display box).
    try:
        # Select-all + delete clears the focused input cross-platform.
        await page.keyboard.press("Control+A")
        await page.keyboard.press("Meta+A")  # mac
        await page.keyboard.press("Backspace")
    except Exception:  # noqa: BLE001
        pass
    try:
        await page.keyboard.type(value, delay=70)
    except Exception as exc:  # noqa: BLE001
        return f"could not type into the focused input for [{loom_id}]: {exc}"

    # 4. Wait for the suggestion dropdown to render (poll).
    appeared = False
    for _ in range(16):  # ~4s total
        try:
            n = await page.evaluate(_COUNT_SUGGESTIONS_JS)
        except Exception:  # noqa: BLE001
            n = 0
        if n and n > 0:
            appeared = True
            break
        await asyncio.sleep(0.25)

    # 5. Commit the first suggestion via keyboard (robust vs portal DOM).
    try:
        await page.keyboard.press("ArrowDown")
        await asyncio.sleep(0.2)
        await page.keyboard.press("Enter")
    except Exception:  # noqa: BLE001
        try:
            await page.keyboard.press("Enter")
        except Exception:  # noqa: BLE001
            pass

    await asyncio.sleep(0.5)

    # 6. Read what committed — try the original element, else the focused
    #    element (the dialog input may differ from the display box).
    committed = ""
    try:
        committed = await el.input_value()
    except Exception:  # noqa: BLE001
        pass
    if not committed:
        try:
            committed = await page.evaluate(_FOCUSED_VALUE_JS)
        except Exception:  # noqa: BLE001
            committed = "(unknown)"

    note = "" if appeared else (
        " (no suggestion list detected — if the value is wrong, the field "
        "may need a different element; re-observe and check)"
    )
    return (
        f'filled [{loom_id}] with "{value}" → now shows "{committed}"{note}. '
        "page_check to confirm it stuck before the next field."
    )
