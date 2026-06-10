"""Verify page state by reading the live DOM.

loomflow has no image-to-model (vision) input yet, so instead of a
screenshot+vision check we read the ACTUAL values the page holds — the
text content of the inputs/fields visible on screen — and return them so
the agent can confirm "did Delhi land in the origin field?". This catches
the wrong-field / didn't-stick failures without needing vision: the
field's real ``value`` is ground truth.

It also saves a screenshot to disk so the user (and a future vision
upgrade) can inspect what the page looked like.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

# Set-of-Marks: draw a numbered box on each interactive element (using
# the data-loom-id tags observe() already placed) so a screenshot the
# model SEES is annotated with the SAME [ids] it acts on. Returns a
# teardown function name so we can remove the overlay after the shot.
_DRAW_MARKS_JS = r"""
() => {
  const old = document.getElementById('__loom_marks__');
  if (old) old.remove();
  const layer = document.createElement('div');
  layer.id = '__loom_marks__';
  layer.style.cssText =
    'position:fixed;inset:0;z-index:2147483647;pointer-events:none;';
  const COLORS = ['#e6194B','#3cb44b','#4363d8','#f58231','#911eb4',
                  '#469990','#9A6324','#800000','#808000','#000075'];
  let n = 0;
  for (const el of document.querySelectorAll('[data-loom-id]')) {
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) continue;
    if (r.bottom < 0 || r.top > window.innerHeight) continue;  // off-screen
    const id = el.getAttribute('data-loom-id');
    const c = COLORS[n % COLORS.length]; n++;
    const box = document.createElement('div');
    box.style.cssText =
      `position:fixed;left:${r.left}px;top:${r.top}px;width:${r.width}px;`+
      `height:${r.height}px;border:2px solid ${c};box-sizing:border-box;`;
    const tag = document.createElement('div');
    tag.textContent = id;
    tag.style.cssText =
      `position:fixed;left:${r.left}px;top:${Math.max(0,r.top-14)}px;`+
      `background:${c};color:#fff;font:bold 11px monospace;padding:0 3px;`+
      `line-height:14px;`;
    layer.appendChild(box); layer.appendChild(tag);
  }
  document.body.appendChild(layer);
}
"""

_REMOVE_MARKS_JS = (
    "() => { const m = document.getElementById('__loom_marks__'); "
    "if (m) m.remove(); }"
)


async def screenshot_b64(page: Any, marks: bool = True) -> str | None:
    """Base64 PNG of the current viewport. With ``marks`` (default), draw
    the numbered Set-of-Marks overlay first so the model sees the [ids],
    then remove it. Returns None on failure."""
    drew = False
    try:
        if marks:
            await page.evaluate(_DRAW_MARKS_JS)
            drew = True
        png = await page.screenshot(type="png")
        return base64.b64encode(png).decode("ascii")
    except Exception:  # noqa: BLE001
        return None
    finally:
        if drew:
            try:
                await page.evaluate(_REMOVE_MARKS_JS)
            except Exception:  # noqa: BLE001
                pass

# Snapshot the values of every labelled input/field + recent prices, so a
# "what's currently entered / shown" question is answerable from the DOM.
_STATE_JS = r"""
() => {
  const clean = (s) => (s || "").replace(/\s+/g, " ").trim();
  const fields = [];
  for (const el of document.querySelectorAll(
        "input, textarea, select, [role=combobox], [contenteditable]")) {
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) continue;
    const label = clean(
      el.getAttribute("aria-label") ||
      el.getAttribute("placeholder") ||
      el.getAttribute("name") || "");
    const val = clean(el.value || el.getAttribute("value") ||
                      el.innerText || "");
    if (label || val) fields.push({ label, value: val.slice(0, 60) });
  }
  return { title: document.title, url: location.href, fields };
}
"""


async def verify(page: Any, question: str, model: str | None = None) -> str:
    """Return the page's current title/URL + the values held in its
    fields, so the agent can answer ``question`` from real DOM state.

    (``model`` is accepted for API stability / a future vision upgrade
    but unused — loomflow has no image input yet.)"""
    try:
        state = await page.evaluate(_STATE_JS)
    except Exception as exc:  # noqa: BLE001
        return f"could not read page state ({exc}); try page_observe."

    # Best-effort screenshot to disk for the user / future vision.
    try:
        shot = Path.home() / ".loom-code" / "last_page.png"
        shot.parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(shot), type="png")
    except Exception:  # noqa: BLE001
        pass

    lines = [
        f"Checking: {question}",
        f"PAGE: {state.get('title', '')}",
        f"URL: {state.get('url', '')}",
        "Current field values:",
    ]
    fields = state.get("fields") or []
    if not fields:
        lines.append("  (no field values found)")
    for f in fields:
        lbl = f.get("label") or "(unlabelled)"
        val = f.get("value") or ""
        lines.append(f'  {lbl}: "{val}"')
    lines.append(
        "\n→ Compare the values above to what you intended. If a field has "
        "the WRONG value (e.g. origin shows your location, not what you "
        "typed), fix it: re-observe, clear that field, type again, and for "
        "autocompletes CLICK the matching suggestion."
    )
    return "\n".join(lines)


async def look(page: Any, question: str, model: str | None = None) -> str:
    """VISION: screenshot the page (with Set-of-Marks [id] overlay) and
    ask the session's multimodal model ``question`` about it. This is how
    the agent SEES the page — reads prices/results, confirms layout,
    disambiguates fields — instead of inferring from DOM text alone.

    Sends the image via loomflow's new image input
    (metadata['_loom_images']) so it works with whichever model the user
    runs (Claude or GPT — both multimodal). Degrades to a DOM read if no
    model / vision is available."""
    b64 = await screenshot_b64(page, marks=True)
    if b64 is None:
        return await verify(page, question, model)
    if not model:
        return (
            "no model available for page_look — using DOM values:\n\n"
            + await verify(page, question, model)
        )
    try:
        from loomflow import Agent
    except Exception:  # noqa: BLE001
        return await verify(page, question, model)
    prompt = (
        "You are looking at a screenshot of a web page. Numbered colored "
        "boxes mark interactive elements (the number is the element id you "
        "can act on). Answer concisely and concretely. If asked for prices/"
        "results, read the actual values you see and list them.\n\n"
        f"Question: {question}"
    )
    try:
        probe = Agent(prompt, model=model)
        result = await probe.run(
            "(see attached screenshot)",
            metadata={
                "_loom_images": [
                    {"data": b64, "media_type": "image/png"}
                ]
            },
        )
        out = getattr(result, "output", None) or str(result)
        return str(out).strip() or "(model returned nothing)"
    except Exception as exc:  # noqa: BLE001
        # Vision failed (model not multimodal / API issue) → DOM fallback.
        return (
            f"page_look vision unavailable ({type(exc).__name__}); "
            "using DOM values instead:\n\n"
            + await verify(page, question, model)
        )
