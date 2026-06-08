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

from pathlib import Path
from typing import Any

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
