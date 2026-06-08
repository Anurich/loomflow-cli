"""Observe a page: find interactive elements, tag them with stable ids,
serialize a compact view for the LLM.

The JS below runs in the page and:
  1. Finds interactive elements — native controls (a/button/input/
     select/textarea/summary), ARIA roles (button/link/checkbox/combobox/
     menuitem/tab/option/switch/radio), and contenteditable. Plus a
     visibility + size filter so we don't list hidden/zero-area noise.
  2. Tags each with ``data-loom-id="N"`` (stable handle for acting).
  3. Returns ``[{id, tag, role, label, value, kind}]``.

The Python side turns that into lines the model reads:
    [12] textbox "Where from?"  (value="Kathmandu")
    [13] button "Search"
``kind`` ("input"|"button"|"link"|"select") helps the model pick the
right action.
"""

from __future__ import annotations

from typing import Any

# JS: tag + collect interactive elements. Returns a JSON-able list.
# Kept as one expression-returning function so page.evaluate gets the
# array back directly.
_COLLECT_JS = r"""
() => {
  const INTERACTIVE_ROLES = new Set([
    "button","link","checkbox","combobox","menuitem","tab","option",
    "switch","radio","textbox","searchbox","slider","spinbutton",
  ]);
  const NATIVE = new Set(["A","BUTTON","INPUT","SELECT","TEXTAREA","SUMMARY"]);
  const out = [];
  let id = 0;

  const isVisible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) return false;
    const s = window.getComputedStyle(el);
    if (s.display === "none" || s.visibility === "hidden" || s.opacity === "0")
      return false;
    return true;
  };

  // Resolve an element's human label as robustly as possible — this is
  // what lets the model understand "which field is which". Order matters:
  // explicit accessible name first, then associated <label>, then nearby
  // text, then attributes.
  const clean = (s) => (s || "").replace(/\s+/g, " ").trim().slice(0, 90);
  const labelFor = (el) => {
    // 1. Explicit accessible name.
    const aria = el.getAttribute("aria-label");
    if (aria) return clean(aria);
    // 2. aria-labelledby → referenced element(s) text.
    const lb = el.getAttribute("aria-labelledby");
    if (lb) {
      const txt = lb.split(/\s+/)
        .map((id) => document.getElementById(id))
        .filter(Boolean)
        .map((n) => n.innerText || n.textContent || "")
        .join(" ");
      if (clean(txt)) return clean(txt);
    }
    // 3. <label for=id> or wrapping <label>.
    if (el.id) {
      const lab = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (lab && clean(lab.innerText)) return clean(lab.innerText);
    }
    const wrapLab = el.closest("label");
    if (wrapLab && clean(wrapLab.innerText)) return clean(wrapLab.innerText);
    // 4. placeholder / title.
    const ph = el.getAttribute("placeholder");
    if (ph) return clean(ph);
    const title = el.getAttribute("title");
    if (title) return clean(title);
    // 5. The element's own visible text (good for buttons/links).
    const own = clean(el.innerText || el.textContent || "");
    if (own) return own;
    // 6. For inputs with no label yet: the nearest preceding text node /
    //    sibling label-ish element (handles "Where from?" rendered as a
    //    separate span above the input).
    let p = el.previousElementSibling;
    for (let i = 0; i < 3 && p; i++) {
      const t = clean(p.innerText || p.textContent || "");
      if (t && t.length < 40) return t;
      p = p.previousElementSibling;
    }
    // 7. Last resort: name / id attribute.
    const name = el.getAttribute("name") || el.getAttribute("id") || "";
    return clean(name);
  };

  const kindFor = (el, role) => {
    const t = el.tagName;
    if (t === "INPUT" || t === "TEXTAREA" || role === "textbox" ||
        role === "searchbox" || role === "combobox" ||
        el.isContentEditable) return "input";
    if (t === "SELECT") return "select";
    if (t === "A" || role === "link") return "link";
    return "button";
  };

  const isInteractive = (el) => {
    if (NATIVE.has(el.tagName)) return true;
    const role = el.getAttribute("role");
    if (role && INTERACTIVE_ROLES.has(role)) return true;
    if (el.isContentEditable) return true;
    // Cursor:pointer is a strong custom-button signal.
    if (window.getComputedStyle(el).cursor === "pointer" &&
        (el.onclick || el.getAttribute("jsaction"))) return true;
    return false;
  };

  const all = document.querySelectorAll("*");
  for (const el of all) {
    try {
      if (!isInteractive(el)) continue;
      if (!isVisible(el)) continue;
      const role = el.getAttribute("role") || "";
      el.setAttribute("data-loom-id", String(id));
      out.push({
        id,
        tag: el.tagName.toLowerCase(),
        role,
        kind: kindFor(el, role),
        label: labelFor(el),
        value: (el.value != null ? String(el.value) : "").slice(0, 60),
      });
      id++;
    } catch (e) { /* skip problematic node */ }
  }
  return out;
}
"""


# JS: extract the page's VISIBLE TEXT content — what a human reads
# (prices, listings, results, headings). Unlike observe (which lists
# CLICKABLE elements), this is for READING outcomes. Walks visible text
# nodes, collapses whitespace, dedupes, and keeps meaningful lines.
_READ_TEXT_JS = r"""
() => {
  const isVisible = (el) => {
    if (!el) return false;
    const s = window.getComputedStyle(el);
    if (s.display === "none" || s.visibility === "hidden" ||
        s.opacity === "0") return false;
    const r = el.getBoundingClientRect();
    return r.width > 1 && r.height > 1;
  };
  // Skip script/style/nav-chrome-heavy containers.
  const SKIP = new Set(["SCRIPT","STYLE","NOSCRIPT","SVG","PATH"]);
  const out = [];
  const seen = new Set();
  const walker = document.createTreeWalker(
    document.body, NodeFilter.SHOW_TEXT, null);
  let n;
  while ((n = walker.nextNode())) {
    const t = (n.nodeValue || "").replace(/\s+/g, " ").trim();
    if (!t || t.length < 2) continue;
    const p = n.parentElement;
    if (!p || SKIP.has(p.tagName)) continue;
    if (!isVisible(p)) continue;
    if (seen.has(t)) continue;
    seen.add(t);
    out.push(t);
    if (out.length > 400) break;  // safety cap
  }
  return out;
}
"""


async def read_text(page: Any) -> str:
    """Return the page's visible text content (for reading prices /
    results / listings). Compact + capped so it stays affordable."""
    try:
        lines: list[str] = await page.evaluate(_READ_TEXT_JS)
    except Exception as exc:  # noqa: BLE001
        return f"(could not read page text: {exc})"
    title = url = ""
    try:
        title = await page.title()
        url = page.url
    except Exception:  # noqa: BLE001
        pass
    body = "\n".join(lines)
    # Hard char cap so a huge results page can't blow the budget.
    if len(body) > 8000:
        body = body[:8000] + "\n… (truncated — ask to read more / scroll)"
    return f"PAGE: {title}\nURL: {url}\n\n{body}"


async def observe(page: Any) -> tuple[list[dict[str, Any]], str]:
    """Tag + collect interactive elements; return (elements, rendered).

    ``rendered`` is the compact text the LLM reads. ``elements`` is the
    structured list (also drives the selector map / id validity)."""
    try:
        elements: list[dict[str, Any]] = await page.evaluate(_COLLECT_JS)
    except Exception as exc:  # noqa: BLE001 — never break the run
        return [], f"(could not read the page: {exc})"

    title = ""
    url = ""
    try:
        title = await page.title()
        url = page.url
    except Exception:  # noqa: BLE001
        pass

    lines: list[str] = [f"PAGE: {title}", f"URL: {url}", ""]
    if not elements:
        lines.append("(no interactive elements found — the page may still "
                     "be loading; try page_observe again, or page_check to "
                     "see it visually.)")
        return elements, "\n".join(lines)

    # COST CONTROL: a page can have 800+ interactive elements (nav, footer
    # links, ad chrome) — re-sending all of them every turn is the token
    # blowup. Prioritise the elements that matter (inputs/comboboxes +
    # buttons with real labels) and cap the rendered list. The full
    # ``elements`` list (with ids) is still returned for acting; we only
    # trim what the LLM READS.
    PRIORITY = {"input": 0, "select": 1, "button": 2, "link": 3}

    def _score(e: dict[str, Any]) -> tuple[int, int]:
        kind = e.get("kind", "link")
        has_label = 0 if (e.get("label") or "").strip() else 1
        return (PRIORITY.get(kind, 9), has_label)

    ranked = sorted(elements, key=_score)
    CAP = 60
    shown = ranked[:CAP]
    # Keep them in id order within the shown set so the list reads
    # naturally.
    shown.sort(key=lambda e: e["id"])
    for e in shown:
        label = e.get("label") or "(no label)"
        val = e.get("value") or ""
        val_part = f'  (value="{val}")' if val else ""
        lines.append(f'[{e["id"]}] {e["kind"]} "{label}"{val_part}')
    if len(elements) > len(shown):
        lines.append(
            f"… (+{len(elements) - len(shown)} more elements not shown — "
            "mostly nav/footer. If you need one that isn't listed, say which "
            "and I'll surface it.)"
        )
    return elements, "\n".join(lines)
