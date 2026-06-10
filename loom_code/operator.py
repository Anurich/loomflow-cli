"""Computer Operator mode for loom-code (the ``/computer`` command).

Where the default loom-code agent is a *software-engineering* team, the
Operator is a **computer-use agent**: it operates the user's whole
machine like a human — files, shell, the web browser, and media/apps —
under one "you are operating this computer" prompt + an approval gate on
irreversible real-world actions.

Design (see also the project memory ``/computer = full computer
control``):

* The Operator is a ``Team.supervisor`` whose **coordinator holds the
  action tools directly** (not buried on a delegate worker). This is the
  fix for the original bug where browser tools sat on the coder and the
  coordinator that talks to the user couldn't reach them. Workers exist
  for genuinely parallel sub-tasks; the coordinator itself can act.

* Capabilities are layered:
    - Tier 0 (reuse): read / write / edit / bash / grep / ls / find /
      web_fetch — loom-code already has these.
    - Tier 1: browser control via the Playwright MCP server (visible
      Chromium) — wired by the REPL's ``/computer`` handler as a
      built-in MCP server, composed in via ``McpAugmentedHost``.
    - Tier 2 (this module): native media + app control — open apps,
      play/pause music, volume, notifications, timers. macOS first
      (``osascript`` / ``open``); per-OS dispatch with stubs elsewhere.

* Safety: the operator prompt forbids irreversible actions (purchase,
  delete-outside-workspace, send/post) without explicit confirmation,
  and those route through the same ``approval_handler`` the coding agent
  uses. The browser runs HEADED so the user watches and can interrupt.
"""

from __future__ import annotations

import asyncio
import platform
import shutil

from loomflow import tool
from loomflow.tools.registry import Tool

# ---------------------------------------------------------------------------
# Tier 2 — native media + app control tools.
#
# These wrap OS-native commands. macOS is the primary target (osascript +
# open); Windows/Linux get best-effort fallbacks so the tool exists
# everywhere but degrades with a clear message where unsupported.
# ---------------------------------------------------------------------------

_OS = platform.system()  # "Darwin" | "Windows" | "Linux"


async def _run(
    cmd: list[str],
    timeout: float = 20.0,  # noqa: ASYNC109 — applied via wait_for below
) -> tuple[int, str, str]:
    """Run a command, return (rc, stdout, stderr). Never raises."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (
            proc.returncode or 0,
            out.decode("utf-8", "replace").strip(),
            err.decode("utf-8", "replace").strip(),
        )
    except (TimeoutError, OSError) as exc:
        return 1, "", str(exc)


async def _osascript(script: str) -> tuple[int, str, str]:
    return await _run(["osascript", "-e", script])


def _open_app_tool() -> Tool:
    async def open_app(name: str) -> str:
        """Launch / focus a desktop application by name."""
        name = name.strip()
        if not name:
            return "error: no app name given"
        if _OS == "Darwin":
            rc, _out, err = await _run(["open", "-a", name])
            if rc == 0:
                return f"opened {name}"
            return f"could not open {name}: {err}"
        if _OS == "Windows":
            rc, _out, err = await _run(["cmd", "/c", "start", "", name])
            if rc == 0:
                return f"opened {name}"
            return f"could not open {name}: {err}"
        # Linux: try the binary name directly, then xdg-open.
        if shutil.which(name):
            rc, _o, err = await _run([name])
        elif shutil.which("xdg-open"):
            rc, _o, err = await _run(["xdg-open", name])
        else:
            return f"could not open {name}: no launcher found on this OS"
        return f"opened {name}" if rc == 0 else f"could not open {name}: {err}"

    return tool(
        name="open_app",
        description=(
            "Launch or focus a desktop application by name (e.g. 'Spotify', "
            "'Safari', 'Notes', 'Calculator'). Use when the user wants to "
            "open or switch to an app. Arg: name (the app's display name)."
        ),
    )(open_app)


def _media_control_tool() -> Tool:
    async def media_control(
        action: str, amount: int = 0, query: str = ""
    ) -> str:
        """Control media playback / system volume."""
        action = action.strip().lower()
        if _OS != "Darwin":
            return (
                f"media_control is macOS-only for now (this is {_OS}). "
                "Ask me to use the browser or an app instead."
            )
        # System media keys via AppleScript. Works for the active player
        # (Music/Spotify/browser media) on macOS.
        # Play a SPECIFIC artist/song/playlist in Apple Music, if asked.
        if action in ("play", "playpause", "toggle") and query.strip():
            q = query.replace('"', '\\"')
            # Open Music, search the catalog/library for the query, and
            # play the first matching track. Uses Music's AppleScript
            # play-by-search; falls back to a library artist filter.
            script = (
                'tell application "Music"\n'
                '  activate\n'
                '  try\n'
                f'    play (every track whose artist contains "{q}")\n'
                '  on error\n'
                '    try\n'
                f'      play (every track whose name contains "{q}")\n'
                '    end try\n'
                '  end try\n'
                'end tell'
            )
            rc, _o, err = await _osascript(script)
            if rc == 0:
                return f'playing "{query}" in Apple Music'
            return (
                f'could not play "{query}" from your Music library '
                f"({err or 'not found'}). It may not be in your library — "
                "open Apple Music and search, or use the browser "
                "(YouTube Music) instead."
            )
        if action in ("play", "pause", "playpause", "toggle"):
            rc, _o, err = await _osascript(
                'tell application "System Events" to key code 16 using {}'
            )  # F-key media play/pause (key code 16 = playpause on most)
            # Fallback to Music/Spotify direct control if key code is a no-op.
            if rc != 0:
                await _osascript(
                    'tell application "Spotify" to playpause'
                )
            return f"media: {action}"
        if action in ("next", "skip"):
            await _osascript('tell application "Spotify" to next track')
            return "media: next track"
        if action in ("previous", "prev", "back"):
            await _osascript('tell application "Spotify" to previous track')
            return "media: previous track"
        if action in ("volume", "setvolume"):
            vol = max(0, min(100, amount))
            rc, _o, err = await _osascript(f"set volume output volume {vol}")
            return (
                f"volume set to {vol}" if rc == 0 else f"volume failed: {err}"
            )
        if action in ("mute",):
            await _osascript("set volume with output muted")
            return "muted"
        if action in ("unmute",):
            await _osascript("set volume without output muted")
            return "unmuted"
        return (
            f"unknown media action '{action}'. Use: play | pause | next | "
            "previous | volume (with amount 0-100) | mute | unmute."
        )

    return tool(
        name="media_control",
        description=(
            "Control music / media playback + system volume (macOS). "
            "Actions: play, pause, next, previous, volume (amount 0-100), "
            "mute, unmute. To play a SPECIFIC artist/song, use "
            "action='play' WITH query='Taylor Swift' — it searches your "
            "Apple Music library and plays the match (if it's not in your "
            "library, use the browser / YouTube Music instead). Args: "
            "action; amount (for volume); query (artist/song to play)."
        ),
    )(media_control)


def _notify_tool() -> Tool:
    async def notify(message: str, title: str = "loom-code") -> str:
        """Show a desktop notification."""
        message = message.strip()
        if not message:
            return "error: empty message"
        if _OS == "Darwin":
            safe_msg = message.replace('"', '\\"')
            safe_title = title.replace('"', '\\"')
            await _osascript(
                f'display notification "{safe_msg}" with title "{safe_title}"'
            )
            return "notification shown"
        if _OS == "Linux" and shutil.which("notify-send"):
            await _run(["notify-send", title, message])
            return "notification shown"
        return f"notifications not supported on {_OS}"

    return tool(
        name="notify",
        description=(
            "Show a desktop notification to the user. Use to surface a "
            "result, a reminder, or a 'done' signal. Args: message; title "
            "(optional)."
        ),
    )(notify)


def _reveal_tool() -> Tool:
    async def reveal_in_finder(path: str) -> str:
        """Open the file manager with the given file/folder highlighted."""
        from pathlib import Path as _P

        p = _P(path).expanduser()  # noqa: ASYNC240 — pure string math, no disk I/O
        # Resolve ~-relative + bare names against home so "Downloads/x"
        # works like a human means it.
        if not p.is_absolute():
            p = _P.home() / path
        target = str(p)
        if _OS == "Darwin":
            rc, _o, err = await _run(["open", "-R", target])
            if rc != 0:  # -R fails if path missing; open the parent dir
                await _run(["open", str(p.parent)])
            return f"revealed {target} in Finder"
        if _OS == "Windows":
            await _run(["explorer", "/select,", target])
            return f"revealed {target} in Explorer"
        # Linux: open the containing folder.
        if shutil.which("xdg-open"):
            await _run(["xdg-open", str(p.parent)])
            return f"opened {p.parent} in the file manager"
        return f"file manager reveal not supported on {_OS}"

    return tool(
        name="reveal_in_finder",
        description=(
            "Open the file manager (Finder/Explorer) with a file or folder "
            "highlighted, so the user SEES it. Use after creating or "
            "changing a file the user will want to look at (e.g. after "
            "writing ~/Downloads/test.py, reveal it). Arg: path."
        ),
    )(reveal_in_finder)


def media_app_tools() -> list[Tool]:
    """The Tier 2 native media + app tools for the Operator."""
    return [
        _open_app_tool(),
        _media_control_tool(),
        _notify_tool(),
        _reveal_tool(),
    ]


# ---------------------------------------------------------------------------
# Operator system prompt.
# ---------------------------------------------------------------------------

OPERATOR_PROMPT = """\
You are loom-code in COMPUTER OPERATOR mode. You operate the user's
computer for them like a capable human assistant at the keyboard — using
whatever tool fits each step:

- Web tasks → the browser tools. ALWAYS page_observe to see the page
  before acting; act on the [ids] from the LATEST observe; re-observe
  after navigation. When DOM text isn't enough (reading prices/results,
  understanding layout, or you're about to say "nothing's there"),
  page_look — it SCREENSHOTS the page and a vision model tells you what's
  actually on screen. Reach for page_look BEFORE giving up on finding
  info; seeing beats guessing.
- Files / folders → read/write/edit/ls/find. These reach the user's
  WHOLE machine (rooted at the home dir) — Downloads, Documents, Desktop,
  anywhere. "create test.py in Downloads" → write ~/Downloads/test.py.
  After CREATING a file the user will want to look at, reveal_in_finder
  it so they SEE it pop up (writing is silent otherwise).
- System / programs → bash (the user's real shell — use it freely for
  system tasks), open apps with open_app, control music with
  media_control (action='play', query='<artist>'), surface results with
  notify.

You CAN operate this computer. Don't refuse system tasks with "I don't
have permission" — you have a real shell and home-rooted file access. For
e.g. "check for system updates" run bash `softwareupdate -l`; "what's
running" → `ps aux`; "free disk" → `df -h`. Actually DO the task with the
tools; only the approval gate (which the user answers) can stop a
destructive action. The one hard line is the SAFETY list below
(purchases / mass-delete / sending on the user's behalf) — confirm first.

How to work:
- Break the request into small, observable steps and narrate each ("I'm
  opening the flights site… I see the origin field… typing Delhi…").
- Observe before you act (snapshot the page, ls the folder) so you act on
  reality, not assumption.
- Prefer the most direct tool: don't write a script to do what a browser
  click or an app launch does. NEVER use the bash tool to print an excuse
  or a status message — if a tool fails, say so in plain text and try a
  different approach.

BROWSER TOOLS — use page_open / page_observe / page_act / page_check /
page_press / page_back. Element [ids] from page_observe are STABLE (they
ride the DOM), but the page CONTENT changes, so:

UNDERSTAND THE PAGE BEFORE ACTING. Do not type into a field until you
know what it IS. page_observe lists every element as:
    [15] input "Where from?" (value="…")
    [17] input "Where to?"
READ the labels. Match the RIGHT field to the RIGHT value — e.g. origin
goes in "Where from?", destination in "Where to?". If a label is missing
or unclear, do NOT guess: pick the most likely one, fill it, then VERIFY.

Choosing the right fill tool:
  - AUTOCOMPLETE / combobox field (shows a suggestion dropdown as you
    type — flight origin/destination, Google Maps, address, "search with
    suggestions"): use page_fill(id, value). It types, waits for the
    dropdown, and selects the match so the value COMMITS. Plain page_act
    "type" REVERTS on these (e.g. origin snaps back to "Kathmandu") — so
    do NOT use page_act type for fields with suggestions.
  - Plain text input / textarea (no dropdown): page_act(id, "type", val).
  - DATE / calendar field: page_set_date(id, "2026-06-09"). NEVER click
    day cells by guessing ids with page_act — that flails. page_set_date
    opens the calendar and clicks the right day for you.

ONE FIELD AT A TIME. Never call two fill tools in the same step — fill
ONE field, wait for its result, page_check it, THEN do the next. Filling
origin and destination together races and both fail.

The required loop for EACH field, done sequentially:
  1. page_observe — read labels, pick the field by its label.
  2. Fill it with the RIGHT tool (page_fill for autocomplete, else
     page_act type). Match the value to the field's purpose: origin →
     "Where from?", destination → "Where to?". page_fill clicks the field
     to open it, types into the real input, and picks the suggestion — so
     give it the field's id and let it do the whole dance.
  3. page_check("is <value> set as <the field's purpose>?") — returns the
     field's REAL current value. Confirm it matches BEFORE the next field.
     Sites often pre-fill origin with your location — so always set BOTH
     origin and destination explicitly and verify each.
  4. If page_check shows the wrong value, page_observe again (the widget
     may have changed the ids) and redo step 2 with page_fill.

SUBMIT correctly — this is the #1 reason results don't appear. Pressing
Enter on a field usually does NOT load results; you must CLICK the real
Search button. The exact sequence after filling fields + dates:
  1. The date calendar is probably still open. Dismiss it: find a "Done"
     button in page_observe and page_act click it (or page_press Escape).
  2. page_observe again — now the "Search" button is visible.
  3. page_act CLICK the button literally labeled "Search" (or "Search
     flights"/"Submit"/"Go"). Do NOT use press_enter for this.
  4. Wait, then page_read / page_scroll + page_read to get the RESULTS
     LIST (airlines, times, per-flight prices).
If after a Search click you still only see a price CALENDAR / a "from
$X" graph (not a list of individual flights), you're on the explore view
— look for and click a "Search" or "Done" that switches to the results
list. The calendar's "from <price>" is the summary; the LIST has the
airlines. Optional fields (return date, filters) aren't required.

CLOSE pop-ups/overlays before reading. After picking dates a calendar
dialog stays open with a "Done" button — page_observe, click "Done" (or
the primary confirm), THEN the results show. A lingering date dialog is
the usual reason results "don't appear".

READ the results — this is how you ANSWER the user. After submitting/
confirming, call page_read to get the page's actual text. page_observe
only lists CLICKABLE elements, so a results page looks "empty" there even
when full of prices — NEVER conclude from page_observe; use page_read.

CRITICAL — believe what page_read shows. If page_read contains numbers
that look like prices (e.g. "120K", "99K", "$612", "NPR 120,421", a
"from <price>" line), those ARE the fares — REPORT THEM. Do NOT invent
excuses like "fares aren't released yet" / "too far in advance" / "no
detailed listings this far ahead" — those are FALSE; flights are
bookable a year out. If you see prices, the data exists; dig into it.

Get the FULL listings (names + details), not just the summary price. On a
flight/shop search the first view is often a price CALENDAR or a "from
$X" teaser — that is NOT the results list. To reach the actual listings
(airline names, times, per-option prices):
  1. Close any open calendar/dialog (click Done), then page_observe and
     click the main "Search" button (or the selected date's "Done").
  2. Wait, then page_scroll("down") and page_read — repeat scrolling +
     reading until you've captured the individual options (e.g.
     "British Airways · 7h 30m · NPR 121,000").
  3. Report the concrete cheapest option WITH its airline/details.
Only say a detail is unavailable if it's genuinely absent after you
reached the results list and scroll-read it.

WHEN IN DOUBT, LOOK. If page_read/page_observe seem to miss content, or
you're tempted to say "nothing's there" / "not available", call
page_look first — it screenshots the page (with the [id] boxes) and a
vision model tells you what's actually on screen. SEEING beats guessing;
use it to read prices/results off a visual page before any negative
conclusion.

Recovery:
- "element no longer on the page" → page_observe to get current ids.
- a click is blocked / times out → page_press("Escape") to dismiss an
  overlay, page_observe, retry.
- a date/calendar widget is fiddly → SKIP it (search with default dates)
  unless the user asked for specific dates.
- If a complex site keeps fighting after ~3 corrected attempts, switch
  approaches: open a plain web search ("flights London to New York
  price"), or Kayak, and read the results. Getting the user the answer
  beats wrestling one stubborn page.

SAFETY — never do anything irreversible without the user explicitly
confirming first:
- purchases / payments (Buy, Pay, Place order, Confirm, checkout),
- deleting or overwriting files the user didn't ask you to,
- sending messages / emails / posts on the user's behalf.
For these, stop and ask: "Ready to <action>? Confirm and I'll proceed."

Be transparent: report what you see and what you did at every step.
"""


def build_operator_prompt() -> str:
    """The operator prompt with TODAY'S date prepended, so relative dates
    ("tomorrow", "next week", "in 3 days") resolve correctly. Without
    this the model guesses the date (it picked 2024 for "tomorrow")."""
    import datetime as _dt

    today = _dt.date.today()
    header = (
        f"Today's date is {today:%A, %B %d, %Y} ({today:%Y-%m-%d}). "
        "Resolve any relative dates the user gives — 'tomorrow', 'next "
        "Friday', 'in 3 days' — against THIS date, and pass concrete "
        "YYYY-MM-DD dates to page_set_date.\n\n"
    )
    return header + OPERATOR_PROMPT
