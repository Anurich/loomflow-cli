# TUI integration checklist (branch: tui-bottom-input)

Goal: full-screen pinned-bottom chat box is the **default**;
`--classic` restores the inline box. Must work **exactly** like the
inline box for every surface (esp. `/` completion). Full manual sweep
before merge.

## Proven (committed on this branch)
- `loom_code/tui.py` — `ChatTUI`: continuous full-screen app, scroll
  pane fed by a redirected Rich console, fixed bottom box that grows
  as text wraps, status-line slot, in-app `select()` (filter-gated
  ↑/↓+Enter+hotkeys), `attach_completer()` + CompletionsMenu float so
  `/` pops the command menu. All verified in PTY.
- `loom_code/approval.py` — `ApprovalGate.select_hook`: when set, the
  Yes/All/No prompt routes through `tui.select` instead of the
  raw-termios worker-thread selector. Compiles.

## Remaining wiring (do next, carefully — each is a touch point where
## parity can break; test after each)

1. **cli.py**: add `--classic` flag (store_true). Thread `classic` →
   `run_repl(..., classic=classic)`. Default False (TUI on).
   Also force classic when `not sys.stdout.isatty()` (piped/CI) and
   for `--output-format json`.

2. **repl.py `run_repl`**: accept `classic: bool=False`; pass to
   `Repl(..., use_tui=not classic)`.

3. **repl.py `Repl.__init__`** (the real one, not `_SlashCompleter`):
   - accept `use_tui: bool`.
   - if use_tui and stdout.isatty(): construct `self._tui = ChatTUI()`,
     else `self._tui = None`.
   - if self._tui: `import loom_code.render as _r; _r.console =
     self._tui.console` — the ONE redirect that makes all output flow
     to the pane. (Keep a ref to restore on exit.)
   - after building the completer: `self._tui.attach_completer(
     self._prompt_session.completer)`.
   - `self._gate.select_hook = self._tui.select` when tui active.

4. **repl.py `run()`**: wrap the loop body in the tui session:
   ```
   if self._tui:
       async with self._tui.session():
           return await self._run_inner()
   else:
       return await self._run_inner()
   ```
   (keep the MCP/browser/bg teardown in the existing finally.)

5. **repl.py `_read_line`**: if self._tui → `return await
   self._tui.read_line()` (raises EOFError on empty Ctrl-D, matching
   the existing except-EOFError branch). Else the current box.

6. **repl.py `_select_menu`** (the /set_model etc picker): if
   self._tui → `key = await self._tui.select(title, options,
   default=default); return None if key == "\x00cancel" else key` —
   reuse the existing option list (already includes a Cancel row).
   Else the current `_select_option` thread path.

7. **Spinner → status line**: `set_status(label)` →
   `self._tui.set_status(label)`; `pause_status()` →
   `self._tui.clear_status()`. Skip constructing the Rich
   `console.status` when tui active (it can't render in the pane).
   The renderer already calls these callbacks — just point them at
   the tui when present.

8. **Flush**: after each `console.print` batch the tui needs a
   `self._tui.flush()`. Simplest: wrap the redirected console so
   `print` auto-flushes, OR call `flush()` in the renderer's
   chunk/`_end_text` paths + after each command handler. Prefer a
   thin `_TuiConsole(Console)` subclass whose `print` calls super then
   `tui.flush()` — one place, no scattered flush calls.

9. **Teardown**: on exit restore `render.console` to the original;
   `session()` already exits the app.

## Full manual sweep before merge (the bar the user set)
- [ ] `/` pops the completion menu; picking one fills the box
- [ ] plain task streams into the pane; box stays pinned
- [ ] an approval-gated edit shows the diff + the in-app Yes/All/No
- [ ] `/set_model` provider+model menus navigate in-app
- [ ] `/context`, `/cost`, `/fork`, `/tree`, `/help` render right
- [ ] `!bash` inline output
- [ ] paste a multi-line blob (paste keybindings)
- [ ] `/resume`, Ctrl-C twice, Ctrl-D exit
- [ ] `--classic` → old inline box unchanged
- [ ] non-TTY (piped) → falls back to classic, no crash
- [ ] full `pytest` green

Delete this file before merge.
