# loom-code — Design

loom-code is a CLI coding agent built **on top of** the released
`loomflow` framework. loomflow is consumed as a library; loom-code never
modifies it. Everything here is composition: prompts, tool scoping, a
worker roster, and persistence wiring around `loomflow.Agent` /
`Team.supervisor`.

Two surfaces consume the same agent:
- the **terminal REPL** (`repl.py`, `cli.py`, `render.py`), and
- the **loomflow-desktop sidecar**, which imports `build_agent` directly.

So the agent's behaviour is defined once, in this package, and shared.

---

## The agent topology

`build_agent` (`loom_code/agent.py`) returns `(coordinator, workspace)`.
The coordinator is a single `Team.supervisor` Agent whose job is to
**plan and delegate** — it is deliberately **read-only**.

```
coordinator (Team.supervisor, READ-ONLY)
  tools: read · grep · find · ls · web_fetch · plan_write/plan_read
         · notebook (search_notes/read_note/…) · recall_past_plans
  delegates to workers via `delegate` / `send_message`
        │
        ├── coder      — the ONLY writer (read/write/edit/multi_edit/grep/find/ls/bash)
        ├── explorer   — read-only investigator → returns a briefing
        ├── auditor    — read-only defect/security/perf hunter
        └── reviewer   — independent verifier (re-reads, runs tests, pass/fail)
```

The worker roster is built by `build_workers` (`loom_code/workers.py`).
All four run on the **same model** as the coordinator — the specialism
is in the prompt + tool scoping, not a weaker model. Only `coder` and
`reviewer` get a permissions policy + approval handler (they hold
destructive tools); `explorer` and `auditor` are purely read-only.

### Why the coordinator is read-only

An earlier "unified coordinator" had the full tool kernel and, in
practice, **ground**: it used its own `edit`/`bash` and rarely
delegated. Making the coordinator structurally read-only (no writer
tools at all) forces the intended pattern — it *must* hand every change
to `coder` and every verification to `reviewer`. This is a structural
guarantee, not a prompt nudge.

### Why one writer

`coder` is the sole writer so two delegations can never race on the
filesystem in the same turn. It also holds `bash`, so it can check a
real installed API or run tests itself while implementing.

---

## Planning model

`living_plan=True` wires `plan_write(goal, steps)` / `plan_read()` onto
the coordinator. The plan is the coordinator's durable working memory
and the user's view of progress. It mirrors to the workspace notebook
(`kind="plan"` notes) so it survives across runs.

The coordinator prompt (`prompts.py`, `_UNIFIED_COORDINATOR`) drives the
lifecycle:

1. **RESUME before RESTART.** For any task that could continue earlier
   work, the first actions are `recall_past_plans` + `search_notes` —
   recall the prior plan/findings instead of re-investigating.
2. **Decide by GOAL MATCH, not topic overlap.** Recall surfaces plans
   from the same project, so a hit is *not* automatically a
   continuation: resume only when the prior plan's **goal** matches the
   request (and verify it still holds via `reviewer` — never report a
   stale result); for a **different** goal, write a **fresh** plan
   rather than riding the old one.
3. **PLAN FIRST** for real work — an OUTCOME-level plan shaped
   `INVESTIGATE → IMPLEMENT → VERIFY`, last step always a `reviewer`
   verification. Trivial questions/greetings get no plan.
4. **Work one step at a time, recording findings** into the plan so a
   long run can't forget what's done or what was discovered.

A bounded stop-hook ("Ralph") loop re-prompts while the plan still has
`todo`/`doing` steps; `max_stop_hook_iterations` is kept low to avoid
re-planning spins.

---

## Persistence (all under `<project>/.loom/`)

| Concern    | Backing store                  | Notes |
|------------|--------------------------------|-------|
| Episodes / facts | `sqlite:.loom/memory.db`  | multi-tenant by `user_id`; recall is cross-session |
| Notebook   | `.loom/notebook` (`LocalDiskWorkspace`) | shared notes; bridges workers (each runs in a fresh session) |
| Living plan | mirrors into the notebook     | recallable via `recall_past_plans` |
| Repo map   | `.loom/repomap.md` (mirror)    | deterministic, LLM-free (see below) |

---

## Repo map (`loom_code/loominit/repomap.py`)

A deterministic, **LLM-free** structural overview — an Aider-style
ranked symbol list grouped by file, scored by structural importance and
capped by a token budget. It's injected into the agent's `loom_index`
working block each turn (the structural index is fresh-by-construction:
re-walked only when the source tree changes). Being deterministic, it's
cache-friendly and cheap to regenerate; the desktop sidecar can layer an
LLM consolidation on top for very large repos.

---

## Tools & robustness

- `edit_tool.py` — `multi_edit` with **whitespace-flexible matching**:
  when an exact `old_string` isn't found, it retries against a uniquely
  matching block ignoring per-line leading/trailing whitespace and
  re-indents the replacement. Fixes recurring "old_string not found"
  failures from indentation drift.
- `grep_tool.py`, `web_fetch.py` — project-rooted, read-only. `web_fetch`
  rewrites GitHub blob URLs to raw and steers repo-root URLs toward the
  contents API instead of dumping a full HTML page into context.
- `compact.py` — conversation compaction helpers.

---

## Extensions

- **Skills** (`loom_code/skills/`, via loomflow's `SkillRegistry`) —
  markdown + optional tools, lazy-loaded on `load_skill`.
- **Hooks** (`hooks.py`) — REPL-lifecycle hooks (SessionStart,
  UserPromptSubmit) fired by the REPL/sidecar from a trust-filtered
  bundle; loomflow has no native hook point for these.
- **Trust** (`trust.py`) — project-hook trust gating before any
  project-authored hook runs.
- **Custom subagents** — user-authored specs become extra delegate
  roles; the builtin names (`coder`/`explorer`/`auditor`/`reviewer`) are
  protected from being shadowed, and a spec with no declared tools
  defaults to a read-only kernel.

---

## Hard constraints

- **Never modify the `loomflow` framework.** loom-code is strictly a
  consumer; all behaviour lives here. Requires `loomflow>=0.10.22`.
- **Prefer `build_agent` over a raw `loomflow.Agent`** so every surface
  (REPL, desktop) gets the same topology, prompts, and persistence.
