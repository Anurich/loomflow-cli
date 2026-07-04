# loom-code — Harness Gap Analysis & Roadmap to Best-in-Class

> **Audience:** an AI agent (or engineer) implementing improvements to
> loom-code. Every gap below has: the evidence, what the best harness
> does, and a concrete implementation spec (files to touch, shape of
> the change). Sourced from a mid-2026 multi-source research pass
> (Claude Code docs, pi/badlogic + earendil-works, opencode, aider,
> codex CLI, goose, cline, amp; Terminal-Bench 2.x & SWE-bench
> analyses; user pain-point threads) cross-checked against the actual
> loom-code codebase.
>
> **How to read priorities:** P0 = benchmark- or trust-critical, do
> first. P1 = strong user-demand differentiator. P2 = polish / parity.
> Each gap notes whether the evidence says it *moves benchmark scores*
> vs *is a UX/adoption differentiator* — they are not the same lever.

---

## 0. The single most important finding

Multiple independent 2026 sources converge on one conclusion that
should shape everything:

> **Harness design — not model choice — dominates within a fixed
> model, and the gains live in TOOLS, MIDDLEWARE, and MEMORY, not in
> the system prompt.**

Evidence:
- The same LLM's SWE-bench score varied **42% → 78%** purely from
  scaffolding changes; scaffolds spread **22+ points** on SWE-bench
  Pro with an identical model.
- LangChain's `deepagents-cli` gained **+13.7 points** on
  Terminal-Bench 2.0 (52.8 → 66.5) with the model held constant
  (GPT-5.2-Codex), purely from harness engineering.
- Automatic Harness Evolution raised pass@1 on Terminal-Bench 2 from
  **69.7 → 77.0**, beating the human-designed Codex harness (71.9),
  base model fixed. **Ablation: the gain lives in tools, middleware,
  and long-term memory — each carries it alone; a system-prompt-only
  change REGRESSED performance.**
- Edit-tool format alone is a massive lever: Grok Code Fast reportedly
  went **6.7% → 68.3%** by changing only its edit-tool format.
- Frontier models have **converged** on SWE-bench Verified (Opus 4.5
  80.9, Opus 4.6 80.8, Gemini 3.1 Pro 80.6, GPT-5.4 ~80.0, Sonnet 4.6
  79.6 — within ~1.3 pts). So model choice is nearly a wash at the
  top; **the harness is where the remaining points are.**
- Terminal-Bench 2.0 found **essentially no correlation** between
  turns/tokens and success — more turns ≠ better. And model swaps
  moved Codex CLI by 52 pts while scaffold swaps for a fixed model
  moved it only 17 pts (so pick a good model AND a good harness, but
  the harness deltas are real and stackable).

**Implication for loom-code:** invest in the *tool kernel quality*,
*middleware* (loop detection, verify-before-done, context
observability), and *long-term memory* — loom-code already has the
memory seed, which the research singles out as unusually high-value.
Do NOT chase feature-count parity with Claude Code; chase the small
set of features the evidence says actually move the needle.

There is also a strong **counter-signal from pi** (Mario Zechner) and
Terminal-Bench's own Terminus 2: a *deliberately minimal* harness (4
tools, <1000-token prompt, no sub-agents, no MCP, no plan mode) is
**competitive with the most feature-rich commercial harnesses**. The
lesson is not "add everything" — it's "add only what pays for its
context cost." loom-code sits in the middle and should stay lean.

---

## 1. What loom-code already has (verified against the code)

So the report doesn't tell you to build things you have. Confirmed
present in the codebase today:

| Capability | Where | Notes |
|---|---|---|
| Tool kernel: read/write/edit/multi_edit/grep/find/ls/bash/web_fetch | `workers.py` | The 4-category (read/search/edit/execute) table-stakes set — **met**. |
| Living plan (= Claude Code TodoWrite) + Ralph auto-continue | `agent.py` (`living_plan=True`, StopHook) | The "planning" table-stakes — **met**. |
| Permission modes (default/accept-edits/plan/yolo) + allow/ask/deny globs | `permissions.py`, `approval.py` | Matches opencode/codex granularity — **met**. |
| OS bash sandbox | `sandboxed_bash.py` | Table-stakes — **met**. |
| Git worktree isolation (`/isolate`) + checkpoints + `/undo` | `worktree.py`, `checkpoint.py` | The #1 named scaffold differentiator (rollback) — **met**. |
| Per-project memory + self-improvement notebook (`/good` `/bad`) | loomflow memory | Research singles out long-term memory as top-tier lever — **met, and rare**. |
| MCP client (lazy connect) | `mcp_host.py` | Table-stakes — **met**. |
| Sub-agents (explore/review/auditor) | `workers.py` | Present but FIXED + BLOCKING (see gap 6). |
| Hooks: Pre/PostToolUse, Session*, UserPromptSubmit, Stop | `hooks.py`, `extensions.py` | Partial vs Claude Code's 8 events (see gap 4). |
| Extensions: user/project skills + custom agents + hooks, trust-gated | `extensions.py`, `trust.py` | Strong. Missing: custom slash commands + plugin bundles (gap 5). |
| LSP nav (jedi): go_to_definition/find_references/hover | `lsp_tools.py` | **Python only** (see gap 2). |
| codebase_search (semantic, optional embedder) | `workers.py` | Good — many competitors lack this. |
| JSON headless mode | `cli.py` (`--output-format json`) | Present; verify SDK-shape parity (gap 9). |
| Repo map injection | `loominit/repomap.py` | Matches aider's signature feature. |
| Session resume | `repl.py` `/resume` | **Linear only** — no branching (gap 1). |
| Per-turn cost display | `repl.py` | Differentiator vs Claude Code's opacity. |
| Goal mode (`/goal` until condition) | `repl.py` | Codex added stateful multi-day `/goal`; loom-code's is per-session (gap 7). |
| Any-model incl. free NVIDIA tier (litellm) | `credentials.py` | Strong adoption lever — the research says good harness helps cheap models MOST (+10.1 pp on deepseek-v4-flash). |
| Operator mode `/computer` (Playwright + vision) | `operator.py` | HIDDEN. Includes browser + screenshot vision (gap 3 is partly already solved here — expose it). |

**loom-code is already past table-stakes.** The gaps below are
differentiators and benchmark levers, not missing basics.

---

## 2. THE GAPS — ranked

### P0-A · Verify-before-done middleware (benchmark lever)

**Evidence.** A pre-completion verification gate (middleware that
intercepts the agent before it exits and forces a verification pass),
plus build-then-test self-verification loops, were named as harness
features with *measurable* Terminal-Bench gains. Separately: "error
recovery and rollback is the single biggest scaffolding
differentiator — high performers don't make fewer mistakes, they
recover from them." loom-code has rollback (`/undo`) but no
*forced verification before the turn ends*.

**What best-in-class does.** LangChain's deepagents added middleware
that blocks the agent from declaring done until a verification pass
(run tests / build) succeeds. loom-code already has the *hook point*
(StopHook / living_plan DONE-transition verification in loomflow
requires real `tool_call_id`s) — but it doesn't force a test/build
run before accepting completion.

**Implementation.**
- File: `loom_code/agent.py` (StopHook wiring) + a new
  `loom_code/verify_gate.py`.
- On a turn that claims completion (plan all-DONE, or a
  completion-claim in output — you already detect this via
  `_looks_like_completion_claim` in `repl.py`), if the project has a
  detectable test/build command (look for `pytest`, `npm test`,
  `cargo test`, `make`, a `justfile`, etc.), inject a
  system-reminder-style continuation: *"Before finishing, run the
  project's tests and report the result. If they fail, fix and
  re-run."*
- Make it opt-outable (`/verify off`) and skip when no test command is
  discoverable. Gate it behind the same living_plan verification you
  already trust so it can't loop forever (bounded by
  `max_stop_hook_iterations`).
- This composes with your existing anti-hallucination gate (the one
  that deletes unverified "done" episodes) — now instead of just
  *deleting* the false claim, you *prevent* it.

**Why P0:** directly benchmark-moving, and you already have 80% of the
machinery (completion detection + StopHook + checkpoints).

---

### P0-B · Loop / doom-loop detection middleware (benchmark lever)

**Evidence.** "Loop detection implemented as tool-call hooks tracking
per-file edit counts (LoopDetectionMiddleware) was one of the harness
features that improved Terminal-Bench 2.0 performance by preventing
agents from getting stuck in repeated-edit doom loops." Weak/cheap
models (loom-code's free-tier NVIDIA users) are the MOST prone to this
— and the research says good scaffolding helps weak models most.

**What best-in-class does.** Track (file, edit-count) and
(command, repeat-count) across the turn; when the same file is edited
N times or the same failing command re-run N times, interrupt with a
steering message: *"You've edited X three times without progress. Stop
and re-read the file / try a different approach."*

**Implementation.**
- File: new `loom_code/loop_guard.py`; wire as a PostToolUse hook in
  `hooks.py` (you already have the PostToolUse hook path).
- Keep a per-turn `Counter` keyed by `(tool, normalized_target)`
  (file path for edit/write; command string for bash). On threshold
  (default 3), emit a steering `additionalContext` that loomflow
  appends to the next model turn.
- Reset counters at turn start. Cheap, pure-Python, no model calls.
- Bonus: detect the Terminal-Bench #1 failure — "executable not found
  / not on PATH" (24.1% of all command failures). When a bash result
  contains `command not found` / `No such file`, append a hint:
  *"That binary isn't installed or on PATH. Check with `which` / your
  package manager before retrying."*

**Why P0:** cheap, benchmark-proven, and disproportionately helps the
free-tier weak-model users that are loom-code's differentiator.

---

### P0-C · Context observability (trust differentiator — the #1 pain point)

**Evidence (very strong, multi-source).**
- pi's author's **core critique** of mainstream harnesses (Claude Code
  included): "lack of context observability and control — they inject
  content into the model's context that is never surfaced in the UI,
  making it impossible to inspect every model interaction."
- Anthropic's own April 2026 postmortem: a Claude Code harness bug
  stripped thinking blocks every turn (quality regression); a silent
  default effort change (high→medium) shown as "high" in the UI became
  "a top trust-destroying pain point." Users demand
  **system-prompt changelogs and opt-out of harness changes** — "no
  major harness currently offers this."
- Kilo differentiates specifically on "no silent context compression,
  visible context window sizes"; "harness-config observability" is
  named an emerging differentiator.
- Users report Claude Code "forgets" CLAUDE.md and blame invisible
  config drift.

**This is loom-code's single biggest opportunity for positioning:**
be the harness you can *see into*. You already show per-turn cost —
extend that to full context transparency.

**Implementation.**
- A `/context` command (new handler in `repl.py`): print exactly what
  is in the model's context right now — system prompt size (tokens),
  each working block (repo map, learned notes, project rules,
  plan) with its token count, conversation history token count, and
  **% of the model's context window used**. loomflow exposes token
  counts via `count_tokens`; the working blocks are already named
  (`loom_index`, `learned_notes`, `project_rules`).
- A persistent **context-usage indicator** in the per-turn summary
  line you already render: `… · 34% ctx` next to the cost. This alone
  addresses a recurring complaint across Claude Code, cline, kilo.
- A `/prompt` command that dumps the FULL rendered system prompt to
  the terminal (or a file) — nobody else does this; it's a one-line
  win for the "show me everything" crowd.
- When auto-compaction fires, print *what* was compacted and the
  before/after token count (you have `_maybe_compact`; add a one-line
  visible report). Never compact silently.

**Why P0:** pure differentiator, cheap, and hits the loudest 2026
complaint. It's also *on-brand* — loom-code already leans "transparent"
(cost display, open source, free models).

---

### P1-1 · Conversation branching / session tree (pi's signature)

**Evidence.** pi stores sessions as a **tree** (JSONL, each entry has
`id`+`parentId`), with `/tree` navigation, `/fork`, `/clone`,
`--fork`, bookmarks, and export to HTML/gist. This enables "fix a
broken tool in a side branch without consuming context in the main
session." It's repeatedly cited as pi's key differentiator that
"loom-code's linear session resume lacks." loom-code's `/resume` is
linear only.

**What best-in-class does.** Any point in history can become a branch
point; the main thread stays clean while you explore a tangent, then
you either merge the learning back or discard the branch. Full history
preserved in one file even after lossy compaction.

**Implementation.**
- loomflow already keys rehydration on `session_id`. Add a parent
  pointer: when the user types `/fork`, mint a new `session_id`, record
  `{id, parent_id, forked_at_turn}` in `.loom/sessions.jsonl` (which
  you already maintain), and copy the parent's memory episodes up to
  the fork point into the new session (you have the sqlite episode
  table + the `_migrate_legacy_per_route_episodes` pattern for
  re-keying episodes — reuse it).
- `/tree` renders the `.loom/sessions.jsonl` parent-graph as an ASCII
  tree with turn counts + summaries (you already render a history
  preview in `_render_resumed_history_preview` — extend it).
- `/resume pick` already lists sessions; add tree indentation showing
  parentage.
- MVP: `/fork` (branch here), `/tree` (see the graph), resume any node.
  Merge-back can come later.

**Why P1:** it's the one thing pi fans point to that loom-code can't
do, it's genuinely useful (tangent without context pollution), and the
data model is a small extension of what you already persist.

---

### P1-2 · Custom slash commands from markdown files

**Evidence.** Claude Code, opencode, and codex all let users drop a
markdown file (with frontmatter) to define a reusable prompt/command.
Claude Code unified slash commands + skills; opencode puts custom
agents in `.opencode/agents/*.md`. loom-code has skills + custom
*agents* from files, but **not user-defined slash commands** (prompt
templates the user invokes with `/mycommand`).

**Implementation.**
- File: extend `extensions.py` discovery to read
  `~/.loom-code/commands/*.md` and `<repo>/.loom/commands/*.md`.
- Each file = one command: frontmatter `{description, model?,
  allowed-tools?}` + a markdown body that is the prompt template.
  Support `$ARGUMENTS` / `$1 $2` substitution and `!`bash-injection`
  (Claude Code semantics) so `/review-pr 123` expands.
- Register discovered commands into the `_COMMAND_DEFS` list (so they
  appear in `/help` + autocomplete — you already have one source of
  truth there) and dispatch in `_handle_slash` by loading the template,
  substituting args, and running it as a turn.

**Why P1:** high user demand, cheap (you have the discovery +
autocomplete machinery), and turns power users into contributors.

---

### P1-3 · Background / non-blocking execution (subagents AND bash)

**Evidence.**
- Claude Code (v2.1.201): **subagents run in the background by
  default**, main agent keeps working, notified on completion.
- opencode (v1.14.51): experimental background subagents, non-blocking.
- loom-code's explore/review sub-agents are **blocking**; there is **no
  background bash** (confirmed: no `run_in_background` path in
  `workers.py`/`repl.py`). pi deliberately omits this (says "use tmux")
  — so this is optional, but it's a real UX gap for long tasks (dev
  servers, long test runs, parallel research).

**Implementation (two parts, independent).**
- **Background bash:** add a `bash(..., background=True)` variant that
  spawns detached, returns a handle, and a `bash_output(handle)` /
  `bash_kill(handle)` tool pair (mirrors Claude Code). Store handles in
  a per-session dict. Lets the agent start `npm run dev` and keep
  working. File: `loom_code/workers.py` + a small process registry.
- **Async subagents:** loomflow's Supervisor already does parallel
  delegation; the blocker is the REPL awaits the whole turn. Lower-risk
  MVP: keep sub-agents synchronous but let the *user* fire a
  background task (`/bg <task>`) that runs a solo agent in a worker
  thread and notifies on completion — reuse your `anyio.to_thread`
  pattern from the selector.

**Why P1:** genuine capability gap for real dev loops; but scoped —
don't over-build. Background bash alone covers 80% of the need.

---

### P1-4 · Richer hook events + hook action types (parity + power)

**Evidence.** Claude Code exposes **8** lifecycle events
(PreToolUse, PostToolUse, Notification, Stop, SubagentStop, PreCompact,
PostCompact, UserPromptSubmit, SessionStart, SessionEnd) and hook
*actions* can be a shell command, **an HTTP request, an LLM prompt, or
a subagent** — plus `updatedInput` (PreToolUse can mutate tool args)
and `"async": true`. loom-code has 5-6 events and shell-command actions
only.

**Implementation.**
- File: `extensions.py` (`STOP_HOOK_EVENTS`, event set) + `hooks.py`
  (executor).
- Add `SubagentStop`, `PreCompact`, `PostCompact` events (you have the
  compaction lifecycle in `_maybe_compact` — fire hooks around it).
- Add `updatedInput` support: let a PreToolUse hook return JSON with
  `{updatedInput: {...}}` and apply it before the tool runs (your
  `_run` already parses hook stdout JSON — extend the outcome to carry
  a mutated-args field, and thread it through the tool-call path).
- `async: true` for fire-and-forget hooks (you already handle the
  stdin pipe race — a background hook is a natural extension).
- HTTP/LLM/subagent action types are lower priority; shell can shell
  out to `curl` already.

**Why P1:** the automation/power-user crowd lives here, and each
addition is small given your existing hook plumbing.

---

### P1-5 · Multi-language code intelligence (currently Python-only)

**Evidence.** LSP integration is repeatedly named table-stakes for
2026 ("multi-provider, MCP, memory/checkpointing, **LSP-aware code
intelligence**, sandboxed execution, skill systems, git-native
worktrees"). opencode auto-detects and configures language servers;
Claude Code ships per-language LSP plugins that surface type
errors after each edit. cline's *lack* of LSP is a cited cost
complaint (a GitHub issue frames LSP as a token-cost-efficiency
feature). loom-code's LSP is **jedi = Python only**.

**Implementation (staged).**
- MVP (cheap, big coverage gain): after each `edit`/`write`, run the
  project's own type/lint checker if present (`tsc --noEmit`,
  `pyright`, `cargo check`, `go vet`, `ruff`) and feed diagnostics back
  as a tool result — this is the "surface type errors after edit"
  behavior without implementing an LSP client. Reuses your bash tool +
  PostToolUse hook. File: `loom_code/diagnostics.py`.
- Full LSP (later): a real LSP client (via `pygls`/`multilspy`) giving
  go-to-def/references across TS/Go/Rust. Big; only if demand is real.
- Tree-sitter repo map: your `loominit/_ast_walk.py` is Python-AST;
  swapping in tree-sitter (grammars already ship via graphifyy dep)
  extends the repo map to all languages. Medium effort, high value —
  the repo map is your aider-parity feature and it's Python-blind today.

**Why P1:** it's named table-stakes, and the MVP (post-edit
type-check) is a small, high-value slice that also feeds the
verify-before-done gate (P0-A).

---

### P1-6 · Stateful, durable `/goal` (persist across sessions)

**Evidence.** Codex CLI (v0.128+) `/goal` "persists multi-day
workflows statefully across sessions, surviving restarts and context
compaction." loom-code's `/goal` runs until a condition within the
current session but isn't persisted — a restart loses it.

**Implementation.**
- Persist the active goal (condition text + progress notes) to
  `.loom/goal.json` on each turn; on SessionStart, if a goal is active,
  offer to resume it. Reuse your session-pointer persistence pattern
  (`_save_session_pointer`).
- Tie goal-progress to the living plan so "goal met" is verified by
  real tool calls, not a claim (composes with P0-A).

**Why P1:** modest effort, and "run overnight toward a goal" is
exactly the loom-code + free-model + `/isolate` story.

---

### P2-1 · TUI quality (scrollback, no-flicker, statusline, themes)

**Evidence.** pi's appeal is attributed *partly to implementation
quality* — "reliable, memory-efficient, does not flicker as a TUI."
opencode/amp differentiate on TUI polish. loom-code is line-based
(good — you deliberately avoided a full TUI after the box experiments
this session, correctly). This is a *don't-regress* item plus small
adds.

**Implementation (small, safe).**
- A statusline: model · mode · context% · cost, updated in place
  (you have the per-turn summary; a persistent bottom line is a small
  step, but beware the spinner-conflict lessons from this session —
  keep it print-based, not a Live widget).
- `/theme` for the markdown code theme (you hardcode `ansi_dark`) —
  trivial config.
- Do NOT build a full-screen TUI. The research validates line-based
  minimal harnesses (Terminus 2). Stay lean.

**Why P2:** polish, not a lever. Ship if idle, don't prioritize.

---

### P2-2 · Plugin/distribution layer + IDE/ACP bridge

**Evidence.** Claude Code has plugins (bundle skills+hooks+commands+MCP,
namespaced, marketplace-distributable) and IDE integration (VS Code +
JetBrains, diff overlay, selection bridge). opencode + goose speak
ACP (Zed's agent-client-protocol). loom-code has none.

**Implementation.** Defer. A plugin bundle is just a directory of your
existing extension types + a manifest — cheap when you want it, but low
demand for a young CLI. ACP is a real integration surface (lets Zed
drive loom-code) but large. Both are P2 — parity flexes, not levers.

---

## 3. Explicit NON-goals (the research says DON'T)

- **Don't bloat the system prompt.** Prompt-only changes *regressed*
  performance in the ablation; a 3% eval drop came from one Claude Code
  verbosity edit. Keep loom-code's prompt lean (pi runs <1000 tokens).
  Put behavior in tools/middleware/memory.
- **Don't add features for parity's sake.** pi + Terminus 2 prove
  minimal harnesses compete at the top. Every tool costs context. MCP
  servers alone can eat 7-9% of the window (Playwright's = 13.7k
  tokens). Your `McpAugmentedHost` lazy-connect is the right instinct —
  keep tools out of context until used (consider MCP "tool search"
  like Claude Code: descriptions only, load schemas on demand).
- **Don't chase "more turns / more tokens."** No correlation with
  success on Terminal-Bench. Chase *recovery* (checkpoints — have it)
  and *verification* (P0-A) instead.
- **Don't build a heavy multi-agent orchestration layer** to match
  Claude Code's "tens to hundreds of agents / 5-deep recursion." The
  evidence says sub-agent delegation is a *differentiator, not
  table-stakes* (only 5/13 agents have it), and adds complexity pi
  calls "a black box within a black box." Your explore/review is
  enough; make it async (P1-3) before making it bigger.

---

## 4. Recommended sequence

1. **P0-C context observability** (`/context`, ctx% in statusline,
   visible compaction, `/prompt`) — cheapest, biggest positioning win,
   on-brand.
2. **P0-B loop guard + missing-binary hint** — cheap, benchmark-proven,
   helps weak free-tier models most.
3. **P0-A verify-before-done gate** — benchmark lever, reuses your
   completion-detection + StopHook + checkpoints.
4. **P1-5 post-edit type-check (MVP LSP)** — feeds P0-A, closes the
   named table-stakes gap cheaply.
5. **P1-1 session tree (`/fork` + `/tree`)** — pi's signature, small
   data-model extension.
6. **P1-2 custom slash commands** — power-user demand, reuses discovery.
7. **P1-3 background bash** — real capability gap, scoped.
8. **P1-4 richer hooks / P1-6 durable goal** — power-user surface.
9. **P2** — TUI polish, plugins, ACP — only when idle.

Items 1-3 are days of work each and are where the measured wins and
the loudest complaints both live. Do them first.

---

## 5. Positioning takeaway

The market is bifurcating into **feature-maximal but opaque/expensive**
(Claude Code — users churning over rate limits, silent regressions,
CLAUDE.md amnesia, no config observability) and **minimal but
bare** (pi — no permissions, no plan mode, YOLO). loom-code can own the
middle with a sharp story the research directly supports:

> **The transparent, any-model coding agent.** See your whole context
> and cost. Run frontier or free models — good scaffolding helps the
> cheap ones most. Recover from anything (checkpoints, worktrees, session
> tree). Verify before it claims done. No lock-in, no silent changes,
> open source.

Every clause of that is backed by a verified finding above, and most of
it is 1-3 features away from what you already ship.
