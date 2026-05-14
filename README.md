# loom-code

A terminal coding agent built on [loomflow](https://github.com/Anurich/LoomFlow).

loom-code is a thin terminal shell — the brain is loomflow. The CLI
detects your project, builds a loomflow `Agent`, streams the run to
the terminal, and gates destructive tool calls behind an approval
prompt. Everything load-bearing — the agent loop, tools, planning,
memory, the self-improvement notebook — is loomflow.

## What it does

- **Plans before it codes.** Every task gets a living plan
  (`plan_write` / `plan_read`) — TodoWrite-style, visible, hard to
  drift from.
- **A 7-tool kernel.** `read` / `write` / `edit` / `grep` / `find` /
  `ls` / `bash`, all scoped to the project root.
- **Specialist sub-agents on demand.** The main ReAct loop can call
  `explore` (read-only investigation) and `review` (independent
  verification) as tools — one coherent main thread, specialists
  when they earn their keep.
- **Asks before destructive changes.** Writes, edits, and shell
  commands route through an approval gate with a unified-diff
  preview and an allow-all-session option.
- **Gets sharper at your repo.** A per-project notebook
  (`.loom/notebook`) plus episode memory (`.loom/memory.db`) —
  notes the agent reads get credited when a turn goes well, so
  future runs surface what worked.

## Usage

```bash
# one-shot
loom-code "add a retry decorator to the http client"

# interactive REPL
loom-code
```

`loom-code` with no args drops into a REPL with slash commands
(`/help`, `/plan`, `/cost`, `/good`, `/bad`, `/model`, `/clear`,
`/exit`).

## Install (development)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in your environment
(or a local `.env` — which is gitignored).

## Project context

loom-code reads a `LOOM.md` / `CLAUDE.md` / `AGENTS.md` /
`.loom/context.md` file at the project root, if present, and
treats it as binding house rules.
