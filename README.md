# loom-code

**A terminal coding agent built on [loomflow](https://github.com/Anurich/LoomFlow).**
Plans before it codes, asks before it breaks things, works with any model —
including free ones.

```
›  add a retry decorator to the http client

● loom
Added `retry(max_attempts=3, backoff=2.0)` to http/client.py and wired it
onto get() and post(). Tests pass (14/14).
───────────────────────────────────────────── 12,431 in · 217 out · $0.0043
```

loom-code is a thin terminal shell — the brain is loomflow. The CLI detects
your project, builds a loomflow `Agent`, streams the run to your terminal,
and gates destructive tool calls behind an approval prompt. Everything
load-bearing — the agent loop, tools, planning, memory — is loomflow.

## Highlights

- **Plans before it codes.** Every task gets a living plan
  (TodoWrite-style), visible as it progresses, hard to drift from.
- **Any model, including free ones.** OpenAI, Anthropic, NVIDIA's free
  NIM tier, local Ollama, or anything LiteLLM routes (Groq, Together,
  Azure, Bedrock, Vertex…). `/set_model` walks you through provider →
  API key → model with arrow-key menus.
- **Claude-Code-style permissions.** Reads are lenient, writes are
  strict. Every write/edit/shell command routes through an approval
  gate with a unified-diff preview. Allow/ask/deny rules, approval
  modes (`default` / `accept-edits` / `plan` / `yolo`), and an
  optional OS-level bash sandbox (`--sandbox`).
- **Specialist sub-agents on demand.** The main loop can call
  `explore` (read-only investigation) and `review` (independent
  verification) as tools — one coherent thread, specialists when they
  earn their keep.
- **Session isolation.** `/isolate` runs the session in its own git
  worktree; `/review` shows the diff, `/merge` or `/discard` ends it.
  Auto-checkpoints before every edit; `/undo` restores.
- **Gets sharper at your repo.** A per-project notebook plus episode
  memory (`.loom/`) — notes the agent used get credited when a turn
  goes well, so future runs surface what worked. `/good` and `/bad`
  train it.
- **Cost you can see.** Every response closes with that turn's tokens
  and dollar cost (`free` on free tiers). `/cost` has session totals.
- **MCP out of the box.** Connect Linear, Sentry, Postgres,
  Playwright, or any MCP server; `/mcp` lists what's live.
- **Goal mode.** `/goal make all tests pass` keeps working until the
  condition is verifiably met.

## Install

```bash
pipx install loom-code
```

(`pip install loom-code` works too; `pipx` keeps CLI tools in their
own venvs. No pipx? `brew install pipx` or
`python -m pip install --user pipx`.)

Requires Python 3.11+. To update: `pipx upgrade loom-code`.

## Quickstart

```bash
cd ~/your-project
loom-code
```

First run: type `/set_model`, pick a provider with the arrow keys,
paste your API key once (it's saved for future sessions), pick a
model. **No paid key?** Pick NVIDIA — free at
[build.nvidia.com](https://build.nvidia.com).

Then just type what you want:

```
›  fix the failing test in tests/test_auth.py
›  add a /users endpoint with pagination
›  why is startup slow? profile it
```

One-shot mode (does the task, prints a summary, exits):

```bash
loom-code "add a retry decorator to the http client"
loom-code --yes "scaffold a FastAPI backend"   # skip approval prompts
```

Works on existing code and empty directories alike — scaffolding new
projects is a first-class path.

## The interface

loom-code runs a **full-screen chat UI**: the conversation scrolls in
a pane, the input box stays pinned at the bottom, and it grows as you
type. Enter sends; Alt+Enter adds a newline. Scroll the history with
the mouse wheel, PageUp/PageDown, or Ctrl-Up/Down; End jumps back to
live. Typing `/` pops the command menu.

Prefer a plain inline prompt (or on a terminal the full-screen UI
doesn't suit)? `loom-code --classic`. Piped/non-TTY use falls back to
classic automatically.

## Models

| model string | provider | env key |
|---|---|---|
| `claude-opus-4-8`, `claude-sonnet-4-6`, … | Anthropic | `ANTHROPIC_API_KEY` |
| `gpt-4.1`, `gpt-4.1-mini`, `o4-mini`, … | OpenAI | `OPENAI_API_KEY` |
| `nvidia/…` (Nemotron, Llama, DeepSeek) | NVIDIA NIM — **free tier** | `NVIDIA_NIM_API_KEY` |
| `ollama/llama3`, `ollama/qwen2.5-coder`, … | local [Ollama](https://ollama.com) — free, offline | — |
| `litellm/<provider>/<model>` | anything LiteLLM routes | provider's own |

Switch anytime with `/model <name>` or the guided `/set_model`.
Reasoning models support `/effort low|medium|high`.

> Tip: tool-heavy agent work needs a model with solid function
> calling. On the free NVIDIA tier, `deepseek-v4-pro` and
> `nemotron-super-49b` hold up well; tiny models fumble tool calls.

## Safety & permissions

The permission layer is the boundary, not the working directory:

- **Reads** anywhere are allowed; **writes outside the project** are
  only possible for files *you* referenced, and always show a diff
  prompt — in every mode, even `--yes`.
- **Approval modes** (`/mode`): `default` asks for writes and shell;
  `accept-edits` auto-approves in-project edits; `plan` is read-only;
  `yolo` approves everything except your deny rules.
- **Rules** live in `.loom/settings.toml` — glob-based
  `allow` / `ask` / `deny` per tool (e.g. `deny = ["edit(*.env)"]`).
  Deny always wins, even in yolo.
- **Sandbox**: `--sandbox` runs bash under OS-level isolation
  (writes limited to the repo, network off unless
  `--sandbox-allow-network`).
- Irreversible commands (`git push --force`, `rm -rf`, …) always
  get an explicit prompt.

## Commands

Type `/` in the REPL — the menu autocompletes. Highlights:

| | |
|---|---|
| `/plan` | show or start the living plan |
| `/goal <condition>` | work until the condition is met |
| `/undo` · `/checkpoints` | restore / list auto-checkpoints |
| `/isolate` · `/review` · `/merge` · `/discard` | worktree-isolated sessions |
| `/model` · `/set_model` · `/effort` · `/mode` | model + approval setup |
| `/set_web` | web search (Serper / DuckDuckGo) |
| `/mcp` | list connected MCP servers |
| `/cost` · `/compact` · `/export` | session accounting + history |
| `/resume` | pick up the last session for this project |
| `/good` · `/bad` | credit / debit the agent's notes |

## Project context

loom-code reads `LOOM.md` / `CLAUDE.md` / `AGENTS.md` /
`.loom/context.md` at the project root and treats it as binding house
rules. `/init-loom` creates a starter file.

## Development

```bash
git clone https://github.com/Anurich/loomflow-cli
cd loomflow-cli
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q          # 465 tests
ruff check .
```

Architecture in one line: **loom-code is deliberately thin** — if a
capability belongs in the agent loop, it goes in
[loomflow](https://github.com/Anurich/LoomFlow), not here. See
`DESIGN.md` for the boundary.

## License

MIT — see [LICENSE](LICENSE).
