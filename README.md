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

## Install

One command, no clone needed:

```bash
pipx install git+https://github.com/Anurich/loomflow-cli
```

`pipx` is the standard installer for Python CLIs (`aider`,
`ruff`, `httpie` all ship this way). It clones the repo into its
own isolated cache, creates a private venv, and puts `loom-code`
on your PATH — globally. No `cd` into our repo, no venv to
manage. If you don't have pipx: `brew install pipx` (macOS) or
`python -m pip install --user pipx`.

Then set one model key in your environment (or a local `.env`):

```bash
export OPENAI_API_KEY=sk-...        # OpenAI / gpt-4.1-mini default
# or
export ANTHROPIC_API_KEY=sk-ant-... # Anthropic / Claude
```

Ollama needs no key (it's local) — see [Models](#models) below.

To update later: `pipx upgrade loom-code`. To remove:
`pipx uninstall loom-code`.

## Use it

`loom-code` always operates on the **current working directory**
— walks up to find `.git` and roots itself there. So:

```bash
cd ~/your-project

# one-shot — agent does the task, prints a summary, exits
loom-code "add a retry decorator to the http client"

# interactive REPL — chat, code, approve, repeat
loom-code
```

The REPL ships with slash commands — type `/` and the menu pops
with everything available (`/help`, `/plan`, `/cost`, `/good`,
`/bad`, `/model`, `/clear`, `/compress_token_length`, `/exit`).

## Install (for development)

If you want to work *on* loom-code's own source:

```bash
git clone https://github.com/Anurich/loomflow-cli
cd loomflow-cli
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Models

`--model "<name>"` accepts any string loomflow's resolver routes.
The common patterns:

| flag value | provider | env you need |
|---|---|---|
| `claude-sonnet-4-6`, `claude-opus-4-7`, ... | Anthropic | `ANTHROPIC_API_KEY` |
| `gpt-4.1-mini`, `gpt-4.1`, `o4-mini`, ... | OpenAI | `OPENAI_API_KEY` |
| `ollama/llama3`, `ollama/qwen2.5-coder`, ... | local [Ollama](https://ollama.com) (free, private, offline) | (optional) `OLLAMA_API_BASE` — defaults to `http://localhost:11434` |
| `litellm/<anything>` | force LiteLLM for any provider (`groq/`, `together_ai/`, `azure/`, `bedrock/`, `vertex_ai/`, ...) | provider's own env |

The default is `gpt-4.1-mini` — override with `--model "<name>"` on
the CLI, or `/model <name>` inside the REPL. Switching mid-REPL
starts a fresh conversation (the previous model's history doesn't
carry over).

Ollama support comes from the `[litellm]` extra on the loomflow
dependency, which is included by default — no extra install
needed.

## Project context

loom-code reads a `LOOM.md` / `CLAUDE.md` / `AGENTS.md` /
`.loom/context.md` file at the project root, if present, and
treats it as binding house rules.
