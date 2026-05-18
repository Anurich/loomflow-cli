---
name: graphify
description: Build + query a knowledge graph of this project's code. Reach for it on STRUCTURAL questions (cross-file deps, paths between concepts, which file/module everything connects through) that grep can't traverse — never for single-file questions grep wins on.
---

# graphify

Turn the project's source files into a knowledge graph: nodes are
symbols and files, edges are imports / calls / references / shared
data. Persisted to ``.loom/graphify/graph.json`` so queries are
cheap across runs. AST-only extraction — deterministic, no LLM
cost, no provider key required.

## Tools (after `load_skill`)

**IMPORTANT — these are PYTHON @tool FUNCTIONS, not shell
commands.** Call them via the model's tool-call mechanism, the
same way you call ``read`` / ``edit`` / ``grep``. Do NOT pass
them to ``bash`` — there is no ``graphify__build`` executable on
disk, only a registered tool with that name. If you try
``bash graphify__build ...`` you'll get "command not found"
because graphify__build is an in-process Python function, not a
CLI. Use tool-call syntax exclusively.

* ``graphify__build(path=".")`` — walk the project, extract,
  cluster, write ``.loom/graphify/graph.json``. Idempotent and
  incremental: re-running on an unchanged repo is fast (file
  hashes gate re-extraction). Run this ONCE per project (or after
  major refactors) before the query tools below; the post-commit
  hook keeps it current every 5 commits afterward.

* ``graphify__query(question, path=".")`` — BFS traversal over
  the graph from nodes whose label matches keywords in
  ``question``. Returns a list of related nodes + their edges,
  one paragraph each. Use for "what's involved in X" / "how does
  Y work in this codebase" questions.

* ``graphify__path(a, b, path=".")`` — shortest path between two
  named concepts. Returns the hop-by-hop trail with edge labels.
  Use for "how does A get to B" / "what's the connection between
  X and Y" — exactly what grep can't answer.

* ``graphify__explain(node, path=".")`` — plain-language
  explanation of one node: what it is, its source file/line, its
  immediate neighbors, the community/cluster it belongs to. Use
  when the user asks about a specific symbol or file.

## When to reach for graphify

Use the graph tools when the question is **structural** — about
how things connect across files. Concrete trigger shapes:

* "What connects A to B?"
* "Where is the auth code used?"
* "What are the dependencies of foo?"
* "Show me the path between X and Y."
* "Which file is central to this codebase?" → ``graphify__query("god nodes")`` returns highest-degree symbols.

**Do NOT reach for graphify when:**

* The question is about ONE file's content — use ``read`` /
  ``grep`` directly. Graph queries waste tokens on one-file
  answers.
* You haven't run ``graphify__build`` yet — call it first, then
  query. If ``.loom/graphify/graph.json`` doesn't exist any
  graph_* call returns an error pointing here.
* The user wants raw source text — that's a ``read`` / ``grep``
  job, not a graph job. The graph holds *structure*, not bodies.

## Cost shape

* ``graphify__build``: ~5-30s on typical loom-code-sized
  projects (100-500 files). Up to 2 min on monorepos. Pure
  Python, no LLM call. AST-only — code files only; docs / PDFs /
  images need the standalone ``/graphify`` skill in Claude Code,
  not this one.

* ``graphify__query`` / ``graphify__path`` / ``graphify__explain``:
  milliseconds. They load ``graph.json`` once and traverse in
  memory.
