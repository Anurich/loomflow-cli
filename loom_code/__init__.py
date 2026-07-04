"""loom-code — a loomflow-native terminal coding agent.

The entire agent brain is loomflow:

* ``Agent`` + ``ReAct`` — the agent loop
* ``living_plan=True`` — the task tracker (Claude Code's TodoWrite)
* ``LocalDiskWorkspace`` — per-project memory + the self-improvement
  loop (citation tracking + relevance-aware recall)
* ``read`` / ``write`` / ``edit`` / ``bash`` / ``grep`` / ``find``
  / ``ls`` builtin tools — the file-and-shell kernel
* ``StandardPermissions`` + ``approval_handler`` — the safety gate
* ``Agent.stream()`` — streaming output

This package is ONLY the terminal shell: REPL, ``rich`` rendering,
slash commands, project detection, the diff-approval prompt. If
agent-loop / memory / tool-dispatch logic ever shows up here, that
is a bug — it means loomflow is missing something and the fix
belongs in the framework, not here. loom-code is the dogfood test
that keeps loomflow honest.
"""

__version__ = "0.2.0"
