"""loom-code's file tools — Claude-Code-style path boundary.

The built-in loomflow ``read_tool`` hard-refuses any path outside its
workdir (``_resolve_within`` raises). That makes "point at a file one
directory up" impossible — the tool returns "file not found" no matter
what path the model passes. Claude Code instead lets the tool reach
any path the OS allows and puts the boundary in the PERMISSION layer:
reads are lenient (in-project auto-allowed, outside approvable), writes
are strict (outside always confirmed).

This module builds loom-code's ``read`` tool to that model:

* Resolve the path with :func:`loom_code.paths.resolve_path` (``~``,
  cwd-relative, absolute) — one canonical resolver.
* IN the project → read normally.
* OUTSIDE the project → allowed only when the user REFERENCED the file
  this session (``@``-mention / pasted path → :mod:`loom_code.consent`)
  or a config rule permits it. Reads never mutate, so — matching Claude
  Code — an outside read the user asked for goes straight through
  rather than nagging; a self-initiated outside read the user never
  named is refused (prompt-injection guard).

The underlying read is delegated to a loomflow ``read_tool`` rooted at
the filesystem root, so an absolute path never "escapes" it — the
containment decision has already been made here.

Edit/Write keep their existing loom-code wrappers (``edit_tool.py``),
which already consent-gate outside writes and always show a diff.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loomflow import tool
from loomflow.tools import read_tool as _loomflow_read_tool

from .paths import is_within, resolve_path


def loom_read_tool(workdir: Path | str) -> Any:
    """A ``read`` tool whose boundary is policy, not the cwd.

    In-project reads behave exactly as before. An outside-project read
    is permitted when the user referenced that file this session (see
    :mod:`loom_code.consent`); otherwise it's refused with a message
    telling the model to ask the user to reference the file — so a
    prompt-injected "read ~/.ssh/id_rsa" the user never named fails.
    """
    root = Path(workdir).resolve()
    anchor = root.anchor or "/"
    # Delegate the actual read to a loomflow tool rooted at the
    # filesystem anchor ("/"), so an already-resolved absolute path is
    # always accepted by its own ``_resolve_within`` (nothing escapes
    # the root). The containment decision is made HERE, not there.
    inner = _loomflow_read_tool(Path(anchor))

    async def read(
        path: str,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        target = resolve_path(path, root)
        if not is_within(target, root):
            from . import consent

            if not consent.is_granted(target):
                return (
                    f"ERROR: {path} is outside the project and was not "
                    "referenced by the user. Ask the user to share the "
                    "file (they can paste its path or @-mention it); "
                    "do not read outside the project on your own."
                )
        # Hand the inner ("/"-rooted) tool a path relative to ITS root:
        # the absolute target with the leading anchor stripped.
        rel = str(target)
        if rel.startswith(anchor):
            rel = rel[len(anchor):]
        return await inner.fn(path=rel, offset=offset, limit=limit)

    return tool(
        name="read",
        description=(
            "Read a text file, returned with line numbers. The path may "
            "be relative to the project, absolute, or start with ~. "
            "Files the user referenced this session (pasted a path or "
            "@-mentioned) are readable even outside the project; other "
            "outside paths are refused — ask the user to share the file "
            "rather than reading outside the project yourself. Args: "
            "path, offset=0, limit=None."
        ),
    )(read)
