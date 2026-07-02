"""Pin the contract of the loom-code → graphify integration so the
shared build helper + the post-commit refresh stay aligned, and so the
retired `/loominit` / `/graphify` setup commands can't sneak back.

Guards:
1. ``graphify_build_impl`` is the single shared entrypoint that both
   the ``@tool`` wrapper and ``_refresh_graphify`` (post-commit hook)
   route through — so the submodule-import shim + git-ls-files fast
   path + correct ``cluster`` / ``to_json`` arg shapes can't drift.
2. Neither ``/graphify`` nor ``/loominit`` exists anymore — graphify
   builds on demand via the ``graphify__build`` tool; the codebase
   structure reaches the agent through the deterministic repo map.
"""

from __future__ import annotations

import inspect


def test_graphify_build_impl_is_shared_helper() -> None:
    """The skill's tools module exports a non-tool helper that the
    ``@tool`` wrapper + post-commit hook call directly."""
    from loom_code.skills.graphify import tools as graphify_tools

    assert hasattr(graphify_tools, "graphify_build_impl"), (
        "graphify_build_impl missing — the tool wrapper + post-commit "
        "hook can't share the same build path"
    )
    assert hasattr(graphify_tools, "GraphifyBuildResult"), (
        "GraphifyBuildResult dataclass missing"
    )


def test_tool_wrapper_uses_shared_helper() -> None:
    """The ``@tool``-decorated ``build`` is a thin formatter around
    the shared helper. Lock the contract: if a future edit
    re-implements the body inline, the submodule-import shim + git
    fast path could silently drift."""
    from loom_code.skills.graphify import tools as graphify_tools

    src = inspect.getsource(graphify_tools.build.fn)
    assert "graphify_build_impl" in src, (
        "build @tool no longer delegates to graphify_build_impl — "
        "risks divergence from the post-commit refresh path"
    )


def test_post_commit_refresh_uses_shared_helper() -> None:
    """``_refresh_graphify`` must route through the shared helper so
    the four historical hand-rolled-pipeline bugs (extract submodule,
    wrong arg shape, cluster overwrite, missing communities arg) can't
    come back."""
    from loom_code._post_commit import _refresh_graphify

    src = inspect.getsource(_refresh_graphify)
    assert "graphify_build_impl" in src, (
        "_refresh_graphify no longer uses the shared helper — "
        "post-commit refresh likely broken"
    )


def test_setup_commands_are_gone() -> None:
    """Both ``/graphify`` and ``/loominit`` were removed: graphify
    builds on demand via its tool, and the codebase map is the
    deterministic repo map (no LLM-narrated LOOM.md / index.json).
    Pin that neither the handlers nor the command defs survive so we
    can't accidentally re-introduce the retired setup-command UX."""
    from loom_code import repl as repl_mod

    assert not hasattr(repl_mod.Repl, "_handle_graphify"), (
        "_handle_graphify still defined on Repl"
    )
    assert not hasattr(repl_mod.Repl, "_handle_loominit"), (
        "_handle_loominit still defined on Repl"
    )
    cmds = {entry[0] for entry in repl_mod._COMMAND_DEFS}
    assert "/graphify" not in cmds, "/graphify still in _COMMAND_DEFS"
    assert "/loominit" not in cmds, "/loominit still in _COMMAND_DEFS"
