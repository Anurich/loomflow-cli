"""Pin the contract of the loom-code → graphify integration so the
shared helper + the LOOM.md section renderer + the post-commit
refresh all stay aligned. Three things this file guards:

1. ``graphify_build_impl`` is the single shared entrypoint that all
   three callers (the ``@tool`` wrapper, ``_handle_loominit``, and
   ``_refresh_graphify`` in the post-commit hook) route through —
   so the submodule-import-shim + git-ls-files fast-path + correct
   ``cluster`` / ``to_json`` arg shapes can't drift across them.
2. ``_render_graphify_section`` in the REPL produces a body that
   names every graphify tool so the agent can pick one without
   having to ``load_skill('graphify')`` first.
3. ``/graphify`` is gone — the only entrypoint is now ``/loominit``,
   which runs the build during the index pass.
"""

from __future__ import annotations

import inspect


def test_graphify_build_impl_is_shared_helper() -> None:
    """The skill's tools module exports a non-tool helper that the
    REPL + post-commit hook can call directly."""
    from loom_code.skills.graphify import tools as graphify_tools

    assert hasattr(graphify_tools, "graphify_build_impl"), (
        "graphify_build_impl missing — REPL + post-commit hook can't "
        "share the same build path"
    )
    assert hasattr(graphify_tools, "GraphifyBuildResult"), (
        "GraphifyBuildResult dataclass missing — REPL needs it to "
        "render the LOOM.md section"
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
        "this risks divergence between the agent-callable path and "
        "the /loominit-callable path"
    )


def test_post_commit_refresh_uses_shared_helper() -> None:
    """``_refresh_graphify`` originally hand-rolled the pipeline AND
    had four bugs (graphify.extract was a submodule, [extraction]
    was the wrong arg shape, cluster was overwriting graph_obj
    with the community map, to_json was missing the communities
    arg). Pin that it now routes through the shared helper so
    those bugs can't come back."""
    from loom_code._post_commit import _refresh_graphify

    src = inspect.getsource(_refresh_graphify)
    assert "graphify_build_impl" in src, (
        "_refresh_graphify no longer uses the shared helper — "
        "post-commit refresh likely broken"
    )


def test_render_graphify_section_lists_all_tools() -> None:
    """The LOOM.md ``## Knowledge Graph`` body must name every
    graphify tool by name. That's the efficiency win — the agent
    sees the tool surface in its always-injected LOOM.md context,
    no ``load_skill`` round-trip needed before picking the right
    call."""
    from loom_code.repl import _render_graphify_section

    body = _render_graphify_section(
        graph_rel_path=".loom/graphify/graph.json",
        n_nodes=700,
        n_edges=1000,
        n_communities=40,
        source="git ls-files",
    )
    for hint in (
        "graphify__build",
        "graphify__query",
        "graphify__path",
        "graphify__explain",
    ):
        assert hint in body, f"{hint} missing from rendered section"
    # The body must also state the counts + path so the agent knows
    # the graph exists and what shape it is.
    assert ".loom/graphify/graph.json" in body
    assert "700 nodes" in body
    assert "git ls-files" in body


def test_graphify_repl_command_is_gone() -> None:
    """``/graphify`` was removed in favour of folding the build into
    ``/loominit``. Both the dispatcher branch and the handler must
    be absent so we can't accidentally re-introduce the dual
    setup-command UX."""
    from loom_code import repl as repl_mod

    assert not hasattr(repl_mod.Repl, "_handle_graphify"), (
        "_handle_graphify still defined on Repl"
    )
    # The command-defs list is the single source of truth for
    # autocomplete + /help; assert /graphify isn't there.
    cmds = {cmd for cmd, _desc in repl_mod._COMMAND_DEFS}
    assert "/graphify" not in cmds, (
        "/graphify still listed in _COMMAND_DEFS"
    )
