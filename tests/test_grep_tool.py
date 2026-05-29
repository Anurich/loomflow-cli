"""Tests for ``enhanced_grep_tool`` — the loom-code grep wrapper.

The four behaviours this file guards:

1. Grouped-by-file output with surrounding context (default mode).
2. Test-file collapsing — hits under ``tests/`` / matching
   ``test_*.py`` etc. don't drown out prod code unless explicitly
   asked for.
3. ``raw=True`` escape hatch — flat ``path:line: content`` shape
   for callers who want one-line-per-match.
4. ``type=`` filter narrows the file walk by extension.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loom_code.grep_tool import enhanced_grep_tool


def _make_repo(root: Path) -> None:
    """Seed a tiny fake project: 2 prod files, 1 test file, 1
    file under a deeply nested test dir."""
    (root / "src").mkdir()
    (root / "src" / "a.py").write_text(
        "# top comment\n"
        "def authenticate(user):\n"
        "    # check token\n"
        "    return token_valid(user.token)\n"
        "\n"
        "def logout(user):\n"
        "    pass\n"
    )
    (root / "src" / "b.py").write_text(
        "from a import authenticate\n"
        "result = authenticate('alice')\n"
    )
    (root / "tests").mkdir()
    (root / "tests" / "test_a.py").write_text(
        "from src.a import authenticate\n"
        "def test_authenticate():\n"
        "    assert authenticate('x')\n"
    )
    # noise dirs should be skipped silently
    (root / ".venv").mkdir()
    (root / ".venv" / "ignored.py").write_text("authenticate()\n")


def test_grouped_output_shows_context(tmp_path: Path) -> None:
    """Default mode renders one block per file, with line
    numbers and ±2 context around each hit. The ▸ marker
    flags the actual match line."""
    _make_repo(tmp_path)
    grep = enhanced_grep_tool(tmp_path)

    result = asyncio.run(grep.fn(pattern="authenticate"))

    # Prod files appear, with file header + context.
    assert "src/a.py" in result
    assert "src/b.py" in result
    # The ▸ marker fires on the hit line.
    assert "▸" in result
    # Context lines visible (line 1 is "# top comment", before
    # the def authenticate match on line 2).
    assert "# top comment" in result


def test_collapses_test_file_matches_by_default(
    tmp_path: Path,
) -> None:
    """Hits in test files don't show by default. A one-line
    summary appears at the bottom telling the agent how many
    were collapsed + how to surface them."""
    _make_repo(tmp_path)
    grep = enhanced_grep_tool(tmp_path)

    result = asyncio.run(grep.fn(pattern="authenticate"))

    # Test file body NOT rendered.
    assert "tests/test_a.py" not in result
    assert "def test_authenticate" not in result
    # But the collapse summary IS present, with the include_tests
    # hint so the agent knows how to opt in.
    assert "test file" in result.lower()
    assert "include_tests" in result


def test_include_tests_opens_collapse(tmp_path: Path) -> None:
    """include_tests=True unmasks the test-file matches."""
    _make_repo(tmp_path)
    grep = enhanced_grep_tool(tmp_path)

    result = asyncio.run(
        grep.fn(pattern="authenticate", include_tests=True)
    )
    assert "tests/test_a.py" in result
    assert "def test_authenticate" in result


def test_raw_mode_returns_flat_lines(tmp_path: Path) -> None:
    """raw=True drops the grouped/contextual rendering and
    falls back to the path:lineno: content shape so callers
    needing tight one-line-per-match output can opt out."""
    _make_repo(tmp_path)
    grep = enhanced_grep_tool(tmp_path)

    result = asyncio.run(
        grep.fn(pattern="authenticate", raw=True)
    )
    # No tier headers or markers.
    assert "▸" not in result
    assert "─" not in result
    # Classic path:line: shape present.
    assert "src/a.py:2:" in result
    assert "src/b.py:2:" in result


def test_type_filter_narrows_extension(tmp_path: Path) -> None:
    """type='md' should miss the .py matches entirely; type='py'
    should include them."""
    _make_repo(tmp_path)
    (tmp_path / "src" / "notes.md").write_text(
        "authenticate this section is about auth\n"
    )
    grep = enhanced_grep_tool(tmp_path)

    md_only = asyncio.run(
        grep.fn(pattern="authenticate", type="md")
    )
    assert "src/notes.md" in md_only
    assert "src/a.py" not in md_only

    py_only = asyncio.run(
        grep.fn(pattern="authenticate", type="py")
    )
    assert "src/a.py" in py_only
    assert "src/notes.md" not in py_only


def test_skips_noise_dirs(tmp_path: Path) -> None:
    """`.venv` / `node_modules` / `__pycache__` etc. must be
    silently skipped — without this, every grep returns
    thousands of irrelevant matches in installed deps."""
    _make_repo(tmp_path)
    grep = enhanced_grep_tool(tmp_path)

    result = asyncio.run(grep.fn(pattern="authenticate"))
    assert ".venv" not in result


def test_invalid_regex_returns_error_not_silent_no_matches(
    tmp_path: Path,
) -> None:
    """A bad regex should surface a clear error so the agent
    can fix the call. Silently returning 'no matches' would
    let the agent assume the keyword wasn't in the codebase."""
    _make_repo(tmp_path)
    grep = enhanced_grep_tool(tmp_path)

    result = asyncio.run(grep.fn(pattern="[unclosed"))
    assert "invalid regex" in result.lower()


def test_path_outside_workdir_refused(tmp_path: Path) -> None:
    """Refuse traversal attempts to escape the workdir — agents
    shouldn't be able to grep ../../.."""
    _make_repo(tmp_path)
    grep = enhanced_grep_tool(tmp_path)

    result = asyncio.run(
        grep.fn(pattern="authenticate", path="../../../")
    )
    assert "refusing" in result.lower()


def test_tool_name_and_destructive_flag(tmp_path: Path) -> None:
    """The tool registers as ``grep`` (replacing loomflow's
    grep_tool slot in the agent's tool surface) and is marked
    non-destructive (it's read-only)."""
    grep = enhanced_grep_tool(tmp_path)
    assert grep.name == "grep"
    assert grep.destructive is False


def test_string_typed_args_coerced_not_crash(tmp_path: Path) -> None:
    """The headline crash: the tool-call layer serialises typed
    params as strings (context='2', ignore_case='true'), and the
    line-window math did ``lineno - context`` → 'int - str'
    TypeError. Coercion must make string args work."""
    _make_repo(tmp_path)
    grep = enhanced_grep_tool(tmp_path)

    # Pass EVERY typed param as a string, exactly as a weak model's
    # serialised tool call would.
    result = asyncio.run(
        grep.fn(
            pattern="authenticate",
            context="2",          # str, not int
            ignore_case="true",   # str, not bool
            include_tests="false",  # str, not bool
            raw="false",          # str, not bool
        )
    )
    # Must NOT crash — must return real grouped output.
    assert "src/a.py" in result
    assert "▸" in result


def test_string_context_controls_window(tmp_path: Path) -> None:
    """A string context value must still control the window size,
    not silently fall back to default."""
    f = tmp_path / "big.py"
    f.write_text("\n".join(f"line_{i}" for i in range(1, 31)) + "\n")
    grep = enhanced_grep_tool(tmp_path)

    # context="0" → only the hit line, no surrounding context.
    result = asyncio.run(
        grep.fn(pattern="line_15", context="0")
    )
    assert "line_15" in result
    assert "line_14" not in result  # context=0 → no neighbours


def test_as_bool_and_as_int_helpers() -> None:
    """Unit-pin the coercion helpers — they guard the whole tool
    surface against weak-model string serialisation."""
    from loom_code.grep_tool import _as_bool, _as_int

    assert _as_int("2", 99) == 2
    assert _as_int(2, 99) == 2
    assert _as_int("garbage", 99) == 99  # unparseable → default
    assert _as_int(True, 99) == 99  # bool not treated as int

    assert _as_bool("true") is True
    assert _as_bool("false") is False
    assert _as_bool("1") is True
    assert _as_bool("0") is False
    assert _as_bool(True) is True
    assert _as_bool("garbage", default=True) is True  # → default
    assert _as_bool("", default=True) is False  # empty → falsy


# ---- ripgrep fast path + fallback parity --------------------------


def _seed(d: Path) -> None:
    (d / "a.py").write_text("def foo():\n    return 1\nfoo_bar = 2\n")
    (d / "b.py").write_text("x = foo()\n# foo comment\n")
    sub = d / "sub"
    sub.mkdir()
    (sub / "c.py").write_text("foo_again = 3\n")


def test_rg_and_python_paths_agree(tmp_path: Path) -> None:
    """The ripgrep fast path must produce byte-for-byte the same output
    as the pure-Python fallback — the renderer is shared, rg only
    changes HOW matches are found, never how they're shown."""
    from unittest.mock import patch

    _seed(tmp_path)
    tool = enhanced_grep_tool(tmp_path)
    rg_out = asyncio.run(tool.fn(pattern="foo", path="."))
    with patch("loom_code.grep_tool._rg_path", return_value=None):
        py_out = asyncio.run(tool.fn(pattern="foo", path="."))
    assert rg_out == py_out


def test_python_fallback_used_when_rg_absent(tmp_path: Path) -> None:
    """When rg isn't on PATH the tool still works via the Python walk."""
    from unittest.mock import patch

    _seed(tmp_path)
    tool = enhanced_grep_tool(tmp_path)
    with patch("loom_code.grep_tool._rg_path", return_value=None):
        out = asyncio.run(tool.fn(pattern="foo_bar", path="."))
    assert "foo_bar" in out


def test_lookahead_pattern_falls_back_to_python(tmp_path: Path) -> None:
    """rg's Rust regex rejects lookahead (exit 2) → _collect_with_ripgrep
    returns None → the tool falls back to Python, which supports it. No
    capability regression for exotic regex."""
    from loom_code.grep_tool import _collect_with_ripgrep

    _seed(tmp_path)
    # rg should reject this and signal fallback.
    res = _collect_with_ripgrep(
        tmp_path,
        "foo(?=_)",
        ignore_case=False,
        glob="*",
        type_filter=None,
        max_files=30,
        max_per_file=10,
    )
    assert res is None
    # And the tool as a whole still resolves the lookahead via Python.
    tool = enhanced_grep_tool(tmp_path)
    out = asyncio.run(tool.fn(pattern="foo(?=_)", path="."))
    assert "foo_bar" in out or "foo_again" in out


def test_rg_path_returns_str_or_none() -> None:
    from loom_code.grep_tool import _rg_path

    val = _rg_path()
    assert val is None or isinstance(val, str)
