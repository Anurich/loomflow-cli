"""Tests for the per-file AST walker.

The walker is the foundation of the structural extractor — every
downstream pass (PageRank, API surface, clustering, annotation)
trusts its symbol / import / decorator output. Test the surprising
shapes explicitly: nested classes, async functions, multi-line
signatures, decorator AST node shapes, broken Python, dunders.
"""

from __future__ import annotations

from loom_code.loominit._ast_walk import (
    LANDMARK_DECORATORS,
    walk_python_file,
)


def _walk(src: str):
    return walk_python_file(src, "test.py")


# ---- top-level functions and classes ---------------------------------


def test_walks_top_level_function() -> None:
    symbols, _imports, _decs = _walk(
        '''
def foo(x: int) -> int:
    """First line.

    Second line.
    """
    return x
'''
    )
    assert len(symbols) == 1
    s = symbols[0]
    assert s.name == "foo"
    assert s.qualified_name == "foo"
    assert s.kind == "function"
    assert s.is_public is True
    assert s.docstring_first_line == "First line."
    assert "def foo(x: int) -> int:" in s.signature


def test_walks_async_function() -> None:
    symbols, _, _ = _walk("async def bar():\n    pass\n")
    assert len(symbols) == 1
    assert symbols[0].kind == "function"
    assert symbols[0].name == "bar"


def test_walks_class_and_methods() -> None:
    symbols, _, _ = _walk(
        """
class Outer:
    def method(self):
        pass
    class Inner:
        def deep(self):
            pass
"""
    )
    by_qual = {s.qualified_name: s for s in symbols}
    assert by_qual["Outer"].kind == "class"
    assert by_qual["Outer.method"].kind == "method"
    assert by_qual["Outer.Inner"].kind == "class"
    assert by_qual["Outer.Inner.deep"].kind == "method"


def test_underscore_prefix_marks_private() -> None:
    symbols, _, _ = _walk("def _helper(): pass\n")
    assert symbols[0].is_public is False


# ---- decorators ------------------------------------------------------


def test_decorator_names_simple_name() -> None:
    symbols, _, _ = _walk(
        """
@decorator
def fn():
    pass
"""
    )
    assert symbols[0].decorators == ("decorator",)


def test_decorator_names_attribute_chain() -> None:
    symbols, _, _ = _walk(
        """
@click.command
def fn():
    pass
"""
    )
    assert symbols[0].decorators == ("click.command",)


def test_decorator_names_call_form_drops_args() -> None:
    """``@app.route("/x")`` is recorded as ``"app.route"`` — the
    annotator wants the *kind* of decorator, not the literal call."""
    symbols, _, _ = _walk(
        """
@app.route("/login", methods=["POST"])
def login():
    pass
"""
    )
    assert symbols[0].decorators == ("app.route",)


def test_landmark_decorator_is_captured() -> None:
    """``@click.command`` is in :data:`LANDMARK_DECORATORS` so it
    should land in the decorators list AS WELL AS on the symbol."""
    assert "click.command" in LANDMARK_DECORATORS
    _, _, decs = _walk(
        """
@click.command
def main():
    pass
"""
    )
    assert len(decs) == 1
    assert decs[0].decorator == "click.command"
    assert decs[0].target_qualname == "main"


def test_non_landmark_decorator_skipped() -> None:
    """A random decorator gets attached to the symbol but is NOT
    promoted to a landmark — landmark detection has to be allow-
    list-based or the annotator drowns in noise."""
    _, _, decs = _walk(
        """
@something_random
def fn():
    pass
"""
    )
    assert decs == []


# ---- imports ---------------------------------------------------------


def test_plain_import_records_module() -> None:
    _, imports, _ = _walk("import foo.bar\n")
    assert len(imports) == 1
    assert imports[0].to_module == "foo.bar"
    assert imports[0].level == 0


def test_import_as_records_real_module() -> None:
    """``import foo as bar`` → we record ``foo`` (the real module),
    not the alias. Alias resolution happens in the import-graph
    resolver, not here."""
    _, imports, _ = _walk("import foo.bar as fb\n")
    assert imports[0].to_module == "foo.bar"


def test_from_import_records_module() -> None:
    _, imports, _ = _walk("from foo.bar import baz\n")
    assert imports[0].to_module == "foo.bar"
    assert imports[0].level == 0


def test_relative_import_records_level() -> None:
    """``from .. import x`` → level 2, module empty. The resolver
    needs the level to compute the absolute target."""
    _, imports, _ = _walk("from .. import x\n")
    assert imports[0].to_module == ""
    assert imports[0].level == 2


# ---- constants -------------------------------------------------------


def test_uppercase_module_constant_indexed() -> None:
    symbols, _, _ = _walk("MAX_RETRIES = 5\n")
    assert len(symbols) == 1
    assert symbols[0].kind == "constant"
    assert symbols[0].name == "MAX_RETRIES"


def test_lowercase_module_local_skipped() -> None:
    """``x = 1`` at module level is plumbing, not a "constant" worth
    indexing. Filter avoids drowning the index."""
    symbols, _, _ = _walk("x = 1\n")
    assert symbols == []


def test_pascal_case_module_alias_indexed() -> None:
    """``MyType = list[str]`` is a type alias — worth indexing."""
    symbols, _, _ = _walk("MyType = list[str]\n")
    assert len(symbols) == 1
    assert symbols[0].name == "MyType"
    assert symbols[0].kind == "constant"


def test_annotated_module_constant_indexed() -> None:
    symbols, _, _ = _walk("MAX_RETRIES: int = 5\n")
    assert len(symbols) == 1
    assert symbols[0].name == "MAX_RETRIES"
    assert symbols[0].kind == "constant"


def test_dunders_indexed() -> None:
    """``__all__`` / ``__version__`` are useful entries even though
    they're dunder-prefixed."""
    symbols, _, _ = _walk('__version__ = "1.0"\n')
    assert symbols[0].name == "__version__"


def test_locals_inside_function_not_indexed() -> None:
    """``def foo(): x = 1`` — ``x`` is local; must not pollute the
    symbol table."""
    symbols, _, _ = _walk(
        """
def foo():
    X = 1
"""
    )
    names = {s.name for s in symbols}
    assert "X" not in names
    assert "foo" in names


# ---- multi-line signatures ------------------------------------------


def test_multi_line_signature_collapsed() -> None:
    symbols, _, _ = _walk(
        """
def foo(
    a: int,
    b: str,
) -> bool:
    return True
"""
    )
    sig = symbols[0].signature
    assert "def foo(" in sig
    assert "a: int" in sig
    assert "b: str" in sig
    assert "-> bool" in sig


# ---- error tolerance -------------------------------------------------


def test_syntax_error_returns_empty_lists() -> None:
    """A broken file in the middle of a repo must NOT abort indexing.
    Walker returns empty results; the caller logs the broken path
    and continues."""
    symbols, imports, decs = _walk("def broken(:\n")
    assert symbols == []
    assert imports == []
    assert decs == []


def test_empty_file_returns_empty_lists() -> None:
    symbols, imports, decs = _walk("")
    assert symbols == []
    assert imports == []
    assert decs == []
