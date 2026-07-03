"""The Claude-Code-style file boundary: policy, not the cwd.

The failure this locks down (observed live): pointing loom-code at a
file one directory ABOVE the project — "read/edit this" — used to fail
with "file not found" / "refusing to edit outside workdir", because
loomflow's tools hard-refuse any path outside their workdir. loom-code
now moves the boundary to the permission layer: the tools reach any
path, and a file the USER referenced this session (consent) is
readable + editable even outside the project, while a self-initiated
outside access the user never named is refused (prompt-injection
guard).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loom_code import consent
from loom_code.edit_tool import multi_edit_tool, verifying_edit_tool
from loom_code.file_tools import loom_read_tool
from loom_code.paths import is_within, resolve_path


@pytest.fixture(autouse=True)
def _clean_consent():
    consent.reset()
    yield
    consent.reset()


# ---- resolve_path ---------------------------------------------------------


def test_resolve_relative_anchors_to_project(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    assert resolve_path("a/b.py", proj) == (proj / "a/b.py").resolve()


def test_resolve_absolute_passes_through(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    outside = tmp_path / "x.py"
    assert resolve_path(str(outside), proj) == outside.resolve()


def test_is_within(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    assert is_within(proj / "a.py", proj)
    assert not is_within(tmp_path / "b.py", proj)


# ---- read boundary --------------------------------------------------------


@pytest.mark.anyio
async def test_read_relative_in_project(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "in.py").write_text("inside\n")
    out = await loom_read_tool(proj).fn(path="in.py")
    assert "inside" in out


@pytest.mark.anyio
async def test_read_outside_requires_reference(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    outside = tmp_path / "out.py"
    outside.write_text("secret\n")
    tool = loom_read_tool(proj)
    # not referenced → refused (injection guard)
    assert "was not referenced" in await tool.fn(path=str(outside))
    # referenced → readable
    consent.grant(outside)
    assert "secret" in await tool.fn(path=str(outside))


# ---- edit / multi_edit boundary -------------------------------------------


@pytest.mark.anyio
async def test_edit_outside_referenced_file(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    outside = tmp_path / "bench.py"
    outside.write_text("PDF = '/hardcoded.pdf'\n")
    consent.grant(outside)  # user @-mentioned it
    result = await verifying_edit_tool(proj).fn(
        path=str(outside),
        old_string="'/hardcoded.pdf'",
        new_string="args.pdf",
    )
    assert not result.splitlines()[0].startswith("ERROR")
    assert outside.read_text() == "PDF = args.pdf\n"


@pytest.mark.anyio
async def test_edit_outside_unreferenced_refused(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    outside = tmp_path / "x.py"
    outside.write_text("a = 1\n")
    result = await verifying_edit_tool(proj).fn(
        path=str(outside), old_string="a = 1", new_string="a = 2"
    )
    assert "refusing" in result
    assert outside.read_text() == "a = 1\n"  # untouched


@pytest.mark.anyio
async def test_in_project_edit_shows_clean_path(tmp_path: Path) -> None:
    # In-project edits keep the short relative path in the result
    # message (not the sprawling absolute form).
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.py").write_text("x = 1\n")
    result = await verifying_edit_tool(proj).fn(
        path="a.py", old_string="x = 1", new_string="x = 2"
    )
    assert "edited a.py" in result


@pytest.mark.anyio
async def test_multi_edit_outside_referenced(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    outside = tmp_path / "bench.py"
    outside.write_text("A = 1\nB = 2\n")
    consent.grant(outside)
    await multi_edit_tool(proj).fn(
        path=str(outside),
        edits=[
            {"old_string": "A = 1", "new_string": "A = 9"},
            {"old_string": "B = 2", "new_string": "B = 8"},
        ],
    )
    assert outside.read_text() == "A = 9\nB = 8\n"
