"""Shared fixtures for the loom-code test suite.

The whole suite is offline and deterministic: every test that
builds agents uses ``model="echo"`` (loomflow's zero-key
EchoModel), and nothing here makes a network call. Tests assert on
*structure* (rosters, tool scoping, wiring, event-payload
handling), not on model behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loom_code.project import Project


@pytest.fixture
def project(tmp_path: Path) -> Project:
    """A minimal Project rooted at an isolated tmp dir — no git, no
    context file. Tests that need git / a context file craft their
    own Project or call detect_project on a built-up tree."""
    return Project(
        root=tmp_path,
        is_git=False,
        context_file=None,
        context_text="",
    )
