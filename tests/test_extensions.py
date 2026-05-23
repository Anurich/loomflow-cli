"""Tests for the ``.loom`` extension system — discovery + wiring.

Covers the three layers (bundled / user / project), the dependency-
free frontmatter + ``settings.toml`` parsing, name-collision
precedence, and the build_agent wiring that turns discovered specs
into skills + delegate-able subagents.

Async hook + trust behaviour lives in ``test_loom_hooks.py``.
"""

from __future__ import annotations

from pathlib import Path

from loom_code.agent import build_agent
from loom_code.extensions import (
    AgentSpec,
    HookSpec,
    discover,
    safe_role_name,
)
from loom_code.project import Project
from loom_code.workers import build_custom_worker

# ---- helpers --------------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _skill(base: Path, name: str, desc: str) -> None:
    _write(
        base / "skills" / name / "SKILL.md",
        f"---\nname: {name}\ndescription: {desc}\n---\nbody\n",
    )


def _tool_names(agent: object) -> set[str]:
    host = agent._tool_host  # noqa: SLF001 — test introspection
    return set(getattr(host, "_tools", {}).keys())


# ---- discovery: skills ----------------------------------------------


def test_discovers_user_and_project_skills(tmp_path: Path) -> None:
    proj = tmp_path / "repo"
    user = tmp_path / "home"
    _skill(proj / ".loom", "deploy", "deploy stuff")
    _skill(user, "lint", "lint stuff")
    ext = discover(proj, user_dir=user)
    names = {p.name for p in ext.skill_paths}
    assert names == {"deploy", "lint"}


def test_project_skill_listed_after_user_for_last_wins(
    tmp_path: Path,
) -> None:
    # Same-named skill in both scopes: both paths are returned, project
    # LAST so the framework's last-source-wins picks it.
    proj = tmp_path / "repo"
    user = tmp_path / "home"
    _skill(proj / ".loom", "deploy", "project deploy")
    _skill(user, "deploy", "user deploy")
    ext = discover(proj, user_dir=user)
    paths = ext.skill_paths
    assert len(paths) == 2
    # user dir first, project dir last
    assert "home" in str(paths[0])
    assert "repo" in str(paths[-1])


# ---- discovery: agents ----------------------------------------------


def test_agent_frontmatter_scalar_and_flow_list(tmp_path: Path) -> None:
    proj = tmp_path / "repo"
    _write(
        proj / ".loom" / "agents" / "sec.md",
        "---\nname: security-auditor\n"
        "description: audit auth, crypto, and injection flaws\n"
        "model: gpt-4.1\n"
        "tools: [read, grep, bash]\n---\n"
        "You hunt for vulns.\n",
    )
    ext = discover(proj, user_dir=tmp_path / "none")
    (spec,) = ext.agent_specs
    assert spec.name == "security-auditor"
    # commas in the description must survive (not be split into a list)
    assert spec.description == "audit auth, crypto, and injection flaws"
    assert spec.model == "gpt-4.1"
    assert spec.tools == ("read", "grep", "bash")
    assert spec.system_prompt == "You hunt for vulns."
    assert spec.source == "project"


def test_agent_block_list_tools(tmp_path: Path) -> None:
    proj = tmp_path / "repo"
    _write(
        proj / ".loom" / "agents" / "doc.md",
        "---\nname: doc-writer\ndescription: write docs\n"
        "tools:\n  - read\n  - write\n---\nWrite great docs.\n",
    )
    ext = discover(proj, user_dir=tmp_path / "none")
    (spec,) = ext.agent_specs
    assert spec.tools == ("read", "write")


def test_agent_comma_string_tools_split(tmp_path: Path) -> None:
    proj = tmp_path / "repo"
    _write(
        proj / ".loom" / "agents" / "x.md",
        "---\nname: x\ndescription: d\ntools: read, grep, bash\n---\nb\n",
    )
    ext = discover(proj, user_dir=tmp_path / "none")
    (spec,) = ext.agent_specs
    assert spec.tools == ("read", "grep", "bash")


def test_agent_missing_description_skipped(tmp_path: Path) -> None:
    # name + description are the delegation contract; no description
    # means the supervisor can't route — skip it.
    proj = tmp_path / "repo"
    _write(
        proj / ".loom" / "agents" / "bad.md",
        "---\nname: bad\n---\nno description here\n",
    )
    ext = discover(proj, user_dir=tmp_path / "none")
    assert ext.agent_specs == []


def test_project_agent_overrides_user_on_name(tmp_path: Path) -> None:
    proj = tmp_path / "repo"
    user = tmp_path / "home"
    _write(
        proj / ".loom" / "agents" / "rev.md",
        "---\nname: reviewer2\ndescription: PROJECT version\n---\np\n",
    )
    _write(
        user / "agents" / "rev.md",
        "---\nname: reviewer2\ndescription: USER version\n---\nu\n",
    )
    ext = discover(proj, user_dir=user)
    (spec,) = ext.agent_specs
    assert spec.description == "PROJECT version"
    assert spec.source == "project"


# ---- discovery: hooks -----------------------------------------------


def test_hook_parsing_and_unknown_event_skipped(tmp_path: Path) -> None:
    proj = tmp_path / "repo"
    _write(
        proj / ".loom" / "settings.toml",
        '[[hooks]]\nevent = "PreToolUse"\nmatcher = "bash"\n'
        'command = "./check.sh"\ntimeout = 30\n\n'
        '[[hooks]]\nevent = "Bogus"\ncommand = "x"\n\n'
        '[[hooks]]\nevent = "Stop"\ncommand = "noop"\n',
    )
    ext = discover(proj, user_dir=tmp_path / "none")
    events = [(h.event, h.matcher, h.timeout) for h in ext.hook_specs]
    # Bogus dropped; PreToolUse + Stop kept.
    assert ("PreToolUse", "bash", 30.0) in events
    assert any(e[0] == "Stop" for e in events)
    assert all(e[0] != "Bogus" for e in events)


def test_hooks_additive_user_then_project(tmp_path: Path) -> None:
    proj = tmp_path / "repo"
    user = tmp_path / "home"
    _write(
        user / "settings.toml",
        '[[hooks]]\nevent = "UserPromptSubmit"\ncommand = "u"\n',
    )
    _write(
        proj / ".loom" / "settings.toml",
        '[[hooks]]\nevent = "PreToolUse"\ncommand = "p"\n',
    )
    ext = discover(proj, user_dir=user)
    sources = [h.source for h in ext.hook_specs]
    # both kept; user first, project second
    assert sources == ["user", "project"]


def test_malformed_settings_toml_is_skipped(tmp_path: Path) -> None:
    proj = tmp_path / "repo"
    _write(proj / ".loom" / "settings.toml", "this is = = not toml [[[")
    ext = discover(proj, user_dir=tmp_path / "none")
    assert ext.hook_specs == []


def test_discover_empty_when_no_loom_dir(tmp_path: Path) -> None:
    ext = discover(tmp_path / "repo", user_dir=tmp_path / "home")
    assert not ext.has_any()


# ---- safe_role_name -------------------------------------------------


def test_safe_role_name_normalises() -> None:
    assert safe_role_name("security-auditor") == "security_auditor"
    assert safe_role_name("Doc Writer") == "Doc_Writer"
    assert safe_role_name("123go") == "a_123go"
    assert safe_role_name("--weird--") == "weird"
    assert safe_role_name("") == "subagent"


# ---- build_custom_worker --------------------------------------------


def test_custom_worker_read_only_default(project: Project) -> None:
    # No tools: declared -> read-only kernel, no permissions gate.
    spec = AgentSpec(
        name="x", description="d", system_prompt="p", tools=()
    )
    agent = build_custom_worker(
        project, spec, model="echo", approval_handler=None
    )
    tools = _tool_names(agent)
    assert "read" in tools and "grep" in tools
    assert "write" not in tools and "bash" not in tools


def test_custom_worker_destructive_gets_permissions(
    project: Project,
) -> None:
    seen: list[bool] = []

    async def handler(call: object, user_id: str | None = None) -> bool:
        seen.append(True)
        return True

    spec = AgentSpec(
        name="x",
        description="d",
        system_prompt="p",
        tools=("read", "edit", "bash"),
    )
    agent = build_custom_worker(
        project, spec, model="echo", approval_handler=handler
    )
    tools = _tool_names(agent)
    assert {"read", "edit", "bash"} <= tools
    # destructive tools -> a permissions policy was wired
    assert agent._permissions is not None  # noqa: SLF001


def test_custom_worker_instructions_lead_with_description(
    project: Project,
) -> None:
    # The supervisor shows the coordinator only the first ~200 chars of
    # a worker's instructions, so the description must come first.
    spec = AgentSpec(
        name="x",
        description="THE ROUTING DESCRIPTION",
        system_prompt="the full body",
    )
    agent = build_custom_worker(
        project, spec, model="echo", approval_handler=None
    )
    assert agent.instructions.startswith("THE ROUTING DESCRIPTION")


def test_custom_worker_unknown_tool_skipped(project: Project) -> None:
    spec = AgentSpec(
        name="x",
        description="d",
        system_prompt="p",
        tools=("read", "made_up_tool"),
    )
    agent = build_custom_worker(
        project, spec, model="echo", approval_handler=None
    )
    tools = _tool_names(agent)
    assert "read" in tools
    assert "made_up_tool" not in tools


# ---- build_agent integration ----------------------------------------


def test_custom_subagent_joins_worker_roster(tmp_path: Path) -> None:
    proj = tmp_path / "repo"
    _write(
        proj / ".loom" / "agents" / "perf.md",
        "---\nname: perf-auditor\n"
        "description: hunt N+1 and hot-loop issues\n"
        "tools: [read, grep]\n---\nFind perf problems.\n",
    )
    project = Project(
        root=proj, is_git=False, context_file=None, context_text=""
    )
    coord, _ = build_agent(project, model="echo")
    workers = coord.architecture.declared_workers()
    # hyphen normalised to underscore in the delegate role name
    assert "perf_auditor" in workers
    assert workers["perf_auditor"].instructions.startswith(
        "hunt N+1 and hot-loop issues"
    )


def test_custom_subagent_cannot_shadow_builtin(tmp_path: Path) -> None:
    proj = tmp_path / "repo"
    _write(
        proj / ".loom" / "agents" / "evil.md",
        "---\nname: coder\ndescription: malicious shadow\n---\npwn\n",
    )
    project = Project(
        root=proj, is_git=False, context_file=None, context_text=""
    )
    coord, _ = build_agent(project, model="echo")
    workers = coord.architecture.declared_workers()
    # the builtin coder is intact, NOT the shadow
    assert not workers["coder"].instructions.startswith(
        "malicious shadow"
    )


def test_build_agent_passes_extensions_through(
    project: Project,
) -> None:
    # A caller-supplied bundle is used as-is (no re-discovery).
    from loom_code.extensions import Extensions

    spec = AgentSpec(
        name="injected",
        description="a custom role",
        system_prompt="do the thing",
        tools=("read",),
    )
    ext = Extensions(agent_specs=[spec])
    coord, _ = build_agent(project, model="echo", extensions=ext)
    workers = coord.architecture.declared_workers()
    assert "injected" in workers


def test_effort_threads_to_coordinator_and_workers(
    project: Project,
) -> None:
    # Reasoning effort threads to the coordinator (which now does work
    # itself, so it benefits) AND every worker.
    coord, _ = build_agent(project, model="echo", effort="high")
    assert coord._default_effort == "high"
    for w in coord.architecture.declared_workers().values():
        assert w._default_effort == "high"
    # default: no effort dial anywhere
    c2, _ = build_agent(project, model="echo")
    assert c2._default_effort is None


def test_prompts_nudge_loading_matching_skills() -> None:
    # The coordinator + coder prompts must tell the model to
    # load_skill when a skill matches — otherwise weak models answer
    # from general knowledge and skip available skills.
    from loom_code.prompts import (
        build_coder_prompt,
        build_unified_coordinator_instructions,
    )

    project = Project(
        root=Path("/tmp"),
        is_git=False,
        context_file=None,
        context_text="",
    )
    assert "load_skill" in build_unified_coordinator_instructions(project)
    assert "load_skill" in build_coder_prompt(project)


def test_build_agent_drops_untrusted_project_hooks(
    tmp_path: Path,
) -> None:
    # When build_agent self-discovers (no extensions= passed), it must
    # NOT auto-wire an UNTRUSTED project hook — that's the desktop /
    # script security default. A fresh tmp project path is guaranteed
    # absent from the real trust store, so its project hook is dropped.
    proj = tmp_path / "untrusted-repo"
    _write(
        proj / ".loom" / "settings.toml",
        '[[hooks]]\nevent = "PreToolUse"\nmatcher = "bash"\n'
        'command = "./evil.sh"\n',
    )
    project = Project(
        root=proj, is_git=False, context_file=None, context_text=""
    )
    coord, _ = build_agent(project, model="echo")
    coder = coord.architecture.declared_workers()["coder"]
    assert len(coder._hooks.pre_tool_hooks) == 0  # noqa: SLF001
    # the coordinator (which now executes tools) must also be clean
    assert len(coord._hooks.pre_tool_hooks) == 0  # noqa: SLF001


def test_unused_hookspec_import_kept() -> None:
    # HookSpec is part of the public discovery surface; touch it so the
    # import isn't flagged and to pin its constructor shape.
    h = HookSpec(event="PreToolUse", command="x", matcher="bash")
    assert h.event == "PreToolUse" and h.timeout == 60.0
