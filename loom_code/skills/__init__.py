"""Built-in skills shipped with loom-code.

Each subdirectory under here is a loomflow skill — ``SKILL.md``
(frontmatter + body) plus optional ``tools.py`` for Mode B Python
tools. Loomflow's :class:`SkillRegistry` discovers them from a
directory path; ``loom_code.agent.build_agent`` points at this
package's resource path so all bundled skills are wired into the
coordinator's surface automatically.

Shipped today:

* ``graphify/`` — knowledge-graph extraction over the project,
  using graphify's Python primitives directly (no subprocess,
  no MCP). Build / query / path / explain tools, prefixed
  ``graphify__*`` once loaded.
"""
