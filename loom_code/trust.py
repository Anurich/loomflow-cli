"""Trust gating for project-scope hooks.

A user-scope hook (``~/.loom-code/settings.toml``) is something YOU
wrote — it runs without ceremony. A project-scope hook
(``<repo>/.loom/settings.toml``) is a stranger's code: clone a repo,
open it in loom-code, and its hooks would otherwise run shell commands
on your machine automatically. That's a real code-execution vector, so
project hooks are gated:

* The first time loom-code sees a repo's project hooks — or whenever
  the set of commands changes — it shows you the commands and asks
  whether to trust them.
* Your answer is remembered (keyed by repo path + a fingerprint of the
  hook commands) in ``~/.loom-code/trusted_hooks.json``, so you're
  asked again only if the hooks change.

Skills and subagents are NOT gated here: they run only when the model
invokes them, and every tool they call still goes through the normal
approval gate. Hooks are the one thing that fires automatically, so
they're the one thing that needs up-front consent.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from .extensions import Extensions, HookSpec
from .extensions import discover as _discover

_DEFAULT_TRUST_STORE = Path.home() / ".loom-code" / "trusted_hooks.json"


def deny_untrusted(_specs: list[HookSpec]) -> bool:
    """A non-interactive trust prompt that always declines.

    The secure default for callers that can't show a prompt (the
    desktop sidecar, scripts, ``build_agent``'s self-discovery path):
    project hooks run ONLY if already recorded as trusted; everything
    not-yet-trusted is dropped rather than auto-run."""
    return False


def discover_trusted(
    project_root: Path,
    *,
    prompt: Callable[[list[HookSpec]], bool] = deny_untrusted,
    user_dir: Path | None = None,
    trust_store: Path | None = None,
) -> Extensions:
    """Discover ``.loom`` extensions and apply the project-hook trust
    gate in one call.

    The convenience entry every non-REPL caller should use: it returns
    a bundle whose project hooks are already trust-filtered. ``prompt``
    defaults to :func:`deny_untrusted` (secure, non-interactive); pass
    an interactive callback to ask the user."""
    ext = _discover(project_root, user_dir=user_dir)
    return filter_trusted_hooks(
        ext,
        project_root=project_root,
        prompt=prompt,
        trust_store=trust_store,
    )


def filter_trusted_hooks(
    extensions: Extensions,
    *,
    project_root: Path,
    prompt: Callable[[list[HookSpec]], bool],
    trust_store: Path | None = None,
) -> Extensions:
    """Return ``extensions`` with untrusted project hooks removed.

    User-scope hooks always pass through. Project-scope hooks pass only
    when their fingerprint is already recorded as trusted for
    ``project_root``, or when ``prompt(project_specs)`` returns True
    (in which case the fingerprint is recorded so we don't ask again).
    On denial the project hooks are dropped — skills, subagents, and
    user hooks are kept untouched.

    ``trust_store`` overrides the on-disk record path (tests pass a tmp
    file)."""
    project_hooks = [
        h for h in extensions.hook_specs if h.source == "project"
    ]
    if not project_hooks:
        return extensions

    if is_trusted(project_root, project_hooks, trust_store=trust_store):
        return extensions  # already trusted, unchanged

    if prompt(project_hooks):
        record_trust(project_root, project_hooks, trust_store=trust_store)
        return extensions

    # Denied — strip project hooks, keep everything else.
    kept = [h for h in extensions.hook_specs if h.source != "project"]
    return Extensions(
        skill_paths=extensions.skill_paths,
        agent_specs=extensions.agent_specs,
        hook_specs=kept,
    )


def is_trusted(
    project_root: Path,
    project_hooks: list[HookSpec],
    *,
    trust_store: Path | None = None,
) -> bool:
    """Has the user already trusted *exactly* these project hooks for
    this repo?

    True when the project hooks' fingerprint matches the recorded one
    for ``project_root`` (or when there are no project hooks). Lets an
    async caller (the desktop sidecar) decide whether to prompt without
    going through the sync ``filter_trusted_hooks`` callback."""
    if not project_hooks:
        return True
    store = trust_store if trust_store is not None else _DEFAULT_TRUST_STORE
    key = str(project_root.resolve())
    return _load(store).get(key) == _fingerprint(project_hooks)


def record_trust(
    project_root: Path,
    project_hooks: list[HookSpec],
    *,
    trust_store: Path | None = None,
) -> None:
    """Record the user's decision to trust exactly these project hooks
    for this repo, so they aren't re-prompted until the hooks change."""
    if not project_hooks:
        return
    store = trust_store if trust_store is not None else _DEFAULT_TRUST_STORE
    _record(
        store, str(project_root.resolve()), _fingerprint(project_hooks)
    )


def _fingerprint(specs: list[HookSpec]) -> str:
    """A stable hash over the hooks' executable surface. Changing any
    command / matcher / event / timeout invalidates trust (re-prompts);
    reordering does not."""
    payload = sorted(
        (s.event, s.matcher, s.command, s.timeout) for s in specs
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _load(store: Path) -> dict[str, str]:
    if not store.exists():
        return {}
    try:
        data = json.loads(store.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _record(store: Path, key: str, fingerprint: str) -> None:
    data = _load(store)
    data[key] = fingerprint
    try:
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        # A failed write just means we'll ask again next time — never
        # fatal to the session.
        pass
