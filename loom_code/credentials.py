"""User-level API-key storage for loom-code.

Persists keys at ``~/.loom-code/credentials`` (chmod 600) so users
don't have to ``export OPENAI_API_KEY=...`` in every new shell.
Same pattern ``gh``, ``aws cli``, ``aider`` use.

Flow:

1. On startup, :func:`load_credentials` reads the file and sets
   any missing env vars. Env always wins — if the user already
   ``export``ed something in their shell, we don't overwrite it.
2. :func:`ensure_key_for_model` checks if the chosen model has a
   key it can use. If not, prompts the user inline (hidden
   input), saves to the credentials file, and updates env.
3. loomflow's model resolver reads env in the normal way — it
   has no idea this layer exists.

Security: the file is written with ``chmod 600`` (user-only
read/write). Plaintext on disk; an OS keyring would be stronger
but adds a dependency we don't have yet. Worth revisiting.
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path

from rich.console import Console

# User-level config. Per-project state stays at ``<project>/.loom/``
# (lowercase + dot, no hyphen) — the hyphen here is the on-purpose
# differentiator: "this is loom-code's GLOBAL config, not a project."
_CREDENTIALS_DIR = Path.home() / ".loom-code"
_CREDENTIALS_FILE = _CREDENTIALS_DIR / "credentials"

# Where users go to grab a key, surfaced in the prompt so they
# don't have to guess.
_KEY_SIGNUP_URL = {
    "OPENAI_API_KEY": "https://platform.openai.com/api-keys",
    "ANTHROPIC_API_KEY": "https://console.anthropic.com/settings/keys",
}


def load_credentials() -> None:
    """Read ``~/.loom-code/credentials`` and populate any env vars
    that aren't already set. Silent no-op if the file is missing.

    The file format is plain ``KEY=value`` lines, comments with
    ``#``, blank lines ignored. Surrounding quotes on the value
    (single or double) are stripped.
    """
    if not _CREDENTIALS_FILE.exists():
        return
    for raw in _CREDENTIALS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        # Env wins — if the user already ``export``ed something,
        # we don't second-guess them.
        if name and value and not os.environ.get(name):
            os.environ[name] = value


def save_credential(name: str, value: str) -> None:
    """Write or update ``name=value`` in
    ``~/.loom-code/credentials`` with ``chmod 600``. The dir is
    created if missing. Preserves comments + other entries; only
    replaces the line for ``name`` if it already exists."""
    _CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    replaced = False
    if _CREDENTIALS_FILE.exists():
        for raw in _CREDENTIALS_FILE.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if (
                stripped
                and not stripped.startswith("#")
                and "=" in stripped
                and stripped.split("=", 1)[0].strip() == name
            ):
                out.append(f"{name}={value}")
                replaced = True
            else:
                out.append(raw)
    if not replaced:
        out.append(f"{name}={value}")
    _CREDENTIALS_FILE.write_text("\n".join(out) + "\n", encoding="utf-8")
    # User-only read/write; cheap defence vs. accidental world-
    # readability (e.g. if HOME ends up shared). POSIX-only — on
    # Windows chmod can't express owner-only perms (the file inherits
    # NTFS ACLs from the user profile dir, which is already private).
    if os.name == "posix":
        _CREDENTIALS_FILE.chmod(0o600)


def required_env_for_model(model: str) -> str | None:
    """Which env var loomflow's resolver needs to talk to ``model``.

    Returns ``None`` for models that need no key (local Ollama,
    EchoModel, and ``litellm/<provider>`` where the answer
    depends on which provider — we can't reliably guess).
    """
    lower = model.lower()
    if lower == "echo":
        return None
    if lower.startswith("ollama/"):
        return None
    if lower.startswith("litellm/"):
        # Could split further (litellm/groq/* → GROQ_API_KEY,
        # litellm/together_ai/* → TOGETHERAI_API_KEY, ...). Skip
        # for now — `litellm/` users are advanced enough to set
        # the env themselves; we won't surprise them with a prompt.
        return None
    if "claude" in lower:
        return "ANTHROPIC_API_KEY"
    # Everything else (gpt-*, o-series, etc.) → OpenAI.
    return "OPENAI_API_KEY"


def ensure_key_for_model(model: str, console: Console) -> bool:
    """If ``model`` needs a key that isn't set, prompt the user
    for one (hidden input), save it, and load it into env. Returns
    ``True`` when we can proceed, ``False`` if the user cancelled.

    Sync because it's called from ``main()`` before any async
    event loop is running — and ``getpass`` is sync-blocking
    anyway.
    """
    env_name = required_env_for_model(model)
    if env_name is None:
        return True
    if os.environ.get(env_name):
        return True

    console.print()
    console.print(
        f"  [yellow]No {env_name} set.[/yellow] "
        f"loom-code needs it to use [cyan]{model}[/cyan]."
    )
    signup = _KEY_SIGNUP_URL.get(
        env_name, "your provider's dashboard"
    )
    console.print(f"  Get one at [dim]{signup}[/dim].\n")
    try:
        # getpass hides the input so the key doesn't appear in
        # the terminal or shell history.
        value = getpass.getpass(f"  Paste your {env_name}: ")
    except (EOFError, KeyboardInterrupt):
        console.print("\n  [dim]cancelled[/dim]")
        return False
    value = value.strip()
    if not value:
        console.print("  [yellow]no key entered — aborting[/yellow]")
        return False
    save_credential(env_name, value)
    os.environ[env_name] = value
    console.print(
        f"  [green]✓[/green] saved to "
        f"[dim]{_CREDENTIALS_FILE}[/dim] (chmod 600)\n"
    )
    return True
