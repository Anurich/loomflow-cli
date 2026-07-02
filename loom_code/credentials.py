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
# Remembers the user's chosen model across sessions, so launching
# loom-code reuses the last model (via /model or /set_model) instead of
# always reverting to the built-in default.
_MODEL_FILE = _CREDENTIALS_DIR / "model"


def save_preferred_model(model: str) -> None:
    """Persist the chosen model to ``~/.loom-code/model``. Best-effort."""
    try:
        _CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
        _MODEL_FILE.write_text(model.strip() + "\n", encoding="utf-8")
    except OSError:
        pass


def load_preferred_model() -> str | None:
    """The model saved on a previous run, or None if none/unreadable."""
    try:
        m = _MODEL_FILE.read_text(encoding="utf-8").strip()
        return m or None
    except OSError:
        return None

# --- LiteLLM provider registry ------------------------------------------
#
# loomflow routes ``litellm/<provider>/<model>`` through LiteLLM, which
# reads a provider-specific API key from the environment. Stock loomflow
# knows the *routing*; loom-code adds the friendly layer on top — so a
# user who picks one of these providers gets the same "paste your key +
# here's where to get one" flow as OpenAI/Anthropic, instead of a raw
# LiteLLM KeyError.
#
# Keyed by LiteLLM provider slug (the segment after ``litellm/``). Each
# entry is (env var LiteLLM reads, where to sign up). NVIDIA NIM is here
# because build.nvidia.com hands out a FREE tier — the cheapest way to
# drive loom-code, and what we use for the Terminal-Bench run. Adding a
# provider is one line: no other code changes needed, since the whole
# credential/prompt path is table-driven off this dict.
#
# Deliberately NOT exhaustive — only providers we've verified route
# cleanly. An unknown ``litellm/<x>/...`` still returns None from
# required_env_for_model (we don't guess and risk prompting for the
# wrong key), preserving the "advanced users are on their own" escape
# hatch for anything not listed here.
_BUILTIN_LITELLM_PROVIDERS: dict[str, tuple[str, str]] = {
    "nvidia_nim": ("NVIDIA_NIM_API_KEY", "https://build.nvidia.com"),
    "groq": ("GROQ_API_KEY", "https://console.groq.com/keys"),
    "together_ai": ("TOGETHERAI_API_KEY", "https://api.together.xyz/settings/api-keys"),
    "gemini": ("GEMINI_API_KEY", "https://aistudio.google.com/apikey"),
    "mistral": ("MISTRAL_API_KEY", "https://console.mistral.ai/api-keys"),
    "deepseek": ("DEEPSEEK_API_KEY", "https://platform.deepseek.com/api_keys"),
}


# Context-window sizes (tokens) for models loomflow's context_window_for
# doesn't recognise — chiefly litellm-routed providers. Without this,
# an unknown model falls back to a conservative 8192, which makes
# auto-compact fire far too early (compacting live state into a lossy
# summary mid-task). Keyed by a substring matched case-insensitively
# against the model string; first match wins, so order longest/most-
# specific first. Users can override per-model via
# ``auto_compact_at_tokens`` or a ``[[provider]] context_window`` block.
_MODEL_CONTEXT_WINDOWS: list[tuple[str, int]] = [
    # NVIDIA Nemotron family (build.nvidia.com) — 128K context.
    ("nemotron", 128_000),
    # Groq-hosted Llama 3.3 / 3.1 — 128K.
    ("llama-3.3", 128_000),
    ("llama-3.1", 128_000),
    # DeepSeek chat/coder — 64K.
    ("deepseek", 64_000),
]


def context_window_override(model: str) -> int | None:
    """A known context-window size for ``model`` when loomflow's own
    ``context_window_for`` would fall back to its conservative default.

    Returns None if we have no better number (caller keeps its default).
    Consulted before the fallback so litellm-routed models (NVIDIA
    Nemotron, Groq Llama, ...) get a realistic compaction threshold.
    A user ``[[provider]]`` block may add ``context_window`` to override.
    """
    lower = model.lower()
    slug = _litellm_provider_slug(model)
    if slug is not None:
        _, user_windows = _load_user_provider_windows()
        if slug in user_windows:
            return user_windows[slug]
    for needle, size in _MODEL_CONTEXT_WINDOWS:
        if needle in lower:
            return size
    return None


def _load_user_provider_windows() -> tuple[dict[str, str], dict[str, int]]:
    """Read optional ``context_window`` from user ``[[provider]]`` blocks.
    Returns ``({}, {slug: window})`` — the first element is a placeholder
    kept for shape symmetry with other loaders. Lenient, never crashes.
    """
    windows: dict[str, int] = {}
    settings = _CREDENTIALS_DIR / "settings.toml"
    try:
        import tomllib

        data = tomllib.loads(settings.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, windows
    raw = data.get("provider")
    if not isinstance(raw, list):
        return {}, windows
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        slug = str(entry.get("slug", "")).strip()
        cw = entry.get("context_window")
        if slug and isinstance(cw, int) and cw > 0:
            windows[slug] = cw
    return {}, windows


def _load_user_providers() -> dict[str, tuple[str, str]]:
    """Read ``[[provider]]`` blocks from ``~/.loom-code/settings.toml`` so a
    user can register ANY OpenAI-compatible / LiteLLM provider WITHOUT
    editing loom-code's source — the general "integrate any API" hatch.

    Each block::

        [[provider]]
        slug = "myproxy"          # the litellm/<slug>/ segment
        env = "MYPROXY_API_KEY"   # env var LiteLLM reads for the key
        signup = "https://..."    # optional, shown in the key prompt
        alias = "myai"            # optional, short --model prefix
        # For a bare OpenAI-compatible endpoint, point litellm at it:
        # slug = "openai" with api_base set via env, per LiteLLM docs.

    Returns ``{slug: (env, signup)}``. Lenient: a missing file, bad TOML,
    or a malformed block yields ``{}`` / skips the block rather than
    crashing — same never-abort posture as the extensions discovery.
    User entries OVERRIDE built-ins of the same slug (lets a user repoint
    a provider's env var). Aliases are collected separately.
    """
    out, _ = _load_user_providers_and_aliases()
    return out


def _load_user_providers_and_aliases() -> (
    tuple[dict[str, tuple[str, str]], dict[str, tuple[str, bool]]]
):
    """Backing loader for both the provider registry and the alias map.

    Returns ``(providers, aliases)`` where ``aliases`` maps a short
    ``--model`` prefix to ``(slug, keep_prefix=False)``. Kept private;
    :func:`litellm_providers` and :func:`normalize_model` consume it.
    """
    providers: dict[str, tuple[str, str]] = {}
    aliases: dict[str, tuple[str, bool]] = {}
    settings = _CREDENTIALS_DIR / "settings.toml"
    try:
        import tomllib

        data = tomllib.loads(settings.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return providers, aliases
    raw = data.get("provider")
    if not isinstance(raw, list):
        return providers, aliases
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        slug = str(entry.get("slug", "")).strip()
        env = str(entry.get("env", "")).strip()
        if not slug or not env:
            continue
        signup = str(entry.get("signup", "")).strip() or (
            "your provider's dashboard"
        )
        providers[slug] = (env, signup)
        alias = str(entry.get("alias", "")).strip().lower()
        if alias:
            aliases[alias] = (slug, False)
    return providers, aliases


def quiet_litellm_model_warnings(model: str) -> None:
    """Silence loomflow's two "unknown model" ``UserWarning``s for a
    litellm-routed model, where they're expected noise rather than a
    real problem:

    * ``context_window_for: unknown model ...`` — loom-code supplies the
      real window via :func:`context_window_override`, so loomflow's
      conservative fallback is never actually used.
    * ``cost estimation: unknown model ...`` — providers like NVIDIA's
      free tier have no entry in loomflow's pricing table; reporting
      $0.00 is correct for a free model, and budget caps are moot.

    Scoped to litellm models ONLY: a native ``gpt-*``/``claude-*`` that
    somehow warns is a genuine signal we must not hide. No-op otherwise.
    """
    if not model.lower().startswith("litellm/"):
        return
    import warnings

    for pattern in (
        r"context_window_for: unknown model.*",
        r"cost estimation: unknown model.*",
    ):
        warnings.filterwarnings(
            "ignore", message=pattern, category=UserWarning
        )


def litellm_providers() -> dict[str, tuple[str, str]]:
    """The effective provider registry: built-ins overlaid with any
    user-declared ``[[provider]]`` blocks. Recomputed on each call so a
    settings.toml edit takes effect without restarting the process."""
    merged = dict(_BUILTIN_LITELLM_PROVIDERS)
    merged.update(_load_user_providers())
    return merged


def _litellm_provider_slug(model: str) -> str | None:
    """For a ``litellm/<provider>/<model>`` string, return the provider
    slug (``nvidia_nim``, ``groq``, ...) if it's one in the effective
    registry (built-ins + user-declared), else None. Case-insensitive on
    the prefix; the slug is matched exactly."""
    lower = model.lower()
    if not lower.startswith("litellm/"):
        return None
    rest = model[len("litellm/"):]
    slug = rest.split("/", 1)[0]
    return slug if slug in litellm_providers() else None


# First-party signup links. LiteLLM provider links are merged in at
# lookup time by :func:`signup_url_for` so user-declared providers get
# their link too, without a module-load-time snapshot going stale.
_FIRST_PARTY_SIGNUP_URL = {
    "OPENAI_API_KEY": "https://platform.openai.com/api-keys",
    "ANTHROPIC_API_KEY": "https://console.anthropic.com/settings/keys",
}


def signup_url_for(env_name: str) -> str:
    """Where to get a key for ``env_name`` — first-party or any provider
    in the effective registry. Falls back to a generic hint."""
    if env_name in _FIRST_PARTY_SIGNUP_URL:
        return _FIRST_PARTY_SIGNUP_URL[env_name]
    for env, url in litellm_providers().values():
        if env == env_name:
            return url
    return "your provider's dashboard"


def normalize_model(model: str) -> str:
    """Expand short provider aliases into the ``litellm/<provider>/``
    form loomflow's resolver understands.

    Lets a user type the friendly ``nvidia/nemotron-...`` instead of the
    verbose ``litellm/nvidia_nim/nvidia/nemotron-...``. Only rewrites a
    leading ``<alias>/`` for an alias we recognise; everything else —
    already-prefixed ``litellm/...``, native ``gpt-*``/``claude-*``,
    ``ollama/...`` — passes through untouched. Idempotent: running it on
    an already-normalised string is a no-op.

    ``nvidia`` is the alias for the ``nvidia_nim`` LiteLLM provider (the
    provider slug isn't an obvious thing to type). NVIDIA's own model
    IDs are vendor-namespaced (``nvidia/nemotron-...``, ``meta/llama-...``),
    so the ``nvidia`` alias PRESERVES the segment as part of the model
    ID rather than consuming it — ``nvidia/nemotron-x`` becomes
    ``litellm/nvidia_nim/nvidia/nemotron-x``, not ``.../nemotron-x``.
    Aliases whose provider uses flat model IDs consume the segment.
    """
    # alias -> (litellm provider slug, keep_prefix). keep_prefix=True
    # re-attaches the alias segment to the model ID (for providers with
    # vendor-namespaced IDs like NVIDIA); False consumes it. Built-in
    # aliases first; user-declared ``[[provider]] alias = ...`` entries
    # merged on top so a custom provider gets a short prefix too.
    aliases: dict[str, tuple[str, bool]] = {
        "nvidia": ("nvidia_nim", True),
        "nvidia_nim": ("nvidia_nim", False),
        "groq": ("groq", False),
        "together": ("together_ai", False),
        "together_ai": ("together_ai", False),
        "gemini": ("gemini", False),
        "mistral": ("mistral", False),
        "deepseek": ("deepseek", False),
    }
    _, user_aliases = _load_user_providers_and_aliases()
    aliases.update(user_aliases)
    spec = model.strip()
    if spec.lower().startswith("litellm/"):
        return spec
    head, sep, rest = spec.partition("/")
    entry = aliases.get(head.lower())
    if sep and entry and rest:
        slug, keep_prefix = entry
        model_id = f"{head}/{rest}" if keep_prefix else rest
        return f"litellm/{slug}/{model_id}"
    return spec


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
        # A KNOWN LiteLLM provider (nvidia_nim, groq, ...) maps to its
        # env var so we prompt for the right key with a signup link.
        # An UNKNOWN litellm/<x>/... still returns None — advanced
        # users routing through an unlisted provider set the env
        # themselves, and we won't surprise them with a wrong-key
        # prompt we can't reliably name.
        slug = _litellm_provider_slug(model)
        if slug is not None:
            return litellm_providers()[slug][0]
        return None
    if "claude" in lower:
        return "ANTHROPIC_API_KEY"
    # Everything else (gpt-*, o-series, etc.) → OpenAI.
    return "OPENAI_API_KEY"


def cheap_model_for(model: str) -> str | None:
    """The CHEAP, fast sibling of ``model`` in the SAME provider —
    for low-stakes utility calls (compaction summaries, tool-result
    compression, /goal's DONE/NOT_DONE checker).

    Staying in-provider avoids switching to an account that may be
    unfunded (a set key doesn't prove credits). Returns ``None``
    when no cheap sibling is usable (local / litellm / echo models,
    or the cheap model's key isn't set) — callers fall back to the
    main model.
    """
    lower = model.lower()
    # Escape BEFORE the substring checks: ``litellm/anthropic/claude-*``
    # contains "claude" but is routed through a proxy/Bedrock the user
    # chose — silently sending compaction summaries or tool output
    # direct to api.anthropic.com would bypass that routing. Local and
    # fake models have no cheap sibling either.
    if lower == "echo" or lower.startswith(("ollama/", "litellm/")):
        return None
    if "claude" in lower:
        target = "claude-haiku-4-5"
    elif lower.startswith(("gpt", "o1", "o3", "o4")):
        target = "gpt-4.1-mini"
    else:
        # Unknown provider — don't guess.
        return None
    if lower == target:
        return target
    env = required_env_for_model(target)
    if env is None or os.environ.get(env):
        return target
    return None


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
    signup = signup_url_for(env_name)
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
