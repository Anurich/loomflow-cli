"""Tests for the user-level credentials layer.

Two failure modes worth locking down:

* Reading a credentials file that *exists* but is malformed (extra
  spaces, quoted values, comments, blank lines) — a brittle parser
  would silently mis-set keys and the user would get cryptic 401s.
* ``required_env_for_model`` getting a model string wrong — would
  either prompt for the wrong key or fail to prompt at all,
  surprising the user.

The interactive ``ensure_key_for_model`` is not unit-tested here
(it calls ``getpass`` which needs a TTY).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loom_code import credentials as creds_mod
from loom_code.credentials import (
    load_credentials,
    required_env_for_model,
    save_credential,
)


@pytest.fixture
def tmp_creds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Redirect the credentials file into a tmp path for each test
    so we never touch the real ~/.loom-code/."""
    tmp_dir = tmp_path / ".loom-code"
    tmp_file = tmp_dir / "credentials"
    monkeypatch.setattr(creds_mod, "_CREDENTIALS_DIR", tmp_dir)
    monkeypatch.setattr(creds_mod, "_CREDENTIALS_FILE", tmp_file)
    return tmp_file


def test_save_then_load_roundtrip(
    tmp_creds: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    save_credential("OPENAI_API_KEY", "sk-test-12345")
    # File must exist + be chmod 600 (user-only read/write).
    assert tmp_creds.exists()
    assert tmp_creds.stat().st_mode & 0o777 == 0o600
    # load_credentials picks it up.
    load_credentials()
    import os
    assert os.environ.get("OPENAI_API_KEY") == "sk-test-12345"


def test_save_updates_existing_entry(
    tmp_creds: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Replacing an existing key must not append a duplicate line.
    save_credential("OPENAI_API_KEY", "old-value")
    save_credential("OPENAI_API_KEY", "new-value")
    text = tmp_creds.read_text()
    assert text.count("OPENAI_API_KEY=") == 1
    assert "new-value" in text
    assert "old-value" not in text


def test_load_preserves_existing_env(
    tmp_creds: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Env always wins. If the user `export`ed a key, the file
    # must NOT overwrite it (which would silently undo their
    # intent).
    save_credential("OPENAI_API_KEY", "file-value")
    monkeypatch.setenv("OPENAI_API_KEY", "env-value")
    load_credentials()
    import os
    assert os.environ["OPENAI_API_KEY"] == "env-value"


def test_load_handles_messy_lines(
    tmp_creds: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Comments, blanks, quoted values — all should be tolerated.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    tmp_creds.parent.mkdir(parents=True, exist_ok=True)
    tmp_creds.write_text(
        '# comment\n'
        '\n'
        '   OPENAI_API_KEY = "sk-quoted"   \n'
        "ANTHROPIC_API_KEY='sk-singlequoted'\n"
    )
    load_credentials()
    import os
    assert os.environ["OPENAI_API_KEY"] == "sk-quoted"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-singlequoted"


def test_load_no_file_is_silent_noop(tmp_creds: Path) -> None:
    # Brand-new install — no file yet. Must not error.
    assert not tmp_creds.exists()
    load_credentials()  # should just return


# ---- required_env_for_model -----------------------------------------------


def test_anthropic_models_route_to_anthropic_key() -> None:
    assert (
        required_env_for_model("claude-sonnet-4-6")
        == "ANTHROPIC_API_KEY"
    )
    assert (
        required_env_for_model("claude-opus-4-7")
        == "ANTHROPIC_API_KEY"
    )


def test_openai_default_route() -> None:
    # Anything that doesn't match another rule → OpenAI.
    assert required_env_for_model("gpt-4.1-mini") == "OPENAI_API_KEY"
    assert required_env_for_model("o4-mini") == "OPENAI_API_KEY"


def test_ollama_needs_no_key() -> None:
    # Local, free, offline — no key prompt should fire.
    assert required_env_for_model("ollama/llama3") is None
    assert (
        required_env_for_model("ollama/qwen2.5-coder") is None
    )


def test_litellm_known_provider_maps_to_its_key() -> None:
    # A KNOWN LiteLLM provider (in the registry) prompts for the
    # right env var + signup link.
    assert (
        required_env_for_model("litellm/nvidia_nim/nvidia/nemotron")
        == "NVIDIA_NIM_API_KEY"
    )
    assert (
        required_env_for_model("litellm/groq/llama-3.1")
        == "GROQ_API_KEY"
    )
    assert (
        required_env_for_model("litellm/together_ai/some-model")
        == "TOGETHERAI_API_KEY"
    )


def test_litellm_unknown_provider_returns_none() -> None:
    # An UNKNOWN litellm/<x>/... still returns None — we don't guess
    # and risk prompting for a key we can't reliably name.
    assert (
        required_env_for_model("litellm/some_exotic_proxy/m") is None
    )


def test_nvidia_alias_normalizes_to_litellm_form() -> None:
    from loom_code.credentials import normalize_model

    assert (
        normalize_model("nvidia/nemotron-nano-9b-v2")
        == "litellm/nvidia_nim/nvidia/nemotron-nano-9b-v2"
    )
    # Idempotent + passthrough for native/prefixed strings.
    assert (
        normalize_model("litellm/nvidia_nim/nvidia/x")
        == "litellm/nvidia_nim/nvidia/x"
    )
    assert normalize_model("gpt-4.1-mini") == "gpt-4.1-mini"
    assert normalize_model("claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert normalize_model("ollama/llama3") == "ollama/llama3"


def test_echo_model_needs_no_key() -> None:
    # The test fake.
    assert required_env_for_model("echo") is None


# ---- user-declared providers (settings.toml [[provider]]) -----------------


def test_user_provider_registers_via_settings_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A user can integrate ANY provider without editing source: a
    # [[provider]] block in ~/.loom-code/settings.toml adds it to the
    # registry, so its key is prompted for and its alias normalizes.
    from loom_code.credentials import (
        litellm_providers,
        normalize_model,
        signup_url_for,
    )

    monkeypatch.setattr(creds_mod, "_CREDENTIALS_DIR", tmp_path)
    (tmp_path / "settings.toml").write_text(
        '[[provider]]\n'
        'slug = "myproxy"\n'
        'env = "MYPROXY_API_KEY"\n'
        'signup = "https://example.com/keys"\n'
        'alias = "myai"\n'
    )
    # Registry now includes the custom provider alongside built-ins.
    reg = litellm_providers()
    assert reg["myproxy"] == ("MYPROXY_API_KEY", "https://example.com/keys")
    assert "nvidia_nim" in reg  # built-ins still present
    # required_env_for_model + signup link route to it.
    assert (
        required_env_for_model("litellm/myproxy/some-model")
        == "MYPROXY_API_KEY"
    )
    assert signup_url_for("MYPROXY_API_KEY") == "https://example.com/keys"
    # The declared alias normalizes.
    assert (
        normalize_model("myai/some-model")
        == "litellm/myproxy/some-model"
    )


def test_malformed_settings_toml_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bad TOML must not crash startup — falls back to built-ins only.
    from loom_code.credentials import litellm_providers

    monkeypatch.setattr(creds_mod, "_CREDENTIALS_DIR", tmp_path)
    (tmp_path / "settings.toml").write_text("this is not valid = = toml [[")
    reg = litellm_providers()
    assert "nvidia_nim" in reg  # built-ins survive
    assert "myproxy" not in reg
