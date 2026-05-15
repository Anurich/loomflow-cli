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


def test_litellm_returns_none_to_skip_prompt() -> None:
    # litellm/<provider>/... — too many shapes to map reliably;
    # don't surprise advanced users with a wrong-key prompt.
    assert (
        required_env_for_model("litellm/groq/llama-3.1") is None
    )


def test_echo_model_needs_no_key() -> None:
    # The test fake.
    assert required_env_for_model("echo") is None
