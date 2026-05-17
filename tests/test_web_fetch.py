"""Tests for loom_code.web_fetch — URL normalization + tool shape.

Network calls (httpx) are exercised through a monkeypatched
``httpx.AsyncClient`` — no real HTTP fires from this suite. The
pure URL-rewriting + schema-shape assertions don't need any
patching at all.
"""

from __future__ import annotations

import httpx
import pytest

from loom_code.web_fetch import _normalize_url, web_fetch_tool  # noqa: I001

# --- URL normalization (pure) ---------------------------------


def test_https_passes_through_unchanged() -> None:
    norm, err = _normalize_url("https://example.com/foo")
    assert err is None
    assert norm == "https://example.com/foo"


def test_http_upgrades_to_https() -> None:
    norm, err = _normalize_url("http://example.com/foo")
    assert err is None
    assert norm == "https://example.com/foo"


def test_github_blob_rewrites_to_raw() -> None:
    norm, err = _normalize_url(
        "https://github.com/Anurich/LoomFlow/blob/main/README.md"
    )
    assert err is None
    assert norm == (
        "https://raw.githubusercontent.com/"
        "Anurich/LoomFlow/main/README.md"
    )


def test_github_www_subdomain_also_rewrites() -> None:
    norm, err = _normalize_url(
        "https://www.github.com/foo/bar/blob/main/x.py"
    )
    assert err is None
    assert norm is not None
    assert "raw.githubusercontent.com" in norm


def test_github_tree_url_rejected_with_directive() -> None:
    # /tree/ pages are React HTML (~700kB) and waste tokens; the
    # tool refuses and points the model at the GitHub contents API.
    url = "https://github.com/foo/bar/tree/main"
    norm, err = _normalize_url(url)
    assert norm is None
    assert err is not None
    assert err.startswith("ERROR:")
    # Suggested API URL is constructed from the parsed owner/repo/ref.
    assert "api.github.com/repos/foo/bar/contents" in err
    assert "?ref=main" in err


def test_github_tree_url_with_subpath_rejected() -> None:
    # Ref + path parse correctly into the suggested API URL.
    url = "https://github.com/foo/bar/tree/develop/src/lib"
    norm, err = _normalize_url(url)
    assert norm is None
    assert err is not None
    assert "api.github.com/repos/foo/bar/contents/src/lib" in err
    assert "?ref=develop" in err


def test_non_http_scheme_rejected() -> None:
    norm, err = _normalize_url("file:///etc/passwd")
    assert norm is None
    assert err is not None
    assert err.startswith("ERROR:")


def test_empty_url_rejected() -> None:
    norm, err = _normalize_url("")
    assert norm is None
    assert err is not None
    assert err.startswith("ERROR:")


def test_whitespace_only_url_rejected() -> None:
    norm, err = _normalize_url("   \n  ")
    assert norm is None
    assert err is not None


# --- Tool shape -----------------------------------------------


def test_tool_has_expected_shape() -> None:
    t = web_fetch_tool()
    assert t.name == "web_fetch"
    assert "url" in t.input_schema["properties"]
    assert t.input_schema["required"] == ["url"]
    assert not t.destructive
    assert len(t.description) > 50


def test_tool_name_is_overridable() -> None:
    t = web_fetch_tool(name="fetch_url")
    assert t.name == "fetch_url"


# --- Behaviour via monkeypatched httpx ------------------------


class _FakeResponse:
    """Minimal duck-type of httpx.Response — only what _fetch reads."""

    def __init__(
        self,
        *,
        status_code: int,
        text: str,
        url: str = "https://example.com/x",
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.content = text.encode()
        self.url = url


class _FakeAsyncClient:
    """Stand-in for httpx.AsyncClient. Class-level ``response``
    is what the test sets up; instances are throwaway."""

    response: object = _FakeResponse(status_code=200, text="ok")

    def __init__(self, **kwargs: object) -> None:
        pass

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass

    async def get(self, url: str) -> _FakeResponse:
        if isinstance(self.response, Exception):
            raise self.response
        assert isinstance(self.response, _FakeResponse)
        return self.response


@pytest.mark.anyio
async def test_fetch_returns_body_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeAsyncClient.response = _FakeResponse(
        status_code=200,
        text="hello world",
        url="https://example.com/hi",
    )
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    tool = web_fetch_tool()
    out = await tool.execute({"url": "https://example.com/hi"})
    assert "status: 200" in out
    assert "hello world" in out
    assert "https://example.com/hi" in out  # final URL surfaced


@pytest.mark.anyio
async def test_fetch_returns_error_string_on_http_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeAsyncClient.response = httpx.ConnectError("boom")
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    tool = web_fetch_tool()
    out = await tool.execute({"url": "https://example.com/x"})
    assert out.startswith("ERROR:")
    assert "boom" in out


@pytest.mark.anyio
async def test_fetch_rejects_oversized_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    huge = "x" * 100
    _FakeAsyncClient.response = _FakeResponse(
        status_code=200, text=huge
    )
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    tool = web_fetch_tool(max_bytes=50)
    out = await tool.execute({"url": "https://example.com/big"})
    assert out.startswith("ERROR:")
    assert "exceeds" in out


@pytest.mark.anyio
async def test_fetch_non_http_url_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If _normalize_url rejects, we should never even attempt
    # to construct the httpx client.
    called = {"get": False}

    class _ShouldNotCall(_FakeAsyncClient):
        async def get(self, url: str) -> _FakeResponse:
            called["get"] = True
            return await super().get(url)

    monkeypatch.setattr(httpx, "AsyncClient", _ShouldNotCall)
    tool = web_fetch_tool()
    out = await tool.execute({"url": "file:///etc/passwd"})
    assert out.startswith("ERROR:")
    assert not called["get"]
