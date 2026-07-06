"""Grab an image off the system clipboard (for Ctrl-V vision paste).

Terminals only deliver TEXT through the normal paste path (bracketed
paste), so a screenshot on the clipboard never reaches the input box
by itself. The TUI binds Ctrl-V to this module: if the clipboard holds
an image, we return its bytes and the prompt gains an ``[image-N]``
placeholder while the bytes ride to the model via loomflow's
``_loom_images`` metadata.

Per-platform, no third-party deps:

* macOS  — ``osascript`` coerces the clipboard to PNG (``«class
  PNGf»``) and prints it as a hex blob we decode.
* Linux  — ``wl-paste`` (Wayland) then ``xclip`` (X11), asking for
  ``image/png``.
* Windows — PowerShell's ``Get-Clipboard -Format Image`` saved to PNG
  via .NET, printed base64.

All failures (no image on the clipboard, tool missing, weird format)
return ``None`` — the caller shows a gentle hint, never an error.
"""

from __future__ import annotations

import base64
import subprocess
import sys

_TIMEOUT_S = 10.0


def grab_clipboard_image() -> tuple[bytes, str] | None:
    """Return ``(png_bytes, "image/png")`` from the clipboard, or
    ``None`` when there's no image (or no way to read one)."""
    try:
        if sys.platform == "darwin":
            return _grab_macos()
        if sys.platform.startswith("linux"):
            return _grab_linux()
        if sys.platform == "win32":
            return _grab_windows()
    except Exception:  # noqa: BLE001 — clipboard is best-effort
        return None
    return None


def _grab_macos() -> tuple[bytes, str] | None:
    # ``the clipboard as «class PNGf»`` errors when the clipboard has
    # no image — that error IS the "no image" signal.
    proc = subprocess.run(
        ["osascript", "-e", "the clipboard as «class PNGf»"],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
    )
    if proc.returncode != 0:
        return None
    # Output shape: «data PNGf89504E47...» — hex after the tag.
    out = proc.stdout.strip()
    start = out.find("PNGf")
    if start == -1:
        return None
    hexdata = out[start + 4 :].rstrip("»").strip()
    if not hexdata:
        return None
    data = bytes.fromhex(hexdata)
    return (data, "image/png") if data else None


def _grab_linux() -> tuple[bytes, str] | None:
    for cmd in (
        ["wl-paste", "-t", "image/png"],
        ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
    ):
        try:
            proc = subprocess.run(
                cmd, capture_output=True, timeout=_TIMEOUT_S
            )
        except FileNotFoundError:
            continue
        if proc.returncode == 0 and proc.stdout[:8] == b"\x89PNG\r\n\x1a\n":
            return (proc.stdout, "image/png")
    return None


def _grab_windows() -> tuple[bytes, str] | None:
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$img = [Windows.Forms.Clipboard]::GetImage(); "
        "if ($img) { $ms = New-Object IO.MemoryStream; "
        "$img.Save($ms, [Drawing.Imaging.ImageFormat]::Png); "
        "[Convert]::ToBase64String($ms.ToArray()) }"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-STA", "-Command", script],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
    )
    b64 = proc.stdout.strip()
    if proc.returncode != 0 or not b64:
        return None
    data = base64.b64decode(b64)
    return (data, "image/png") if data else None


def to_loom_image(data: bytes, media_type: str) -> dict[str, str]:
    """The dict shape loomflow's ``_loom_images`` metadata accepts."""
    return {
        "data": base64.b64encode(data).decode("ascii"),
        "media_type": media_type,
    }


def model_supports_vision(model: str) -> bool | None:
    """True/False from litellm's model-capability DB; None = unknown.

    The honesty guard: a text-only model that receives multimodal
    content via an OpenAI-compatible endpoint often ANSWERS anyway —
    the server silently drops the image parts and the model
    confabulates a description from the filename + conversation
    (observed live: a store banner "described" as a GitHub diff).
    Unknown (None) stays silent — the DB can lag brand-new models,
    and a false warning is worse than none."""
    try:
        import litellm

        m = str(model)
        if m.startswith("litellm/"):
            m = m[len("litellm/") :]
        return bool(litellm.supports_vision(model=m))
    except Exception:  # noqa: BLE001 — capability check is advisory
        return None
