"""Shared plumbing for evidence tools: .env loading, HTTP, evidence schema.

Design rules enforced here:
  * API keys are read from .env and NEVER copied into evidence/provenance.
  * Every tool returns the same evidence shape via `evidence()`.
  * HTTP helpers mirror the smoke-tested patterns in tests/api_tools/.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT_DIR / ".env"

DEFAULT_TIMEOUT = 10


# ── .env ─────────────────────────────────────────────────────────────
def load_dotenv(path: Path = ENV_PATH) -> None:
    """Populate os.environ from .env without overwriting existing values."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Load once on import so tools can be used standalone.
load_dotenv()


class ToolError(RuntimeError):
    """Raised when an external resource fails. Surfaced, never swallowed."""


# ── HTTP ─────────────────────────────────────────────────────────────
def _request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[int, str, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.headers.get("content-type", ""), resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise ToolError(f"HTTP {e.code} from {url}: {detail}") from e
    except urllib.error.URLError as e:
        raise ToolError(f"URL error from {url}: {e.reason}") from e


def get_text(url: str, params: dict[str, Any] | None = None,
             headers: dict[str, str] | None = None, timeout: int = DEFAULT_TIMEOUT) -> str:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    _status, _ct, raw = _request("GET", url, headers=headers, timeout=timeout)
    return raw.decode("utf-8", errors="replace")


def get_json(url: str, params: dict[str, Any] | None = None,
             headers: dict[str, str] | None = None, timeout: int = DEFAULT_TIMEOUT) -> Any:
    text = get_text(url, params=params, headers=headers, timeout=timeout)
    return json.loads(text)


def post_json(url: str, data: dict[str, Any], headers: dict[str, str],
              timeout: int = DEFAULT_TIMEOUT) -> Any:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    merged = {"content-type": "application/json; charset=UTF-8", **headers}
    _status, _ct, raw = _request("POST", url, headers=merged, body=body, timeout=timeout)
    return json.loads(raw.decode("utf-8", errors="replace"))


def require_key(name: str) -> str:
    key = os.environ.get(name)
    if not key:
        raise ToolError(f"missing required API key: {name} (set it in .env)")
    return key


# ── Evidence schema ──────────────────────────────────────────────────
def evidence(
    tool: str,
    slot: str,
    *,
    signals: dict | None = None,
    spans: list[dict] | None = None,
    sources: list[dict] | None = None,
    strength: str = "none",
    summary: str = "",
) -> dict:
    """Build one evidence object. `strength` ∈ {none, weak, strong}."""
    if strength not in ("none", "weak", "strong"):
        raise ValueError(f"invalid strength: {strength}")
    return {
        "tool": tool,
        "slot": slot,
        "signals": signals or {},
        "spans": spans or [],
        "sources": sources or [],
        "strength": strength,
        "summary": summary,
    }


def source(name: str, *, ref: str | None = None, url: str | None = None,
           retrieved_at: str | None = None) -> dict:
    """Provenance record. Never contains secrets."""
    out: dict[str, Any] = {"name": name}
    if ref is not None:
        out["ref"] = ref
    if url is not None:
        out["url"] = url
    if retrieved_at is not None:
        out["retrieved_at"] = retrieved_at
    return out


# ── Shared local analyzers ───────────────────────────────────────────
_KIWI = None


def get_kiwi():
    """Lazy KIWI singleton (local morphological analyzer, no network)."""
    global _KIWI
    if _KIWI is None:
        from kiwipiepy import Kiwi
        _KIWI = Kiwi()
    return _KIWI

