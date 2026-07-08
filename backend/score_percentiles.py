"""Per-prompt percentile ranking from human-score distributions.

Loads data/score_distributions.json (built by scripts/build_score_distributions.py)
and computes, for a given prompt, what "top %" a final score is — measured against
*that prompt's* full dataset distribution, not the global one.

top% = share of essays (for the same prompt) scoring at-or-above the given score.
Lower top% is better (상위 5% = only ~5% scored as high or higher).
"""
from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "data" / "score_distributions.json"


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", (s or "").strip())


@lru_cache(maxsize=1)
def _prompts() -> dict:
    if not _PATH.exists():
        return {}
    try:
        return json.loads(_PATH.read_text(encoding="utf-8")).get("prompts", {})
    except Exception:
        return {}


def _top_pct(hist: dict, score: float | None) -> int | None:
    """top% = count(value >= score) / n * 100, clamped to [1, 100]."""
    if not hist or score is None:
        return None
    n = 0
    ge = 0
    for k, count in hist.items():
        try:
            val = float(k)
        except (TypeError, ValueError):
            continue
        n += count
        if val >= float(score) - 1e-9:
            ge += count
    if n <= 0:
        return None
    return max(1, min(100, round(ge / n * 100)))


def percentile_for(prompt: str, total: float | None = None,
                   slot_scores: dict | None = None) -> dict | None:
    """Return {"n", "total": top%, "slots": {slot: top%}} or None if the prompt
    is not in the dataset (e.g. a custom topic)."""
    entry = _prompts().get(_norm(prompt))
    if not entry:
        return None
    out = {"n": entry.get("n"), "total": None, "slots": {}}
    if total is not None:
        out["total"] = _top_pct(entry.get("total") or {}, total)
    for slot, sc in (slot_scores or {}).items():
        out["slots"][slot] = _top_pct((entry.get("slots") or {}).get(slot) or {}, sc)
    return out
