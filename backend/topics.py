"""Load the pre-built topics catalog (data/topics.json).

The JSON file is produced by scripts/build_topics.py and contains the
227 unique trained prompts (deduplicated by prompt text, not id).
Each topic carries its own 8-slot `rubric_names` list.
"""
from __future__ import annotations

import json
import os
import unicodedata
from functools import lru_cache
from pathlib import Path


DEFAULT_TOPICS_PATH = os.environ.get(
    "KANANA_TOPICS_JSON",
    str(Path(__file__).resolve().parent.parent / "data" / "topics.json"),
)


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", (s or "").strip())


@lru_cache(maxsize=1)
def load_topics(path: str = DEFAULT_TOPICS_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        return {"source": None, "total": 0, "subjects": [], "levels": [], "groups": []}
    with p.open(encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _prompt_to_rubric(path: str = DEFAULT_TOPICS_PATH) -> dict[str, list[str]]:
    """prompt-text (NFC-normalized) → list of 8 rubric full names."""
    data = load_topics(path)
    lookup: dict[str, list[str]] = {}
    for g in data.get("groups", []):
        for t in g.get("topics", []):
            names = t.get("rubric_names") or []
            prompt = _norm(t.get("prompt", ""))
            if prompt and len(names) == 8 and all(names):
                lookup[prompt] = list(names)
    return lookup


def rubric_for_prompt(prompt: str) -> list[str] | None:
    """Return the 8 rubric full names trained for this prompt, or None if unknown.

    Custom topics (not in the training set) will return None — callers should
    fall back to the default rubric in that case.
    """
    if not prompt:
        return None
    return _prompt_to_rubric().get(_norm(prompt))


@lru_cache(maxsize=1)
def _prompt_to_entry(path: str = DEFAULT_TOPICS_PATH) -> dict[str, dict]:
    """prompt-text (NFC-normalized) → full topics.json entry."""
    data = load_topics(path)
    lookup: dict[str, dict] = {}
    for g in data.get("groups", []):
        for t in g.get("topics", []):
            prompt = _norm(t.get("prompt", ""))
            if prompt:
                lookup[prompt] = t
    return lookup


def topic_entry_for_prompt(prompt: str, keywords: str | None = None) -> dict:
    """Return the full topics.json entry for a prompt.

    For custom prompts not in the training set, synthesize a minimal entry so the
    agent's tools still work (rubric falls back to default; keywords optional).
    """
    entry = _prompt_to_entry().get(_norm(prompt)) if prompt else None
    if entry is not None:
        return dict(entry)
    return {
        "prompt": prompt,
        "topic": prompt,
        "keyword": keywords or "",
        "rubric_names": None,
        "type_short": None,
        "rubric_type": None,
    }
