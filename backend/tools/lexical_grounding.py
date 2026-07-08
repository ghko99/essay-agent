"""lexical_grounding — vocabulary level/existence evidence (expression_1).

For a list of surface words we ask official dictionaries (no hardcoded grade
lists):
  * 한국어기초사전 (krdict)  → word_grade (초급/중급/고급), pos, definition
  * 우리말샘 (opendict)      → existence fallback for words krdict lacks

A word found in krdict carries an official grade. A word in neither dictionary
is reported as `unverified` — a signal the agent may read as a coinage, a
proper noun, or a possible misspelling (it is NOT auto-judged here).

Both APIs require num ≥ 10 (smaller `num` fails — see SESSION §4).
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from ._common import ToolError, evidence, get_kiwi, get_text, require_key, source

_KRDICT = "https://krdict.korean.go.kr/api/search"
_OPENDICT = "https://opendict.korean.go.kr/api/search"

_NOUN_TAGS = ("NNG", "NNP")
_MAX_TOKENS = 12


def _text(el: ET.Element | None, tag: str) -> str | None:
    if el is None:
        return None
    child = el.find(tag)
    return child.text.strip() if child is not None and child.text else None


def _krdict_lookup(word: str) -> dict | None:
    key = require_key("KRD_API_KEY")
    params = {"key": key, "q": word, "num": "10", "part": "word", "sort": "dict",
              "translated": "n"}
    text = get_text(_KRDICT, params=params)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise ToolError(f"krdict returned non-XML for '{word}': {e}") from e
    for item in root.findall(".//item"):
        if _text(item, "word") == word:
            sense = item.find("sense")
            return {
                "word": word,
                "pos": _text(item, "pos"),
                "word_grade": _text(item, "word_grade"),
                "definition": _text(sense if sense is not None else item, "definition"),
                "link": _text(item, "link"),
            }
    return None


def _opendict_exists(word: str) -> bool:
    key = require_key("OPENDICT_API_KEY")
    params = {"key": key, "q": word, "req_type": "json", "num": "10", "start": "1",
              "part": "word", "sort": "dict"}
    import json
    text = get_text(_OPENDICT, params=params)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False
    channel = data.get("channel", data) if isinstance(data, dict) else {}
    items = channel.get("item", []) if isinstance(channel, dict) else []
    if isinstance(items, dict):
        items = [items]
    return any(it.get("word") == word for it in items)


def candidate_nouns(essay: str, limit: int = _MAX_TOKENS) -> list[str]:
    """Distinct content nouns from the essay (KIWI), in first-seen order."""
    seen: list[str] = []
    for tok in get_kiwi().tokenize(essay):
        if tok.tag in _NOUN_TAGS and len(tok.form) >= 2 and tok.form not in seen:
            seen.append(tok.form)
            if len(seen) >= limit:
                break
    return seen


def lexical_grounding(tokens: list[str]) -> dict:
    """Ground each token's vocabulary level/existence in official dictionaries."""
    tokens = [t for t in dict.fromkeys(tokens) if t.strip()][:_MAX_TOKENS]
    if not tokens:
        raise ToolError("lexical_grounding got no tokens")

    graded: list[dict] = []
    unverified: list[str] = []
    grade_counts: dict[str, int] = {}

    for tok in tokens:
        entry = _krdict_lookup(tok)
        if entry:
            grade = entry.get("word_grade") or "등급없음"
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
            graded.append(entry)
        elif _opendict_exists(tok):
            grade_counts["우리말샘"] = grade_counts.get("우리말샘", 0) + 1
            graded.append({"word": tok, "word_grade": None, "in_opendict": True})
        else:
            unverified.append(tok)

    n = len(tokens)
    summary = (
        f"어휘 {n}개 조회 — 등급분포 {grade_counts}"
        + (f", 사전 미확인 {len(unverified)}개: {', '.join(unverified)}" if unverified else "")
    )

    return evidence(
        tool="lexical_grounding",
        slot="expression_1",
        signals={
            "n_tokens": n,
            "grade_counts": grade_counts,
            "graded": graded,
            "unverified": unverified,
        },
        sources=[
            source("한국어기초사전", url=_KRDICT),
            source("우리말샘", url=_OPENDICT),
        ],
        strength="weak" if graded or unverified else "none",
        summary=summary,
    )
