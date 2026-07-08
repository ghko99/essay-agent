"""norm_lookup — map a BAREUN correction to its 국립국어원 어문 규범 article.

Two static paths (no LLM, no network):
  1. BAREUN `helps[helpId].ruleArticle` like "한글맞춤법, 제1장 총칙, 제2항"
     → parsed into (category, article) → matched against korean_norms.jsonl.
  2. BAREUN `helpId` with no ruleArticle → looked up in
     data/norms/bareun_helpid_map.json (manually curated helpId → norm_id).

If neither resolves, we return None — honestly reporting "no norm grounding"
rather than guessing. (An LLM-based fallback that picks an article is left as a
future enhancement; it must never silently invent a citation.)
"""
from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path

_NORMS = Path(__file__).resolve().parents[2] / "data" / "norms" / "processed" / "korean_norms.jsonl"
_HELPMAP = Path(__file__).resolve().parents[2] / "data" / "norms" / "bareun_helpid_map.json"

# BAREUN spells categories without spaces; jsonl uses spaced official names.
_CATEGORY_ALIASES = {
    "한글맞춤법": "한글 맞춤법",
    "표준어규정": "표준어 규정",
    "외래어표기법": "외래어 표기법",
    "국어의로마자표기법": "국어의 로마자 표기법",
}


def _nospace(s: str) -> str:
    return unicodedata.normalize("NFC", s).replace(" ", "").strip()


@lru_cache(maxsize=1)
def _rows() -> list[dict]:
    rows = []
    with _NORMS.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@lru_cache(maxsize=1)
def _by_id() -> dict[str, dict]:
    return {r["norm_id"]: r for r in _rows()}


@lru_cache(maxsize=1)
def _by_cat_article() -> dict[tuple[str, str], list[dict]]:
    idx: dict[tuple[str, str], list[dict]] = {}
    for r in _rows():
        key = (_nospace(r["category"]), _nospace(r["article"]))
        idx.setdefault(key, []).append(r)
    return idx


@lru_cache(maxsize=1)
def _helpid_map() -> dict[str, str]:
    if not _HELPMAP.exists():
        return {}
    data = json.loads(_HELPMAP.read_text(encoding="utf-8"))
    return data.get("map", {})


def _normalize_category(raw: str) -> str:
    key = _nospace(raw)
    return _nospace(_CATEGORY_ALIASES.get(key, raw))


def _parse_rule_article(rule_article: str) -> tuple[str | None, str | None, str | None]:
    """"한글맞춤법, 제1장 총칙, 제2항" → (category, chapter, article).

    Last comma-part is the article (제n항/제n조 등); first is the category.
    """
    parts = [p.strip() for p in rule_article.split(",") if p.strip()]
    if not parts:
        return None, None, None
    category = parts[0]
    article = parts[-1] if len(parts) >= 2 else None
    chapter = parts[1] if len(parts) >= 3 else None
    return category, chapter, article


def _trim(row: dict) -> dict:
    """Report-safe slice — title/summary/source only, not the full body dump."""
    return {
        "norm_id": row["norm_id"],
        "category": row["category"],
        "chapter": row.get("chapter", ""),
        "article": row.get("article", ""),
        "title": row.get("title", ""),
        "body": row.get("body", ""),
        "examples": row.get("examples", [])[:3],
        "source_url": row.get("source_url", ""),
    }


def _bigrams(s: str) -> set[str]:
    s = _nospace(s)
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def norm_search(query: str, category: str | None = None,
                top_k: int = 5) -> list[dict]:
    """검색어로 어문규범 조항을 찾는다 (로컬, 네트워크/LLM 없음).

    한글 자모 2-gram 겹침으로 (분류·항·제목·본문·예시)를 점수화해 상위 조항을 반환.
    검색어와 전혀 겹치지 않는 조항은 제외한다. 일치하는 게 없으면 빈 리스트.
    """
    qg = _bigrams(query)
    if not qg:
        return []
    cat = _normalize_category(category) if category else None

    scored: list[tuple[float, dict]] = []
    for r in _rows():
        if cat and _nospace(r.get("category", "")) != cat:
            continue
        haystack = " ".join([
            r.get("category", ""), r.get("article", ""), r.get("title", ""),
            r.get("body", ""), " ".join(r.get("examples", []) or []),
        ])
        tg = _bigrams(haystack)
        if not tg:
            continue
        inter = len(qg & tg)
        if inter == 0:
            continue
        scored.append((inter / len(qg), r))

    scored.sort(key=lambda x: -x[0])
    return [{**_trim(r), "score": round(s, 3)} for s, r in scored[:top_k]]


def norm_lookup(rule_article: str | None = None,
                help_id: str | None = None) -> dict | None:
    """Resolve a single norm article from a BAREUN correction.

    Returns a trimmed norm row dict, or None if nothing matched.
    """
    # Path 1: explicit ruleArticle string.
    if rule_article:
        category, chapter, article = _parse_rule_article(rule_article)
        if category and article:
            cat = _normalize_category(category)
            hits = _by_cat_article().get((cat, _nospace(article)), [])
            if len(hits) == 1:
                return _trim(hits[0])
            if len(hits) > 1 and chapter:
                ch = _nospace(chapter)
                for h in hits:
                    if _nospace(h.get("chapter", "")) == ch:
                        return _trim(h)
            if hits:
                return _trim(hits[0])

    # Path 2: curated helpId → norm_id.
    if help_id:
        norm_id = _helpid_map().get(help_id)
        if norm_id and norm_id in _by_id():
            return _trim(_by_id()[norm_id])

    return None
