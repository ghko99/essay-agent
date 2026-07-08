"""keyword_coverage — does the essay cover the topic's required keywords?

The topic's `keyword` field (from data/topics.json) lists the content words the
prompt expects. We test coverage by *lemma* using KIWI morphological analysis
on both the keywords and the essay — not raw substring matching — so that
inflected forms (조사/어미 변화) still count. No hardcoded vocabulary.

Primary slots: task_1 (과제 충실성), content_3 (적절성/타당성).
"""
from __future__ import annotations

from ._common import evidence, source, get_kiwi

# KIWI POS tags worth indexing as content words (nouns, verbs, adjectives,
# roots, foreign/number). Particles/endings are excluded.
_CONTENT_TAGS = ("NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "XR", "SL", "SH", "SN")


def _content_lemmas(text: str) -> set[str]:
    kiwi = get_kiwi()
    lemmas: set[str] = set()
    for tok in kiwi.tokenize(text):
        if tok.tag in _CONTENT_TAGS and len(tok.form) >= 1:
            lemmas.add(tok.form)
    return lemmas


def _split_keywords(keyword_field: str) -> list[str]:
    """Topic keywords are comma/space separated phrases. Keep them as phrases."""
    raw = keyword_field.replace("·", ",").replace("/", ",")
    parts = [p.strip() for chunk in raw.split(",") for p in [chunk] if chunk.strip()]
    return [p for p in parts if p]


def keyword_coverage(essay: str, topic: dict) -> dict:
    """topic is a topics.json entry (carries `keyword`, `prompt`, `topic`).

    A keyword counts as covered if *every* content lemma of the keyword phrase
    appears among the essay's content lemmas.
    """
    keyword_field = (topic or {}).get("keyword", "") or ""
    keywords = _split_keywords(keyword_field)
    essay_lemmas = _content_lemmas(essay)

    covered: list[str] = []
    missing: list[str] = []
    for kw in keywords:
        kw_lemmas = _content_lemmas(kw) or {kw}
        if kw_lemmas <= essay_lemmas:
            covered.append(kw)
        else:
            missing.append(kw)

    total = len(keywords)
    ratio = round(len(covered) / total, 3) if total else None
    strength = "none"
    if total:
        strength = "strong" if (ratio == 1.0 or ratio <= 0.5) else "weak"

    if total == 0:
        summary = "토픽에 핵심 키워드가 지정되지 않음."
    else:
        summary = f"핵심 키워드 {len(covered)}/{total} 충족" + (
            f" — 누락: {', '.join(missing)}" if missing else " (전부 충족)"
        )

    return evidence(
        tool="keyword_coverage",
        slot="task_1",
        signals={
            "keywords": keywords,
            "covered": covered,
            "missing": missing,
            "coverage_ratio": ratio,
            "n_total": total,
            "n_covered": len(covered),
        },
        sources=[source("data/topics.json", ref=(topic or {}).get("topic"))],
        strength=strength,
        summary=summary,
    )
