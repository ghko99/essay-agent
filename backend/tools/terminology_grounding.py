"""terminology_grounding — domain/technical-term evidence (content_3).

Checks whether content words the student used are real, sourced technical terms
in 국립국어원 온용어(K-term): for each grounded term we keep its subject category,
origin (한자/영어 등), and definition. Useful for science/social-studies prompts
where correct terminology signals 적절성/타당성. No hardcoded term lists.
"""
from __future__ import annotations

import html

from ._common import ToolError, evidence, get_text, require_key, source

_KTERM = "https://kli.korean.go.kr/term/api/search.do"
_MAX_TERMS = 10


def _kterm_lookup(term: str) -> dict | None:
    key = require_key("KTERM_API_KEY")
    params = {"key": key, "apiSearchWord": term, "start": "1", "num": "10"}
    import json
    text = get_text(_KTERM, params=params)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ToolError(f"K-term returned non-JSON for '{term}': {e}") from e

    # K-term returns return_object as a *string* ("검색 결과가 없습니다.") when there
    # are no hits — not a list — so every level must be type-guarded.
    channel = data.get("channel") if isinstance(data, dict) else None
    if not isinstance(channel, dict):
        return None
    ros = channel.get("return_object")
    ros = [ros] if isinstance(ros, dict) else (ros if isinstance(ros, list) else [])
    for ro in ros:
        if not isinstance(ro, dict):
            continue
        rlist = ro.get("resultlist")
        rlist = [rlist] if isinstance(rlist, dict) else (rlist if isinstance(rlist, list) else [])
        for entry in rlist:
            if isinstance(entry, dict) and entry.get("word") == term:
                return {
                    "word": term,
                    "category_main": entry.get("category_main"),
                    "category_sub": entry.get("category_sub"),
                    "origin": entry.get("origin"),
                    "source": entry.get("source"),
                    "glossary": entry.get("glossary"),
                    "definition": html.unescape(entry.get("definition", "") or ""),
                }
    return None


def terminology_grounding(tokens: list[str]) -> dict:
    """Ground each token against the K-term technical dictionary."""
    tokens = [t for t in dict.fromkeys(tokens) if t.strip()][:_MAX_TERMS]
    if not tokens:
        raise ToolError("terminology_grounding got no tokens")

    grounded: list[dict] = []
    not_terms: list[str] = []
    for tok in tokens:
        entry = _kterm_lookup(tok)
        if entry:
            grounded.append(entry)
        else:
            not_terms.append(tok)

    categories = sorted({g.get("category_sub") for g in grounded if g.get("category_sub")})
    summary = (
        f"전문어 grounding {len(grounded)}/{len(tokens)}"
        + (f" — 분야: {', '.join(categories)}" if categories else "")
    )

    return evidence(
        tool="terminology_grounding",
        slot="content_3",
        signals={
            "n_tokens": len(tokens),
            "n_grounded": len(grounded),
            "grounded": grounded,
            "not_terms": not_terms,
            "categories": categories,
        },
        sources=[source("국립국어원 온용어(K-term)", url=_KTERM)],
        strength="weak" if grounded else "none",
        summary=summary,
    )
