"""rubric_retrieve — local lookup into data/rubric_criteria.json.

Given a topic's 8 rubric full-names (+ an optional Korean type hint) and a
slot/score, return the matching 5-level criterion text. No network, no LLM.

Score → level mapping comes straight from rubric_criteria.json `score_to_level`
(1-2→evaluation_1, 3-4→evaluation_2, 5→evaluation_3, 6-7→evaluation_4,
8-9→evaluation_5).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ._common import evidence, source

_DATA = Path(__file__).resolve().parents[2] / "data" / "rubric_criteria.json"

# Topic Korean type label → preferred rubric_type code prefix.
_LABEL_PREFIX = {
    "서술형": "A",
    "논술형": "C",
    "주제별": "B",
}


@lru_cache(maxsize=1)
def _load() -> dict:
    with _DATA.open(encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _names_index() -> dict[tuple[str, ...], list[str]]:
    """Map the 8-slot name tuple → list of rubric_type codes sharing it."""
    data = _load()
    idx: dict[tuple[str, ...], list[str]] = {}
    for code, rt in data["rubrics"].items():
        names = tuple(rt["slots"][s]["name"] for s in rt["slots"])
        idx.setdefault(names, []).append(code)
    return idx


def resolve_rubric_type(rubric_names: list[str] | None,
                        type_hint: str | None = None) -> str:
    """Resolve a rubric_type code from 8 full names + an optional Korean hint.

    Several codes share the same 8 names (e.g. A-00Z / B-00A / C-00A all use the
    설명 variant). The Korean type hint (서술형/논술형/주제별) breaks the tie via
    the code prefix; otherwise the first matching code is used. Falls back to
    A-00Z when names are unknown.
    """
    data = _load()
    if rubric_names and len(rubric_names) == 8:
        codes = _names_index().get(tuple(rubric_names))
        if codes:
            if type_hint:
                prefix = _LABEL_PREFIX.get(type_hint.strip())
                for code in codes:
                    if prefix and code.startswith(prefix):
                        return code
            return codes[0]
    return "A-00Z" if "A-00Z" in data["rubrics"] else next(iter(data["rubrics"]))


def score_to_level(score: int) -> str:
    mapping = _load()["score_to_level"]
    return mapping.get(str(int(score)), "evaluation_3")


def rubric_retrieve(
    slot: str,
    score: int,
    rubric_names: list[str] | None = None,
    type_hint: str | None = None,
    rubric_type: str | None = None,
) -> dict:
    """Return an evidence object holding the criterion text for (slot, score).

    The criterion at the score's level is highlighted, but the full 5-level
    ladder is included under signals so the agent can see neighbouring levels.
    """
    data = _load()
    code = rubric_type or resolve_rubric_type(rubric_names, type_hint)
    rubric = data["rubrics"].get(code)
    if rubric is None:
        raise KeyError(f"unknown rubric_type: {code}")
    slot_def = rubric["slots"].get(slot)
    if slot_def is None:
        raise KeyError(f"unknown slot: {slot} in {code}")

    level = score_to_level(score)
    criteria = slot_def["criteria"]
    current = criteria.get(level, "")

    return evidence(
        tool="rubric_retrieve",
        slot=slot,
        signals={
            "rubric_type": code,
            "rubric_key": slot_def.get("rubric_key"),
            "name": slot_def.get("name"),
            "category": slot_def.get("category"),
            "score": int(score),
            "level": level,
            "criterion": current,
            "ladder": criteria,
        },
        sources=[source(
            "data/rubric_criteria.json",
            ref=f"{code}/{slot}/{level}",
        )],
        strength="strong",
        summary=f"[{slot_def.get('name')}] {score}점 기준({level}): {current}",
    )
