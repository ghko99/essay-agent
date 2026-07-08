"""8-criteria Korean essay rubric shared by training data and this agent.

Three training rubric variants exist (sets A/B/C). They all share the same
8 canonical slots (task_1 → expression_2) and categories (과제/내용/조직/표현);
only the per-slot **full** and **short** names vary. `resolve_rubric(full_names)`
returns the 8 Criterion objects with the labels matching the given topic.

Set A (설명형 · 115 topics)
Set B (정서/주제형 · 62 topics)
Set C (주장/근거형 · 50 topics)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Criterion:
    key: str
    short: str
    full: str
    category: str
    feedback_header: str


SLOT_KEYS: list[str] = [
    "task_1",
    "content_1",
    "content_2",
    "content_3",
    "organization_1",
    "organization_2",
    "expression_1",
    "expression_2",
]

SLOT_CATEGORIES: list[str] = [
    "과제", "내용", "내용", "내용",
    "조직", "조직", "표현", "표현",
]


# Per-variant (full, short) labels.  Indexed by slot 0..7.
VARIANT_A: list[tuple[str, str]] = [
    ("과제 수행의 충실성", "과제충실성"),
    ("설명의 명료성",     "설명명료성"),
    ("설명의 구체성",     "설명구체성"),
    ("설명의 적절성",     "설명적절성"),
    ("문장의 연결성",     "문장연결성"),
    ("글의 통일성",       "글통일성"),
    ("어휘의 적절성",     "어휘적절성"),
    ("어법의 적절성",     "어법적절성"),
]

VARIANT_B: list[tuple[str, str]] = [
    ("과제 수행의 충실성",              "과제충실성"),
    ("주제 전달 및 정서 표현의 명료성", "주제표현명료성"),
    ("주제 전달 및 정서 표현의 구체성", "주제표현구체성"),
    ("주제 전달 및 정서 표현의 적절성", "주제표현적절성"),
    ("문장 및 문단의 연결성",           "문단연결성"),
    ("글의 통일성",                     "글통일성"),
    ("어휘 및 문장의 적절성",           "어휘문장적절성"),
    ("어법의 정확성",                   "어법정확성"),
]

VARIANT_C: list[tuple[str, str]] = [
    ("과제 수행의 충실성",     "과제충실성"),
    ("주장의 명료성",           "주장명료성"),
    ("주장의 적절성",           "주장적절성"),
    ("근거의 타당성",           "근거타당성"),
    ("문장 및 문단의 연결성",   "문단연결성"),
    ("글의 통일성",             "글통일성"),
    ("어휘 및 문장의 적절성",   "어휘문장적절성"),
    ("어법의 정확성",           "어법정확성"),
]

# Fast lookup: tuple of 8 full names → short names
_FULL_TUPLE_TO_SHORTS: dict[tuple[str, ...], list[str]] = {
    tuple(full for full, _ in variant): [short for _, short in variant]
    for variant in (VARIANT_A, VARIANT_B, VARIANT_C)
}


def _shorts_for(full_names: list[str] | None) -> list[str]:
    """Resolve per-slot short labels for the given full names.

    Falls back to Set A shorts if the tuple doesn't match any known variant
    (e.g. custom topics or unexpected model output).
    """
    if full_names and len(full_names) == 8:
        shorts = _FULL_TUPLE_TO_SHORTS.get(tuple(full_names))
        if shorts is not None:
            return shorts
    return [short for _, short in VARIANT_A]


def resolve_rubric(full_names: list[str] | None = None) -> list[Criterion]:
    """Return 8 Criterion objects for a topic.

    * `full_names` — list of 8 full rubric names (from topics.json). If omitted
      or malformed, defaults to Set A.
    """
    if not full_names or len(full_names) != 8:
        full_names = [full for full, _ in VARIANT_A]
    shorts = _shorts_for(full_names)
    return [
        Criterion(
            key=SLOT_KEYS[i],
            short=shorts[i],
            full=full_names[i],
            category=SLOT_CATEGORIES[i],
            feedback_header=full_names[i],
        )
        for i in range(8)
    ]


# Default rubric used by /api/rubric when no topic is given.
RUBRIC: list[Criterion] = resolve_rubric(None)

MIN_SCORE = 1
MAX_SCORE = 9


def rubric_as_dict(rubric: list[Criterion] | None = None) -> list[dict]:
    rubric = rubric or RUBRIC
    return [
        {"key": c.key, "short": c.short, "full": c.full, "category": c.category}
        for c in rubric
    ]


# ── Scoring rules (agent's correction policy) ────────────────────────────
# Encoded in scoring_rules.py based on empirical analysis of the training set
# (analysis/regression_r2_summary.md, analysis/corr_partial_length.csv).
from .scoring_rules import (  # noqa: E402,F401
    ScoringRule,
    apply_rule,
    rules_for,
    has_rule,
    coverage_summary,
    ALL_RULES,
)
