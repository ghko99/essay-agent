"""Scoring rules — agent's deterministic correction policy per (rubric_key, slot).

Rules are derived from the empirical analysis in analysis/ (see
regression_r2_summary.md, corr_partial_length.csv).  For each (rubric_key,
slot) we encode a structured rule that the agent applies on top of the
LoRA score.

Rule types:
    DIRECT  — replace LoRA score with feature-derived score (used only when
              tool→score mapping is essentially 1-to-1; e.g. A content_3
              keyword count). Confidence ≥ 0.7.
    CAP     — feature value enforces a score upper bound (e.g. high spacing
              error rate caps expression_2 below a threshold). LoRA cannot
              exceed the cap. Confidence ≥ 0.3.
    FLOOR   — symmetric: feature value enforces a lower bound.
    NUDGE   — soft adjustment: feature suggests ±1 in the indicated
              direction. Used as evidence for the LLM agent, not enforced
              automatically. Confidence 0.15 ~ 0.4.
    DEFER   — no rule; trust LoRA. Used when CV R² < 0.15.

Thresholds for tier rules:
    A list of (threshold, score). Sorted ascending by threshold.
    For DIRECT/CAP/FLOOR, the resolved score is the `score` of the
    *highest* threshold whose value the feature exceeds.

    e.g. thresholds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
         feature_val = 2.5  → threshold (2, 3) is highest exceeded → score 3
         feature_val = 4.0  → threshold (4, 5) is highest exceeded → score 5
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


RuleType = Literal["DIRECT", "CAP", "FLOOR", "NUDGE", "DEFER"]


@dataclass(frozen=True)
class ScoringRule:
    rubric_key: str
    slot: str
    rule_type: RuleType
    feature: str
    confidence: float            # 5-fold CV R² (or |partial-corr| for NUDGE)
    thresholds: tuple[tuple[float, int], ...] = ()
    direction: int = 0           # +1 / −1 for NUDGE; sign of correlation w/ score
    rationale: str = ""


def _resolve_tier(value: float, thresholds: tuple[tuple[float, int], ...]) -> int:
    """Return score of highest threshold whose value <= `value`."""
    out = thresholds[0][1]
    for thr, sc in thresholds:
        if value >= thr:
            out = sc
        else:
            break
    return out


def apply_rule(rule: ScoringRule, lora_score: int, feature_val: float | None) -> tuple[int, str]:
    """Apply a rule to a LoRA score.  Returns (final_score, explanation)."""
    if feature_val is None or rule.rule_type == "DEFER":
        return lora_score, "no rule applied"

    if rule.rule_type == "DIRECT":
        sc = _resolve_tier(feature_val, rule.thresholds)
        return sc, f"{rule.feature}={feature_val:.2f} → score {sc} (direct)"

    if rule.rule_type == "CAP":
        cap = _resolve_tier(feature_val, rule.thresholds)
        if lora_score > cap:
            return cap, f"{rule.feature}={feature_val:.2f} caps score at {cap} (was {lora_score})"
        return lora_score, f"{rule.feature}={feature_val:.2f} cap={cap}, LoRA within bounds"

    if rule.rule_type == "FLOOR":
        floor = _resolve_tier(feature_val, rule.thresholds)
        if lora_score < floor:
            return floor, f"{rule.feature}={feature_val:.2f} raises floor to {floor} (was {lora_score})"
        return lora_score, f"{rule.feature}={feature_val:.2f} floor={floor}, LoRA within bounds"

    # NUDGE — emit suggested direction; the agent LLM decides whether to apply
    return lora_score, f"NUDGE {rule.direction:+d} on {rule.feature}={feature_val:.2f}"


# ── Rule definitions ────────────────────────────────────────────────────
#
# Bins below were calibrated on raw_datasets/valid (n≈8000) — see
# regression_r2_summary.md and the empirical thresholds dump.

# A-00Z content_3 (서술형 - 핵심어 매칭) — DIRECT, R²=0.798
_RULE_A_CONTENT_3 = ScoringRule(
    rubric_key="A-00Z-1B-2H",
    slot="content_3",
    rule_type="DIRECT",
    feature="T1.kw_matched",
    confidence=0.798,
    thresholds=((0.0, 1), (1.0, 2), (2.0, 3), (3.0, 4), (4.0, 5)),
    direction=+1,
    rationale="루브릭이 핵심어 개수 0/1/2/3/4+ → 점수 1/2/3/4/5로 직접 매핑",
)

# expression_2 — 모든 변형: 맞춤법·띄어쓰기 오류율로 CAP
# Bins from valid data: rate=0~0.5% median 4-5, 0.5-2% median 4, 2-5% median 3,
# 5-10% median 2, 10%+ median 1. Use generous CAP so we don't false-cap mid essays.
def _spacing_cap(rubric_key: str, conf: float) -> ScoringRule:
    return ScoringRule(
        rubric_key=rubric_key, slot="expression_2",
        rule_type="CAP", feature="T4.spacing_error_rate", confidence=conf,
        # Empirical bins on valid (median expression_2 score):
        #   rate 0~0.5%: median 4-5     → no cap
        #   rate 0.5~2%: median 4       → no cap (cap at 5)
        #   rate 2~5%:   median 2.75-4  → cap at 4
        #   rate 5~10%:  median 1-2     → cap at 3
        #   rate 10~20%: median 1-2     → cap at 2
        #   rate >20%:   median 1-2     → cap at 1
        thresholds=((0.0, 5), (0.02, 4), (0.05, 3), (0.10, 2), (0.20, 1)),
        direction=-1,
        rationale="맞춤법 오류율이 일정 수준 초과하면 어법 점수 상한 강제",
    )

_SPACING_CAP_RULES = [
    _spacing_cap("A-00Z-1D-2I", 0.211),
    _spacing_cap("B-00A-1D-2I", 0.10),   # weak — keep but low confidence
    _spacing_cap("B-00B-1D-2J", 0.226),
    _spacing_cap("B-00C-1D-2J", 0.200),
    _spacing_cap("C-00A-1D-2I", 0.567),
    _spacing_cap("C-00B-1D-2J", 0.439),
    _spacing_cap("C-00C-1D-2J", 0.460),
]

# train_emo_diff NUDGE for C variants — high CV R² across slots
_C_VARIANT_PREFIXES = ("C-00A", "C-00B", "C-00C")

_C_NUDGE_RULES: list[ScoringRule] = []
# Only on slots where CV R² (with this feature dominant) is ≥ 0.20
_C_TARGET_SLOTS_CONFIDENCE = {
    # (variant_prefix, slot): cv_r2
    ("C-00A", "expression_1"): 0.402,
    ("C-00A", "expression_2"): 0.567,
    ("C-00A", "organization_1"): 0.325,
    ("C-00A", "organization_2"): 0.335,
    ("C-00A", "content_1"): 0.342,
    ("C-00A", "content_2"): 0.215,
    ("C-00B", "expression_1"): 0.335,
    ("C-00B", "expression_2"): 0.439,
    ("C-00B", "organization_1"): 0.229,
    ("C-00B", "organization_2"): 0.222,
    ("C-00B", "content_1"): 0.210,
    ("C-00C", "expression_1"): 0.330,
    ("C-00C", "expression_2"): 0.460,
    ("C-00C", "organization_2"): 0.301,
    ("C-00C", "organization_1"): 0.254,
}

# Rubric-key suffix per slot for variants (verified against raw rubric_keys)
_KEY_SUFFIX = {
    "C-00A": {
        "task_1": "1A-2A", "content_1": "1B-2G", "content_2": "1B-2F", "content_3": "1B-2H",
        "organization_1": "1C-2E", "organization_2": "1C-2D",
        "expression_1": "1D-2L", "expression_2": "1D-2I",
    },
    "C-00B": {
        "task_1": "1A-2A", "content_1": "1B-2M", "content_2": "1B-2N", "content_3": "1B-2B",
        "organization_1": "1C-2C", "organization_2": "1C-2D",
        "expression_1": "1D-2K", "expression_2": "1D-2J",
    },
    "C-00C": {
        "task_1": "1A-2A", "content_1": "1B-2O", "content_2": "1B-2P", "content_3": "1B-2Q",
        "organization_1": "1C-2C", "organization_2": "1C-2D",
        "expression_1": "1D-2K", "expression_2": "1D-2J",
    },
}

for (prefix, slot), conf in _C_TARGET_SLOTS_CONFIDENCE.items():
    suffix = _KEY_SUFFIX[prefix][slot]
    _C_NUDGE_RULES.append(ScoringRule(
        rubric_key=f"{prefix}-{suffix}",
        slot=slot,
        rule_type="NUDGE",
        feature="T1.train_emo_diff",
        confidence=conf,
        direction=+1,
        rationale="train-mined emotion/quality lexicon — 양수면 점수↑, 음수면↓",
    ))

# A-00Z train_qualA_diff NUDGE — broad but weak signal across A slots
_A_NUDGE_RULES = [
    ScoringRule(rubric_key="A-00Z-1A-2A", slot="task_1",
                rule_type="NUDGE", feature="T1.train_qualA_diff", confidence=0.240,
                direction=+1, rationale="A 품질 어휘"),
    ScoringRule(rubric_key="A-00Z-1B-2G", slot="content_1",
                rule_type="NUDGE", feature="T1.train_qualA_diff", confidence=0.186,
                direction=+1, rationale="A 품질 어휘"),
    ScoringRule(rubric_key="A-00Z-1B-2F", slot="content_2",
                rule_type="NUDGE", feature="T1.train_qualA_diff", confidence=0.157,
                direction=+1, rationale="A 품질 어휘"),
    ScoringRule(rubric_key="A-00Z-1C-2D", slot="organization_2",
                rule_type="NUDGE", feature="T1.train_qualA_diff", confidence=0.223,
                direction=+1, rationale="A 품질 어휘"),
    ScoringRule(rubric_key="A-00Z-1D-2I", slot="expression_2",
                rule_type="NUDGE", feature="T1.train_qualA_diff", confidence=0.211,
                direction=+1, rationale="A 품질 어휘 (보조)"),
]

# topic_mean_cos NUDGE for B/C task_1 — moderate signal
_TOPIC_NUDGE_RULES = [
    ScoringRule(rubric_key="B-00A-1A-2A", slot="task_1",
                rule_type="NUDGE", feature="T3.topic_mean_cos", confidence=0.205,
                direction=+1, rationale="주제↔본문 의미 유사도"),
    ScoringRule(rubric_key="B-00B-1A-2A", slot="task_1",
                rule_type="NUDGE", feature="T3.topic_mean_cos", confidence=0.252,
                direction=+1, rationale="주제↔본문 의미 유사도"),
]

# T2.inner_nll NUDGE — base-LM fluency signal.
# Most useful on B variants where train-mined lexicons (emo / qualA) don't fire.
# Single-feature Spearman ρ on valid (n>=478):
#   B-00A: task_1 -0.370, content_1 -0.342, content_3 -0.265, org_1 -0.274,
#          org_2 -0.321, expression_2 -0.387, content_2 -0.235, expression_1 -0.238
#   B-00B: task_1 -0.404, content_1 -0.352, content_3 -0.297, org_1 -0.275,
#          org_2 -0.368, expression_2 -0.304, content_2 -0.271
#   B-00C: expression_2 -0.400 (only entry strong enough)
_FLUENCY_NUDGE_RULES: list[ScoringRule] = []
_T2_TARGETS = {
    # (rubric_key, slot): |ρ|
    ("B-00A-1A-2A", "task_1"): 0.370,
    ("B-00A-1B-2G", "content_1"): 0.342,
    ("B-00A-1B-2F", "content_2"): 0.235,
    ("B-00A-1B-2H", "content_3"): 0.265,
    ("B-00A-1C-2E", "organization_1"): 0.274,
    ("B-00A-1C-2D", "organization_2"): 0.321,
    ("B-00A-1D-2L", "expression_1"): 0.238,
    ("B-00A-1D-2I", "expression_2"): 0.387,
    ("B-00B-1A-2A", "task_1"): 0.404,
    ("B-00B-1B-2M", "content_1"): 0.352,
    ("B-00B-1B-2N", "content_2"): 0.271,
    ("B-00B-1B-2B", "content_3"): 0.297,
    ("B-00B-1C-2C", "organization_1"): 0.275,
    ("B-00B-1C-2D", "organization_2"): 0.368,
    ("B-00B-1D-2K", "expression_1"): 0.223,
    ("B-00B-1D-2J", "expression_2"): 0.304,
    ("B-00C-1D-2J", "expression_2"): 0.400,
    # A variants — supplementary
    ("A-00Z-1D-2I", "expression_2"): 0.259,
    ("A-00Z-1D-2L", "expression_1"): 0.217,
    # C variants — secondary (T1 features already dominate, T2 as backup)
    ("C-00A-1D-2I", "expression_2"): 0.407,
    ("C-00C-1D-2J", "expression_2"): 0.409,
}
for (rk, slot), abs_rho in _T2_TARGETS.items():
    _FLUENCY_NUDGE_RULES.append(ScoringRule(
        rubric_key=rk, slot=slot,
        rule_type="NUDGE", feature="T2.inner_nll",
        confidence=abs_rho ** 2,   # use ρ² as confidence proxy
        direction=-1,              # lower NLL → higher score
        rationale="base LM 문장 내부 NLL — 유창성 신호 (낮을수록 자연스러움)",
    ))


# ── Master list & lookup ────────────────────────────────────────────────

ALL_RULES: list[ScoringRule] = (
    [_RULE_A_CONTENT_3]
    + _SPACING_CAP_RULES
    + _C_NUDGE_RULES
    + _A_NUDGE_RULES
    + _TOPIC_NUDGE_RULES
    + _FLUENCY_NUDGE_RULES
)

_RULES_BY_KEY: dict[tuple[str, str], list[ScoringRule]] = {}
for r in ALL_RULES:
    _RULES_BY_KEY.setdefault((r.rubric_key, r.slot), []).append(r)


def rules_for(rubric_key: str, slot: str) -> list[ScoringRule]:
    """Return ordered list of rules for (rubric_key, slot).  Higher confidence first."""
    rs = _RULES_BY_KEY.get((rubric_key, slot), [])
    return sorted(rs, key=lambda r: -r.confidence)


def has_rule(rubric_key: str, slot: str) -> bool:
    return (rubric_key, slot) in _RULES_BY_KEY


def coverage_summary() -> dict:
    """Diagnostic — how many (rubric_key, slot) cells we cover."""
    by_type: dict[str, int] = {}
    for r in ALL_RULES:
        by_type[r.rule_type] = by_type.get(r.rule_type, 0) + 1
    return {"total_rules": len(ALL_RULES), "by_type": by_type,
            "unique_cells": len(_RULES_BY_KEY)}
