"""Self-verifying tool-calling audit agent.

The LoRA scorer produces a hypothesis (8 slot scores + confidence + feedback).
This agent hands that hypothesis to the **base** Kanana model (LoRA OFF) and lets
it run *native tool calling* (functionary v3-llama3.1 format): the model decides
which evidence tools to call, reads the results, and then self-verifies every
slot — agreeing with or adjusting each LoRA score, always with a stated reason.

Design
  * The model calls tools by emitting `<function=name>{json args}</function>`.
    `parse_tool_calls()` extracts them; `ToolRunner` executes them and feeds the
    observation back as a `tool` (ipython) message; the loop repeats.
  * essay / topic / criteria are bound in the runner, so the model only passes
    tiny args (slot, score) — robust against arg hallucination.
  * Deterministic scoring rules (scoring_rules.py) are exposed as ONE advisory
    tool (`statistical_prior`); they are evidence only and never enforce a score.
  * The final answer is a JSON verdict (no tool call); `extract_json()` parses it.

No external LLM. No hardcoded linguistic knowledge. API keys never enter output.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import AsyncIterator

from .rubric import SLOT_KEYS, resolve_rubric
from .score_percentiles import percentile_for
from .scoring_rules import apply_rule, rules_for
from .tools import (
    ToolError,
    candidate_nouns,
    keyword_coverage,
    lexical_grounding,
    norm_search,
    orthography_probe,
    rubric_retrieve,
    terminology_grounding,
)
from .tools.rubric_retrieve import resolve_rubric_type


# Rubric 5-level descriptors map to 1–9 score ranges. We always show the agent
# the level WITH its score range so it never confuses the 1–5 level scale with
# the 1–9 score scale (which made it call 3점 "lower than" 2점).
_LEVEL_LABEL = {
    "evaluation_1": "1단계(1~2점·최저)",
    "evaluation_2": "2단계(3~4점)",
    "evaluation_3": "3단계(5점)",
    "evaluation_4": "4단계(6~7점)",
    "evaluation_5": "5단계(8~9점·최고)",
}

MAX_ITERS = 8             # tool-calling rounds before we force a final verdict
GEN_MAX_TOKENS = 3600     # the base model tends to emit reasoning *before* the
                          # JSON; at 1500 the 8-slot verdict was getting truncated
                          # mid-object → salvage kept only the first few slots and
                          # the rest silently fell back to LoRA. Reasons now cite
                          # concrete evidence (1–2 sentences), so give extra room.
REPORT_MAX_TOKENS = 1300  # phase-3 종합 리포트 (summary/strengths/improvements
                          # + 보정 항목별 설명 adjustments 최대 8건)
GEN_TEMPERATURE = 0.2     # low temp → stable verdicts. The 8B judge is high-variance
                          # run to run (same essay, different adjustments); keep it
                          # as deterministic as sampling allows.

# Independent judgment (2026-07). Whenever the LoRA numbers are visible to the
# model — at the start (2026-06 study) or in a later "reconciliation" turn
# (2026-07 test: same copy behavior, just less reliably) — the 8B verifier
# anchors and copies them. So the agent NEVER sees the LoRA scores while
# judging: HIDE_LORA_ANCHOR=True drops them from the user message, the rubric
# observations, and the echoed prefetch args. Its independent, evidence-based
# verdict IS the final verdict; LoRA (the trained, calibrated scorer) anchors
# the result only through the post-hoc ±MAX_ADJUST_DELTA clamp. The hypothesis
# comparison is revealed to the model only afterwards, when it writes the
# learner-facing report (prose only — scores are already locked).
HIDE_LORA_ANCHOR = True
MAX_ADJUST_DELTA = 3      # safety bound: even a reasoned adjustment can't move the
                          # final score more than ± this from LoRA per slot. Widened
                          # 2→3 (2026-07, user call): the two-stage agent was still
                          # settling too close to the hypothesis; only runaway swings
                          # (e.g. 7→2) should be caught, not reasoned adjustments.
TOOL_ROUND_CAP = 4        # after this many tool rounds, force a final JSON answer
MAX_REPROMPTS = 2         # consecutive no-progress turns before giving up the loop


# ── functionary tool-call parsing ────────────────────────────────────
_FUNC_RE = re.compile(r"<function=([^>]+?)>(.*?)</function>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict]:
    """Extract `<function=name>{json}</function>` calls from model output.

    Returns a list of {"name", "args", "raw_args"} dicts. Malformed JSON args
    are surfaced (args={}, parse_error set) rather than silently dropped.
    """
    calls: list[dict] = []
    for m in _FUNC_RE.finditer(text):
        name = m.group(1).strip()
        raw = m.group(2).strip()
        args: dict = {}
        parse_error = None
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    args = parsed
                else:
                    parse_error = "args were not a JSON object"
            except json.JSONDecodeError as e:
                parse_error = f"invalid JSON args: {e}"
        call = {"name": name, "args": args, "raw_args": raw}
        if parse_error:
            call["parse_error"] = parse_error
        calls.append(call)
    return calls


def extract_json(text: str) -> dict | None:
    """Return the first balanced top-level JSON object in `text`, or None.

    String-aware brace matching so braces inside JSON string values don't throw
    off the depth counter.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    blob = text[start:i + 1]
                    try:
                        return json.loads(blob)
                    except json.JSONDecodeError:
                        break  # try next '{'
        start = text.find("{", start + 1)
    return None


# Per-slot object salvage: even a truncated/oversized verdict (JSON cut off after
# a few slots) yields usable per-slot objects, which is far better than dropping
# the whole verdict and falling back to LoRA. Each slot object is small and
# self-contained, so we match `{...”slot”:”task_1”...}` chunks individually.
_SLOT_OBJ_RE = re.compile(
    r'\{[^{}]*?"slot"\s*:\s*"(?:' + "|".join(SLOT_KEYS) + r')"[^{}]*\}', re.DOTALL)
_OVERALL_RE = re.compile(r'"overall"\s*:\s*(\{[^{}]*\})', re.DOTALL)


def salvage_verdict(text: str) -> dict | None:
    """Return a verdict dict, tolerating truncated/oversized JSON.

    First try a clean top-level parse; if that fails (or has no slots), scrape
    individual slot objects so a partial verdict still survives.
    """
    v = extract_json(text)
    if v and v.get("slots"):
        return v
    slots: list[dict] = []
    seen: set[str] = set()
    for m in _SLOT_OBJ_RE.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        slot = obj.get("slot")
        if slot in SLOT_KEYS and slot not in seen:
            seen.add(slot)
            slots.append(obj)
    if not slots:
        return None
    overall: dict = {}
    mo = _OVERALL_RE.search(text)
    if mo:
        try:
            overall = json.loads(mo.group(1))
        except json.JSONDecodeError:
            overall = {}
    return {"slots": slots, "overall": overall}


# ── Tool schemas exposed to the model (functionary flat schema) ──────
TOOL_SCHEMAS: list[dict] = [
    {
        "name": "rubric_criteria",
        "description": "특정 평가 항목(slot)에서 주어진 점수의 루브릭 기준(해당 등급 + 5단계 사다리)을 조회한다. 점수 타당성 판단의 기준이 된다.",
        "parameters": {
            "type": "object",
            "properties": {
                "slot": {"type": "string", "enum": SLOT_KEYS,
                         "description": "평가 항목 키"},
                "score": {"type": "integer",
                          "description": "확인할 점수(1-9)"},
            },
            "required": ["slot", "score"],
        },
    },
    {
        "name": "check_orthography",
        "description": "에세이의 맞춤법·띄어쓰기를 BAREUN과 KIWI로 교차검증해 오류율과 의심 구간을 반환한다. 어법의 적절성(expression_2) 검증용.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "check_keywords",
        "description": "논제가 요구하는 핵심 키워드가 에세이에 포함됐는지 형태소 기준으로 확인한다. 과제 수행의 충실성(task_1) 검증용.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "check_vocabulary",
        "description": "에세이의 주요 명사를 국립국어원 사전에서 조회해 어휘 등급/사전 등재 여부를 확인한다. 어휘의 적절성(expression_1) 검증용.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "check_terminology",
        "description": "에세이의 주요 명사가 실제 전문용어인지 국립국어원 온용어에서 확인한다. 내용의 적절성/근거의 타당성(content_3) 검증용.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "statistical_prior",
        "description": "해당 평가 항목에 대해 학습 데이터에서 도출한 통계적 점수 경향(참고용·강제 아님)을 반환한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "slot": {"type": "string", "enum": SLOT_KEYS},
            },
            "required": ["slot"],
        },
    },
    {
        "name": "norm_search",
        "description": "국립국어원 어문규범(한글 맞춤법·띄어쓰기·표준어 규정·외래어 표기법·로마자 표기법)에서 검색어로 관련 조항을 직접 찾는다. 어법 오류의 근거 조항이 증거에 연결돼 있지 않을 때, 직접 규범을 검색해 근거를 보강하라.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "찾을 규범 키워드 (예: '의존 명사 띄어쓰기', '받침 표기', '사이시옷', '보조 용언')"},
                "category": {"type": "string",
                             "description": "선택: 한글 맞춤법 | 띄어쓰기 | 표준어 규정 | 외래어 표기법 | 국어의 로마자 표기법"},
            },
            "required": ["query"],
        },
    },
]


def _agent_tool_schemas() -> list[dict]:
    """Tools offered to the model during evaluation. statistical_prior is
    LoRA-relative — its notes literally quote the LoRA score ("was N") — so in
    anchor-hidden mode offering it would leak the number we hid. Drop it."""
    if not HIDE_LORA_ANCHOR:
        return TOOL_SCHEMAS
    return [s for s in TOOL_SCHEMAS if s["name"] != "statistical_prior"]


# ── Tool runner (binds essay/topic/criteria; memoizes evidence) ──────
def _lora_score(criterion: dict) -> int:
    s = criterion.get("score")
    return int(s) if s is not None else 5


class ToolRunner:
    """Executes the agent's tool calls against the real evidence tools.

    Holds the essay/topic/criteria context so the model only sends small args.
    Caches the heavyweight evidence tools so repeat calls (and statistical_prior's
    feature lookups) don't re-hit external APIs.
    """

    def __init__(self, essay: str, topic_entry: dict, criteria: list[dict]):
        self.essay = essay
        self.topic_entry = topic_entry
        self.criteria = criteria
        self.rubric_names = topic_entry.get("rubric_names")
        type_hint = topic_entry.get("type_short") or topic_entry.get("rubric_type")
        self.rubric_type = resolve_rubric_type(self.rubric_names, type_hint)
        self._nouns: list[str] | None = None
        self._cache: dict[str, dict] = {}     # tool name → full evidence
        self.collected: list[dict] = []       # every evidence object, for the report

    # -- cached raw evidence tools --
    def _nouns_list(self) -> list[str]:
        if self._nouns is None:
            self._nouns = candidate_nouns(self.essay)
        return self._nouns

    def _orthography(self) -> dict:
        if "orthography_probe" not in self._cache:
            self._cache["orthography_probe"] = orthography_probe(self.essay)
        return self._cache["orthography_probe"]

    def _keywords(self) -> dict:
        if "keyword_coverage" not in self._cache:
            self._cache["keyword_coverage"] = keyword_coverage(self.essay, self.topic_entry)
        return self._cache["keyword_coverage"]

    def _rubric(self, slot: str, score: int) -> dict:
        return rubric_retrieve(slot, int(score), rubric_names=self.rubric_names,
                               rubric_type=self.rubric_type)

    def _rubric_key(self, slot: str) -> str | None:
        idx = SLOT_KEYS.index(slot) if slot in SLOT_KEYS else 0
        crit = self.criteria[idx] if idx < len(self.criteria) else {}
        ev = self._rubric(slot, _lora_score(crit))
        return ev["signals"].get("rubric_key")

    # -- dispatch --
    def call(self, name: str, args: dict) -> dict:
        """Run one tool. Returns the full evidence dict (with a `slot`).

        Raises ToolError on failure (surfaced by the loop, never swallowed).
        Successful evidence is appended to `self.collected` for the report.
        """
        ev = self._dispatch(name, args)
        if ev is not None:
            self.collected.append(ev)
        return ev

    def _dispatch(self, name: str, args: dict) -> dict:
        if name == "rubric_criteria":
            slot = args.get("slot")
            score = args.get("score")
            if slot not in SLOT_KEYS:
                raise ToolError(f"unknown slot: {slot!r}")
            if not isinstance(score, int) or not (1 <= score <= 9):
                raise ToolError(f"score must be an int 1-9, got {score!r}")
            return self._rubric(slot, score)
        if name == "check_orthography":
            return self._orthography()
        if name == "check_keywords":
            return self._keywords()
        if name == "check_vocabulary":
            nouns = self._nouns_list()
            if not nouns:
                raise ToolError("에세이에서 분석할 명사를 찾지 못했습니다.")
            return lexical_grounding(nouns)
        if name == "check_terminology":
            nouns = self._nouns_list()
            if not nouns:
                raise ToolError("에세이에서 분석할 명사를 찾지 못했습니다.")
            return terminology_grounding(nouns)
        if name == "statistical_prior":
            slot = args.get("slot")
            if slot not in SLOT_KEYS:
                raise ToolError(f"unknown slot: {slot!r}")
            return self._statistical_prior(slot)
        if name == "norm_search":
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ToolError("norm_search에는 query(검색어)가 필요합니다.")
            return self._norm_search(query.strip(), args.get("category"))
        raise ToolError(f"unknown tool: {name!r}")

    def _norm_search(self, query: str, category: str | None) -> dict:
        """Agent-driven search over the official 어문규범 corpus (evidence only)."""
        from .tools import evidence, source

        hits = norm_search(query, category=category, top_k=5)
        if hits:
            top = hits[0]
            summary = (f"어문규범 검색 '{query}' → {len(hits)}건, "
                       f"최상위: {top['category']} {top['article']} — {top.get('body', '')[:50]}")
            strength = "strong"
        else:
            summary = f"어문규범 검색 '{query}' → 일치하는 조항 없음."
            strength = "none"
        return evidence(
            tool="norm_search",
            slot="expression_2",
            signals={"query": query, "category": category, "matches": hits},
            sources=[source("국립국어원 어문 규범",
                            ref="data/norms/processed/korean_norms.jsonl")],
            strength=strength,
            summary=summary,
        )

    def _statistical_prior(self, slot: str) -> dict:
        """Advisory evidence from the empirical scoring rules. Never enforced."""
        from .tools import evidence, source  # local import to keep module top tidy

        rubric_key = self._rubric_key(slot)
        idx = SLOT_KEYS.index(slot)
        crit = self.criteria[idx] if idx < len(self.criteria) else {}
        lora = _lora_score(crit)
        rules = rules_for(rubric_key, slot) if rubric_key else []

        notes: list[str] = []
        suggested: int | None = None
        for rule in rules:
            feature_val = self._feature_for(rule.feature, slot)
            sc, expl = apply_rule(rule, lora, feature_val)
            if rule.rule_type in ("CAP", "FLOOR", "DIRECT") and feature_val is not None:
                suggested = sc
            notes.append(f"[{rule.rule_type}] {expl} (R²/|ρ|≈{rule.confidence:.2f})")

        if not notes:
            notes.append("이 항목에는 통계적 사전 규칙이 없습니다(LoRA 점수를 신뢰).")

        summary = f"[{slot}] 통계적 사전(참고용): " + " / ".join(notes)
        return evidence(
            tool="statistical_prior",
            slot=slot,
            signals={
                "rubric_key": rubric_key,
                "lora_score": lora,
                "suggested_score": suggested,
                "notes": notes,
                "enforced": False,
            },
            sources=[source("scoring_rules.py", ref="empirical (analysis/)")],
            strength="weak" if suggested is not None else "none",
            summary=summary,
        )

    def _feature_for(self, feature: str, slot: str) -> float | None:
        """Resolve the feature value a rule needs, from cached evidence.

        Only features we can actually measure at serving time are filled
        (spacing rate, keyword count). Train-mined / LM features (inner_nll,
        train_emo_diff, topic_mean_cos) are unavailable → None, so apply_rule
        returns the rule description without a number.
        """
        try:
            if feature == "T4.spacing_error_rate":
                return self._orthography()["signals"].get("spacing_error_rate")
            if feature == "T1.kw_matched":
                return float(self._keywords()["signals"].get("n_covered") or 0)
        except ToolError:
            return None
        return None


# ── Observations fed back to the model (compact, token-budgeted) ─────
def _clip(s, n: int = 160):
    if isinstance(s, str) and len(s) > n:
        return s[:n] + "…"
    return s


def _observation(name: str, ev: dict) -> dict:
    """Compact view of an evidence object for the tool (ipython) message."""
    sig = ev.get("signals", {})
    base = {"tool": name, "summary": _clip(ev.get("summary", ""), 200),
            "strength": ev.get("strength")}
    if name == "rubric_criteria":
        # Surface the FULL 5-level ladder (with score ranges) so the agent can
        # compare the essay against every level, not just LoRA's current one.
        ladder = sig.get("ladder") or {}
        levels = [f"{_LEVEL_LABEL.get(lv, lv)}: {_clip(ladder.get(lv, ''), 90)}"
                  for lv in ("evaluation_1", "evaluation_2", "evaluation_3",
                             "evaluation_4", "evaluation_5")]
        base.update({"slot": ev["slot"], "rubric_5levels": levels})
        if not HIDE_LORA_ANCHOR:
            base.update({"lora_score": sig.get("score"),
                         "lora_level": _LEVEL_LABEL.get(sig.get("level"), sig.get("level"))})
    elif name == "check_orthography":
        base.update({k: sig.get(k) for k in
                     ("spacing_error_rate", "n_spacing", "n_typo", "n_strong",
                      "n_corrections", "n_norm_grounded")})
        # Surface actual corrections with their norm article + candidates so the
        # agent can cite/choose the right 어문규범 조항 (C) instead of guessing.
        def _norm_label(n):
            return f"{n.get('category','')} {n.get('article','')} {n.get('title','')}".strip()
        corrections = []
        for s in (ev.get("spans") or [])[:6]:
            item = {"원문": s.get("origin"), "교정": s.get("revised"),
                    "유형": s.get("category")}
            n = s.get("norm")
            if n:
                item["규범"] = _norm_label(n) + (" (검색)" if n.get("matched_by") == "search" else "")
            cands = s.get("norm_candidates") or []
            if cands:
                item["규범후보"] = [_norm_label(c) for c in cands[:3]]
            corrections.append(item)
        if corrections:
            base["corrections"] = corrections
    elif name == "check_keywords":
        base.update({k: sig.get(k) for k in
                     ("covered", "missing", "coverage_ratio", "n_total")})
    elif name == "check_vocabulary":
        unv = sig.get("unverified") or []
        n_tok = sig.get("n_tokens") or 0
        base.update({"grade_counts": sig.get("grade_counts"),
                     "n_unverified": len(unv),
                     "unverified_ratio": round(len(unv) / n_tok, 2) if n_tok else None,
                     "unverified": unv,
                     "hint": ("사전 미확인 내용어 존재 → 지어낸 말·오기 의심, 어휘 적절성(expression_1) "
                              "부정 신호 — 이 항목 점수를 올리지 말 것"
                              if len(unv) >= 1 else None)})
    elif name == "check_terminology":
        base.update({"n_grounded": sig.get("n_grounded"),
                     "categories": sig.get("categories"),
                     "not_terms": sig.get("not_terms")})
    elif name == "statistical_prior":
        notes = [_clip(n, 120) for n in (sig.get("notes") or [])[:2]]
        base.update({"slot": ev["slot"], "suggested_score": sig.get("suggested_score"),
                     "notes": notes, "enforced": False})
    elif name == "norm_search":
        matches = sig.get("matches") or []
        base.update({"query": sig.get("query"),
                     "matches": [{"category": m.get("category"), "article": m.get("article"),
                                  "body": _clip(m.get("body", ""), 110)}
                                 for m in matches[:3]]})
    elif name == "perplexity_probe":
        # Only flow sentences carry a PPL; the anchor (first) sentence is None.
        flow = [s for s in (sig.get("sentences") or []) if s.get("ppl") is not None]
        sents = sorted(flow, key=lambda s: -s.get("ppl", 0))
        base.update({"overall_ppl": sig.get("overall_ppl"),
                     "worst_sentences": [{"ppl": s.get("ppl"), "text": _clip(s.get("text", ""), 45)}
                                         for s in sents[:3]]})
    return base


def _perplexity_evidence(ppl: dict) -> dict:
    """Wrap base-model perplexity into a 문장 연결성(organization_1) evidence object."""
    from .tools import evidence, source

    sents = ppl.get("sentences") or []
    flow = [s for s in sents if s.get("ppl") is not None]  # anchor sentence has no PPL
    worst = max(flow, key=lambda s: s.get("ppl", 0)) if flow else None
    summary = f"문장 흐름 perplexity 평균 {ppl.get('overall_ppl')} (낮을수록 앞 문맥과 자연스럽게 이어짐)"
    if worst:
        summary += f"; 가장 거친 연결 {worst.get('ppl')}: \"{(worst.get('text') or '')[:30]}\""
    return evidence(
        tool="perplexity_probe",
        slot="organization_1",
        signals={
            "overall_ppl": ppl.get("overall_ppl"),
            "overall_mean_nll": ppl.get("overall_mean_nll"),
            "n_tokens": ppl.get("n_tokens"),
            "sentences": sents,
        },
        sources=[source("Kanana base model (LoRA OFF)", ref="prompt_logprobs perplexity")],
        strength="weak",
        summary=summary,
    )


# ── Prompts ──────────────────────────────────────────────────────────
# All 8 slots pre-listed so the 8B model fills a rail instead of free-writing
# (fewer omissions/truncations). Field order = thinking order: the model writes
# the evidence-citing reason FIRST, then commits to level, then score — an
# autoregressive model that emits the number first rationalizes it afterwards,
# which is exactly the reason↔score contradiction we saw.
_VERDICT_SKELETON = (
    '{"slots":[\n'
    + ",\n".join(
        f' {{"slot":"{k}","reason":"<측정값을 인용한 근거 1~2문장>","level":<1-5>,"score":<점수>}}'
        for k in SLOT_KEYS)
    + '\n],"overall":{"summary":"<전체 총평 2~3문장>"}}'
)

_SYSTEM = (
    "너는 한국어 에세이 채점관이다. 에세이 본문과 함께 외부 도구의 측정 결과(맞춤법·띄어쓰기, "
    "핵심 키워드, 어휘 등급, 전문어, 루브릭 5단계 기준, 문장 흐름 PPL)가 대화에 제공된다. "
    "8개 평가 항목 각각에 1~9점(1=최저, 9=최고)을 매긴다.\n\n"
    "## 항목별 판단 절차 — 반드시 이 순서\n"
    "1. 그 항목과 관련된 도구 측정값을 확인한다.\n"
    "2. reason: 측정값을 인용해 잘한 점과 결함을 1~2문장으로 적는다.\n"
    "3. level: 그 reason이 루브릭 5단계 설명 중 어느 단계에 해당하는지 고른다.\n"
    "4. score: 그 단계의 점수 범위 안에서 정한다 — 1단계=1~2점, 2단계=3~4점, 3단계=5점, "
    "4단계=6~7점, 5단계=8~9점.\n\n"
    "## 규칙\n"
    "- reason·level·score는 한 방향이다: 결함을 적었으면 낮은 단계, 잘했다고 적었으면 높은 단계.\n"
    "- 증거가 낮은 단계를 가리키면 1~3점도 주고, 높은 단계에 부합하면 8~9점도 준다. "
    "애매하다고 5~6점으로 도피하지 않는다.\n"
    "- 측정 결과에 없는 수치·사실을 지어내지 않는다. 어법 오류는 '규범/규범후보' 조항을 "
    "인용한다(없으면 norm_search로 찾고, 그래도 없으면 '근거 조항 없음').\n"
    "- reason은 최대 2문장. 길게 쓰면 8개 항목이 잘려 채점이 무효가 된다.\n\n"
    "## 출력\n"
    "필요하면 도구를 더 호출할 수 있다. 채점을 마치면 설명 문장 없이 아래 형식의 JSON 객체 "
    "하나만 출력한다. <>를 채워라:\n"
    + _VERDICT_SKELETON
)


# Phase 3: after the verdict, ask for a learner-facing detailed report. Kept as a
# separate turn so the fragile 8-slot verdict JSON stays short and parseable.
_REPORT_INSTR = (
    "채점이 끝났다. 이제 위 도구 증거와 항목별 판정을 종합해 학습자에게 보여줄 종합 리포트를 작성하라. "
    "증거에 나온 구체 수치·사례(오류 건수, 누락 키워드, 어휘 등급, PPL 등)를 인용하고, 증거에 없는 "
    "사실은 쓰지 마라. adjustments에는 위 비교표에서 점수가 바뀐(▲/▼) 항목 각각에 대해, 기존 채점과 "
    "왜 다르게 판단했는지를 증거를 인용해 설명한다. 설명 문장 없이 아래 JSON 객체 하나만 출력한다:\n"
    '{"summary": "글 전체에 대한 평가 4~6문장 — 무엇을 잘했고 무엇이 부족한지",\n'
    ' "strengths": ["이 글의 강점 2~4개, 각 1문장"],\n'
    ' "improvements": ["구체적인 개선 제안 2~4개, 각 1문장 — 어떻게 고치면 되는지까지"],\n'
    ' "adjustments": [{"slot": "<점수가 바뀐 항목키>", "explanation": "<기존 채점보다 올린/내린 이유, '
    '증거 인용 1~2문장>"}, ... 바뀐 항목 각각]}'
)


def _user_message(essay: str, topic_entry: dict, criteria: list[dict]) -> str:
    lines = [
        f"논제: {topic_entry.get('prompt') or topic_entry.get('topic') or ''}",
    ]
    kw = topic_entry.get("keyword")
    if kw:
        lines.append(f"논제 핵심 키워드: {kw}")
    header = "\n채점할 8개 항목:" if HIDE_LORA_ANCHOR else "\nLoRA 채점 결과(검증 대상):"
    lines.append(header)
    for idx, slot in enumerate(SLOT_KEYS):
        crit = criteria[idx] if idx < len(criteria) else {}
        name = crit.get("full") or slot
        if HIDE_LORA_ANCHOR:
            lines.append(f"- {slot} ({name})")
        else:
            conf = (crit.get("confidence") or {}).get("confidence")
            conf_txt = f", 신뢰도 {conf:.2f}" if isinstance(conf, (int, float)) else ""
            lines.append(f"- {slot} ({name}): {_lora_score(crit)}점{conf_txt}")
    lines.append(f"\n에세이 본문:\n{essay}")
    lines.append("\n도구로 검증한 뒤 각 항목을 판정하고 최종 JSON을 출력하라.")
    return "\n".join(lines)


# ── Report assembly ──────────────────────────────────────────────────
# Rubric level (1–5) ↔ score (1–9) ranges, mirroring data/rubric_criteria.json.
_LEVEL_RANGE = {1: (1, 2), 2: (3, 4), 3: (5, 5), 4: (6, 7), 5: (8, 9)}


def _clamped_agent_score(slot_verdict: dict, lora: int) -> int:
    """Final score for a slot: the agent's independent score, bounded to
    LoRA ± MAX_ADJUST_DELTA. The agent judges freely (it never saw the LoRA
    number); LoRA — the trained, calibrated scorer — is the gravitational
    center that catches runaway swings."""
    raw = slot_verdict.get("score", slot_verdict.get("agent_score", lora))
    try:
        score = int(raw)
    except (TypeError, ValueError):
        score = lora
    # Schema consistency: the ladder level is the more grounded judgment (picked
    # against the rubric text); a score that drifted outside its range is snapped
    # back in before any clamping.
    try:
        level = int(slot_verdict.get("level"))
    except (TypeError, ValueError):
        level = None
    if level in _LEVEL_RANGE:
        lo, hi = _LEVEL_RANGE[level]
        score = max(lo, min(hi, score))
    score = max(1, min(9, score))
    return max(lora - MAX_ADJUST_DELTA, min(lora + MAX_ADJUST_DELTA, score))


def _severity(flag: bool, delta: int) -> str:
    if not flag:
        return "none"
    ad = abs(delta)
    if ad >= 3:
        return "high"
    if ad == 2:
        return "medium"
    return "low"


def _first_feedback(crit: dict) -> str:
    fbs = crit.get("feedbacks") or []
    return fbs[0]["feedback"] if fbs and fbs[0].get("feedback") else (crit.get("feedback") or "")


def _evidence_view(ev: dict) -> dict:
    return {
        "tool": ev["tool"],
        "strength": ev.get("strength"),
        "summary": ev.get("summary", ""),
        "spans": ev.get("spans", []),
        "sources": ev.get("sources", []),
        "signals": ev.get("signals", {}),
    }


def _build_report(topic_entry: dict, rubric_type: str, criteria: list[dict],
                  verdict: dict, collected: list[dict], trace: list[dict]) -> dict:
    by_slot_ev: dict[str, list[dict]] = {k: [] for k in SLOT_KEYS}
    for ev in collected:
        by_slot_ev.setdefault(ev.get("slot", ""), []).append(ev)

    verdict_slots = {s.get("slot"): s for s in (verdict.get("slots") or [])
                     if isinstance(s, dict)}

    slot_reports: list[dict] = []
    for idx, slot in enumerate(SLOT_KEYS):
        crit = criteria[idx] if idx < len(criteria) else {}
        slot_name = crit.get("full") or slot
        lora = _lora_score(crit)
        v = verdict_slots.get(slot, {})

        agent_score = _clamped_agent_score(v, lora)

        reason = (v.get("reason") or "").strip()
        adjusted = agent_score != lora
        # Integrity guard: a change with no stated reason is reverted to LoRA, so
        # every surviving adjustment carries a rationale the UI shows on the card.
        if adjusted and not reason:
            agent_score = lora
            adjusted = False
        flag = bool(adjusted)
        delta = agent_score - lora

        slot_reports.append({
            "slot": slot,
            "rubric_name": slot_name,
            "category": crit.get("category"),
            "lora": {
                "score": lora,
                "confidence": (crit.get("confidence") or {}).get("confidence"),
                "feedback": _first_feedback(crit),
            },
            "agent": {
                "score": agent_score,
                "flag": flag,
                "severity": _severity(flag, delta),
                "reasoning": reason,
                "adjust_reason": (v.get("adjust_reason") or "").strip(),
                "deterministic_note": "",
                "evidence_tools": v.get("evidence_tools") or [],
            },
            "evidence": [_evidence_view(ev) for ev in by_slot_ev.get(slot, [])],
        })

    lora_total = sum(s["lora"]["score"] for s in slot_reports)
    agent_total = sum(s["agent"]["score"] for s in slot_reports)
    overall = verdict.get("overall") or {}
    if not isinstance(overall, dict):
        overall = {}

    def _str_list(v) -> list[str]:
        if not isinstance(v, list):
            return []
        return [s.strip() for s in v if isinstance(s, str) and s.strip()]

    # Per-prompt percentile of the FINAL (agent) scores — measured against this
    # prompt's full human-scored dataset (not the global distribution).
    prompt = topic_entry.get("prompt") or topic_entry.get("topic")
    pct = percentile_for(
        prompt, total=agent_total,
        slot_scores={s["slot"]: s["agent"]["score"] for s in slot_reports},
    )
    if pct:
        for s in slot_reports:
            s["agent"]["percentile"] = (pct.get("slots") or {}).get(s["slot"])

    totals = {
        "lora": lora_total,
        "agent": agent_total,
        "delta": agent_total - lora_total,
        "max": len(slot_reports) * 9,
        "percentile": pct.get("total") if pct else None,
    }
    return {
        "topic": prompt,
        "rubric_type": rubric_type,
        "totals": totals,
        "percentile_n": pct.get("n") if pct else None,
        "n_flagged": sum(1 for s in slot_reports if s["agent"]["flag"]),
        "overall": {
            "confidence": overall.get("confidence"),
            "summary": (overall.get("summary") or "").strip(),
            "strengths": _str_list(overall.get("strengths")),
            "improvements": _str_list(overall.get("improvements")),
        },
        "slots": slot_reports,
        "trace": trace,
    }


# ── Tool execution (shared by prefetch + verdict loop) ───────────────
async def _exec_tool(runner: ToolRunner, trace: list[dict], iteration: int,
                     name: str, args: dict) -> tuple[list[dict], dict]:
    """Run one validated tool call. Returns (events_to_yield, tool_message).

    Tool failures are surfaced in the events/trace (status="error"), never
    swallowed — the observation simply carries the error for that tool.
    Runs in a worker thread: the tools do blocking HTTP (urllib) and would
    otherwise stall the event loop for every concurrent request.
    """
    events: list[dict] = [{"type": "tool_call", "iteration": iteration,
                           "tool": name, "args": args}]
    t0 = time.time()
    try:
        ev = await asyncio.to_thread(runner.call, name, args)
        ms = int((time.time() - t0) * 1000)
        obs = _observation(name, ev)
        trace.append({"phase": "VERIFY", "tool": name, "ms": ms, "status": "ok"})
        events.append({"type": "tool_result", "iteration": iteration, "tool": name,
                       "status": "ok", "ms": ms, "summary": ev.get("summary", ""),
                       "strength": ev.get("strength")})
    except ToolError as e:
        ms = int((time.time() - t0) * 1000)
        obs = {"tool": name, "error": str(e)}
        trace.append({"phase": "VERIFY", "tool": name, "ms": ms, "status": "error",
                      "message": str(e)})
        events.append({"type": "tool_result", "iteration": iteration, "tool": name,
                       "status": "error", "ms": ms, "summary": str(e)})
    except Exception as e:  # unexpected — surface, don't hide
        ms = int((time.time() - t0) * 1000)
        obs = {"tool": name, "error": f"{type(e).__name__}: {e}"}
        trace.append({"phase": "VERIFY", "tool": name, "ms": ms, "status": "error",
                      "message": f"{type(e).__name__}: {e}"})
        events.append({"type": "tool_result", "iteration": iteration, "tool": name,
                       "status": "error", "ms": ms, "summary": f"{type(e).__name__}: {e}"})
    return events, {"role": "tool", "content": json.dumps(obs, ensure_ascii=False)}


def _evidence_recap(runner: ToolRunner) -> str:
    """Compact factual recap of the key measurements, restated in the final
    instruction right before the verdict is requested. An 8B model weighs the
    most recent message heavily; without this the numbers sit a dozen messages
    back and the judgment drifts off them. Facts only — no thresholds/judgment."""
    def _sig(tool: str) -> dict:
        for ev in runner.collected:
            if ev.get("tool") == tool:
                return ev.get("signals") or {}
        return {}

    lines: list[str] = []
    kw = _sig("keyword_coverage")
    if kw:
        covered = kw.get("covered") or []
        ratio = kw.get("coverage_ratio")
        pct = f"{int(ratio * 100)}%" if ratio is not None else "—"
        missing = ", ".join(kw.get("missing") or []) or "없음"
        lines.append(f"- 핵심 키워드: {len(covered)}/{kw.get('n_total', '—')}개 충족({pct}), "
                     f"누락: {missing}")
    ort = _sig("orthography_probe")
    if ort:
        lines.append(f"- 맞춤법·띄어쓰기: 교정 {ort.get('n_corrections') or 0}건"
                     f"(띄어쓰기 {ort.get('n_spacing') or 0}·맞춤법 {ort.get('n_typo') or 0}, "
                     f"KIWI 교차검증 {ort.get('n_strong') or 0}건)")
    lex = _sig("lexical_grounding")
    if lex:
        unv = lex.get("unverified") or []
        unv_txt = f" ({', '.join(unv[:3])})" if unv else ""
        lines.append(f"- 어휘: 등급분포 {json.dumps(lex.get('grade_counts') or {}, ensure_ascii=False)}, "
                     f"사전 미확인 {len(unv)}개{unv_txt}")
    term = _sig("terminology_grounding")
    if term:
        lines.append(f"- 전문어: {term.get('n_grounded') or 0}개 인정")
    ppl = _sig("perplexity_probe")
    if ppl:
        lines.append(f"- 문장 흐름 PPL 평균: {ppl.get('overall_ppl')} (높을수록 연결이 거침)")
    return "\n".join(lines)


def _prefetch_plan(criteria: list[dict]) -> list[tuple[str, dict]]:
    """Evidence tools to force-run before the model verifies, so it always sees
    the core measurements instead of skipping them. statistical_prior / norm_search
    stay agent-callable (not forced) to keep the context within budget."""
    plan: list[tuple[str, dict]] = [
        ("check_keywords", {}),
        ("check_orthography", {}),
        ("check_vocabulary", {}),
        ("check_terminology", {}),
    ]
    for i, slot in enumerate(SLOT_KEYS):
        score = _lora_score(criteria[i]) if i < len(criteria) else 5
        plan.append(("rubric_criteria", {"slot": slot, "score": score}))
    return plan


# ── The agent loop (streaming) ───────────────────────────────────────
async def agent_verify_stream(
    scorer, essay: str, topic_entry: dict, criteria: list[dict],
) -> AsyncIterator[dict]:
    """Tool-calling verification. Every tool is force-run first (so the model
    always has full evidence), then the base model self-verifies each slot.

    Yields SSE-ready events: start / tool_call / tool_result / token / done / error
    """
    runner = ToolRunner(essay, topic_entry, criteria)
    trace: list[dict] = []

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _user_message(essay, topic_entry, criteria)},
    ]

    yield {"type": "start", "n_slots": len(SLOT_KEYS),
           "lora_total": sum(_lora_score(c) for c in criteria)}

    # ── Phase 1a: base-model perplexity (async; needs the engine) ──
    yield {"type": "tool_call", "iteration": 0, "tool": "perplexity_probe", "args": {}}
    _t0 = time.time()
    try:
        ppl = await scorer.essay_perplexity(essay)
        ev = _perplexity_evidence(ppl)
        runner.collected.append(ev)
        _ms = int((time.time() - _t0) * 1000)
        trace.append({"phase": "VERIFY", "tool": "perplexity_probe", "ms": _ms, "status": "ok"})
        yield {"type": "tool_result", "iteration": 0, "tool": "perplexity_probe",
               "status": "ok", "ms": _ms, "summary": ev["summary"], "strength": ev["strength"]}
        messages.append({"role": "assistant", "content": "<function=perplexity_probe>{}</function>"})
        messages.append({"role": "tool", "content": json.dumps(_observation("perplexity_probe", ev), ensure_ascii=False)})
    except Exception as e:  # surface, don't abort verification
        _ms = int((time.time() - _t0) * 1000)
        trace.append({"phase": "VERIFY", "tool": "perplexity_probe", "ms": _ms,
                      "status": "error", "message": f"{type(e).__name__}: {e}"})
        yield {"type": "tool_result", "iteration": 0, "tool": "perplexity_probe",
               "status": "error", "ms": _ms, "summary": f"{type(e).__name__}: {e}"}

    # ── Phase 1b: force-run the rest of the tools, recorded as if the agent called them ──
    for name, args in _prefetch_plan(criteria):
        echo_args = dict(args)
        if HIDE_LORA_ANCHOR and name == "rubric_criteria":
            # the tool runs with the LoRA score (correct evidence for the UI), but
            # the echoed call must not leak that number into the model's context
            echo_args.pop("score", None)
        messages.append({"role": "assistant",
                         "content": f"<function={name}>{json.dumps(echo_args, ensure_ascii=False)}</function>"})
        events, tool_msg = await _exec_tool(runner, trace, 0, name, args)
        for e in events:
            yield e
        messages.append(tool_msg)

    recap = _evidence_recap(runner)
    messages.append({"role": "user", "content":
                     ("모든 도구 측정이 끝났다. 핵심 측정값 요약:\n" + recap + "\n\n"
                      if recap else "모든 도구 측정이 끝났다.\n\n")
                     + "어법 오류 중 어문규범 조항이 연결되지 않은 것이 있으면 norm_search로 "
                     "근거 조항을 찾아 보강하라. 그런 다음 8개 항목을 판단 절차(측정값 확인 → "
                     "reason → level → score)에 따라 채점하고, 설명 문장 없이 정해진 JSON "
                     "형식의 <>만 채워 출력하라."})

    # ── Phase 2: model self-verification (may still call more tools) ─────────
    verdict: dict | None = None
    assistant_texts: list[str] = []   # for end-of-loop salvage
    tool_rounds = 0                    # how many turns actually called tools
    reprompts = 0                      # consecutive no-progress (no call, no verdict)
    for iteration in range(1, MAX_ITERS + 1):
        # Once the model has had enough tool rounds, stop offering tools so it is
        # forced to emit the final verdict instead of looping on more calls.
        offer_tools = tool_rounds < TOOL_ROUND_CAP
        prompt = scorer.chat_prompt(messages,
                                    tools=_agent_tool_schemas() if offer_tools else None)

        # Stream one base-model (LoRA OFF) assistant turn, forwarding token
        # deltas live so the UI shows the agent reasoning in real time.
        text = ""
        async for gev in scorer.generate_prompt_ensemble_stream(
            prompt=prompt,
            n_samples=1,
            temperature=GEN_TEMPERATURE,
            max_new_tokens=GEN_MAX_TOKENS,
            emit_tokens=True,
            use_lora=False,
        ):
            gt = gev.get("type")
            if gt == "token":
                delta = gev.get("text", "")
                text += delta
                yield {"type": "token", "iteration": iteration, "text": delta}
            elif gt == "done":
                samples = gev.get("samples") or []
                if samples and samples[0]:
                    text = samples[0]
            elif gt == "error":
                yield {"type": "error", "message": gev.get("message", "generation error")}
                return
        text = text.strip()
        assistant_texts.append(text)

        # If tools are no longer offered, ignore any stray tool-call syntax and
        # treat the turn as a (possibly final) answer.
        calls = parse_tool_calls(text) if offer_tools else []
        # Feed the assistant turn back verbatim so the model keeps its context.
        messages.append({"role": "assistant", "content": text})

        if not calls:
            verdict = salvage_verdict(text)
            if verdict and verdict.get("slots"):
                break
            # No tool call and no parseable verdict → nudge for JSON only, but
            # bail out after a couple of stale turns instead of burning every
            # iteration (which makes the UI look frozen).
            reprompts += 1
            if reprompts >= MAX_REPROMPTS or iteration == MAX_ITERS:
                break
            messages.append({
                "role": "user",
                "content": "함수 호출과 설명 문장 없이, 시스템에 제시된 JSON 형식의 <>만 "
                           "채워 출력하라. 출력의 첫 글자는 '{'여야 한다.",
            })
            trace.append({"phase": "VERIFY", "tool": "reparse_request",
                          "ms": 0, "status": "ok"})
            continue

        reprompts = 0
        tool_rounds += 1
        for call in calls:
            name, args = call["name"], call["args"]
            if call.get("parse_error"):
                yield {"type": "tool_call", "iteration": iteration, "tool": name, "args": args}
                trace.append({"phase": "VERIFY", "tool": name, "ms": 0,
                              "status": "error", "message": call["parse_error"]})
                yield {"type": "tool_result", "iteration": iteration, "tool": name,
                       "status": "error", "ms": 0, "summary": call["parse_error"]}
                messages.append({"role": "tool", "content": json.dumps(
                    {"tool": name, "error": call["parse_error"]}, ensure_ascii=False)})
                continue
            events, tool_msg = await _exec_tool(runner, trace, iteration, name, args)
            for e in events:
                yield e
            messages.append(tool_msg)

        # Reached the tool-round cap → tell the model to stop calling tools and
        # produce the final verdict on the next turn.
        if tool_rounds == TOOL_ROUND_CAP:
            messages.append({"role": "user", "content":
                             "이제 도구 호출을 멈추고, 설명 문장 없이 시스템에 제시된 JSON "
                             "형식의 <>만 채워 출력하라. 출력의 첫 글자는 '{'여야 한다."})

    # Last-chance salvage: a verdict may have been emitted alongside tool calls or
    # truncated across turns — scrape it from everything the model said.
    if verdict is None or not verdict.get("slots"):
        verdict = salvage_verdict("\n".join(assistant_texts))

    verdict_ok = bool(verdict and verdict.get("slots"))
    if not verdict_ok:
        # Could not get a structured verdict — keep LoRA scores, record it visibly.
        trace.append({"phase": "VERIFY", "tool": "verdict_parse",
                      "ms": 0, "status": "error",
                      "message": "구조화된 검증 결과(JSON)를 얻지 못해 LoRA 점수를 유지했습니다."})
        verdict = {"slots": [{"slot": s, "verdict": "agree",
                              "agent_score": _lora_score(criteria[i]), "reason": "",
                              "evidence_tools": []}
                             for i, s in enumerate(SLOT_KEYS)],
                   "overall": {"confidence": "low",
                               "summary": "자동 검증이 구조화된 결과를 내지 못했습니다."}}

    # ── Phase 3: learner-facing detailed report (summary / strengths / improvements).
    # The verdict is locked, so the LoRA-vs-agent comparison can now be revealed to
    # the model — it explains the adjustments in prose without touching the scores.
    if verdict_ok:
        yield {"type": "tool_call", "iteration": MAX_ITERS + 1,
               "tool": "final_report", "args": {}}
        _t0 = time.time()
        verdict_slots = {s.get("slot"): s for s in (verdict.get("slots") or [])
                         if isinstance(s, dict)}
        cmp_lines = []
        for i, slot in enumerate(SLOT_KEYS):
            crit = criteria[i] if i < len(criteria) else {}
            lora = _lora_score(crit)
            final_score = _clamped_agent_score(verdict_slots.get(slot, {}), lora)
            d = final_score - lora
            mark = "유지" if d == 0 else (f"▲{d}" if d > 0 else f"▼{abs(d)}")
            cmp_lines.append(f"- {slot} ({crit.get('full') or slot}): "
                             f"기존 채점 {lora}점 → 네 판정 {final_score}점 ({mark})")
        messages.append({"role": "user", "content":
                         "참고 — 사람 채점 데이터로 학습된 채점 모델의 점수와 네 판정 비교"
                         "(점수는 이미 확정됐다):\n" + "\n".join(cmp_lines) + "\n\n"
                         + _REPORT_INSTR})
        text = ""
        try:
            async for gev in scorer.generate_prompt_ensemble_stream(
                prompt=scorer.chat_prompt(messages, tools=None),
                n_samples=1,
                temperature=GEN_TEMPERATURE,
                max_new_tokens=REPORT_MAX_TOKENS,
                emit_tokens=True,
                use_lora=False,
            ):
                gt = gev.get("type")
                if gt == "token":
                    delta = gev.get("text", "")
                    text += delta
                    yield {"type": "token", "iteration": MAX_ITERS + 1, "text": delta}
                elif gt == "done":
                    samples = gev.get("samples") or []
                    if samples and samples[0]:
                        text = samples[0]
            detail = extract_json(text)
        except Exception as e:  # report is best-effort; never break the verdict
            detail = None
            trace.append({"phase": "REPORT", "tool": "final_report",
                          "ms": int((time.time() - _t0) * 1000), "status": "error",
                          "message": f"{type(e).__name__}: {e}"})
            yield {"type": "tool_result", "iteration": MAX_ITERS + 1,
                   "tool": "final_report", "status": "error",
                   "ms": int((time.time() - _t0) * 1000),
                   "summary": f"{type(e).__name__}: {e}"}
        else:
            _ms = int((time.time() - _t0) * 1000)
            if isinstance(detail, dict) and (detail.get("summary")
                                             or detail.get("strengths")
                                             or detail.get("improvements")):
                overall = verdict.get("overall")
                if not isinstance(overall, dict):
                    overall = {}
                if detail.get("summary"):
                    overall["summary"] = detail["summary"]
                overall["strengths"] = detail.get("strengths") or []
                overall["improvements"] = detail.get("improvements") or []
                verdict["overall"] = overall
                # 바뀐 항목별 "왜 기존 채점과 다르게 봤는지" 설명 — 판정 근거와 별개로,
                # UI의 '보정 근거' 자리에 그대로 표시된다.
                adj_expl = {}
                for a in (detail.get("adjustments") or []):
                    if (isinstance(a, dict) and a.get("slot") in SLOT_KEYS
                            and isinstance(a.get("explanation"), str)
                            and a["explanation"].strip()):
                        adj_expl[a["slot"]] = a["explanation"].strip()
                for s in (verdict.get("slots") or []):
                    if isinstance(s, dict) and s.get("slot") in adj_expl:
                        s["adjust_reason"] = adj_expl[s["slot"]]
                trace.append({"phase": "REPORT", "tool": "final_report",
                              "ms": _ms, "status": "ok"})
                yield {"type": "tool_result", "iteration": MAX_ITERS + 1,
                       "tool": "final_report", "status": "ok", "ms": _ms,
                       "summary": "종합 리포트 작성 완료"}
            else:
                trace.append({"phase": "REPORT", "tool": "final_report",
                              "ms": _ms, "status": "error",
                              "message": "리포트 JSON을 파싱하지 못해 한 줄 총평만 표시합니다."})
                yield {"type": "tool_result", "iteration": MAX_ITERS + 1,
                       "tool": "final_report", "status": "error", "ms": _ms,
                       "summary": "리포트 JSON 파싱 실패 — 한 줄 총평으로 대체"}

    report = _build_report(topic_entry, runner.rubric_type, criteria,
                           verdict, runner.collected, trace)
    yield {"type": "done", **report}


async def run_verify(scorer, essay: str, topic_entry: dict,
                     criteria: list[dict]) -> dict:
    """Non-streaming variant: drive the loop and return the final report."""
    report: dict | None = None
    async for ev in agent_verify_stream(scorer, essay, topic_entry, criteria):
        if ev["type"] == "done":
            report = {k: v for k, v in ev.items() if k != "type"}
        elif ev["type"] == "error":
            raise RuntimeError(ev.get("message", "verify error"))
    if report is None:
        raise RuntimeError("verify loop did not complete")
    return report
