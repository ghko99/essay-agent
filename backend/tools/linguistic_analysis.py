"""linguistic_analysis — ETRI WiseNLU 형태소·의존구문·의미역 분석 (시각화용).

에세이 전체를 ETRI WiseNLU `srl` 파이프라인으로 한 번에 분석해, 프런트엔드가 그대로
시각화할 수 있는 문장별 구조를 돌려준다. 채점/검증 흐름과 분리된 온디맨드 분석이라
evidence 스키마가 아니라 평범한 dict를 반환한다.

ETRI 응답 핵심(분석으로 확인):
  word        [{id, text, begin, end}]          begin~end = 그 어절의 형태소 id 범위
  morp        [{id, lemma, type(POS), position}]
  dependency  [{id, text, head, label, mod}]    head = 지배 어절 id(-1=root)
  SRL         [{verb, word_id, argument:[{type(역할), word_id, text}]}]

No external LLM. API 키는 .env에서만 읽고 출력에 넣지 않는다.
"""
from __future__ import annotations

from ._common import ToolError, post_json, require_key
import os


_DEFAULT_URL = "http://aiopen.etri.re.kr:8000/WiseNLU"


def _etri(analysis_code: str, text: str) -> dict:
    key = require_key("ETRI_API_KEY")
    url = os.environ.get("ETRI_WISENLU_URL", _DEFAULT_URL)
    payload = {"argument": {"analysis_code": analysis_code, "text": text}}
    data = post_json(url, payload, headers={"Authorization": key}, timeout=30)
    if not isinstance(data, dict):
        raise ToolError("ETRI WiseNLU returned a non-object response")
    if data.get("result") not in (0, None):
        raise ToolError(f"ETRI WiseNLU error (result={data.get('result')}): "
                        f"{str(data.get('reason') or data)[:200]}")
    return data


def _word_morphemes(word: dict, morp: list[dict]) -> list[dict]:
    """Morphemes belonging to one word (word.begin..end are morpheme ids)."""
    b = word.get("begin", 0)
    e = word.get("end", -1)
    out = []
    for m in morp:
        mid = m.get("id")
        if isinstance(mid, int) and b <= mid <= e:
            out.append({"lemma": m.get("lemma", ""), "tag": m.get("type", "")})
    return out


def analyze_text(essay: str) -> dict:
    """Return per-sentence morphology + dependency + SRL for the whole essay."""
    essay = (essay or "").strip()
    if not essay:
        raise ToolError("빈 텍스트는 분석할 수 없습니다.")

    data = _etri("srl", essay)
    ret = data.get("return_object") or {}
    if not isinstance(ret, dict):
        raise ToolError("ETRI WiseNLU returned no analysable sentences.")
    sentences = ret.get("sentence") or []

    out: list[dict] = []
    for s in sentences:
        if not isinstance(s, dict):
            continue
        morp = s.get("morp") or []
        words = []
        for w in (s.get("word") or []):
            words.append({
                "id": w.get("id"),
                "text": w.get("text", ""),
                "morphemes": _word_morphemes(w, morp),
            })
        dependency = []
        for d in (s.get("dependency") or []):
            dependency.append({
                "id": d.get("id"),
                "text": d.get("text", ""),
                "head": d.get("head"),
                "label": d.get("label", ""),
            })
        srl = []
        for fr in (s.get("SRL") or []):
            srl.append({
                "predicate": {"word_id": fr.get("word_id"), "verb": fr.get("verb", "")},
                "args": [
                    {"role": a.get("type", ""), "word_id": a.get("word_id"),
                     "text": a.get("text", "")}
                    for a in (fr.get("argument") or [])
                ],
            })
        out.append({
            "text": s.get("text", ""),
            "words": words,
            "dependency": dependency,
            "srl": srl,
        })

    return {
        "source": "ETRI WiseNLU",
        "n_sentences": len(out),
        "sentences": out,
    }
