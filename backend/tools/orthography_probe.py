"""orthography_probe — cross-checked 어법/맞춤법/띄어쓰기 evidence (expression_2).

No single checker is trusted. We gather independent signals and let agreement
decide strength:

  bareun_spellcheck   — real correction candidates (span, category, helpId, norm)
  kiwi_spacing_diff   — local reproducible spacing change offsets
  etri_morp_view      — public-API morpheme baseline (auxiliary)
  bareun_morp_view    — corrector-linked morpheme analysis (auxiliary)
  kiwi_morp_view      — local morpheme boundary signal (auxiliary)

Strength rule (SESSION §6): one tool flags a span → weak; two independent tools
flag the same span/type → strong. Concretely, a BAREUN SPACING correction whose
offset coincides with a KIWI spacing change is upgraded to strong.

Error policy: the core signals (BAREUN spellcheck + KIWI) are strict — failures
raise ToolError, never a silent fallback. The auxiliary morpheme views (ETRI,
BAREUN morphology) record their failure *visibly* under signals instead of
crashing the whole probe.
"""
from __future__ import annotations

import os

from ._common import (
    ToolError,
    evidence,
    get_kiwi,
    post_json,
    require_key,
    source,
)
from .norm_lookup import norm_lookup, norm_search

# Deterministic norm-search fallback: when norm_lookup can't resolve a correction
# to an article (no ruleArticle and the helpId map is empty), search the corpus
# with BAREUN's own explanation (metalanguage) as the query and attach the top
# hit. Gated by a minimum 2-gram overlap so we don't attach noisy citations, and
# tagged matched_by="search" to distinguish it from exact rule matches.
_NORM_SEARCH_MIN_SCORE = 0.12

# TYPO cross-validation: KIWI is an independent analyzer. We compare KIWI's mean
# per-morpheme log-prob of the corrected form vs the original form of the SAME
# word. If the correction is clearly more probable under KIWI (by this margin),
# KIWI independently corroborates BAREUN's spelling fix → strong. Comparing the
# two forms of one word self-normalizes, so rare-but-correct words don't misfire.
_KIWI_TYPO_MARGIN = 0.5


def _norm_candidates(comment: str | None) -> list[dict]:
    """Top norm-article candidates for a correction, from its BAREUN comment."""
    if not comment or not comment.strip():
        return []
    hits = norm_search(comment.strip(), top_k=3)
    return [{"norm_id": h["norm_id"], "category": h["category"],
             "article": h["article"], "title": h.get("title", ""),
             "body": h.get("body", ""), "source_url": h.get("source_url", ""),
             "score": h.get("score", 0)} for h in hits]


# ── BAREUN spellcheck (core) ─────────────────────────────────────────
def _bareun_spellcheck(essay: str) -> dict:
    key = require_key("BAREUN_API_KEY")
    base = os.environ.get("BAREUN_API_BASE", "https://api.bareun.ai").rstrip("/")
    url = f"{base}/bareun.RevisionService/CorrectError"
    payload = {
        "document": {"content": essay, "language": "ko-KR"},
        "encoding_type": "UTF8",
        "auto_split_sentence": True,
    }
    data = post_json(url, payload, headers={"api-key": key})
    if not isinstance(data, dict):
        raise ToolError("BAREUN spellcheck returned non-dict response")
    return data


def _byte_to_char(essay: str, byte_offset: int) -> int:
    """BAREUN reports UTF-8 *byte* offsets; convert to a character index so the
    span aligns with KIWI's character-based offsets and with frontend slicing."""
    raw = essay.encode("utf-8")
    return len(raw[:max(byte_offset, 0)].decode("utf-8", errors="ignore"))


def _spell_spans(essay: str, bareun: dict) -> list[dict]:
    """Flatten revisedBlocks → correction spans with norm grounding.

    Offsets are normalized to character indices (begin_char/end_char)."""
    helps = bareun.get("helps", {}) or {}
    spans: list[dict] = []
    for block in bareun.get("revisedBlocks", []) or []:
        origin = block.get("origin", {}) or {}
        content = origin.get("content", "")
        begin_char = _byte_to_char(essay, origin.get("beginOffset", 0))
        end_char = begin_char + len(content)
        for rev in block.get("revisions", []) or []:
            help_id = rev.get("helpId")
            help_obj = helps.get(help_id, {}) if help_id else {}
            rule_article = help_obj.get("ruleArticle")
            comment = help_obj.get("comment")
            norm = norm_lookup(rule_article=rule_article, help_id=help_id)
            candidates: list[dict] = []
            if norm is not None:
                norm = {**norm, "matched_by": "rule"}
            else:
                # No exact rule/helpId match → search the corpus with BAREUN's
                # explanation. Surface top candidates (C) so the agent can pick
                # the right article; auto-attach top-1 only if it clears the bar (A).
                candidates = _norm_candidates(comment)
                if candidates and candidates[0]["score"] >= _NORM_SEARCH_MIN_SCORE:
                    top = candidates[0]
                    norm = {"norm_id": top["norm_id"], "category": top["category"],
                            "chapter": "", "article": top["article"],
                            "title": top["title"], "body": top.get("body", ""),
                            "examples": [], "source_url": top.get("source_url", ""),
                            "matched_by": "search"}
            spans.append({
                "begin": begin_char,
                "end": end_char,
                "origin": content,
                "revised": rev.get("revised"),
                "category": rev.get("category"),
                "help_id": help_id,
                "help_comment": help_obj.get("comment"),
                "norm": norm,
                "norm_candidates": candidates,   # for agent selection (C)
                "kiwi_agrees": False,        # SPACING: KIWI spacing diff overlap
                "kiwi_morp_agrees": False,   # TYPO: KIWI rates correction likelier
            })
    return spans


# ── KIWI spacing diff (core, local) ──────────────────────────────────
def _kiwi_spacing_offsets(essay: str) -> list[int]:
    """Offsets (in the original) where KIWI's spacing differs from the original.

    Both strings share the same non-space character sequence; only whitespace
    between them differs. We record the original offset of the non-space
    character that follows a changed gap.
    """
    spaced = get_kiwi().space(essay)
    a = [(i, ch) for i, ch in enumerate(essay) if not ch.isspace()]
    b = [ch for ch in spaced if not ch.isspace()]
    if len(a) != len(b):
        # Defensive: space() should preserve non-space chars; if not, bail loudly.
        raise ToolError("KIWI spacing changed non-space characters unexpectedly")

    def _has_space_before(s: str, nonspace_positions: list[int], k: int) -> bool:
        if k == 0:
            return False
        prev_end = nonspace_positions[k - 1] + 1
        cur = nonspace_positions[k]
        return any(s[j].isspace() for j in range(prev_end, cur))

    orig_pos = [i for i, _ in a]
    spaced_pos = [i for i, ch in enumerate(spaced) if not ch.isspace()]
    diffs: list[int] = []
    for k in range(1, len(a)):
        o = _has_space_before(essay, orig_pos, k)
        s = _has_space_before(spaced, spaced_pos, k)
        if o != s:
            diffs.append(orig_pos[k])
    return diffs


# ── Auxiliary morpheme views ─────────────────────────────────────────
def _kiwi_morp_count(essay: str) -> int:
    return len(get_kiwi().tokenize(essay))


def _kiwi_mean_score(text: str) -> float | None:
    """Mean per-morpheme log-prob KIWI assigns to `text` (higher = more likely)."""
    toks = get_kiwi().tokenize(text or "")
    if not toks:
        return None
    return sum(t.score for t in toks) / len(toks)


def _etri_morp_view(essay: str) -> dict:
    try:
        key = require_key("ETRI_API_KEY")
        url = os.environ.get("ETRI_WISENLU_URL", "https://epretx.etri.re.kr/api/WiseNLU")
        data = post_json(
            url,
            {"argument": {"text": essay, "analysis_code": "morp"}},
            headers={"Authorization": key},
        )
        ro = data.get("return_object", {}) if isinstance(data, dict) else {}
        n_morp = sum(len(s.get("morp", [])) for s in ro.get("sentence", []))
        return {"status": "ok", "morpheme_count": n_morp}
    except Exception as e:  # visible, not masked
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def _bareun_morp_view(essay: str) -> dict:
    try:
        key = require_key("BAREUN_API_KEY")
        base = os.environ.get("BAREUN_API_BASE", "https://api.bareun.ai").rstrip("/")
        url = f"{base}/bareun.LanguageService/AnalyzeSyntax"
        data = post_json(
            url,
            {
                "document": {"content": essay, "language": "ko-KR"},
                "encoding_type": "UTF8",
                "auto_split_sentence": True,
                "auto_spacing": False,
                "auto_jointing": False,
            },
            headers={"api-key": key},
        )
        n_morp = 0
        for s in (data.get("sentences", []) if isinstance(data, dict) else []):
            for tok in s.get("tokens", []):
                n_morp += len(tok.get("morphemes", []))
        return {"status": "ok", "morpheme_count": n_morp}
    except Exception as e:  # visible, not masked
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


# ── Probe ────────────────────────────────────────────────────────────
def orthography_probe(essay: str) -> dict:
    """Return one expression_2 evidence object combining all signals."""
    if not essay or not essay.strip():
        raise ToolError("orthography_probe got empty essay")

    bareun = _bareun_spellcheck(essay)             # strict core
    spans = _spell_spans(essay, bareun)
    spacing_offsets = _kiwi_spacing_offsets(essay)  # strict core (char offsets)

    # Independent corroboration → strong:
    #  · SPACING: a KIWI spacing change overlaps the correction span.
    #  · TYPO:    KIWI rates the corrected form clearly more probable than the
    #             original (its own analyzer agrees the original is off).
    for sp in spans:
        if sp["category"] == "SPACING":
            lo, hi = sp["begin"], sp["end"]
            sp["kiwi_agrees"] = any(lo <= off <= hi for off in spacing_offsets)
        elif sp["category"] == "TYPO":
            origin, revised = sp.get("origin"), sp.get("revised")
            if origin and revised and origin != revised:
                so = _kiwi_mean_score(origin)
                sr = _kiwi_mean_score(revised)
                if so is not None and sr is not None and (sr - so) >= _KIWI_TYPO_MARGIN:
                    sp["kiwi_morp_agrees"] = True

    n_eojeol = max(len(essay.split()), 1)
    n_spacing = sum(1 for s in spans if s["category"] == "SPACING")
    n_typo = sum(1 for s in spans if s["category"] == "TYPO")
    spacing_error_rate = round(n_spacing / n_eojeol, 4)

    strong_spans = [s for s in spans if s["kiwi_agrees"] or s.get("kiwi_morp_agrees")]
    if strong_spans:
        strength = "strong"
    elif spans:
        strength = "weak"
    else:
        strength = "none"

    n_norm = sum(1 for s in spans if s["norm"])

    sources = [
        source("BAREUN CorrectError", url="https://api.bareun.ai"),
        source("KIWI", ref="kiwipiepy"),
    ]
    if n_norm:
        sources.append(source("국립국어원 어문 규범", ref="data/norms/korean_norms.jsonl"))

    if spans:
        summary = (
            f"맞춤법/띄어쓰기 교정 후보 {len(spans)}건"
            f"(띄어쓰기 {n_spacing}·맞춤법 {n_typo}), KIWI 교차검증 일치 {len(strong_spans)}건"
            + (f", 어문규범 조항 연결 {n_norm}건" if n_norm else "")
            + "."
        )
    else:
        summary = "맞춤법/띄어쓰기 교정 후보 없음."

    return evidence(
        tool="orthography_probe",
        slot="expression_2",
        signals={
            "n_corrections": len(spans),
            "n_spacing": n_spacing,
            "n_typo": n_typo,
            "n_strong": len(strong_spans),
            "n_norm_grounded": n_norm,
            "spacing_error_rate": spacing_error_rate,
            "kiwi_spacing_offsets": spacing_offsets,
            "etri_morp_view": _etri_morp_view(essay),
            "bareun_morp_view": _bareun_morp_view(essay),
            "kiwi_morp_count": _kiwi_morp_count(essay),
        },
        spans=spans,
        sources=sources,
        strength=strength,
        summary=summary,
    )
