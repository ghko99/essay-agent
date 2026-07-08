"""FastAPI app for the essay scoring agent (vLLM-backed ensemble)."""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import agent as audit_agent
from .tools import analyze_text
from .model import EssayScorer
from .rubric import SLOT_KEYS, rubric_as_dict, resolve_rubric
from .topics import load_topics, rubric_for_prompt, topic_entry_for_prompt


APP_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = APP_ROOT / "frontend"

scorer = EssayScorer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the engine in the BACKGROUND so the HTTP server (and the web page)
    # is reachable immediately instead of blocking ~30s on vLLM load. Scoring
    # requests await the same load() (engine lock dedups), so the first score
    # waits only if the warm-up hasn't finished yet.
    if os.environ.get("KANANA_LAZY_LOAD") != "1":
        async def _warm():
            try:
                print("[startup] warming vLLM engine in background…")
                await scorer.load()
                print("[startup] vLLM engine ready.")
            except Exception as e:  # surface, don't crash the server
                print(f"[startup] engine warm-up failed: {e}")
        asyncio.create_task(_warm())
    yield


app = FastAPI(title="Kanana AES Agent — vLLM ensemble", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _no_cache_static(request, call_next):
    """Force the browser to revalidate the frontend assets.

    Without this, `index.html` updates but the browser keeps serving a cached
    `/static/app.js`/`style.css` (same URL), so frontend changes don't take
    effect until a hard refresh. no-store keeps dev iterations honest.
    """
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


# ── Schemas ─────────────────────────────────────────────────────────
class EnsembleRequest(BaseModel):
    topic: str
    essay: str
    keywords: str | None = None
    n_samples: int = 30
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    max_new_tokens: int = 1024
    top_feedback_k: int = 3
    fallback_score: int = 5


# ── Helpers ─────────────────────────────────────────────────────────
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _confidence(
    probs: list[float] | None,
    expected_score: float | None = None,
    confidence_scalar: float | None = None,
) -> dict | None:
    """Peak / expected / scalar confidence summary from the soft distribution.

    All values come from the *ensemble's aggregated digit distribution*
    — there is no greedy fallback path anymore.
    """
    if not probs or len(probs) != 9:
        return None
    peak_idx = max(range(9), key=lambda i: probs[i])
    if expected_score is None:
        expected_score = sum((i + 1) * p for i, p in enumerate(probs))
    return {
        "peak_score": peak_idx + 1,
        "peak_prob": round(probs[peak_idx], 4),
        "expected_score": round(float(expected_score), 2),
        "confidence": round(float(confidence_scalar), 4) if confidence_scalar is not None else None,
    }


def _build_criteria(
    scores: list[int],
    expected_scores: list[float] | None,
    reps: list[dict],
    rubric_names: list[str] | None,
    score_probs: list[list[float]] | None = None,
    confidence_scalars: list[float] | None = None,
) -> list[dict]:
    rubric = resolve_rubric(rubric_names)
    criteria = []
    for idx, c in enumerate(rubric):
        fbs = []
        for r in reps:
            fb = r["feedback"][idx] if idx < len(r["feedback"]) else ""
            if fb:
                fbs.append({
                    "score": r["scores"][idx] if idx < len(r["scores"]) else None,
                    "feedback": fb,
                    "distance": r["distance"],
                })
        probs_i = score_probs[idx] if (score_probs and idx < len(score_probs)) else None
        exp_i = expected_scores[idx] if (expected_scores and idx < len(expected_scores)) else None
        conf_i = confidence_scalars[idx] if (confidence_scalars and idx < len(confidence_scalars)) else None
        criteria.append({
            "key": c.key,
            "short": c.short,
            "full": c.full,
            "category": c.category,
            "score": scores[idx],
            "expected_score": exp_i,
            "feedbacks": fbs,
            "score_probs": probs_i,
            "confidence": _confidence(probs_i, exp_i, conf_i),
        })
    return criteria


# ── Routes ──────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "ready": scorer.is_ready(),
        "adapter_attached": scorer.adapter_attached,
        "base_model": scorer.base_model_path,
        "adapter": scorer.adapter_path,
    }


@app.get("/api/rubric")
def get_rubric(topic: str | None = None):
    names = rubric_for_prompt(topic) if topic else None
    rubric = resolve_rubric(names)
    return {
        "rubric": rubric_as_dict(rubric),
        "min_score": 1,
        "max_score": 9,
        "matched_topic": bool(names),
    }


@app.get("/api/rubric/guide")
def rubric_guide():
    """채점 기준 안내 페이지용: 루브릭 유형 × 슬롯별 5단계 평가 기준 전체."""
    path = APP_ROOT / "data" / "rubric_criteria.json"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"rubric_criteria.json 로드 실패: {e}")
    types = []
    for key, t in (data.get("rubrics") or {}).items():
        slots = t.get("slots") or {}
        types.append({
            "key": key,
            "label": t.get("label"),
            "purpose": t.get("purpose"),
            "description": t.get("description"),
            "slots": [
                {
                    "slot": sk,
                    "category": (slots.get(sk) or {}).get("category"),
                    "name": (slots.get(sk) or {}).get("name"),
                    "criteria": (slots.get(sk) or {}).get("criteria") or {},
                }
                for sk in SLOT_KEYS if sk in slots
            ],
        })
    return {"score_to_level": data.get("score_to_level"), "types": types}


@app.get("/api/topics")
def get_topics():
    return load_topics()


@app.post("/api/score/ensemble")
async def score_ensemble(req: EnsembleRequest):
    if not scorer.is_ready():
        await scorer.load()
    try:
        result = await scorer.score_ensemble(
            topic=req.topic,
            essay=req.essay,
            keywords=req.keywords,
            n_samples=req.n_samples,
            temperature=req.temperature,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            max_new_tokens=req.max_new_tokens,
            top_feedback_k=req.top_feedback_k,
            fallback_score=req.fallback_score,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    reps_list = [
        {"scores": r.scores, "feedback": r.feedback, "distance": r.distance}
        for r in result.representative
    ]
    rubric_names = rubric_for_prompt(req.topic)
    criteria = _build_criteria(
        result.scores,
        result.expected_scores,
        reps_list,
        rubric_names,
        score_probs=result.score_probs,
        confidence_scalars=result.confidence,
    )
    return {
        "step": "ensemble",
        "adapter_attached": scorer.adapter_attached,
        "total": result.total,
        "n_samples": result.n_samples,
        "n_valid": result.n_valid,
        "n_soft_valid": result.n_soft_valid,
        "criteria": criteria,
        "matched_topic": rubric_names is not None,
        "generation_ms": result.generation_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }


@app.post("/api/score/ensemble/stream")
async def score_ensemble_stream(req: EnsembleRequest):
    if not scorer.is_ready():
        await scorer.load()

    rubric_names = rubric_for_prompt(req.topic)

    async def event_gen():
        try:
            async for ev in scorer.score_ensemble_stream(
                topic=req.topic,
                essay=req.essay,
                keywords=req.keywords,
                n_samples=req.n_samples,
                temperature=req.temperature,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                max_new_tokens=req.max_new_tokens,
                top_feedback_k=req.top_feedback_k,
                fallback_score=req.fallback_score,
            ):
                t = ev.pop("type")
                if t == "done":
                    criteria = _build_criteria(
                        ev["scores"],
                        ev["expected_scores"],
                        ev["representative"],
                        rubric_names,
                        score_probs=ev.get("score_probs"),
                        confidence_scalars=ev.get("confidence"),
                    )
                    payload = {
                        "step": "ensemble",
                        "adapter_attached": scorer.adapter_attached,
                        "total": ev["total"],
                        "n_samples": ev["n_samples"],
                        "n_valid": ev["n_valid"],
                        "n_soft_valid": ev.get("n_soft_valid"),
                        "criteria": criteria,
                        "matched_topic": rubric_names is not None,
                        "generation_ms": ev["generation_ms"],
                        "input_tokens": ev["input_tokens"],
                        "output_tokens": ev["output_tokens"],
                    }
                    yield _sse("done", payload)
                else:
                    yield _sse(t, ev)
        except Exception as e:
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ── Verify agent (tool-calling self-verification, base Kanana LoRA OFF) ──
async def _score_to_criteria(req: EnsembleRequest) -> tuple[list[dict], dict]:
    """Run the LoRA scorer and return (criteria, raw ensemble summary)."""
    if not scorer.is_ready():
        await scorer.load()
    result = await scorer.score_ensemble(
        topic=req.topic,
        essay=req.essay,
        keywords=req.keywords,
        n_samples=req.n_samples,
        temperature=req.temperature,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        max_new_tokens=req.max_new_tokens,
        top_feedback_k=req.top_feedback_k,
        fallback_score=req.fallback_score,
    )
    reps_list = [
        {"scores": r.scores, "feedback": r.feedback, "distance": r.distance}
        for r in result.representative
    ]
    rubric_names = rubric_for_prompt(req.topic)
    criteria = _build_criteria(
        result.scores, result.expected_scores, reps_list, rubric_names,
        score_probs=result.score_probs, confidence_scalars=result.confidence,
    )
    summary = {
        "total": result.total,
        "n_samples": result.n_samples,
        "n_valid": result.n_valid,
        "generation_ms": result.generation_ms,
    }
    return criteria, summary


class VerifyRequest(BaseModel):
    topic: str
    essay: str
    keywords: str | None = None
    criteria: list[dict]


@app.post("/api/agent/verify")
async def verify_endpoint(req: VerifyRequest):
    """Tool-calling self-verification over already-computed scorer criteria.

    The base Kanana model (LoRA OFF) decides which evidence tools to call, reads
    the results, and self-verifies every slot. Non-streaming variant.
    """
    if not scorer.is_ready():
        await scorer.load()
    try:
        topic_entry = topic_entry_for_prompt(req.topic, req.keywords)
        report = await audit_agent.run_verify(scorer, req.essay, topic_entry, req.criteria)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    report["step"] = "verify"
    return report


@app.post("/api/agent/verify/stream")
async def verify_stream_endpoint(req: VerifyRequest):
    """Streaming tool-calling self-verification (SSE).

    Emits: start / token / tool_call / tool_result / done / error so the UI can
    show the agent calling tools and reasoning in real time.
    """
    if not scorer.is_ready():
        await scorer.load()
    topic_entry = topic_entry_for_prompt(req.topic, req.keywords)

    async def event_gen():
        try:
            async for ev in audit_agent.agent_verify_stream(
                scorer, req.essay, topic_entry, req.criteria,
            ):
                t = ev.pop("type")
                yield _sse(t, ev)
        except Exception as e:
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


class LinguisticRequest(BaseModel):
    essay: str


@app.post("/api/analyze/linguistic")
def analyze_linguistic_endpoint(req: LinguisticRequest):
    """ETRI 형태소·의존구문·의미역 분석 (시각화용, 온디맨드). 엔진 불필요."""
    try:
        return analyze_text(req.essay)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Backwards-compatible alias for the old audit route (now self-verification).
@app.post("/api/audit/from_criteria")
async def audit_from_criteria_endpoint(req: VerifyRequest):
    if not scorer.is_ready():
        await scorer.load()
    try:
        topic_entry = topic_entry_for_prompt(req.topic, req.keywords)
        report = await audit_agent.run_verify(scorer, req.essay, topic_entry, req.criteria)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    report["step"] = "verify"
    return report


# ── Static frontend ─────────────────────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
