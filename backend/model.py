"""vLLM-based async essay scorer with **weighted self-consistency**.

Pipeline (matches lora-self-consistency-aes/weighted_self_consistency.py):
  1. Generate N score-prefix samples one score token at a time, constraining
     each score position to digit tokens 1..9 (`allowed_token_ids`).
  2. At each score position, sum the probability mass of every token that
     decodes to digit 1..9, renormalize over digits only, then compute
     Σ d · P(d). For the Kanana AES tokenizer, this is exactly the 9 single
     digit tokens `1`..`9`.
  3. Aggregate across samples:
       agg_dist[j][s]  = mean over samples of P(score=s) at rubric j
       expected[j]     = mean over samples of expected_digit at rubric j
       final[j]        = round(clip(expected[j], 1, 9))
       confidence[j]   = 1 - H(agg_dist[j]) / log(9)

Key change from previous self-consistency:
  - The greedy single-pass probability extraction (`_score_probs`) is GONE.
  - All probability info comes from the ensemble's own score-prefix logprobs.
"""
from __future__ import annotations

import asyncio
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.lora.request import LoRARequest
from vllm.sampling_params import RequestOutputKind

from .prompt import build_chat_messages, parse_assistant


def _split_sentences_regex(text: str) -> list[tuple[int, int, str]]:
    """Fallback splitter: break on sentence-ending punctuation + whitespace/EOS."""
    sents: list[tuple[int, int, str]] = []
    start = 0
    for m in re.finditer(r"[.!?…]+(?:\s|$)", text):
        end = m.end()
        seg = text[start:end].strip()
        if seg:
            sents.append((start, end, seg))
        start = end
    if start < len(text):
        seg = text[start:].strip()
        if seg:
            sents.append((start, len(text), seg))
    return sents


def _split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Split into (begin, end, sentence) keeping char offsets for token mapping.

    Uses KIWI's Korean sentence splitter (robust to quotes, missing periods,
    decimals); falls back to punctuation-based splitting if KIWI is unavailable.
    """
    try:
        from .tools._common import get_kiwi
        sents: list[tuple[int, int, str]] = []
        for s in get_kiwi().split_into_sents(text):
            seg = (s.text or "").strip()
            if seg:
                sents.append((s.start, s.end, seg))
        if sents:
            return sents
    except Exception:
        pass
    return _split_sentences_regex(text)


DEFAULT_BASE_MODEL = os.environ.get("KANANA_BASE", "/home/khko/models/kanana")
DEFAULT_ADAPTER = os.environ.get(
    "KANANA_ADAPTER",
    str(Path(__file__).resolve().parent.parent / "adapter"),
)


@dataclass
class EnsembleSample:
    scores: list[int | None]                       # hard scores parsed from text
    feedback: list[str]
    raw: str
    distance: float = 0.0
    expected_scores: list[float] | None = None     # per-rubric expected digit (soft)
    digit_dists: list[list[float]] | None = None   # [8][9] per-sample distribution


@dataclass
class EnsembleResult:
    scores: list[int]                              # final hard = round(expected)
    expected_scores: list[float]                   # per-rubric soft expected (8,)
    score_probs: list[list[float]]                 # aggregated [8][9] from ensemble
    confidence: list[float]                        # per-rubric 0-1 (1 - H/log9)
    total: int | None
    representative: list[EnsembleSample]
    samples: list[EnsembleSample]
    n_samples: int
    n_valid: int                                   # samples with all 8 hard digits parsed
    n_soft_valid: int                              # samples with all 8 digit positions located
    generation_ms: int
    input_tokens: int
    output_tokens: int


class EssayScorer:
    """Async vLLM scorer using weighted self-consistency."""

    def __init__(
        self,
        base_model_path: str = DEFAULT_BASE_MODEL,
        adapter_path: str = DEFAULT_ADAPTER,
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.8,
        max_lora_rank: int = 16,
    ):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self._max_model_len = max_model_len
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_lora_rank = max_lora_rank

        self._engine: AsyncLLMEngine | None = None
        self._engine_lock = asyncio.Lock()
        self._tokenizer = None
        self._stop_ids: list[int] = []
        self._digit_token_ids: dict[int, int] = {}
        self._lora_req: LoRARequest | None = None
        self.adapter_attached = True

    # ── Lifecycle ────────────────────────────────────────────────────
    async def load(self) -> None:
        if self._engine is not None:
            return
        async with self._engine_lock:
            if self._engine is not None:
                return

            tok = AutoTokenizer.from_pretrained(
                self.adapter_path, use_fast=True, trust_remote_code=True
            )
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            stop_ids: list[int] = []
            if getattr(tok, "eos_token_id", None) is not None:
                stop_ids.append(tok.eos_token_id)
            for t in ["<|eot_id|>", "<|end_of_text|>", "</s>"]:
                try:
                    tid = tok.convert_tokens_to_ids(t)
                    if isinstance(tid, int) and tid != tok.unk_token_id and tid >= 0:
                        stop_ids.append(tid)
                except Exception:
                    pass
            self._stop_ids = sorted(set(stop_ids))

            digit_ids: dict[int, int] = {}
            decode_number = getattr(tok, "decode_number_token", None)
            if callable(decode_number):
                for token, token_id in tok.get_vocab().items():
                    try:
                        value = decode_number(token)
                    except ValueError:
                        continue
                    except Exception:
                        continue
                    try:
                        if float(value).is_integer():
                            digit = int(value)
                            if 1 <= digit <= 9:
                                digit_ids[token_id] = digit
                    except Exception:
                        continue

            if not digit_ids:
                for d in range(1, 10):
                    for form in [str(d), f" {d}"]:
                        ids = tok.encode(form, add_special_tokens=False)
                        if len(ids) == 1 and ids[0] not in digit_ids:
                            digit_ids[ids[0]] = d
            self._digit_token_ids = digit_ids

            args = AsyncEngineArgs(
                model=self.base_model_path,
                tensor_parallel_size=1,
                gpu_memory_utilization=self._gpu_memory_utilization,
                max_model_len=self._max_model_len,
                disable_log_stats=True,
                enable_lora=True,
                max_loras=1,
                max_lora_rank=self._max_lora_rank,
                trust_remote_code=True,
            )
            engine = AsyncLLMEngine.from_engine_args(args)

            self._tokenizer = tok
            self._engine = engine
            self._lora_req = LoRARequest("aes_step1", 1, self.adapter_path)
            self.adapter_attached = True

    def is_ready(self) -> bool:
        return self._engine is not None

    # ── Prompt ───────────────────────────────────────────────────────
    def _build_prompt(self, topic: str, essay: str, keywords: str | None) -> str:
        messages = build_chat_messages(topic, essay, keywords)
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def chat_prompt(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        """Render arbitrary chat messages with the model's template.

        Used by the auditing agent to prompt the **base** model (LoRA OFF).
        When `tools` is given, the functionary (v3-llama3.1) template renders the
        function schemas + the `<function=name>{...}</function>` calling
        instructions into the leading system block — this is how the verify agent
        does native tool calling.
        """
        if self._tokenizer is None:
            raise RuntimeError("engine/tokenizer not loaded; call load() first")
        return self._tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True
        )

    async def essay_perplexity(self, essay: str) -> dict:
        """Base-model (LoRA OFF) per-sentence perplexity as an inter-sentence
        *flow* signal: for each sentence, how naturally it follows the sentences
        before it. One forward pass — vLLM `prompt_logprobs` gives each token's
        log-prob in full left context; we segment per-token NLL by sentence.

        The first sentence has no preceding sentence, so "how well does it follow?"
        is undefined for it — it is the anchor. We therefore report it with
        ``ppl: None`` (flow=False) and exclude it from the overall figure, instead
        of letting its context-free opening token inflate it and distort the
        comparison. Sentences 2..N each condition on everything before them.

        Returns: {overall_ppl, overall_mean_nll, n_tokens,
        sentences:[{text, ppl, mean_nll, n_tokens, flow}]}.
        """
        if self._engine is None:
            await self.load()
        text = (essay or "").strip()
        if not text:
            raise RuntimeError("essay_perplexity: empty text")

        enc = self._tokenizer(text, add_special_tokens=False,
                              return_offsets_mapping=True)
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        if not ids:
            raise RuntimeError("essay_perplexity: no tokens")

        sampling = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1,
                                  output_kind=RequestOutputKind.FINAL_ONLY)
        plps = None
        async for out in self._engine.generate(
            prompt={"prompt_token_ids": ids},
            sampling_params=sampling,
            request_id=f"ppl-{uuid.uuid4()}",
            lora_request=None,          # base model, LoRA OFF
        ):
            if getattr(out, "prompt_logprobs", None):
                plps = out.prompt_logprobs
            if out.finished:
                break
        if not plps:
            raise RuntimeError("essay_perplexity: no prompt_logprobs returned")

        tok_nll: list[tuple[int, int, float]] = []
        for i, tid in enumerate(ids):
            lp_dict = plps[i] if i < len(plps) else None
            if not lp_dict:
                continue
            entry = lp_dict.get(tid)
            if entry is None:
                continue
            val = entry.logprob if hasattr(entry, "logprob") else float(entry)
            b, e = offsets[i]
            tok_nll.append((b, e, -float(val)))
        if not tok_nll:
            raise RuntimeError("essay_perplexity: could not extract token logprobs")

        # Group token NLLs by sentence (in document order, keeping non-empty ones).
        segs: list[tuple[str, list[float]]] = []
        for sb, se, stext in _split_sentences(text):
            sn = [n for (b, _e, n) in tok_nll if sb <= b < se]
            if sn:
                segs.append((stext, sn))

        sentences: list[dict] = []
        flow_nll: list[float] = []
        for idx, (stext, sn) in enumerate(segs):
            if idx == 0 and len(segs) > 1:
                # Anchor sentence: no preceding context → no flow PPL.
                sentences.append({"text": stext, "ppl": None, "mean_nll": None,
                                  "n_tokens": len(sn), "flow": False})
                continue
            m = sum(sn) / len(sn)
            sentences.append({
                "text": stext,
                "ppl": round(math.exp(min(m, 20.0)), 2),
                "mean_nll": round(m, 3),
                "n_tokens": len(sn),
                "flow": True,
            })
            flow_nll.extend(sn)

        # Overall = mean over the flow sentences (excludes the anchor). For a
        # single-sentence essay there is no flow, so fall back to all tokens.
        base = flow_nll if flow_nll else [n for *_, n in tok_nll]
        overall_nll = sum(base) / len(base)
        return {
            "overall_ppl": round(math.exp(min(overall_nll, 20.0)), 2),
            "overall_mean_nll": round(overall_nll, 3),
            "n_tokens": len(base),
            "sentences": sentences,
        }

    async def _generate_score_prefix_sample(
        self,
        prompt: str,
        sample_idx: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        logprobs_k: int,
    ) -> tuple[list[int], list[dict | None]]:
        """Sample the 8 leading score digits with a digit-only candidate set.

        vLLM in this environment caps returned logprobs at 20, so requesting
        full-vocab logprobs is not available. The AES tokenizer exposes exactly
        nine score digit tokens, so constraining each score step to those token
        IDs gives the desired 1..9 distribution directly.
        """
        digit_token_ids = sorted(self._digit_token_ids)
        if len(digit_token_ids) > 20:
            raise RuntimeError(
                f"digit token set has {len(digit_token_ids)} ids; vLLM logprobs cap is 20"
            )
        if not digit_token_ids:
            raise RuntimeError("digit token ids are not initialized")

        sampling = SamplingParams(
            n=1,
            max_tokens=1,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            logprobs=max(logprobs_k, len(digit_token_ids)),
            allowed_token_ids=digit_token_ids,
            stop_token_ids=self._stop_ids,
            ignore_eos=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

        prefix = ""
        token_ids: list[int] = []
        logprobs: list[dict | None] = []
        for slot_idx in range(8):
            chosen_id: int | None = None
            chosen_logprobs: dict | None = None
            async for out in self._engine.generate(
                prompt=prompt + prefix,
                sampling_params=sampling,
                request_id=f"ensemble-score-{sample_idx}-{slot_idx}-{uuid.uuid4()}",
                lora_request=self._lora_req,
            ):
                for completion in out.outputs:
                    if completion.token_ids:
                        chosen_id = int(completion.token_ids[-1])
                    if completion.logprobs:
                        chosen_logprobs = completion.logprobs[-1]
                if out.finished:
                    break

            if chosen_id is None:
                raise RuntimeError(f"score-prefix generation failed at sample={sample_idx}, slot={slot_idx}")

            token_ids.append(chosen_id)
            logprobs.append(chosen_logprobs)
            digit = self._digit_token_ids.get(chosen_id)
            prefix += str(digit) if digit is not None else self._tokenizer.decode([chosen_id])
            if slot_idx < 7:
                prefix += " "
        return token_ids, logprobs

    # ── Per-sample soft extraction ──────────────────────────────────
    def _per_sample_soft(
        self,
        token_ids: list[int],
        logprobs: list[dict | None],
    ) -> tuple[list[list[float]] | None, list[float] | None]:
        """Walk a sample's tokens; at the first 8 *digit* positions extract a
        digit-only renormalized distribution + expected digit.

        Returns (dists [8][9], expected [8]) or (None, None) on failure.
        """
        if not token_ids:
            return None, None
        dists: list[list[float]] = []
        expected: list[float] = []
        n = min(len(token_ids), len(logprobs)) if logprobs else len(token_ids)
        for i in range(n):
            tid = token_ids[i]
            if tid not in self._digit_token_ids:
                continue
            lp_dict = logprobs[i] if logprobs and i < len(logprobs) else None

            # Sum probability mass for every token that decodes to each digit.
            digit_mass = [0.0] * 9
            if lp_dict:
                for ltid, lp in lp_dict.items():
                    if ltid in self._digit_token_ids:
                        d = self._digit_token_ids[ltid]
                        v = lp.logprob if hasattr(lp, "logprob") else float(lp)
                        digit_mass[d - 1] += math.exp(v)

            total = sum(digit_mass)
            if total <= 0.0:
                # fallback: one-hot on the chosen digit (no top-K coverage)
                chosen_d = self._digit_token_ids[tid]
                dist = [0.0] * 9
                dist[chosen_d - 1] = 1.0
            else:
                # Renormalize over digit mass only, as in weighted_digit_inference.
                dist = [m / total for m in digit_mass]

            dists.append([round(p, 6) for p in dist])
            expected.append(sum((s + 1) * dist[s] for s in range(9)))
            if len(dists) == 8:
                return dists, expected

        return (None, None) if len(dists) < 8 else (dists, expected)

    # ── Cross-sample aggregation ─────────────────────────────────────
    @staticmethod
    def _aggregate_soft(
        per_sample_dists: list[list[list[float]] | None],
        per_sample_expected: list[list[float] | None],
        fallback_score: int = 5,
    ) -> tuple[list[list[float]], list[float], list[int], list[float], int]:
        """Weighted self-consistency aggregation.

        Returns:
            agg_dist        [8][9] mean P(score=s)
            expected_scores [8]    mean expected digit
            final_scores    [8]    round(clip(expected, 1, 9))
            confidence      [8]    1 − H(agg_dist) / log 9, clipped to [0,1]
            n_soft_valid    int    samples that contributed
        """
        valid_dists: list[list[list[float]]] = []
        valid_exp: list[list[float]] = []
        for d, e in zip(per_sample_dists, per_sample_expected):
            if d is not None and e is not None:
                valid_dists.append(d)
                valid_exp.append(e)
        n_soft = len(valid_dists)

        if n_soft == 0:
            uniform = [round(1 / 9, 6)] * 9
            return (
                [uniform[:] for _ in range(8)],
                [float(fallback_score)] * 8,
                [fallback_score] * 8,
                [0.0] * 8,
                0,
            )

        # mean distribution per rubric
        agg: list[list[float]] = []
        for j in range(8):
            sum_dist = [0.0] * 9
            for s_dist in valid_dists:
                for k in range(9):
                    sum_dist[k] += s_dist[j][k]
            agg.append([round(v / n_soft, 6) for v in sum_dist])

        # expected per rubric (use mean of per-sample expected for stability;
        # mathematically identical to Σ s · agg[j][s] up to rounding)
        expected_scores: list[float] = []
        for j in range(8):
            mean_exp = sum(s_exp[j] for s_exp in valid_exp) / n_soft
            expected_scores.append(round(mean_exp, 4))

        final_scores = [
            int(max(1, min(9, round(e)))) for e in expected_scores
        ]

        # entropy → confidence ∈ [0, 1]
        log9 = math.log(9)
        confidence: list[float] = []
        for d in agg:
            h = 0.0
            for p in d:
                if p > 0:
                    h -= p * math.log(p)
            c = max(0.0, min(1.0, 1.0 - h / log9))
            confidence.append(round(c, 4))

        return agg, expected_scores, final_scores, confidence, n_soft

    # ── Ensemble generation ──────────────────────────────────────────
    async def score_ensemble_stream(
        self,
        topic: str,
        essay: str,
        keywords: str | None = None,
        n_samples: int = 30,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 1024,
        top_feedback_k: int = 3,
        fallback_score: int = 5,
        emit_tokens: bool = True,
        logprobs_k: int = 9,
    ) -> AsyncIterator[dict]:
        """Yield events: start / token / sample / score_probs / done / error."""
        if self._engine is None:
            await self.load()

        prompt = self._build_prompt(topic, essay, keywords)
        prompt_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        input_tokens = len(prompt_ids)

        yield {"type": "start", "n_samples": n_samples, "input_tokens": input_tokens}

        t0 = time.time()
        out_tokens = 0

        try:
            score_results = await asyncio.gather(*[
                self._generate_score_prefix_sample(
                    prompt=prompt,
                    sample_idx=i,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    logprobs_k=logprobs_k,
                )
                for i in range(n_samples)
            ])
        except Exception as e:
            yield {"type": "error", "message": str(e)}
            return
        out_tokens += sum(len(ids) for ids, _ in score_results)

        # Per-sample soft score distributions from the short score-prefix pass.
        per_sample_dists: list[list[list[float]] | None] = []
        per_sample_expected: list[list[float] | None] = []
        for ids, lps in score_results:
            dists, expected = self._per_sample_soft(ids, lps)
            per_sample_dists.append(dists)
            per_sample_expected.append(expected)

        # Cross-sample aggregation
        agg_dist, expected_scores, final_scores, confidence, n_soft_valid = \
            self._aggregate_soft(per_sample_dists, per_sample_expected, fallback_score)

        yield {"type": "score_probs", "probs": agg_dist}

        feedback_sampling = SamplingParams(
            n=n_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_token_ids=self._stop_ids,
            ignore_eos=False,
            output_kind=RequestOutputKind.DELTA,
        )
        full_texts: list[str] = [""] * n_samples
        finished = [False] * n_samples

        try:
            async for out in self._engine.generate(
                prompt=prompt,
                sampling_params=feedback_sampling,
                request_id=f"ensemble-feedback-{uuid.uuid4()}",
                lora_request=self._lora_req,
            ):
                for completion in out.outputs:
                    idx = completion.index
                    delta = completion.text
                    if delta:
                        full_texts[idx] += delta
                        if emit_tokens and idx == 0:
                            yield {"type": "token", "text": delta}
                    if completion.token_ids:
                        out_tokens += len(completion.token_ids)
                    if completion.finish_reason is not None and not finished[idx]:
                        finished[idx] = True
                        parsed = parse_assistant(full_texts[idx])
                        yield {
                            "type": "sample",
                            "index": idx + 1,
                            "n_samples": n_samples,
                            "scores": parsed["scores"],
                        }
                if out.finished:
                    break
        except Exception as e:
            yield {"type": "error", "message": str(e)}
            return

        elapsed_ms = int((time.time() - t0) * 1000)

        samples: list[EnsembleSample] = []
        for i, text in enumerate(full_texts):
            parsed = parse_assistant(text)
            samples.append(EnsembleSample(
                scores=parsed["scores"],
                feedback=parsed["feedback"],
                raw=parsed["raw"],
                expected_scores=per_sample_expected[i],
                digit_dists=per_sample_dists[i],
            ))

        # Distance to final for representative selection (still uses hard scores
        # for compatibility with existing parse_assistant feedback alignment)
        for s in samples:
            d = 0.0
            for j in range(8):
                v = s.scores[j] if s.scores[j] is not None else fallback_score
                d += abs(v - final_scores[j])
            s.distance = d
        ordered = sorted(samples, key=lambda s: s.distance)
        reps = ordered[:top_feedback_k]
        n_valid = sum(1 for s in samples if all(v is not None for v in s.scores))

        yield {
            "type": "done",
            "scores": final_scores,
            "expected_scores": expected_scores,
            "score_probs": agg_dist,
            "confidence": confidence,
            "total": sum(final_scores),
            "representative": [
                {
                    "scores": r.scores, "feedback": r.feedback, "distance": r.distance,
                    "expected_scores": r.expected_scores,
                }
                for r in reps
            ],
            "samples": [
                {
                    "scores": s.scores, "feedback": s.feedback, "distance": s.distance,
                    "expected_scores": s.expected_scores,
                }
                for s in samples
            ],
            "n_samples": n_samples,
            "n_valid": n_valid,
            "n_soft_valid": n_soft_valid,
            "generation_ms": elapsed_ms,
            "input_tokens": input_tokens,
            "output_tokens": out_tokens,
        }

    async def score_ensemble(
        self,
        topic: str,
        essay: str,
        keywords: str | None = None,
        n_samples: int = 30,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 1024,
        top_feedback_k: int = 3,
        fallback_score: int = 5,
    ) -> EnsembleResult:
        done_payload: dict | None = None
        async for ev in self.score_ensemble_stream(
            topic=topic,
            essay=essay,
            keywords=keywords,
            n_samples=n_samples,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            top_feedback_k=top_feedback_k,
            fallback_score=fallback_score,
            emit_tokens=False,
        ):
            if ev["type"] == "done":
                done_payload = ev
            elif ev["type"] == "error":
                raise RuntimeError(ev["message"])
        if done_payload is None:
            raise RuntimeError("ensemble did not complete")

        samples = [
            EnsembleSample(
                scores=s["scores"],
                feedback=s["feedback"],
                distance=s["distance"],
                raw="",
                expected_scores=s.get("expected_scores"),
            )
            for s in done_payload["samples"]
        ]
        reps = [
            EnsembleSample(
                scores=r["scores"],
                feedback=r["feedback"],
                distance=r["distance"],
                raw="",
                expected_scores=r.get("expected_scores"),
            )
            for r in done_payload["representative"]
        ]
        return EnsembleResult(
            scores=done_payload["scores"],
            expected_scores=done_payload["expected_scores"],
            score_probs=done_payload["score_probs"],
            confidence=done_payload["confidence"],
            total=done_payload["total"],
            representative=reps,
            samples=samples,
            n_samples=done_payload["n_samples"],
            n_valid=done_payload["n_valid"],
            n_soft_valid=done_payload["n_soft_valid"],
            generation_ms=done_payload["generation_ms"],
            input_tokens=done_payload["input_tokens"],
            output_tokens=done_payload["output_tokens"],
        )

    async def generate_prompt_ensemble_stream(
        self,
        prompt: str,
        n_samples: int = 30,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 1024,
        emit_tokens: bool = False,
        use_lora: bool = True,
    ) -> AsyncIterator[dict]:
        """Generic prompt ensemble generation.

        `use_lora=False` runs the **base** Kanana model with no adapter — this is
        how the auditing agent reasons (the LoRA scorer stays untouched).
        """
        if self._engine is None:
            await self.load()
        lora_request = self._lora_req if use_lora else None

        prompt_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        input_tokens = len(prompt_ids)
        yield {"type": "start", "n_samples": n_samples, "input_tokens": input_tokens}

        sampling = SamplingParams(
            n=n_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_token_ids=self._stop_ids,
            ignore_eos=False,
            output_kind=RequestOutputKind.DELTA,
        )
        request_id = f"agent-ensemble-{uuid.uuid4()}"

        full_texts: list[str] = [""] * n_samples
        finished = [False] * n_samples
        t0 = time.time()
        out_tokens = 0

        try:
            async for out in self._engine.generate(
                prompt=prompt,
                sampling_params=sampling,
                request_id=request_id,
                lora_request=lora_request,
            ):
                for completion in out.outputs:
                    idx = completion.index
                    delta = completion.text
                    if delta:
                        full_texts[idx] += delta
                        if emit_tokens and idx == 0:
                            yield {"type": "token", "text": delta}
                    if completion.token_ids:
                        out_tokens += len(completion.token_ids)
                    if completion.finish_reason is not None and not finished[idx]:
                        finished[idx] = True
                        yield {
                            "type": "sample",
                            "index": idx + 1,
                            "n_samples": n_samples,
                            "text": full_texts[idx],
                        }
                if out.finished:
                    break
        except Exception as e:
            yield {"type": "error", "message": str(e)}
            return

        elapsed_ms = int((time.time() - t0) * 1000)
        n_valid = sum(1 for t in full_texts if t.strip())
        yield {
            "type": "done",
            "samples": full_texts,
            "n_samples": n_samples,
            "n_valid": n_valid,
            "generation_ms": elapsed_ms,
            "input_tokens": input_tokens,
            "output_tokens": out_tokens,
        }

    async def generate_prompt_ensemble(
        self,
        prompt: str,
        n_samples: int = 30,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 1024,
        emit_tokens: bool = False,
        use_lora: bool = True,
    ) -> dict:
        done_payload: dict | None = None
        async for ev in self.generate_prompt_ensemble_stream(
            prompt=prompt,
            n_samples=n_samples,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            emit_tokens=emit_tokens,
            use_lora=use_lora,
        ):
            if ev["type"] == "done":
                done_payload = ev
            elif ev["type"] == "error":
                raise RuntimeError(ev["message"])
        if done_payload is None:
            raise RuntimeError("agent ensemble did not complete")
        return done_payload
