"""Prompt formatter that matches the training-time format of the LoRA adapter.

Training format (from aes-llm-training/collator.py + aes_datasets/*.jsonl):
  system: "에세이 채점기. 8개 루브릭(...)별 1-9점 채점 후 피드백을 작성한다."
  user:   "질문: {topic}\n에세이: {essay}\n핵심 키워드: {keywords}"

The assistant output begins with 8 space-separated scores (1-9),
then a blank line, then "### Feedback:" and 8 bullet items.

The 8 bullets always appear in the canonical slot order below, but the
**bullet header name** varies per topic depending on which of three rubric
variants the topic was trained with:

  slot 0 (task_1)         → 과제 수행의 충실성              (all topics)
  slot 1 (content_1)      → 설명의 명료성 | 주장의 명료성 | 주제 전달 및 정서 표현의 명료성
  slot 2 (content_2)      → 설명의 구체성 | 주장의 적절성 | 주제 전달 및 정서 표현의 구체성
  slot 3 (content_3)      → 설명의 적절성 | 근거의 타당성 | 주제 전달 및 정서 표현의 적절성
  slot 4 (organization_1) → 문장의 연결성 | 문장 및 문단의 연결성
  slot 5 (organization_2) → 글의 통일성                     (all topics)
  slot 6 (expression_1)   → 어휘의 적절성 | 어휘 및 문장의 적절성
  slot 7 (expression_2)   → 어법의 적절성 | 어법의 정확성

`parse_assistant()` builds a name→slot lookup from the union of these
variants, so any header the model emits is routed to the correct slot.
"""
from __future__ import annotations

import re
import unicodedata

SYSTEM_PROMPT = (
    "에세이 채점기. 8개 루브릭(과제충실성, 설명명료성, 설명구체성, 설명적절성, "
    "문장연결성, 글통일성, 어휘적절성, 어법적절성)별 1-9점 채점 후 피드백을 작성한다."
)


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def build_user_message(topic: str, essay: str, keywords: str | None = None) -> str:
    topic = normalize(topic)
    essay = normalize(essay)
    parts = [f"질문: {topic}", f"에세이: {essay}"]
    if keywords and keywords.strip():
        parts.append(f"핵심 키워드: {normalize(keywords)}")
    return "\n".join(parts)


def build_chat_messages(topic: str, essay: str, keywords: str | None = None) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(topic, essay, keywords)},
    ]


# ── Header variants → slot index ─────────────────────────────────────
# Union of all names observed across the 3 rubric variants in raw_datasets.
_HEADER_VARIANTS: list[tuple[int, str]] = [
    (0, "과제 수행의 충실성"),
    (1, "설명의 명료성"),
    (1, "주장의 명료성"),
    (1, "주제 전달 및 정서 표현의 명료성"),
    (2, "설명의 구체성"),
    (2, "주장의 적절성"),
    (2, "주제 전달 및 정서 표현의 구체성"),
    (3, "설명의 적절성"),
    (3, "근거의 타당성"),
    (3, "주제 전달 및 정서 표현의 적절성"),
    (4, "문장의 연결성"),
    (4, "문장 및 문단의 연결성"),
    (5, "글의 통일성"),
    (6, "어휘의 적절성"),
    (6, "어휘 및 문장의 적절성"),
    (7, "어법의 적절성"),
    (7, "어법의 정확성"),
]


def _canon(s: str) -> str:
    """Lowercase + strip + collapse whitespace for forgiving header match."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", s)).strip()


_NAME_TO_SLOT: dict[str, int] = {_canon(n): i for i, n in _HEADER_VARIANTS}
# Longest-first so regex greedily captures the fullest header string.
_HEADERS_SORTED = sorted({n for _, n in _HEADER_VARIANTS}, key=len, reverse=True)
_HEADER_PATTERN = re.compile(
    r"(?m)^[ \t]*[-*•●·][ \t]*(" + "|".join(re.escape(h) for h in _HEADERS_SORTED) + r")[ \t]*:[ \t]*\n?"
)


def parse_assistant(text: str) -> dict:
    """Parse the adapter's generation into structured scores + feedback.

    Returns:
        {
          "scores":   [int | None] * 8,
          "feedback": [str]        * 8,
          "raw":      str
        }

    Robust to:
      * any of the 3 rubric name variants per slot
      * bullet chars: `-`, `*`, `•`, `●`, `·`
      * minor whitespace differences inside the header
      * output that is cut mid-section (remaining slots stay empty)
    """
    raw = text.strip()
    scores: list[int | None] = [None] * 8
    feedback: list[str] = [""] * 8

    # First non-empty line should contain the 8 scores.
    first_line = ""
    for line in raw.splitlines():
        if line.strip():
            first_line = line.strip()
            break

    nums = re.findall(r"\d+", first_line)
    for i in range(min(8, len(nums))):
        try:
            v = int(nums[i])
            if 1 <= v <= 9:
                scores[i] = v
        except ValueError:
            pass

    # Body after "### Feedback" if present; else the whole text.
    fb_start = raw.find("### Feedback")
    fb_body = raw[fb_start:] if fb_start >= 0 else raw

    # Find every header occurrence and slice body up to the next one.
    matches = list(_HEADER_PATTERN.finditer(fb_body))
    for mi, m in enumerate(matches):
        header = m.group(1)
        slot = _NAME_TO_SLOT.get(_canon(header))
        if slot is None:
            continue
        body_start = m.end()
        body_end = matches[mi + 1].start() if mi + 1 < len(matches) else len(fb_body)
        body = fb_body[body_start:body_end].strip()
        if body and not feedback[slot]:
            feedback[slot] = body

    return {"scores": scores, "feedback": feedback, "raw": raw}
