"""Evidence tools for the auditing agent.

Each tool *measures* (returns deterministic, provenance-carrying evidence);
the agent *interprets*. No hardcoded linguistic knowledge lives here — every
signal is grounded in an official API, a local analyzer, or a curated dataset.
API keys are read from .env and never written into evidence output.
"""
from __future__ import annotations

from ._common import ToolError, evidence, source
from .rubric_retrieve import rubric_retrieve, resolve_rubric_type
from .keyword_coverage import keyword_coverage
from .orthography_probe import orthography_probe
from .lexical_grounding import lexical_grounding, candidate_nouns
from .terminology_grounding import terminology_grounding
from .norm_lookup import norm_lookup, norm_search
from .linguistic_analysis import analyze_text

__all__ = [
    "ToolError",
    "evidence",
    "source",
    "rubric_retrieve",
    "resolve_rubric_type",
    "keyword_coverage",
    "orthography_probe",
    "lexical_grounding",
    "candidate_nouns",
    "terminology_grounding",
    "norm_lookup",
    "norm_search",
    "analyze_text",
]
