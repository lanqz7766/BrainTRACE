#!/usr/bin/env python3
"""BrainTRACE - LLM-as-judge scorer (spec v0.5).

Augments the deterministic v0.3 rubric scorer (and the v0.4 chain
aggregator that wraps it) with a structured LLM-as-judge pass. The
v0.3 regex rubric is high-precision but low-recall: a paraphrase that
swaps "left cerebellar tonsil" for "left inferior cerebellar
hemisphere abutting the medulla" is clinically equivalent yet misses
the literal regex. v0.5 layers a board-certified-neuroradiologist-
persona LLM judge on top of every step to recover semantic equivalence
while preserving full audit trail.

Design contract
---------------
- **Per-step holistic judging.** One judge call evaluates *all* rubric
  slots for one step in a single structured-JSON response. Cost-
  optimal for the case-level reasoning layout (6 steps -> 6 calls/item).
- **Pinned judge model.** ``gpt-4o-mini-2024-07-18`` by default;
  decoding ``temperature=0, top_p=1, max_tokens=600``.
- **Full audit trail.** Every API call is persisted to disk with the
  exact messages, raw response, parsed assessment, model version,
  token counts, latency, and timestamp. ``replay.py`` reads these
  files to reproduce a judgement; ``aggregate.py`` rebuilds per-item
  metrics from the persisted per-step records.
- **Resume-friendly.** ``--skip-existing`` checks for a prior judge
  call file at ``judge_calls/<item>__<step>.json`` and skips it; the
  per-step / per-item JSONL outputs are rewritten on every run from
  the union of fresh + cached records.
- **Async parallelism.** A bounded ``asyncio.Semaphore`` over a single
  ``httpx.AsyncClient`` lets us run up to ``--max-parallel`` (default
  8) concurrent OpenAI calls without exceeding rate limits.

CLI
---
::

    python llm_judge.py \\
        --scored-jsonl results/<run>/scored_v0_3.jsonl \\
        --gt-root benchmark/gt/case_reasoning \\
        --questions-root benchmark/questions/case_reasoning \\
        --out-dir results/_cross_model/_llm_judge_2026-04-27/<model_slug> \\
        --judge-model gpt-4o-mini-2024-07-18 \\
        --max-parallel 8 \\
        --skip-existing \\
        --limit 10 \\
        --dry-run

Operates on either the v0.4 chain layout (``--scored-jsonl`` is the
``scored_v0_3.jsonl`` produced by ``scoring_chain.py``, *and* the
inference output dir lives under ``<run>/outputs/<template>/<item>.output.json``)
or a non-chain single-step layout (``--outputs-dir`` points to the
flat directory of ``<item>.output.json``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# httpx is only needed for the actual API calls. The pure functions in
# this module (parse_judge_response, build_prompt, aggregation, etc.)
# must remain importable without httpx so that test_llm_judge can run
# in lightweight environments. The CLI entrypoint enforces the dep.
try:
    import httpx  # type: ignore
    _HTTPX_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as exc:  # pragma: no cover -- import guard
    httpx = None  # type: ignore
    _HTTPX_IMPORT_ERROR = exc


SCORING_SPEC_VERSION = "0.5.0"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_MAX_PARALLEL = 8
DEFAULT_RETRIES = 3

# Token-budget guard: if model_output exceeds this many *characters*
# (a conservative ~4 chars/token rule of thumb), we truncate the body
# and append a marker. 24_000 chars ~= 6_000 tokens, leaving headroom
# for the ~2_000-token prompt scaffolding under the 8K input cap.
MODEL_OUTPUT_CHAR_CAP = 24_000
TRUNCATION_MARKER = "\n\n[...truncated for judge]"

# Pricing table for the supported judge models. Update as OpenAI
# pricing changes; values are USD per 1 *million* tokens.
JUDGE_PRICING_PER_M_TOKENS: Dict[str, Tuple[float, float]] = {
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
}
DEFAULT_PRICING = (0.15, 0.60)


# ---------------------------------------------------------------------------
# Prompt scaffolding
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are a board-certified neuroradiologist acting as a factual coverage auditor for a vision-language model answer.

Your task is not to grade writing quality or radiology style. Your task is to decide whether the model answer explicitly covers the required clinical facts in the reference rubric without contradicting them or adding unsupported critical claims.

== CRITICAL RULE 1: Reference-Anchored Required Facts ==
The required_fact_results MUST be derived from the REFERENCE answer's specific claims, NOT from anything the model said. If the reference says "T2-weighted MRI demonstrating a right cerebellar focal abnormality", the required facts are exactly:
  - modality is T2-weighted MRI
  - region is right cerebellar
  - focal abnormality present
Do NOT invent required facts that match what the model said (e.g., do NOT add "modality is FLAIR" just because the model said FLAIR). Anchor every required fact to a specific phrase in the reference.

== CRITICAL RULE 2: Final-Answer-Only Evaluation (Anti-Thinking-Token Game) ==
If the model output contains chain-of-thought reasoning (signs: "Let me think...", "Step 1...", "Wait, actually...", "Let me look closer...", numbered analysis steps, "The user wants..."), evaluate whether the model's FINAL stated conclusion matches the reference. Mid-reasoning consideration of multiple possibilities does NOT count as the model "mentioning" a fact:
  - If the model wrote "This could be T2... wait, actually it looks like FLAIR" and the reference says "T2", the model's final claim is FLAIR — that is a critical_error of type wrong_modality_reading.
  - Do NOT pass a slot just because the correct word appeared somewhere mid-reasoning. The model must commit to the correct answer.

== CRITICAL RULE 3: Slot Evaluation Anchors to Reference ==
For each rubric slot, criterion_met=true ONLY if the model's stated value matches the REFERENCE's specific value. Substituting an alternative semantically-similar value (e.g., FLAIR for T2-weighted, frontal for temporal, left for right) is criterion_met=false and a critical_error.

Default to clinically_acceptable=false. Set clinically_acceptable=true only if ALL of the following are true:
1. Every required clinical fact from the reference is explicitly present or clinically-equivalently paraphrased.
2. Every critical rubric slot is satisfied with the correct finding, anatomy, laterality, direction, timepoint or interval, and current status when applicable.
3. The answer contains no critical factual contradiction, including wrong anatomy, wrong laterality, wrong direction of change, wrong timepoint, wrong diagnosis, wrong modality-specific reading, wrong temporal phase, wrong severity, or hallucinated landmark.
4. The answer contains no unsupported critical clinical claim that would change diagnosis, treatment response, disease status, or longitudinal interpretation.

Do not infer a missing named finding, diagnosis, laterality, trajectory, or timepoint from generic radiology-style language. Generic phrases such as "abnormality", "lesion", "stable disease", "overall status", "post-treatment change", or "clinical correlation recommended" do not satisfy a specific required fact unless the specific clinical entity and required direction/status are stated.

For trajectory or interval-change criteria, require the correct subject AND correct direction. An answer that names a lesion but reverses progression, improvement, stability, new appearance, or resolution fails the relevant criterion and must produce a critical error.

For L5 case-synthesis answers, fluency is insufficient. The answer must name the key diagnosis or finding(s), preserve required location/laterality when present, state the longitudinal/current status, and include required evidence or differential support when the rubric asks for it.

Be strict but clinically fair: accept genuinely equivalent medical paraphrases, but reject vague hand-waving, missing named findings, hallucinated landmarks, wrong laterality, wrong anatomy, wrong direction of change, wrong timepoint, or unsupported disease-status claims.

== Calibration examples ==

Example 1 (PASS — correct paraphrase):
  Reference fact: "left frontal arteriovenous malformation is stable."
  Model answer: "Unchanged left frontal AVM without interval enlargement."
  Verdict: clinically_acceptable=true. Named finding (AVM), location (left frontal), trajectory (stable) all preserved.

Example 2 (FAIL — fluent but missing named finding):
  Reference fact: "stable chronic demyelinating plaques with a new left frontal subcortical plaque."
  Model answer: "There is stable intracranial abnormality without acute change."
  Verdict: clinically_acceptable=false. Generic "abnormality" does not name "demyelinating plaques", and the new left frontal subcortical plaque is omitted. List both as missing required facts.

Example 3 (FAIL — wrong direction):
  Reference fact: "enhancement has decreased compared with prior study."
  Model answer: "Enhancement has increased, concerning for progression."
  Verdict: clinically_acceptable=false. critical_error type=wrong_direction.

Example 4 (FAIL — unsupported extra finding):
  Reference fact: "right temporal lesion is stable; no cerebellar lesion is described."
  Model answer: "Stable right temporal lesion with a new cerebellar metastasis."
  Verdict: clinically_acceptable=false. List "new cerebellar metastasis" in unsupported_claims; this is clinically material.

Return ONLY a JSON object matching the requested schema. Do NOT wrap it in markdown fences, do NOT prepend prose, do NOT trail commentary."""

JUDGE_USER_TEMPLATE = """## Question
{question}

## Reference answer
{reference_answer}

## Acceptable phrasing variants
{accepted_answer_variants}

## Auto-fail conditions
{rejection_criteria}

## Required clinical facts to check
{required_clinical_facts}

If the section above is empty, infer the required clinical facts from the reference answer and rubric slots. Treat named diagnoses, named findings, required anatomy/laterality, trajectory direction, timepoint/interval, new-or-absent finding status, and current overall status as required when clinically central to the reference.

## Sub-criteria to evaluate
{rubric_slots_with_criteria}

## Model's answer
{model_output}

## Decision procedure
1. Extract the model's clinically meaningful claims.
2. Check every required clinical fact. Mark it missing if it is absent, only generically implied, or replaced by a non-equivalent finding.
3. Check every rubric slot in order. A slot is met only when the answer contains the required fact with correct clinical meaning.
4. Check contradictions and unsupported claims. Contradictions in anatomy, laterality, trajectory direction, timepoint, diagnosis, temporal phase, severity, or modality reading are critical errors.
5. Set clinically_acceptable=false if any required clinical fact is missing, any critical slot is missed, any critical error is present, or any unsupported critical clinical claim is present.

## Your task
Output ONLY a JSON object with this schema:
{{
  "required_fact_results": [
    {{"fact": "<short required clinical fact>", "met": true|false, "evidence": "<short quote from the model answer, or empty string>", "issue": "<if met=false, short reason; else null>"}}
  ],
  "slot_results": [
    {{"slot_name": "<exact slot_name>", "criterion_met": true|false, "evidence": "<short quote from the model's answer that supports your decision, or empty string>", "issue": "<if criterion_met=false, a short reason; else null>"}}
  ],
  "critical_errors": [
    {{"type": "wrong_anatomy|wrong_laterality|wrong_direction|wrong_timepoint|wrong_modality_reading|wrong_temporal_phase|wrong_severity|wrong_diagnosis|hallucinated_landmark|other", "description": "..."}}
  ],
  "unsupported_claims": [
    {{"claim": "<short phrase from the model answer>", "rationale": "..."}}
  ],
  "clinically_acceptable": true|false,
  "reasoning": "<1-2 sentences explaining the overall verdict>"
}}

Rules:
- slot_results MUST list every slot_name from "Sub-criteria to evaluate", in the same order, with no additions and no omissions.
- required_fact_results MUST list every required clinical fact (provided or inferred), each with a met=true|false decision.
- clinically_acceptable MUST be false if any required clinical fact is missing.
- clinically_acceptable MUST be false if any critical rubric slot is missed.
- clinically_acceptable MUST be false if any critical_error is present.
- clinically_acceptable MUST be false if any unsupported critical claim would alter diagnosis, disease status, response assessment, or longitudinal interpretation.
- A generic statement does NOT satisfy a named fact. Example: "stable abnormality" does not satisfy "stable chronic demyelinating plaques with a new left frontal subcortical plaque" unless those facts are explicitly covered.
- For a trajectory slot, the answer must state both the clinical subject and the correct direction of change.
- Output JSON only, no markdown fences."""

JSON_RETRY_SUFFIX = (
    "\n\nReminder: Return ONLY a single JSON object. No markdown, no prose "
    "before or after. Begin your reply with `{` and end with `}`."
)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("llm_judge")
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Raised when the judge's response cannot be parsed into valid JSON
    matching the expected schema, even after a retry."""


class JudgeAPIError(Exception):
    """Raised when the judge API call fails after exhausting retries
    (non-rate-limit error, e.g. 4xx auth, malformed request)."""


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

@dataclass
class CostAccumulator:
    """Running tally of token usage and dollar cost across all calls."""

    judge_model: str
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    n_failures: int = 0

    def pricing(self) -> Tuple[float, float]:
        return JUDGE_PRICING_PER_M_TOKENS.get(self.judge_model, DEFAULT_PRICING)

    def record(self, prompt_tokens: int, completion_tokens: int) -> float:
        in_rate, out_rate = self.pricing()
        cost = (prompt_tokens / 1e6) * in_rate + (completion_tokens / 1e6) * out_rate
        self.total_calls += 1
        self.total_input_tokens += prompt_tokens
        self.total_output_tokens += completion_tokens
        self.total_cost_usd += cost
        return cost

    def record_failure(self) -> None:
        self.n_failures += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "judge_model": self.judge_model,
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "n_failures": self.n_failures,
        }


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _format_list_block(items: Optional[Sequence[Any]]) -> str:
    """Render a list of strings as a bullet block, or "(none)" if empty."""
    if not items:
        return "(none)"
    out_lines: List[str] = []
    for entry in items:
        if entry is None:
            continue
        text = str(entry).strip()
        if not text:
            continue
        out_lines.append(f"- {text}")
    return "\n".join(out_lines) if out_lines else "(none)"


def _format_rubric_block(rubric_slots: Sequence[Mapping[str, Any]]) -> str:
    """Render rubric_slots as an enumerated block of ``slot_name -> criterion``."""
    if not rubric_slots:
        return "(no rubric slots provided)"
    lines: List[str] = []
    for idx, slot in enumerate(rubric_slots, start=1):
        name = str(slot.get("slot_name") or f"slot_{idx}")
        criterion = (
            slot.get("llm_judge_criterion")
            or slot.get("criterion")
            or "Does the answer satisfy this rubric slot semantically?"
        )
        weight = slot.get("weight")
        weight_suffix = f" (weight {weight})" if weight is not None else ""
        lines.append(f"{idx}. slot_name: {name}{weight_suffix}\n   criterion: {criterion}")
    return "\n".join(lines)


def _truncate_model_output(text: str, char_cap: int = MODEL_OUTPUT_CHAR_CAP) -> Tuple[str, bool]:
    """Truncate model output to char_cap characters; return (text, was_truncated)."""
    if not isinstance(text, str):
        text = str(text or "")
    if len(text) <= char_cap:
        return text, False
    return text[:char_cap].rstrip() + TRUNCATION_MARKER, True


def build_prompt(
    question: str,
    gt_step: Mapping[str, Any],
    model_output: str,
) -> Tuple[str, str, bool]:
    """Build (system_msg, user_msg, was_truncated) for one step's judge call.

    Parameters
    ----------
    question : str
        The chain step's question text. Empty string is acceptable but
        should be logged upstream.
    gt_step : Mapping[str, Any]
        The per-step GT dict. Required keys consulted: ``reference_answer``,
        ``accepted_answer_variants``, ``rejection_criteria``,
        ``rubric_slots`` (with ``slot_name`` + ``llm_judge_criterion``).
    model_output : str
        The raw model answer text for this step.

    Returns
    -------
    (system_msg, user_msg, was_truncated)
    """
    reference_answer = str(gt_step.get("reference_answer") or "(no reference provided)")
    variants_block = _format_list_block(gt_step.get("accepted_answer_variants"))
    rejections_block = _format_list_block(gt_step.get("rejection_criteria"))
    rubric_block = _format_rubric_block(gt_step.get("rubric_slots") or [])
    truncated_text, was_truncated = _truncate_model_output(model_output or "")

    user = JUDGE_USER_TEMPLATE.format(
        question=question or "(question text unavailable)",
        reference_answer=reference_answer,
        accepted_answer_variants=variants_block,
        rejection_criteria=rejections_block,
        required_clinical_facts="",
        rubric_slots_with_criteria=rubric_block,
        model_output=truncated_text or "(empty answer)",
    )
    return JUDGE_SYSTEM_PROMPT, user, was_truncated


# ---------------------------------------------------------------------------
# Response parsing + validation
# ---------------------------------------------------------------------------

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", flags=re.DOTALL)


def _extract_json_blob(raw: str) -> str:
    """Pull a JSON object out of a possibly-fenced or prose-padded reply.

    Returns a substring expected to be a JSON object. Raises
    :class:`ParseError` if the reply is empty, has no braces, or appears
    to be a top-level JSON array (which would violate the schema).
    """
    if not isinstance(raw, str):
        raise ParseError(f"judge returned non-string body: {type(raw).__name__}")
    text = raw.strip()
    if not text:
        raise ParseError("judge returned empty body")
    # Strip markdown fences if present.
    fence_match = _FENCED_JSON_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()
    # Hard reject a top-level JSON array (judge ignored "JSON object" instruction).
    if text.startswith("["):
        raise ParseError("judge returned a JSON array at top level; expected object")
    # If reply has prose then JSON, snip from first `{` to matching `}`.
    if not text.startswith("{"):
        first = text.find("{")
        last = text.rfind("}")
        if first == -1 or last == -1 or last < first:
            raise ParseError("no JSON object braces found in judge reply")
        text = text[first:last + 1]
    return text


def parse_judge_response(raw_response: str, expected_slot_names: Sequence[str]) -> Dict[str, Any]:
    """Parse a judge raw response into a validated assessment dict.

    Validates:
    - top-level is a dict
    - ``slot_results`` is a list of dicts with ``slot_name`` (str) and
      ``criterion_met`` (bool); fills in ``evidence`` / ``issue`` if absent
    - ``critical_errors`` / ``unsupported_claims`` are lists (possibly empty)
    - ``clinically_acceptable`` is bool
    - ``reasoning`` is str
    - missing slots (in ``expected_slot_names`` but absent from response)
      are added as ``criterion_met=False`` with ``issue="missing from judge response"``;
      extra slots not in ``expected_slot_names`` are kept but flagged

    Raises
    ------
    ParseError
        If the JSON cannot be decoded or the top-level shape is wrong.
    """
    blob = _extract_json_blob(raw_response)
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ParseError(f"json.loads failed: {exc.msg} at pos {exc.pos}") from exc
    if not isinstance(parsed, dict):
        raise ParseError(f"judge reply is not a JSON object (got {type(parsed).__name__})")

    # --- slot_results ------------------------------------------------------
    raw_slots = parsed.get("slot_results")
    if not isinstance(raw_slots, list):
        raise ParseError("slot_results missing or not a list")

    slot_index: Dict[str, Dict[str, Any]] = {}
    extra_slots: List[Dict[str, Any]] = []
    for entry in raw_slots:
        if not isinstance(entry, dict):
            continue
        name = entry.get("slot_name")
        if not isinstance(name, str) or not name:
            continue
        norm: Dict[str, Any] = {
            "slot_name": name,
            "criterion_met": bool(entry.get("criterion_met", False)),
            "evidence": str(entry.get("evidence") or ""),
            "issue": entry.get("issue") if entry.get("issue") not in ("", None) else None,
        }
        if name in slot_index:
            # Duplicate slot in judge reply: keep the first, log via issue.
            continue
        slot_index[name] = norm

    canonical_slots: List[Dict[str, Any]] = []
    for name in expected_slot_names:
        if name in slot_index:
            canonical_slots.append(slot_index.pop(name))
        else:
            canonical_slots.append({
                "slot_name": name,
                "criterion_met": False,
                "evidence": "",
                "issue": "missing from judge response",
            })
    # Anything still in slot_index is an "extra" slot the judge invented.
    for leftover in slot_index.values():
        leftover["issue"] = (leftover.get("issue") or "extra slot not in rubric")
        extra_slots.append(leftover)

    # --- critical_errors ---------------------------------------------------
    raw_crit = parsed.get("critical_errors") or []
    crit: List[Dict[str, Any]] = []
    if isinstance(raw_crit, list):
        for entry in raw_crit:
            if not isinstance(entry, dict):
                continue
            crit.append({
                "type": str(entry.get("type") or "other"),
                "description": str(entry.get("description") or ""),
            })

    # --- unsupported_claims -----------------------------------------------
    raw_unsup = parsed.get("unsupported_claims") or []
    unsup: List[Dict[str, Any]] = []
    if isinstance(raw_unsup, list):
        for entry in raw_unsup:
            if not isinstance(entry, dict):
                continue
            unsup.append({
                "claim": str(entry.get("claim") or ""),
                "rationale": str(entry.get("rationale") or ""),
            })

    # --- required_fact_results (v0.6+, optional/backward-compat) ---------
    raw_facts = parsed.get("required_fact_results") or []
    required_facts: List[Dict[str, Any]] = []
    if isinstance(raw_facts, list):
        for entry in raw_facts:
            if not isinstance(entry, dict):
                continue
            fact = str(entry.get("fact") or "")
            if not fact:
                continue
            required_facts.append({
                "fact": fact,
                "met": bool(entry.get("met", False)),
                "evidence": str(entry.get("evidence") or ""),
                "issue": entry.get("issue") if entry.get("issue") not in ("", None) else None,
            })

    # --- top-level booleans / reasoning -----------------------------------
    clinically_acceptable = bool(parsed.get("clinically_acceptable", False))
    reasoning = str(parsed.get("reasoning") or "")

    return {
        "required_fact_results": required_facts,
        "slot_results": canonical_slots,
        "extra_slot_results": extra_slots,
        "critical_errors": crit,
        "unsupported_claims": unsup,
        "clinically_acceptable": clinically_acceptable,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# OpenAI Chat Completions wrapper
# ---------------------------------------------------------------------------

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


async def _post_chat_completion(
    client: httpx.AsyncClient,
    api_key: str,
    judge_model: str,
    system_msg: str,
    user_msg: str,
    *,
    max_tokens: int = 600,
    temperature: float = 0.0,
    top_p: float = 1.0,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Single Chat Completions POST. Returns the parsed JSON envelope."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": judge_model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    }
    resp = await client.post(OPENAI_CHAT_URL, headers=headers, json=body, timeout=timeout)
    if resp.status_code != 200:
        # Re-raise with the body for diagnostics.
        text = resp.text[:1000]
        raise httpx.HTTPStatusError(
            f"OpenAI returned {resp.status_code}: {text}",
            request=resp.request,
            response=resp,
        )
    return resp.json()


async def call_judge(
    client: httpx.AsyncClient,
    api_key: str,
    judge_model: str,
    system_msg: str,
    user_msg: str,
    expected_slot_names: Sequence[str],
    *,
    max_tokens: int = 600,
    retries: int = DEFAULT_RETRIES,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Make one judge call with retry-on-rate-limit and one parse-retry.

    Returns a dict with keys:
    - ``raw_response``: full text body returned by the judge
    - ``parsed_assessment``: validated assessment dict
    - ``prompt_tokens`` / ``completion_tokens``: token counts (0 if absent)
    - ``model_version``: model id reported by the API
    - ``wall_ms``: total wall-clock ms across retries
    - ``parse_retry_used``: bool
    """
    logger = logger or logging.getLogger("llm_judge")
    backoff_schedule = [4.0, 8.0, 16.0, 32.0]
    attempt = 0
    last_exc: Optional[Exception] = None
    t_start = time.monotonic()
    parse_retry_used = False

    current_user = user_msg
    while attempt < retries:
        attempt += 1
        try:
            envelope = await _post_chat_completion(
                client, api_key, judge_model, system_msg, current_user,
                max_tokens=max_tokens,
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429 or status >= 500:
                delay = backoff_schedule[min(attempt - 1, len(backoff_schedule) - 1)]
                logger.warning(
                    "judge HTTP %s on attempt %d/%d; backing off %.0fs",
                    status, attempt, retries, delay,
                )
                last_exc = exc
                await asyncio.sleep(delay)
                continue
            # 4xx other than 429: do not retry, fail fast.
            raise JudgeAPIError(f"judge API HTTP {status}: {str(exc)[:500]}") from exc
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ReadError) as exc:
            delay = backoff_schedule[min(attempt - 1, len(backoff_schedule) - 1)]
            logger.warning(
                "judge transport error %s on attempt %d/%d; backing off %.0fs",
                type(exc).__name__, attempt, retries, delay,
            )
            last_exc = exc
            await asyncio.sleep(delay)
            continue

        # ----- got 200; parse body -----
        choices = envelope.get("choices") or []
        if not choices:
            last_exc = ParseError("envelope had no choices[]")
            await asyncio.sleep(2.0)
            continue
        msg = choices[0].get("message") or {}
        raw_text = msg.get("content") or ""
        usage = envelope.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        model_version = envelope.get("model") or judge_model

        try:
            assessment = parse_judge_response(raw_text, expected_slot_names)
            wall_ms = int((time.monotonic() - t_start) * 1000)
            return {
                "raw_response": raw_text,
                "parsed_assessment": assessment,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model_version": model_version,
                "wall_ms": wall_ms,
                "parse_retry_used": parse_retry_used,
            }
        except ParseError as exc:
            if not parse_retry_used:
                logger.warning(
                    "judge JSON parse failed on attempt %d (%s); retrying once with reminder",
                    attempt, exc,
                )
                parse_retry_used = True
                current_user = user_msg + JSON_RETRY_SUFFIX
                # Don't burn an attempt on parse retry; reset attempt counter
                # by 1 so we still respect the retries budget.
                continue
            logger.error("judge JSON parse failed after retry; treating as failure: %s", exc)
            wall_ms = int((time.monotonic() - t_start) * 1000)
            # Synthesize a graceful failure assessment.
            failure_assessment = {
                "slot_results": [
                    {"slot_name": name, "criterion_met": False, "evidence": "",
                     "issue": f"parse_failure: {exc}"}
                    for name in expected_slot_names
                ],
                "extra_slot_results": [],
                "critical_errors": [{"type": "other", "description": "judge response unparseable"}],
                "unsupported_claims": [],
                "clinically_acceptable": False,
                "reasoning": f"PARSE_FAILURE: {exc}",
                "_parse_failure": True,
            }
            return {
                "raw_response": raw_text,
                "parsed_assessment": failure_assessment,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model_version": model_version,
                "wall_ms": wall_ms,
                "parse_retry_used": parse_retry_used,
            }

    # Exhausted retries on transport / 5xx errors.
    wall_ms = int((time.monotonic() - t_start) * 1000)
    raise JudgeAPIError(
        f"call_judge exhausted {retries} attempts: {type(last_exc).__name__ if last_exc else '?'}: "
        f"{str(last_exc)[:300] if last_exc else ''} (wall_ms={wall_ms})"
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_step_metrics(
    judge_result: Mapping[str, Any],
    gt_step: Mapping[str, Any],
) -> Dict[str, Any]:
    """Aggregate one step's judge output into headline metrics.

    Returns
    -------
    dict
        ``slot_match_rate``, ``n_slots_total``, ``n_slots_met``,
        ``n_critical_errors``, ``n_unsupported_claims``,
        ``clinically_acceptable``, ``critical_slots_met``, ``step_pass``.

    The ``step_pass`` rule combines the per-slot judgement, critical-slot
    requirement, and the overall ``clinically_acceptable`` flag:

        step_pass = clinically_acceptable
                    AND critical_slots_met
                    AND (slot_match_rate >= per_step_pass_threshold_fraction)

    where ``per_step_pass_threshold_fraction`` defaults to
    ``pass_threshold / total_weight`` from the GT, falling back to 0.5.
    """
    slot_results = judge_result.get("slot_results") or []
    n_slots_total = len(slot_results)
    n_slots_met = sum(1 for sr in slot_results if sr.get("criterion_met"))
    slot_match_rate = (n_slots_met / n_slots_total) if n_slots_total else 0.0

    critical = list(gt_step.get("critical_slots") or [])
    slot_lookup = {sr.get("slot_name"): bool(sr.get("criterion_met")) for sr in slot_results}
    critical_slots_met = all(slot_lookup.get(name, False) for name in critical) if critical else True

    # Pass threshold fraction from GT (mirrors v0.3 strict_pass arithmetic).
    total_weight = float(gt_step.get("total_weight") or 0.0)
    pass_threshold = float(gt_step.get("pass_threshold") or 0.0)
    threshold_frac = (pass_threshold / total_weight) if total_weight > 0 else 0.5

    clinically_acceptable = bool(judge_result.get("clinically_acceptable", False))
    n_critical_errors = len(judge_result.get("critical_errors") or [])
    n_unsupported_claims = len(judge_result.get("unsupported_claims") or [])

    # v0.6: required_fact_results coverage
    required_facts = judge_result.get("required_fact_results") or []
    n_required_facts_total = len(required_facts)
    n_required_facts_met = sum(1 for rf in required_facts if rf.get("met"))
    n_missing_required_facts = n_required_facts_total - n_required_facts_met

    step_pass = bool(
        clinically_acceptable
        and critical_slots_met
        and (slot_match_rate + 1e-9 >= threshold_frac)
        and n_missing_required_facts == 0
    )

    return {
        "n_slots_total": n_slots_total,
        "n_slots_met": n_slots_met,
        "slot_match_rate": float(slot_match_rate),
        "critical_slots_met": bool(critical_slots_met),
        "n_critical_errors": n_critical_errors,
        "n_unsupported_claims": n_unsupported_claims,
        "n_required_facts_total": n_required_facts_total,
        "n_required_facts_met": n_required_facts_met,
        "n_missing_required_facts": n_missing_required_facts,
        "clinically_acceptable": clinically_acceptable,
        "step_pass_threshold_fraction": float(threshold_frac),
        "step_pass": step_pass,
    }


def aggregate_item_metrics(step_assessments: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-step assessments into per-item headline metrics.

    Returns
    -------
    dict
        ``item_pass`` (AND across steps), ``n_steps``, ``n_steps_passed``,
        ``mean_slot_match_rate``, ``mean_clinically_acceptable``,
        ``first_failed_step``, ``prefix_fraction_passed``,
        ``total_critical_errors``, ``total_unsupported_claims``.
    """
    n = len(step_assessments)
    if n == 0:
        return {
            "item_pass": False, "n_steps": 0, "n_steps_passed": 0,
            "mean_slot_match_rate": 0.0, "mean_clinically_acceptable": 0.0,
            "first_failed_step": None, "prefix_fraction_passed": 0.0,
            "total_critical_errors": 0, "total_unsupported_claims": 0,
        }

    n_passed = 0
    sum_slot_rate = 0.0
    sum_clin_ok = 0
    total_crit = 0
    total_unsup = 0
    first_failed: Optional[str] = None
    prefix_pass = 0
    prefix_locked = False

    for entry in step_assessments:
        step_pass = bool(entry.get("step_pass"))
        if step_pass:
            n_passed += 1
            if not prefix_locked:
                prefix_pass += 1
        else:
            prefix_locked = True
            if first_failed is None:
                first_failed = entry.get("step_id")
        sum_slot_rate += float(entry.get("slot_match_rate") or 0.0)
        sum_clin_ok += int(bool(entry.get("clinically_acceptable")))
        total_crit += int(entry.get("n_critical_errors") or 0)
        total_unsup += int(entry.get("n_unsupported_claims") or 0)

    return {
        "item_pass": (n_passed == n),
        "n_steps": n,
        "n_steps_passed": n_passed,
        "mean_slot_match_rate": sum_slot_rate / n,
        "mean_clinically_acceptable": sum_clin_ok / n,
        "first_failed_step": first_failed,
        "prefix_fraction_passed": prefix_pass / n,
        "total_critical_errors": total_crit,
        "total_unsupported_claims": total_unsup,
    }


# ---------------------------------------------------------------------------
# Audit-trail persistence
# ---------------------------------------------------------------------------

def save_call_record(
    out_dir: Path,
    item_id: str,
    step_id: str,
    *,
    system_msg: str,
    user_msg: str,
    raw_response: str,
    parsed_assessment: Mapping[str, Any],
    judge_model: str,
    model_version: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    wall_ms: int,
    was_truncated: bool,
    parse_retry_used: bool,
    timestamp_iso: str,
) -> Path:
    """Persist one judge call as a complete-audit JSON file."""
    calls_dir = out_dir / "judge_calls"
    calls_dir.mkdir(parents=True, exist_ok=True)
    path = calls_dir / f"{item_id}__{step_id}.json"
    record = {
        "item_id": item_id,
        "step_id": step_id,
        "judge_model": judge_model,
        "model_version": model_version,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "raw_response": raw_response,
        "parsed_assessment": dict(parsed_assessment),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": float(cost_usd),
        "wall_ms": int(wall_ms),
        "model_output_was_truncated": bool(was_truncated),
        "parse_retry_used": bool(parse_retry_used),
        "timestamp": timestamp_iso,
        "scoring_spec_version": SCORING_SPEC_VERSION,
    }
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(record, indent=2))
    tmp_path.replace(path)
    return path


def load_call_record(out_dir: Path, item_id: str, step_id: str) -> Optional[Dict[str, Any]]:
    path = out_dir / "judge_calls" / f"{item_id}__{step_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Layout discovery
# ---------------------------------------------------------------------------

@dataclass
class StepWorkUnit:
    """One unit of work for the async judge worker pool."""
    item_id: str
    step_id: str
    template: str
    question: str
    gt_step: Mapping[str, Any]
    model_output: str
    expected_slot_names: List[str]


@dataclass
class _FileTreeIndex:
    """One-pass directory scan with root-level and nested file maps."""

    root_files: Dict[str, Path] = field(default_factory=dict)
    nested_files: Dict[str, Path] = field(default_factory=dict)
    all_paths: List[Path] = field(default_factory=list)
    all_path_set: set[str] = field(default_factory=set)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _scan_tree_once(root: Path) -> _FileTreeIndex:
    """Recursively walk a tree once and memoize root vs nested files."""
    index = _FileTreeIndex()
    stack: List[Tuple[Path, bool]] = [(root, True)]

    while stack:
        current_dir, is_root = stack.pop()
        try:
            with os.scandir(current_dir) as it:
                entries = sorted(it, key=lambda entry: entry.name)
        except OSError:
            continue

        child_dirs: List[Path] = []
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    child_dirs.append(Path(entry.path))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
            except OSError:
                continue

            path = Path(entry.path)
            index.all_paths.append(path)
            index.all_path_set.add(path.as_posix())
            target = index.root_files if is_root else index.nested_files
            target.setdefault(entry.name, path)

        for child_dir in reversed(child_dirs):
            stack.append((child_dir, False))

    return index


def _find_question_file(
    questions_root: Path,
    item_id: str,
    *,
    index: Optional[_FileTreeIndex] = None,
) -> Optional[Path]:
    if index is None:
        candidates = [
            questions_root / f"{item_id}.question.json",
            questions_root / f"{item_id}.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        # Fallback: nested template subdir search.
        for c in questions_root.glob(f"**/{item_id}.question.json"):
            return c
        return None

    for name in (f"{item_id}.question.json", f"{item_id}.json"):
        path = index.root_files.get(name)
        if path is not None:
            return path
    return index.nested_files.get(f"{item_id}.question.json")


def _find_gt_file(
    gt_root: Path,
    item_id: str,
    *,
    index: Optional[_FileTreeIndex] = None,
) -> Optional[Path]:
    if index is None:
        candidates = [
            gt_root / f"{item_id}.gt.json",
            gt_root / f"{item_id}.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        for c in gt_root.glob(f"**/{item_id}.gt.json"):
            return c
        return None

    for name in (f"{item_id}.gt.json", f"{item_id}.json"):
        path = index.root_files.get(name)
        if path is not None:
            return path
    return index.nested_files.get(f"{item_id}.gt.json")


def _find_output_file(
    outputs_root: Path,
    template: Optional[str],
    item_id: str,
    *,
    index: Optional[_FileTreeIndex] = None,
) -> Optional[Path]:
    if index is None:
        candidates: List[Path] = []
        if template:
            candidates.append(outputs_root / template / f"{item_id}.output.json")
        candidates.append(outputs_root / f"{item_id}.output.json")
        for c in candidates:
            if c.exists():
                return c
        for c in outputs_root.glob(f"**/{item_id}.output.json"):
            return c
        return None

    if template:
        template_name = f"{item_id}.output.json"
        template_path = outputs_root / template / template_name
        if template_path.as_posix() in index.all_path_set:
            return template_path
    root_path = index.root_files.get(f"{item_id}.output.json")
    if root_path is not None:
        return root_path
    return index.nested_files.get(f"{item_id}.output.json")


def _index_chain_questions(question_json: Mapping[str, Any]) -> Dict[str, str]:
    """Map step_id -> question_text. Handles chain (chain_questions[]) and single-step (rendered_question)."""
    mti = question_json.get("model_text_input") or {}
    chain = mti.get("chain_questions") or []
    if chain:
        return {str(c.get("step_id")): str(c.get("question") or "") for c in chain if c.get("step_id")}
    rq = mti.get("rendered_question") or ""
    if rq:
        return {"step_1": str(rq)}
    return {}


def _strip_inline_thinking(text: str) -> str:
    """Strip pre-`</think>` chain-of-thought from inline-reasoning model outputs.

    Open-source thinking models (e.g., Qwen3.5-27B with thinking mode ON)
    emit reasoning text directly in raw_output, terminated by a literal
    `</think>` tag, followed by the final answer. API thinking models
    (GPT-5.x, Gemini 2.5+) hide CoT server-side and return only the final
    answer. To make these comparable for the judge, we strip everything
    up to and including the first `</think>`.

    Returns the original text if `</think>` is not present.
    """
    if not text:
        return text
    idx = text.find("</think>")
    if idx < 0:
        return text
    after = text[idx + len("</think>") :]
    return after.lstrip("\r\n").lstrip()


def _index_step_outputs(output_json: Mapping[str, Any]) -> Dict[str, str]:
    """Map step_id -> raw_output text for a chain output file.

    Inline CoT (`</think>`-tagged) is stripped so the judge sees only the
    final answer, matching API thinking models' deployment surface.
    """
    out: Dict[str, str] = {}
    for entry in output_json.get("steps") or []:
        sid = str(entry.get("step_id") or "")
        if not sid:
            continue
        text = (
            entry.get("raw_output")
            or entry.get("answer")
            or entry.get("answer_text")
            or entry.get("response")
            or ""
        )
        out[sid] = _strip_inline_thinking(str(text or ""))
    return out


def discover_work_units(
    *,
    scored_jsonl: Optional[Path],
    outputs_dir: Optional[Path],
    gt_root: Path,
    questions_root: Path,
    limit: Optional[int],
    logger: logging.Logger,
) -> List[StepWorkUnit]:
    """Build the ordered list of (item_id, step_id) work units to judge."""
    if scored_jsonl is None and outputs_dir is None:
        raise SystemExit("Must pass either --scored-jsonl or --outputs-dir")

    t_discovery_start = time.monotonic()

    # Locate inference outputs root if scored_jsonl path was used.
    if outputs_dir is None:
        outputs_dir = scored_jsonl.parent / "outputs"
        if not outputs_dir.exists():
            outputs_dir = scored_jsonl.parent

    gt_index_start = time.monotonic()
    gt_index = _scan_tree_once(gt_root)
    gt_index_seconds = time.monotonic() - gt_index_start

    questions_index_start = time.monotonic()
    questions_index = _scan_tree_once(questions_root)
    questions_index_seconds = time.monotonic() - questions_index_start

    outputs_index_start = time.monotonic()
    outputs_index = _scan_tree_once(outputs_dir)
    outputs_index_seconds = time.monotonic() - outputs_index_start

    item_ids: List[Tuple[str, Optional[str]]] = []  # (item_id, template)
    if scored_jsonl is not None:
        records = _load_jsonl(scored_jsonl)
        seen = set()
        for rec in records:
            iid = rec.get("item_id")
            if not iid or iid in seen:
                continue
            seen.add(iid)
            item_ids.append((str(iid), rec.get("template") or rec.get("template_id")))
    else:
        for f in sorted(
            (path for path in outputs_index.all_paths if path.name.endswith(".output.json")),
            key=lambda path: path.as_posix(),
        ):
            iid = f.stem.replace(".output", "")
            item_ids.append((iid, f.parent.name))

    if limit is not None and limit > 0:
        item_ids = item_ids[:limit]
    logger.info("discovered %d items to judge", len(item_ids))

    gt_lookup_seconds = 0.0
    question_lookup_seconds = 0.0
    output_lookup_seconds = 0.0
    units: List[StepWorkUnit] = []
    for item_id, template in item_ids:
        t0 = time.monotonic()
        gt_path = _find_gt_file(gt_root, item_id, index=gt_index)
        gt_lookup_seconds += time.monotonic() - t0
        if gt_path is None:
            logger.warning("GT file not found for %s under %s; skipping", item_id, gt_root)
            continue
        try:
            gt = json.loads(gt_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("failed to read GT %s: %s", gt_path, exc)
            continue

        t0 = time.monotonic()
        q_path = _find_question_file(questions_root, item_id, index=questions_index)
        question_lookup_seconds += time.monotonic() - t0
        question_index: Dict[str, str] = {}
        if q_path is not None:
            try:
                question_index = _index_chain_questions(json.loads(q_path.read_text()))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("failed to read questions %s: %s", q_path, exc)

        t0 = time.monotonic()
        out_path = _find_output_file(outputs_dir, template, item_id, index=outputs_index)
        output_lookup_seconds += time.monotonic() - t0
        output_index: Dict[str, str] = {}
        if out_path is not None:
            try:
                output_index = _index_step_outputs(json.loads(out_path.read_text()))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("failed to read output %s: %s", out_path, exc)
        else:
            logger.warning("output file not found for %s; will judge against empty answer", item_id)

        steps_gt = list(gt.get("step_ground_truth") or [])
        if not steps_gt:
            # Single-step (non-chain) item: synthesize one work unit
            slots = list(gt.get("rubric_slots") or [])
            slot_names = [str(s.get("slot_name")) for s in slots if s.get("slot_name")]
            answer = output_index.get("step_1") or output_index.get("step_0") or ""
            if not answer and out_path is not None:
                # Fall back: open output and try to find a top-level answer field.
                try:
                    obj = json.loads(out_path.read_text())
                    answer = (
                        obj.get("raw_output") or obj.get("answer")
                        or obj.get("answer_text") or ""
                    )
                    answer = _strip_inline_thinking(answer)
                except (OSError, json.JSONDecodeError):
                    answer = ""
            units.append(StepWorkUnit(
                item_id=item_id,
                step_id="step_1",
                template=template or gt.get("template_id") or "",
                question=question_index.get("step_1") or gt.get("question") or "",
                gt_step=gt,
                model_output=answer,
                expected_slot_names=slot_names,
            ))
            continue

        for step_gt in steps_gt:
            sid = str(step_gt.get("step_id") or "")
            if not sid:
                continue
            slot_names = [
                str(s.get("slot_name"))
                for s in step_gt.get("rubric_slots") or []
                if s.get("slot_name")
            ]
            units.append(StepWorkUnit(
                item_id=item_id,
                step_id=sid,
                template=template or gt.get("template_id") or "",
                question=question_index.get(sid, ""),
                gt_step=step_gt,
                model_output=output_index.get(sid, ""),
                expected_slot_names=slot_names,
            ))

    logger.info(
        "discovery timing: total=%.3fs index_gt=%.3fs index_questions=%.3fs "
        "index_outputs=%.3fs gt_lookup=%.3fs question_lookup=%.3fs output_lookup=%.3fs",
        time.monotonic() - t_discovery_start,
        gt_index_seconds,
        questions_index_seconds,
        outputs_index_seconds,
        gt_lookup_seconds,
        question_lookup_seconds,
        output_lookup_seconds,
    )
    logger.info("expanded into %d step-level work units", len(units))
    return units


# ---------------------------------------------------------------------------
# Main async pipeline
# ---------------------------------------------------------------------------

async def _process_unit(
    unit: StepWorkUnit,
    *,
    client: httpx.AsyncClient,
    api_key: str,
    judge_model: str,
    out_dir: Path,
    skip_existing: bool,
    cost: CostAccumulator,
    semaphore: asyncio.Semaphore,
    verbose: bool,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Run one work unit (one step), returning the per-step assessment row."""
    if skip_existing:
        cached = load_call_record(out_dir, unit.item_id, unit.step_id)
        if cached is not None:
            assessment = cached.get("parsed_assessment") or {}
            metrics = aggregate_step_metrics(assessment, unit.gt_step)
            metrics.update({
                "item_id": unit.item_id, "step_id": unit.step_id,
                "template": unit.template, "from_cache": True,
            })
            if verbose:
                logger.info(
                    "[cache] %s/%s pass=%s slot_rate=%.2f",
                    unit.item_id, unit.step_id, metrics["step_pass"], metrics["slot_match_rate"],
                )
            return metrics

    system_msg, user_msg, was_truncated = build_prompt(
        question=unit.question, gt_step=unit.gt_step, model_output=unit.model_output,
    )
    if was_truncated:
        logger.warning(
            "truncated model_output for %s/%s (>%d chars)",
            unit.item_id, unit.step_id, MODEL_OUTPUT_CHAR_CAP,
        )

    async with semaphore:
        try:
            result = await call_judge(
                client=client,
                api_key=api_key,
                judge_model=judge_model,
                system_msg=system_msg,
                user_msg=user_msg,
                expected_slot_names=unit.expected_slot_names,
                logger=logger,
            )
        except JudgeAPIError as exc:
            logger.error("judge API failed for %s/%s: %s", unit.item_id, unit.step_id, exc)
            cost.record_failure()
            assessment = {
                "slot_results": [
                    {"slot_name": name, "criterion_met": False, "evidence": "",
                     "issue": f"api_failure: {exc}"}
                    for name in unit.expected_slot_names
                ],
                "extra_slot_results": [],
                "critical_errors": [{"type": "other", "description": "judge API failure"}],
                "unsupported_claims": [],
                "clinically_acceptable": False,
                "reasoning": f"API_FAILURE: {exc}",
                "_api_failure": True,
            }
            metrics = aggregate_step_metrics(assessment, unit.gt_step)
            metrics.update({
                "item_id": unit.item_id, "step_id": unit.step_id,
                "template": unit.template, "from_cache": False,
                "api_failed": True,
            })
            return metrics

    cost_usd = cost.record(result["prompt_tokens"], result["completion_tokens"])
    timestamp = datetime.now(timezone.utc).isoformat()
    save_call_record(
        out_dir=out_dir,
        item_id=unit.item_id, step_id=unit.step_id,
        system_msg=system_msg, user_msg=user_msg,
        raw_response=result["raw_response"],
        parsed_assessment=result["parsed_assessment"],
        judge_model=judge_model, model_version=result["model_version"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        cost_usd=cost_usd, wall_ms=result["wall_ms"],
        was_truncated=was_truncated,
        parse_retry_used=result["parse_retry_used"],
        timestamp_iso=timestamp,
    )

    metrics = aggregate_step_metrics(result["parsed_assessment"], unit.gt_step)
    metrics.update({
        "item_id": unit.item_id, "step_id": unit.step_id,
        "template": unit.template, "from_cache": False,
    })

    if verbose:
        logger.info(
            "[done] %s/%s pass=%s slot_rate=%.2f clin_ok=%s tokens=%d/%d $%.4f wall=%dms",
            unit.item_id, unit.step_id, metrics["step_pass"], metrics["slot_match_rate"],
            metrics["clinically_acceptable"],
            result["prompt_tokens"], result["completion_tokens"],
            cost_usd, result["wall_ms"],
        )

    if cost.total_calls % 100 == 0 and cost.total_calls > 0:
        logger.info(
            "[progress] %d calls, $%.4f total cost, %d input toks, %d output toks, %d failures",
            cost.total_calls, cost.total_cost_usd,
            cost.total_input_tokens, cost.total_output_tokens, cost.n_failures,
        )

    return metrics


async def _amain(args: argparse.Namespace, logger: logging.Logger) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    units = discover_work_units(
        scored_jsonl=Path(args.scored_jsonl) if args.scored_jsonl else None,
        outputs_dir=Path(args.outputs_dir) if args.outputs_dir else None,
        gt_root=Path(args.gt_root),
        questions_root=Path(args.questions_root),
        limit=args.limit,
        logger=logger,
    )
    if not units:
        logger.error("no work units discovered; aborting")
        return 2

    # Always emit prompt template + manifest skeleton up-front, even on dry-run.
    (out_dir / "prompt_template.md").write_text(_render_prompt_template_doc())

    if args.dry_run:
        logger.info("--dry-run: printing first 3 prompts then exiting")
        for unit in units[:3]:
            sys_msg, user_msg, was_trunc = build_prompt(
                unit.question, unit.gt_step, unit.model_output,
            )
            print("=" * 80)
            print(f"ITEM={unit.item_id}  STEP={unit.step_id}  truncated={was_trunc}")
            print(f"expected_slot_names={unit.expected_slot_names}")
            print("--- SYSTEM ---")
            print(sys_msg)
            print("--- USER ---")
            print(user_msg)
        return 0

    if httpx is None:
        raise SystemExit(
            "llm_judge requires httpx for non-dry-run mode. "
            f"Install via `pip install httpx` (import error: {_HTTPX_IMPORT_ERROR})"
        )
    api_key = os.environ.get("OPENAI_API_KEY") or ""
    if not api_key:
        raise SystemExit("OPENAI_API_KEY env var is required (unless --dry-run)")

    cost = CostAccumulator(judge_model=args.judge_model)
    semaphore = asyncio.Semaphore(max(1, args.max_parallel))

    run_started = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        tasks = [
            asyncio.create_task(_process_unit(
                unit,
                client=client, api_key=api_key,
                judge_model=args.judge_model, out_dir=out_dir,
                skip_existing=args.skip_existing,
                cost=cost, semaphore=semaphore,
                verbose=args.verbose, logger=logger,
            ))
            for unit in units
        ]
        step_rows: List[Dict[str, Any]] = []
        for fut in asyncio.as_completed(tasks):
            row = await fut
            step_rows.append(row)

    elapsed = time.monotonic() - t0

    # Group by item_id (preserve original item discovery order).
    item_order: List[str] = []
    seen_items = set()
    for u in units:
        if u.item_id not in seen_items:
            seen_items.add(u.item_id)
            item_order.append(u.item_id)
    by_item: Dict[str, List[Dict[str, Any]]] = {iid: [] for iid in item_order}
    for row in step_rows:
        by_item.setdefault(row["item_id"], []).append(row)
    # Within each item, preserve step order from the discovery units list.
    step_order_per_item: Dict[str, List[str]] = {}
    for u in units:
        step_order_per_item.setdefault(u.item_id, []).append(u.step_id)
    for iid, rows in by_item.items():
        ordering = {sid: idx for idx, sid in enumerate(step_order_per_item.get(iid, []))}
        rows.sort(key=lambda r: ordering.get(r["step_id"], 1_000_000))

    # --- write per_step_assessments.jsonl ----------------------------------
    per_step_path = out_dir / "per_step_assessments.jsonl"
    with per_step_path.open("w") as f:
        for iid in item_order:
            for row in by_item.get(iid, []):
                f.write(json.dumps(row) + "\n")

    # --- write per_item_assessments.jsonl ----------------------------------
    per_item_path = out_dir / "per_item_assessments.jsonl"
    item_records: List[Dict[str, Any]] = []
    for iid in item_order:
        rows = by_item.get(iid, [])
        agg = aggregate_item_metrics(rows)
        agg["item_id"] = iid
        agg["template"] = rows[0].get("template") if rows else ""
        agg["scoring_spec_version"] = SCORING_SPEC_VERSION
        agg["judge_model"] = args.judge_model
        item_records.append(agg)
    with per_item_path.open("w") as f:
        for rec in item_records:
            f.write(json.dumps(rec) + "\n")

    # --- write manifest.json ----------------------------------------------
    manifest = {
        "scoring_spec_version": SCORING_SPEC_VERSION,
        "judge_model": args.judge_model,
        "run_started_utc": run_started,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(elapsed, 2),
        "scored_jsonl": str(args.scored_jsonl) if args.scored_jsonl else None,
        "outputs_dir": str(args.outputs_dir) if args.outputs_dir else None,
        "gt_root": str(args.gt_root),
        "questions_root": str(args.questions_root),
        "out_dir": str(out_dir),
        "max_parallel": args.max_parallel,
        "skip_existing": bool(args.skip_existing),
        "limit": args.limit,
        "n_items_total": len(item_order),
        "n_step_units": len(units),
        "n_steps_passed": sum(1 for r in step_rows if r.get("step_pass")),
        "n_items_passed": sum(1 for r in item_records if r.get("item_pass")),
        **cost.to_dict(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # --- write README.md ---------------------------------------------------
    (out_dir / "README.md").write_text(_render_run_readme(manifest))

    # ---- final summary ----
    logger.info("=" * 60)
    logger.info("LLM-judge run finished")
    logger.info("  items judged:        %d", len(item_order))
    logger.info("  step calls:          %d (failures: %d)", cost.total_calls, cost.n_failures)
    logger.info("  items passed:        %d / %d (%.1f%%)",
                manifest["n_items_passed"], len(item_order),
                100.0 * manifest["n_items_passed"] / max(1, len(item_order)))
    logger.info("  steps passed:        %d / %d (%.1f%%)",
                manifest["n_steps_passed"], len(step_rows),
                100.0 * manifest["n_steps_passed"] / max(1, len(step_rows)))
    logger.info("  total cost:          $%.4f (%d in, %d out tokens)",
                cost.total_cost_usd, cost.total_input_tokens, cost.total_output_tokens)
    logger.info("  wall:                %.1fs", elapsed)
    logger.info("  outputs:             %s", out_dir)
    return 0


# ---------------------------------------------------------------------------
# README / prompt-template renderers (kept here so the run is self-documenting)
# ---------------------------------------------------------------------------

def _render_prompt_template_doc() -> str:
    return (
        "# LLM-judge prompt template (v0.5)\n\n"
        "## SYSTEM\n```\n" + JUDGE_SYSTEM_PROMPT + "\n```\n\n"
        "## USER (template; `{...}` are filled per step)\n```\n"
        + JUDGE_USER_TEMPLATE + "\n```\n\n"
        "Decoding: `temperature=0`, `top_p=1`, `max_tokens=600`, "
        "`response_format={\"type\":\"json_object\"}`.\n"
    )


def _render_run_readme(manifest: Mapping[str, Any]) -> str:
    return (
        "# LLM-judge run\n\n"
        f"- spec: v{manifest['scoring_spec_version']}\n"
        f"- judge_model: `{manifest['judge_model']}`\n"
        f"- started: {manifest['run_started_utc']}\n"
        f"- finished: {manifest['run_finished_utc']}\n"
        f"- wall: {manifest['wall_seconds']}s\n"
        f"- items: {manifest['n_items_total']} (passed: {manifest['n_items_passed']})\n"
        f"- step calls: {manifest['total_calls']} (failures: {manifest['n_failures']})\n"
        f"- cost: ${manifest['total_cost_usd']:.4f} "
        f"({manifest['total_input_tokens']} in, {manifest['total_output_tokens']} out tokens)\n\n"
        "## Files\n"
        "- `manifest.json` -- run metadata, token totals, cost.\n"
        "- `prompt_template.md` -- verbatim system + user template used.\n"
        "- `judge_calls/<item>__<step>.json` -- one per call: messages, raw response, parsed assessment, tokens, cost, latency.\n"
        "- `per_step_assessments.jsonl` -- one row per step: slot_match_rate, clinically_acceptable, step_pass, etc.\n"
        "- `per_item_assessments.jsonl` -- one row per item: item_pass (AND across steps), aggregate stats.\n\n"
        "## Replay\n"
        "Use `replay.py` to re-print a specific call's prompt + parsed assessment:\n"
        "```\npython replay.py --call judge_calls/<item>__<step>.json\n```\n\n"
        "## Re-aggregate\n"
        "If you change the per-step pass threshold logic, regenerate per-item rows from the saved per-step:\n"
        "```\npython aggregate.py --in per_step_assessments.jsonl --out per_item_assessments.jsonl\n```\n"
    )


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BrainTRACE v0.5 LLM-as-judge scorer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--scored-jsonl", default=None,
                   help="Path to scored_v0_3.jsonl (chain layout). "
                        "Mutually exclusive-ish with --outputs-dir; pick one.")
    p.add_argument("--outputs-dir", default=None,
                   help="Path to a directory of <item>.output.json files (non-chain layout).")
    p.add_argument("--gt-root", required=True,
                   help="Path to GT root (e.g. benchmark/gt/case_reasoning).")
    p.add_argument("--questions-root", required=True,
                   help="Path to questions root (e.g. benchmark/questions/case_reasoning).")
    p.add_argument("--out-dir", required=True,
                   help="Output directory; will be created if absent.")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                   help=f"Judge model id (default: {DEFAULT_JUDGE_MODEL}).")
    p.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL,
                   help=f"Max concurrent API calls (default: {DEFAULT_MAX_PARALLEL}).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip steps whose judge_calls/<item>__<step>.json already exists.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap to first N items (for smoke testing).")
    p.add_argument("--dry-run", action="store_true",
                   help="Build prompts and print first 3 (system+user) without calling the API.")
    p.add_argument("--verbose", action="store_true",
                   help="Log each call's parsed result one-line summary.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    logger = _setup_logger(args.verbose)
    if args.scored_jsonl and args.outputs_dir:
        logger.warning(
            "both --scored-jsonl and --outputs-dir given; "
            "--scored-jsonl filters items while --outputs-dir locates output files"
        )
    return asyncio.run(_amain(args, logger))


if __name__ == "__main__":
    sys.exit(main())
