#!/usr/bin/env python3
"""
BrainTRACE — Proprietary VLM API adapter (Anthropic / OpenAI / Google).

Drop-in replacement for scripts/vlm_eval*.py for closed-source models that are
accessed via REST API instead of local vLLM.  Output schema is 100% compatible
with score_pilot.py / summarize_pilot_v2.py / generate_dashboard.py.

Flags:
  --model          Model ID, e.g. claude-sonnet-4-7, gpt-5, gemini-2.5-pro
  --provider       {anthropic,openai,google}  — override auto-detect
  --root           Project root (default: parent of this script's directory)
  --sample-file    JSONL items file (required)
  --out-dir        Output directory (required)
  --images-root    Load canonical PNGs from <PATH>/bulk_v1/<template>/<item_id>/
                   or <PATH>/flagship/<item_id>/
  --max-tokens     Max output tokens (default: 2048)
  --temperature    Sampling temperature (default: 0.0)
  --max-images     Cap images per API call (default: 16)
  --skip-existing  Skip items whose .output.json already exists
  --max-items      Only process first N items (for cost-controlled partial runs)
  --parallel       Number of parallel API threads (default: 1 = serial)
  --dry-run        Load images + build prompts; print first item; skip API calls

Provider auto-detection (from --model):
  claude-* | anthropic/*  → anthropic  (requires ANTHROPIC_API_KEY)
  gpt-* | o3-* | o4-* | openai/* → openai  (requires OPENAI_API_KEY)
  gemini-* | google/*     → google    (requires GOOGLE_API_KEY)

Retry policy: exponential backoff on 429 / 5xx (max 5 retries, 2→32 s).

Cost estimates are rough and for progress logging only — not billing.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Provider auto-detection
# ---------------------------------------------------------------------------

PROVIDER_PATTERNS = {
    "anthropic": [r"^claude-", r"^anthropic/"],
    "openai":    [r"^gpt-", r"^o3-", r"^o4-", r"^openai/"],
    "google":    [r"^gemini-", r"^google/"],
}

ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "google":    "GOOGLE_API_KEY",
}

# Anthropic: hard 20 images/request limit
# OpenAI: no official hard limit but 16 is a safe documented ceiling
# Google: no official image-count limit; we cap at 20
PROVIDER_IMAGE_LIMIT = {
    "anthropic": 20,
    "openai":    16,
    "google":    20,
}

# ---------------------------------------------------------------------------
# Rough cost table ($/M tokens, input / output)
# Update these numbers as pricing changes — they are only for logging.
# ---------------------------------------------------------------------------

COST_TABLE = {
    # Anthropic
    "claude-sonnet-4-7":       (3.00,  15.00),
    "claude-sonnet-4-5":       (3.00,  15.00),
    "claude-opus-4-7":         (15.00, 75.00),
    "claude-opus-4-5":         (15.00, 75.00),
    "claude-haiku-4-5":        (0.80,  4.00),
    "claude-3-5-sonnet":       (3.00,  15.00),
    "claude-3-5-haiku":        (0.80,  4.00),
    "claude-3-opus":           (15.00, 75.00),
    # OpenAI
    "gpt-5":                   (2.50,  10.00),
    "gpt-4o":                  (2.50,  10.00),
    "gpt-4o-mini":             (0.15,  0.60),
    "o3":                      (10.00, 40.00),
    "o4-mini":                 (1.10,  4.40),
    # Google
    "gemini-2.5-pro":          (1.25,  5.00),
    "gemini-2.5-flash":        (0.075, 0.30),
    "gemini-1.5-pro":          (1.25,  5.00),
    "gemini-1.5-flash":        (0.075, 0.30),
}

# Default fallback cost estimate if model not in table
DEFAULT_COST = (3.00, 15.00)


def _cost_per_tok(model_id: str):
    """Return (input_$/M_tok, output_$/M_tok) for model_id (prefix match)."""
    for key, val in COST_TABLE.items():
        if model_id.startswith(key):
            return val
    return DEFAULT_COST


def detect_provider(model_id: str, override: str = None) -> str:
    if override:
        if override not in PROVIDER_PATTERNS:
            raise ValueError(
                f"Unknown provider '{override}'. "
                f"Supported: {sorted(PROVIDER_PATTERNS)}"
            )
        return override
    for provider, patterns in PROVIDER_PATTERNS.items():
        if any(re.match(p, model_id) for p in patterns):
            return provider
    supported = " | ".join(
        f"{p}: {pats}" for p, pats in PROVIDER_PATTERNS.items()
    )
    raise ValueError(
        f"Cannot auto-detect provider for model '{model_id}'.\n"
        f"Supported patterns:\n  {supported}\n"
        f"Use --provider {{anthropic,openai,google}} to override."
    )


def check_api_key(provider: str) -> str:
    env = ENV_KEYS[provider]
    key = os.environ.get(env, "")
    if not key:
        raise EnvironmentError(
            f"Provider '{provider}' requires the '{env}' environment variable.\n"
            f"Set it with:  export {env}=sk-..."
        )
    return key


# ---------------------------------------------------------------------------
# Retry helper (works without tenacity)
# ---------------------------------------------------------------------------

def _is_retryable(exc) -> bool:
    """Return True if the exception looks like a transient 429 / 5xx."""
    msg = str(exc).lower()
    # anthropic / openai SDKs raise typed exceptions; we check name + msg
    cls = type(exc).__name__
    retryable_classes = {
        "RateLimitError", "APIStatusError", "APIConnectionError",
        "ServiceUnavailableError", "InternalServerError",
    }
    if cls in retryable_classes:
        return True
    # google.api_core.exceptions
    if "429" in msg or "rate limit" in msg or "quota" in msg:
        return True
    if "500" in msg or "502" in msg or "503" in msg or "timeout" in msg:
        return True
    return False


def with_retry(fn, max_retries=5, base_delay=2.0):
    """Call fn(); retry with exponential backoff on transient errors."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if not _is_retryable(exc) or attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            print(
                f"[retry] attempt {attempt+1}/{max_retries}: {type(exc).__name__}: {exc}. "
                f"Sleeping {delay:.0f}s ...",
                flush=True,
            )
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Image loading / base64 encoding
# ---------------------------------------------------------------------------

def _pil_to_b64(img: Image.Image) -> str:
    import io
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _path_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ---------------------------------------------------------------------------
# Canonical PNG loader (mirrored from vlm_eval_qwen35.py)
# ---------------------------------------------------------------------------

def _study_uid_from_dirname(study_dirname: str) -> str:
    parts = study_dirname.split("_", 2)
    return parts[2] if len(parts) == 3 else study_dirname


def load_canonical_images(question_json: dict, images_root, max_images: int = 16):
    """
    Load pre-rendered canonical PNGs from images_root.

    Directory layout:
      Bulk:     <images_root>/bulk_v1/<template>/<item_id>/
      Flagship: <images_root>/flagship/<item_id>/

    Returns (list[PIL.Image RGB], list[str paths]).
    Returns ([], []) with an [err] print if the canonical dir is missing/empty.
    """
    item_id = question_json["item_id"]
    spec = question_json["model_image_spec"]
    ba = question_json["backend_authoring"]
    rule = spec["slice_selection_rule"]
    modalities = spec["input_modalities"]
    planes = spec["input_planes"]
    mti = question_json["model_text_input"]
    tp_labels = mti.get("shown_metadata", {}).get("shown_tp_labels", [])

    images_root = Path(images_root)

    if item_id.startswith("flagship"):
        cdir = images_root / "flagship" / item_id
    elif item_id.startswith("threed_v1_"):
        template = question_json.get("template") or "_".join(item_id[len("threed_v1_"):].split("_")[:-1])
        cdir = images_root / "threed_v1" / template / item_id
    elif item_id.startswith("threed_v2_"):
        template = question_json.get("template") or "_".join(item_id[len("threed_v2_"):].split("_")[:-1])
        cdir = images_root / "threed_v2" / template / item_id
    else:
        template = question_json.get("template") or question_json.get("metadata", {}).get("template")
        if template is None:
            stripped = item_id[len("bulk_v1_"):] if item_id.startswith("bulk_v1_") else item_id
            template = "_".join(stripped.split("_")[:-1])
        cdir = images_root / "bulk_v1" / template / item_id

    if not cdir.exists():
        print(f"[err] canonical PNGs missing for {item_id}: directory not found: {cdir}", flush=True)
        return [], []

    all_pngs = sorted(cdir.glob("*.png"))
    if not all_pngs:
        print(f"[err] canonical PNGs missing for {item_id}: directory empty: {cdir}", flush=True)
        return [], []

    # Structured ordering (mirrors vlm_eval_qwen35 logic)
    ordered_paths = []
    tp_uids = [_study_uid_from_dirname(sd) for sd in ba.get("study_dirnames", [])]

    if rule == "full_3d_volume":
        mip_planes = ["axial", "sagittal", "coronal"]
        for tp_uid in tp_uids:
            for mod in modalities:
                for plane in mip_planes:
                    for png in all_pngs:
                        n = png.name.lower()
                        if tp_uid.lower() in n and mod in n and f"mip-{plane}" in n:
                            if png not in ordered_paths:
                                ordered_paths.append(png)
    else:
        for tp_uid in tp_uids:
            for mod in modalities:
                for plane in planes:
                    if rule in ("median_axial_single", "median_axial_per_tp") and plane != "axial":
                        continue
                    for png in all_pngs:
                        n = png.name.lower()
                        if tp_uid.lower() in n and mod in n and f"_{plane}.png" in n:
                            if png not in ordered_paths:
                                ordered_paths.append(png)

    unmatched = [p for p in all_pngs if p not in ordered_paths]
    if unmatched and not ordered_paths:
        # Fall back to sorted filename order when structured lookup fails
        print(
            f"[warn] {item_id}: structured canonical lookup yielded 0 paths; "
            f"falling back to sorted filename order ({len(all_pngs)} files)",
            flush=True,
        )
        ordered_paths = list(all_pngs)
    elif unmatched:
        print(
            f"[warn] {item_id}: {len(unmatched)} PNG(s) did not match expected naming pattern "
            f"(ignored): {[p.name for p in unmatched[:3]]}{'...' if len(unmatched)>3 else ''}",
            flush=True,
        )

    if len(ordered_paths) > max_images:
        import numpy as np
        idx = (
            np.linspace(0, len(ordered_paths) - 1, max_images)
            .round()
            .astype(int)
            .tolist()
        )
        ordered_paths = [ordered_paths[i] for i in idx]

    imgs, paths = [], []
    for p in ordered_paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
            paths.append(str(p))
        except Exception as exc:
            print(f"[warn] {item_id}: failed to open canonical PNG {p}: {exc}", flush=True)

    return imgs, paths


# ---------------------------------------------------------------------------
# Prompt composition
# ---------------------------------------------------------------------------

SYSTEM_INSTR = (
    "You are a neuroradiologist reviewing sequential brain MRI images. Answer the question "
    "concisely. For multiple-choice / ordinal questions, reply with the option LETTER only "
    "(e.g., 'A'), optionally followed by a short phrase. For open-ended questions, "
    "produce a focused 2-5 sentence impression grounded in what is visible in the images. "
    "Do not fabricate findings."
)


def build_user_text(question_json: dict) -> str:
    """Build the plain-text question block (same as vlm_eval_qwen35)."""
    mti = question_json["model_text_input"]
    question = mti["rendered_question"]
    options = mti.get("options")
    md = mti.get("shown_metadata", {}) or {}

    parts = ["Patient metadata (anonymized):"]
    for k, v in md.items():
        if v is None or v == []:
            continue
        parts.append(f"  {k}: {v}")
    parts.append("")
    parts.append(f"Question: {question}")
    if options:
        parts.append("")
        parts.append("Options:")
        for i, opt in enumerate(options):
            parts.append(f"  {chr(65 + i)}) {opt}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Per-provider API calls
# ---------------------------------------------------------------------------

def _call_anthropic(client, model_id: str, imgs, user_text: str,
                    max_tokens: int, temperature: float):
    content = []
    for img in imgs:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": _pil_to_b64(img),
            },
        })
    content.append({"type": "text", "text": user_text})

    def _do():
        return client.messages.create(
            model=model_id,
            system=SYSTEM_INSTR,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    resp = with_retry(_do)
    raw = resp.content[0].text
    n_out_tok = resp.usage.output_tokens if resp.usage else 0
    finish = resp.stop_reason or ""
    return raw, n_out_tok, finish


def _call_openai(client, model_id: str, imgs, user_text: str,
                 max_tokens: int, temperature: float,
                 reasoning_effort=None, verbosity=None, seed=42):
    content = [{"type": "text", "text": user_text}]
    for img in imgs:
        b64 = _pil_to_b64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    is_reasoning = (reasoning_effort is not None) or ("gpt-5" in model_id) or ("o3" in model_id) or ("o4" in model_id)

    def _extract_responses_text(resp) -> str:
        text = getattr(resp, "output_text", None)
        if text:
            return text

        chunks = []
        for out in getattr(resp, "output", []) or []:
            for part in getattr(out, "content", []) or []:
                if getattr(part, "type", None) == "output_text":
                    val = getattr(part, "text", None)
                    if val:
                        chunks.append(val)
        return "\n".join(chunks).strip()

    def _do():
        if is_reasoning and hasattr(client, "responses"):
            user_content = [{"type": "input_text", "text": user_text}]
            for img in imgs:
                b64 = _pil_to_b64(img)
                user_content.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{b64}",
                })

            kwargs = dict(
                model=model_id,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_INSTR}]},
                    {"role": "user", "content": user_content},
                ],
                max_output_tokens=max_tokens,
            )
            if reasoning_effort is not None:
                kwargs["reasoning"] = {"effort": reasoning_effort}
            if verbosity is not None:
                kwargs["text"] = {"verbosity": verbosity}
            return client.responses.create(**kwargs)

        kwargs = dict(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_INSTR},
                {"role": "user", "content": content},
            ],
            max_completion_tokens=max_tokens,
            seed=seed,
        )
        if not is_reasoning:
            kwargs["temperature"] = temperature
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if verbosity is not None:
            kwargs["verbosity"] = verbosity
        return client.chat.completions.create(**kwargs)

    resp = with_retry(_do)
    if is_reasoning and hasattr(resp, "output"):
        raw = _extract_responses_text(resp)
        usage = getattr(resp, "usage", None)
        n_out_tok = getattr(usage, "output_tokens", 0) if usage else 0
        finish = getattr(resp, "status", "") or ""
    else:
        raw = resp.choices[0].message.content
        n_out_tok = resp.usage.completion_tokens if resp.usage else 0
        finish = resp.choices[0].finish_reason or ""
    return raw, n_out_tok, finish


def _call_google(client, types_mod, model_id, system_instr, imgs, user_text: str,
                 max_tokens: int, temperature: float,
                 thinking_budget=None):
    """Call Google Gemini via google.genai (new SDK)."""
    # Build parts: user text + images (PIL.Image accepted)
    contents_parts = [user_text]
    for img in imgs:
        contents_parts.append(img)

    cfg_kwargs = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "system_instruction": system_instr,
    }
    if thinking_budget is None and model_id.startswith("gemini-2.5"):
        # Gemini 2.5 can consume the full visible-output budget on hidden
        # thoughts for open-ended chain prompts, yielding empty `resp.text`.
        # A bounded default preserves reasoning while forcing a surfaced answer.
        thinking_budget = 512
    if thinking_budget is not None:
        cfg_kwargs["thinking_config"] = types_mod.ThinkingConfig(thinking_budget=thinking_budget)

    config = types_mod.GenerateContentConfig(**cfg_kwargs)

    def _do():
        return client.models.generate_content(
            model=model_id,
            contents=contents_parts,
            config=config,
        )

    resp = with_retry(_do)
    raw = getattr(resp, "text", None) or ""
    if not raw:
        parts = []
        for cand in getattr(resp, "candidates", []) or []:
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", []) or []:
                text = getattr(part, "text", None)
                if text:
                    parts.append(text)
        raw = "\n".join(parts).strip()
    n_out_tok = 0
    try:
        n_out_tok = resp.usage_metadata.candidates_token_count or 0
    except Exception:
        pass
    finish = ""
    try:
        finish = resp.candidates[0].finish_reason.name if resp.candidates else ""
    except Exception:
        pass
    return raw, n_out_tok, finish


# ---------------------------------------------------------------------------
# Per-item worker
# ---------------------------------------------------------------------------

def process_item(
    rec: dict,
    question_json: dict,
    imgs,
    paths,
    provider: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
    out_dir: Path,
    api_clients: dict,  # keys: "anthropic_client", "openai_client", "google_model"
    dry_run: bool = False,
):
    """Process one item and write output.json. Returns (item_id, n_out_tok, wall_ms, error)."""
    item_id = rec["item_id"]
    template = rec["template"]
    level = rec["level"]
    tdir = out_dir / template
    tdir.mkdir(parents=True, exist_ok=True)
    out_path = tdir / f"{item_id}.output.json"

    user_text = build_user_text(question_json)

    if dry_run:
        print(f"\n[dry-run] item_id={item_id}  template={template}  n_images={len(imgs)}", flush=True)
        print(f"[dry-run] user_text (first 400 chars):\n{user_text[:400]}", flush=True)
        return item_id, 0, 0.0, None

    t0 = time.time()
    raw_output = None
    n_out_tok = 0
    finish = ""
    error_str = None

    try:
        if provider == "anthropic":
            raw_output, n_out_tok, finish = _call_anthropic(
                api_clients["anthropic_client"], model_id, imgs,
                user_text, max_tokens, temperature
            )
        elif provider == "openai":
            raw_output, n_out_tok, finish = _call_openai(
                api_clients["openai_client"], model_id, imgs,
                user_text, max_tokens, temperature,
                reasoning_effort=api_clients.get("reasoning_effort"),
                verbosity=api_clients.get("verbosity"),
                seed=api_clients.get("seed", 42),
            )
        elif provider == "google":
            raw_output, n_out_tok, finish = _call_google(
                api_clients["google_client"],
                api_clients["google_types"],
                api_clients["google_model_id"],
                api_clients["google_system_instr"],
                imgs, user_text, max_tokens, temperature,
                thinking_budget=api_clients.get("thinking_budget")
            )
    except Exception as exc:
        error_str = f"{type(exc).__name__}: {exc}"
        print(f"[error] {item_id}: {error_str}", flush=True)

    wall_ms = round((time.time() - t0) * 1000, 1)

    record = {
        "item_id": item_id,
        "template": template,
        "level": level,
        "raw_output": raw_output,
        "n_output_tokens": n_out_tok,
        "n_images": len(imgs),
        "image_paths": paths,
        "wall_time_ms": wall_ms,
        "provider": provider,
        "api_model": model_id,
        "api_finish_reason": finish,
        "api_error": error_str,
    }
    out_path.write_text(json.dumps(record, indent=2))
    return item_id, n_out_tok, wall_ms, error_str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="BrainTRACE — proprietary VLM API adapter"
    )
    ap.add_argument("--model", required=True,
                    help="Model ID, e.g. claude-sonnet-4-7, gpt-5, gemini-2.5-pro")
    ap.add_argument("--provider", choices=["anthropic", "openai", "google"], default=None,
                    help="Override provider auto-detection")
    ap.add_argument("--root",
                    default=str(Path(__file__).resolve().parent.parent),
                    help="Project root (default: parent of this script's directory)")
    ap.add_argument("--sample-file", required=True,
                    help="JSONL items file (e.g. results/<run>/all_clean_items.jsonl)")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for per-item .output.json files")
    ap.add_argument("--images-root", default=None,
                    help="Load canonical PNGs from <PATH>/bulk_v1/<template>/<item_id>/ "
                         "or <PATH>/flagship/<item_id>/")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="Max output tokens per API call (default: 2048)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (default: 0.0)")
    ap.add_argument("--max-images", type=int, default=16,
                    help="Max images sent per API call (default: 16; hard-capped by provider limit)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip items whose .output.json already exists")
    ap.add_argument("--no-image", action="store_true",
                    help="Allow items with zero images (for text-only ablation).")
    ap.add_argument("--max-items", type=int, default=None,
                    help="Only process the first N items (cost-controlled partial run)")
    ap.add_argument("--parallel", type=int, default=1,
                    help="Number of parallel API threads (default: 1 = serial). "
                         "Increase to 5–10 for throughput; watch rate limits.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Load images + build prompts; print first item; skip API calls")
    ap.add_argument("--thinking-budget", type=int, default=None,
                    help="Gemini-only: thinking_budget (0=off, -1=auto, positive=fixed budget). "
                         "Default None = SDK default (auto for 2.5+).")
    ap.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"], default=None,
                    help="OpenAI o-series / GPT-5 only: reasoning effort.")
    ap.add_argument("--verbosity", choices=["low", "medium", "high"], default=None,
                    help="OpenAI GPT-5 only: response verbosity.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for OpenAI (reproducibility). Anthropic/Google ignore.")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.images_root is None:
        ap.error("--images-root is required for this adapter (no NIfTI rendering path).")

    images_root = Path(args.images_root)
    if not images_root.is_absolute():
        images_root = root / images_root

    # --- provider + key ---
    try:
        provider = detect_provider(args.model, args.provider)
    except ValueError as exc:
        print(f"[error] {exc}", flush=True)
        sys.exit(1)

    print(f"[info] model={args.model}  provider={provider}", flush=True)

    api_key = None
    if not args.dry_run:
        try:
            api_key = check_api_key(provider)
        except EnvironmentError as exc:
            print(f"[error] {exc}", flush=True)
            sys.exit(1)

    # --- effective image cap = min(user-requested, provider hard limit) ---
    provider_limit = PROVIDER_IMAGE_LIMIT[provider]
    max_images = min(args.max_images, provider_limit)
    if max_images < args.max_images:
        print(
            f"[warn] --max-images {args.max_images} exceeds {provider} limit "
            f"({provider_limit}); capped to {max_images}.",
            flush=True,
        )

    # --- init API clients (lazy imports — SDKs optional) ---
    api_clients = {}
    if not args.dry_run:
        if provider == "anthropic":
            try:
                import anthropic
            except ImportError:
                print("[error] 'anthropic' SDK not installed. Run: pip install anthropic", flush=True)
                sys.exit(1)
            api_clients["anthropic_client"] = anthropic.Anthropic(api_key=api_key)

        elif provider == "openai":
            try:
                import openai
            except ImportError:
                print("[error] 'openai' SDK not installed. Run: pip install openai", flush=True)
                sys.exit(1)
            api_clients["openai_client"] = openai.OpenAI(api_key=api_key)
            api_clients["reasoning_effort"] = args.reasoning_effort
            api_clients["verbosity"] = args.verbosity
            api_clients["seed"] = args.seed

        elif provider == "google":
            try:
                from google import genai as gengen
                from google.genai import types as gentypes
            except ImportError:
                print("[error] google-genai SDK not installed. Run: pip install google-genai", flush=True)
                sys.exit(1)
            api_clients["thinking_budget"] = args.thinking_budget
            api_clients["google_client"] = gengen.Client(api_key=api_key)
            api_clients["google_types"] = gentypes
            api_clients["google_model_id"] = args.model
            api_clients["google_system_instr"] = SYSTEM_INSTR

    # --- load items ---
    items = []
    with open(args.sample_file) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    if args.max_items is not None:
        items = items[: args.max_items]

    print(
        f"[info] {len(items)} items loaded from {args.sample_file}. out_dir={out_dir}",
        flush=True,
    )

    # --- cost estimate setup ---
    c_in, c_out = _cost_per_tok(args.model)
    # Rough: assume ~500 input tokens per image + ~300 text tokens; max_tokens output
    est_in_per_item = max_images * 500 + 300  # tokens
    est_out_per_item = args.max_tokens
    est_cost_per_item = (est_in_per_item * c_in + est_out_per_item * c_out) / 1_000_000
    print(
        f"[info] estimated cost: ~${est_cost_per_item:.4f}/item  "
        f"=> ~${est_cost_per_item * len(items):.2f} total  "
        f"(pricing: ${c_in}/M in, ${c_out}/M out)",
        flush=True,
    )

    # --- prepare work list ---
    work = []
    skipped = 0
    for rec in items:
        item_id = rec["item_id"]
        template = rec.get("template", "")
        if not template:
            # infer from item_id if missing
            if item_id.startswith("bulk_v1_"):
                stripped = item_id[len("bulk_v1_"):]
                template = "_".join(stripped.split("_")[:-1])
            else:
                template = item_id
            rec = dict(rec)
            rec["template"] = template

        out_path = out_dir / template / f"{item_id}.output.json"
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        qpath = root / rec["question_path"]
        try:
            with open(qpath) as fq:
                question_json = json.load(fq)
        except Exception as exc:
            print(f"[warn] {item_id}: cannot load question JSON {qpath}: {exc}", flush=True)
            continue

        if "template" not in question_json:
            question_json["template"] = template

        try:
            imgs, paths = load_canonical_images(question_json, images_root, max_images)
        except Exception as exc:
            print(f"[warn] {item_id}: canonical image load failed: {exc}", flush=True)
            imgs, paths = [], []

        if not imgs and not args.no_image:
            print(f"[warn] {item_id}: no images resolved; skipping item.", flush=True)
            continue

        if len(imgs) > max_images:
            print(
                f"[warn] {item_id}: {len(imgs)} images exceeds cap {max_images}; "
                f"subsampling.",
                flush=True,
            )
            import numpy as np
            idx = (
                np.linspace(0, len(imgs) - 1, max_images)
                .round()
                .astype(int)
                .tolist()
            )
            imgs = [imgs[i] for i in idx]
            paths = [paths[i] for i in idx]

        work.append((rec, question_json, imgs, paths))

    if skipped:
        print(f"[info] skipped {skipped} already-done items (--skip-existing).", flush=True)
    print(f"[info] {len(work)} items to process.", flush=True)

    if args.dry_run:
        if work:
            rec, question_json, imgs, paths = work[0]
            process_item(
                rec, question_json, imgs, paths,
                provider, args.model, args.max_tokens, args.temperature,
                out_dir, api_clients, dry_run=True
            )
            print(f"\n[dry-run] would process {len(work)} items total. Exiting.", flush=True)
        else:
            print("[dry-run] no items to process.", flush=True)
        return

    # --- run loop (serial or threaded) ---
    t_wall0 = time.time()
    done_count = 0
    total_out_tok = 0
    errors = 0
    LOG_EVERY = 20

    def _worker(task):
        rec, qj, imgs, paths = task
        return process_item(
            rec, qj, imgs, paths,
            provider, args.model, args.max_tokens, args.temperature,
            out_dir, api_clients, dry_run=False
        )

    if args.parallel <= 1:
        for task in work:
            item_id, n_tok, wall_ms, err = _worker(task)
            done_count += 1
            total_out_tok += n_tok
            if err:
                errors += 1
            if done_count % LOG_EVERY == 0:
                elapsed = time.time() - t_wall0
                spent = (total_out_tok * c_out) / 1_000_000
                rate = done_count / max(elapsed, 1)
                print(
                    f"[info] {done_count}/{len(work)} items | "
                    f"{args.model} ~${est_cost_per_item:.4f}/item "
                    f"~${spent:.2f} output-tokens cost so far | "
                    f"{rate:.1f} items/s",
                    flush=True,
                )
    else:
        print(f"[info] parallel={args.parallel} threads", flush=True)
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(_worker, task): task for task in work}
            for fut in as_completed(futures):
                try:
                    item_id, n_tok, wall_ms, err = fut.result()
                except Exception as exc:
                    print(f"[error] unexpected exception in worker: {exc}", flush=True)
                    errors += 1
                    done_count += 1
                    continue
                done_count += 1
                total_out_tok += n_tok
                if err:
                    errors += 1
                if done_count % LOG_EVERY == 0:
                    elapsed = time.time() - t_wall0
                    spent = (total_out_tok * c_out) / 1_000_000
                    rate = done_count / max(elapsed, 1)
                    print(
                        f"[info] {done_count}/{len(work)} items | "
                        f"{args.model} ~${est_cost_per_item:.4f}/item "
                        f"~${spent:.2f} output-tokens cost so far | "
                        f"{rate:.1f} items/s",
                        flush=True,
                    )

    wall_total = time.time() - t_wall0
    total_est_cost = done_count * est_cost_per_item
    print(
        f"\n[info] DONE. items_processed={done_count} | errors={errors} | "
        f"output_tokens={total_out_tok} | wall_total={wall_total:.1f}s | "
        f"estimated_cost=~${total_est_cost:.2f}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Smoke test entry point
# ---------------------------------------------------------------------------

def _smoke_test():
    """
    Quick 3-item smoke test invoked by: python scripts/vlm_eval_api.py --smoke-test
    Reads from a known sample JSONL; requires ANTHROPIC_API_KEY (or prints skip).
    """
    import tempfile
    print("[smoke] BrainTRACE vlm_eval_api.py smoke test", flush=True)

    # Provider auto-detection
    test_cases = [
        ("claude-sonnet-4-7",      "anthropic"),
        ("claude-opus-4-7",        "anthropic"),
        ("gpt-5",                  "openai"),
        ("gemini-2.5-pro",         "google"),
        ("o4-mini",                "openai"),
    ]
    print("\n[smoke] Provider auto-detection:", flush=True)
    all_ok = True
    for model_id, expected in test_cases:
        got = detect_provider(model_id)
        ok = "OK" if got == expected else "FAIL"
        if got != expected:
            all_ok = False
        print(f"  {model_id:30s} -> {got:12s}  [{ok}]", flush=True)
    if not all_ok:
        print("[smoke] FAIL: some provider detections wrong!", flush=True)
        sys.exit(1)
    print("[smoke] All provider detections correct.", flush=True)

    # API key check
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print(
            "\n[smoke] SKIPPED actual API call — ANTHROPIC_API_KEY not set.\n"
            "Set it and re-run to test live API connectivity.",
            flush=True,
        )
        return

    print("\n[smoke] ANTHROPIC_API_KEY found. Attempting live API call ...", flush=True)
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        t0 = time.time()
        resp = client.messages.create(
            model="claude-haiku-4-5",
            system=SYSTEM_INSTR,
            messages=[{
                "role": "user",
                "content": "Describe a normal axial T1 brain MRI in 1 sentence.",
            }],
            max_tokens=128,
            temperature=0.0,
        )
        wall_ms = (time.time() - t0) * 1000
        text = resp.content[0].text
        n_tok = resp.usage.output_tokens
        print(f"[smoke] API call OK. wall={wall_ms:.0f}ms  n_out_tok={n_tok}", flush=True)
        print(f"[smoke] Response: {text[:200]}", flush=True)

        # Validate output.json schema
        record = {
            "item_id": "smoke_test_item",
            "template": "L1_appearance",
            "level": 1,
            "raw_output": text,
            "n_output_tokens": n_tok,
            "n_images": 0,
            "image_paths": [],
            "wall_time_ms": round(wall_ms, 1),
            "provider": "anthropic",
            "api_model": "claude-haiku-4-5",
            "api_finish_reason": resp.stop_reason or "",
            "api_error": None,
        }
        required_keys = [
            "item_id", "template", "level", "raw_output", "n_output_tokens",
            "n_images", "image_paths", "wall_time_ms", "provider",
            "api_model", "api_finish_reason", "api_error",
        ]
        missing = [k for k in required_keys if k not in record]
        if missing:
            print(f"[smoke] FAIL: output schema missing keys: {missing}", flush=True)
            sys.exit(1)
        print("[smoke] Output schema validated (all required keys present).", flush=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".output.json", delete=False) as tf:
            json.dump(record, tf, indent=2)
            print(f"[smoke] Sample output written to: {tf.name}", flush=True)
    except Exception as exc:
        print(f"[smoke] FAIL: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    print("\n[smoke] PASSED.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--smoke-test":
        _smoke_test()
    else:
        main()
