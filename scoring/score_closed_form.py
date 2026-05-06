"""BrainTRACE - closed-form scorer.

Deterministic exact-match scorer for the ``answer_type == "closed_form"``
subset of the BrainTRACE parquet. Implements the public scoring contract
documented in ``docs/SCORING.md`` Section 3.

Pipeline (per item):
  1. Strip any pre-``</think>`` chain-of-thought block from the prediction
     so inline-reasoning models (e.g. open-weights "thinking" variants)
     are comparable to API thinking models that hide CoT server-side.
  2. Light text normalisation: remove a leading ``Answer:`` style label,
     remove a leading MCQ prefix (``(A)``, ``A.``, ``A)``, ``[A]``, ``A -``),
     lowercase, trim whitespace and one trailing period.
  3. If the normalised prediction still starts with a bare letter ``A``-``E``
     (with optional brackets/parens, terminated by whitespace, punctuation,
     or end-of-string), resolve that letter against the parquet's
     ``options`` array and compare the resolved option text to ``gt_value``.
  4. ``"D (FLAIR)"`` echo style: a leading letter immediately followed by
     a parenthesised substantive phrase is also accepted, with the
     parenthesised phrase compared as the answer.

The headline metric is per-(track, sub_category) accuracy plus a 1000-iter
percentile bootstrap 95% CI. Per-level we additionally report the random
baseline (mean of ``1/|options|``) and the chance-debiased score
``CDS = max(0, (acc - chance) / (1 - chance))``.

The module exposes a small functional surface that ``score.py`` (the
unified entry-point) imports directly; running this file as a script is
also supported for closed-form-only scoring runs.

Usage as a script::

    python score_closed_form.py \\
        --dataset path/to/braintrace_dataset \\
        --predictions path/to/predictions \\
        --out-dir path/to/scores
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
# Regex constants — duplicated minimally; mirror the released SCORING.md
# ---------------------------------------------------------------------------

# Strip a leading MCQ prefix in any of these forms:
#   "(A) ...", "A) ...", "A. ...", "A: ...", "[A] ...", "A - ..."
# The letter must be A-E (we cap option count at 5 in the dataset);
# the prefix must be followed by at least one whitespace char.
_MCQ_PREFIX = re.compile(r"^\s*(?:\(?\[?([A-Ea-e])\]?[\.\)\:\-]|\(([A-Ea-e])\))\s+")

# Strip a leading "Answer:" style label.
_ANSWER_LABEL = re.compile(
    r"^\s*(?:the\s+)?(?:correct\s+|final\s+)?answer\s*(?:is)?\s*[:\-]?\s*",
    re.IGNORECASE,
)

# Permissive "starts with a letter" detector for the letter-only fallback.
# Accepts forms like "D", "D.", "(D)", "[D]", "D ...", "D-explanation".
# Uses lookahead so the match does not consume the trailing context.
_LEADING_LETTER = re.compile(r"^\s*\(?\[?([A-Ea-e])\]?\)?(?=[^A-Za-z0-9]|$)")

# "D (FLAIR)" / "B) FLAIR (post-contrast)" echo: a leading letter
# immediately followed by a parenthesised substantive phrase. We strip
# the letter and accept the parenthesised text as the substantive answer.
_LETTER_THEN_PAREN = re.compile(r"^\s*\(?\[?[A-Ea-e]\]?\)?\s*\(([^)]+)\)\s*$")


# ---------------------------------------------------------------------------
# Core string ops
# ---------------------------------------------------------------------------

def strip_think(text: str) -> str:
    """Strip everything up to and including the *last* ``</think>`` tag.

    Inline-reasoning models (e.g. some open-weights variants with
    "thinking" mode on) emit ``<think>...</think>`` blocks directly in
    their output. API thinking models hide that text server-side. To
    keep scoring uniform we drop the pre-``</think>`` text and keep only
    the post-tag final answer. Returns the input unchanged if no
    ``</think>`` tag is present.

    Uses ``rfind`` (string-based, O(n)) rather than a regex to avoid
    catastrophic backtracking on long outputs.
    """
    if not text:
        return text or ""
    s = str(text)
    lower = s.lower()
    if "</think>" not in lower:
        return s
    idx = lower.rfind("</think>")
    return s[idx + len("</think>"):]


def normalise_closed(text: Optional[str]) -> str:
    """Apply the closed-form normalisation pipeline.

    Order of operations:
      1. ``strip_think`` (drop pre-``</think>`` block if present).
      2. ``"D (FLAIR)"`` echo collapse: keep only the parenthesised phrase.
      3. Strip a leading ``Answer:`` style label.
      4. Strip a leading MCQ prefix (``(A)``, ``A.`` etc.).
      5. ``rstrip(".")``, lowercase, trim whitespace.
    """
    if text is None:
        return ""
    s = strip_think(str(text)).strip()
    m_paren = _LETTER_THEN_PAREN.match(s)
    if m_paren:
        s = m_paren.group(1)
    s = _ANSWER_LABEL.sub("", s)
    s = _MCQ_PREFIX.sub("", s)
    return s.strip().rstrip(".").strip().lower()


def letter_only(text: Optional[str]) -> Optional[str]:
    """Return the leading A-E letter (lowercased) if the prediction
    starts with one, else ``None``. Applies ``strip_think`` first.

    Permissive: matches ``"D"``, ``"D."``, ``"(D)"``, ``"[D]"``,
    ``"D - explanation"``, ``"D. it is a FLAIR sequence"`` etc., but
    does *not* match free-text answers that happen to begin with a
    letter followed by another letter (e.g. ``"Diffusion ..."``).
    """
    if text is None:
        return None
    s = strip_think(str(text)).strip()
    m = _LEADING_LETTER.match(s)
    return m.group(1).lower() if m else None


def closed_match(pred: Any, gt: Any, options: Optional[Sequence[Any]]) -> bool:
    """Return True iff the prediction matches the ground truth.

    Resolution order:
      1. Normalised text equality.
      2. Letter-only fallback against ``options``.
    """
    np_, ng = normalise_closed(pred), normalise_closed(gt)
    if np_ and np_ == ng:
        return True
    letter = letter_only(pred)
    if letter and options is not None:
        try:
            opts = [str(o) for o in options if o is not None]
        except TypeError:
            opts = []
        idx = ord(letter) - ord("a")
        if 0 <= idx < len(opts):
            return normalise_closed(opts[idx]) == ng
    return False


# ---------------------------------------------------------------------------
# Bootstrap CI + chance-debiased score
# ---------------------------------------------------------------------------

def bootstrap_ci(
    flags: Sequence[bool],
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for accuracy expressed in percent.

    Returns ``(0.0, 0.0)`` for the empty-input edge case.
    """
    if not flags:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(flags)
    samples = []
    for _ in range(n_iter):
        s = sum(1 for _ in range(n) if flags[rng.randrange(n)])
        samples.append(s / n)
    samples.sort()
    lo = samples[max(0, int(n_iter * (alpha / 2)))]
    hi = samples[min(n_iter - 1, int(n_iter * (1 - alpha / 2)))]
    return (lo * 100.0, hi * 100.0)


def random_baseline(option_lists: Iterable[Optional[Sequence[Any]]]) -> float:
    """Per-item uniform-guess baseline as a percentage in [0, 100]."""
    accs = []
    for opts in option_lists:
        if opts is None:
            continue
        try:
            n = sum(1 for o in opts if o is not None)
        except TypeError:
            continue
        if n > 1:
            accs.append(1.0 / n)
    return (sum(accs) / len(accs) * 100.0) if accs else 0.0


def chance_debiased(acc_pct: float, chance_pct: float) -> float:
    """``CDS = max(0, (acc - chance) / (1 - chance))``, all on [0, 100]."""
    if chance_pct >= 100.0:
        return 0.0
    return max(0.0, (acc_pct - chance_pct) / (100.0 - chance_pct) * 100.0)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

_OUTPUT_KEYS = ("raw_output", "output", "response", "text", "completion")


def _extract_output(obj: Mapping[str, Any]) -> str:
    for k in _OUTPUT_KEYS:
        v = obj.get(k)
        if v:
            return str(v)
    return ""


def load_predictions(pred_path: Path) -> dict[str, dict]:
    """Load predictions from either a JSONL file or a directory tree.

    JSONL: one ``{"item_id": ..., ...}`` per line.
    Directory: scanned recursively for ``*.json`` / ``*.output.json``;
    each file must carry an ``item_id`` field (the file stem is used as
    a fallback when missing).
    """
    preds: dict[str, dict] = {}
    if pred_path.is_dir():
        for fp in pred_path.rglob("*.json"):
            try:
                obj = json.loads(fp.read_text())
            except Exception:
                continue
            iid = obj.get("item_id") or fp.stem.replace(".output", "")
            preds[iid] = obj
        return preds
    if pred_path.suffix in {".jsonl", ".ndjson"}:
        with pred_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                preds[obj["item_id"]] = obj
        return preds
    raise SystemExit(f"unsupported predictions path: {pred_path}")


def load_parquet_index(parquet_path: Path) -> dict[str, dict[str, Any]]:
    """Read the parquet into a dict keyed by ``item_id``."""
    import pyarrow.parquet as pq  # local import keeps unit tests light

    df = pq.read_table(str(parquet_path)).to_pandas()
    out: dict[str, dict[str, Any]] = {}
    for _, r in df.iterrows():
        out[r["item_id"]] = r.to_dict()
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def score_closed_form(
    items: Mapping[str, Mapping[str, Any]],
    preds: Mapping[str, Mapping[str, Any]],
    n_bootstrap: int = 1000,
) -> dict:
    """Score the closed-form subset and return a structured report.

    The report has three keys:
      ``per_item``      list of one record per scored item
      ``per_track_subcat``  dict[(track, sub_category) -> stats]
      ``per_level``     dict[level -> stats] including chance and CDS

    Items missing predictions are excluded from denominators (they are
    not counted as failures).
    """
    rows: list[dict] = []
    flags_by_track_sub: dict[tuple[str, str], list[bool]] = defaultdict(list)
    flags_by_level: dict[int, list[bool]] = defaultdict(list)
    options_by_level: dict[int, list[list[Any]]] = defaultdict(list)

    for iid, item in items.items():
        if item.get("answer_type") != "closed_form":
            continue
        if iid not in preds:
            continue  # missing prediction → skip from denominator
        pred_obj = preds[iid]
        pred = _extract_output(pred_obj) if isinstance(pred_obj, dict) else str(pred_obj)
        gt = item.get("gt_value")

        opts = item.get("options")
        if isinstance(opts, float) and math.isnan(opts):
            opts = None
        if opts is not None:
            try:
                opts = [o for o in opts if o is not None]
            except TypeError:
                opts = None

        ok = closed_match(pred, gt, opts)
        track = item.get("track", "unknown")
        sub_cat = item.get("sub_category") or "_unspecified"
        level = int(item.get("level") or 0)

        rows.append({
            "item_id": iid,
            "track": track,
            "level": level,
            "sub_category": sub_cat,
            "template": item.get("template"),
            "gt": gt,
            "pred": pred,
            "correct": bool(ok),
        })
        flags_by_track_sub[(track, sub_cat)].append(ok)
        flags_by_level[level].append(ok)
        options_by_level[level].append(opts if opts is not None else [])

    per_track_subcat: dict[str, dict] = {}
    for (trk, sub), flags in flags_by_track_sub.items():
        n = len(flags)
        n_correct = sum(1 for f in flags if f)
        acc = n_correct / n * 100.0 if n else 0.0
        lo, hi = bootstrap_ci(flags, n_iter=n_bootstrap)
        per_track_subcat[f"{trk}::{sub}"] = {
            "track": trk,
            "sub_category": sub,
            "n": n,
            "n_correct": n_correct,
            "accuracy_pct": round(acc, 2),
            "ci95_pct": [round(lo, 2), round(hi, 2)],
        }

    per_level: dict[int, dict] = {}
    for lvl, flags in flags_by_level.items():
        n = len(flags)
        n_correct = sum(1 for f in flags if f)
        acc = n_correct / n * 100.0 if n else 0.0
        lo, hi = bootstrap_ci(flags, n_iter=n_bootstrap)
        chance = random_baseline(options_by_level[lvl])
        cds = chance_debiased(acc, chance)
        per_level[lvl] = {
            "level": lvl,
            "n": n,
            "n_correct": n_correct,
            "accuracy_pct": round(acc, 2),
            "ci95_pct": [round(lo, 2), round(hi, 2)],
            "chance_pct": round(chance, 2),
            "cds_pct": round(cds, 2),
        }

    # Track-overall
    by_track_overall: dict[str, dict] = {}
    flags_by_track: dict[str, list[bool]] = defaultdict(list)
    options_by_track: dict[str, list[list[Any]]] = defaultdict(list)
    for r in rows:
        flags_by_track[r["track"]].append(r["correct"])
        item = items.get(r["item_id"], {})
        opts = item.get("options")
        if isinstance(opts, float) and math.isnan(opts):
            opts = None
        if opts is not None:
            try:
                opts = [o for o in opts if o is not None]
            except TypeError:
                opts = None
        options_by_track[r["track"]].append(opts if opts is not None else [])

    for trk, flags in flags_by_track.items():
        n = len(flags)
        n_correct = sum(1 for f in flags if f)
        acc = n_correct / n * 100.0 if n else 0.0
        lo, hi = bootstrap_ci(flags, n_iter=n_bootstrap)
        chance = random_baseline(options_by_track[trk])
        by_track_overall[trk] = {
            "track": trk,
            "n": n,
            "n_correct": n_correct,
            "accuracy_pct": round(acc, 2),
            "ci95_pct": [round(lo, 2), round(hi, 2)],
            "chance_pct": round(chance, 2),
            "cds_pct": round(chance_debiased(acc, chance), 2),
        }

    return {
        "per_item": rows,
        "per_track_subcat": per_track_subcat,
        "per_level": per_level,
        "per_track_overall": by_track_overall,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="BrainTRACE closed-form scorer")
    ap.add_argument("--dataset", type=Path, required=True,
                    help="Path to braintrace_dataset/ (must contain data/test.parquet)")
    ap.add_argument("--predictions", type=Path, required=True,
                    help="Predictions JSONL or directory of per-item JSON")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Where to write scored_closed_form.jsonl + summary_closed_form.json")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    args = ap.parse_args(argv)

    parquet = args.dataset / "data" / "test.parquet"
    if not parquet.exists():
        print(f"missing parquet: {parquet}")
        return 2
    items = load_parquet_index(parquet)
    preds = load_predictions(args.predictions)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = score_closed_form(items, preds, n_bootstrap=args.n_bootstrap)
    (args.out_dir / "scored_closed_form.jsonl").write_text(
        "\n".join(json.dumps(r) for r in report["per_item"])
    )
    summary = {
        "per_track_overall": report["per_track_overall"],
        "per_track_subcat": report["per_track_subcat"],
        "per_level": report["per_level"],
    }
    (args.out_dir / "summary_closed_form.json").write_text(
        json.dumps(summary, indent=2)
    )
    for trk, s in summary["per_track_overall"].items():
        print(f"  {trk:20s} n={s['n']:>5d}  acc={s['accuracy_pct']}%  "
              f"CI95={s['ci95_pct']}  chance={s['chance_pct']}%  "
              f"CDS={s['cds_pct']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
