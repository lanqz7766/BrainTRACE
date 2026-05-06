"""BrainTRACE - open-ended + case-reasoning scorer (LLM-judge wrapper).

Materialises the parquet's open-ended (Pattern C) and chain
(Pattern D, ``answer_type == "chain"``, formerly "case_reasoning")
items into the on-disk layout that ``llm_judge.py`` expects, runs the
judge as a subprocess, then re-aggregates the per-item assessments
into headline metrics matching ``docs/SCORING.md`` Sections 4-5.

Inline ``<think>...</think>`` blocks in model outputs are stripped
before judging via the always-on ``_strip_inline_thinking`` hook
inside ``llm_judge.py`` (no extra flag is required).

Open-ended aggregation buckets (paper Table 3 categories):

  L1-ABN   L1.6     abnormality detection (open form)
  L2-APP   L2.6     appearance characterisation
  L3-EFF   L3.6     mass-effect / interval characterisation
  L4-TRJ   L4.4     trajectory description
  IMP      L5.1+L5.2 diagnostic impression (synthesis)
  CMP      L5.3     comparison / interval reasoning
  DIF      L5.4     differential diagnosis

Headline metrics (per bucket):
  ``item_pass_rate``         fraction of items with all step rubrics passed
  ``mean_slot_match_rate``   mean over items of slot-pass fraction
  ``mean_quality_1_5``       arithmetic mean of derived 1-5 quality score
  ``faithfulness_composite`` mean of slot-match * unsupported-discount * critical-discount

Chain (case_reasoning) aggregation:
  ``step_pass_pct``          steps_passed / 420 * 100
  ``cesr_pct``               item_pass / 70 * 100
  ``ge_k_of_6``              fraction with at least k steps passed (k=1..6)
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


# ---------------------------------------------------------------------------
# Open-ended bucket map (paper Table 3). Keyed by (level, sub_category).
# Sub-category strings come from the parquet's ``sub_category`` column.
# ---------------------------------------------------------------------------

OPEN_ENDED_BUCKETS: dict[tuple[int, str], str] = {
    (1, "Det"):    "L1-ABN",
    (2, "App"):    "L2-APP",
    (3, "Effect"): "L3-EFF",
    (4, "Traj"):   "L4-TRJ",
    (5, "Imp"):    "IMP",
    (5, "Comp"):   "CMP",
    (5, "Diff"):   "DIF",
}


def open_ended_bucket(level: int, sub_category: Optional[str]) -> Optional[str]:
    """Map a parquet (level, sub_category) pair to its paper-table bucket."""
    return OPEN_ENDED_BUCKETS.get((int(level), str(sub_category or "")))


# ---------------------------------------------------------------------------
# Quality 1-5 score (deterministic; from docs/SCORING.md §4).
# ---------------------------------------------------------------------------

def derive_quality_score(
    *,
    n_steps: int,
    n_steps_passed: int,
    mean_slot_match_rate: float,
    mean_clinically_acceptable: float,
    total_critical_errors: int,
    total_unsupported_claims: int,
    pass_threshold_frac: float = 0.6,
) -> int:
    """Return a 1-5 quality score derived from per-item judge fields.

    Mapping (highest matching rule wins, top to bottom):
      5  clinically_acceptable AND slot_rate == 1 AND no unsupported claims
      4  clinically_acceptable AND slot_rate < 1
      3  no critical errors AND slot_rate > 0.5 AND (some shortfall)
      2  no critical errors AND slot_rate <= 0.5
      1  any critical error
    """
    if total_critical_errors > 0:
        return 1
    clin_ok = mean_clinically_acceptable >= 0.5  # majority of steps acceptable
    slot_rate = float(mean_slot_match_rate or 0.0)
    if clin_ok and slot_rate >= 1.0 - 1e-9 and total_unsupported_claims == 0:
        return 5
    if clin_ok and slot_rate < 1.0:
        return 4
    if slot_rate > 0.5 and (
        total_unsupported_claims > 0 or slot_rate < pass_threshold_frac
    ):
        return 3
    if slot_rate <= 0.5:
        return 2
    return 3


def faithfulness_score(
    *,
    mean_slot_match_rate: float,
    total_critical_errors: int,
    total_unsupported_claims: int,
) -> float:
    """Composite faithfulness in [0, 1]:

        slot_match * (1 - 0.3 * min(1, unsupported / 3))
                   * (1 - min(1, critical / 3))
    """
    slot = max(0.0, min(1.0, float(mean_slot_match_rate or 0.0)))
    unsup_pen = 1.0 - 0.3 * min(1.0, total_unsupported_claims / 3.0)
    crit_pen = 1.0 - min(1.0, total_critical_errors / 3.0)
    return slot * unsup_pen * crit_pen


# ---------------------------------------------------------------------------
# Materialise inputs for llm_judge.py
# ---------------------------------------------------------------------------

def materialise_judge_inputs(
    items: Mapping[str, Mapping[str, Any]],
    preds: Mapping[str, Mapping[str, Any]],
    work_dir: Path,
    *,
    include_open: bool = True,
    include_chain: bool = True,
) -> tuple[Path, Path, Path, int]:
    """Write three trees that ``llm_judge.py`` expects:

      ``gt_root/<template_underscored>/<item_id>.gt.json``
      ``questions_root/<template_underscored>/<item_id>.question.json``
      ``outputs_dir/<item_id>.output.json``

    Returns ``(gt_root, questions_root, outputs_dir, n_items_written)``.
    """
    gt_root = work_dir / "gt"
    q_root = work_dir / "questions"
    out_dir = work_dir / "outputs"
    gt_root.mkdir(parents=True, exist_ok=True)
    q_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for iid, item in items.items():
        atype = item.get("answer_type")
        if atype == "open_ended":
            if not include_open:
                continue
        elif atype in ("chain", "case_reasoning"):
            if not include_chain:
                continue
        else:
            continue
        if iid not in preds:
            continue

        template_under = (item.get("template") or "").replace(".", "_")
        # GT
        try:
            gt_obj = json.loads(item["rubric_json"]) if item.get("rubric_json") else {}
        except Exception:
            gt_obj = {}
        gt_dir = gt_root / template_under
        gt_dir.mkdir(parents=True, exist_ok=True)
        (gt_dir / f"{iid}.gt.json").write_text(json.dumps(gt_obj, ensure_ascii=False))

        # Question
        q_dir = q_root / template_under
        q_dir.mkdir(parents=True, exist_ok=True)
        q_obj = {
            "item_id": iid,
            "template_id": item.get("template"),
            "level": int(item.get("level") or 0),
            "model_text_input": {
                "rendered_question": item.get("question", ""),
                "answer_type": (
                    "open_ended_paragraph" if atype == "open_ended" else "case_reasoning"
                ),
            },
        }
        if atype in ("chain", "case_reasoning") and item.get("chain_steps"):
            try:
                q_obj["model_text_input"]["chain_questions"] = json.loads(item["chain_steps"])
            except Exception:
                pass
        (q_dir / f"{iid}.question.json").write_text(json.dumps(q_obj, ensure_ascii=False))

        # Output
        pred_obj = preds[iid]
        if not isinstance(pred_obj, dict):
            pred_obj = {"item_id": iid, "raw_output": str(pred_obj)}
        else:
            pred_obj = dict(pred_obj)
        pred_obj.setdefault("item_id", iid)
        (out_dir / f"{iid}.output.json").write_text(json.dumps(pred_obj, ensure_ascii=False))
        n += 1
    return gt_root, q_root, out_dir, n


def run_llm_judge(
    *,
    judge_script: Path,
    gt_root: Path,
    questions_root: Path,
    outputs_dir: Path,
    out_dir: Path,
    judge_model: str = "gpt-4o-mini-2024-07-18",
    max_parallel: int = 8,
    extra_args: Sequence[str] = (),
) -> int:
    cmd = [
        sys.executable, str(judge_script),
        "--outputs-dir", str(outputs_dir),
        "--gt-root", str(gt_root),
        "--questions-root", str(questions_root),
        "--out-dir", str(out_dir),
        "--judge-model", judge_model,
        "--max-parallel", str(max_parallel),
        "--skip-existing",
        *extra_args,
    ]
    print("[score_open_judge] running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _read_assessments(judge_out: Path) -> list[dict]:
    f = judge_out / "per_item_assessments.jsonl"
    if not f.exists():
        return []
    rows = []
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def aggregate_open_ended(
    items: Mapping[str, Mapping[str, Any]],
    judge_rows: Sequence[Mapping[str, Any]],
) -> dict:
    """Aggregate per-item open-ended assessments into paper-table buckets.

    Returns a dict keyed by bucket name (or ``"_unbucketed"``) with
    ``n``, ``item_pass_rate_pct``, ``mean_slot_match_rate``,
    ``mean_quality_1_5``, ``faithfulness_composite`` and per-track totals.
    """
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    by_track: dict[str, list[dict]] = defaultdict(list)
    by_track_bucket: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for r in judge_rows:
        iid = r.get("item_id")
        item = items.get(iid)
        if item is None or item.get("answer_type") != "open_ended":
            continue
        level = int(item.get("level") or 0)
        sub = item.get("sub_category")
        bucket = open_ended_bucket(level, sub) or "_unbucketed"
        track = item.get("track", "unknown")
        by_bucket[bucket].append(r)
        by_track[track].append(r)
        by_track_bucket[(track, bucket)].append(r)

    def _agg(rows: Sequence[Mapping[str, Any]]) -> dict:
        n = len(rows)
        if n == 0:
            return {"n": 0}
        n_pass = sum(1 for r in rows if r.get("item_pass"))
        sum_slot = 0.0
        sum_q = 0
        sum_faith = 0.0
        sum_crit = 0
        sum_unsup = 0
        for r in rows:
            slot = float(r.get("mean_slot_match_rate") or 0.0)
            crit = int(r.get("total_critical_errors") or 0)
            unsup = int(r.get("total_unsupported_claims") or 0)
            n_steps = int(r.get("n_steps") or 1)
            n_steps_pass = int(r.get("n_steps_passed") or 0)
            mean_clin = float(r.get("mean_clinically_acceptable") or 0.0)
            sum_slot += slot
            sum_crit += crit
            sum_unsup += unsup
            sum_q += derive_quality_score(
                n_steps=n_steps,
                n_steps_passed=n_steps_pass,
                mean_slot_match_rate=slot,
                mean_clinically_acceptable=mean_clin,
                total_critical_errors=crit,
                total_unsupported_claims=unsup,
            )
            sum_faith += faithfulness_score(
                mean_slot_match_rate=slot,
                total_critical_errors=crit,
                total_unsupported_claims=unsup,
            )
        return {
            "n": n,
            "item_pass_rate_pct": round(n_pass / n * 100.0, 2),
            "mean_slot_match_rate": round(sum_slot / n, 4),
            "mean_quality_1_5": round(sum_q / n, 3),
            "faithfulness_composite": round(sum_faith / n, 4),
            "n_critical_errors_mean": round(sum_crit / n, 3),
            "n_unsupported_claims_mean": round(sum_unsup / n, 3),
        }

    return {
        "by_bucket": {k: _agg(v) for k, v in by_bucket.items()},
        "by_track": {k: _agg(v) for k, v in by_track.items()},
        "by_track_bucket": {
            f"{trk}::{buc}": _agg(v) for (trk, buc), v in by_track_bucket.items()
        },
    }


def aggregate_chain(
    items: Mapping[str, Mapping[str, Any]],
    judge_rows: Sequence[Mapping[str, Any]],
    *,
    n_steps_per_case: int = 6,
    n_cases_total: int = 70,
) -> dict:
    """Compute Step Pass + CESR + per-k distribution for the chain track."""
    chain_rows: list[dict] = []
    for r in judge_rows:
        iid = r.get("item_id")
        item = items.get(iid)
        if item is None or item.get("answer_type") not in ("chain", "case_reasoning"):
            continue
        chain_rows.append(dict(r))

    n_items = len(chain_rows)
    n_steps_total = sum(int(r.get("n_steps", 0)) for r in chain_rows)
    n_steps_passed = sum(int(r.get("n_steps_passed", 0)) for r in chain_rows)
    n_item_pass = sum(1 for r in chain_rows if r.get("item_pass"))
    expected_steps = n_cases_total * n_steps_per_case

    ge_k = {}
    for k in range(1, n_steps_per_case + 1):
        ge_k[f"ge_{k}_of_{n_steps_per_case}"] = round(
            sum(1 for r in chain_rows if int(r.get("n_steps_passed", 0)) >= k) /
            max(1, n_cases_total) * 100.0,
            2,
        )

    return {
        "n_items_judged": n_items,
        "n_items_expected": n_cases_total,
        "n_steps_total": n_steps_total,
        "n_steps_passed": n_steps_passed,
        "n_steps_expected": expected_steps,
        "step_pass_pct": round(n_steps_passed / max(1, expected_steps) * 100.0, 2),
        "cesr_pct": round(n_item_pass / max(1, n_cases_total) * 100.0, 2),
        **ge_k,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_parquet_index(parquet_path: Path) -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as pq
    df = pq.read_table(str(parquet_path)).to_pandas()
    return {r["item_id"]: r.to_dict() for _, r in df.iterrows()}


def load_predictions(pred_path: Path) -> dict[str, dict]:
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="BrainTRACE open + chain scorer (LLM judge)")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--judge-model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--max-parallel", type=int, default=8)
    ap.add_argument("--mode", choices=["open", "chain", "both"], default="both")
    ap.add_argument("--reuse-judge-dir", type=Path, default=None,
                    help="Skip the judge subprocess and re-aggregate from this directory.")
    args = ap.parse_args(argv)

    items = load_parquet_index(args.dataset / "data" / "test.parquet")
    preds = load_predictions(args.predictions)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent
    judge_script = repo_root / "scoring" / "llm_judge.py"

    if args.reuse_judge_dir:
        judge_dir = args.reuse_judge_dir
    else:
        judge_dir = args.out_dir / "_llm_judge"
        judge_dir.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="braintrace_judge_") as tmp:
            tmp_p = Path(tmp)
            gt_root, q_root, out_dir, n = materialise_judge_inputs(
                items, preds, tmp_p,
                include_open=args.mode in ("open", "both"),
                include_chain=args.mode in ("chain", "both"),
            )
            print(f"[score_open_judge] materialised {n} items in {tmp_p}")
            if n == 0:
                print("[score_open_judge] nothing to judge — done")
                return 0
            rc = run_llm_judge(
                judge_script=judge_script,
                gt_root=gt_root, questions_root=q_root,
                outputs_dir=out_dir, out_dir=judge_dir,
                judge_model=args.judge_model,
                max_parallel=args.max_parallel,
            )
            if rc != 0:
                print(f"[score_open_judge] WARN: llm_judge exited rc={rc}")

    rows = _read_assessments(judge_dir)
    summary: dict = {"judge_dir": str(judge_dir), "n_assessments": len(rows)}
    if args.mode in ("open", "both"):
        summary["open_ended"] = aggregate_open_ended(items, rows)
    if args.mode in ("chain", "both"):
        summary["chain"] = aggregate_chain(items, rows)
    (args.out_dir / "summary_open_judge.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
