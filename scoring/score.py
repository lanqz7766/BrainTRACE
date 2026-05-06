"""BrainTRACE - unified scoring entry point.

Single CLI that scores a predictions set against the BrainTRACE parquet
and writes a unified ``summary.json``. Internally it dispatches to two
specialist modules:

  * ``score_closed_form.py``  — deterministic exact-match (Patterns A/B)
  * ``score_open_judge.py``   — LLM-as-judge (Patterns C/D)

Inputs
------
``--dataset``      directory containing ``data/test.parquet``
``--predictions``  JSONL/NDJSON file or directory tree of per-item JSON
``--out-dir``      output directory; created if absent

Modes
-----
``--mode all``     (default) closed-form + open-ended + chain
``--mode closed``  closed-form only
``--mode open``    open-ended + chain only

Other flags
-----------
``--skip-judge``         alias for ``--mode closed``
``--judge-model``        OpenAI model id; default ``gpt-4o-mini-2024-07-18``
``--max-parallel``       judge call concurrency (default 8)
``--reuse-judge-dir``    re-aggregate from existing judge output without re-calling
``--n-bootstrap``        bootstrap iterations for closed-form CI (default 1000)

Inline-thinking handling
------------------------
Predictions that contain a ``<think>...</think>`` reasoning block (some
open-weights "thinking" model variants emit this) are stripped before
scoring on both pathways. The closed-form scorer drops everything up to
the last ``</think>``; the LLM judge applies the same rule before
sending text to the judge model. This keeps inline-CoT and
hidden-CoT model surfaces directly comparable.

Outputs
-------
``out-dir/scored_closed_form.jsonl``       per-item closed-form rows
``out-dir/summary_closed_form.json``       closed-form aggregates
``out-dir/_llm_judge/``                    raw judge artefacts (audit trail)
``out-dir/summary_open_judge.json``        open + chain aggregates
``out-dir/summary.json``                   merged headline summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

# Allow being run either as a module or as a script from the scoring/ dir.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import score_closed_form as cf  # noqa: E402
import score_open_judge as oj  # noqa: E402


def _detect_thinking_rate(preds: dict, sample_n: int = 200) -> float:
    """Estimate the fraction of predictions that contain a ``<think>`` tag.

    Used only for an informational log line — stripping itself is
    unconditional in both pathways.
    """
    if not preds:
        return 0.0
    iids = list(preds.keys())[:sample_n]
    n_hit = 0
    for iid in iids:
        obj = preds[iid]
        if not isinstance(obj, dict):
            text = str(obj)
        else:
            text = ""
            for k in ("raw_output", "output", "response", "text", "completion"):
                v = obj.get(k)
                if v:
                    text = str(v)
                    break
        if "</think>" in text.lower():
            n_hit += 1
    return n_hit / max(1, len(iids))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.strip().split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset", type=Path, required=True,
                    help="Path to braintrace_dataset/ (must contain data/test.parquet)")
    ap.add_argument("--predictions", type=Path, required=True,
                    help="Predictions JSONL or directory tree of per-item JSON")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Where to write scored outputs and summary.json")
    ap.add_argument("--mode", choices=["all", "closed", "open"], default="all")
    ap.add_argument("--skip-judge", action="store_true",
                    help="Alias for --mode closed (closed-form only)")
    ap.add_argument("--judge-model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--max-parallel", type=int, default=8)
    ap.add_argument("--reuse-judge-dir", type=Path, default=None,
                    help="Skip judge subprocess; re-aggregate from this directory")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    args = ap.parse_args(argv)

    if args.skip_judge:
        args.mode = "closed"

    parquet = args.dataset / "data" / "test.parquet"
    if not parquet.exists():
        print(f"[score] missing parquet: {parquet}", file=sys.stderr)
        return 2

    print(f"[score] loading dataset index from {parquet}")
    items = cf.load_parquet_index(parquet)
    print(f"[score] {len(items):,} items in dataset")

    print(f"[score] loading predictions from {args.predictions}")
    preds = cf.load_predictions(args.predictions)
    print(f"[score] {len(preds):,} predictions loaded")
    missing = sum(1 for iid in items if iid not in preds)
    if missing:
        print(f"[score] note: {missing} parquet items have no prediction "
              f"(excluded from denominators)")

    think_rate = _detect_thinking_rate(preds)
    if think_rate > 0.05:
        print(f"[score] inline <think> tag detected in ~{think_rate * 100:.1f}% "
              f"of sampled predictions; pre-</think> text will be stripped")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "n_items_in_parquet": len(items),
        "n_predictions_loaded": len(preds),
        "mode": args.mode,
        "judge_model": args.judge_model if args.mode != "closed" else None,
        "thinking_strip_applied": True,
        "scoring_spec": "BrainTRACE Scoring v1.0",
    }

    # ---- Closed-form -----------------------------------------------------
    if args.mode in ("all", "closed"):
        cf_report = cf.score_closed_form(items, preds, n_bootstrap=args.n_bootstrap)
        (args.out_dir / "scored_closed_form.jsonl").write_text(
            "\n".join(json.dumps(r) for r in cf_report["per_item"])
        )
        cf_summary = {
            "per_track_overall": cf_report["per_track_overall"],
            "per_track_subcat": cf_report["per_track_subcat"],
            "per_level": cf_report["per_level"],
        }
        (args.out_dir / "summary_closed_form.json").write_text(
            json.dumps(cf_summary, indent=2)
        )
        summary["closed_form"] = cf_summary
        print("[score] closed-form summary (per track):")
        for trk, s in cf_summary["per_track_overall"].items():
            print(f"  {trk:20s} n={s['n']:>5d}  acc={s['accuracy_pct']}%  "
                  f"CI95={s['ci95_pct']}  chance={s['chance_pct']}%  "
                  f"CDS={s['cds_pct']}%")

    # ---- Open-ended + chain ---------------------------------------------
    if args.mode in ("all", "open"):
        repo_root = Path(__file__).resolve().parent.parent
        judge_script = repo_root / "scoring" / "llm_judge.py"

        if args.reuse_judge_dir:
            judge_dir = args.reuse_judge_dir
        else:
            judge_dir = args.out_dir / "_llm_judge"
            judge_dir.mkdir(exist_ok=True)
            import tempfile
            with tempfile.TemporaryDirectory(prefix="braintrace_judge_") as tmp:
                tmp_p = Path(tmp)
                gt_root, q_root, out_dir, n = oj.materialise_judge_inputs(
                    items, preds, tmp_p, include_open=True, include_chain=True,
                )
                print(f"[score] materialised {n} items for LLM judge")
                if n == 0:
                    print("[score] no open-ended / chain items to judge")
                    judge_dir = None
                else:
                    rc = oj.run_llm_judge(
                        judge_script=judge_script,
                        gt_root=gt_root, questions_root=q_root,
                        outputs_dir=out_dir, out_dir=judge_dir,
                        judge_model=args.judge_model,
                        max_parallel=args.max_parallel,
                    )
                    if rc != 0:
                        print(f"[score] WARN: llm_judge exited rc={rc}")

        if judge_dir is not None:
            rows = oj._read_assessments(judge_dir)
            open_summary = oj.aggregate_open_ended(items, rows)
            chain_summary = oj.aggregate_chain(items, rows)
            judge_summary = {
                "judge_dir": str(judge_dir),
                "n_assessments": len(rows),
                "open_ended": open_summary,
                "chain": chain_summary,
            }
            (args.out_dir / "summary_open_judge.json").write_text(
                json.dumps(judge_summary, indent=2)
            )
            summary["open_ended"] = open_summary
            summary["chain"] = chain_summary

            if open_summary.get("by_track"):
                print("[score] open-ended summary (per track):")
                for trk, s in open_summary["by_track"].items():
                    print(f"  {trk:20s} n={s['n']:>4d}  "
                          f"item_pass={s['item_pass_rate_pct']}%  "
                          f"slot={s['mean_slot_match_rate']}  "
                          f"Q1-5={s['mean_quality_1_5']}  "
                          f"faith={s['faithfulness_composite']}")
            if chain_summary.get("n_items_judged", 0) > 0:
                print(f"[score] chain summary: "
                      f"step_pass={chain_summary['step_pass_pct']}%  "
                      f"CESR={chain_summary['cesr_pct']}%")

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[score] wrote {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
