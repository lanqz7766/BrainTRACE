#!/usr/bin/env python3
"""
BrainTRACE — case-level reasoning inference wrapper.

Iterates each case-level reasoning item, breaks it into 6 per-step inference calls,
delegates each step to an existing 2D-PNG or 3D-volume adapter
(`vlm_eval_qwen35.py`, `vlm_eval_radfm_2d.py`, `vlm_eval_m3d.py`, ...),
and aggregates the per-step outputs into a single chain output JSON.

Design:

- Per step k, the model only sees timepoints listed in
  ``chain_questions[k].timepoints_used``. No information leakage.
- A greedy cap-fit policy packs (TP, modality, slice-tag) PNG triples up to
  ``--max-images``; falls back through (drop btm → drop top → drop t1w →
  drop t2w → evenly subsample TPs) if the (mid, all-mods) baseline overshoots.
- The wrapper is non-invasive: it spawns the existing adapter as a subprocess
  via ``subprocess.run([sys.executable, args.adapter_script, ...])`` with a
  one-line JSONL sample-file pointing at a synthesized per-step question JSON.
  The adapter does its own model loading, prompt formatting, and output
  writing. We re-load its output JSON, copy the relevant fields into our
  chain schema, and clean up.

For 3D mode, instead of a PNG cap-fit pass we slice the source
``volume.npy`` along the (modality, timepoint) axes for the current step,
write a per-step ``volume.npy`` + ``volume_meta.json`` into a unique temp
directory, and pass that as the adapter's ``--threed-root`` so the existing
3D loader picks it up unchanged.

CLI flags:
  --sample-file       JSONL of case-level reasoning items
  --out-dir           Output dir for per-item chain JSONs
  --root              Project root (auto-detected from this script's dir)
  --images-root       Default <root>/benchmark/images
  --threed-root       Default <root>/benchmark/threed_v2_case_reasoning
  --adapter-script    Path to underlying adapter (e.g. scripts/vlm_eval_qwen35.py)
  --adapter-flags     Extra args, will be ``shlex.split``'d and passed through
  --max-images        Per-step image cap (default 16)
  --mode              "2d" (PNG) or "3d" (volume.npy)
  --skip-existing     Skip items whose chain output already exists
  --gpu-id            GPU id, forwarded to adapter if it accepts ``--gpu-id``

Output schema (per paper §6) — see ``aggregate_steps`` below.
"""

from __future__ import annotations

import argparse
import json
import shlex
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


# --- constants ---------------------------------------------------------------

SLICE_TAGS_ORDER = ("mid", "top", "btm")  # greedy pass order
MODALITY_DROP_ORDER = ("t1w", "t2w")       # dropped first-to-last; flair last
ALL_MODALITIES = ("t1w", "t2w", "flair")   # order for bookkeeping


# --- helpers: question / sample-file synthesis -------------------------------

def _tp_label_to_index(tp_label: str) -> int:
    """Convert "TP3" → 3; tolerant of "tp3" / "TP_3"."""
    s = tp_label.strip().upper().lstrip("TP").lstrip("_")
    return int(s)


def synth_per_step_question(question_json: dict, step: dict) -> dict:
    """
    Build a per-step question.json the existing adapter can consume.

    Mirrors the bulk_v1 schema:
      - ``model_text_input.rendered_question`` is set to the step's question
        (adapter's prompt builder reads this key).
      - ``model_text_input.options`` is forced to None (chain steps are open).
      - ``model_text_input.shown_metadata`` is preserved but
        ``shown_tp_labels`` / ``shown_relative_days_from_first_tp`` are
        restricted to the step's used TPs (so adapter prompt/metadata reflect
        the actual visible window).
      - ``model_image_spec`` is preserved verbatim (rule = multi_tp_key_slices).
      - ``backend_authoring`` retains tp_indices/study_dirnames *restricted to
        used TPs* so the 3D adapter's per-(M, T) loop stays consistent with
        what we sliced in the temp volume.

    The step's id / subclass / timepoints_used are echoed under a non-standard
    key ``_chain_step`` for downstream debugging only — adapters ignore it.
    """
    used_labels = list(step["timepoints_used"])
    used_idxs = [_tp_label_to_index(t) for t in used_labels]

    base_mti = dict(question_json["model_text_input"])
    md = dict(base_mti.get("shown_metadata", {}) or {})

    # Restrict shown_metadata to used TPs.
    orig_tp_labels = md.get("shown_tp_labels", []) or []
    orig_days = md.get("shown_relative_days_from_first_tp", []) or []
    label_to_pos = {t: i for i, t in enumerate(orig_tp_labels)}
    keep_pos = [label_to_pos[t] for t in used_labels if t in label_to_pos]
    if keep_pos:
        md["shown_tp_labels"] = [orig_tp_labels[p] for p in keep_pos]
        md["n_shown_tps"] = len(keep_pos)
        if len(orig_days) == len(orig_tp_labels):
            md["shown_relative_days_from_first_tp"] = [orig_days[p] for p in keep_pos]

    new_mti = {
        "answer_type": "open_ended",   # adapter's prompt builder treats as free-form
        "options": None,
        "rendered_question": step["question"],
        "shown_metadata": md,
    }

    new_qj = dict(question_json)
    new_qj["model_text_input"] = new_mti

    # Restrict backend_authoring to used TPs (for 3D adapter's M×T loop).
    ba = dict(question_json.get("backend_authoring", {}) or {})
    orig_tp_indices = ba.get("tp_indices", []) or []
    orig_study_dirs = ba.get("study_dirnames", []) or []
    if orig_tp_indices and orig_study_dirs:
        # Map by position in shown_tp_labels (not by tp_index value, since some
        # items may have non-contiguous TPs).
        if keep_pos and len(orig_tp_indices) == len(orig_tp_labels):
            ba["tp_indices"] = [orig_tp_indices[p] for p in keep_pos]
            ba["study_dirnames"] = [orig_study_dirs[p] for p in keep_pos]
            if "study_dates" in ba and len(ba["study_dates"]) == len(orig_tp_labels):
                ba["study_dates"] = [ba["study_dates"][p] for p in keep_pos]
    new_qj["backend_authoring"] = ba

    new_qj["_chain_step"] = {
        "step_id": step.get("step_id"),
        "subclass": step.get("subclass"),
        "timepoints_used": used_labels,
        "tp_indices_used": used_idxs,
    }
    return new_qj


# --- helpers: greedy 2D image packing ---------------------------------------

def _png_path(images_root: Path, item_id: str, tp_idx: int, mod: str, tag: str) -> Path:
    fname = f"TP{tp_idx}_{mod}_axial-{tag}.png"
    preferred = images_root / "case_reasoning" / item_id / fname
    if preferred.exists():
        return preferred
    legacy_name = "bulk_v2_" + "_".join(["L5", "5"])
    return images_root / legacy_name / item_id / fname


def greedy_pack_images(
    images_root: Path,
    item_id: str,
    used_tp_indices: list[int],
    modalities: list[str],
    cap: int,
    chain_anchor_tp_indices: list[int] | None = None,
) -> dict:
    """
    Pack PNGs for one chain step under the cap, recording which fallbacks
    triggered (for audit logging in the per-step output).

    Pack order:
      Pass 1: every (used_tp, mod, mid)
      Pass 2: every (used_tp, mod, top)
      Pass 3: every (used_tp, mod, btm)
    Stop the moment we hit ``cap``.

    Fallback ladder when even Pass 1 (mid, all mods, all used_tps) > cap:
      1. Drop modalities in order t1w → t2w (keep flair).
      2. If still over cap, evenly subsample TPs while preserving any in
         ``chain_anchor_tp_indices``.

    Returns a dict::
        {
            "image_paths": [<absolute paths>],
            "slice_levels_used": [...],     # subset of {mid, top, btm}
            "n_modalities_used": int,
            "n_tps_used": int,
            "modalities_used": [...],
            "tp_indices_used": [...],
            "dropped_modalities": [...],
            "subsampled_tps": bool,
        }
    """
    anchor = set(chain_anchor_tp_indices or [])

    def _list_existing(tps: list[int], mods: list[str], tag: str) -> list[Path]:
        out = []
        for tp in tps:
            for mod in mods:
                p = _png_path(images_root, item_id, tp, mod, tag)
                if p.exists():
                    out.append(p)
        return out

    tps_eff = list(used_tp_indices)
    mods_eff = [m for m in ALL_MODALITIES if m in modalities]
    if not mods_eff:
        mods_eff = list(modalities)
    dropped_mods: list[str] = []
    subsampled_tps = False

    # Step A: shrink "mid-only" baseline to fit cap.
    while len(_list_existing(tps_eff, mods_eff, "mid")) > cap:
        # 1) drop modalities in order t1w → t2w
        droppable = [m for m in MODALITY_DROP_ORDER if m in mods_eff and m != "flair"]
        if droppable and len(mods_eff) > 1:
            drop = droppable[0]
            mods_eff = [m for m in mods_eff if m != drop]
            dropped_mods.append(drop)
            continue
        # 2) evenly subsample TPs, keeping anchors first
        if len(tps_eff) <= 1:
            break  # cannot shrink further; final mid-only count > cap
        keep_n = max(1, cap // max(1, len(mods_eff)))
        kept_anchors = [t for t in tps_eff if t in anchor]
        non_anchors = [t for t in tps_eff if t not in anchor]
        budget = max(0, keep_n - len(kept_anchors))
        if budget > 0 and non_anchors:
            idx = np.linspace(0, len(non_anchors) - 1, budget).round().astype(int).tolist()
            picked = sorted({non_anchors[i] for i in idx})
        else:
            picked = []
        new_tps = sorted(set(kept_anchors) | set(picked))
        if not new_tps:
            new_tps = tps_eff[:keep_n]
        if new_tps == tps_eff:
            break  # cannot shrink further
        tps_eff = new_tps
        subsampled_tps = True

    # Step B: greedy multi-pass fill up to cap.
    paths: list[Path] = []
    levels_used: set[str] = set()
    for tag in SLICE_TAGS_ORDER:
        if len(paths) >= cap:
            break
        for tp in tps_eff:
            if len(paths) >= cap:
                break
            for mod in mods_eff:
                if len(paths) >= cap:
                    break
                p = _png_path(images_root, item_id, tp, mod, tag)
                if p.exists():
                    paths.append(p)
                    levels_used.add(tag)

    # Preserve a stable, audit-friendly slice-level order.
    levels_ordered = [t for t in SLICE_TAGS_ORDER if t in levels_used]
    return {
        "image_paths": [str(p) for p in paths],
        "slice_levels_used": levels_ordered,
        "n_modalities_used": len(mods_eff),
        "n_tps_used": len(tps_eff),
        "modalities_used": list(mods_eff),
        "tp_indices_used": list(tps_eff),
        "dropped_modalities": list(dropped_mods),
        "subsampled_tps": subsampled_tps,
    }


# --- helpers: 3D volume slicing ---------------------------------------------

def slice_volume_for_step(
    threed_root: Path,
    item_id: str,
    used_tp_indices: list[int],
    modalities_spec: list[str],
    out_dir: Path,
) -> dict:
    """
    Read ``<threed_root>/<item_id>/volume.npy`` and ``volume_meta.json``, slice
    along (modality, timepoint) axes per the step, and write a per-step
    ``<out_dir>/<item_id>/volume.npy`` + ``volume_meta.json`` so an unchanged
    3D adapter can be pointed at ``<out_dir>`` via ``--threed-root``.

    Returns a dict with ``modalities_used``, ``tp_indices_used``,
    ``n_modalities_used``, ``n_tps_used``, ``volume_path`` for audit.
    """
    src_dir = Path(threed_root) / item_id
    meta = json.loads((src_dir / "volume_meta.json").read_text())
    vol = np.load(src_dir / "volume.npy")  # (M, T, H, W, D) float16

    src_mods = list(meta.get("modalities_present") or meta.get("modalities") or [])
    src_tp_labels = list(meta["tp_labels"])

    # Modality intersection (preserve spec order).
    keep_mod_idx = [src_mods.index(m) for m in modalities_spec if m in src_mods]
    keep_mods = [src_mods[i] for i in keep_mod_idx]

    # TP index → position in source volume_meta.
    label_to_pos = {lbl: i for i, lbl in enumerate(src_tp_labels)}
    used_labels = [f"TP{i}" for i in used_tp_indices]
    keep_tp_idx = [label_to_pos[t] for t in used_labels if t in label_to_pos]
    keep_tp_labels = [src_tp_labels[i] for i in keep_tp_idx]

    if not keep_mod_idx or not keep_tp_idx:
        # Defensive: write empty meta but no volume so adapter can fail loud.
        raise RuntimeError(
            f"slice_volume_for_step: no overlap for {item_id}: "
            f"mods spec={modalities_spec} src={src_mods} ; "
            f"used_tps={used_labels} src={src_tp_labels}"
        )

    sub = vol[np.ix_(keep_mod_idx, keep_tp_idx)]  # (m', t', H, W, D)
    sub_dir = Path(out_dir) / item_id
    sub_dir.mkdir(parents=True, exist_ok=True)
    np.save(sub_dir / "volume.npy", sub.astype(np.float16))
    sub_meta = {
        "modalities": keep_mods,
        "tp_labels": keep_tp_labels,
        "source_item_id": item_id,
        "sliced_for_chain_step": True,
    }
    (sub_dir / "volume_meta.json").write_text(json.dumps(sub_meta, indent=2))

    return {
        "modalities_used": keep_mods,
        "tp_indices_used": list(used_tp_indices),
        "n_modalities_used": len(keep_mods),
        "n_tps_used": len(keep_tp_labels),
        "volume_path": str(sub_dir / "volume.npy"),
        "slice_levels_used": [],          # not applicable in 3D
        "image_paths": [str(sub_dir / "volume.npy")],
        "dropped_modalities": [m for m in modalities_spec if m not in keep_mods],
        "subsampled_tps": False,
    }


# --- helpers: subprocess invocation -----------------------------------------

def invoke_adapter(
    adapter_script: Path,
    sample_file: Path,
    adapter_out_dir: Path,
    extra_flags: list[str],
    mode: str,
    threed_root: Path | None,
    images_root: Path | None,
    gpu_id: int,
    root: Path,
) -> subprocess.CompletedProcess:
    """
    Launch the underlying adapter for a single one-line JSONL sample-file.

    Adds the universal flags (`--root`, `--sample-file`, `--out-dir`),
    plus mode-specific flags:
      - 2d: `--images-root <images_root>`
      - 3d: `--threed-root <threed_root>` (the per-step temp dir)
    Forwards `--gpu-id` (best-effort; not all adapters accept it but most do).
    Caller-supplied `extra_flags` (e.g. `--model`, `--max-tokens`,
    `--max-model-len`) are appended last so they can override defaults.
    """
    cmd: list[str] = [
        sys.executable,
        str(adapter_script),
        "--root", str(root),
        "--sample-file", str(sample_file),
        "--out-dir", str(adapter_out_dir),
    ]
    if mode == "2d":
        if images_root is not None:
            cmd += ["--images-root", str(images_root)]
    elif mode == "3d":
        if threed_root is not None:
            cmd += ["--threed-root", str(threed_root)]

    # Best-effort gpu-id pass-through. Some adapters accept `--gpu-id` flag
    # (vlm_eval_radfm_2d, vlm_eval_m3d), others use CUDA_VISIBLE_DEVICES env
    # (vlm_eval.py for Qwen2.5/3-VL, vlm_eval_qwen35.py for Qwen3.5). We
    # always set the env var (works for both) and only add the CLI flag when
    # the caller explicitly passes it through --adapter-flags.
    cmd += list(extra_flags)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def parse_adapter_output(adapter_out_dir: Path, item_id: str, template: str) -> dict | None:
    """
    Locate and parse the adapter's per-item output JSON.

    Adapters write to ``<out_dir>/<template>/<item_id>.output.json`` (qwen35,
    radfm_2d, m3d, ...). Some adapters write a flat path; we fall back to that.
    """
    candidates = [
        adapter_out_dir / template / f"{item_id}.output.json",
        adapter_out_dir / f"{item_id}.output.json",
    ]
    for c in candidates:
        if c.exists():
            try:
                return json.loads(c.read_text())
            except Exception:
                continue
    # Last resort: any file named <item_id>.output.json under out_dir.
    for p in adapter_out_dir.rglob(f"{item_id}.output.json"):
        try:
            return json.loads(p.read_text())
        except Exception:
            continue
    return None


# --- per-step driver ---------------------------------------------------------

def run_one_step(
    *,
    item_id: str,
    template: str,
    level: int,
    step_idx: int,
    step: dict,
    question_json: dict,
    args: argparse.Namespace,
    root: Path,
    tmp_root: Path,
) -> dict:
    """
    Drive one chain step end-to-end: build images / slice volume, write
    per-step question.json and one-line sample-file, invoke the adapter,
    parse output, return a normalized step record.
    """
    step_id = step.get("step_id", f"step_{step_idx + 1}")
    subclass = step.get("subclass")
    used_labels = list(step.get("timepoints_used") or [])
    used_idxs = [_tp_label_to_index(t) for t in used_labels]

    spec = question_json.get("model_image_spec", {}) or {}
    spec_mods = list(spec.get("input_modalities") or ALL_MODALITIES)
    ba = question_json.get("backend_authoring", {}) or {}
    anchor = ba.get("chain_anchor_tp_indices") or []

    workdir = tmp_root / f"_chain_step_{item_id}_{step_id}"
    workdir.mkdir(parents=True, exist_ok=True)
    adapter_out = workdir / "out"
    adapter_out.mkdir(parents=True, exist_ok=True)

    pack_info: dict[str, Any]
    threed_temp_root: Path | None = None
    images_root_eff: Path | None = None

    if args.mode == "2d":
        images_root_eff = Path(args.images_root) if args.images_root else (root / "benchmark" / "images")
        pack_info = greedy_pack_images(
            images_root=images_root_eff,
            item_id=item_id,
            used_tp_indices=used_idxs,
            modalities=spec_mods,
            cap=args.max_images,
            chain_anchor_tp_indices=anchor,
        )
        # Bridge case-level reasoning PNG layout to the adapter's expected per-template layout.
        # which existing 2D adapters' load_canonical_images expects. Symlink the
        # selected PNGs into a per-step temp tree so the adapter resolves them
        # without any adapter-side patch.
        adapter_img_root = workdir / "img_root"
        adapter_layout_dir = adapter_img_root / "bulk_v1" / template / item_id
        adapter_layout_dir.mkdir(parents=True, exist_ok=True)
        for src_path_str in pack_info.get("image_paths", []):
            src = Path(src_path_str)
            if not src.is_absolute():
                src = (root / src).resolve()
            dst = adapter_layout_dir / src.name
            if dst.exists() or dst.is_symlink():
                continue
            try:
                dst.symlink_to(src)
            except OSError:
                shutil.copy2(src, dst)
        images_root_eff = adapter_img_root
    elif args.mode == "3d":
        if args.threed_root:
            threed_src = Path(args.threed_root)
        else:
            threed_src = root / "benchmark" / "threed_v2_case_reasoning"
            legacy_src = root / "benchmark" / ("threed_v2_" + "_".join(["L5", "5"]))
            if not threed_src.exists() and legacy_src.exists():
                threed_src = legacy_src
        threed_temp_root = workdir / "threed"
        threed_temp_root.mkdir(parents=True, exist_ok=True)
        pack_info = slice_volume_for_step(
            threed_root=threed_src,
            item_id=item_id,
            used_tp_indices=used_idxs,
            modalities_spec=spec_mods,
            out_dir=threed_temp_root,
        )
    else:
        raise ValueError(f"unknown --mode: {args.mode}")

    # Synthesize per-step question.json + one-line JSONL sample-file.
    step_qj = synth_per_step_question(question_json, step)
    step_qj_path = workdir / f"{item_id}.{step_id}.question.json"
    step_qj_path.write_text(json.dumps(step_qj, indent=2))
    sample_path = workdir / "sample.jsonl"
    sample_path.write_text(
        json.dumps({
            "item_id": item_id,
            "template": template,
            "level": level,
            "question_path": str(step_qj_path),
            "gt_path": "",   # not needed for inference
        }) + "\n"
    )

    extra_flags = shlex.split(args.adapter_flags) if args.adapter_flags else []
    # Force --max-images on the adapter to match our packer's cap (some
    # adapters subsample internally otherwise).
    if "--max-images" not in extra_flags:
        extra_flags = ["--max-images", str(args.max_images)] + extra_flags

    t0 = time.time()
    proc = invoke_adapter(
        adapter_script=Path(args.adapter_script),
        sample_file=sample_path,
        adapter_out_dir=adapter_out,
        extra_flags=extra_flags,
        mode=args.mode,
        threed_root=threed_temp_root,
        images_root=images_root_eff,
        gpu_id=args.gpu_id,
        root=root,
    )
    wall_ms = int((time.time() - t0) * 1000)

    raw_out = parse_adapter_output(adapter_out, item_id, template)
    if raw_out is None:
        # Surface adapter stderr tail to ease debugging.
        tail = (proc.stderr or "").splitlines()[-20:]
        return {
            "step_id": step_id,
            "subclass": subclass,
            "raw_output": None,
            "n_output_tokens": 0,
            "n_images": len(pack_info.get("image_paths", [])),
            "image_paths": pack_info.get("image_paths", []),
            "wall_time_ms": wall_ms,
            "n_modalities_used": pack_info.get("n_modalities_used", 0),
            "n_tps_used": pack_info.get("n_tps_used", 0),
            "slice_levels_used": pack_info.get("slice_levels_used", []),
            "dropped_modalities": pack_info.get("dropped_modalities", []),
            "subsampled_tps": pack_info.get("subsampled_tps", False),
            "step_pass": False,
            "skip_reason": "adapter_no_output",
            "stderr_tail": "\n".join(tail),
        }

    return {
        "step_id": step_id,
        "subclass": subclass,
        "raw_output": raw_out.get("raw_output"),
        "n_output_tokens": int(raw_out.get("n_output_tokens", 0) or 0),
        "n_images": int(raw_out.get("n_images", len(pack_info.get("image_paths", []))) or 0),
        "image_paths": pack_info.get("image_paths", []),
        "wall_time_ms": int(raw_out.get("wall_time_ms", wall_ms) or wall_ms),
        "n_modalities_used": pack_info.get("n_modalities_used", 0),
        "n_tps_used": pack_info.get("n_tps_used", 0),
        "slice_levels_used": pack_info.get("slice_levels_used", []),
        "dropped_modalities": pack_info.get("dropped_modalities", []),
        "subsampled_tps": pack_info.get("subsampled_tps", False),
    }


# --- per-item driver ---------------------------------------------------------

def aggregate_steps(item_id: str, template: str, level: int, steps_out: list[dict]) -> dict:
    """Build the final per-item chain output JSON (paper §6 schema)."""
    n_imgs = sum(int(s.get("n_images") or 0) for s in steps_out)
    wall_total = sum(int(s.get("wall_time_ms") or 0) for s in steps_out)
    completed = all(s.get("raw_output") is not None for s in steps_out)
    skip_reason = None
    if not completed:
        first_bad = next((s for s in steps_out if s.get("raw_output") is None), None)
        if first_bad:
            skip_reason = first_bad.get("skip_reason") or "adapter_no_output"
    return {
        "item_id": item_id,
        "template": template,
        "level": level,
        "track": "case_reasoning",
        "steps": steps_out,
        "all_steps_completed": completed,
        "skip_reason": skip_reason,
        "n_images_total": n_imgs,
        "wall_time_ms_total": wall_total,
    }


def run_one_item(
    rec: dict,
    args: argparse.Namespace,
    root: Path,
    tmp_root: Path,
    out_dir: Path,
) -> Path | None:
    item_id = rec["item_id"]
    template = rec.get("template", "case_reasoning")
    level = int(rec.get("level", 5))
    final_path = out_dir / f"{item_id}.output.json"
    if args.skip_existing and final_path.exists():
        print(f"[skip] {item_id}: chain output already exists", flush=True)
        return final_path

    qpath = Path(rec["question_path"])
    if not qpath.is_absolute():
        qpath = root / qpath
    try:
        question_json = json.loads(qpath.read_text())
    except Exception as e:
        print(f"[err] {item_id}: cannot read question_path={qpath}: {e}", flush=True)
        return None

    chain = (question_json.get("model_text_input", {}) or {}).get("chain_questions") or []
    if not chain:
        print(f"[err] {item_id}: no chain_questions in question.json", flush=True)
        return None

    # Inject GT subclass if available (so steps can be tagged for analysis).
    gt_subclass_by_id: dict[str, str] = {}
    gt_path_str = rec.get("gt_path") or ""
    if gt_path_str:
        gt_path = Path(gt_path_str)
        if not gt_path.is_absolute():
            gt_path = root / gt_path
        if gt_path.exists():
            try:
                gt = json.loads(gt_path.read_text())
                for sg in gt.get("step_ground_truth", []) or []:
                    sid = sg.get("step_id")
                    sc = sg.get("subclass")
                    if sid and sc:
                        gt_subclass_by_id[sid] = sc
            except Exception as e:
                print(f"[warn] {item_id}: gt parse failed: {e}", flush=True)

    steps_out: list[dict] = []
    for k, step in enumerate(chain):
        step = dict(step)
        if step.get("step_id") in gt_subclass_by_id and "subclass" not in step:
            step["subclass"] = gt_subclass_by_id[step["step_id"]]
        try:
            srec = run_one_step(
                item_id=item_id,
                template=template,
                level=level,
                step_idx=k,
                step=step,
                question_json=question_json,
                args=args,
                root=root,
                tmp_root=tmp_root,
            )
        except Exception as e:
            print(f"[err] {item_id} {step.get('step_id')}: {e}", flush=True)
            srec = {
                "step_id": step.get("step_id", f"step_{k + 1}"),
                "subclass": step.get("subclass"),
                "raw_output": None,
                "n_output_tokens": 0,
                "n_images": 0,
                "image_paths": [],
                "wall_time_ms": 0,
                "n_modalities_used": 0,
                "n_tps_used": 0,
                "slice_levels_used": [],
                "dropped_modalities": [],
                "subsampled_tps": False,
                "skip_reason": f"wrapper_exception: {type(e).__name__}",
            }
        steps_out.append(srec)
        print(
            f"[step] {item_id} {srec['step_id']} "
            f"n_imgs={srec['n_images']} mods={srec['n_modalities_used']} "
            f"tps={srec['n_tps_used']} levels={srec['slice_levels_used']} "
            f"dropped={srec['dropped_modalities']} sub={srec['subsampled_tps']} "
            f"wall={srec['wall_time_ms']}ms",
            flush=True,
        )

    aggregated = aggregate_steps(item_id, template, level, steps_out)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps(aggregated, indent=2))
    print(
        f"[done] {item_id}: "
        f"completed={aggregated['all_steps_completed']} "
        f"n_images_total={aggregated['n_images_total']} "
        f"wall_total={aggregated['wall_time_ms_total']}ms",
        flush=True,
    )
    return final_path


# --- CLI / main --------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = str(Path(__file__).resolve().parent.parent)
    ap.add_argument("--sample-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--root", default=default_root)
    ap.add_argument("--images-root", default=None)
    ap.add_argument("--threed-root", default=None)
    ap.add_argument("--adapter-script", required=True)
    ap.add_argument("--adapter-flags", default="")
    ap.add_argument("--max-images", type=int, default=16)
    ap.add_argument("--mode", choices=["2d", "3d"], default="2d")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--keep-temp", action="store_true",
                    help="Keep per-step temp dirs (default: delete on success)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(tempfile.mkdtemp(prefix="chain_inf_", dir="/tmp"))
    print(f"[info] tmp_root={tmp_root} ; out_dir={out_dir} ; mode={args.mode}", flush=True)

    items: list[dict] = []
    sample_path = Path(args.sample_file)
    with sample_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"[info] {len(items)} items loaded from {sample_path}", flush=True)

    n_ok = 0
    n_fail = 0
    t0 = time.time()
    for i, rec in enumerate(items):
        print(f"[item] ({i + 1}/{len(items)}) {rec.get('item_id')}", flush=True)
        try:
            res = run_one_item(rec=rec, args=args, root=root, tmp_root=tmp_root, out_dir=out_dir)
            if res is None:
                n_fail += 1
            else:
                n_ok += 1
        except Exception as e:
            print(f"[err] {rec.get('item_id')}: top-level exception: {e}", flush=True)
            n_fail += 1

    if not args.keep_temp:
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass

    dt = time.time() - t0
    print(f"[summary] ok={n_ok} fail={n_fail} elapsed={dt:.1f}s", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
