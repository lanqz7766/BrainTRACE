#!/usr/bin/env python3
"""Unified BrainTRACE image renderer.

Inputs:
  - patients/: output of benchmark/extraction/extract_from_mr_rate.py
  - questions/: question.json files with `model_image_spec` rendering recipes
  - labels/seg_volumes/01_label_map_brain.csv (for tri_plane_structure_centric)

Output:
  - benchmark/images/<item_id>/<mr_rate_nifti_basename>_<render_plane>.png
  - benchmark/images/<item_id>/<basename>_<plane>_centroid-<StructureLabel>.png
      (tri_plane_structure_centric)
  - benchmark/images/<item_id>/<basename>_MIP-<axis>.png  (full_3d_volume)
  - benchmark/images/_manifest.sha256   (when --manifest)

Determinism / naming contract (v0.2):
  - PNG filename = source NIfTI basename (stripping '.nii.gz') + '_' + render-plane.
    Example: YLLNUBJLP3_coreg_flair-raw-sag.nii.gz @ render-plane=axial ->
             YLLNUBJLP3_coreg_flair-raw-sag_axial.png.
    This inherits MR-RATE provenance and removes the old pilot convention
    (<modality>_<tp_label>_<plane>.png) which discarded which raw file was used.
  - PIL.Image.save(..., optimize=True) — bit-exact match to v0.1 pilot cache
    is INTENTIONALLY broken; pilot cache will be re-rendered in full Phase-2.
  - Per-slice intensity window = 1st / 99th percentile over strictly-positive
    voxels; fall back to (min, max) if no positive voxel. sl = np.rot90(sl).
    Then thumbnail to <=512x512 with Image.LANCZOS, mode='L'.

Sharding (v0.2):
  --shard-idx / --n-shards let N parallel workers render disjoint items.
  Sharding key: blake2b(item_id) % n_shards == shard_idx. item-level so each
  shard owns a full per-item dir; no write collisions across shards.

Version history:
  - v0.1 (2026-04-20): bit-exact reproduction of pilot cache
    (<modality>_<tp_label>_<plane>.png, optimize off).
  - v0.2 (2026-04-20): new naming from MR-RATE basename, optimize=True,
    true tri_plane_structure_centric via nvseg-ctmr-brain centroids,
    shard-idx/n-shards CLI.
  - v0.2.1 (2026-04-20): output path now follows canonical benchmark/images
    hierarchy (bulk_v1/<L?_?>/<item_id>/ and flagship/<item_id>/).

Script replaces: scripts/prepare_images.py + scripts/prerender_images.py.
Downstream: external users run this after extract_from_mr_rate.py to reproduce
benchmark/images/.

Observed spec/rule inventory (bulk_v1 + flagship, 2026-04-20):
  median_axial_single            1020
  median_axial_per_tp            1896
  tri_plane_key_slices           2029
  multi_plane_stack               212
  median_slice_of_primary_plane   200
  full_3d_volume                  410
  tri_plane_structure_centric      24   (stage1 + stage3 each have 12 copies)

Deviations / defaults flagged for PI review:
  - Structure->label map (v0.2): derived from
    labels/seg_volumes/01_label_map_brain.csv, LUMIR labels 214-345.
    Structure-name matching is exact string match (case-sensitive) on
    `structure_name`. If match fails, we warn and skip that structure.
  - Short CamelCase label (filename): rule = strip '-', '_', ' ', '.',
    collapse consecutive '-'/'_' boundaries into CamelCase. Examples:
        Left-Hippocampus         -> LeftHippocampus
        Right-Lateral-Ventricle  -> RightLateralVentricle
        Brain-Stem               -> BrainStem
        3rd-Ventricle            -> 3rdVentricle
    The PI asked whether "LHippocampus" (single L/R letter) was preferred.
    We default to full "Left"/"Right" to avoid losing information; flip via
    constants at top of the module.
  - tri_plane_structure_centric source image: spec says
    `input_modalities=['t1w']` and `input_planes=['axial','coronal','sagittal']`.
    We fetch a t1w file matching each requested render-plane suffix (axi/cor/sag)
    AND compute the centroid from the brain-segmentation volume (in voxel-space
    of the same study's *_nvseg-ctmr-brain.nii.gz, which is native-aligned to
    the coreg T1 stack, all with identical affine and shape). If the source
    file's shape differs from the seg shape we fall back to geometric center.
  - specific_slice_index applies to the primary plane (input_planes[0]); unseen
    in pilot; untested.
  - full_3d_volume MIP: filename becomes <basename>_MIP-<axis>.png where basename
    is the primary (first-found) t1w/flair/etc NIfTI; we emit 3 MIPs per modality
    from the SAME source file.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image

from render_volume_utils import build_item_volume, save_item

# --------------------------------------------------------------------------
# Config / constants
# --------------------------------------------------------------------------
# Brain label CSV, shipped with the benchmark (produced by build_label_map.py)
DEFAULT_LABEL_MAP_CSV = Path(
    "labels/seg_volumes/01_label_map_brain.csv"
)

MOD_PATTERNS = {
    "t1w":   [r"_t1w-raw-", r"_coreg_t1w-raw-"],
    "t2w":   [r"_t2w-raw", r"_coreg_t2w-raw"],
    "flair": [r"_flair-raw", r"_coreg_flair-raw"],
    "swi":   [r"_swi-raw", r"_coreg_swi-raw"],
}
PLANE_SUFFIX = {
    "axial":    ["-axi"],
    "sagittal": ["-sag"],
    "coronal":  ["-cor"],
    "oblique":  ["-obl"],
}


# --------------------------------------------------------------------------
# Structure-name helpers
# --------------------------------------------------------------------------
def _load_label_map(csv_path: Path):
    """Return dict {structure_name_lowercase: int_label_index} for LUMIR brain labels."""
    out = {}
    if not csv_path.is_file():
        return out
    with csv_path.open() as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 3:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            name = parts[1].strip()
            observed = parts[3].strip().lower() if len(parts) > 3 else ""
            if observed != "true":
                continue
            out[name.lower()] = idx
    return out


def _structure_camel(name: str) -> str:
    """Map 'Left-Hippocampus' -> 'LeftHippocampus', '3rd-Ventricle' -> '3rdVentricle'.

    Rule: split on any of '-', '_', ' ', '.', then CamelCase-join. We keep the
    first char of each token as-is (so '3rd' stays '3rd'). If a token is already
    all lowercase we title-case it; otherwise preserve it. Drop empty tokens.
    """
    parts = re.split(r"[-_\s\.]+", name.strip())
    parts = [p for p in parts if p]
    out = []
    for p in parts:
        # If the token has any lowercase and is not already CamelCase-like, title-case it
        if p.islower() or (p[:1].isalpha() and p[:1].islower()):
            out.append(p[:1].upper() + p[1:])
        else:
            out.append(p)
    return "".join(out) if out else "Unknown"


# --------------------------------------------------------------------------
# Path / NIfTI discovery
# --------------------------------------------------------------------------
def find_nifti(patient_dir: Path, study_dirname: str, modality: str, plane: str):
    """Return the first NIfTI matching modality + plane for this study, else None."""
    study = patient_dir / study_dirname
    for sub in ("image_center_coreg", "images_coreg_optional"):
        d = study / sub
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if not f.name.endswith(".nii.gz"):
                continue
            name = f.name.lower()
            if not any(re.search(p, name) for p in MOD_PATTERNS.get(modality, [])):
                continue
            if any(s in name for s in PLANE_SUFFIX.get(plane, [])):
                return f
    return None


def find_brain_seg(patient_dir: Path, study_dirname: str):
    """Return the first segmentation_brain/*_nvseg-ctmr-brain.nii.gz, else None."""
    d = patient_dir / study_dirname / "segmentation_brain"
    if not d.is_dir():
        return None
    for f in sorted(d.iterdir()):
        if f.name.endswith("_nvseg-ctmr-brain.nii.gz"):
            return f
    return None


def _nii_basename(nii_path: Path) -> str:
    """Strip '.nii.gz' from a filename, e.g. 'YLLNUBJLP3_coreg_flair-raw-sag'."""
    name = nii_path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


# --------------------------------------------------------------------------
# Slice + MIP writers
# --------------------------------------------------------------------------
def _slice_index_for_axis(arr, axis, override=None):
    if override is not None:
        return int(override)
    if axis == "axial":
        return arr.shape[2] // 2
    if axis == "sagittal":
        return arr.shape[0] // 2
    if axis == "coronal":
        return arr.shape[1] // 2
    return arr.shape[2] // 2


def _extract_slice(arr, axis, idx):
    if axis == "axial":
        return arr[:, :, idx]
    if axis == "sagittal":
        return arr[idx, :, :]
    if axis == "coronal":
        return arr[:, idx, :]
    return arr[:, :, idx]


def _to_png(arr2d: np.ndarray, out_path: Path) -> None:
    pos = arr2d > 0
    if pos.any():
        lo, hi = np.percentile(arr2d[pos], [1, 99])
    else:
        lo, hi = arr2d.min(), arr2d.max()
    sl = np.clip((arr2d - lo) / max(hi - lo, 1e-6), 0, 1)
    sl = np.rot90(sl)
    sl = (sl * 255).astype(np.uint8)
    im = Image.fromarray(sl, mode="L")
    im.thumbnail((512, 512), Image.LANCZOS)
    # v0.2: optimize=True (v0.1 used default off, bit-exact match with pilot)
    im.save(out_path, optimize=True)


def save_slice(nii_path: Path, out_path: Path, axis: str, slice_index=None) -> None:
    img = nib.load(str(nii_path))
    arr = img.get_fdata()
    while arr.ndim > 3:
        arr = arr[..., 0]
    idx = _slice_index_for_axis(arr, axis, override=slice_index)
    sl = _extract_slice(arr, axis, idx)
    _to_png(sl, out_path)


def save_mip(nii_path: Path, out_path: Path, axis: str) -> None:
    img = nib.load(str(nii_path))
    arr = img.get_fdata()
    while arr.ndim > 3:
        arr = arr[..., 0]
    if axis == "axial":
        mip = arr.max(axis=2)
    elif axis == "sagittal":
        mip = arr.max(axis=0)
    elif axis == "coronal":
        mip = arr.max(axis=1)
    else:
        mip = arr.max(axis=2)
    _to_png(mip, out_path)


# --------------------------------------------------------------------------
# tri_plane_structure_centric centroid computation
# --------------------------------------------------------------------------
def _structure_centroid(seg_path: Path, label_index: int):
    """Return (ci, cj, ck) voxel-space centroid of `label_index` within seg_path,
    or None if the label is absent (empty mask).
    """
    img = nib.load(str(seg_path))
    arr = img.get_fdata()
    while arr.ndim > 3:
        arr = arr[..., 0]
    # The seg files are stored as integer label volumes; fdata returns float.
    # Tolerate tiny float noise with a tight eq.
    mask = np.abs(arr - label_index) < 0.5
    if not mask.any():
        return None
    coords = np.argwhere(mask)
    c = coords.mean(axis=0)
    return tuple(int(round(x)) for x in c)


def save_slice_at_index(nii_path: Path, out_path: Path, axis: str, voxel_idx: int) -> None:
    """Extract and save a slice at voxel_idx along the given axis."""
    img = nib.load(str(nii_path))
    arr = img.get_fdata()
    while arr.ndim > 3:
        arr = arr[..., 0]
    # Clamp index into range (tolerate shape mismatch vs seg)
    if axis == "axial":
        voxel_idx = max(0, min(arr.shape[2] - 1, voxel_idx))
    elif axis == "sagittal":
        voxel_idx = max(0, min(arr.shape[0] - 1, voxel_idx))
    elif axis == "coronal":
        voxel_idx = max(0, min(arr.shape[1] - 1, voxel_idx))
    sl = _extract_slice(arr, axis, voxel_idx)
    _to_png(sl, out_path)


def save_multi_slice_contiguous(nii_path: Path, out_dir: Path, basename: str, n: int = 16) -> list:
    """Render N evenly-spaced axial slices and write one PNG per slice.

    Output filenames: <basename>_axial_slice_<NN>.png  (NN zero-padded to 2 digits).
    Uses the same intensity-windowing and thumbnail conventions as save_slice().
    Returns a list of Path objects for written files.
    """
    img = nib.load(str(nii_path))
    arr = img.get_fdata()
    while arr.ndim > 3:
        arr = arr[..., 0]
    depth = arr.shape[2]
    # Evenly space n indices across the axial dimension.
    indices = [int(round(i * (depth - 1) / max(n - 1, 1))) for i in range(n)]
    written = []
    for seq_num, idx in enumerate(indices):
        sl = _extract_slice(arr, "axial", idx)
        out_path = out_dir / f"{basename}_axial_slice_{seq_num:02d}.png"
        _to_png(sl, out_path)
        written.append(out_path)
    return written


# --------------------------------------------------------------------------
# Renderer per-item
# --------------------------------------------------------------------------
def _plan_standard(spec, ba, tp_labels):
    """Yield (modality, tp_label, study_dirname, render_plane, is_mip,
    slice_override) for non-structure-centric rules."""
    rule = spec["slice_selection_rule"]
    modalities = spec.get("input_modalities", [])
    planes = spec.get("input_planes", [])
    spec_idx = spec.get("specific_slice_index")
    tp_indices = ba.get("tp_indices", [])
    study_dirs = ba.get("study_dirnames", [])

    if rule == "full_3d_volume":
        for (tp_idx, study_dir, tp_label) in zip(tp_indices, study_dirs, tp_labels):
            for mod in modalities:
                for axis in ("axial", "sagittal", "coronal"):
                    yield mod, tp_label, study_dir, axis, True, None
        return

    for i, (tp_idx, study_dir, tp_label) in enumerate(zip(tp_indices, study_dirs, tp_labels)):
        for mod in modalities:
            for plane in planes:
                if rule in ("median_axial_single", "median_axial_per_tp") and plane != "axial":
                    continue
                if rule == "median_axial_single" and i > 0:
                    continue
                override = None
                if spec_idx is not None and plane == (planes[0] if planes else None):
                    override = spec_idx
                yield mod, tp_label, study_dir, plane, False, override


_BULK_V1_RE = re.compile(r"^bulk_v1_(L\d+_\d+)_\d+$")
_BULK_V2_RE = re.compile(r"^bulk_v2_(L\d+_\d+)_\d+$")
_THREED_V1_RE = re.compile(r"^threed_v1_(L\d+_\d+)_\d+$")
_THREED_V2_RE = re.compile(r"^threed_v2_(L\d+_\d+)_\d+$")


def item_subdir(item_id: str) -> str:
    """Map item_id -> relative output sub-path under out_root (canonical hierarchy v0.2.1).

    - bulk_v1_L?_?_<n>   -> 'bulk_v1/<L?_?>/<item_id>'
    - bulk_v2_L?_?_<n>   -> 'bulk_v2_<L?_?>/<item_id>'  (v0.3 chain track)
    - flagship_v1_...    -> 'flagship/<item_id>'
    - otherwise          -> '<item_id>'  (fallback, warns)
    """
    m = _BULK_V1_RE.match(item_id)
    if m:
        return f"bulk_v1/{m.group(1)}/{item_id}"
    m3d = _THREED_V1_RE.match(item_id)
    if m3d:
        return f"threed_v1/{m3d.group(1)}/{item_id}"
    m3d_v2 = _THREED_V2_RE.match(item_id)
    if m3d_v2:
        return f"threed_v2/{m3d_v2.group(1)}/{item_id}"
    m2 = _BULK_V2_RE.match(item_id)
    if m2:
        return f"bulk_v2_{m2.group(1)}/{item_id}"
    if item_id.startswith("flagship_v1_"):
        return f"flagship/{item_id}"
    print(f"[warn] item_subdir: unrecognized item_id prefix '{item_id}'; "
          f"using flat fallback <out_root>/{item_id}", flush=True)
    return item_id


def threed_subdir(item_id: str) -> str:
    """Map 3D-capable item_id -> benchmark-relative volume output subdir.

    Returns None if item_id should not dual-emit a volume.
    """
    m3d = _THREED_V1_RE.match(item_id)
    if m3d:
        return f"threed_v1/{item_id}"
    m3d_v2 = _THREED_V2_RE.match(item_id)
    if m3d_v2:
        return f"threed_v2/{item_id}"
    m2 = _BULK_V2_RE.match(item_id)
    if m2:
        return f"threed_v2_{m2.group(1)}/{item_id}"
    return None


def benchmark_root_from_out_root(out_root: Path) -> Path:
    """Infer benchmark root when PNGs are rooted under benchmark/images."""
    if out_root.name == "images":
        return out_root.parent
    return out_root


# --------------------------------------------------------------------------
# multi_tp_key_slices helpers (case-level reasoning track)
# --------------------------------------------------------------------------
# Three depth-fraction slices per (TP, modality, axial)
_KEY_SLICE_FRACTIONS = (("top", 0.25), ("mid", 0.5), ("btm", 0.75))
# Trilinear-resampled volume target for 3D-native baselines
_VOL_TARGET_SHAPE = (64, 128, 128)  # (D, H, W)


def _load_volume_3d(nii_path: Path) -> np.ndarray:
    """Load NIfTI, squeeze to 3D float array. Returns (X, Y, Z)."""
    img = nib.load(str(nii_path))
    arr = img.get_fdata()
    while arr.ndim > 3:
        arr = arr[..., 0]
    return arr


def _normalize_volume_pos_pct(arr: np.ndarray) -> np.ndarray:
    """Normalize volume to [0, 1] using 1st/99th percentile of strictly-positive
    voxels, consistent with PNG per-slice windowing.
    """
    pos = arr > 0
    if pos.any():
        lo, hi = np.percentile(arr[pos], [1, 99])
    else:
        lo, hi = float(arr.min()), float(arr.max())
    out = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return out


def _trilinear_resize_to_target(arr: np.ndarray, target=_VOL_TARGET_SHAPE) -> np.ndarray:
    """Resize a 3D volume (X, Y, Z) -> (D, H, W) using torch trilinear interpolation.

    Note: input arr axes are (X, Y, Z) per nibabel convention; we treat the
    nibabel z-axis as 'depth' (axial slice axis) so that target depth (D=64)
    aligns with the axial slicing direction used for PNGs. Output axis order
    is (D, H, W) where D corresponds to nibabel's z-axis.
    """
    import torch  # lazy import: only needed for bulk_v2 dual-emit
    # Reorder from (X, Y, Z) -> (D=Z, H=Y, W=X) so D is the axial axis.
    src = np.transpose(arr, (2, 1, 0)).astype(np.float32)
    t = torch.from_numpy(src)[None, None, ...]  # (1, 1, D, H, W)
    out = torch.nn.functional.interpolate(
        t, size=tuple(target), mode="trilinear", align_corners=False
    )[0, 0].cpu().numpy()
    return out


def save_key_slices(nii_path: Path, out_paths_by_tag: dict) -> list:
    """Render axial slices at depth fractions 0.25/0.5/0.75 and save to provided paths.

    out_paths_by_tag: {"top": Path, "mid": Path, "btm": Path}
    Returns list of written Paths.
    """
    arr = _load_volume_3d(nii_path)
    depth = arr.shape[2]
    written = []
    for tag, frac in _KEY_SLICE_FRACTIONS:
        out_path = out_paths_by_tag.get(tag)
        if out_path is None:
            continue
        idx = int(round(frac * (depth - 1)))
        idx = max(0, min(depth - 1, idx))
        sl = _extract_slice(arr, "axial", idx)
        _to_png(sl, out_path)
        written.append(out_path)
    return written


def _handle_multi_tp_key_slices(question_json: dict, patient_dir: Path,
                                out_root: Path, only_missing: bool,
                                item_dir_override: Path = None,
                                volume_dir_override: Path = None):
    """Render `multi_tp_key_slices` PNGs and emit `volume.npy` for bulk_v2 items.

    Output layout:
      <out_root>/bulk_v2_<Lx_y>/<item_id>/TP{i}_{mod}_axial-{tag}.png
      <out_root>/threed_v2_<Lx_y>/<item_id>/volume.npy
      <out_root>/threed_v2_<Lx_y>/<item_id>/volume_meta.json

    Returns (item_id, n_written, n_existing, errors).
    """
    item_id = question_json["item_id"]
    spec = question_json["model_image_spec"]
    ba = question_json["backend_authoring"]
    tp_labels = question_json["model_text_input"]["shown_metadata"]["shown_tp_labels"]
    modalities = spec.get("input_modalities") or ["t1w"]
    study_dirs = ba.get("study_dirnames", [])

    item_dir = item_dir_override if item_dir_override is not None else out_root / item_subdir(item_id)
    item_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_existing = 0
    errors = []

    # Per-(tp, mod) NIfTI cache, reused for PNG + 3D emit
    nii_by_tp_mod: dict = {}
    skipped_modalities_per_tp: list = []

    for i, study_dir in enumerate(study_dirs):
        tp_label = f"TP{i}"  # canonical TP index label, independent of shown_tp_labels mapping
        skipped_here = []
        for mod in modalities:
            nii = find_nifti(patient_dir, study_dir, mod, "axial")
            if nii is None:
                skipped_here.append(mod)
                continue
            nii_by_tp_mod[(i, mod)] = nii
            out_paths = {
                tag: item_dir / f"{tp_label}_{mod}_axial-{tag}.png"
                for tag, _frac in _KEY_SLICE_FRACTIONS
            }
            # Skip rendering if all targets already exist (only_missing)
            if only_missing and all(p.exists() for p in out_paths.values()):
                n_existing += len(out_paths)
                continue
            try:
                # Filter out already-existing per-tag if only_missing
                if only_missing:
                    out_paths = {t: p for t, p in out_paths.items() if not p.exists()}
                    n_existing += (len(_KEY_SLICE_FRACTIONS) - len(out_paths))
                written = save_key_slices(nii, out_paths)
                n_written += len(written)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{item_id}:TP{i}/{mod}: "
                              f"{type(e).__name__}: {e}")
        skipped_modalities_per_tp.append({"tp": tp_label, "study_dirname": study_dir,
                                          "skipped_modalities": skipped_here})

    # Determine modalities present (intersection-by-presence): include a modality
    # only if at least one TP has that modality (so volume.npy stays rectangular).
    modalities_present = [m for m in modalities
                          if any((i, m) in nii_by_tp_mod for i in range(len(study_dirs)))]
    n_tp = len(study_dirs)
    n_mod = len(modalities_present)

    threed_rel = threed_subdir(item_id)
    if threed_rel is None or n_mod == 0 or n_tp == 0:
        # Not a bulk_v2 item, or no usable modalities -- no 3D emit
        if threed_rel is None:
            errors.append(f"{item_id}: not a bulk_v2 item; skipping volume.npy emit")
        # Still log skipped-modalities metadata under PNG dir for transparency
        try:
            (item_dir / "render_meta.json").write_text(json.dumps({
                "item_id": item_id,
                "rule": "multi_tp_key_slices",
                "modalities_requested": list(modalities),
                "modalities_present": modalities_present,
                "tp_count": n_tp,
                "skipped_per_tp": skipped_modalities_per_tp,
            }, indent=2))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{item_id}: render_meta.json write failed: "
                          f"{type(e).__name__}: {e}")
        return item_id, n_written, n_existing, errors

    threed_dir = volume_dir_override if volume_dir_override is not None else out_root / threed_rel
    threed_dir.mkdir(parents=True, exist_ok=True)
    vol_path = threed_dir / "volume.npy"
    meta_path = threed_dir / "volume_meta.json"
    target_d, target_h, target_w = _VOL_TARGET_SHAPE
    vol_shape = (n_mod, n_tp, target_d, target_h, target_w)

    if only_missing and vol_path.exists() and meta_path.exists():
        n_existing += 1
    else:
        try:
            volume = np.zeros(vol_shape, dtype=np.float16)
            for mi, mod in enumerate(modalities_present):
                for ti in range(n_tp):
                    nii = nii_by_tp_mod.get((ti, mod))
                    if nii is None:
                        # Missing slot: leave as zeros (rectangular tensor)
                        continue
                    arr = _load_volume_3d(nii)
                    arr = _normalize_volume_pos_pct(arr)
                    resized = _trilinear_resize_to_target(arr, _VOL_TARGET_SHAPE)
                    # Re-clamp into [0,1] (interpolation can produce small overshoot)
                    resized = np.clip(resized, 0.0, 1.0)
                    volume[mi, ti] = resized.astype(np.float16)
            np.save(vol_path, volume)
            meta = {
                "item_id": item_id,
                "modalities_present": modalities_present,
                "tp_labels": [f"TP{i}" for i in range(n_tp)],
                "study_dirnames": list(study_dirs),
                "volume_shape": list(vol_shape),
                "dtype": "float16",
                "value_range": [0.0, 1.0],
                "skipped_per_tp": skipped_modalities_per_tp,
            }
            meta_path.write_text(json.dumps(meta, indent=2))
            n_written += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"{item_id}: volume.npy emit failed: "
                          f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    return item_id, n_written, n_existing, errors


def render_item(question_json: dict, patients_root: Path, out_root: Path,
                label_map: dict, only_missing: bool = False,
                item_dir_override: Path = None,
                volume_dir_override: Path = None):
    """Render all PNGs for one item. Returns (item_id, n_written, n_skipped_existing, errors).

    When ``item_dir_override`` is provided, write PNGs there instead of
    ``out_root / item_subdir(item_id)`` (used by the parquet-native entrypoint
    to honour ``test.parquet.image_relpaths[0]``). ``volume_dir_override``
    plays the same role for ``.npy`` volume emission.
    """
    item_id = question_json["item_id"]
    spec = question_json["model_image_spec"]
    ba = question_json["backend_authoring"]
    tp_labels = question_json["model_text_input"]["shown_metadata"]["shown_tp_labels"]
    patient_dir = patients_root / ba["patient_id"]

    item_dir = item_dir_override if item_dir_override is not None else out_root / item_subdir(item_id)
    item_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_existing = 0
    errors = []

    rule = spec["slice_selection_rule"]

    if rule == "multi_tp_key_slices":
        return _handle_multi_tp_key_slices(
            question_json, patient_dir, out_root, only_missing=only_missing,
            item_dir_override=item_dir_override,
            volume_dir_override=volume_dir_override,
        )

    if rule == "multi_slice_contiguous_N":
        modalities = spec.get("input_modalities", ["t1w"])
        n = spec.get("n_slices", 16)
        for i, (tp_idx, study_dir, tp_label) in enumerate(
                zip(ba.get("tp_indices", []), ba.get("study_dirnames", []), tp_labels)):
            for mod in modalities:
                # Prefer axial; fall back to sagittal then coronal.
                nii = (find_nifti(patient_dir, study_dir, mod, "axial")
                       or find_nifti(patient_dir, study_dir, mod, "sagittal")
                       or find_nifti(patient_dir, study_dir, mod, "coronal"))
                if nii is None:
                    errors.append(f"{item_id}:{mod}/{tp_label}: no source NIfTI "
                                  f"(study={study_dir})")
                    continue
                base = _nii_basename(nii)
                try:
                    written = save_multi_slice_contiguous(nii, item_dir, base, n=n)
                    n_written += len(written)
                except Exception as e:
                    errors.append(f"{item_id}:{mod}/{tp_label}: "
                                  f"{type(e).__name__}: {e}")
        if spec.get("emit_volume_npy") is True:
            if volume_dir_override is not None:
                threed_dir = volume_dir_override
            else:
                threed_rel = threed_subdir(item_id)
                if threed_rel is None:
                    errors.append(f"{item_id}: emit_volume_npy requested but no 3D output track mapping exists")
                    threed_dir = None
                else:
                    threed_dir = benchmark_root_from_out_root(out_root) / threed_rel
            if threed_dir is not None:
                vol_path = threed_dir / "volume.npy"
                meta_path = threed_dir / "volume_meta.json"
                if only_missing and vol_path.exists() and meta_path.exists():
                    n_existing += 1
                else:
                    try:
                        stack, meta = build_item_volume(
                            question_json=question_json,
                            patients_root=patients_root,
                            target_shape=(128, 128, 64),
                            np_dtype=np.float16,
                        )
                        save_item(threed_dir, stack, meta)
                        n_written += 1
                    except Exception as e:
                        errors.append(f"{item_id}: volume.npy emit failed: {type(e).__name__}: {e}")
        return item_id, n_written, n_existing, errors

    if rule == "tri_plane_structure_centric":
        # Structure-centric dispatch
        structures = spec.get("seg_structures_referenced") or []
        modalities = spec.get("input_modalities") or ["t1w"]
        planes = spec.get("input_planes") or ["axial", "sagittal", "coronal"]
        for (tp_idx, study_dir, tp_label) in zip(
                ba.get("tp_indices", []), ba.get("study_dirnames", []), tp_labels):
            seg_path = find_brain_seg(patient_dir, study_dir)
            if seg_path is None:
                errors.append(f"{item_id}:{tp_label}: no segmentation_brain file (study={study_dir})")
                continue
            for struct in structures:
                key = struct.strip().lower()
                if key not in label_map:
                    errors.append(f"{item_id}:{tp_label}: unknown structure '{struct}' "
                                  f"(not in label_map_brain.csv)")
                    continue
                label_idx = label_map[key]
                centroid = _structure_centroid(seg_path, label_idx)
                if centroid is None:
                    errors.append(f"{item_id}:{tp_label}: empty mask for '{struct}' "
                                  f"(label={label_idx}, seg={seg_path.name})")
                    continue
                ci, cj, ck = centroid
                struct_label = _structure_camel(struct)
                for mod in modalities:
                    for plane in planes:
                        nii = find_nifti(patient_dir, study_dir, mod, plane)
                        if nii is None:
                            errors.append(f"{item_id}:{tp_label}:{struct}: no source NIfTI "
                                          f"(mod={mod}, plane={plane}, study={study_dir})")
                            continue
                        base = _nii_basename(nii)
                        out_path = item_dir / f"{base}_{plane}_centroid-{struct_label}.png"
                        if only_missing and out_path.exists():
                            n_existing += 1
                            continue
                        try:
                            # Pick axis index from centroid
                            if plane == "axial":
                                vidx = ck
                            elif plane == "sagittal":
                                vidx = ci
                            elif plane == "coronal":
                                vidx = cj
                            else:
                                vidx = ck
                            save_slice_at_index(nii, out_path, plane, vidx)
                            n_written += 1
                        except Exception as e:
                            errors.append(f"{item_id}:{out_path.name}: "
                                          f"{type(e).__name__}: {e}")
        return item_id, n_written, n_existing, errors

    # Standard rules (incl. full_3d_volume)
    for mod, tp_label, study_dir, plane, is_mip, slice_override in _plan_standard(spec, ba, tp_labels):
        if is_mip:
            nii = (find_nifti(patient_dir, study_dir, mod, "axial")
                   or find_nifti(patient_dir, study_dir, mod, "sagittal")
                   or find_nifti(patient_dir, study_dir, mod, "coronal"))
        else:
            nii = find_nifti(patient_dir, study_dir, mod, plane)
        if nii is None:
            errors.append(f"{item_id}:{mod}/{tp_label}/{plane}: no source NIfTI (study={study_dir})")
            continue
        base = _nii_basename(nii)
        if is_mip:
            out_name = f"{base}_MIP-{plane}.png"
        else:
            out_name = f"{base}_{plane}.png"
        out_path = item_dir / out_name
        if only_missing and out_path.exists():
            n_existing += 1
            continue
        try:
            if is_mip:
                save_mip(nii, out_path, plane)
            else:
                save_slice(nii, out_path, plane, slice_index=slice_override)
            n_written += 1
        except Exception as e:
            errors.append(f"{item_id}:{out_name}: {type(e).__name__}: {e}")

    return item_id, n_written, n_existing, errors


# --------------------------------------------------------------------------
# Question-file discovery
# --------------------------------------------------------------------------
def discover_questions(roots):
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        seen = set()
        for p in sorted(root.rglob("*.question.json")):
            seen.add(p.resolve())
            yield p
        for p in sorted(root.rglob("question.json")):
            if p.resolve() in seen:
                continue
            yield p


def load_question(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"[err] {path}: load failed: {e}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------
def write_manifest(out_root: Path, manifest_path: Path):
    entries = []
    for p in sorted(out_root.rglob("*.png")):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        rel = p.relative_to(out_root).as_posix()
        entries.append((rel, h))
    entries.sort(key=lambda t: t[0])
    manifest_path.write_text("\n".join(f"{h}  {rel}" for rel, h in entries) + "\n")
    return len(entries)


# --------------------------------------------------------------------------
# Sharding
# --------------------------------------------------------------------------
def _shard_of(item_id: str, n_shards: int) -> int:
    """Stable shard assignment: blake2b(item_id) % n_shards.

    Using blake2b(digest_size=8) for a stable 64-bit integer across Python
    versions/processes (built-in hash() is randomized per-process).
    """
    h = hashlib.blake2b(item_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % n_shards


# --------------------------------------------------------------------------
# Parquet-native helpers (released-dataset entrypoint)
# --------------------------------------------------------------------------
def _row_to_question_json(row) -> dict:
    """Build an in-memory question_json shim from a single test.parquet row.

    The shim mirrors the fields render_item() reads from question.json files
    so we can reuse the rendering pipeline against the released parquet
    without separately authoring per-item question.json files.
    """
    def _aslist(v):
        if v is None:
            return []
        try:
            return list(v)
        except TypeError:
            return [v]

    rule = row["slice_selection_rule"]
    n_tp = int(row["n_shown_tps"])
    has_volume = row.get("volume_relpath") is not None and bool(row["volume_relpath"])
    spec = {
        "input_modalities": _aslist(row["input_modalities"]),
        "input_planes": _aslist(row["input_planes"]),
        "slice_selection_rule": rule,
        # Default 16-slice budget for the contiguous-N rule used by the 3D track.
        "n_slices": 16,
        # multi_slice_contiguous_N items in the 3D track also emit a packed
        # volume.npy when the parquet row carries a volume_relpath.
        "emit_volume_npy": (rule == "multi_slice_contiguous_N" and has_volume),
    }
    backend = {
        "patient_id": row["patient_id_anon"],
        "study_dirnames": _aslist(row["study_uids"]),
        "tp_indices": list(range(n_tp)),
    }
    return {
        "item_id": row["item_id"],
        "model_image_spec": spec,
        "backend_authoring": backend,
        "model_text_input": {
            "shown_metadata": {
                "shown_tp_labels": _aslist(row["shown_tp_labels"]),
            },
        },
    }


def _resolve_image_dir(out_root: Path, image_relpaths) -> Path:
    """Pick the canonical PNG output directory from `image_relpaths[0]`.

    The parquet stores the relative directory (e.g.
    ``images/broadQA/L1_1/bulk_v1_L1_1_001/``); we resolve it under
    ``out_root`` and strip any trailing slash.
    """
    rel = ""
    if image_relpaths is not None:
        try:
            rel = list(image_relpaths)[0]
        except (TypeError, IndexError):
            rel = ""
    rel = (rel or "").rstrip("/")
    return (out_root / rel) if rel else out_root


def _resolve_volume_dir(out_root: Path, volume_relpath):
    """Pick the canonical volume.npy parent directory from ``volume_relpath``."""
    if not volume_relpath:
        return None
    p = out_root / str(volume_relpath)
    return p.parent


def _check_mr_rate_root(mr_rate_root: Path) -> None:
    """Validate that ``mr_rate_root`` looks plausible; raise SystemExit otherwise.

    The renderer expects ``<mr_rate_root>/<patient_id>/<study_uid>/image_center_coreg/...``
    or the equivalent ``images_coreg_optional/`` layout produced by the upstream
    MR-RATE coregistration release. We only sanity-check that the root exists
    and contains at least one patient subdirectory; deeper structure is the
    upstream contract.
    """
    if not mr_rate_root.exists():
        raise SystemExit(
            f"[fatal] --mr-rate-root does not exist: {mr_rate_root}\n"
            f"        Sign the MR-RATE Data Use Agreement at "
            f"https://huggingface.co/datasets/Forithmus/MR-RATE and download "
            f"the dataset before rendering."
        )
    if not mr_rate_root.is_dir():
        raise SystemExit(
            f"[fatal] --mr-rate-root is not a directory: {mr_rate_root}"
        )
    has_patient_dir = False
    for child in mr_rate_root.iterdir():
        if child.is_dir():
            has_patient_dir = True
            break
    if not has_patient_dir:
        raise SystemExit(
            f"[fatal] --mr-rate-root has no patient subdirectories: {mr_rate_root}\n"
            f"        Expected MR-RATE layout: "
            f"<mr-rate-root>/<patient_id>/<study_uid>/image_center_coreg/*.nii.gz"
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _build_parquet_work(args, ap):
    """Build the per-item work list for parquet mode."""
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit(
            "[fatal] --dataset (parquet mode) requires pandas + pyarrow.\n"
            "        Install with: pip install pandas pyarrow"
        ) from e

    dataset_root = args.dataset.resolve()
    parquet_path = dataset_root / "data" / "test.parquet"
    if not parquet_path.is_file():
        raise SystemExit(
            f"[fatal] --dataset must contain data/test.parquet; "
            f"not found at {parquet_path}"
        )

    if args.mr_rate_root is None:
        ap.error("--mr-rate-root is required in parquet mode (--dataset)")
    if args.out_root is None:
        ap.error("--out-root is required in parquet mode (--dataset)")

    mr_rate_root = args.mr_rate_root.resolve()
    out_root = args.out_root.resolve()
    _check_mr_rate_root(mr_rate_root)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    work = []
    seen = set()
    dropped_shard = 0
    for row in df.itertuples(index=False):
        rd = row._asdict()
        iid = rd["item_id"]
        if iid in seen:
            continue
        seen.add(iid)
        if args.n_shards > 1 and _shard_of(iid, args.n_shards) != args.shard_idx:
            dropped_shard += 1
            continue
        qj = _row_to_question_json(rd)
        item_dir = _resolve_image_dir(out_root, rd["image_relpaths"])
        volume_dir = _resolve_volume_dir(out_root, rd["volume_relpath"])
        work.append((qj, item_dir, volume_dir))
        if args.limit and len(work) >= args.limit:
            break
    return mr_rate_root, out_root, work, dropped_shard


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    # Parquet-native (released-dataset) mode
    ap.add_argument("--dataset", type=Path, default=None,
                    help="Path to the released BrainTRACE dataset root "
                         "(must contain data/test.parquet). When set, the "
                         "renderer walks every parquet row and writes PNGs / "
                         "volumes to the canonical layout from "
                         "image_relpaths and volume_relpath.")
    ap.add_argument("--mr-rate-root", type=Path, default=None,
                    help="Path to the user's local MR-RATE download (used as "
                         "the patients-root in parquet mode).")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Output root for rendered PNGs and 3D-track .npy "
                         "volumes (parquet mode). Typically the same as "
                         "--dataset so that image_relpaths resolve in place.")
    # Legacy (internal) mode used by extract_from_mr_rate.py / question.json
    # authoring pipelines.
    ap.add_argument("--patients-root", type=Path, default=None,
                    help="(legacy mode) Patients root produced by "
                         "extract_from_mr_rate.py.")
    ap.add_argument("--questions-roots", type=Path, nargs="+", default=None,
                    help="(legacy mode) Roots containing question.json files.")
    ap.add_argument("--out", type=Path, default=None,
                    help="(legacy mode) Output root.")
    # Shared
    ap.add_argument("--only-missing", action="store_true",
                    help="Skip items whose expected PNGs already exist.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N items (0 = all). For smoke tests.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--manifest", action="store_true",
                    help="After rendering, write <out>/_manifest.sha256 "
                         "(sorted by relpath).")
    ap.add_argument("--shard-idx", type=int, default=0,
                    help="0..n_shards-1; keep items where blake2b(item_id) %% "
                         "n_shards == shard_idx.")
    ap.add_argument("--n-shards", type=int, default=1,
                    help="Total number of shards (default 1 = no sharding).")
    ap.add_argument("--label-map-csv", type=Path, default=None,
                    help="Path to labels/seg_volumes/01_label_map_brain.csv. "
                         "Optional; only consumed by tri_plane_structure_centric "
                         "items, which are not present in the released parquet.")
    args = ap.parse_args()

    if args.n_shards < 1:
        ap.error("--n-shards must be >= 1")
    if not (0 <= args.shard_idx < args.n_shards):
        ap.error(f"--shard-idx must be in [0, {args.n_shards})")

    parquet_mode = args.dataset is not None
    legacy_mode = (args.patients_root is not None
                   or args.questions_roots is not None
                   or args.out is not None)
    if parquet_mode and legacy_mode:
        ap.error("Use either --dataset (parquet mode) OR "
                 "--patients-root/--questions-roots/--out (legacy mode), not both.")
    if not parquet_mode and not legacy_mode:
        ap.error("Provide --dataset (parquet mode) OR "
                 "--patients-root + --questions-roots + --out (legacy mode).")

    # ---- assemble work + roots ----
    if parquet_mode:
        patients_root, out_root, work_pq, dropped_shard = _build_parquet_work(args, ap)
        # work_pq entries are (question_json, item_dir, volume_dir)
        n_total = len(work_pq)
        # Locate label map (optional in parquet mode)
        lbl_csv = args.label_map_csv
        label_map = _load_label_map(lbl_csv) if lbl_csv else {}
    else:
        if args.patients_root is None or args.questions_roots is None or args.out is None:
            ap.error("Legacy mode requires --patients-root, --questions-roots, and --out")
        patients_root = args.patients_root
        out_root = args.out
        out_root.mkdir(parents=True, exist_ok=True)
        lbl_csv = args.label_map_csv or (patients_root.parent / DEFAULT_LABEL_MAP_CSV)
        label_map = _load_label_map(lbl_csv)
        if not label_map:
            print(f"[warn] label_map at {lbl_csv} empty or missing; "
                  f"tri_plane_structure_centric items will be skipped.",
                  file=sys.stderr)
        else:
            print(f"[init] loaded {len(label_map)} LUMIR brain labels from {lbl_csv}",
                  flush=True)
        seen_ids = set()
        work_legacy = []
        dropped_shard = 0
        for qpath in discover_questions(args.questions_roots):
            qj = load_question(qpath)
            if qj is None:
                continue
            iid = qj.get("item_id")
            if not iid or iid in seen_ids:
                continue
            seen_ids.add(iid)
            if args.n_shards > 1 and _shard_of(iid, args.n_shards) != args.shard_idx:
                dropped_shard += 1
                continue
            work_legacy.append(qj)
            if args.limit and len(work_legacy) >= args.limit:
                break
        n_total = len(work_legacy)

    print(f"[plan] {n_total} items to process "
          f"(shard {args.shard_idx}/{args.n_shards}; dropped_by_shard={dropped_shard}); "
          f"out_root={out_root}", flush=True)

    t0 = time.time()
    n_items_done = 0
    n_items_all_ok = 0
    n_items_with_errors = 0
    total_written = 0
    total_existing = 0
    all_errors = []

    def _task_pq(qj, item_dir, volume_dir):
        try:
            return render_item(qj, patients_root, out_root,
                               label_map=label_map,
                               only_missing=args.only_missing,
                               item_dir_override=item_dir,
                               volume_dir_override=volume_dir)
        except Exception as e:  # noqa: BLE001
            return qj.get("item_id", "?"), 0, 0, [
                f"fatal: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            ]

    def _task_legacy(qj):
        try:
            return render_item(qj, patients_root, out_root,
                               label_map=label_map, only_missing=args.only_missing)
        except Exception as e:  # noqa: BLE001
            return qj.get("item_id", "?"), 0, 0, [
                f"fatal: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            ]

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        if parquet_mode:
            futs = {ex.submit(_task_pq, qj, idir, vdir): qj
                    for (qj, idir, vdir) in work_pq}
        else:
            futs = {ex.submit(_task_legacy, qj): qj for qj in work_legacy}
        for fut in as_completed(futs):
            item_id, n_w, n_e, errs = fut.result()
            n_items_done += 1
            total_written += n_w
            total_existing += n_e
            if errs:
                n_items_with_errors += 1
                for e in errs[:5]:
                    print(f"[err] {e}", file=sys.stderr, flush=True)
                all_errors.extend(errs)
            else:
                n_items_all_ok += 1
            if n_items_done % 50 == 0:
                el = time.time() - t0
                rate = n_items_done / max(el, 1e-6)
                print(f"[prog] {n_items_done}/{n_total} items ({rate:.1f}/s) "
                      f"written={total_written} existing={total_existing} "
                      f"err_items={n_items_with_errors}", flush=True)

    el = time.time() - t0
    print(f"[done] items={n_items_done} ok={n_items_all_ok} "
          f"with_errors={n_items_with_errors} "
          f"written={total_written} existing={total_existing} "
          f"elapsed={el:.1f}s", flush=True)
    if all_errors:
        print(f"[done] {len(all_errors)} per-file errors logged; continuing.",
              file=sys.stderr)

    if args.manifest:
        mpath = out_root / "_manifest.sha256"
        n = write_manifest(out_root, mpath)
        size = mpath.stat().st_size
        print(f"[manifest] wrote {mpath} ({n} PNGs, {size} bytes)", flush=True)


if __name__ == "__main__":
    main()
