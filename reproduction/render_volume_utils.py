#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


MOD_PATTERNS = {
    "t1w": [r"_t1w-raw-", r"_coreg_t1w-raw-"],
    "t2w": [r"_t2w-raw", r"_coreg_t2w-raw"],
    "flair": [r"_flair-raw", r"_coreg_flair-raw"],
    "swi": [r"_swi-raw", r"_coreg_swi-raw"],
}
PLANE_SUFFIX = {
    "axial": ["-axi"],
    "sagittal": ["-sag"],
    "coronal": ["-cor"],
    "oblique": ["-obl"],
}


def find_nifti(patient_dir: Path, study_dirname: str, modality: str, plane: str) -> Optional[Path]:
    study = patient_dir / study_dirname
    for sub in ["image_center_coreg", "images_coreg_optional"]:
        directory = study / sub
        if not directory.is_dir():
            continue
        for file_path in sorted(directory.iterdir()):
            if not file_path.name.endswith(".nii.gz"):
                continue
            name = file_path.name.lower()
            mod_hit = any(re.search(pattern, name) for pattern in MOD_PATTERNS.get(modality, []))
            if not mod_hit:
                continue
            plane_hit = any(suffix in name for suffix in PLANE_SUFFIX.get(plane, []))
            if plane_hit:
                return file_path
    return None


def load_volume_ras(nii_path: Path) -> np.ndarray:
    img = nib.load(str(nii_path))
    img = nib.as_closest_canonical(img)
    arr = img.get_fdata(dtype=np.float32)
    while arr.ndim > 3:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"{nii_path}: unexpected ndim={arr.ndim}")
    return arr


def normalize_p1_p99(arr: np.ndarray) -> np.ndarray:
    pos = arr > 0
    if pos.any():
        lo, hi = np.percentile(arr[pos], [1, 99])
    else:
        lo, hi = float(arr.min()), float(arr.max())
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return arr.astype(np.float32)


def _center_fit(arr: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    out = arr
    for axis, (cur, tgt) in enumerate(zip(out.shape, target_shape)):
        if cur == tgt:
            continue
        if cur > tgt:
            start = (cur - tgt) // 2
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(start, start + tgt)
            out = out[tuple(sl)]
        else:
            pad_total = tgt - cur
            pad_lo = pad_total // 2
            pad_hi = pad_total - pad_lo
            pad_width = [(0, 0)] * out.ndim
            pad_width[axis] = (pad_lo, pad_hi)
            out = np.pad(out, pad_width, mode="constant", constant_values=0.0)
    return out


def resample_to_shape(arr: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    if arr.shape == target_shape:
        return arr
    factors = tuple(t / s for t, s in zip(target_shape, arr.shape))
    out = zoom(arr, factors, order=1, mode="nearest", prefilter=False)
    if out.shape != target_shape:
        out = _center_fit(out, target_shape)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def process_nifti(nii_path: Path, target_shape: tuple[int, int, int]) -> np.ndarray:
    arr = load_volume_ras(nii_path)
    arr = normalize_p1_p99(arr)
    arr = resample_to_shape(arr, target_shape)
    return arr


def build_item_volume(
    question_json: dict,
    patients_root: Path,
    target_shape: tuple[int, int, int],
    np_dtype: np.dtype,
) -> tuple[np.ndarray, dict]:
    item_id = question_json["item_id"]
    spec = question_json["model_image_spec"]
    backend = question_json["backend_authoring"]
    tp_labels = question_json["model_text_input"]["shown_metadata"]["shown_tp_labels"]

    modalities = list(spec["input_modalities"])
    planes = list(spec["input_planes"])
    if not modalities:
        raise ValueError(f"{item_id}: empty input_modalities")
    if not planes:
        raise ValueError(f"{item_id}: empty input_planes")
    plane = planes[0]

    patient_id = backend["patient_id"]
    study_dirnames = list(backend["study_dirnames"])
    if len(study_dirnames) != len(tp_labels):
        raise ValueError(
            f"{item_id}: len(study_dirnames)={len(study_dirnames)} != len(tp_labels)={len(tp_labels)}"
        )

    patient_dir = patients_root / str(patient_id)
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"{item_id}: patient_dir missing: {patient_dir}")

    height, width, depth = target_shape
    n_modalities = len(modalities)
    n_timepoints = len(study_dirnames)
    stack = np.zeros((n_modalities, n_timepoints, height, width, depth), dtype=np_dtype)

    source_niftis: list[str] = []
    missing: list[str] = []

    for modality_idx, modality in enumerate(modalities):
        for timepoint_idx, (study_dirname, tp_label) in enumerate(zip(study_dirnames, tp_labels)):
            nii_path = find_nifti(patient_dir, study_dirname, modality, plane)
            if nii_path is None:
                missing.append(f"{modality}@{tp_label}({study_dirname})")
                source_niftis.append("")
                continue
            try:
                volume = process_nifti(nii_path, target_shape)
                stack[modality_idx, timepoint_idx] = volume.astype(np_dtype)
                source_niftis.append(str(nii_path))
            except Exception as exc:
                missing.append(f"{modality}@{tp_label}({study_dirname}): {exc}")
                source_niftis.append("")

    meta = {
        "item_id": item_id,
        "template_id": question_json.get("template_id"),
        "shape": list(stack.shape),
        "dtype": str(np_dtype.__name__ if hasattr(np_dtype, "__name__") else np_dtype),
        "modalities": modalities,
        "tp_labels": tp_labels,
        "plane": plane,
        "patient_id": str(patient_id),
        "study_dirnames": study_dirnames,
        "target_shape_hwd": [height, width, depth],
        "normalization": "p1_p99_on_positive_voxels_clipped_0_1",
        "orientation": "RAS (nibabel.as_closest_canonical)",
        "resample": "scipy.ndimage.zoom order=1",
        "source_niftis": source_niftis,
        "missing_niftis": missing,
    }
    return stack, meta


def save_item(out_dir: Path, stack: np.ndarray, meta: dict) -> tuple[int, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / "volume.npy"
    meta_path = out_dir / "volume_meta.json"
    np.save(npy_path, stack)
    size = npy_path.stat().st_size
    meta["file_bytes"] = int(size)
    with meta_path.open("w") as handle:
        json.dump(meta, handle, indent=2)
    return size, npy_path
