# `reproduction/` — Render images, run end-to-end pipeline

## Files

| File | Purpose |
|---|---|
| `render_images.py` | Reads BrainTRACE per-item metadata from `data/test.parquet`, pulls the corresponding MR-RATE study/series/timepoints, applies the per-item `slice_selection_rule`, and writes deterministic PNG mosaics to the canonical `image_relpaths[0]` directory and (for the 3D track) a `.npy` volume to the canonical `volume_relpath`. |
| `render_volume_utils.py` | Volume-stack builder used by `render_images.py` for the 3D track. |
| `reproduce.sh` | Glue script that runs render → infer → score for a single model. |
| `leaderboard_runs.csv` | Cost / wall-time / config record for every model in the published leaderboard. |

## Rendering imagery from MR-RATE

Imagery is **not** redistributed with the dataset. You must obtain MR-RATE
yourself (see `../DUA_NOTICE.md` and the [MR-RATE HF page](https://huggingface.co/datasets/Forithmus/MR-RATE)).

After downloading MR-RATE locally (e.g. to `./mr_rate`):

```bash
python reproduction/render_images.py \
  --dataset      ./braintrace_dataset \
  --mr-rate-root ./mr_rate \
  --out-root     ./braintrace_dataset
```

The renderer walks every row of `data/test.parquet`, opens the upstream
NIfTI files under
`<mr-rate-root>/<patient_id_anon>/<study_uid>/image_center_coreg/*.nii.gz`
(or `images_coreg_optional/`), applies the per-item `slice_selection_rule`,
and writes the rendered evidence to the canonical paths recorded in the
parquet:

```
braintrace_dataset/
├── data/test.parquet
├── images/
│   ├── broadQA/<template>/<item_id>/<basename>_<plane>.png
│   ├── 3D/<template>/<item_id>/<basename>_axial_slice_<NN>.png
│   └── clinical_reasoning_QA/case_reasoning/<item_id>/TP{i}_<modality>_axial-<tag>.png
└── volumes/
    └── 3D/<item_id>/volume.npy             (3D-track items only)
```

Useful flags:

| Flag | Default | Notes |
|---|---|---|
| `--limit N` | `0` (all 6,923) | Smoke-test fewer items. |
| `--workers N` | `4` | Render thread pool size. |
| `--shard-idx i --n-shards N` | `0 / 1` | Distribute across N parallel workers; each worker processes the subset where `blake2b(item_id) % N == i`. |
| `--only-missing` | off | Skip items whose target PNGs already exist on disk. |
| `--manifest` | off | After rendering, write `<out-root>/_manifest.sha256`. |

If `--mr-rate-root` does not exist or contains no patient subdirectories,
the renderer fails fast with a pointer to the upstream DUA page rather than
silently producing a partial benchmark.

The renderer is deterministic given the same MR-RATE snapshot: per-slice
intensity windowing uses the 1st / 99th percentile of strictly-positive
voxels, slices are rotated `np.rot90(...)`, then thumbnailed to ≤512×512 with
PIL's LANCZOS filter and saved with `optimize=True`. Verify a fresh render
with `--manifest` and diff against the shard hashes recorded in the
render manifest from a previous run.

Slice-selection logic and per-template image budget are implemented in
`render_images.py` and summarized above.

### Recovering the source-cohort scope

The released dataset card ships two cohort manifests
(`braintrace_dataset/cohort/braintrace_cohort_patients_1778.csv` and
`braintrace_cohort_studies_7299.csv`) that map BrainTRACE items back to
MR-RATE pseudonymous identifiers. If you only want to download the studies
needed to render BrainTRACE evidence, scope your MR-RATE pull to the 1,778
`patient_uid` values (or the 7,299 `(patient_uid, study_uid)` pairs)
listed in those manifests.

## End-to-end pipeline

`reproduce.sh` runs render → infer → score for a single model:

```bash
./reproduction/reproduce.sh gpt-5.4
./reproduction/reproduce.sh gemini-2.5-pro --out ./outputs/gemini
./reproduction/reproduce.sh claude-opus-4-6 --dataset ./bt_data --mr-rate ./mr_rate
```

Phases can be skipped:

```bash
./reproduction/reproduce.sh gpt-5.4 --render-only   # just render images
./reproduction/reproduce.sh gpt-5.4 --score-only    # skip render + infer
```

For open-weight models the script expects you to drop a
`./outputs/<model>/predictions.jsonl` produced by your own adapter and
will only run the render + score phases.

## Cost & wall-time reference

`leaderboard_runs.csv` records, for every model in the published
leaderboard:

- adapter flags (max-tokens, temperature, parallel, reasoning-effort)
- inference cost in USD (API) or GPU-hours (open-weight)
- wall-time per phase (render / infer / score)
- total wall-time

Use this file to budget a re-run before firing it.
