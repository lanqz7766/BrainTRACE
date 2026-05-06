# BrainTRACE — Task Taxonomy

BrainTRACE comprises 31 task templates (22 closed-form, 8 open-ended,
1 case-level reasoning), organised across five cognitive levels. The five-level taxonomy probes increasing cognitive demand on
longitudinal brain MRI: from single-image recognition (L1) to multi-step
clinical synthesis (L5). Each template targets a distinct skill that a
clinical neuroradiologist exercises during a routine longitudinal read.

For the answer-format and scoring split, see [`SCORING.md`](./SCORING.md).
For the per-row schema, see the dataset card.

## Five cognitive levels

| Level | Demand | n templates | Family |
|---:|---|---:|---|
| **L1** | Single-image recognition (no context) | 5 | `perception` |
| **L2** | Within-study reasoning (single timepoint) | 8 | `attribute_extraction`, `volumetric_analysis` |
| **L3** | Two-timepoint interval comparison | 7 | `interval_change`, `volumetric_analysis` |
| **L4** | Multi-timepoint trajectory (≥3 TPs) | 6 | `trajectory`, `treatment_response`, `volumetric_analysis` |
| **L5** | Synthesis & clinical reasoning | 5 (4 single-turn + case-level reasoning) | `case_impression`, `status_summary`, `differential_diagnosis`, `case_reasoning` |

## Per-template registry

Closed-form (`gt_value` exact match), open-ended (LLM judge with
slot rubric), and case-level reasoning (stepwise judge) are denoted in
the **format** column.

### L1 — Recognition (single image, no context)

| Template | Name | Sub-cat | Format | Probes |
|---|---|---|---|---|
| `L1.1` | pulse_sequence_identification | Acq | closed | Identify the MRI sequence (T1, T2, FLAIR, DWI, SWI, …) |
| `L1.2` | plane_id | Acq | closed | Identify the acquisition plane (axial / coronal / sagittal) |
| `L1.3` | laterality_of_focal_finding | Det | closed | Localise the dominant focal finding (left / right / midline / bilateral) |
| `L1.4` | focal_abnormality_presence | Det | closed | Yes/no — is a focal abnormality visible? |
| `L1.6` | one_sentence_image_description | Det | open | One-sentence description of the displayed image |

### L2 — Within-study reasoning (single TP)

| Template | Name | Sub-cat | Format | Probes |
|---|---|---|---|---|
| `L2.1` | discrete_abnormality_count | Burd | closed | Count discrete focal abnormalities (1, 2, 3, ≥4) |
| `L2.2` | location_bucket_dominant | Loc | closed | Anatomical region of the dominant abnormality |
| `L2.3` | size_bucket_dominant | Burd | closed | Size bucket of the dominant abnormality (from report) |
| `L2.4` | signal_pattern | App | closed | Signal characteristics across MRI sequences (rendered as bucketed MCQ) |
| `L2.5` | enhancement_pattern | App | closed | Pattern of contrast enhancement (rendered as bucketed MCQ) |
| `L2.6` | morphology_and_boundary | App | open | Morphology, margin sharpness, internal heterogeneity |
| `L2.7` | structure_volume_bucket | Loc | closed | Quartile bucket of an anatomical structure's volume (seg-derived) |
| `L2.8` | bilateral_volumetric_asymmetry | Loc | closed | Detect bilateral asymmetry in paired structures |

### L3 — Two-timepoint interval

| Template | Name | Sub-cat | Format | Probes |
|---|---|---|---|---|
| `L3.1` | size_change_at_site | Quant | closed | Size change at a named site between TPs |
| `L3.2` | new_focal_abnormality | Event | closed | Yes/no — has a new focal finding appeared? |
| `L3.3` | resolution_of_prior | Event | closed | Yes/no — has a prior finding resolved? |
| `L3.4` | enhancement_change_at_site | Quant | closed | Change in enhancement pattern at a named site |
| `L3.5` | mass_effect_change_at_site | Effect | closed | Change in mass effect / midline shift / hydrocephalus (rendered as bucketed MCQ) |
| `L3.6` | open_interval_change_at_site | Effect | open | Open description of the interval change |
| `L3.7` | structure_volume_change | Effect | closed | Anatomical structure volume change (seg-derived; rendered as ordinal bucket MCQ) |

### L4 — Multi-timepoint trajectory (≥3 TPs)

| Template | Name | Sub-cat | Format | Probes |
|---|---|---|---|---|
| `L4.1` | trajectory_class_closed | Traj | closed | Trajectory class MCQ (improving / stable / worsening / mixed / pseudo-progression …) |
| `L4.2` | new_lesion_timing | Time | closed | Earliest timepoint at which a new lesion appears |
| `L4.3` | peak_extent_timepoint | Time | closed | Timepoint of peak burden / largest extent |
| `L4.4` | open_trajectory_summary | Traj | open | Open-paragraph trajectory summary |
| `L4.5` | response_proxy_RANO_lite | Resp | closed | RANO-lite response category MCQ (CR / PR / SD / PD) |
| `L4.6` | structure_volume_trajectory | Traj | closed | Anatomical structure volume trajectory class (seg-derived; rendered as MCQ) |

### L5 — Synthesis & clinical reasoning

| Template | Name | Sub-cat | Format | Probes |
|---|---|---|---|---|
| `L5.1` | full_impression_3to5_sentences | Imp | open | Generate a 3–5 sentence radiology impression |
| `L5.2` | brief_imaging_status_summary | Imp | open | Brief imaging status summary (1–2 sentences) |
| `L5.3` | comparative_interval_summary | Comp | open | Comparative interval summary across two timepoints |
| `L5.4` | differential_with_visual_evidence | Diff | open | Top differential diagnosis with visual evidence cited |
| Case-level reasoning | diagnostic_reasoning_chain | — | case_reasoning | Six decomposed VQA steps from baseline anchor → interval comparisons → final synthesis |

## L5: two evaluation tracks

L5 is split deliberately into two columns in the leaderboard:

- **Single-turn synthesis.** Open-ended generation that
  mirrors the style of a real radiology impression. These probe whether a
  model can produce *fluent radiology-style prose*, not whether the prose
  is *factually correct*. Frontier VLMs typically score 20–40 % on lenient
  rubrics here yet collapse on strict factual grading.
- **Case-level reasoning.** Six sequential sub-questions that force a
  baseline-anchor → interval-comparison → final-synthesis sequence. Each
  decomposed VQA step has its own slot rubric, and Case Success requires
  every step in the case to pass. Step Pass is the diagnostic that
  surfaces which sub-question type a model fails on.

The bench's strongest paper claim comes from this split: every frontier
VLM we tested converges to roughly 24 % Step Pass on case-level reasoning regardless of
test-time-compute scaling, while still scoring 20–40 % on L5.1. **Fluent
report-style output does not imply visual reasoning.**

## Cross-cutting axes

Each template additionally implicates one or more of four reasoning
axes (used to design ablations):

| Axis | Description |
|---|---|
| **A — Longitudinal** | ≥2 timepoints, requires temporal integration |
| **B — Multimodal** | ≥2 MRI sequences, requires cross-sequence integration |
| **C — Volumetric (3D)** | Native 3D volume *or* multi-plane 2.5D mosaic |
| **D — Clinical reasoning** | Demands clinician-level judgement (response classification, differential, treatment effect) |

The 3D track (`subset` ∈ {`v1`, `v2`} of `track="3D"`) carries the
multi-slice 2.5D mosaic *and* a paired `volume.npy` so native-volumetric
models (e.g. M3D-LaMed, RadFM) consume the same items the 2D models do.

## Sub-category codes (for the `sub_category` parquet column)

| Code | Expansion | Templates |
|---|---|---|
| `Acq` | Acquisition | L1.1, L1.2 |
| `Det` | Detection (presence + localization) | L1.3, L1.4, L1.6 |
| `Burd` | Burden | L2.1, L2.3 |
| `Loc` | Location | L2.2, L2.7, L2.8 |
| `App` | Appearance | L2.4, L2.5, L2.6 |
| `Quant` | Quantitative | L3.1, L3.4 |
| `Event` | Event | L3.2, L3.3 |
| `Effect` | Effect | L3.5, L3.6, L3.7 |
| `Traj` | Trajectory | L4.1, L4.4, L4.6 |
| `Time` | Timing | L4.2, L4.3 |
| `Resp` | Response | L4.5 |
| `Imp` | Impression | L5.1, L5.2 |
| `Comp` | Comparison | L5.3 |
| `Diff` | Differential | L5.4 |
| (case-level reasoning) | — | Decomposed case-level reasoning steps |

## Note on template numbering

Template ids are stable, public-facing identifiers — they intentionally
match the ids referenced throughout the BrainTRACE companion paper, all
ground-truth files, and all model output files in the leaderboard.

You will notice **L1 has a small gap**: the L1 family contains
`L1.1, L1.2, L1.3, L1.4, L1.6` — there is no `L1.5`. The number was used
during early benchmark design and the corresponding template was merged
into `L1.6` (`one_sentence_image_description`). We deliberately preserve
the numbering rather than reindexing because every existing item id, GT
file, output file, and leaderboard entry carries the original ids;
reindexing would invalidate every prior model run. **Levels L2 through
L5 are continuous** with no gaps.

## Notes on what the taxonomy *avoids*

- **No tumour-volume questions from the segmentation atlas.** The
  segmentation labels we ship (anatomical structures only) do *not*
  include tumour, MS plaque, cyst, etc. Tumour size questions
  (`L2.3`, `L3.1`) draw their ground truth from the upstream radiology
  report, not from a segmentation-derived bucket.
- **Treatment-effect templates (`L4.5`) avoid an explicit "N/A" foil**
  to prevent models from hedging into a non-clinical option.
- **Volumetric templates (`L2.7/8`, `L3.7`, `L4.6`) only use anatomical
  white-list structures** — left/right hippocampus, left/right lateral
  ventricle, third ventricle, brain stem.
