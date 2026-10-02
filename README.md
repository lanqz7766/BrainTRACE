<h1 align="center">BrainTRACE</h1>

<p align="center"><strong>Tracing Longitudinal, Multimodal, and Volumetric Evidence<br>in Brain MRI Clinical Reasoning</strong></p>

<p align="center">
  <a href="https://openreview.net/forum?id=hXktXOmwp6"><img src="https://img.shields.io/badge/NeurIPS_2026-Accepted_%C2%B7_E%26D-2563eb?style=flat-square" alt="Accepted at NeurIPS 2026, Evaluations & Datasets Track"></a>
  <a href="https://huggingface.co/datasets/BrainTRACE-anon/BrainTRACE"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-f5c451?style=flat-square" alt="Hugging Face dataset"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/Code-Apache--2.0-64748b?style=flat-square" alt="Code license: Apache-2.0"></a>
</p>

<p align="center">
  Qizhen Lan<sup>*</sup> · Mengchen Fan<sup>*</sup> · Hang Zhang<sup>*</sup> · Jingwei Duan<br>
  Moule Lin · Jialin Chen · Baocheng Geng · Xiaoqian Jiang<sup>†</sup><br>
  <sub><sup>*</sup>Equal contribution · <sup>†</sup>Corresponding author: Xiaoqian Jiang<br>
  Contact: <a href="mailto:qizhen.lan@uth.tmc.edu">qizhen.lan@uth.tmc.edu</a></sub>
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=hXktXOmwp6">Paper / OpenReview</a> ·
  <a href="https://huggingface.co/datasets/BrainTRACE-anon/BrainTRACE">Dataset</a> ·
  <a href="#key-results">Results</a> · <a href="#get-started">Get started</a> · <a href="#citation">Citation</a>
</p>

**Accepted at NeurIPS 2026, Evaluations & Datasets Track (Poster).**

BrainTRACE evaluates whether vision-language models can follow the evidence chain of brain MRI interpretation: locate a finding, compare studies across time, integrate MRI sequences and spatial evidence, and reach a report-grounded clinical assessment. It separates success on individual recognition tasks from success on a complete longitudinal reasoning case.

![BrainTRACE benchmark overview](docs/assets/braintrace-overview.png)

| Scored VQA instances | Longitudinal patients | MRI studies | Model configurations |
|:---:|:---:|:---:|:---:|
| **7,273** | **1,778** | **7,299** | **20** |

## Benchmark at a glance

The release contains **6,923 source items**. Each of the 70 case-reasoning items expands into six scored steps, giving **7,273 scored VQA instances** in total.

| Track | Source items | What it tests |
|---|---:|---|
| Broad-compatible (`broadQA`) | 5,653 | Recognition, localization, burden, interval change, trajectory, and synthesis using rendered visual inputs |
| 3D-evidence (`3D`) | 1,200 | Shared spatial-evidence targets using rendered views or supported native-volume interfaces |
| Case reasoning (`clinical_reasoning_QA`) | 70 cases / 420 steps | A six-step longitudinal evidence chain, from anchor localization to final synthesis |

**Five reasoning levels:** L1 recognition → L2 within-study reasoning → L3 interval comparison → L4 multi-timepoint trajectory → L5 synthesis and clinical reasoning. Thirty single-turn task templates accompany the case-reasoning track.

**Construction:** MR-RATE studies, reports, and segmentation-derived measurements provide the reference evidence. Task construction links that evidence to questions, targets, and scoring rubrics; quality checks cover report support, timepoint alignment, visual availability, and scoring consistency. Models receive permitted visual inputs and questions, without source reports, masks, measurements, or hidden references. The documented item-level clinical review covers the 70 case trajectories and 420 steps; it does not establish a second clinical read of every benchmark item.

## Key results

The paper evaluates **20 configurations** across general, medical, frontier API, and native-volume VLMs.

- **Recognition remains fragmented:** the best broad-compatible closed-form accuracy is **41.9%**.
- **Grounded generation is harder:** the best broad-compatible open-ended clinical pass rate is **19.8%**.
- **Local successes rarely compose:** the best Step Pass is **24.5%**, while the best full Case Success is **1.4%**.

Selected configurations from the paper are shown below. All values are percentages; each headline above is the maximum for its metric across configurations.

| Model | Closed<br>Broad | Closed<br>3D | Open pass<br>Broad | Open pass<br>3D | Step<br>Pass | Case<br>Success |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.4 | 40.4 | 42.2 | 19.8 | 15.6 | 24.0 | 0.0 |
| GPT-5 | 41.9 | 38.3 | 10.5 | 11.7 | 21.7 | 1.4 |
| GPT-5-mini | 38.2 | 37.5 | 10.6 | 10.4 | 24.5 | 0.0 |
| Qwen3-VL-30B | 35.1 | 34.8 | 14.4 | 11.7 | 11.7 | 0.0 |
| Qwen2.5-VL-7B | 33.2 | 33.2 | 4.9 | 3.9 | 9.3 | 1.4 |
| Lingshu-7B | 36.5 | 35.4 | 6.8 | 3.9 | 6.9 | 0.0 |
| M3D-LaMed | — | 29.9 | — | 0.0 | 4.8 | 0.0 |
| RadFM | 18.1 | 15.0 | 4.3 | 2.6 | 0.0 | 0.0 |

[**Full results for all 20 configurations →**](docs/RESULTS.md)

Closed columns report exact-match accuracy by track. Open columns report rubric-based clinical pass, using the release judge `gpt-4o-mini-2024-07-18`. Step Pass averages over 420 decomposed steps; Case Success requires **all six steps in the same case** to pass. **—** denotes an unreported entry. These are distinct metrics and denominators, rather than interchangeable measures of clinical ability.

![Attrition through the six-step case-reasoning chain](docs/assets/case-reasoning-attrition.png)

*Paper figure: the left panel follows same-case survival for GPT-5-mini; the right panel shows per-step pass rates. A model can pass individual steps in different cases while completing no full case.*

The scores are conditioned on the released inputs and judging protocol. Judge-sensitivity audits change exact scores, and supported volume/rendered interfaces do not isolate a causal benefit of volumetric representation. The paper's physician MCQ answerability check uses a separate format and is not included as a directly comparable model baseline.

## Get started

### 1. Install and download the benchmark metadata

```bash
git clone https://github.com/lanqz7766/BrainTRACE.git
cd BrainTRACE
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -c "from huggingface_hub import snapshot_download; snapshot_download('BrainTRACE-anon/BrainTRACE', repo_type='dataset', local_dir='./braintrace_dataset')"
```

The dataset remains at its existing Hugging Face namespace. It contains questions, ground truth, rubrics, and pointers to the required evidence; **MRI imagery is not redistributed**.

### 2. Obtain MR-RATE and render inputs locally

Obtain the upstream data from [MR-RATE](https://huggingface.co/datasets/Forithmus/MR-RATE), accept its Data Use Agreement, and place the download under `./mr_rate`. Follow the expected layout in the [rendering guide](reproduction/README.md).

```bash
python reproduction/render_images.py \
  --dataset ./braintrace_dataset \
  --mr-rate-root ./mr_rate \
  --out-root ./braintrace_dataset
```

### 3. Run a VLM

The bundled single-turn adapter supports OpenAI, Anthropic, and Google APIs. Set the relevant API key in your shell environment; OpenAI inference and the default audit judge use `OPENAI_API_KEY`.

```bash
python eval/vlm_eval_api.py \
  --model gpt-5.4 \
  --provider openai \
  --sample-file ./braintrace_dataset/data/test.parquet \
  --images-root ./braintrace_dataset/images \
  --out-dir ./outputs/gpt-5.4/main_outputs \
  --reasoning-effort medium \
  --max-tokens 2048 \
  --parallel 4 \
  --skip-existing
```

For the case-reasoning track, use the dedicated driver described in [the inference guide](eval/README.md). Open-weight and native-volume models require their own inference backends and should write predictions in the documented adapter format. Image limits and provider settings are part of each model's evaluation configuration; see the [run manifest](reproduction/leaderboard_runs.csv).

### 4. Score predictions

```bash
python scoring/score.py \
  --dataset ./braintrace_dataset \
  --predictions ./outputs/gpt-5.4/main_outputs \
  --out-dir ./scores/gpt-5.4 \
  --judge-model gpt-4o-mini-2024-07-18
```

Use the corresponding case-reasoning output directory when scoring that track. Results are written to `summary.json`, with closed-form summaries and judge audit records in the score directory. To score only closed-form items without judge calls, add `--mode closed`.

Complete prediction coverage, the same inputs and rendering rules, and matching model/judge settings are needed to compare against the paper. API models and judges can vary between runs; the table reports the recorded paper results, rather than guaranteeing identical scores on every rerun.

## Repository guide

| Resource | Purpose |
|---|---|
| [Rendering](reproduction/README.md) | Materialize permitted images and volumes from your MR-RATE download |
| [Inference](eval/README.md) | API adapters, case-reasoning driver, and prediction format |
| [Scoring](scoring/README.md) | Unified scorer, output files, and judge settings |
| [Scoring specification](docs/SCORING.md) | Rubrics, critical-fact checks, and aggregation |
| [Taxonomy](docs/TAXONOMY.md) | Task templates and reasoning levels |
| [Paper results](docs/RESULTS.md) | Track-level metrics for all 20 configurations |
| [Run manifest](reproduction/leaderboard_runs.csv) | Recorded adapter flags and run settings |

## Citation

```bibtex
@inproceedings{lan2026braintrace,
  title     = {BrainTRACE: Tracing Longitudinal, Multimodal, and Volumetric Evidence in Brain MRI Clinical Reasoning},
  author    = {Lan, Qizhen and Fan, Mengchen and Zhang, Hang and Duan, Jingwei and Lin, Moule and Chen, Jialin and Geng, Baocheng and Jiang, Xiaoqian},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {Evaluations and Datasets Track},
  url       = {https://openreview.net/forum?id=hXktXOmwp6}
}
```

## License and contact

- **Code:** [Apache-2.0](LICENSE).
- **Benchmark metadata and task definitions:** CC-BY-NC-SA 4.0, as specified on the dataset card.
- **MR-RATE source imagery, reports, and imagery derivatives:** upstream access conditions and Data Use Agreement; see [DUA_NOTICE.md](DUA_NOTICE.md).

For bugs or clarification requests, open a [GitHub issue](https://github.com/lanqz7766/BrainTRACE/issues). Research enquiries: [qizhen.lan@uth.tmc.edu](mailto:qizhen.lan@uth.tmc.edu).
