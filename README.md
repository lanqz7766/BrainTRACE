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

BrainTRACE benchmarks longitudinal, multimodal, and volumetric reasoning over brain MRI.

![BrainTRACE benchmark overview](docs/assets/braintrace-overview.png)

| Scored VQA instances | Longitudinal patients | MRI studies | Model configurations |
|:---:|:---:|:---:|:---:|
| **7,273** | **1,778** | **7,299** | **20** |

## Benchmark design

<p align="center">
  <a href="docs/assets/case-reasoning-example.png"><img src="docs/assets/case-reasoning-example.png" width="900" alt="Figure 2: longitudinal MRI case and task examples across five reasoning levels"></a><br>
  <sub>Figure 2 · Longitudinal case and task examples</sub>
</p>

<p align="center">
  <a href="docs/assets/benchmark-construction.png"><img src="docs/assets/benchmark-construction.png" width="900" alt="Figure 3: report-grounded benchmark construction pipeline, with 7,273 scored instances"></a><br>
  <sub>Figure 3 · Benchmark construction</sub>
</p>

## Benchmark coverage

<p align="center">
  <a href="docs/assets/benchmark-coverage.png"><img src="docs/assets/benchmark-coverage.png" width="40%" align="middle" alt="Task coverage across five reasoning levels and three evidence conditions"></a>
  <a href="docs/assets/cohort-diversity.png"><img src="docs/assets/cohort-diversity.png" width="56%" align="middle" alt="Cohort demographics, longitudinal depth, MRI sequence availability, and acquisition mix"></a>
</p>

## Key results

<p align="center">
  <a href="docs/assets/closed-form-results.png"><img src="docs/assets/closed-form-results.png" width="49%" align="top" alt="Paper Table 2: closed-form accuracy across all model configurations, including its caption and notes"></a>
  <a href="docs/assets/open-case-results.png"><img src="docs/assets/open-case-results.png" width="49%" align="top" alt="Paper Table 3: open-ended and case reasoning results, including the separate physician sanity-check note"></a>
</p>

<p align="center">
  <a href="docs/assets/case-reasoning-attrition.png"><img src="docs/assets/case-reasoning-attrition.png" width="400" alt="Figure 4: case reasoning attrition across six steps"></a>
</p>

[Full metrics and scoring details →](docs/RESULTS.md)

## Get started

Download the [benchmark metadata](https://huggingface.co/datasets/BrainTRACE-anon/BrainTRACE) and obtain [MR-RATE](https://huggingface.co/datasets/Forithmus/MR-RATE) under its Data Use Agreement. MRI imagery is not redistributed.

<details>
<summary><strong>Quickstart commands</strong></summary>

### 1. Install and download metadata

```bash
git clone https://github.com/lanqz7766/BrainTRACE.git
cd BrainTRACE
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -c "from huggingface_hub import snapshot_download; snapshot_download('BrainTRACE-anon/BrainTRACE', repo_type='dataset', local_dir='./braintrace_dataset')"
```

### 2. Render MR-RATE inputs

Place MR-RATE under `./mr_rate` using the layout in the [rendering guide](reproduction/README.md).

```bash
python reproduction/render_images.py \
  --dataset ./braintrace_dataset \
  --mr-rate-root ./mr_rate \
  --out-root ./braintrace_dataset
```

### 3. Run a VLM

Set `OPENAI_API_KEY` in your shell for this OpenAI example and the default scoring judge.

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

See the [inference guide](eval/README.md) for other APIs, case reasoning, and prediction formats. Open-weight and native-volume models require separate backends.

### 4. Score predictions

```bash
python scoring/score.py \
  --dataset ./braintrace_dataset \
  --predictions ./outputs/gpt-5.4/main_outputs \
  --out-dir ./scores/gpt-5.4 \
  --judge-model gpt-4o-mini-2024-07-18
```

Outputs include `summary.json` and judge audit records. Add `--mode closed` to score closed-form items without judge calls; use the case output directory for case-reasoning predictions. Paper comparisons require the recorded inputs and model/judge settings in the [run manifest](reproduction/leaderboard_runs.csv).

</details>

**Documentation:** [Rendering](reproduction/README.md) · [Inference](eval/README.md) · [Scoring](scoring/README.md) · [Rubrics](docs/SCORING.md) · [Taxonomy](docs/TAXONOMY.md) · [Full results](docs/RESULTS.md) · [Run manifest](reproduction/leaderboard_runs.csv)

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
