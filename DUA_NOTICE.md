# Data Use Notice — BrainTRACE

This repository contains **code only** (renderer, evaluation adapter,
scorers). It does not contain medical imagery or clinical reports. Everything
those scripts process is your own copy of MR-RATE, obtained directly from the
upstream maintainer.

## Before you run anything in this repository

1. **Read and sign the MR-RATE Data Use Agreement.**
   MR-RATE is hosted at <https://huggingface.co/datasets/Forithmus/MR-RATE>
   and is gated behind a DUA. Sign it through the HF interface, then download
   the dataset to your local machine.
2. **Do not redistribute MR-RATE imagery, reports, or any imagery derived
   from MR-RATE** — including images or `.npy` volumes produced by
   `render_images.py` in this repository. The MR-RATE DUA prohibits
   redistribution of the source dataset and its derivatives.
3. **Do not attempt to re-identify any individual** whose data appears in
   MR-RATE, whether directly or by combining MR-RATE with external sources.
4. **Use this benchmark only for non-commercial research.** Any commercial
   use requires explicit written permission from the upstream MR-RATE
   maintainer.
5. **Do not use this benchmark for clinical decision-making, patient-facing
   tools, or any other clinical deployment** without an independent
   prospective validation study reviewed by appropriate ethics and
   regulatory bodies.

## What this repository redistributes

| Component | Redistributed here | Source / License |
|---|---|---|
| Renderer code (this repository) | ✅ Yes | Original work, Apache-2.0 |
| Evaluation adapters and scorers (this repository) | ✅ Yes | Original work, Apache-2.0 |
| BrainTRACE task definitions / ground truth | Hosted separately on HF | Original work, CC-BY-NC-SA 4.0 |
| MR-RATE imagery / NIfTI / DICOM / reports | ❌ No — fetch from upstream | MR-RATE DUA |
| Rendered PNG mosaics or `.npy` volumes | ❌ No — render locally | MR-RATE DUA |

## License of this code

The contents of this repository (Python, shell, Markdown) are released under
the [Apache License 2.0](./LICENSE). This permits non-commercial **and**
commercial use of the code itself, but does not relax the MR-RATE DUA terms
that govern the data the code processes.

## Citation

Please cite both the BrainTRACE dataset and the upstream MR-RATE dataset in
any publication that uses this code. See the dataset card on Hugging Face
for the BibTeX block.
