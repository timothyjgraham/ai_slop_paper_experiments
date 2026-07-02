# Measuring Aesthetic Homogenization in Text-to-Image AI: Diversity Collapse in DALL-E 3 and Imagen 4

This repository contains the code, data (precomputed embeddings), and figures for the measurement paper above. It releases everything needed to reproduce the paper's synthetic validation, its empirical study of two commercial models, and its open-source ablation.

## Abstract (of the paper)

Text-to-image models produce outputs that are strikingly similar to one another, but the scale of this homogenization has not been systematically measured. We introduce a reproducible methodology for quantifying aesthetic and cultural diversity collapse in generative image models, operationalizing three distributional measures — effective support radius, diversity index, and effective dimension — in CLIP and DINOv2 embedding space. Applying the method to 1,177 images from two leading commercial models, DALL-E 3 and Imagen 4, we find that within-prompt outputs collapse to about 10 of 768 effective dimensions; that minority cultural representations are 17% less diverse than majority ones; and that the two independently developed models converge to cosine similarity 0.86 on identical prompts. A controlled ablation on Stable Diffusion XL isolates the mean-squared-error objective from preference training and shows that, at the standard guidance scale, 91% of the majority/minority diversity gap is already present in a base model with no preference training.

## Key findings

- Commercial models compress within-prompt outputs to roughly **10 effective dimensions out of 768**, discarding over 98% of available aesthetic variation.
- Minority cultural representations are **17% less diverse** than majority ones (*p* < 0.001 by exact cluster-aware permutation test; Hedges' *g* = 1.51).
- Independently developed models (DALL-E 3, Imagen 4) converge to **cosine similarity 0.86** on the same prompts — well above the 0.64 baseline for mismatched prompts; the matched centroid is the nearest cross-model centroid for 27 of 30 prompts.
- The ablation shows **91%** of the majority/minority diversity gap is present, at the standard guidance scale (w = 7.5), in a base model with **no preference training**, indicating the denoising objective — not RLHF — is the primary structural driver.

## Repository structure

```
.
├── README.md                       # This file
├── LICENSE                         # MIT License
├── CITATION.cff                    # How to cite this work
├── requirements.txt                # Python dependencies
├── DATA.md                         # Data dictionary for the embeddings/metadata
├── paper/
│   ├── ccr-measurement.tex         # LaTeX source of the paper
│   ├── ccr-measurement.pdf         # Compiled PDF
│   └── refs.bib                    # Bibliography
├── experiments/
│   ├── run_experiments.py          # Synthetic validation suite (Gaussian mixtures; Section 5 of the paper)
│   ├── figures/                    # Synthetic-validation figures (incl. overview_scatter.png, Fig. 1)
│   ├── results/summary.json        # Synthetic results (the numbers quoted in Section 5)
│   └── empirical/
│       ├── README_EMPIRICAL.md     # Step-by-step guide to the commercial-model study
│       ├── prompts.json            # 30 structured prompts (3 categories)
│       ├── generate_images.py      # API-based image generation (DALL-E 3, Imagen 4)
│       ├── run_empirical_analysis.py  # CLIP embedding + diversity measures + figures (Section 6)
│       ├── revision_bootstrap_ci_fast.py, revision_dinov2_replication.py  # Robustness checks (Section 6.5)
│       ├── analyze_base_ablation.py, revision_ablation_study.py  # SDXL ablation (Section 7)
│       ├── embeddings/             # Precomputed embeddings (the analysis runs from these)
│       │   ├── embeddings.npz      # CLIP ViT-L/14, 1177 × 768, + per-image metadata
│       │   └── embeddings_dinov2.npz  # DINOv2, 1177 × 1024
│       ├── figures/                # Empirical figures + result JSONs
│       └── ablation/
│           ├── ablation_embeddings.npz  # CLIP embeddings for the 880-image SDXL ablation
│           ├── ablation_index.json      # Condition index (model × CFG scale × prompt)
│           └── figures/            # Ablation figures + results
```

## Data availability

The **raw generated images are not redistributed** in this repository, for three reasons: they are outputs of commercial models subject to provider terms; the full set is ~3.6 GB; and the cultural-identity subset consists of AI-generated images of people grouped by nationality, which we prefer not to redistribute. Instead we release everything needed to reproduce the analysis:

- The exact **prompts** (`experiments/empirical/prompts.json`).
- The **image-generation scripts**, so the corpus can be regenerated from the prompts (`generate_images.py`).
- The **precomputed CLIP and DINOv2 embeddings** with full per-image metadata (platform, category, majority/minority group, nationality) — see `DATA.md`.
- The **precomputed CLIP embeddings for the SDXL ablation** (`experiments/empirical/ablation/ablation_embeddings.npz` + `ablation_index.json`), so the ablation results are likewise reproducible without regenerating images.
- All **analysis code** and **result JSONs**.

The analysis pipeline (`run_empirical_analysis.py --analysis-only`) runs directly from the released embeddings, so all reported numbers are reproducible without regenerating images.

### Sample sizes
30 prompts × 20 images × 2 models = 1,200 images designed; 1,198 generated successfully; **1,177 analysed** after filtering failed/invalid generations (OpenAI/DALL-E 3: 598; Google/Imagen 4: 579). By category: cultural_identity 588, open_ended 394, artistic_style 195. Ablation: 2 models × 4 CFG scales × 11 prompts × 10 images = **880 images**.

## Reproducing the experiments

### Synthetic validation (no API keys, no GPU)
```bash
pip install -r requirements.txt
python experiments/run_experiments.py
```

### Empirical analysis from released embeddings (no API keys, no GPU)
```bash
python experiments/empirical/run_empirical_analysis.py --analysis-only \
    --embeddings experiments/empirical/embeddings/embeddings.npz
```

### Regenerating the image corpus (requires API keys + cost)
See `experiments/empirical/README_EMPIRICAL.md` for full instructions. In brief:
```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AIza..."
python experiments/empirical/generate_images.py --platform all --n-per-prompt 20
python experiments/empirical/run_empirical_analysis.py --image-dir ./images --prompts ./prompts.json
```

## Compiling the paper
```bash
cd paper
pdflatex ccr-measurement.tex
bibtex ccr-measurement
pdflatex ccr-measurement.tex
pdflatex ccr-measurement.tex
```
(The figure search path in the source resolves against `../experiments/...` from `paper/`, so no symlinks are needed.)

## Companion paper

The cultural and critical interpretation of these measurements is developed in a companion paper (in submission). This repository contains only the materials for the measurement paper.

## Citation
See `CITATION.cff`. Please cite the paper if you use this code or data.

## License
Code is released under the MIT License (`LICENSE`). The paper text and figures are © the authors.
