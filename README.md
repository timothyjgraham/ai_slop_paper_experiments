# Variance Collapse in Generative AI: The Mode Averaging Principle and the Statistical Levelling of Originality (SLOP)

This repository contains the full source code, experimental suite, and compiled paper for our work on aesthetic homogenisation in diffusion-based generative AI models.

## Abstract

We prove that the denoising diffusion objective is a conditional expectation operator that systematically averages aesthetic modes under noise, with weighting proportional to each mode's statistical mass in the training data. We call this the **Mode Averaging Principle (MAP)** and show that it combines with **classifier-free guidance (CFG)** and **recursive data contamination** to form a compound convergence system that progressively narrows the diversity of generative outputs. The observable consequence — the **Statistical Levelling of Originality Principle (SLOP)** — provides a rigorous mathematical foundation for the cultural critique of "platform realism" and the generic "AI aesthetic."

We validate all theoretical predictions through controlled synthetic experiments on Gaussian mixture distributions, demonstrating mode averaging convergence across five orders of magnitude, prior transmission with deviation below 10⁻⁶, systematic collapse of all three homogenisation metrics, monotonic diversity loss under CFG, and complete absorption of minority modes at high noise.

## Repository Structure

```
.
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── paper/
│   ├── sn-article-v5.tex             # LaTeX source (main paper)
│   ├── sn-bibliography-v5.bib        # Bibliography (35 entries)
│   ├── sn-article-v5.pdf             # Compiled PDF (24 pages)
│   └── sn-jnl.cls                    # Springer Nature journal class
├── experiments/
│   ├── run_experiments.py            # Complete experiment suite
│   ├── figures/                      # Generated figures (7 PNGs)
│   │   ├── overview_scatter.png      # 3-panel: original → denoised
│   │   ├── exp1_mode_averaging.png   # Convergence to global mean
│   │   ├── exp2_prior_transmission.png  # Posterior → prior convergence
│   │   ├── exp3_homogenisation_metrics.png  # ρ_α, δ, γ collapse
│   │   ├── exp4_cfg_amplification.png    # CFG monotonic diversity loss
│   │   ├── exp4_cfg_densities.png        # CFG density contour plots
│   │   └── exp5_minority_suppression.png # Mode absorption dynamics
│   └── results/
│       └── summary.json              # Machine-readable results
└── supplementary/
    └── (reserved for additional materials)
```

## Key Results

All five theoretical predictions are confirmed empirically:

| Prediction | Metric | Predicted | Observed |
|:-----------|:-------|:----------|:---------|
| **P1**: Mode averaging | ‖E[x₀\|xₜ] − μ_global‖ | → 0 | 2.97 × 10⁻⁵ |
| **P2**: Prior transmission | \|P(Cᵢ\|xₜ) − πᵢ\| | → 0 | 1.22 × 10⁻⁶ |
| **P3**: Homogenisation | ρ_α, δ, γ | < 1 | 0.001, ≈0, ≈0 |
| **P4**: CFG amplification | ρ_α, δ, γ at w=7 | decreasing | 0.29, 0.45, 0.54 |
| **P5**: Minority suppression | Mode C share at σ=10 | < πC | 0.00 (eliminated) |

## Reproducing the Experiments

### Prerequisites

- Python 3.8+
- NumPy, SciPy, Matplotlib

### Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### Running the Experiments

```bash
cd experiments
python run_experiments.py
```

This will regenerate all seven figures in `experiments/figures/` and the results summary in `experiments/results/summary.json`. The script uses a fixed random seed (`np.random.seed(42)`) for exact reproducibility. Expected runtime: ~2–5 minutes on a standard laptop.

### Understanding the Output

The experiment suite tests five predictions derived from the Mode Averaging Principle:

**Experiment 1 — Mode Averaging Under High Noise** (`exp1_mode_averaging.png`):
Computes the analytical conditional expectation E[x₀|xₜ] at 25 noise levels and measures its distance to the global mean μ_global = (2.7, 0.3). Confirms convergence across five orders of magnitude as σ/d_mode increases.

**Experiment 2 — Prior Transmission** (`exp2_prior_transmission.png`):
Computes the Bayesian posterior P(Cᵢ|xₜ) using the known mixture parameters and measures deviation from the prior weights π = (0.70, 0.20, 0.10). Shows convergence to 10⁻⁶ at high noise, confirming that the denoising objective transmits training data asymmetry with near-perfect fidelity.

**Experiment 3 — Homogenisation Metrics** (`exp3_homogenisation_metrics.png`):
Measures three complementary diversity metrics on the denoised distribution relative to the original:
- **Effective support radius** (ρ_α): spatial extent, measured via the 95th percentile of distances from the mean
- **Diversity index** (δ): entropic richness, estimated via Gaussian-fit entropy
- **Effective dimension** (γ): distributional complexity, computed as the inverse participation ratio (1/∫p²dx)

All three ratios fall below 1 and approach zero, confirming the SLOP corollary.

**Experiment 4 — CFG Amplification** (`exp4_cfg_amplification.png`, `exp4_cfg_densities.png`):
Computes the CFG-modified density analytically on a 300×300 grid:

```
p̃(x) ∝ p_cond(x)^(1+w) / p_uncond(x)^w
```

Measures entropy, effective dimension, and support radius of the resulting density for guidance scales w ∈ {0, 1, 3, 5, 7, 10, 15}. All metrics decrease monotonically. Density contour plots at w = 0, 7, and 15 visualise the progressive concentration onto the dominant mode.

**Experiment 5 — Minority Mode Suppression** (`exp5_minority_suppression.png`):
Assigns each denoised output to its nearest mode centre (within 2σ) and tracks mode fractions across noise levels. Mode C (πC = 0.10) falls to 1.3% at σ = 5.0 and is completely eliminated at σ = 10.0, while Mode A (πA = 0.70) absorbs the entire output distribution.

## Compiling the Paper

### Prerequisites

- A LaTeX distribution (TeX Live 2021+ recommended)
- The `natbib`, `amsmath`, `graphicx`, and `hyperref` packages

### Compilation

```bash
cd paper
pdflatex sn-article-v5.tex
bibtex sn-article-v5
pdflatex sn-article-v5.tex
pdflatex sn-article-v5.tex
```

Note: The paper references figures from `experiments/figures/`. When compiling, either run `pdflatex` from the repository root, or create a symlink:

```bash
cd paper
ln -s ../experiments experiments
pdflatex sn-article-v5.tex
# ... (bibtex + two more pdflatex passes)
```

A pre-compiled PDF (`sn-article-v5.pdf`) is included for convenience.

## Mathematical Framework

The paper develops three interlocking results:

1. **Lemma 3.1 (Within-Component Shrinkage)**: For a single Gaussian component, E[x₀|xₜ, Cᵢ] = μᵢ + (1/(1+σ²))(xₜ − μᵢ), showing that the conditional expectation shrinks each observation toward its mode centre, with the shrinkage factor approaching total collapse as σ → ∞.

2. **Theorem 3.2 (Mode Averaging Principle)**: The unconditional expectation E[x₀|xₜ] is a weighted average of mode centres with weights P(Cᵢ|xₜ), which converge to the prior πᵢ under high noise. The denoiser therefore outputs the prior-weighted global mean, regardless of input.

3. **Corollary 3.7 (SLOP)**: The denoised distribution Q has strictly lower effective support radius, diversity index, and effective dimension than the training distribution P. The generative process is a contraction mapping on aesthetic diversity.

These are supplemented by:
- **Proposition 3.4**: CFG amplifies mode concentration by raising the conditional density to power (1+w)
- **Proposition 3.5**: Trajectory lock-in connects per-step averaging to output homogenisation

## Cultural Theory

The paper interprets these mathematical results through three cultural-theoretic lenses:

- **Platform realism** (Meyer 2023): The convergence to the statistical mean corresponds to the "platform realistic" aesthetic observed by cultural critics
- **Mimicry** (Bhabha 1984, 1994): The posterior convergence mechanism produces outputs that are "almost the same, but not quite" — mimicking cultural diversity while flattening it
- **Statistical monoculture**: Minority aesthetic traditions with low statistical mass are algorithmically marginalised, not through data bias, but through the mathematics of conditional expectation

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{slop2026,
  title={Variance Collapse in Generative AI: The Mode Averaging Principle
         and the Statistical Levelling of Originality Principle (SLOP)},
  author={[Authors]},
  year={2026},
  note={Preprint. Code available at https://github.com/<your-username>/<repo-name>}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The paper content (LaTeX source and PDF) is copyright the authors. The experiment code is released under MIT for full reproducibility.
