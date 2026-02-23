"""
Synthetic Experiment Suite for the SLOP Paper (v2)
===================================================
Validates the Mode Averaging Principle (MAP) and its predictions
on a controlled 2D Gaussian mixture distribution.

Predictions tested:
  1. Mode averaging under high noise  (E[x0|xt] → μ_global)
  2. Prior transmission               (P(Ci|xt) → πi)
  3. Homogenisation metrics            (ρ_α, δ, γ < 1)
  4. CFG distributional sharpening     (entropy drops with guidance scale)
  5. Minority mode suppression         (low-π modes absorbed)
"""

import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ─── Configuration ───────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# ── Mixture parameters ──
# Design rationale:
#   - 3 modes with well-separated centres (d_mode ≈ 8.6, ~8.6σ apart)
#     so that modes are clearly distinguishable at low noise.
#   - Prior weights 0.70 / 0.20 / 0.10 model a "dominant / secondary /
#     minority" scenario.  This is *conservative*: real web-scraped
#     datasets have far more extreme asymmetry.
#   - Identical isotropic unit covariances: the most neutral geometric
#     choice — no mode has a structural advantage in shape or spread.
MODES = {
    "A_dominant":  {"mu": np.array([ 5.0,  0.0]), "sigma": np.eye(2), "pi": 0.70},
    "B_secondary": {"mu": np.array([-3.0,  4.0]), "sigma": np.eye(2), "pi": 0.20},
    "C_minority":  {"mu": np.array([-2.0, -5.0]), "sigma": np.eye(2), "pi": 0.10},
}

MU    = np.array([m["mu"]    for m in MODES.values()])   # (3, 2)
SIGMA = np.array([m["sigma"] for m in MODES.values()])   # (3, 2, 2)
PI    = np.array([m["pi"]    for m in MODES.values()])   # (3,)
K     = len(PI)

MU_GLOBAL = PI @ MU        # prior-weighted mean = (2.7, 0.3)
MU_GEOM   = MU.mean(axis=0)  # geometric (uniform) centre

D_MODE = min(
    np.linalg.norm(MU[i] - MU[j])
    for i in range(K) for j in range(i + 1, K)
)

# Noise levels (log-spaced)
NOISE_LEVELS = np.logspace(np.log10(0.1), np.log10(50), 25)

N_SAMPLES = 50_000
ALPHA     = 0.90    # for effective support radius

# ── Plot styling ──
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})
MODE_COLORS = ["#2166ac", "#b2182b", "#4dac26"]
MODE_LABELS = [
    r"Mode A (dominant, $\pi$=0.70)",
    r"Mode B (secondary, $\pi$=0.20)",
    r"Mode C (minority, $\pi$=0.10)",
]


# ═══════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def sample_mixture(n):
    """Sample n points from the 3-component Gaussian mixture."""
    assignments = np.random.choice(K, size=n, p=PI)
    samples = np.zeros((n, 2))
    for i in range(K):
        mask = assignments == i
        samples[mask] = np.random.multivariate_normal(
            MU[i], SIGMA[i], size=mask.sum()
        )
    return samples, assignments


def forward_diffuse(x0, sigma_t):
    """x_t = √ᾱ_t · x_0 + σ_t · ε,  with ᾱ_t = 1/(1+σ_t²)."""
    alpha_bar_t = 1.0 / (1.0 + sigma_t**2)
    sqrt_ab = np.sqrt(alpha_bar_t)
    eps = np.random.randn(*x0.shape)
    xt = sqrt_ab * x0 + sigma_t * eps
    return xt, alpha_bar_t


def compute_posterior(xt, sigma_t, prior=None):
    """P(Ci | xt) via Bayes, using an arbitrary prior vector."""
    if prior is None:
        prior = PI
    alpha_bar_t = 1.0 / (1.0 + sigma_t**2)
    sqrt_ab = np.sqrt(alpha_bar_t)
    n = xt.shape[0]
    log_posteriors = np.zeros((n, K))
    for i in range(K):
        mean_i = sqrt_ab * MU[i]
        cov_i  = alpha_bar_t * SIGMA[i] + sigma_t**2 * np.eye(2)
        rv = multivariate_normal(mean=mean_i, cov=cov_i)
        log_posteriors[:, i] = np.log(prior[i] + 1e-300) + rv.logpdf(xt)
    log_posteriors -= log_posteriors.max(axis=1, keepdims=True)
    posteriors = np.exp(log_posteriors)
    posteriors /= posteriors.sum(axis=1, keepdims=True)
    return posteriors


def compute_conditional_expectation(xt, sigma_t):
    """E[x₀ | x_t] = Σ P(Ci|xt) · E[x₀ | xt, Ci]  (Lemma 3.1)."""
    alpha_bar_t = 1.0 / (1.0 + sigma_t**2)
    sqrt_ab = np.sqrt(alpha_bar_t)
    posteriors = compute_posterior(xt, sigma_t)
    n = xt.shape[0]
    result = np.zeros((n, 2))
    for i in range(K):
        cov_xt = alpha_bar_t * SIGMA[i] + sigma_t**2 * np.eye(2)
        S = alpha_bar_t * SIGMA[i] @ np.linalg.inv(cov_xt)
        dev = xt / sqrt_ab - MU[i]
        E_i = MU[i] + dev @ S.T
        result += posteriors[:, i:i+1] * E_i
    return result


# ── Metric helpers ────────────────────────────────────────────

def effective_support_radius(samples, alpha=ALPHA):
    """R_α: radius around mean containing α-fraction of mass."""
    mu = samples.mean(axis=0)
    dists = np.linalg.norm(samples - mu, axis=1)
    return np.percentile(dists, alpha * 100)


def diversity_index_from_samples(samples):
    """D(P) = exp(H), using Gaussian-fit entropy (upper-bound proxy).

    For a mixture, the true differential entropy is ≤ the Gaussian with
    the same covariance, so this slightly overestimates diversity.
    Crucially, the *ratio* D(Q)/D(P) using the same estimator on both
    distributions remains a valid measure of relative concentration.
    """
    cov = np.cov(samples.T)
    d   = samples.shape[1]
    entropy = 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * np.log(
        np.linalg.det(cov) + 1e-300
    )
    return np.exp(entropy)


def effective_dimension_from_samples(samples, n_bins=80):
    """d_eff = 1/∫p²dx, estimated via 2-D histogram."""
    H, xe, ye = np.histogram2d(
        samples[:, 0], samples[:, 1], bins=n_bins, density=True
    )
    dx = xe[1] - xe[0]
    dy = ye[1] - ye[0]
    integral_p2 = np.sum(H**2) * dx * dy
    return 1.0 / max(integral_p2, 1e-300)


def mixture_density(x, weights):
    """Evaluate Gaussian mixture density at points x with given weights."""
    result = np.zeros(x.shape[0])
    for i in range(K):
        rv = multivariate_normal(mean=MU[i], cov=SIGMA[i])
        result += weights[i] * rv.pdf(x)
    return result


# ── Grid-based metrics (for analytical density comparisons) ──

def make_grid(bounds=(-12, 14, -12, 10), n=300):
    x = np.linspace(bounds[0], bounds[1], n)
    y = np.linspace(bounds[2], bounds[3], n)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return grid, xx, yy, dx, dy


def grid_entropy(density, dx, dy):
    """Differential entropy from a grid-evaluated density."""
    p = density[density > 0]
    return -np.sum(p * np.log(p)) * dx * dy


def grid_effective_dimension(density, dx, dy):
    return 1.0 / max(np.sum(density**2) * dx * dy, 1e-300)


def grid_support_radius(grid, density, dx, dy, alpha=ALPHA):
    """α-quantile radius from the density's mean."""
    total = np.sum(density) * dx * dy
    p_norm = density / total
    mu = np.array([
        np.sum(grid[:, 0] * p_norm) * dx * dy,
        np.sum(grid[:, 1] * p_norm) * dx * dy,
    ])
    dists = np.linalg.norm(grid - mu, axis=1)
    # sort by distance, accumulate mass
    order = np.argsort(dists)
    cum = np.cumsum(density[order]) * dx * dy
    cum /= cum[-1]
    idx = np.searchsorted(cum, alpha)
    return dists[order[min(idx, len(dists) - 1)]]


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 1 — Mode Averaging Under High Noise
# ═══════════════════════════════════════════════════════════════

def experiment_1():
    """E[x₀|x_t] → μ_global as noise ↑."""
    print("Running Experiment 1: Mode Averaging Under High Noise …")
    x0, _ = sample_mixture(N_SAMPLES)
    distances = []
    for sigma_t in NOISE_LEVELS:
        xt, _ = forward_diffuse(x0, sigma_t)
        E_x0 = compute_conditional_expectation(xt, sigma_t)
        distances.append(np.linalg.norm(E_x0.mean(axis=0) - MU_GLOBAL))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(NOISE_LEVELS / D_MODE, distances, "o-",
            color="#2166ac", markersize=4, lw=1.5)
    ax.axhline(0, color="gray", ls="--", alpha=.5)
    ax.set_xlabel(r"Noise ratio $\sigma_t\,/\,d_{\mathrm{mode}}$")
    ax.set_ylabel(
        r"$\|\,\overline{\mathbb{E}[x_0 \mid x_t]}"
        r"\;-\;\mu_{\mathrm{global}}\|$"
    )
    ax.set_title("Prediction 1: Conditional expectation converges to global mean")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, alpha=.3)
    fig.savefig(OUTPUT_DIR / "exp1_mode_averaging.png")
    plt.close()
    print(f"  σ/d=0.01 → dist = {distances[0]:.4f}")
    print(f"  σ/d=5.8  → dist = {distances[-1]:.6f}")
    return distances


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 2 — Prior Transmission
# ═══════════════════════════════════════════════════════════════

def experiment_2():
    """P(Ci|x_t) → πi as noise ↑."""
    print("Running Experiment 2: Prior Transmission …")
    x0, _ = sample_mixture(N_SAMPLES)
    devs, posts = [], []
    for sigma_t in NOISE_LEVELS:
        xt, _ = forward_diffuse(x0, sigma_t)
        post = compute_posterior(xt, sigma_t).mean(axis=0)
        posts.append(post)
        devs.append(np.mean(np.abs(post - PI)))
    posts = np.array(posts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(NOISE_LEVELS / D_MODE, devs, "o-",
             color="#2166ac", ms=4, lw=1.5)
    ax1.axhline(0, color="gray", ls="--", alpha=.5)
    ax1.set_xlabel(r"$\sigma_t / d_{\mathrm{mode}}$")
    ax1.set_ylabel(r"Mean $|P(C_i|x_t) - \pi_i|$")
    ax1.set_title("Posterior deviation from prior")
    ax1.set_xscale("log"); ax1.grid(True, alpha=.3)

    for i in range(K):
        ax2.plot(NOISE_LEVELS / D_MODE, posts[:, i], "o-",
                 color=MODE_COLORS[i], ms=4, lw=1.5, label=MODE_LABELS[i])
        ax2.axhline(PI[i], color=MODE_COLORS[i], ls="--", alpha=.5)
    ax2.set_xlabel(r"$\sigma_t / d_{\mathrm{mode}}$")
    ax2.set_ylabel(r"Mean $P(C_i \mid x_t)$")
    ax2.set_title("Posterior weights → prior weights (dashed)")
    ax2.set_xscale("log"); ax2.legend(fontsize=8); ax2.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp2_prior_transmission.png")
    plt.close()
    print(f"  Low-noise dev:  {devs[0]:.4f}")
    print(f"  High-noise dev: {devs[-1]:.6f}")
    return devs


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 3 — Homogenisation Metrics
# ═══════════════════════════════════════════════════════════════

def experiment_3():
    """ρ_α, δ, γ  < 1 for denoised distributions."""
    print("Running Experiment 3: Homogenisation Metrics …")
    x0, _ = sample_mixture(N_SAMPLES)
    R_P     = effective_support_radius(x0)
    D_P     = diversity_index_from_samples(x0)
    deff_P  = effective_dimension_from_samples(x0)

    rho, delta, gamma = [], [], []
    for sigma_t in NOISE_LEVELS:
        xt, _ = forward_diffuse(x0, sigma_t)
        E_x0 = compute_conditional_expectation(xt, sigma_t)
        rho.append(effective_support_radius(E_x0) / R_P)
        delta.append(diversity_index_from_samples(E_x0) / D_P)
        gamma.append(effective_dimension_from_samples(E_x0) / deff_P)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(NOISE_LEVELS / D_MODE, rho, "o-",   color="#2166ac", ms=4, lw=1.5,
            label=r"$\rho_\alpha$ (support concentration)")
    ax.plot(NOISE_LEVELS / D_MODE, delta, "s-",  color="#b2182b", ms=4, lw=1.5,
            label=r"$\delta$ (diversity loss)")
    ax.plot(NOISE_LEVELS / D_MODE, gamma, "^-",  color="#4dac26", ms=4, lw=1.5,
            label=r"$\gamma$ (dimensional collapse)")
    ax.axhline(1.0, color="gray", ls="--", alpha=.5, label="No change")
    ax.set_xlabel(r"$\sigma_t / d_{\mathrm{mode}}$")
    ax.set_ylabel("Ratio  $Q\,/\,P$")
    ax.set_title("Prediction 3: All homogenisation metrics < 1")
    ax.set_xscale("log"); ax.legend(fontsize=9)
    ax.grid(True, alpha=.3); ax.set_ylim(0, 1.5)
    fig.savefig(OUTPUT_DIR / "exp3_homogenisation_metrics.png")
    plt.close()
    print(f"  At σ/d = {NOISE_LEVELS[-1]/D_MODE:.1f}:")
    print(f"    ρ = {rho[-1]:.4f},  δ = {delta[-1]:.4f},  γ = {gamma[-1]:.4f}")
    return rho, delta, gamma


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 4 — CFG Distributional Sharpening  (REVISED)
# ═══════════════════════════════════════════════════════════════

def experiment_4():
    """CFG sharpens the conditional density, measured on a grid.

    We compute the CFG-modified density analytically:
        p̃(x) ∝ p_cond(x)^{1+w}  /  p_uncond(x)^w
    where p_cond uses the real (skewed) prior and p_uncond uses a
    uniform prior.  We then measure entropy, effective dimension,
    and effective support radius of p̃ for varying w.
    """
    print("Running Experiment 4: CFG Distributional Sharpening …")
    grid, xx, yy, dx, dy = make_grid()

    # Baseline: conditional density (the skewed mixture)
    p_cond   = mixture_density(grid, PI)
    H_base   = grid_entropy(p_cond, dx, dy)
    d_base   = grid_effective_dimension(p_cond, dx, dy)
    R_base   = grid_support_radius(grid, p_cond, dx, dy)

    # Unconditional: uniform-prior mixture
    pi_unif  = np.ones(K) / K
    p_uncond = mixture_density(grid, pi_unif)

    w_values = [0, 0.5, 1, 2, 3, 5, 7, 10, 15]
    H_ratios, d_ratios, R_ratios = [], [], []

    for w in w_values:
        # p̃ ∝ p_cond^(1+w) / p_uncond^w
        log_p = (1 + w) * np.log(p_cond + 1e-300) - w * np.log(p_uncond + 1e-300)
        log_p -= log_p.max()
        p_cfg = np.exp(log_p)
        # normalise
        Z = np.sum(p_cfg) * dx * dy
        p_cfg /= Z

        H_ratios.append(np.exp(grid_entropy(p_cfg, dx, dy)) /
                         np.exp(H_base))
        d_ratios.append(grid_effective_dimension(p_cfg, dx, dy) / d_base)
        R_ratios.append(grid_support_radius(grid, p_cfg, dx, dy) / R_base)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(w_values, R_ratios, "o-", color="#2166ac", ms=5, lw=1.5,
            label=r"$\rho_\alpha$ (support)")
    ax.plot(w_values, H_ratios, "s-", color="#b2182b", ms=5, lw=1.5,
            label=r"$\delta$ (diversity)")
    ax.plot(w_values, d_ratios, "^-", color="#4dac26", ms=5, lw=1.5,
            label=r"$\gamma$ (dimension)")
    ax.axhline(1.0, color="gray", ls="--", alpha=.5, label="Baseline ($w$=0)")
    ax.set_xlabel("CFG guidance scale  $w$")
    ax.set_ylabel("Ratio relative to $w$=0 baseline")
    ax.set_title("Prediction 4: CFG sharpens the conditional distribution")
    ax.legend(fontsize=9); ax.grid(True, alpha=.3)
    ax.set_ylim(0, 1.15)
    fig.savefig(OUTPUT_DIR / "exp4_cfg_amplification.png")
    plt.close()

    print(f"  w= 0: ρ={R_ratios[0]:.3f}  δ={H_ratios[0]:.3f}  γ={d_ratios[0]:.3f}")
    print(f"  w= 7: ρ={R_ratios[6]:.3f}  δ={H_ratios[6]:.3f}  γ={d_ratios[6]:.3f}")
    print(f"  w=15: ρ={R_ratios[-1]:.3f}  δ={H_ratios[-1]:.3f}  γ={d_ratios[-1]:.3f}")

    # ── Bonus panel: visualise the density at w=0, 7, 15 ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, w in enumerate([0, 7, 15]):
        log_p = (1 + w) * np.log(p_cond + 1e-300) - w * np.log(p_uncond + 1e-300)
        log_p -= log_p.max()
        p_cfg = np.exp(log_p)
        p_cfg /= (np.sum(p_cfg) * dx * dy)
        ax = axes2[idx]
        im = ax.contourf(xx, yy, p_cfg.reshape(xx.shape), levels=30, cmap="Blues")
        for i in range(K):
            ax.plot(*MU[i], "x", color=MODE_COLORS[i], ms=10, mew=2)
        ax.plot(*MU_GLOBAL, "*", color="black", ms=12)
        ax.set_title(f"$w = {w}$")
        ax.set_xlabel("$x_1$"); ax.set_aspect("equal")
        if idx == 0:
            ax.set_ylabel("$x_2$")
    fig2.suptitle("CFG-modified density: sharpening with increasing guidance scale",
                   fontsize=13, y=1.02)
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "exp4_cfg_densities.png")
    plt.close()

    return w_values, R_ratios, H_ratios, d_ratios


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 5 — Minority Mode Suppression
# ═══════════════════════════════════════════════════════════════

def experiment_5():
    """Minority modes absorb into the dominant mode."""
    print("Running Experiment 5: Minority Mode Suppression …")
    x0, _ = sample_mixture(N_SAMPLES)

    sigmas = [0.5, 1.0, 2.0, 5.0, 10.0]
    results = {}
    for sigma_t in sigmas:
        xt, _ = forward_diffuse(x0, sigma_t)
        E_x0 = compute_conditional_expectation(xt, sigma_t)
        dists = cdist(E_x0, MU)
        asgn  = dists.argmin(axis=1)
        results[sigma_t] = np.array([(asgn == i).mean() for i in range(K)])

    fig, ax = plt.subplots(figsize=(9, 5))
    xp = np.arange(len(sigmas)); w = 0.22
    for i in range(K):
        fracs = [results[s][i] for s in sigmas]
        ax.bar(xp + i*w, fracs, w, color=MODE_COLORS[i],
               label=MODE_LABELS[i], alpha=.85)
        ax.hlines(PI[i], xp[0]+i*w-.1, xp[-1]+i*w+w+.1,
                  colors=MODE_COLORS[i], ls="--", alpha=.6)
    ax.set_xlabel(r"Noise level $\sigma_t$")
    ax.set_ylabel("Fraction assigned to mode")
    ax.set_title("Prediction 5: Minority mode suppression (dashed = prior)")
    ax.set_xticks(xp + w)
    ax.set_xticklabels([f"σ={s}" for s in sigmas])
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=.2, axis="y")
    fig.savefig(OUTPUT_DIR / "exp5_minority_suppression.png")
    plt.close()

    for s in sigmas:
        f = results[s]
        print(f"  σ={s:5.1f}: A={f[0]:.3f}  B={f[1]:.3f}  C={f[2]:.3f}")
    return results


# ═══════════════════════════════════════════════════════════════
#  OVERVIEW FIGURE — Original vs Denoised Scatter
# ═══════════════════════════════════════════════════════════════

def figure_overview():
    print("Generating overview scatter plot …")
    x0, asgn = sample_mixture(10_000)
    configs = [
        (None,          "Original distribution $P$"),
        (D_MODE * 0.5,  r"Denoised $\mathbb{E}[x_0|x_t]$ at $\sigma/d$=0.5"),
        (D_MODE * 3.0,  r"Denoised $\mathbb{E}[x_0|x_t]$ at $\sigma/d$=3.0"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (sig, title) in zip(axes, configs):
        if sig is None:
            pts = x0
        else:
            xt, _ = forward_diffuse(x0, sig)
            pts = compute_conditional_expectation(xt, sig)
        for i in range(K):
            m = asgn == i
            ax.scatter(pts[m, 0], pts[m, 1], s=1, alpha=.3,
                       color=MODE_COLORS[i],
                       label=MODE_LABELS[i] if sig is None else None)
        ax.plot(*MU_GLOBAL, "*", color="black", ms=14, zorder=5,
                label=(r"$\mu_{\mathrm{global}}$" if sig is None else None))
        for i in range(K):
            ax.plot(*MU[i], "x", color=MODE_COLORS[i], ms=10, mew=2, zorder=5)
        ax.set_title(title)
        ax.set_xlabel("$x_1$"); ax.set_xlim(-10, 12); ax.set_ylim(-10, 10)
        ax.set_aspect("equal"); ax.grid(True, alpha=.2)
        if sig is None:
            ax.set_ylabel("$x_2$")
            ax.legend(fontsize=7, markerscale=5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "overview_scatter.png")
    plt.close()
    print("  Done.")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("SLOP Paper — Synthetic Experiment Suite  (v2)")
    print("=" * 60)
    print(f"  K = {K}   d_mode = {D_MODE:.2f}")
    print(f"  μ_global = ({MU_GLOBAL[0]:.1f}, {MU_GLOBAL[1]:.1f})")
    print(f"  N = {N_SAMPLES:,}")
    print(f"  Noise levels: {len(NOISE_LEVELS)}  "
          f"[{NOISE_LEVELS[0]:.2f} … {NOISE_LEVELS[-1]:.1f}]")
    print(f"  Output → {OUTPUT_DIR}")
    print("=" * 60)

    figure_overview()
    e1 = experiment_1()
    e2 = experiment_2()
    e3 = experiment_3()
    e4 = experiment_4()
    e5 = experiment_5()

    summary = {
        "parameters": {
            "K": K, "d_mode": float(D_MODE),
            "priors": PI.tolist(),
            "mu_global": MU_GLOBAL.tolist(),
            "N": N_SAMPLES,
        },
        "results": {
            "exp1_high_noise_dist": float(e1[-1]),
            "exp2_high_noise_dev":  float(e2[-1]),
            "exp3_at_max_noise": {
                "rho": float(e3[0][-1]),
                "delta": float(e3[1][-1]),
                "gamma": float(e3[2][-1]),
            },
            "exp4_cfg": {
                "w7_rho":   float(e4[1][6]),
                "w7_delta": float(e4[2][6]),
                "w7_gamma": float(e4[3][6]),
            },
        },
        "predictions_confirmed": {
            "P1_mode_averaging":   bool(e1[-1] < 0.1),
            "P2_prior_transmission": bool(e2[-1] < 0.01),
            "P3_rho_below_1":  bool(all(r < 1.0 for r in e3[0][5:])),
            "P3_delta_below_1": bool(all(d < 1.0 for d in e3[1][5:])),
            "P3_gamma_below_1": bool(all(g < 1.0 for g in e3[2][5:])),
            "P4_cfg_decreasing": bool(all(
                e4[2][j+1] <= e4[2][j] + 0.01  # δ non-increasing (tolerance)
                for j in range(len(e4[2]) - 1)
            )),
        },
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for k, v in summary["predictions_confirmed"].items():
        print(f"  {k}: {'✓ CONFIRMED' if v else '✗ CHECK'}")
    print(f"\nFigures → {OUTPUT_DIR}/")
