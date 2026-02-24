#!/usr/bin/env python3
"""
Empirical Analysis Pipeline for MAP/SLOP Validation
=====================================================
Takes a folder of AI-generated images (from generate_images.py or manually
collected), embeds them with CLIP, computes the paper's homogenisation
metrics (rho, delta, gamma), and produces publication-ready figures.

Setup (Google Colab recommended):
  pip install torch torchvision transformers Pillow numpy scipy matplotlib tqdm

Usage:
  # Full pipeline: embed + analyse + plot
  python run_empirical_analysis.py --image-dir ./images --prompts ./prompts.json

  # Just embedding (e.g., to check CLIP works)
  python run_empirical_analysis.py --image-dir ./images --embeddings-only

  # Just analysis (if embeddings already computed)
  python run_empirical_analysis.py --embeddings embeddings/embeddings.npz --analysis-only

Outputs:
  embeddings/embeddings.npz          — CLIP embeddings + metadata
  figures/analysis1_within_prompt_diversity.png
  figures/analysis2_cultural_convergence.png
  figures/analysis2_cultural_heatmap.png
  figures/analysis3_cross_model_agreement.png
  figures/empirical_summary.json      — All metrics in machine-readable format
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Configuration ─────────────────────────────────────────────

ALPHA = 0.90   # For effective support radius (90th percentile)
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"  # 768-dim embeddings
CLIP_DIM = 768

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# Colours matching the paper's existing figure style
PLATFORM_COLORS = {"openai": "#2166ac", "google": "#b2182b"}
CATEGORY_COLORS = {
    "cultural_identity": "#2166ac",
    "open_ended": "#b2182b",
    "artistic_style": "#4dac26",
}
GROUP_COLORS = {"majority": "#2166ac", "minority": "#b2182b"}


# ═══════════════════════════════════════════════════════════════
#  STEP 1: CLIP EMBEDDING
# ═══════════════════════════════════════════════════════════════

def load_clip_model():
    """Load CLIP model and processor. Works on CPU or GPU."""
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        print("ERROR: Install dependencies: pip install torch transformers")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model ({CLIP_MODEL_NAME}) on {device}...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()
    print(f"  CLIP loaded. Embedding dimension: {CLIP_DIM}")
    return model, processor, device


def embed_images(image_dir, prompts_data, output_path, batch_size=32):
    """Embed all images with CLIP and save to .npz file.

    Returns (embeddings, metadata) where:
      - embeddings: np.array of shape (N, 768)
      - metadata: list of dicts with keys: path, prompt_id, category, platform, group
    """
    import torch
    from PIL import Image
    from tqdm import tqdm

    model, processor, device = load_clip_model()

    # Build prompt lookup
    prompt_lookup = {p["id"]: p for p in prompts_data["prompts"]}

    # Discover all images
    image_dir = Path(image_dir)
    image_records = []

    for platform in ["openai", "google"]:
        platform_dir = image_dir / platform
        if not platform_dir.exists():
            print(f"  Warning: {platform_dir} not found, skipping {platform}")
            continue

        # Check for metadata.json first (from generate_images.py)
        meta_path = platform_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            for img in meta["images"]:
                full_path = image_dir / img["path"]
                if full_path.exists():
                    prompt_info = prompt_lookup.get(img["prompt_id"], {})
                    image_records.append({
                        "path": str(full_path),
                        "prompt_id": img["prompt_id"],
                        "category": img["category"],
                        "platform": platform,
                        "group": prompt_info.get("group", "unknown"),
                        "nationality": prompt_info.get("nationality", ""),
                    })
        else:
            # Manual fallback: scan for images matching naming convention
            for img_path in sorted(platform_dir.rglob("*.png")):
                # Try to extract prompt_id from filename: {prompt_id}_{index}.png
                stem = img_path.stem
                parts = stem.rsplit("_", 1)
                if len(parts) == 2:
                    prompt_id = parts[0]
                else:
                    prompt_id = stem

                prompt_info = prompt_lookup.get(prompt_id, {})
                category = prompt_info.get("category", img_path.parent.name)
                image_records.append({
                    "path": str(img_path),
                    "prompt_id": prompt_id,
                    "category": category,
                    "platform": platform,
                    "group": prompt_info.get("group", "unknown"),
                    "nationality": prompt_info.get("nationality", ""),
                })

    if not image_records:
        print("ERROR: No images found! Check --image-dir path.")
        print(f"  Looked in: {image_dir}/openai/ and {image_dir}/google/")
        sys.exit(1)

    print(f"\nFound {len(image_records)} images across "
          f"{len(set(r['platform'] for r in image_records))} platform(s)")

    # Embed in batches
    all_embeddings = []
    print("Computing CLIP embeddings...")

    for batch_start in tqdm(range(0, len(image_records), batch_size),
                            desc="Embedding", unit="batch"):
        batch_records = image_records[batch_start:batch_start + batch_size]
        images = []
        for rec in batch_records:
            try:
                img = Image.open(rec["path"]).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"  Warning: could not open {rec['path']}: {e}")
                images.append(Image.new("RGB", (224, 224)))  # placeholder

        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            features = model.get_image_features(**inputs)
            # Handle newer transformers that return BaseModelOutput
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output if hasattr(features, 'pooler_output') else features[0]
            # L2 normalise (standard practice for CLIP)
            features = features / features.norm(dim=-1, keepdim=True)

        all_embeddings.append(features.cpu().numpy())

    embeddings = np.vstack(all_embeddings)
    print(f"  Embedding matrix shape: {embeddings.shape}")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        metadata=json.dumps(image_records),
    )
    print(f"  Saved to {output_path}")

    return embeddings, image_records


def load_embeddings(embeddings_path):
    """Load precomputed embeddings from .npz file."""
    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = data["embeddings"]
    metadata = json.loads(str(data["metadata"]))
    print(f"Loaded embeddings: {embeddings.shape} from {embeddings_path}")
    return embeddings, metadata


# ═══════════════════════════════════════════════════════════════
#  STEP 2: METRIC COMPUTATION
# ═══════════════════════════════════════════════════════════════
#
#  These are high-dimensional adaptations of the metrics from
#  the paper's synthetic experiments (run_experiments.py).
#

def effective_support_radius(samples, alpha=ALPHA):
    """R_alpha: radius around centroid containing alpha-fraction of mass.

    Identical to the 2D version — works in any dimension.
    Corresponds to Definition 2.2 in the paper.
    """
    mu = samples.mean(axis=0)
    dists = np.linalg.norm(samples - mu, axis=1)
    return float(np.percentile(dists, alpha * 100))


def diversity_index(samples):
    """D(P) = exp(H / d): normalised Gaussian-fit entropy.

    Adapted for high-dimensional stability using slogdet.
    Normalised by dimension d so that values are comparable
    across different embedding spaces.
    Corresponds to Definition 2.1 in the paper.
    """
    n, d = samples.shape
    if n < 2:
        return 0.0
    cov = np.cov(samples.T)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        # Singular covariance (e.g., too few samples for the dimension)
        # Fall back to using only positive eigenvalues
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 1e-12]
        logdet = np.sum(np.log(eigvals)) if len(eigvals) > 0 else -np.inf
    entropy = 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * logdet
    return float(np.exp(entropy / d))  # per-dimension normalisation


def effective_dimension(samples):
    """d_eff: participation ratio of eigenvalues.

    d_eff = (sum lambda_i)^2 / sum(lambda_i^2)

    This is the standard participation ratio from statistical physics,
    equivalent to exp(H_2) where H_2 is Renyi-2 entropy of the
    eigenvalue distribution. It replaces the histogram-based estimator
    from the 2D experiments, which doesn't scale to 768 dimensions.
    Corresponds to Definition 2.1 in the paper.
    """
    n, d = samples.shape
    if n < 2:
        return 0.0
    cov = np.cov(samples.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]  # positive only
    if len(eigenvalues) == 0:
        return 0.0
    total = eigenvalues.sum()
    return float(total ** 2 / (eigenvalues ** 2).sum())


def cosine_distance(a, b):
    """Cosine distance between two vectors (1 - cosine similarity)."""
    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(1.0 - sim)


def compute_metrics_for_group(embeddings_subset):
    """Compute all three metrics for a group of embeddings.

    Args:
        embeddings_subset: np.array of shape (n, d)

    Returns:
        dict with keys: R_alpha, D, d_eff, n, centroid
    """
    n = embeddings_subset.shape[0]
    if n < 3:
        return {
            "R_alpha": 0.0, "D": 0.0, "d_eff": 0.0,
            "n": n, "centroid": embeddings_subset.mean(axis=0).tolist(),
        }
    return {
        "R_alpha": effective_support_radius(embeddings_subset),
        "D": diversity_index(embeddings_subset),
        "d_eff": effective_dimension(embeddings_subset),
        "n": n,
        "centroid": embeddings_subset.mean(axis=0).tolist(),
    }


# ═══════════════════════════════════════════════════════════════
#  STEP 3: ANALYSES
# ═══════════════════════════════════════════════════════════════

def group_embeddings(embeddings, metadata, group_by):
    """Group embeddings by a metadata field (e.g., 'prompt_id', 'platform')."""
    groups = defaultdict(list)
    for i, rec in enumerate(metadata):
        key = rec[group_by]
        groups[key].append(i)
    return {k: embeddings[np.array(v)] for k, v in groups.items()}


def group_embeddings_multi(embeddings, metadata, keys):
    """Group embeddings by multiple metadata fields (tuple key)."""
    groups = defaultdict(list)
    for i, rec in enumerate(metadata):
        key = tuple(rec[k] for k in keys)
        groups[key].append(i)
    return {k: embeddings[np.array(v)] for k, v in groups.items()}


def analysis_1_within_prompt_diversity(embeddings, metadata, figures_dir):
    """Analysis 1: Within-prompt diversity by category and platform.

    Tests MAP Prediction: AI-generated images for the same prompt should
    show low within-prompt diversity (the model converges toward the
    conditional mean each time).
    """
    print("\n" + "=" * 60)
    print("  Analysis 1: Within-Prompt Diversity")
    print("=" * 60)

    # Group by (platform, prompt_id)
    by_platform_prompt = group_embeddings_multi(
        embeddings, metadata, ["platform", "prompt_id"]
    )

    results = []
    for (platform, prompt_id), embs in sorted(by_platform_prompt.items()):
        # Look up category and group
        rec = next((m for m in metadata if m["prompt_id"] == prompt_id
                     and m["platform"] == platform), {})
        metrics = compute_metrics_for_group(embs)
        metrics.update({
            "platform": platform,
            "prompt_id": prompt_id,
            "category": rec.get("category", "unknown"),
            "group": rec.get("group", "unknown"),
        })
        results.append(metrics)
        del metrics["centroid"]  # Don't print the 768-dim vector

    # Summary table
    print(f"\n{'Platform':<10} {'Category':<20} {'Prompt':<20} "
          f"{'R_α':>8} {'D':>10} {'d_eff':>8} {'n':>4}")
    print("-" * 85)
    for r in results:
        print(f"{r['platform']:<10} {r['category']:<20} {r['prompt_id']:<20} "
              f"{r['R_alpha']:>8.4f} {r['D']:>10.4f} {r['d_eff']:>8.2f} {r['n']:>4}")

    # ── Figure: bar chart of mean R_alpha by category x platform ──
    platforms = sorted(set(r["platform"] for r in results))
    categories = ["cultural_identity", "open_ended", "artistic_style"]
    cat_labels = ["Cultural Identity", "Open-Ended", "Artistic Style"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metric_names = [("R_alpha", r"$R_\alpha$ (Support Radius)"),
                    ("D", r"$D$ (Diversity Index)"),
                    ("d_eff", r"$d_{\mathrm{eff}}$ (Effective Dim.)")]

    for ax, (mkey, mlabel) in zip(axes, metric_names):
        x = np.arange(len(categories))
        width = 0.35
        for p_idx, platform in enumerate(platforms):
            means = []
            stds = []
            for cat in categories:
                vals = [r[mkey] for r in results
                        if r["platform"] == platform and r["category"] == cat]
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if vals else 0)
            ax.bar(x + p_idx * width, means, width, yerr=stds,
                   color=PLATFORM_COLORS.get(platform, "#999"),
                   label=platform.capitalize(), alpha=0.85, capsize=3)
        ax.set_xlabel("Prompt Category")
        ax.set_ylabel(mlabel)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(cat_labels, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Analysis 1: Within-Prompt Diversity by Category and Model",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "analysis1_within_prompt_diversity.png")
    plt.close()
    print(f"\n  Figure saved: {figures_dir}/analysis1_within_prompt_diversity.png")

    return results


def analysis_2_cultural_convergence(embeddings, metadata, figures_dir):
    """Analysis 2: Cross-cultural convergence and minority suppression.

    Tests MAP Predictions:
      - Minority nationality centroids should be pulled toward the global mean
      - Minority nationalities should show less within-prompt diversity
    """
    print("\n" + "=" * 60)
    print("  Analysis 2: Cross-Cultural Convergence")
    print("=" * 60)

    # Filter to cultural_identity prompts only
    ci_mask = np.array([m["category"] == "cultural_identity" for m in metadata])
    if not ci_mask.any():
        print("  No cultural_identity images found — skipping Analysis 2")
        return {}

    ci_embeddings = embeddings[ci_mask]
    ci_metadata = [m for m, keep in zip(metadata, ci_mask) if keep]

    # Group by (platform, prompt_id) — each nationality per model
    by_platform_prompt = group_embeddings_multi(
        ci_embeddings, ci_metadata, ["platform", "prompt_id"]
    )

    # Compute per-nationality metrics
    nationality_metrics = {}
    for (platform, prompt_id), embs in sorted(by_platform_prompt.items()):
        rec = next((m for m in ci_metadata
                    if m["prompt_id"] == prompt_id and m["platform"] == platform), {})
        nationality = rec.get("nationality", prompt_id)
        group = rec.get("group", "unknown")
        metrics = compute_metrics_for_group(embs)
        key = (platform, nationality)
        nationality_metrics[key] = {
            **metrics,
            "platform": platform,
            "nationality": nationality,
            "group": group,
        }

    # Compute global centroid (across all cultural identity images per platform)
    for platform in sorted(set(m["platform"] for m in ci_metadata)):
        plat_mask = np.array([m["platform"] == platform for m in ci_metadata])
        global_centroid = ci_embeddings[plat_mask].mean(axis=0)

        print(f"\n  {platform.upper()} — distance from each nationality centroid to global mean:")
        print(f"  {'Nationality':<15} {'Group':<10} {'R_α':>8} {'D':>10} "
              f"{'d_eff':>8} {'Dist→Global':>12}")
        print("  " + "-" * 68)

        for (plat, nat), metrics in sorted(nationality_metrics.items()):
            if plat != platform:
                continue
            centroid = np.array(metrics["centroid"])
            dist_to_global = float(np.linalg.norm(centroid - global_centroid))
            metrics["dist_to_global"] = dist_to_global
            print(f"  {nat:<15} {metrics['group']:<10} {metrics['R_alpha']:>8.4f} "
                  f"{metrics['D']:>10.4f} {metrics['d_eff']:>8.2f} "
                  f"{dist_to_global:>12.4f}")

    # ── Figure A: Bar chart of within-prompt diversity, majority vs minority ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax_idx, platform in enumerate(sorted(set(m["platform"] for m in ci_metadata))):
        ax = axes[ax_idx] if len(axes) > 1 else axes
        nats_majority = sorted([
            (nat, m) for (plat, nat), m in nationality_metrics.items()
            if plat == platform and m["group"] == "majority"
        ], key=lambda x: x[0])
        nats_minority = sorted([
            (nat, m) for (plat, nat), m in nationality_metrics.items()
            if plat == platform and m["group"] == "minority"
        ], key=lambda x: x[0])

        all_nats = nats_majority + nats_minority
        labels = [n for n, _ in all_nats]
        r_alphas = [m["R_alpha"] for _, m in all_nats]
        colors = [GROUP_COLORS["majority"]] * len(nats_majority) + \
                 [GROUP_COLORS["minority"]] * len(nats_minority)

        bars = ax.barh(range(len(labels)), r_alphas, color=colors, alpha=0.85)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(r"$R_\alpha$ (Within-Prompt Support Radius)")
        ax.set_title(f"{platform.capitalize()}: Within-Prompt Diversity by Nationality")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2, axis="x")

        # Add majority/minority legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=GROUP_COLORS["majority"], alpha=0.85, label="Majority"),
            Patch(facecolor=GROUP_COLORS["minority"], alpha=0.85, label="Minority"),
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

        # Add vertical line for mean
        ax.axvline(np.mean(r_alphas), color="gray", ls="--", alpha=0.5,
                   label="Mean")

    fig.suptitle("Analysis 2: Cultural Identity — Within-Prompt Diversity\n"
                 "(MAP predicts minority nationalities have lower diversity)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(figures_dir / "analysis2_cultural_convergence.png")
    plt.close()
    print(f"\n  Figure saved: {figures_dir}/analysis2_cultural_convergence.png")

    # ── Figure B: Cross-cultural distance heatmap ──
    # Compute pairwise cosine distances between nationality centroids
    for platform in sorted(set(m["platform"] for m in ci_metadata)):
        nats_in_platform = sorted([
            (nat, np.array(m["centroid"]))
            for (plat, nat), m in nationality_metrics.items()
            if plat == platform
        ], key=lambda x: x[0])

        if len(nats_in_platform) < 2:
            continue

        nat_names = [n for n, _ in nats_in_platform]
        centroids_arr = np.array([c for _, c in nats_in_platform])
        n_nats = len(nat_names)

        dist_matrix = np.zeros((n_nats, n_nats))
        for i in range(n_nats):
            for j in range(n_nats):
                dist_matrix[i, j] = cosine_distance(centroids_arr[i], centroids_arr[j])

        fig2, ax2 = plt.subplots(figsize=(10, 8))
        im = ax2.imshow(dist_matrix, cmap="YlOrRd", aspect="equal")
        ax2.set_xticks(range(n_nats))
        ax2.set_xticklabels(nat_names, rotation=45, ha="right", fontsize=9)
        ax2.set_yticks(range(n_nats))
        ax2.set_yticklabels(nat_names, fontsize=9)
        fig2.colorbar(im, ax=ax2, label="Cosine Distance", shrink=0.8)
        ax2.set_title(f"{platform.capitalize()}: Pairwise Cosine Distance "
                      f"Between Nationality Centroids\n"
                      f"(MAP predicts compressed distances = cultural convergence)",
                      fontsize=12)
        fig2.tight_layout()
        fig2.savefig(figures_dir / f"analysis2_cultural_heatmap_{platform}.png")
        plt.close()
        print(f"  Heatmap saved: {figures_dir}/analysis2_cultural_heatmap_{platform}.png")

    return nationality_metrics


def analysis_3_cross_model_agreement(embeddings, metadata, figures_dir):
    """Analysis 3: Cross-model agreement on the same prompts.

    Tests Compound Convergence: different models trained on overlapping
    data should produce similar outputs for the same prompt, because
    they're computing conditional expectations over approximately the
    same training distribution.
    """
    print("\n" + "=" * 60)
    print("  Analysis 3: Cross-Model Agreement")
    print("=" * 60)

    platforms = sorted(set(m["platform"] for m in metadata))
    if len(platforms) < 2:
        print("  Only one platform found — skipping cross-model comparison")
        return {}

    # Compute per-(platform, prompt_id) centroids
    by_pp = group_embeddings_multi(embeddings, metadata, ["platform", "prompt_id"])
    centroids = {}
    for (platform, prompt_id), embs in by_pp.items():
        centroids[(platform, prompt_id)] = embs.mean(axis=0)

    # For each prompt, compute cosine similarity between platform centroids
    prompt_ids = sorted(set(m["prompt_id"] for m in metadata))
    cross_model_results = []

    print(f"\n  {'Prompt':<25} {'Category':<20} {'Cosine Sim':>12}")
    print("  " + "-" * 60)

    for prompt_id in prompt_ids:
        p0 = platforms[0]
        p1 = platforms[1]
        if (p0, prompt_id) not in centroids or (p1, prompt_id) not in centroids:
            continue

        c0 = centroids[(p0, prompt_id)]
        c1 = centroids[(p1, prompt_id)]
        sim = float(np.dot(c0, c1) / (np.linalg.norm(c0) * np.linalg.norm(c1) + 1e-12))

        rec = next((m for m in metadata if m["prompt_id"] == prompt_id), {})
        category = rec.get("category", "unknown")

        cross_model_results.append({
            "prompt_id": prompt_id,
            "category": category,
            "cosine_similarity": sim,
        })
        print(f"  {prompt_id:<25} {category:<20} {sim:>12.4f}")

    if not cross_model_results:
        print("  No overlapping prompts between models")
        return {}

    mean_sim = np.mean([r["cosine_similarity"] for r in cross_model_results])
    print(f"\n  Mean cross-model cosine similarity: {mean_sim:.4f}")
    print(f"  (1.0 = identical outputs, 0.0 = orthogonal)")

    # ── Figure: scatter plot of per-prompt cross-model similarity ──
    fig, ax = plt.subplots(figsize=(10, 6))

    for cat, color in CATEGORY_COLORS.items():
        cat_results = [r for r in cross_model_results if r["category"] == cat]
        if not cat_results:
            continue
        x = range(len(cat_results))
        y = [r["cosine_similarity"] for r in cat_results]
        labels = [r["prompt_id"].replace("ci_", "").replace("oe_", "").replace("as_", "")
                  for r in cat_results]
        ax.bar([i + list(CATEGORY_COLORS.keys()).index(cat) * 0.25
                for i in range(len(cat_results))],
               y, 0.25, color=color, alpha=0.85,
               label=cat.replace("_", " ").title())

    ax.axhline(mean_sim, color="black", ls="--", alpha=0.5,
               label=f"Mean = {mean_sim:.3f}")
    ax.set_ylabel("Cosine Similarity (centroid to centroid)")
    ax.set_title(f"Analysis 3: Cross-Model Agreement ({platforms[0].capitalize()} "
                 f"vs {platforms[1].capitalize()})\n"
                 f"(MAP predicts high agreement: both models converge to similar means)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(figures_dir / "analysis3_cross_model_agreement.png")
    plt.close()
    print(f"\n  Figure saved: {figures_dir}/analysis3_cross_model_agreement.png")

    return cross_model_results


# ═══════════════════════════════════════════════════════════════
#  STEP 3b: STATISTICAL SIGNIFICANCE TESTS
# ═══════════════════════════════════════════════════════════════

def bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of observations
        statistic: function to compute (default: np.mean)
        n_bootstrap: number of bootstrap resamples
        ci: confidence level (default: 0.95)
        seed: random seed for reproducibility

    Returns:
        dict with keys: estimate, ci_lower, ci_upper, se
    """
    rng = np.random.RandomState(seed)
    data = np.asarray(data)
    n = len(data)
    boot_stats = np.array([
        statistic(rng.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - ci
    return {
        "estimate": float(statistic(data)),
        "ci_lower": float(np.percentile(boot_stats, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        "se": float(np.std(boot_stats)),
    }


def permutation_test(group_a, group_b, statistic=np.mean, n_permutations=10000,
                     alternative="greater", seed=42):
    """Two-sample permutation test for the difference in a statistic.

    Tests H0: statistic(group_a) = statistic(group_b)
    vs H1: statistic(group_a) > statistic(group_b) (if alternative='greater')
         or statistic(group_a) != statistic(group_b) (if alternative='two-sided')

    Args:
        group_a, group_b: 1D arrays of observations
        statistic: function to compute on each group
        n_permutations: number of random permutations
        alternative: 'greater', 'less', or 'two-sided'
        seed: random seed for reproducibility

    Returns:
        dict with keys: observed_diff, p_value, n_permutations
    """
    rng = np.random.RandomState(seed)
    a = np.asarray(group_a)
    b = np.asarray(group_b)
    observed_diff = float(statistic(a) - statistic(b))

    combined = np.concatenate([a, b])
    n_a = len(a)

    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(combined)
        perm_diff = statistic(perm[:n_a]) - statistic(perm[n_a:])
        if alternative == "greater":
            if perm_diff >= observed_diff:
                count += 1
        elif alternative == "less":
            if perm_diff <= observed_diff:
                count += 1
        else:  # two-sided
            if abs(perm_diff) >= abs(observed_diff):
                count += 1

    p_value = (count + 1) / (n_permutations + 1)  # +1 for continuity correction
    return {
        "observed_diff": observed_diff,
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "alternative": alternative,
    }


def significance_tests(a2_results):
    """Run bootstrap CIs and permutation tests on majority vs minority R_alpha.

    Returns dict with all test results for inclusion in the summary JSON.
    """
    if not a2_results:
        return {}

    majority_r = [m["R_alpha"] for m in a2_results.values()
                  if isinstance(m, dict) and m.get("group") == "majority"]
    minority_r = [m["R_alpha"] for m in a2_results.values()
                  if isinstance(m, dict) and m.get("group") == "minority"]

    if len(majority_r) < 2 or len(minority_r) < 2:
        print("  Insufficient data for significance tests")
        return {}

    print("\n" + "=" * 60)
    print("  Statistical Significance Tests")
    print("=" * 60)

    results = {}

    # Bootstrap CIs for each group mean
    maj_boot = bootstrap_ci(majority_r, n_bootstrap=10000)
    min_boot = bootstrap_ci(minority_r, n_bootstrap=10000)
    results["majority_R_alpha_bootstrap"] = maj_boot
    results["minority_R_alpha_bootstrap"] = min_boot

    print(f"\n  Majority R_α: {maj_boot['estimate']:.4f} "
          f"[{maj_boot['ci_lower']:.4f}, {maj_boot['ci_upper']:.4f}] (95% CI)")
    print(f"  Minority R_α: {min_boot['estimate']:.4f} "
          f"[{min_boot['ci_lower']:.4f}, {min_boot['ci_upper']:.4f}] (95% CI)")

    # Bootstrap CI for the difference
    diff_vals = []
    rng = np.random.RandomState(42)
    maj_arr = np.asarray(majority_r)
    min_arr = np.asarray(minority_r)
    for _ in range(10000):
        boot_maj = rng.choice(maj_arr, size=len(maj_arr), replace=True)
        boot_min = rng.choice(min_arr, size=len(min_arr), replace=True)
        diff_vals.append(np.mean(boot_maj) - np.mean(boot_min))
    diff_vals = np.array(diff_vals)

    diff_boot = {
        "estimate": float(np.mean(maj_arr) - np.mean(min_arr)),
        "ci_lower": float(np.percentile(diff_vals, 2.5)),
        "ci_upper": float(np.percentile(diff_vals, 97.5)),
        "se": float(np.std(diff_vals)),
    }
    results["difference_bootstrap"] = diff_boot
    print(f"\n  Difference (Majority - Minority): {diff_boot['estimate']:.4f} "
          f"[{diff_boot['ci_lower']:.4f}, {diff_boot['ci_upper']:.4f}] (95% CI)")

    # Permutation test: H0: majority R_α = minority R_α
    #                   H1: majority R_α > minority R_α (one-sided)
    perm = permutation_test(majority_r, minority_r, statistic=np.mean,
                            n_permutations=10000, alternative="greater")
    results["permutation_test"] = perm

    print(f"\n  Permutation test (H1: majority > minority):")
    print(f"    Observed difference: {perm['observed_diff']:.4f}")
    print(f"    p-value: {perm['p_value']:.4f}")
    print(f"    {'SIGNIFICANT' if perm['p_value'] < 0.05 else 'NOT SIGNIFICANT'} at α=0.05")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(majority_r) + np.var(minority_r)) / 2)
    if pooled_std > 0:
        cohens_d = (np.mean(majority_r) - np.mean(minority_r)) / pooled_std
    else:
        cohens_d = 0.0
    results["cohens_d"] = float(cohens_d)
    print(f"\n  Cohen's d (effect size): {cohens_d:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════
#  STEP 4: SUMMARY & PREDICTIONS
# ═══════════════════════════════════════════════════════════════

def compile_summary(a1_results, a2_results, a3_results, figures_dir):
    """Compile all results and check MAP predictions."""

    print("\n" + "=" * 60)
    print("  MAP/SLOP PREDICTION SUMMARY")
    print("=" * 60)

    summary = {
        "experiment": "Empirical validation of MAP/SLOP on commercial models",
        "clip_model": CLIP_MODEL_NAME,
        "analyses": {},
        "predictions": {},
    }

    # Analysis 1 summary
    if a1_results:
        by_category = defaultdict(list)
        for r in a1_results:
            by_category[r["category"]].append(r)

        summary["analyses"]["within_prompt_diversity"] = {
            cat: {
                "mean_R_alpha": float(np.mean([r["R_alpha"] for r in recs])),
                "mean_D": float(np.mean([r["D"] for r in recs])),
                "mean_d_eff": float(np.mean([r["d_eff"] for r in recs])),
                "n_prompts": len(set(r["prompt_id"] for r in recs)),
            }
            for cat, recs in by_category.items()
        }

    # Analysis 2 summary
    if a2_results:
        majority_r = [m["R_alpha"] for m in a2_results.values()
                      if isinstance(m, dict) and m.get("group") == "majority"]
        minority_r = [m["R_alpha"] for m in a2_results.values()
                      if isinstance(m, dict) and m.get("group") == "minority"]

        if majority_r and minority_r:
            mean_maj = float(np.mean(majority_r))
            mean_min = float(np.mean(minority_r))
            summary["analyses"]["cultural_convergence"] = {
                "majority_mean_R_alpha": mean_maj,
                "minority_mean_R_alpha": mean_min,
                "ratio_minority_to_majority": float(mean_min / mean_maj) if mean_maj > 0 else 0,
            }

            # MAP predicts minority < majority
            pred_minority_less_diverse = mean_min < mean_maj
            summary["predictions"]["minority_less_diverse"] = bool(pred_minority_less_diverse)
            status = "CONFIRMED" if pred_minority_less_diverse else "NOT CONFIRMED"
            print(f"  P1 (Minority less diverse): {status}")
            print(f"      Majority mean R_α = {mean_maj:.4f}")
            print(f"      Minority mean R_α = {mean_min:.4f}")

            # Statistical significance tests
            sig_results = significance_tests(a2_results)
            if sig_results:
                summary["analyses"]["significance_tests"] = sig_results

    # Analysis 3 summary
    if a3_results:
        sims = [r["cosine_similarity"] for r in a3_results]
        mean_sim = float(np.mean(sims))
        summary["analyses"]["cross_model_agreement"] = {
            "mean_cosine_similarity": mean_sim,
            "min_cosine_similarity": float(np.min(sims)),
            "max_cosine_similarity": float(np.max(sims)),
            "n_prompts": len(sims),
        }

        # MAP predicts high cross-model agreement (> 0.5 as a reasonable threshold)
        pred_high_agreement = mean_sim > 0.5
        summary["predictions"]["cross_model_convergence"] = bool(pred_high_agreement)
        status = "CONFIRMED" if pred_high_agreement else "NOT CONFIRMED"
        print(f"  P2 (Cross-model convergence): {status}")
        print(f"      Mean cosine similarity = {mean_sim:.4f}")

    # Save
    summary_path = figures_dir / "empirical_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved: {summary_path}")

    return summary


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Empirical analysis pipeline for MAP/SLOP validation"
    )
    parser.add_argument(
        "--image-dir", type=str, default=str(Path(__file__).parent / "images"),
        help="Directory containing generated images (default: ./images)",
    )
    parser.add_argument(
        "--prompts", type=str, default=str(Path(__file__).parent / "prompts.json"),
        help="Path to prompts.json (default: ./prompts.json)",
    )
    parser.add_argument(
        "--embeddings", type=str, default=None,
        help="Path to precomputed embeddings.npz (skips CLIP embedding step)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(Path(__file__).parent),
        help="Base output directory (default: same as this script)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="CLIP embedding batch size (default: 32, reduce if OOM)",
    )
    parser.add_argument(
        "--embeddings-only", action="store_true",
        help="Only compute and save embeddings (skip analysis)",
    )
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="Only run analysis (requires --embeddings)",
    )

    args = parser.parse_args()

    # Load prompts
    with open(args.prompts) as f:
        prompts_data = json.load(f)
    print(f"Loaded {len(prompts_data['prompts'])} prompts")

    # Set up paths
    output_dir = Path(args.output_dir)
    embeddings_dir = output_dir / "embeddings"
    figures_dir = output_dir / "figures"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = embeddings_dir / "embeddings.npz"

    # Step 1: Embed (or load)
    if args.analysis_only:
        if args.embeddings:
            embeddings, metadata = load_embeddings(args.embeddings)
        elif embeddings_path.exists():
            embeddings, metadata = load_embeddings(str(embeddings_path))
        else:
            print("ERROR: --analysis-only requires --embeddings or existing embeddings.npz")
            sys.exit(1)
    else:
        embeddings, metadata = embed_images(
            args.image_dir, prompts_data, str(embeddings_path),
            batch_size=args.batch_size,
        )

    if args.embeddings_only:
        print("\nEmbeddings saved. Exiting (--embeddings-only mode).")
        return

    # Step 2-3: Analyses
    print(f"\nRunning analyses on {embeddings.shape[0]} images...")

    a1 = analysis_1_within_prompt_diversity(embeddings, metadata, figures_dir)
    a2 = analysis_2_cultural_convergence(embeddings, metadata, figures_dir)
    a3 = analysis_3_cross_model_agreement(embeddings, metadata, figures_dir)

    # Step 4: Summary
    summary = compile_summary(a1, a2, a3, figures_dir)

    print("\n" + "=" * 60)
    print("  DONE! All analyses complete.")
    print(f"  Figures: {figures_dir}/")
    print(f"  Summary: {figures_dir}/empirical_summary.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
