#!/usr/bin/env python3
"""
DINOv2 Replication — Addresses CLIP Circularity Concern
========================================================
Re-embeds all existing images with DINOv2 ViT-L/14 (self-supervised,
no text-image pairing) and re-runs all diversity analyses.

If the 15% minority suppression finding holds under DINOv2,
the CLIP circularity concern is substantially mitigated.

Requirements:
  pip install torch torchvision transformers Pillow numpy scipy matplotlib tqdm

Usage:
  # Full pipeline: embed with DINOv2 + run analysis
  python revision_dinov2_replication.py

  # Just embedding (skip analysis)
  python revision_dinov2_replication.py --embed-only

  # Just analysis (if DINOv2 embeddings already computed)
  python revision_dinov2_replication.py --analysis-only
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
DINOV2_MODEL = "facebook/dinov2-large"  # 1024-dim embeddings
BATCH_SIZE = 16
ALPHA = 0.90

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

GROUP_COLORS = {"majority": "#2166ac", "minority": "#b2182b"}


# ─── Metrics ────────────────────────────────────────────────────

def effective_support_radius(samples, alpha=ALPHA):
    mu = samples.mean(axis=0)
    dists = np.linalg.norm(samples - mu, axis=1)
    return float(np.percentile(dists, alpha * 100))

def effective_dimension_fast(samples):
    n, d = samples.shape
    if n < 2:
        return 0.0
    centered = samples - samples.mean(axis=0)
    G = centered @ centered.T / (n - 1)
    eigenvalues = np.linalg.eigvalsh(G)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if len(eigenvalues) == 0:
        return 0.0
    total = eigenvalues.sum()
    return float(total ** 2 / (eigenvalues ** 2).sum())

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ─── DINOv2 Embedding ──────────────────────────────────────────

def embed_with_dinov2(image_paths, output_path):
    """Embed images with DINOv2 ViT-L/14.

    DINOv2 is self-supervised (trained on images only, no text pairing),
    so it avoids the CLIP circularity where the embedding space shares
    biases with the text-image training distribution.
    """
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoImageProcessor
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading DINOv2 ({DINOV2_MODEL}) on {device}...")
    model = AutoModel.from_pretrained(DINOV2_MODEL).to(device)
    processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL)
    model.eval()

    all_embeddings = []
    valid_indices = []

    print(f"Embedding {len(image_paths)} images...")
    for batch_start in tqdm(range(0, len(image_paths), BATCH_SIZE),
                            desc="DINOv2 embedding", unit="batch"):
        batch_paths = image_paths[batch_start:batch_start + BATCH_SIZE]
        images = []
        batch_indices = []

        for i, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                batch_indices.append(batch_start + i)
            except Exception as e:
                print(f"  Warning: could not open {p}: {e}")

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token embedding
            features = outputs.last_hidden_state[:, 0]
            # L2 normalize
            features = features / features.norm(dim=-1, keepdim=True)

        all_embeddings.append(features.cpu().numpy())
        valid_indices.extend(batch_indices)

    embeddings = np.vstack(all_embeddings)
    print(f"  DINOv2 embedding shape: {embeddings.shape}")

    np.savez_compressed(output_path, embeddings=embeddings, valid_indices=np.array(valid_indices))
    return embeddings, valid_indices


def run_full_pipeline():
    """Embed with DINOv2, then run comparative analysis against CLIP results."""

    base_dir = Path(__file__).parent
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load CLIP metadata to know which images to embed
    clip_data = np.load(base_dir / "embeddings" / "embeddings.npz", allow_pickle=True)
    clip_embeddings = clip_data["embeddings"]
    metadata = json.loads(str(clip_data["metadata"]))

    prompts_path = base_dir / "prompts.json"
    with open(prompts_path) as f:
        prompt_lookup = {p["id"]: p for p in json.load(f)["prompts"]}

    # Check if DINOv2 embeddings already exist
    dinov2_path = base_dir / "embeddings" / "embeddings_dinov2.npz"
    if dinov2_path.exists():
        print("Loading existing DINOv2 embeddings...")
        d2_data = np.load(dinov2_path, allow_pickle=True)
        dinov2_embeddings = d2_data["embeddings"]
        valid_indices = d2_data["valid_indices"]
    else:
        image_paths = [rec["path"] for rec in metadata]
        dinov2_embeddings, valid_indices = embed_with_dinov2(image_paths, str(dinov2_path))

    # Align metadata with valid indices
    aligned_metadata = [metadata[i] for i in valid_indices]
    aligned_clip = clip_embeddings[valid_indices]

    print(f"\nComparing {len(aligned_metadata)} images across CLIP and DINOv2")
    print(f"  CLIP dim: {aligned_clip.shape[1]}, DINOv2 dim: {dinov2_embeddings.shape[1]}")

    # ── Run comparative analysis ──
    run_comparison(aligned_clip, dinov2_embeddings, aligned_metadata, prompt_lookup, figures_dir)


def run_comparison(clip_embs, dinov2_embs, metadata, prompt_lookup, figures_dir):
    """Compare diversity metrics between CLIP and DINOv2 embeddings."""

    embedding_sets = {
        "CLIP": clip_embs,
        "DINOv2": dinov2_embs,
    }

    # Group by (platform, prompt_id)
    groups = defaultdict(list)
    for i, rec in enumerate(metadata):
        groups[(rec["platform"], rec["prompt_id"])].append(i)

    print(f"\n{'='*80}")
    print("  CLIP vs DINOv2 — Diversity Metrics Comparison")
    print(f"{'='*80}")

    all_results = {name: [] for name in embedding_sets}

    print(f"\n{'Prompt':<25} {'Group':<10} {'n':>4} "
          f"{'CLIP R_α':>10} {'DINO R_α':>10} {'CLIP d_eff':>10} {'DINO d_eff':>10}")
    print("-" * 85)

    for (platform, prompt_id), indices in sorted(groups.items()):
        pinfo = prompt_lookup.get(prompt_id, {})
        group = pinfo.get("group", "unknown")
        n = len(indices)
        idx = np.array(indices)

        row = {"platform": platform, "prompt_id": prompt_id, "group": group, "n": n}

        for name, embs in embedding_sets.items():
            subset = embs[idx]
            r = effective_support_radius(subset)
            d = effective_dimension_fast(subset)
            row[f"{name}_R_alpha"] = r
            row[f"{name}_d_eff"] = d
            all_results[name].append({
                "platform": platform, "prompt_id": prompt_id,
                "group": group, "n": n, "R_alpha": r, "d_eff": d,
            })

        label = f"{platform[:4]}/{prompt_id}"
        print(f"{label:<25} {group:<10} {n:>4} "
              f"{row['CLIP_R_alpha']:>10.4f} {row['DINOv2_R_alpha']:>10.4f} "
              f"{row['CLIP_d_eff']:>10.2f} {row['DINOv2_d_eff']:>10.2f}")

    # ── Key comparison: majority/minority gap in both spaces ──
    print(f"\n{'='*80}")
    print("  MAJORITY vs MINORITY GAP — CLIP vs DINOv2")
    print(f"{'='*80}")

    for name in embedding_sets:
        ci = [r for r in all_results[name] if r["prompt_id"].startswith("ci_")]
        maj_r = [r["R_alpha"] for r in ci if r["group"] == "majority"]
        min_r = [r["R_alpha"] for r in ci if r["group"] == "minority"]

        if not maj_r or not min_r:
            continue

        maj_mean = np.mean(maj_r)
        min_mean = np.mean(min_r)
        gap = maj_mean - min_mean
        gap_pct = (gap / maj_mean) * 100 if maj_mean else 0

        print(f"\n  {name}:")
        print(f"    Majority R_α: {maj_mean:.4f} (n={len(maj_r)})")
        print(f"    Minority R_α: {min_mean:.4f} (n={len(min_r)})")
        print(f"    Gap: {gap:.4f} ({gap_pct:.1f}%)")
        print(f"    → {'FINDING REPLICATES' if gap > 0 else 'FINDING DOES NOT REPLICATE'}")

    # ── Cross-model agreement in both spaces ──
    print(f"\n{'='*80}")
    print("  CROSS-MODEL AGREEMENT — CLIP vs DINOv2")
    print(f"{'='*80}")

    for name, embs in embedding_sets.items():
        # Compute centroids per (platform, prompt_id)
        centroids = {}
        for (platform, prompt_id), indices in groups.items():
            centroids[(platform, prompt_id)] = embs[np.array(indices)].mean(axis=0)

        # Find prompts present in both platforms
        prompt_ids = set()
        platforms = set()
        for (plat, pid) in centroids:
            prompt_ids.add(pid)
            platforms.add(plat)

        if len(platforms) < 2:
            continue

        platforms = sorted(platforms)
        sims = []
        for pid in sorted(prompt_ids):
            if (platforms[0], pid) in centroids and (platforms[1], pid) in centroids:
                s = cosine_sim(centroids[(platforms[0], pid)], centroids[(platforms[1], pid)])
                sims.append(s)

        if sims:
            print(f"  {name}: mean cross-model similarity = {np.mean(sims):.4f} "
                  f"(n={len(sims)} prompts)")

    # Save
    output = {"clip_results": all_results["CLIP"], "dinov2_results": all_results["DINOv2"]}
    with open(figures_dir / "dinov2_comparison_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {figures_dir}/dinov2_comparison_results.json")


def run_analysis_only():
    """Run analysis on pre-computed DINOv2 embeddings."""
    base_dir = Path(__file__).parent
    dinov2_path = base_dir / "embeddings" / "embeddings_dinov2.npz"
    if not dinov2_path.exists():
        print("ERROR: No DINOv2 embeddings found. Run without --analysis-only first.")
        sys.exit(1)
    run_full_pipeline()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINOv2 replication analysis")
    parser.add_argument("--embed-only", action="store_true",
                        help="Only compute DINOv2 embeddings, skip analysis")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Only run analysis on existing DINOv2 embeddings")
    args = parser.parse_args()

    if args.analysis_only:
        run_analysis_only()
    else:
        run_full_pipeline()
