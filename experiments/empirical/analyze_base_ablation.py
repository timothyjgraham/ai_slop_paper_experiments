#!/usr/bin/env python3
"""
Quick analysis of SDXL base ablation results.
Answers two key questions:
  1. Does R_alpha decrease monotonically with CFG scale? (tests Proposition 3.1)
  2. Does the majority/minority gap appear WITHOUT RLHF? (tests MSE as independent cause)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ALPHA = 0.90
ABLATION_DIR = Path(__file__).parent / "ablation" / "sdxl-base"

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

def embed_images_clip(image_paths):
    """Embed a list of images with CLIP. Returns (n, 768) array."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image

    device = "cpu"  # Fine for embedding 440 images
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()

    all_embs = []
    batch_size = 32
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output if hasattr(features, 'pooler_output') else features[0]
            features = features / features.norm(dim=-1, keepdim=True)
        all_embs.append(features.cpu().numpy())
        print(f"  Embedded {min(i+batch_size, len(image_paths))}/{len(image_paths)}", end="\r")

    print()
    return np.vstack(all_embs)

def main():
    # Load prompts for group info
    prompts_path = Path(__file__).parent / "prompts.json"
    with open(prompts_path) as f:
        prompt_lookup = {p["id"]: p for p in json.load(f)["prompts"]}

    # Discover all images
    cfg_scales = sorted([float(d.name.replace("cfg_", ""))
                         for d in ABLATION_DIR.iterdir() if d.is_dir()])
    print(f"Found CFG scales: {cfg_scales}")

    # Collect all image paths with metadata
    records = []
    for cfg in cfg_scales:
        cfg_dir = ABLATION_DIR / f"cfg_{cfg}"
        for cat_dir in sorted(cfg_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            for img_path in sorted(cat_dir.glob("*.png")):
                prompt_id = "_".join(img_path.stem.split("_")[:-1])  # strip _NNN suffix
                pinfo = prompt_lookup.get(prompt_id, {})
                records.append({
                    "path": str(img_path),
                    "cfg": cfg,
                    "prompt_id": prompt_id,
                    "category": pinfo.get("category", cat_dir.name),
                    "group": pinfo.get("group", "unknown"),
                })

    print(f"Total images: {len(records)}")

    # Check for cached embeddings
    cache_path = ABLATION_DIR / "base_embeddings.npz"
    if cache_path.exists():
        print("Loading cached embeddings...")
        data = np.load(cache_path, allow_pickle=True)
        embeddings = data["embeddings"]
        cached_meta = json.loads(str(data["metadata"]))
        if len(cached_meta) == len(records):
            print(f"  Loaded {embeddings.shape}")
        else:
            print(f"  Cache mismatch ({len(cached_meta)} vs {len(records)}), re-embedding...")
            embeddings = embed_images_clip([r["path"] for r in records])
            np.savez_compressed(cache_path, embeddings=embeddings, metadata=json.dumps(records))
    else:
        print("Embedding with CLIP (one-time, ~5 min on CPU)...")
        embeddings = embed_images_clip([r["path"] for r in records])
        np.savez_compressed(cache_path, embeddings=embeddings, metadata=json.dumps(records))
        print(f"  Saved cache: {cache_path}")

    # ── Analysis ──
    # Group by (cfg, prompt_id)
    groups = defaultdict(list)
    for i, rec in enumerate(records):
        groups[(rec["cfg"], rec["prompt_id"])].append(i)

    results = []
    for (cfg, prompt_id), indices in sorted(groups.items()):
        embs = embeddings[np.array(indices)]
        pinfo = prompt_lookup.get(prompt_id, {})
        r_alpha = effective_support_radius(embs)
        d_eff = effective_dimension_fast(embs)
        results.append({
            "cfg": cfg, "prompt_id": prompt_id,
            "group": pinfo.get("group", "unknown"),
            "category": pinfo.get("category", "unknown"),
            "n": len(indices), "R_alpha": r_alpha, "d_eff": d_eff,
        })

    # ═══════════════════════════════════════════════════════════
    #  TEST 1: CFG Scale → Homogenization
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  TEST 1: Does CFG Scale Drive Homogenization?")
    print("  (Proposition 3.1: R_α should DECREASE as CFG increases)")
    print(f"{'='*70}")

    print(f"\n  {'CFG':>6} {'Mean R_α':>10} {'Mean d_eff':>10} {'n_prompts':>10}")
    print("  " + "-" * 40)

    cfg_means = {}
    for cfg in cfg_scales:
        subset = [r for r in results if r["cfg"] == cfg]
        mean_r = np.mean([r["R_alpha"] for r in subset])
        mean_d = np.mean([r["d_eff"] for r in subset])
        cfg_means[cfg] = {"R_alpha": mean_r, "d_eff": mean_d}
        print(f"  {cfg:>6.1f} {mean_r:>10.4f} {mean_d:>10.2f} {len(subset):>10}")

    # Check monotonicity
    r_values = [cfg_means[c]["R_alpha"] for c in sorted(cfg_scales)]
    is_monotonic = all(r_values[i] >= r_values[i+1] for i in range(len(r_values)-1))
    total_decrease = (r_values[0] - r_values[-1]) / r_values[0] * 100

    print(f"\n  R_α at CFG=1.0: {r_values[0]:.4f}")
    print(f"  R_α at CFG=15.0: {r_values[-1]:.4f}")
    print(f"  Total decrease: {total_decrease:.1f}%")
    print(f"  Monotonically decreasing: {'YES ✓' if is_monotonic else 'NO ✗'}")

    if is_monotonic and total_decrease > 10:
        print(f"\n  → STRONG SUPPORT for Proposition 3.1:")
        print(f"    CFG systematically reduces diversity even without RLHF")
    elif total_decrease > 5:
        print(f"\n  → MODERATE SUPPORT for Proposition 3.1")
    else:
        print(f"\n  → WEAK/NO SUPPORT for Proposition 3.1")

    # ═══════════════════════════════════════════════════════════
    #  TEST 2: Majority/Minority Gap WITHOUT RLHF
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  TEST 2: Does the Majority/Minority Gap Appear Without RLHF?")
    print("  (If yes → MSE objective is an independent cause)")
    print(f"{'='*70}")

    # Use CFG=7.5 (the standard default) for the gap comparison
    for cfg in cfg_scales:
        ci = [r for r in results if r["cfg"] == cfg and r["prompt_id"].startswith("ci_")]
        maj = [r["R_alpha"] for r in ci if r["group"] == "majority"]
        mino = [r["R_alpha"] for r in ci if r["group"] == "minority"]

        if not maj or not mino:
            continue

        maj_mean = np.mean(maj)
        min_mean = np.mean(mino)
        gap = maj_mean - min_mean
        gap_pct = (gap / maj_mean) * 100 if maj_mean else 0

        print(f"\n  CFG = {cfg}:")
        print(f"    Majority R_α: {maj_mean:.4f} (n={len(maj)} prompts)")
        print(f"    Minority R_α: {min_mean:.4f} (n={len(mino)} prompts)")
        print(f"    Gap: {gap:.4f} ({gap_pct:.1f}%)")

    # Summary at default CFG
    default_ci = [r for r in results if r["cfg"] == 7.5 and r["prompt_id"].startswith("ci_")]
    if default_ci:
        maj = [r["R_alpha"] for r in default_ci if r["group"] == "majority"]
        mino = [r["R_alpha"] for r in default_ci if r["group"] == "minority"]
        if maj and mino:
            gap = np.mean(maj) - np.mean(mino)
            gap_pct = (gap / np.mean(maj)) * 100

            # Bootstrap the gap
            rng = np.random.RandomState(42)
            gap_boots = []
            for _ in range(2000):
                bm = rng.choice(maj, len(maj), replace=True)
                bn = rng.choice(mino, len(mino), replace=True)
                gap_boots.append(np.mean(bm) - np.mean(bn))
            gap_boots = np.array(gap_boots)

            print(f"\n  === SUMMARY at CFG=7.5 (standard default) ===")
            print(f"  Gap: {gap:.4f} ({gap_pct:.1f}%)")
            print(f"  Bootstrap 95% CI: [{np.percentile(gap_boots, 2.5):.4f}, {np.percentile(gap_boots, 97.5):.4f}]")
            print(f"  CI excludes zero: {'YES' if np.percentile(gap_boots, 2.5) > 0 else 'NO'}")
            print(f"  P(gap > 0): {(gap_boots > 0).mean():.3f}")

            if gap > 0 and (gap_boots > 0).mean() > 0.9:
                print(f"\n  → SUPPORTS MSE objective as independent cause of minority suppression")
                print(f"    The gap appears in a base model with NO RLHF or preference training")
            elif gap > 0:
                print(f"\n  → SUGGESTIVE but not conclusive (gap present but uncertain)")
            else:
                print(f"\n  → DOES NOT SUPPORT MSE as independent cause")
                print(f"    No majority/minority gap without RLHF — effect may be RLHF-driven")

    # ═══════════════════════════════════════════════════════════
    #  TEST 3: Per-prompt detail at CFG=7.5
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  Per-Prompt Detail (CFG=7.5)")
    print(f"{'='*70}")

    print(f"\n  {'Prompt':<20} {'Group':<10} {'R_α':>8} {'d_eff':>8}")
    print("  " + "-" * 50)
    for r in sorted([r for r in results if r["cfg"] == 7.5],
                     key=lambda x: (0 if x["group"]=="majority" else (1 if x["group"]=="minority" else 2), x["prompt_id"])):
        print(f"  {r['prompt_id']:<20} {r['group']:<10} {r['R_alpha']:>8.4f} {r['d_eff']:>8.2f}")

    # ═══════════════════════════════════════════════════════════
    #  Comparison with commercial models
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  COMPARISON: SDXL Base vs Commercial Models (at CFG=7.5)")
    print(f"{'='*70}")

    # Load commercial model results if available
    commercial_emb_path = Path(__file__).parent / "embeddings" / "embeddings.npz"
    if commercial_emb_path.exists():
        comm_data = np.load(commercial_emb_path, allow_pickle=True)
        comm_embs = comm_data["embeddings"]
        comm_meta = json.loads(str(comm_data["metadata"]))

        # Compute commercial metrics for matching prompts
        comm_groups = defaultdict(list)
        for i, rec in enumerate(comm_meta):
            if rec["platform"] == "openai":
                comm_groups[rec["prompt_id"]].append(i)

        ablation_prompts = set(r["prompt_id"] for r in results)
        print(f"\n  {'Prompt':<20} {'SDXL Base R_α':>15} {'DALL-E 3 R_α':>15} {'Ratio':>8}")
        print("  " + "-" * 62)

        sdxl_vals = []
        dalle_vals = []
        for pid in sorted(ablation_prompts):
            sdxl_r = next((r["R_alpha"] for r in results
                           if r["prompt_id"] == pid and r["cfg"] == 7.5), None)
            if pid in comm_groups:
                dalle_embs = comm_embs[np.array(comm_groups[pid])]
                dalle_r = effective_support_radius(dalle_embs)
            else:
                dalle_r = None

            if sdxl_r is not None and dalle_r is not None:
                ratio = sdxl_r / dalle_r
                sdxl_vals.append(sdxl_r)
                dalle_vals.append(dalle_r)
                print(f"  {pid:<20} {sdxl_r:>15.4f} {dalle_r:>15.4f} {ratio:>8.2f}")

        if sdxl_vals:
            mean_ratio = np.mean(sdxl_vals) / np.mean(dalle_vals)
            print(f"\n  Mean SDXL Base R_α: {np.mean(sdxl_vals):.4f}")
            print(f"  Mean DALL-E 3 R_α:  {np.mean(dalle_vals):.4f}")
            print(f"  Ratio (SDXL/DALL-E): {mean_ratio:.2f}")

            if mean_ratio > 1.1:
                print(f"  → SDXL base is MORE diverse than DALL-E 3")
                print(f"    Suggests RLHF/post-training adds additional compression")
            elif mean_ratio < 0.9:
                print(f"  → SDXL base is LESS diverse than DALL-E 3")
            else:
                print(f"  → Similar diversity levels")

    # Save results
    output_path = ABLATION_DIR / "base_analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {output_path}")

if __name__ == "__main__":
    main()
