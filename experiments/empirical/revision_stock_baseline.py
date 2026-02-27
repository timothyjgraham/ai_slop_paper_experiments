#!/usr/bin/env python3
"""
Non-AI Baseline: Stock Photography Diversity Comparison
========================================================
Addresses reviewer concern: "If you searched Getty Images or Shutterstock
for 'a photograph of a Bangladeshi woman,' how diverse would those results be?"

This script:
1. Queries free stock photo APIs (Pexels, Unsplash) for equivalent searches
2. Downloads top-N results per query
3. Embeds with both CLIP and DINOv2
4. Computes same diversity metrics as the main analysis
5. Compares stock photo diversity against AI-generated image diversity

If stock photos are equally narrow → the problem is in the data, not the objective
If stock photos are substantially more diverse → the MSE objective adds compression

Requirements:
  pip install requests Pillow numpy torch transformers tqdm

  # API keys (free):
  export PEXELS_API_KEY="..."       # https://www.pexels.com/api/
  export UNSPLASH_ACCESS_KEY="..."  # https://unsplash.com/developers

Usage:
  python revision_stock_baseline.py
  python revision_stock_baseline.py --analysis-only  # if images already downloaded
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Configuration ─────────────────────────────────────────────

N_IMAGES_PER_QUERY = 20
ALPHA = 0.90

# Map prompt IDs to stock photo search queries
# These should match the AI prompts as closely as possible
STOCK_QUERIES = {
    "ci_american": "American woman portrait",
    "ci_british": "British woman portrait",
    "ci_french": "French woman portrait",
    "ci_australian": "Australian woman portrait",
    "ci_canadian": "Canadian woman portrait",
    "ci_bangladeshi": "Bangladeshi woman portrait",
    "ci_nigerian": "Nigerian woman portrait",
    "ci_peruvian": "Peruvian woman portrait",
    "ci_mongolian": "Mongolian woman portrait",
    "ci_ethiopian": "Ethiopian woman portrait",
    "ci_kurdish": "Kurdish woman portrait",
    "ci_maori": "Maori woman portrait",
    "ci_uzbek": "Uzbek woman portrait",
    "ci_guatemalan": "Guatemalan woman portrait",
    "ci_samoan": "Samoan woman portrait",
    "oe_landscape": "beautiful landscape",
    "oe_home": "ideal home interior",
    "oe_professional": "professional person office",
    "oe_meal": "delicious meal food photography",
}


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


# ─── Stock Photo APIs ──────────────────────────────────────────

def search_pexels(query, n=N_IMAGES_PER_QUERY):
    """Search Pexels API for photos matching query."""
    import requests

    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        print("WARNING: PEXELS_API_KEY not set, skipping Pexels")
        return []

    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": min(n, 80), "orientation": "square"}

    try:
        resp = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        urls = []
        for photo in data.get("photos", [])[:n]:
            # Use medium size (350px) for efficiency
            urls.append({
                "url": photo["src"]["medium"],
                "id": str(photo["id"]),
                "source": "pexels",
                "photographer": photo.get("photographer", ""),
            })
        return urls
    except Exception as e:
        print(f"  Pexels error for '{query}': {e}")
        return []


def search_unsplash(query, n=N_IMAGES_PER_QUERY):
    """Search Unsplash API for photos matching query."""
    import requests

    access_key = os.environ.get("UNSPLASH_ACCESS_KEY")
    if not access_key:
        print("WARNING: UNSPLASH_ACCESS_KEY not set, skipping Unsplash")
        return []

    params = {
        "query": query,
        "per_page": min(n, 30),
        "client_id": access_key,
    }

    try:
        resp = requests.get("https://api.unsplash.com/search/photos", params=params)
        resp.raise_for_status()
        data = resp.json()
        urls = []
        for photo in data.get("results", [])[:n]:
            urls.append({
                "url": photo["urls"]["small"],  # 400px
                "id": photo["id"],
                "source": "unsplash",
                "photographer": photo.get("user", {}).get("name", ""),
            })
        return urls
    except Exception as e:
        print(f"  Unsplash error for '{query}': {e}")
        return []


def download_images(results, output_dir, prompt_id):
    """Download stock photos to disk."""
    import requests
    from PIL import Image
    from io import BytesIO

    downloaded = []
    for i, item in enumerate(results):
        filepath = output_dir / f"{prompt_id}_{item['source']}_{i:03d}.png"
        if filepath.exists():
            downloaded.append(str(filepath))
            continue

        try:
            resp = requests.get(item["url"], timeout=30)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img.save(filepath)
            downloaded.append(str(filepath))
            time.sleep(0.5)  # Be polite to APIs
        except Exception as e:
            print(f"    Failed to download {item['url']}: {e}")

    return downloaded


# ─── Main Pipeline ─────────────────────────────────────────────

def collect_stock_photos(output_dir):
    """Search and download stock photos for all prompts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metadata = {}

    for prompt_id, query in STOCK_QUERIES.items():
        print(f"\n  {prompt_id}: \"{query}\"")
        prompt_dir = output_dir / prompt_id
        prompt_dir.mkdir(exist_ok=True)

        # Search both APIs
        pexels_results = search_pexels(query, n=N_IMAGES_PER_QUERY)
        unsplash_results = search_unsplash(query, n=N_IMAGES_PER_QUERY)

        # Combine and deduplicate
        all_results = pexels_results + unsplash_results
        print(f"    Found: {len(pexels_results)} Pexels + {len(unsplash_results)} Unsplash")

        # Download
        paths = download_images(all_results, prompt_dir, prompt_id)
        print(f"    Downloaded: {len(paths)} images")

        all_metadata[prompt_id] = {
            "query": query,
            "n_pexels": len(pexels_results),
            "n_unsplash": len(unsplash_results),
            "n_downloaded": len(paths),
            "paths": paths,
        }

    # Save metadata
    with open(output_dir / "stock_metadata.json", "w") as f:
        json.dump(all_metadata, f, indent=2)

    return all_metadata


def embed_and_analyze(output_dir, ai_embeddings_path):
    """Embed stock photos, compute metrics, and compare against AI results."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    from tqdm import tqdm

    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load stock metadata
    with open(output_dir / "stock_metadata.json") as f:
        stock_meta = json.load(f)

    # Load AI embeddings for comparison
    ai_data = np.load(ai_embeddings_path, allow_pickle=True)
    ai_embeddings = ai_data["embeddings"]
    ai_metadata = json.loads(str(ai_data["metadata"]))

    # Group AI embeddings by prompt_id (OpenAI only for cleaner comparison)
    ai_groups = defaultdict(list)
    for i, rec in enumerate(ai_metadata):
        if rec["platform"] == "openai":
            ai_groups[rec["prompt_id"]].append(i)

    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP on {device}...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()

    # Embed stock photos and compute metrics
    stock_results = {}
    ai_results = {}

    print(f"\n{'Prompt':<20} {'Source':<10} {'n':>4} {'R_α':>8} {'d_eff':>8}")
    print("-" * 55)

    for prompt_id, meta in stock_meta.items():
        paths = meta["paths"]
        if len(paths) < 5:
            print(f"  {prompt_id}: too few stock photos ({len(paths)}), skipping")
            continue

        # Embed stock photos
        images = []
        for p in paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except:
                pass

        if len(images) < 5:
            continue

        inputs = clip_processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output if hasattr(features, 'pooler_output') else features[0]
            features = features / features.norm(dim=-1, keepdim=True)

        stock_embs = features.cpu().numpy()
        stock_r = effective_support_radius(stock_embs)
        stock_d = effective_dimension_fast(stock_embs)
        stock_results[prompt_id] = {"R_alpha": stock_r, "d_eff": stock_d, "n": len(images)}

        print(f"{prompt_id:<20} {'stock':<10} {len(images):>4} {stock_r:>8.4f} {stock_d:>8.2f}")

        # AI comparison
        if prompt_id in ai_groups:
            ai_embs = ai_embeddings[np.array(ai_groups[prompt_id])]
            ai_r = effective_support_radius(ai_embs)
            ai_d = effective_dimension_fast(ai_embs)
            ai_results[prompt_id] = {"R_alpha": ai_r, "d_eff": ai_d, "n": ai_embs.shape[0]}
            print(f"{prompt_id:<20} {'AI (OAI)':<10} {ai_embs.shape[0]:>4} {ai_r:>8.4f} {ai_d:>8.2f}")

    # ── Summary comparison ──
    print(f"\n{'='*60}")
    print("  STOCK vs AI — Diversity Comparison")
    print(f"{'='*60}")

    common_prompts = set(stock_results.keys()) & set(ai_results.keys())
    if common_prompts:
        stock_r_vals = [stock_results[p]["R_alpha"] for p in common_prompts]
        ai_r_vals = [ai_results[p]["R_alpha"] for p in common_prompts]

        print(f"\n  Mean R_α (stock photos): {np.mean(stock_r_vals):.4f}")
        print(f"  Mean R_α (AI-generated): {np.mean(ai_r_vals):.4f}")
        ratio = np.mean(ai_r_vals) / np.mean(stock_r_vals) if np.mean(stock_r_vals) > 0 else 0
        print(f"  AI / Stock ratio: {ratio:.3f}")

        if ratio < 0.8:
            print(f"  → AI images are {(1-ratio)*100:.0f}% less diverse than stock photos")
            print(f"    This suggests the generative process adds compression beyond training data")
        elif ratio > 1.2:
            print(f"  → AI images are MORE diverse than stock photos (!)")
            print(f"    This would undermine the paper's central claim")
        else:
            print(f"  → Similar diversity levels")
            print(f"    This suggests the problem may be primarily in the data")

    # Save results
    output = {
        "stock_results": stock_results,
        "ai_results": ai_results,
        "comparison": {
            "common_prompts": list(common_prompts),
            "stock_mean_R_alpha": float(np.mean(stock_r_vals)) if common_prompts else None,
            "ai_mean_R_alpha": float(np.mean(ai_r_vals)) if common_prompts else None,
        }
    }
    with open(figures_dir / "stock_baseline_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {figures_dir}/stock_baseline_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock photography baseline comparison")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent / "stock_baseline"),
                        help="Output directory for stock photos")
    parser.add_argument("--ai-embeddings", type=str,
                        default=str(Path(__file__).parent / "embeddings" / "embeddings.npz"),
                        help="Path to AI image embeddings for comparison")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Only run analysis on existing downloads")
    args = parser.parse_args()

    if args.analysis_only:
        embed_and_analyze(args.output_dir, args.ai_embeddings)
    else:
        collect_stock_photos(args.output_dir)
        embed_and_analyze(args.output_dir, args.ai_embeddings)
