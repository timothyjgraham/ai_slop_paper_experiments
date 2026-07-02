#!/usr/bin/env python3
"""
Ablation Study: Disentangling MSE Objective from RLHF/Data Effects
===================================================================
This is the most important new experiment for the revision.

The reviewer's central concern: "The observed homogenization could be caused
by biased training data alone, by RLHF preference training, by content
filtering, or by the MSE objective — or by some combination. The current
experimental design cannot distinguish between these causes."

Design:
  Factor 1: RLHF
    - SDXL base (no RLHF)
    - SDXL + DPO (with RLHF-like preference training)

  Factor 2: CFG scale (tests Proposition 3.1)
    - w ∈ {1.0, 3.0, 5.0, 7.5, 10.0, 15.0}

  Factor 3: (Optional) Loss function
    - SDXL (standard noise prediction / epsilon-parameterization)
    - SD3 (rectified flow matching — different loss)

If the MSE objective is doing the work the paper claims:
  → Homogenization should appear even WITHOUT RLHF (in the base model)
  → Homogenization should increase monotonically with CFG scale
  → The majority/minority gap should be present in the base model

If RLHF is the primary driver:
  → Base model should show substantially MORE diversity
  → The DPO variant should show the homogenization pattern
  → CFG scale should have a smaller effect relative to RLHF

Requirements:
  pip install torch diffusers transformers accelerate safetensors
  pip install Pillow numpy scipy matplotlib tqdm

  # Supports: CUDA GPU, Apple Silicon (MPS), or CPU (very slow)
  # Apple Silicon M3 Max: ~30-45s per image (~8-12 hours for reduced run)
  # CUDA A100: ~3s per image (~2 hours for full run)

Usage:
  # Full ablation (CUDA GPU)
  python revision_ablation_study.py --all

  # Reduced ablation for Apple Silicon (overnight run, ~8-12 hours)
  python revision_ablation_study.py --all --apple-silicon

  # Just SDXL base at multiple CFG scales
  python revision_ablation_study.py --model sdxl-base --cfg-scales 1 3 5 7.5 10 15

  # Just the DPO comparison
  python revision_ablation_study.py --model sdxl-dpo --cfg-scales 7.5

  # Analysis only (after generation)
  python revision_ablation_study.py --analysis-only

  # Dry run to check setup
  python revision_ablation_study.py --dry-run
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Configuration ─────────────────────────────────────────────

# Models to compare
MODELS = {
    "sdxl-base": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "label": "SDXL Base (no RLHF)",
        "description": "Standard SDXL, MSE objective only, no preference training",
        "color": "#2166ac",
    },
    "sdxl-dpo": {
        "model_id": "mhdang/dpo-sdxl-text2image-v1",
        "label": "SDXL + DPO",
        "description": "SDXL fine-tuned with Direct Preference Optimization",
        "color": "#b2182b",
        "unet_only": True,  # DPO model is a finetuned UNet, not a full pipeline
    },
    # Optional: flow-matching comparison
    # "sd3-medium": {
    #     "model_id": "stabilityai/stable-diffusion-3-medium-diffusers",
    #     "label": "SD3 Medium (Flow Matching)",
    #     "description": "Rectified flow, different loss function entirely",
    #     "color": "#4dac26",
    # },
}

# CFG scales to test — this is the key ablation for Proposition 3.1
CFG_SCALES = [1.0, 3.0, 5.0, 7.5, 10.0, 15.0]
CFG_SCALES_REDUCED = [1.0, 5.0, 7.5, 15.0]  # Reduced set for Apple Silicon overnight run

# Use the same prompts as the main experiment
# We'll focus on cultural identity prompts (the 15% gap finding)
# plus a few open-ended prompts for context
ABLATION_PROMPT_IDS = [
    # Majority
    "ci_american", "ci_british", "ci_french",
    # Minority
    "ci_bangladeshi", "ci_nigerian", "ci_mongolian", "ci_ethiopian", "ci_kurdish",
    # Open-ended (to test general convergence)
    "oe_landscape", "oe_home", "oe_professional",
]

N_IMAGES_PER_CONDITION = 20
IMAGE_SIZE = 1024
SEED_BASE = 42  # Use fixed seeds for reproducibility

ALPHA = 0.90

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


# ─── Metrics (same as main pipeline) ───────────────────────────

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


# ─── Image Generation ──────────────────────────────────────────

def get_device():
    """Detect best available device: CUDA > MPS (Apple Silicon) > CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def generate_ablation_images(model_key, cfg_scales, prompts, output_dir, n_per_condition, dry_run=False):
    """Generate images for one model at multiple CFG scales.

    Uses fixed random seeds so results are reproducible and comparable
    across models (same latent noise, different denoising).

    Supports CUDA, Apple Silicon (MPS), and CPU.
    """
    if dry_run:
        total = len(prompts) * len(cfg_scales) * n_per_condition
        print(f"[DRY RUN] Would generate {total} images for {model_key}")
        print(f"  CFG scales: {cfg_scales}")
        print(f"  Prompts: {[p['id'] for p in prompts]}")
        return

    import torch
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel

    model_info = MODELS[model_key]
    device = get_device()

    print(f"\nLoading {model_info['label']}...")
    print(f"  Model: {model_info['model_id']}")
    print(f"  Device: {device}")

    if device == "cpu":
        print("  WARNING: Running on CPU will be very slow.")

    # Determine dtype — float16 on CUDA, float32 on MPS
    # NOTE: float16 on MPS causes black/NaN images with SDXL.
    # Apple Silicon must use float32 (it has unified memory, so this is fine).
    use_fp16 = device == "cuda"
    dtype = torch.float16 if use_fp16 else torch.float32

    # Load pipeline
    # DPO models are finetuned UNets loaded into the base SDXL pipeline
    if model_info.get("unet_only"):
        print(f"  Loading base SDXL pipeline + DPO UNet from {model_info['model_id']}...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if use_fp16 else None,
        )
        # Swap in the DPO-finetuned UNet
        dpo_unet = UNet2DConditionModel.from_pretrained(
            model_info["model_id"],
            subfolder="unet",
            torch_dtype=dtype,
        )
        pipe.unet = dpo_unet
        pipe = pipe.to(device)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_info["model_id"],
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if use_fp16 else None,
        ).to(device)

    # Use DPM++ 2M Karras for consistency
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )

    # Disable NSFW safety checker — it produces black images for legitimate
    # research prompts like "a photograph of an American woman". This is
    # standard practice for academic research with diffusion models.
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    # On MPS, enable attention slicing to reduce peak memory
    if device == "mps":
        pipe.enable_attention_slicing()

    # Track progress
    total_images = len(cfg_scales) * len(prompts) * n_per_condition
    generated_count = 0
    skipped_count = 0
    start_time = time.time()

    # Generate
    for cfg in cfg_scales:
        print(f"\n  CFG scale = {cfg}")

        for prompt_info in prompts:
            prompt_id = prompt_info["id"]
            prompt_text = prompt_info["prompt"]
            cat_dir = output_dir / model_key / f"cfg_{cfg}" / prompt_info["category"]
            cat_dir.mkdir(parents=True, exist_ok=True)

            print(f"    {prompt_id}: \"{prompt_text}\"")

            for i in range(n_per_condition):
                filepath = cat_dir / f"{prompt_id}_{i:03d}.png"
                if filepath.exists():
                    skipped_count += 1
                    continue

                # Fixed seed per (prompt, image_index) — same noise across models/CFG
                # Generator must be on CPU for MPS (MPS doesn't support torch.Generator)
                seed = SEED_BASE + hash(prompt_id) % 10000 + i
                gen_device = "cpu" if device == "mps" else device
                generator = torch.Generator(device=gen_device).manual_seed(seed)

                try:
                    result = pipe(
                        prompt=prompt_text,
                        num_inference_steps=30,
                        guidance_scale=cfg,
                        generator=generator,
                        height=IMAGE_SIZE,
                        width=IMAGE_SIZE,
                    )
                    result.images[0].save(filepath)
                    generated_count += 1

                    # Progress reporting
                    elapsed = time.time() - start_time
                    done = generated_count + skipped_count
                    remaining = total_images - done
                    if generated_count > 0:
                        per_image = elapsed / generated_count
                        eta_min = (remaining * per_image) / 60
                        print(f"      [{done}/{total_images}] saved "
                              f"({per_image:.1f}s/img, ~{eta_min:.0f} min remaining)")

                except Exception as e:
                    print(f"      ERROR generating {filepath.name}: {e}")

    elapsed_total = (time.time() - start_time) / 60
    print(f"\n  Done generating for {model_key}: "
          f"{generated_count} new, {skipped_count} skipped, {elapsed_total:.1f} min total")


# ─── CLIP Embedding ────────────────────────────────────────────

def embed_ablation_images(output_dir, model_keys, cfg_scales, prompts):
    """Embed all ablation images with CLIP and save per-condition embeddings.
    Also embeds with DINOv2 for cross-space validation."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    from tqdm import tqdm

    device = get_device()
    print(f"\nLoading CLIP for embedding on {device}...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()

    results = {}  # (model_key, cfg, prompt_id) -> embeddings array

    for model_key in model_keys:
        for cfg in cfg_scales:
            for prompt_info in prompts:
                prompt_id = prompt_info["id"]
                cat_dir = output_dir / model_key / f"cfg_{cfg}" / prompt_info["category"]

                image_files = sorted(cat_dir.glob(f"{prompt_id}_*.png"))
                if not image_files:
                    continue

                images = []
                for f in image_files:
                    try:
                        images.append(Image.open(f).convert("RGB"))
                    except:
                        pass

                if not images:
                    continue

                # Embed in one batch
                inputs = clip_processor(images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    features = clip_model.get_image_features(**inputs)
                    if not isinstance(features, torch.Tensor):
                        features = features.pooler_output if hasattr(features, 'pooler_output') else features[0]
                    features = features / features.norm(dim=-1, keepdim=True)

                key = (model_key, float(cfg), prompt_id)
                results[key] = features.cpu().numpy()
                print(f"  Embedded {model_key}/cfg={cfg}/{prompt_id}: {features.shape[0]} images")

    # Save
    emb_path = output_dir / "ablation_embeddings.npz"
    save_dict = {}
    index = []
    for (mk, cfg, pid), embs in results.items():
        safe_key = f"{mk}__cfg{cfg}__{pid}"
        save_dict[safe_key] = embs
        index.append({"model": mk, "cfg": cfg, "prompt_id": pid, "key": safe_key, "n": embs.shape[0]})

    np.savez_compressed(emb_path, **save_dict)
    with open(output_dir / "ablation_index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n  Saved ablation embeddings: {emb_path}")
    return results


# ─── Analysis ──────────────────────────────────────────────────

def run_ablation_analysis(output_dir, prompt_lookup):
    """Analyze ablation results and produce key figures."""

    # Load embeddings
    emb_path = output_dir / "ablation_embeddings.npz"
    idx_path = output_dir / "ablation_index.json"

    if not emb_path.exists():
        print("ERROR: No ablation embeddings found. Run generation + embedding first.")
        sys.exit(1)

    data = np.load(emb_path, allow_pickle=True)
    with open(idx_path) as f:
        index = json.load(f)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Compute metrics per condition
    results = []
    for entry in index:
        embs = data[entry["key"]]
        pinfo = prompt_lookup.get(entry["prompt_id"], {})

        r_alpha = effective_support_radius(embs)
        d_eff = effective_dimension_fast(embs)

        results.append({
            "model": entry["model"],
            "cfg": entry["cfg"],
            "prompt_id": entry["prompt_id"],
            "group": pinfo.get("group", "unknown"),
            "category": pinfo.get("category", "unknown"),
            "n": entry["n"],
            "R_alpha": r_alpha,
            "d_eff": d_eff,
        })

    print(f"\n{'='*80}")
    print("  ABLATION STUDY RESULTS")
    print(f"{'='*80}")

    # ── Key Test 1: Does CFG scale monotonically increase homogenization? ──
    print(f"\n  Test 1: CFG Scale → Homogenization (Proposition 3.1)")
    print(f"  {'Model':<15} {'CFG':>6} {'Mean R_α':>10} {'Mean d_eff':>10} {'n_prompts':>10}")
    print("  " + "-" * 55)

    for model in sorted(set(r["model"] for r in results)):
        for cfg in sorted(set(r["cfg"] for r in results)):
            subset = [r for r in results if r["model"] == model and r["cfg"] == cfg]
            if not subset:
                continue
            mean_r = np.mean([r["R_alpha"] for r in subset])
            mean_d = np.mean([r["d_eff"] for r in subset])
            print(f"  {model:<15} {cfg:>6.1f} {mean_r:>10.4f} {mean_d:>10.2f} {len(subset):>10}")

    # ── Key Test 2: Does the majority/minority gap appear WITHOUT RLHF? ──
    print(f"\n  Test 2: Majority/Minority Gap — Base vs DPO")

    for model in sorted(set(r["model"] for r in results)):
        ci = [r for r in results if r["model"] == model and r["prompt_id"].startswith("ci_")]
        if not ci:
            continue

        # Use default CFG (7.5) for this comparison
        ci_default = [r for r in ci if abs(r["cfg"] - 7.5) < 0.1]
        if not ci_default:
            ci_default = ci  # Use whatever CFG is available

        maj = [r["R_alpha"] for r in ci_default if r["group"] == "majority"]
        mino = [r["R_alpha"] for r in ci_default if r["group"] == "minority"]

        if maj and mino:
            gap = np.mean(maj) - np.mean(mino)
            gap_pct = (gap / np.mean(maj)) * 100 if np.mean(maj) else 0
            print(f"\n  {MODELS.get(model, {}).get('label', model)}:")
            print(f"    Majority R_α: {np.mean(maj):.4f}")
            print(f"    Minority R_α: {np.mean(mino):.4f}")
            print(f"    Gap: {gap:.4f} ({gap_pct:.1f}%)")

            if model == "sdxl-base" and gap > 0:
                print(f"    → SUPPORTS MSE objective as independent cause")
            elif model == "sdxl-base" and gap <= 0:
                print(f"    → SUGGESTS gap is due to RLHF, not MSE objective")

    # ── Generate figures ──
    _plot_cfg_ablation(results, figures_dir)
    _plot_model_comparison(results, figures_dir)

    # Save
    with open(figures_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {figures_dir}/ablation_results.json")


def _plot_cfg_ablation(results, figures_dir):
    """Plot diversity metrics as a function of CFG scale, per model."""

    models = sorted(set(r["model"] for r in results))
    cfgs = sorted(set(r["cfg"] for r in results))

    if len(cfgs) < 2:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for model in models:
        color = MODELS.get(model, {}).get("color", "#999")
        label = MODELS.get(model, {}).get("label", model)

        cfg_vals = []
        r_means = []
        d_means = []

        for cfg in cfgs:
            subset = [r for r in results if r["model"] == model and r["cfg"] == cfg]
            if subset:
                cfg_vals.append(cfg)
                r_means.append(np.mean([r["R_alpha"] for r in subset]))
                d_means.append(np.mean([r["d_eff"] for r in subset]))

        if cfg_vals:
            ax1.plot(cfg_vals, r_means, "o-", color=color, label=label, linewidth=2)
            ax2.plot(cfg_vals, d_means, "o-", color=color, label=label, linewidth=2)

    ax1.set_xlabel("Classifier-Free Guidance Scale (w)")
    ax1.set_ylabel(r"Mean $R_\alpha$ (Support Radius)")
    ax1.set_title("Support Radius vs. CFG Scale\n(Proposition 3.1: should decrease with w)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Classifier-Free Guidance Scale (w)")
    ax2.set_ylabel(r"Mean $d_{\mathrm{eff}}$ (Effective Dimension)")
    ax2.set_title("Effective Dimension vs. CFG Scale\n(Should decrease with w)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Ablation: Effect of CFG Scale on Diversity\n"
                 "(Tests whether MSE denoising objective drives homogenization)",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(figures_dir / "ablation_cfg_scale.png")
    plt.close()
    print(f"  Figure saved: {figures_dir}/ablation_cfg_scale.png")


def _plot_model_comparison(results, figures_dir):
    """Bar chart comparing base vs DPO for majority/minority prompts."""

    models = sorted(set(r["model"] for r in results))
    if len(models) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35

    for g_idx, group in enumerate(["majority", "minority"]):
        means = []
        stds = []
        for model in models:
            # Use default CFG
            vals = [r["R_alpha"] for r in results
                    if r["model"] == model and r["group"] == group
                    and abs(r["cfg"] - 7.5) < 0.1]
            if not vals:
                vals = [r["R_alpha"] for r in results
                        if r["model"] == model and r["group"] == group]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)

        color = "#2166ac" if group == "majority" else "#b2182b"
        ax.bar(x + g_idx * width, means, width, yerr=stds,
               color=color, alpha=0.85, label=group.capitalize(), capsize=5)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([MODELS.get(m, {}).get("label", m) for m in models])
    ax.set_ylabel(r"$R_\alpha$ (Support Radius)")
    ax.set_title("Ablation: Base Model vs. DPO — Majority/Minority Diversity\n"
                 "(If gap appears in base model → MSE objective is an independent cause)")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(figures_dir / "ablation_base_vs_dpo.png")
    plt.close()
    print(f"  Figure saved: {figures_dir}/ablation_base_vs_dpo.png")


# ─── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ablation study for paper revision")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all",
                        help="Which model(s) to run")
    parser.add_argument("--cfg-scales", nargs="+", type=float, default=CFG_SCALES,
                        help=f"CFG scales to test (default: {CFG_SCALES})")
    parser.add_argument("--n-per-condition", type=int, default=N_IMAGES_PER_CONDITION,
                        help=f"Images per condition (default: {N_IMAGES_PER_CONDITION})")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent / "ablation"),
                        help="Output directory for ablation images")
    parser.add_argument("--prompts", type=str,
                        default=str(Path(__file__).parent / "prompts.json"),
                        help="Path to prompts.json")
    parser.add_argument("--all", action="store_true",
                        help="Run full ablation (all models, all CFG scales)")
    parser.add_argument("--apple-silicon", action="store_true",
                        help="Use reduced settings optimized for Apple Silicon overnight run: "
                             "4 CFG scales instead of 6, 10 images per condition")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Only run analysis on existing images")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without generating")

    args = parser.parse_args()

    # Load prompts
    with open(args.prompts) as f:
        all_prompts = json.load(f)["prompts"]

    # Filter to ablation subset
    prompts = [p for p in all_prompts if p["id"] in ABLATION_PROMPT_IDS]
    prompt_lookup = {p["id"]: p for p in all_prompts}
    print(f"Ablation prompts: {len(prompts)} of {len(all_prompts)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_keys = list(MODELS.keys()) if (args.model == "all" or args.all) else [args.model]

    # Apply Apple Silicon optimizations
    if args.apple_silicon:
        if args.cfg_scales == CFG_SCALES:  # Only override if user didn't specify custom
            args.cfg_scales = CFG_SCALES_REDUCED
        if args.n_per_condition == N_IMAGES_PER_CONDITION:  # Only override if default
            args.n_per_condition = 10
        print(f"  Apple Silicon mode: using {len(args.cfg_scales)} CFG scales, "
              f"{args.n_per_condition} images/condition")

    if args.analysis_only:
        run_ablation_analysis(output_dir, prompt_lookup)
        return

    # Estimate cost/time
    total_images = len(model_keys) * len(args.cfg_scales) * len(prompts) * args.n_per_condition
    print(f"\nTotal images to generate: {total_images}")
    print(f"  Models: {model_keys}")
    print(f"  CFG scales: {args.cfg_scales}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Per condition: {args.n_per_condition}")

    # Estimate time based on device
    import torch
    if torch.cuda.is_available():
        sec_per_img = 3
        device_label = "CUDA GPU"
    elif torch.backends.mps.is_available():
        sec_per_img = 40
        device_label = "Apple Silicon (MPS)"
    else:
        sec_per_img = 180
        device_label = "CPU"
    est_hours = (total_images * sec_per_img) / 3600
    print(f"  Device: {device_label}")
    print(f"  Estimated time: {est_hours:.1f} hours")

    if args.dry_run:
        print("\n[DRY RUN] No images generated.")
        return

    # Generate images
    for model_key in model_keys:
        generate_ablation_images(
            model_key, args.cfg_scales, prompts, output_dir,
            args.n_per_condition, dry_run=False,
        )

    # Embed
    embed_ablation_images(output_dir, model_keys, args.cfg_scales, prompts)

    # Analyze
    run_ablation_analysis(output_dir, prompt_lookup)


if __name__ == "__main__":
    main()
