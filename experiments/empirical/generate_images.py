#!/usr/bin/env python3
"""
Image Generation Script for MAP/SLOP Empirical Validation
==========================================================
Generates images from OpenAI (DALL-E 3) and Google (Imagen 4) APIs
using the structured prompt set designed to test the Mode Averaging
Principle's predictions on real commercial models.

Setup:
  1. Install dependencies:
       pip install openai google-genai Pillow tqdm

  2. Set API keys as environment variables:
       export OPENAI_API_KEY="sk-..."
       export GOOGLE_API_KEY="AIza..."

     To get these keys:
       - OpenAI:  https://platform.openai.com/api-keys
                  (requires account with billing enabled)
       - Google:  https://aistudio.google.com/apikey
                  (free tier available)

  3. Run:
       python generate_images.py --platform all --n-per-prompt 20
       python generate_images.py --platform openai --n-per-prompt 2   # test run

Usage:
  python generate_images.py [OPTIONS]

Options:
  --platform {openai,google,all}   Which API(s) to use (default: all)
  --prompts PATH                   Path to prompts.json (default: ./prompts.json)
  --n-per-prompt N                 Images per prompt per model (default: 20)
  --output-dir PATH                Where to save images (default: ./images)
  --start-index N                  Resume from image index N (default: 0)
  --dry-run                        Print what would be generated without calling APIs

Estimated costs (at n=20, 30 prompts):
  - OpenAI DALL-E 3: ~$48  (1200 images x $0.040/image at 1024x1024)
  - Google Imagen 4 Fast: ~$24  (1200 images x $0.020/image)
  - Google Imagen 4 Full: ~$324 (1200 images x $0.270/image — significantly more expensive)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ─── Lazy imports (only load what's needed) ────────────────────

def get_openai_client():
    """Initialise and return OpenAI client."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("  Get your key at: https://platform.openai.com/api-keys")
        print("  Then run: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    return OpenAI(api_key=api_key)


def get_google_client():
    """Initialise and return Google GenAI client."""
    try:
        from google import genai
    except ImportError:
        print("ERROR: google-genai package not installed. Run: pip install google-genai")
        sys.exit(1)

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not set.")
        print("  Get your key at: https://aistudio.google.com/apikey")
        print("  Then run: export GOOGLE_API_KEY='AIza...'")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    return client


# ─── Generation functions ──────────────────────────────────────

def generate_openai(client, prompt_text, output_path):
    """Generate a single image with DALL-E 3 and save to output_path.

    Returns True on success, False on failure.
    """
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt_text,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url

        # Download the image
        import urllib.request
        urllib.request.urlretrieve(image_url, str(output_path))
        return True

    except Exception as e:
        error_msg = str(e)
        # Handle content policy rejections gracefully
        if "content_policy" in error_msg.lower() or "safety" in error_msg.lower():
            print(f"    [BLOCKED by content policy] {error_msg[:120]}")
        else:
            print(f"    [ERROR] {error_msg[:120]}")
        return False


def generate_google(client, prompt_text, output_path, max_retries=5):
    """Generate a single image with Google Imagen 4 Fast and save to output_path.

    Returns True on success, False on failure.
    Retries with exponential backoff on rate-limit (429) errors.
    """
    from google.genai import types

    for attempt in range(max_retries):
        try:
            response = client.models.generate_images(
                model="imagen-4.0-fast-generate-001",
                prompt=prompt_text,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="1:1",
                    safety_filter_level="BLOCK_LOW_AND_ABOVE",
                ),
            )

            if response.generated_images and len(response.generated_images) > 0:
                image = response.generated_images[0].image
                with open(output_path, "wb") as f:
                    f.write(image.image_bytes)
                return True
            else:
                print(f"    [NO IMAGE RETURNED] The API returned no images")
                return False

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                wait_time = 30 * (2 ** attempt)  # 30s, 60s, 120s, 240s, 480s
                print(f"    [RATE LIMITED] Waiting {wait_time}s before retry "
                      f"({attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            elif "safety" in error_msg.lower() or "block" in error_msg.lower():
                print(f"    [BLOCKED by safety filter] {error_msg[:120]}")
                return False
            else:
                print(f"    [ERROR] {error_msg[:120]}")
                return False

    print(f"    [GIVING UP] Still rate-limited after {max_retries} retries")
    return False


# ─── Rate limiting ─────────────────────────────────────────────

class RateLimiter:
    """Simple rate limiter to stay within API quotas."""

    def __init__(self, min_interval_seconds=1.5):
        self.min_interval = min_interval_seconds
        self.last_call = 0

    def wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


# ─── Main generation loop ─────────────────────────────────────

def run_generation(platform, prompts, n_per_prompt, output_dir, start_index, dry_run):
    """Run image generation for a single platform."""

    # Set up client
    if platform == "openai":
        client = None if dry_run else get_openai_client()
        generate_fn = generate_openai
        rate_limiter = RateLimiter(min_interval_seconds=2.0)  # DALL-E rate limits
    elif platform == "google":
        client = None if dry_run else get_google_client()
        generate_fn = generate_google
        rate_limiter = RateLimiter(min_interval_seconds=6.0)  # Imagen quota: ~10 RPM on Tier 1
    else:
        raise ValueError(f"Unknown platform: {platform}")

    # Track results
    metadata_path = Path(output_dir) / platform / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {"platform": platform, "generated_at": [], "images": []}

    existing_paths = {img["path"] for img in metadata["images"]}
    total = len(prompts) * n_per_prompt
    generated = 0
    skipped = 0
    failed = 0

    print(f"\n{'='*60}")
    print(f"  Generating images: {platform.upper()}")
    print(f"  {len(prompts)} prompts x {n_per_prompt} images = {total} total")
    if dry_run:
        print(f"  [DRY RUN — no API calls will be made]")
    print(f"{'='*60}\n")

    for p_idx, prompt_info in enumerate(prompts):
        prompt_id = prompt_info["id"]
        prompt_text = prompt_info["prompt"]
        category = prompt_info["category"]

        # Create output directory
        cat_dir = Path(output_dir) / platform / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{p_idx+1}/{len(prompts)}] {prompt_id}: \"{prompt_text}\"")

        for i in range(start_index, n_per_prompt):
            filename = f"{prompt_id}_{i:03d}.png"
            filepath = cat_dir / filename
            rel_path = f"{platform}/{category}/{filename}"

            # Skip if already exists
            if rel_path in existing_paths or filepath.exists():
                skipped += 1
                continue

            if dry_run:
                print(f"  Would generate: {rel_path}")
                generated += 1
                continue

            # Generate
            rate_limiter.wait()
            success = generate_fn(client, prompt_text, filepath)

            if success:
                generated += 1
                metadata["images"].append({
                    "path": rel_path,
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "category": category,
                    "index": i,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                print(f"  [{i+1}/{n_per_prompt}] saved")
            else:
                failed += 1

            # Save metadata periodically (every 10 images)
            if generated % 10 == 0 and not dry_run:
                metadata["generated_at"].append(datetime.now(timezone.utc).isoformat())
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

    # Final metadata save
    if not dry_run:
        metadata["generated_at"].append(datetime.now(timezone.utc).isoformat())
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"\nDone ({platform}): {generated} generated, {skipped} skipped, {failed} failed")
    return generated, skipped, failed


# ─── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate images for MAP/SLOP empirical validation"
    )
    parser.add_argument(
        "--platform",
        choices=["openai", "google", "all"],
        default="all",
        help="Which API to use (default: all)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=str(Path(__file__).parent / "prompts.json"),
        help="Path to prompts.json",
    )
    parser.add_argument(
        "--n-per-prompt",
        type=int,
        default=20,
        help="Number of images per prompt per model (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "images"),
        help="Output directory for images",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Resume from this image index (default: 0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without calling APIs",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip the confirmation prompt (useful for unattended runs)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only generate for prompt IDs matching this substring "
             "(e.g., --filter ci_american for a quick test)",
    )

    args = parser.parse_args()

    # Load prompts
    with open(args.prompts) as f:
        prompt_data = json.load(f)
    prompts = prompt_data["prompts"]

    # Apply filter if specified
    if args.filter:
        prompts = [p for p in prompts if args.filter in p["id"]]
        print(f"Filter '{args.filter}' matched {len(prompts)} prompts")
        if not prompts:
            print("No prompts matched the filter. Exiting.")
            return

    print(f"Loaded {len(prompts)} prompts from {args.prompts}")
    print(f"Output directory: {args.output_dir}")
    print(f"Images per prompt per model: {args.n_per_prompt}")

    # Estimate costs
    n_total_per_platform = len(prompts) * args.n_per_prompt
    if args.platform in ("openai", "all"):
        cost_openai = n_total_per_platform * 0.04
        print(f"Estimated OpenAI cost: ${cost_openai:.2f} ({n_total_per_platform} images)")
    if args.platform in ("google", "all"):
        cost_google = n_total_per_platform * 0.02
        print(f"Estimated Google cost: ${cost_google:.2f} ({n_total_per_platform} images, Imagen 4 Fast)")

    if not args.dry_run and not args.yes:
        response = input("\nProceed? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            return

    # Run generation
    platforms = ["openai", "google"] if args.platform == "all" else [args.platform]
    for platform in platforms:
        run_generation(
            platform=platform,
            prompts=prompts,
            n_per_prompt=args.n_per_prompt,
            output_dir=args.output_dir,
            start_index=args.start_index,
            dry_run=args.dry_run,
        )

    print("\n" + "=" * 60)
    print("Generation complete!")
    print(f"Images saved to: {args.output_dir}/")
    print(f"Next step: python run_empirical_analysis.py --image-dir {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
