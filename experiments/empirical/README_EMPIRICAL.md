# Empirical Validation of MAP/SLOP on Commercial Models

This directory contains the experiment pipeline for testing the Mode Averaging Principle (MAP) and SLOP predictions on real commercial image generation models (OpenAI DALL-E 3 and Google Imagen 4).

## Overview

The experiment generates 1,200 images (30 prompts × 20 images × 2 models), embeds them with CLIP, and computes the paper's homogenisation metrics (ρ, δ, γ) to test three predictions:

1. **Within-prompt diversity is low** — the model converges toward the conditional mean each time
2. **Minority cultures show less diversity** — stronger pull toward the global aesthetic mean
3. **Different models agree** — both converge toward similar statistical centres

---

## Step 1: Get API Keys

### OpenAI (for DALL-E 3)

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Sign in (or create an account)
3. Click "Create new secret key" — give it a name like "SLOP experiment"
4. Copy the key (starts with `sk-`)
5. **Billing**: You need a funded account. Add a payment method at [platform.openai.com/settings/organization/billing](https://platform.openai.com/settings/organization/billing). Set a usage limit (e.g., $60) to avoid surprises.
6. Estimated cost: **~$48** for 1,200 images at $0.04/image (1024×1024, standard quality)

### Google Gemini (for Imagen 4)

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click "Create API key" — select or create a Google Cloud project
4. Copy the key (starts with `AIza`)
5. **Cost**: Imagen 4 Fast costs ~$0.02/image (~$24 for 1,200 images). The full Imagen 4 model is significantly more expensive (~$0.27/image).

---

## Step 2: Set Up Environment

### Option A: Google Colab (recommended)

Open a new Colab notebook and run:

```python
# Install dependencies
!pip install openai google-genai torch torchvision transformers Pillow tqdm scipy matplotlib

# Set API keys
import os
os.environ["OPENAI_API_KEY"] = "sk-..."   # paste your key
os.environ["GOOGLE_API_KEY"] = "AIza..."   # paste your key

# Upload the experiment files (or clone from GitHub)
# Option 1: Upload directly
from google.colab import files
# Then upload: prompts.json, generate_images.py, run_empirical_analysis.py

# Option 2: Clone from GitHub
!git clone https://github.com/timothyjgraham/ai_slop_paper_experiments.git
%cd ai_slop_paper_experiments/experiments/empirical
```

### Option B: Local machine

```bash
# Create a virtual environment
python -m venv slop_env
source slop_env/bin/activate  # or slop_env\Scripts\activate on Windows

# Install dependencies
pip install openai google-genai torch torchvision transformers Pillow tqdm scipy matplotlib

# Set API keys
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AIza..."
```

---

## Step 3: Generate Images

### Test run first (2 images per prompt — costs ~$5)

```bash
python generate_images.py --platform all --n-per-prompt 2
```

This generates 120 images (30 × 2 × 2) to verify everything works.

### Full run (20 images per prompt)

```bash
python generate_images.py --platform all --n-per-prompt 20
```

The script will:
- Show you the estimated cost and ask for confirmation
- Generate images with rate limiting to stay within API quotas
- Save images to `images/{openai,google}/{category}/{prompt_id}_{index}.png`
- Save metadata to `images/{platform}/metadata.json`
- Support resuming — if interrupted, re-run and it skips existing images

### If you prefer to generate manually

Generate images through the ChatGPT and Gemini web interfaces, then save them with this naming convention:

```
images/
├── openai/
│   ├── cultural_identity/
│   │   ├── ci_american_000.png
│   │   ├── ci_american_001.png
│   │   ├── ...
│   │   └── ci_kurdish_019.png
│   ├── open_ended/
│   │   ├── oe_landscape_000.png
│   │   └── ...
│   └── artistic_style/
│       ├── as_impressionism_000.png
│       └── ...
└── google/
    └── [same structure]
```

The analysis script will discover images from the folder structure.

---

## Step 4: Run Analysis

```bash
# Full pipeline (embed + analyse + plot)
python run_empirical_analysis.py --image-dir ./images --prompts ./prompts.json

# Or in two stages:
# Stage 1: Just embed (saves embeddings.npz — slow, needs GPU)
python run_empirical_analysis.py --image-dir ./images --embeddings-only

# Stage 2: Just analyse (fast, from saved embeddings)
python run_empirical_analysis.py --analysis-only
```

### On Google Colab

Make sure you're using a GPU runtime (Runtime → Change runtime type → T4 GPU).
CLIP embedding of 1,200 images takes about 2-3 minutes on a T4.

---

## Step 5: Interpret Results

The analysis produces:

### Figures

| File | What it shows |
|------|---------------|
| `analysis1_within_prompt_diversity.png` | Bar charts of ρ, D, d_eff by category and model |
| `analysis2_cultural_convergence.png` | Within-prompt diversity by nationality (majority vs minority) |
| `analysis2_cultural_heatmap_{platform}.png` | Pairwise cosine distance between nationality centroids |
| `analysis3_cross_model_agreement.png` | Per-prompt cosine similarity between model centroids |

### Summary JSON

`empirical_summary.json` contains all metrics and prediction confirmations.

### What to look for

**If MAP is correct, you should see:**

1. **Low within-prompt diversity** — R_α values well below what you'd see for random images
2. **Minority nationalities have lower R_α** — their outputs are more tightly clustered (stronger pull to global mean)
3. **The cultural heatmap shows compressed distances** — nationality centroids are closer together than real photographs would be
4. **High cross-model similarity** — OpenAI and Google produce similar centroids for the same prompts (>0.5 cosine similarity)

---

## File Structure

```
empirical/
├── README_EMPIRICAL.md         ← this file
├── prompts.json                ← 30 structured prompts
├── generate_images.py          ← API-based image generation
├── run_empirical_analysis.py   ← CLIP embedding + metrics + figures
├── images/                     ← generated images (not in git)
│   ├── openai/{category}/
│   └── google/{category}/
├── embeddings/                 ← CLIP embeddings (not in git)
│   └── embeddings.npz
└── figures/                    ← output plots
    ├── analysis1_*.png
    ├── analysis2_*.png
    ├── analysis3_*.png
    └── empirical_summary.json
```

## Dependencies

```
openai>=1.0
google-genai>=1.0
torch>=2.0
torchvision>=0.15
transformers>=4.30
Pillow>=9.0
numpy>=1.21
scipy>=1.7
matplotlib>=3.5
tqdm>=4.60
```
