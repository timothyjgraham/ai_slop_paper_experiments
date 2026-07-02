#!/bin/bash
# ----------------------------------------------------------------------------
# Finalize the public repository on your Mac (where all iCloud files are local).
#
# This repo is scoped to the CCR MEASUREMENT PAPER ONLY:
#   "Measuring Aesthetic Homogenization in Text-to-Image AI:
#    Diversity Collapse in DALL-E 3 and Imagen 4"
# It must contain nothing from the original combined paper beyond what the
# measurement paper uses directly (code, embeddings, figures).
#
# What it does:
#   1. Forces iCloud to download any evicted files, then verifies that no
#      0-byte placeholders remain and that no legacy files are present.
#   2. Initialises a fresh git repository and makes the first commit.
#   3. Prints the commands to push to GitHub and mint a Zenodo DOI.
#
# Usage:  double-click this file, OR in Terminal:
#   cd "ai_slop_paper_experiments" && bash FINALIZE_REPO.command
# ----------------------------------------------------------------------------
set -e
cd "$(dirname "$0")"
DST="$(pwd)"

echo "==> Staging folder: $DST"

# 1a. Force-download anything iCloud has evicted, so files are local
command -v brctl >/dev/null 2>&1 && brctl download "$DST" 2>/dev/null || true
sleep 2

# 1b. Safety: never include the raw image corpus or OS cruft
rm -rf experiments/empirical/images 2>/dev/null || true
find "$DST" -name .DS_Store -delete 2>/dev/null || true
find "$DST" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# 1c. Guard: no legacy files from the original combined paper
LEGACY=$(find "$DST" -not -path "*/.git/*" \( -name "sn-article*" -o -name "sn-jnl.cls" -o -name "sn-basic.bst" -o -name "sn-bibliography*" \) 2>/dev/null || true)
if [ -n "$LEGACY" ]; then
  echo "   ERROR: legacy original-paper files found — remove before publishing:"
  echo "$LEGACY" | sed 's/^/     /'
  exit 1
fi

# 1d. Guard: required files for the measurement paper
for f in paper/ccr-measurement.tex paper/ccr-measurement.pdf paper/refs.bib \
         experiments/run_experiments.py \
         experiments/empirical/run_empirical_analysis.py \
         experiments/empirical/prompts.json \
         experiments/empirical/embeddings/embeddings.npz \
         experiments/empirical/embeddings/embeddings_dinov2.npz \
         experiments/empirical/ablation/ablation_embeddings.npz \
         experiments/empirical/ablation/ablation_index.json; do
  [ -s "$f" ] || { echo "   ERROR: missing or empty: $f"; exit 1; }
done

# 1e. Guard: no 0-byte placeholders anywhere
if find "$DST" -type f -empty -not -path "*/.git/*" | grep -q .; then
  echo "   ERROR: empty placeholder files remain (iCloud not fully downloaded?):"
  find "$DST" -type f -empty -not -path "*/.git/*" | sed 's/^/     STILL EMPTY: /'
  exit 1
fi
echo "==> All checks passed."

# 2. Fresh git history
echo "==> Initialising git..."
rm -rf .git
git init -b main
git add -A
git -c user.name="Timothy Graham" -c user.email="Timothy.Graham@qut.edu.au" \
    commit -m "Measuring Aesthetic Homogenization in Text-to-Image AI: code, embeddings, and paper"

echo ""
echo "==> Done. Local commit created. Tracked files:"
git ls-files | sed 's/^/   /'
echo ""
echo "==> Next steps to publish:"
echo "   The GitHub repo already exists (currently holds the OLD synthetic-only version),"
echo "   so push your updated content over it with --force:"
echo "        git remote add origin https://github.com/timothyjgraham/ai_slop_paper_experiments.git"
echo "        # if that says 'remote origin already exists', run instead:"
echo "        #   git remote set-url origin https://github.com/timothyjgraham/ai_slop_paper_experiments.git"
echo "        git push -u origin main --force"
echo "   (Safe: the repo has 0 stars/forks and no releases — nothing depends on the old history.)"
echo ""
echo "   Then on GitHub, update the repo Description to match the measurement paper, e.g.:"
echo "     'Code, embeddings, and figures for: Measuring Aesthetic Homogenization in"
echo "      Text-to-Image AI (DALL-E 3, Imagen 4, SDXL ablation)'"
echo "   Optional: rename the repo (Settings > General) if you want the name to match"
echo "   the measurement paper; GitHub redirects the old URL automatically."
echo ""
echo "   For a citable DOI: in Zenodo (zenodo.org), enable the repo under Settings > GitHub,"
echo "   then cut a Release on GitHub. Zenodo mints a DOI - confirm it matches the one in"
echo "   CITATION.cff (or update CITATION.cff and the paper's Data Availability statement)."
echo ""
echo "   (You can delete the old ../github-repo folder once you're happy.)"
