#!/usr/bin/env python3
"""
Fast Bootstrap CI Analysis — addresses reviewer concern about n=20 sample size.
Uses Gram matrix trick (n×n instead of d×d) for efficient eigendecomposition.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ALPHA = 0.90

def effective_support_radius(samples, alpha=ALPHA):
    mu = samples.mean(axis=0)
    dists = np.linalg.norm(samples - mu, axis=1)
    return float(np.percentile(dists, alpha * 100))

def effective_dimension_fast(samples):
    """Compute d_eff using the Gram matrix trick: O(n^2*d + n^3) instead of O(d^3).
    Since n=20 << d=768, this is ~15,000x faster."""
    n, d = samples.shape
    if n < 2:
        return 0.0
    centered = samples - samples.mean(axis=0)
    # Gram matrix: G = X X^T / (n-1), eigenvalues match those of X^T X / (n-1)
    G = centered @ centered.T / (n - 1)
    eigenvalues = np.linalg.eigvalsh(G)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if len(eigenvalues) == 0:
        return 0.0
    total = eigenvalues.sum()
    return float(total ** 2 / (eigenvalues ** 2).sum())

def main():
    emb_path = Path(__file__).parent / "embeddings" / "embeddings.npz"
    data = np.load(emb_path, allow_pickle=True)
    embeddings = data["embeddings"]
    metadata = json.loads(str(data["metadata"]))
    print(f"Loaded {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

    prompts_path = Path(__file__).parent / "prompts.json"
    with open(prompts_path) as f:
        prompt_lookup = {p["id"]: p for p in json.load(f)["prompts"]}

    groups = defaultdict(list)
    for i, rec in enumerate(metadata):
        groups[(rec["platform"], rec["prompt_id"])].append(i)

    N_BOOT = 500
    rng = np.random.RandomState(42)

    print(f"\n{'Platform':<10} {'Prompt':<25} {'Group':<10} {'n':>4} "
          f"{'R_α':>8} {'R_α 95%CI':>20} {'d_eff':>8} {'d_eff 95%CI':>20} {'R_α CV':>8}")
    print("-" * 120)

    all_results = []
    for (platform, prompt_id), indices in sorted(groups.items()):
        embs = embeddings[np.array(indices)]
        n = embs.shape[0]
        pinfo = prompt_lookup.get(prompt_id, {})
        group = pinfo.get("group", "unknown")

        # R_alpha bootstrap
        r_est = effective_support_radius(embs)
        r_boots = [effective_support_radius(embs[rng.choice(n, n, replace=True)]) for _ in range(N_BOOT)]
        r_boots = np.array(r_boots)

        # d_eff bootstrap
        d_est = effective_dimension_fast(embs)
        d_boots = [effective_dimension_fast(embs[rng.choice(n, n, replace=True)]) for _ in range(N_BOOT)]
        d_boots = np.array(d_boots)

        result = {
            "platform": platform, "prompt_id": prompt_id, "group": group, "n": n,
            "R_alpha": {
                "est": r_est,
                "ci_lo": float(np.percentile(r_boots, 2.5)),
                "ci_hi": float(np.percentile(r_boots, 97.5)),
                "se": float(np.std(r_boots)),
                "cv": float(np.std(r_boots) / (abs(r_est) + 1e-12)),
            },
            "d_eff": {
                "est": d_est,
                "ci_lo": float(np.percentile(d_boots, 2.5)),
                "ci_hi": float(np.percentile(d_boots, 97.5)),
                "se": float(np.std(d_boots)),
                "cv": float(np.std(d_boots) / (abs(d_est) + 1e-12)),
            },
        }
        all_results.append(result)

        r = result["R_alpha"]
        d = result["d_eff"]
        print(f"{platform:<10} {prompt_id:<25} {group:<10} {n:>4} "
              f"{r['est']:>8.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] "
              f"{d['est']:>8.2f} [{d['ci_lo']:.2f}, {d['ci_hi']:.2f}] "
              f"{r['cv']:>8.3f}")

    # ── Gap analysis ──
    print(f"\n{'='*70}")
    print("  MAJORITY vs MINORITY GAP ROBUSTNESS")
    print(f"{'='*70}")

    ci_results = [r for r in all_results if r["prompt_id"].startswith("ci_")]

    for mname in ["R_alpha", "d_eff"]:
        maj = [r[mname]["est"] for r in ci_results if r["group"] == "majority"]
        mino = [r[mname]["est"] for r in ci_results if r["group"] == "minority"]
        if not maj or not mino:
            continue

        maj_mean, min_mean = np.mean(maj), np.mean(mino)
        gap = maj_mean - min_mean
        gap_pct = (gap / maj_mean) * 100 if maj_mean else 0

        gap_boots = []
        for _ in range(N_BOOT):
            bm = rng.choice(maj, len(maj), replace=True)
            bn = rng.choice(mino, len(mino), replace=True)
            gap_boots.append(np.mean(bm) - np.mean(bn))
        gap_boots = np.array(gap_boots)

        print(f"\n  {mname}:")
        print(f"    Majority mean: {maj_mean:.4f} (n={len(maj)} prompts)")
        print(f"    Minority mean: {min_mean:.4f} (n={len(mino)} prompts)")
        print(f"    Gap: {gap:.4f} ({gap_pct:.1f}%)")
        print(f"    Gap 95% CI: [{np.percentile(gap_boots, 2.5):.4f}, {np.percentile(gap_boots, 97.5):.4f}]")
        print(f"    CI excludes zero: {'YES' if np.percentile(gap_boots, 2.5) > 0 else 'NO'}")
        print(f"    P(gap > 0): {(gap_boots > 0).mean():.3f}")

    # ── Subsample stability ──
    print(f"\n{'='*70}")
    print("  SUBSAMPLE STABILITY")
    print(f"{'='*70}")

    for test_pid in ["ci_american", "ci_bangladeshi"]:
        key = ("openai", test_pid)
        if key not in groups:
            continue
        embs = embeddings[np.array(groups[key])]
        n_full = embs.shape[0]
        print(f"\n  {test_pid} (n={n_full}):")

        for mname, mfn in [("R_alpha", effective_support_radius), ("d_eff", effective_dimension_fast)]:
            full_est = mfn(embs)
            print(f"    {mname} at n={n_full}: {full_est:.4f}")
            for ss in [5, 8, 10, 15, 18]:
                if ss > n_full:
                    continue
                vals = [mfn(embs[rng.choice(n_full, ss, replace=False)]) for _ in range(100)]
                vals = np.array(vals)
                cv = np.std(vals) / (abs(np.mean(vals)) + 1e-12)
                print(f"      n={ss:>3}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, CV={cv:.3f}")

    # Save
    figures_dir = Path(__file__).parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    with open(figures_dir / "bootstrap_ci_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to figures/bootstrap_ci_results.json")

if __name__ == "__main__":
    main()
