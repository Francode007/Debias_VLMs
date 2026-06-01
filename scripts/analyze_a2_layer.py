"""
Phase 0.8 A2 — reusable per-layer analysis script.

Mirrors the L9 verdict pipeline:
  1. PCA spectrum (rank-1 dominance, cum-var to thresholds)
  2. Kept-PCA accuracy (vs misleading 100-head mean)
  3. SVM-vs-SVM collapse + PC-vs-SVM alignment
  4. Per-cell reward gap (raw + sigma-standardised) per variant, per head_type
  5. Sign-correctness across 9 SVM heads on ambig/neg/non_neg
  6. Pearson correlation of per-head gap vector vs L35 baseline
  7. Final cross-layer summary line you can paste back into chat

Usage:
  python scripts/analyze_a2_layer.py --layer 11
  python scripts/analyze_a2_layer.py --layer 13 --variants base svm_ep1-end
  python scripts/analyze_a2_layer.py --layer 9   # re-validate L9

All paths are resolved relative to the repo root (parent of this script's dir).
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VARIANTS = ["base", "svm_ep1-50pct", "svm_ep1-end", "pca_ep1-step80", "pca_ep1-end"]
SB_CATS = ["Age", "Disab", "Gender", "Natnl", "PhysAp", "Race", "Relig", "SES", "SexOr"]

# Expected sign of (correct − incorrect) reward gap per condition.
# Heads should reward CORRECT answers more on ambig (where C=I don't know is correct),
# and reward correct answers more on neg / non_neg too (correct demographic answer).
# At L35 baseline: SVM gives positive gap on ambig, neg, non_neg.
EXPECTED_SIGN = {"ambig": +1, "neg": +1, "non_neg": +1}


def _path_head_dir(layer: int) -> Path:
    return REPO_ROOT / "Phase0.8" / "a2_results" / f"generated_heads_letter_post_letter_L{layer}"


def _path_reward_dir(layer: int) -> Path:
    return REPO_ROOT / "Phase0.8" / "a2_results" / f"phase08_offline_reward_L{layer}"


def _path_l35_reward_dir() -> Path:
    return REPO_ROOT / "Phase0.7" / "offline_reward" / "phase07_offline_reward"


def section(title: str) -> None:
    print()
    print("=" * 88)
    print(f"  {title}")
    print("=" * 88)


def analyze_heads(layer: int) -> dict:
    """Sections 1–3: spectrum, kept-PCA acc, SVM collapse + PC alignment."""
    head_dir = _path_head_dir(layer)
    out: dict = {"layer": layer, "head_dir_exists": head_dir.exists()}
    if not head_dir.exists():
        section(f"Head-extraction artifacts (L{layer})")
        print(f"  [SKIP] {head_dir} does not exist")
        return out

    # ---- spectrum ----
    section(f"1. PCA spectrum (L{layer})")
    ev = np.load(head_dir / "explained_variance.npy")
    evr = np.load(head_dir / "explained_variance_ratio.npy")
    print(f"  n_components on disk: {len(ev)}")
    print(f"  PC0..PC4 ratios     : {np.round(evr[:5], 4).tolist()}")
    cum = np.cumsum(evr)
    for t in (0.5, 0.8, 0.95, 0.99):
        idx = int(np.argmax(cum >= t)) if cum[-1] >= t else len(cum) - 1
        print(f"    cumvar >= {t*100:>4.0f}% at PC{idx:<3d} (cum={cum[idx]:.3f})")
    rank1 = float(evr[0] / max(evr[1], 1e-12))
    print(f"  Top-1 / Top-2 ratio = {rank1:.2f}  (>>1 = rank-1 dominated)")
    out["pc0_var"] = float(evr[0])
    out["pc1_var"] = float(evr[1])
    out["rank1_ratio"] = rank1
    out["cumvar_to_pc1"] = float(cum[1])

    # ---- kept-PCA accuracy ----
    section(f"2. Kept-PCA accuracy (L{layer})")
    eval_path = head_dir / "drm_head_eval_pca_test.json"
    kept_path = head_dir / "kept_heads_pca.json"
    if eval_path.exists() and kept_path.exists():
        eval_j = json.loads(eval_path.read_text())
        kept_j = json.loads(kept_path.read_text())
        all_acc = np.array(eval_j["overall_per_head"])
        idxs = kept_j["kept_indices"]
        print(f"  All-{len(all_acc)}-heads mean: {eval_j['overall_mean']:.4f}")
        if len(idxs):
            print(f"  Kept ({len(idxs)} heads, thr={kept_j['threshold']}) mean: {all_acc[idxs].mean():.4f}")
            print(f"  Kept min/max         : {all_acc[idxs].min():.4f} / {all_acc[idxs].max():.4f}")
        else:
            print(f"  [WARN] No heads passed threshold {kept_j['threshold']}")
        out["kept_count"] = len(idxs)
        out["kept_mean_acc"] = float(all_acc[idxs].mean()) if idxs else 0.0
    else:
        print(f"  [SKIP] missing {eval_path.name} or {kept_path.name}")

    # ---- SVM collapse + PC alignment ----
    section(f"3. SVM coherence + PC-vs-SVM alignment (L{layer})")
    pcs_path = head_dir / "orthogonal_heads.npy"
    svm_path = head_dir / "svm_normals.npy"
    if pcs_path.exists() and svm_path.exists():
        pcs = np.load(pcs_path)
        svm = np.load(svm_path)

        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

        # SVM pairwise (collapse metric)
        sims = [abs(cos(svm[i], svm[j])) for i in range(len(svm)) for j in range(i + 1, len(svm))]
        print(f"  SVM-vs-SVM pairwise |cos|: mean={np.mean(sims):.3f}  min={min(sims):.3f}  max={max(sims):.3f}")
        out["svm_pairwise_meancos"] = float(np.mean(sims))

        # PCs vs SVMs: best max |cos| across all PCs and SVMs
        k = min(5, pcs.shape[0])
        best = 0.0
        for i in range(pcs.shape[0]):
            for j in range(svm.shape[0]):
                c = abs(cos(pcs[i], svm[j]))
                if c > best:
                    best = c; best_i = i; best_j = j
        print(f"  max |cos(PC_i, SVM_j)| across all = {best:.3f}  (PC{best_i} ↔ {SB_CATS[best_j]})")
        # Top-5 PC alignment one-liner
        for i in range(k):
            row = [abs(cos(pcs[i], svm[j])) for j in range(svm.shape[0])]
            j = int(np.argmax(row))
            print(f"    PC{i} (ev={evr[i]*100:5.2f}%): max|cos|={row[j]:.3f} on {SB_CATS[j]}")
        out["max_pc_svm_cos"] = float(best)
    else:
        print(f"  [SKIP] orthogonal_heads.npy or svm_normals.npy missing")
    return out


def _gap_and_sigma(cell_correct: dict, cell_incorrect: dict) -> tuple[float, float, np.ndarray]:
    """Return (raw_gap, sigma_standardised_gap, per_head_gap_vector)."""
    if not cell_correct or not cell_incorrect:
        return float("nan"), float("nan"), np.array([])
    mc = np.array(cell_correct["mean_per_head"])
    mi = np.array(cell_incorrect["mean_per_head"])
    sc = np.array(cell_correct.get("std_per_head", [1.0] * len(mc)))
    si = np.array(cell_incorrect.get("std_per_head", [1.0] * len(mi)))
    nc, ni = cell_correct.get("n", 1), cell_incorrect.get("n", 1)
    pooled = np.sqrt((sc ** 2 + si ** 2) / 2.0 + 1e-12)
    per_head_gap = mc - mi
    raw = float(per_head_gap.mean())
    sigma = float((per_head_gap / pooled).mean())
    return raw, sigma, per_head_gap


def analyze_rewards(layer: int, variants: list[str]) -> dict:
    """Sections 4–6: per-variant cell gaps, signs, cross-layer correlation."""
    rdir = _path_reward_dir(layer)
    l35_dir = _path_l35_reward_dir()

    section(f"4. Per-variant offline-reward cell gaps (L{layer})")
    rows = []
    for variant in variants:
        for head_type in ("svm", "pca"):
            fn = rdir / f"{variant}__{head_type}_L{layer}__offline_reward.json"
            if not fn.exists():
                print(f"  [SKIP] {fn.name}")
                continue
            j = json.loads(fn.read_text())
            by = j.get("by_cell", {})
            row = {"variant": variant, "head": head_type}
            for cond in ("ambig", "neg", "non_neg"):
                c = by.get(f"{cond}::correct", {})
                i = by.get(f"{cond}::incorrect", {})
                raw, sigma, vec = _gap_and_sigma(c, i)
                row[f"{cond}_raw"] = raw
                row[f"{cond}_sigma"] = sigma
                row[f"{cond}_vec"] = vec
                # sign correctness over per-head vector
                if vec.size and head_type == "svm":
                    correct_signs = int((np.sign(vec) == EXPECTED_SIGN[cond]).sum())
                    row[f"{cond}_sign_ok"] = f"{correct_signs}/{len(vec)}"
            rows.append(row)

    # Pretty print
    header = f"  {'variant':<18s} {'head':<4s} | {'ambig_raw':>10s} {'ambig_σ':>9s} {'sign':>6s} | {'neg_raw':>9s} {'neg_σ':>9s} {'sign':>6s} | {'nneg_raw':>9s} {'nneg_σ':>9s} {'sign':>6s}"
    print(header)
    print(f"  {'-'*len(header)}")
    for r in rows:
        sign_a = r.get("ambig_sign_ok", "-")
        sign_n = r.get("neg_sign_ok", "-")
        sign_nn = r.get("non_neg_sign_ok", "-")
        print(f"  {r['variant']:<18s} {r['head']:<4s} | "
              f"{r['ambig_raw']:>10.3f} {r['ambig_sigma']:>+9.3f} {sign_a:>6s} | "
              f"{r['neg_raw']:>9.3f} {r['neg_sigma']:>+9.3f} {sign_n:>6s} | "
              f"{r['non_neg_raw']:>9.3f} {r['non_neg_sigma']:>+9.3f} {sign_nn:>6s}")

    # Section 5+6: per-head vector correlation with L35 baseline
    section(f"5. Cross-layer per-head correlation: L{layer} vs L35 baseline")
    corrs = []
    for variant in variants:
        for head_type in ("svm", "pca"):
            f_new = rdir / f"{variant}__{head_type}_L{layer}__offline_reward.json"
            f_l35 = l35_dir / f"{variant}__{head_type}__offline_reward.json"
            if not f_new.exists() or not f_l35.exists():
                continue
            jn, j35 = json.loads(f_new.read_text()), json.loads(f_l35.read_text())
            for cond in ("ambig", "neg", "non_neg"):
                a = _gap_and_sigma(jn["by_cell"].get(f"{cond}::correct", {}),
                                   jn["by_cell"].get(f"{cond}::incorrect", {}))[2]
                b = _gap_and_sigma(j35["by_cell"].get(f"{cond}::correct", {}),
                                   j35["by_cell"].get(f"{cond}::incorrect", {}))[2]
                if a.size and b.size and a.size == b.size:
                    r = float(np.corrcoef(a, b)[0, 1])
                    corrs.append({"variant": variant, "head": head_type, "cond": cond, "pearson": r,
                                  "n_heads": len(a)})

    if corrs:
        print(f"  {'variant':<18s} {'head':<4s} {'cond':<7s} {'n_heads':>7s} {'pearson':>9s}")
        for c in corrs:
            print(f"  {c['variant']:<18s} {c['head']:<4s} {c['cond']:<7s} {c['n_heads']:>7d} {c['pearson']:>+9.3f}")
    else:
        print(f"  [no L35 baselines paired]")

    return {"rows": rows, "cross_layer_corrs": corrs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True, help="Layer index, e.g. 9, 11, 13")
    p.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    args = p.parse_args()

    print(f"\n>>> Phase 0.8 A2 analysis for L{args.layer}")
    print(f"    variants: {args.variants}")
    head_summary = analyze_heads(args.layer)
    reward_summary = analyze_rewards(args.layer, args.variants)

    # Final one-liner
    section(f"FINAL SUMMARY — L{args.layer}")
    if head_summary.get("head_dir_exists"):
        print(f"  PCA: kept={head_summary.get('kept_count','?')} | "
              f"kept_mean_acc={head_summary.get('kept_mean_acc',0):.3f} | "
              f"PC0_var={head_summary.get('pc0_var',0)*100:.1f}% | "
              f"rank1_ratio={head_summary.get('rank1_ratio',0):.1f}")
        print(f"  SVM: pairwise_meancos={head_summary.get('svm_pairwise_meancos',0):.3f}  "
              f"(L9=0.60 baseline; lower is better differentiation)")
        print(f"  PC↔SVM max|cos|: {head_summary.get('max_pc_svm_cos',0):.3f}  "
              f"(L9=0.09 = bias buried; >0.3 = PC carries bias)")


if __name__ == "__main__":
    sys.exit(main())
