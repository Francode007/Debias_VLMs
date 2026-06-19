"""Phase 0.8 — Investigate the s1 vwarmup KL=0.044 anomaly.

Context: across the 4 vwarmup PPO runs, the tail KL of s1 was reported at
0.044, vs ~0.0007-0.001 for s2/s3/s4 (~50× outlier). s1 was also the BEST
vwarmup seed at downstream canonical SB-Bench (+0.21 pp). We want to:

  1. Confirm the tail KL spike (compute mean KL over the last N steps).
  2. Find WHERE in training s1 diverged (per-step KL trajectory).
  3. Check whether the spike correlates with any other field (reward,
     pg_loss, value_grad_norm, ratio, ambig_preserve_rate, etc.).
  4. Identify candidate steps for deeper inspection.

Inputs:
  Phase0.8/seed_diag/vwarmup/s{1,2,3,4}_vwarmup_metrics.jsonl       (246 PPO steps)
  Phase0.8/seed_diag/vwarmup/s{1,2,3,4}_vwarmup_warmup_metrics.jsonl (30 warmup steps)

Outputs:
  Phase0.8/seed_diag/vwarmup/s1_kl_anomaly_diagnostic.json
  Console table: per-seed KL summary stats + correlation table for s1
"""
import json
import numpy as np
from pathlib import Path

SEEDS = [1, 2, 3, 4]
ROOT = Path("Phase0.8/seed_diag/vwarmup")
OUT = ROOT / "s1_kl_anomaly_diagnostic.json"


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def field(rows, k):
    return np.array([r.get(k, np.nan) for r in rows], dtype=np.float64)


def summary(arr, name):
    return {
        "field": name,
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "median": float(np.nanmedian(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "p95": float(np.nanpercentile(arr, 95)),
        "p99": float(np.nanpercentile(arr, 99)),
    }


def windowed_mean(arr, n=10):
    if len(arr) < n:
        return float(np.nanmean(arr))
    return float(np.nanmean(arr[-n:]))


def per_step_compare(metrics_by_seed, key, n_show=10):
    n = min(len(m) for m in metrics_by_seed.values())
    out = []
    for i in range(n):
        row = {"step": i}
        for s, m in metrics_by_seed.items():
            row[f"s{s}"] = float(m[i].get(key, float("nan")))
        out.append(row)
    return out


# Load metrics
metrics = {s: load(ROOT / f"s{s}_vwarmup_metrics.jsonl") for s in SEEDS}
warmup_metrics = {s: load(ROOT / f"s{s}_vwarmup_warmup_metrics.jsonl") for s in SEEDS}
print(f'PPO step counts: ' + ', '.join(f's{s}={len(m)}' for s,m in metrics.items()))
print(f'Warmup step counts: ' + ', '.join(f's{s}={len(m)}' for s,m in warmup_metrics.items()))

# 1. Per-seed KL summary (full + last 50 + last 20 + last 10)
print("\n=== Per-seed KL summary ===")
print(f'{"seed":>4}  {"mean_all":>9}  {"mean_50":>9}  {"mean_20":>9}  {"mean_10":>9}  {"max":>9}  {"p99":>9}  {"argmax_step":>11}')
kl_summary = {}
for s in SEEDS:
    kl = field(metrics[s], "kl")
    last_50 = windowed_mean(kl, 50)
    last_20 = windowed_mean(kl, 20)
    last_10 = windowed_mean(kl, 10)
    argmax = int(np.nanargmax(kl))
    print(f's{s:>2}    {np.nanmean(kl):>9.5f}  {last_50:>9.5f}  {last_20:>9.5f}  {last_10:>9.5f}  {np.nanmax(kl):>9.5f}  {np.nanpercentile(kl,99):>9.5f}  {argmax:>11d}')
    kl_summary[f"s{s}"] = {
        "mean_all": float(np.nanmean(kl)),
        "mean_last_50": last_50,
        "mean_last_20": last_20,
        "mean_last_10": last_10,
        "median": float(np.nanmedian(kl)),
        "max": float(np.nanmax(kl)),
        "p95": float(np.nanpercentile(kl, 95)),
        "p99": float(np.nanpercentile(kl, 99)),
        "argmax_step": argmax,
    }

# 2. Where does s1 KL first diverge from others?
print("\n=== s1 KL trajectory (relative to median of s2/s3/s4) ===")
kls = {s: field(metrics[s], "kl") for s in SEEDS}
others = np.median(np.stack([kls[2], kls[3], kls[4]]), axis=0)
s1 = kls[1]
ratio = np.where(others > 1e-8, s1 / others, np.nan)
# bucket into windows of 20 steps
print(f'{"step_range":>12}  {"s1_mean":>9}  {"others_med":>11}  {"ratio":>7}')
for i in range(0, len(s1), 20):
    j = min(i + 20, len(s1))
    s1_m = np.nanmean(s1[i:j])
    ot_m = np.nanmedian(others[i:j])
    r = s1_m / ot_m if ot_m > 1e-8 else float('nan')
    print(f'  {i:>3}-{j-1:<3}      {s1_m:>9.5f}  {ot_m:>11.5f}  {r:>7.2f}')

# 3. Top-10 highest-KL steps in s1
print("\n=== s1 top-10 highest-KL steps ===")
top10 = np.argsort(-s1)[:10]
print(f'{"step":>5}  {"kl":>9}  {"reward":>9}  {"pg_loss":>10}  {"v_loss":>10}  {"vgrad":>10}  {"pgrad":>10}  {"ratio":>7}')
top10_records = []
rows = metrics[1]
for idx in sorted(top10):
    r = rows[idx]
    print(f'  {idx:>3}  {r.get("kl",0):>9.5f}  {r.get("reward_task",0):>+9.5f}  {r.get("pg_loss",0):>+10.5f}  {r.get("v_loss",0):>10.4f}  {r.get("value_grad_norm",0):>10.3f}  {r.get("policy_grad_norm",0):>10.3f}  {r.get("ratio_mean",0):>7.4f}')
    top10_records.append({"step": int(idx), **{k: float(r[k]) for k in ("kl","reward_task","pg_loss","v_loss","value_grad_norm","policy_grad_norm","ratio_mean","mean_abs_logprob_diff")}})

# 4. Correlation analysis (in s1)
print("\n=== s1 — Pearson(KL, X) ===")
def pcorr(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return float('nan')
    a, b = a[m], b[m]
    return float(np.corrcoef(a, b)[0, 1])

corr_fields = ["reward_task", "reward_dense_mean", "reward_bias_aligned_mean",
               "pg_loss", "v_loss", "value_grad_norm", "policy_grad_norm",
               "ratio_mean", "adv_abs_mean", "mean_abs_logprob_diff",
               "binary_accuracy", "ambig_preserve_rate", "parse_success_rate",
               "kl_beta", "loss", "reward_task_std"]
s1_kl = field(metrics[1], "kl")
s1_corrs = {}
for k in corr_fields:
    c = pcorr(s1_kl, field(metrics[1], k))
    s1_corrs[k] = c
for k, c in sorted(s1_corrs.items(), key=lambda x: -abs(x[1])):
    print(f'  {k:<28}  {c:>+.4f}')

# 5. Check kl_beta trajectory (adaptive controller)
print("\n=== kl_beta trajectory tail (last 10 steps) ===")
for s in SEEDS:
    last10_beta = field(metrics[s], "kl_beta")[-10:]
    print(f's{s}: kl_beta tail = {last10_beta}')

# 6. ratio_mean (off-policy drift) tail
print("\n=== ratio_mean tail (last 10 steps) ===")
for s in SEEDS:
    last10_r = field(metrics[s], "ratio_mean")[-10:]
    print(f's{s}: ratio tail = {[round(x,4) for x in last10_r.tolist()]}')

# 7. value_grad_norm trajectory (was the value head exploding?)
print("\n=== value_grad_norm summary ===")
for s in SEEDS:
    vg = field(metrics[s], "value_grad_norm")
    print(f's{s}: mean={np.nanmean(vg):.3f}  median={np.nanmedian(vg):.3f}  max={np.nanmax(vg):.3f}  p99={np.nanpercentile(vg,99):.3f}  argmax_step={int(np.nanargmax(vg))}')

# 8. Save JSON
out = {
    "n_ppo_steps_per_seed": {f"s{s}": len(metrics[s]) for s in SEEDS},
    "n_warmup_steps_per_seed": {f"s{s}": len(warmup_metrics[s]) for s in SEEDS},
    "kl_summary_per_seed": kl_summary,
    "s1_vs_others_kl_ratio_by_20step_window": [
        {"step_range": f"{i}-{min(i+20,len(s1))-1}",
         "s1_mean_kl": float(np.nanmean(s1[i:min(i+20,len(s1))])),
         "others_median_kl": float(np.nanmedian(others[i:min(i+20,len(s1))])),
         "ratio": float(np.nanmean(s1[i:min(i+20,len(s1))]) / np.nanmedian(others[i:min(i+20,len(s1))])) if np.nanmedian(others[i:min(i+20,len(s1))]) > 1e-8 else None}
        for i in range(0, len(s1), 20)
    ],
    "s1_top10_high_kl_steps": top10_records,
    "s1_pearson_kl_vs_field": dict(sorted(s1_corrs.items(), key=lambda x: -abs(x[1]))),
    "kl_beta_tail_last10_per_seed": {f"s{s}": field(metrics[s], "kl_beta")[-10:].tolist() for s in SEEDS},
    "ratio_mean_tail_last10_per_seed": {f"s{s}": field(metrics[s], "ratio_mean")[-10:].tolist() for s in SEEDS},
    "value_grad_norm_summary": {
        f"s{s}": {"mean": float(np.nanmean(field(metrics[s], "value_grad_norm"))),
                  "median": float(np.nanmedian(field(metrics[s], "value_grad_norm"))),
                  "max": float(np.nanmax(field(metrics[s], "value_grad_norm"))),
                  "p99": float(np.nanpercentile(field(metrics[s], "value_grad_norm"), 99)),
                  "argmax_step": int(np.nanargmax(field(metrics[s], "value_grad_norm")))}
        for s in SEEDS
    },
}
OUT.write_text(json.dumps(out, indent=2))
print(f'\nWrote {OUT}')
