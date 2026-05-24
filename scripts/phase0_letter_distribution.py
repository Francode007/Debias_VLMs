"""
Phase 0 diagnostic: letter-distribution analysis across checkpoints.

Can run locally on downloaded JSONLs or as a Modal function against the volume.

Usage (local):
    python scripts/phase0_letter_distribution.py \
        --jsonl_dir ./generations/ \
        --output_json ./letter_dist_analysis.json

Usage (Modal — runs against the persistent volume):
    modal run scripts/phase0_letter_distribution.py::analyse_on_volume \
        --gen-dir /mnt/data/output_ppo_phase0/generations \
        --output-json /mnt/data/output_ppo_phase0/letter_distribution_analysis.json
"""
import argparse
import collections
import glob
import json
import os
import re
import sys


def analyse_jsonl(path: str) -> dict:
    """Compute letter distribution + confusion matrix from a generation JSONL."""
    letters = collections.Counter()
    label_dist = collections.Counter()
    confusion = collections.Counter()  # (pred, true_label) → count
    total = 0
    correct = 0
    per_cat_letters = collections.defaultdict(lambda: collections.Counter())
    per_cat_correct = collections.defaultdict(int)
    per_cat_total = collections.defaultdict(int)

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pred = (rec.get("text") or "").strip().upper()[:1]
            label = rec.get("label")
            cat = rec.get("category", "unknown")
            if not pred or label is None:
                continue

            label_letter = chr(65 + int(label)) if 0 <= int(label) <= 2 else "?"
            letters[pred] += 1
            label_dist[label_letter] += 1
            confusion[(pred, label_letter)] += 1
            per_cat_letters[cat][pred] += 1
            per_cat_total[cat] += 1
            total += 1
            if pred == label_letter:
                correct += 1
                per_cat_correct[cat] += 1

    if total == 0:
        return {"error": "no valid records", "path": path}

    # Compute entropy of predicted distribution (max=log2(3)≈1.585 for uniform)
    import math
    entropy = 0.0
    for cnt in letters.values():
        p = cnt / total
        if p > 0:
            entropy -= p * math.log2(p)
    max_entropy = math.log2(3)

    # Mode collapse indicator: is >60% of predictions a single letter?
    dominant_letter = letters.most_common(1)[0]
    mode_collapsed = dominant_letter[1] / total > 0.60

    result = {
        "path": path,
        "total_samples": total,
        "accuracy": correct / total,
        "predicted_distribution": {l: letters[l] for l in sorted(letters)},
        "predicted_pct": {l: round(100 * letters[l] / total, 1) for l in sorted(letters)},
        "ground_truth_distribution": {l: label_dist[l] for l in sorted(label_dist)},
        "entropy": round(entropy, 4),
        "max_entropy": round(max_entropy, 4),
        "entropy_ratio": round(entropy / max_entropy, 4),  # 1.0 = uniform
        "mode_collapsed": mode_collapsed,
        "dominant_letter": dominant_letter[0],
        "dominant_pct": round(100 * dominant_letter[1] / total, 1),
        "confusion_matrix": {
            f"true_{tl}": {
                f"pred_{pl}": confusion.get((pl, tl), 0)
                for pl in ["A", "B", "C"]
            }
            for tl in ["A", "B", "C"]
        },
        "per_category": {},
    }

    for cat in sorted(per_cat_total):
        n = per_cat_total[cat]
        cat_letters = per_cat_letters[cat]
        dom = cat_letters.most_common(1)[0]
        result["per_category"][cat] = {
            "total": n,
            "accuracy": round(per_cat_correct[cat] / n, 4),
            "distribution_pct": {
                l: round(100 * cat_letters[l] / n, 1)
                for l in sorted(cat_letters)
            },
            "dominant_letter": dom[0],
            "dominant_pct": round(100 * dom[1] / n, 1),
            "mode_collapsed": dom[1] / n > 0.60,
        }

    return result


def analyse_directory(gen_dir: str) -> dict:
    """Analyse all JSONL files in a directory, sort by checkpoint order."""
    jsonls = sorted(glob.glob(os.path.join(gen_dir, "*.jsonl")))
    if not jsonls:
        return {"error": f"No JSONL files found in {gen_dir}"}

    def _sort_key(p):
        name = os.path.basename(p).replace(".jsonl", "")
        m_step = re.search(r"step(\d+)", name)
        m_pct = re.search(r"(\d+)pct", name)
        if "end" in name:
            return (999, 999)
        if m_step:
            return (0, int(m_step.group(1)))
        if m_pct:
            return (1, int(m_pct.group(1)))
        return (2, 0)

    jsonls.sort(key=_sort_key)

    results = {}
    summary_table = []

    for path in jsonls:
        tag = os.path.basename(path).replace(".jsonl", "")
        # Skip eval_results JSONs that happen to be .jsonl
        if "eval_results" in tag:
            continue
        analysis = analyse_jsonl(path)
        results[tag] = analysis
        summary_table.append({
            "tag": tag,
            "accuracy": analysis.get("accuracy", 0),
            "entropy_ratio": analysis.get("entropy_ratio", 0),
            "dominant_letter": analysis.get("dominant_letter", "?"),
            "dominant_pct": analysis.get("dominant_pct", 0),
            "mode_collapsed": analysis.get("mode_collapsed", False),
        })

    # Find the collapse onset: first tag where mode_collapsed goes True
    collapse_onset = None
    for row in summary_table:
        if row["mode_collapsed"] and collapse_onset is None:
            collapse_onset = row["tag"]

    return {
        "gen_dir": gen_dir,
        "num_checkpoints": len(results),
        "collapse_onset_tag": collapse_onset,
        "summary": summary_table,
        "per_checkpoint": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Letter-distribution diagnostic")
    parser.add_argument("--jsonl_dir", type=str, required=True,
                        help="Directory containing checkpoint generation JSONLs")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Output path for analysis JSON (default: stdout)")
    args = parser.parse_args()

    result = analyse_directory(args.jsonl_dir)

    # Print summary table
    print("\n" + "=" * 72)
    print(f"LETTER DISTRIBUTION ANALYSIS — {args.jsonl_dir}")
    print("=" * 72)
    print(f"{'Tag':<18} {'Acc':>6} {'Entropy':>8} {'Dom':>4} {'Dom%':>6} {'Collapse':>9}")
    print("-" * 72)
    for row in result["summary"]:
        print(
            f"{row['tag']:<18} "
            f"{row['accuracy']:>6.3f} "
            f"{row['entropy_ratio']:>8.4f} "
            f"{row['dominant_letter']:>4} "
            f"{row['dominant_pct']:>5.1f}% "
            f"{'YES' if row['mode_collapsed'] else 'no':>9}"
        )
    print("=" * 72)
    if result["collapse_onset_tag"]:
        print(f"⚠️  Mode collapse onset: {result['collapse_onset_tag']}")
    else:
        print("✅ No mode collapse detected.")
    print()

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Full analysis saved to: {args.output_json}")
    else:
        json.dump(result, sys.stdout, indent=2)


# ─── Modal version (runs against the persistent volume) ──────────────────────
try:
    import modal

    _modal_app = modal.App("phase0-letter-dist")
    _volume = modal.Volume.from_name("debias-vlm-persistent-storage")
    _image = modal.Image.debian_slim(python_version="3.10")

    @_modal_app.function(
        image=_image,
        volumes={"/mnt/data": _volume},
        timeout=1800,
    )
    def analyse_on_volume(
        gen_dir: str = "/mnt/data/output_ppo_phase0/generations",
        output_json: str = "/mnt/data/output_ppo_phase0/letter_distribution_analysis.json",
    ):
        """Run letter-distribution analysis on the Modal volume and save results."""
        result = analyse_directory(gen_dir)

        # Print summary
        print("\n" + "=" * 72)
        print(f"LETTER DISTRIBUTION ANALYSIS — {gen_dir}")
        print("=" * 72)
        print(f"{'Tag':<18} {'Acc':>6} {'Entropy':>8} {'Dom':>4} {'Dom%':>6} {'Collapse':>9}")
        print("-" * 72)
        for row in result["summary"]:
            print(
                f"{row['tag']:<18} "
                f"{row['accuracy']:>6.3f} "
                f"{row['entropy_ratio']:>8.4f} "
                f"{row['dominant_letter']:>4} "
                f"{row['dominant_pct']:>5.1f}% "
                f"{'YES' if row['mode_collapsed'] else 'no':>9}"
            )
        print("=" * 72)
        if result["collapse_onset_tag"]:
            print(f"⚠️  Mode collapse onset: {result['collapse_onset_tag']}")
        else:
            print("✅ No mode collapse detected.")

        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        _volume.commit()
        print(f"\n✅ Analysis saved to: {output_json}")

except ImportError:
    pass  # Modal not installed; local-only usage


if __name__ == "__main__":
    main()
