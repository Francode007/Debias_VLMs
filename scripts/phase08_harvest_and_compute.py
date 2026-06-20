#!/usr/bin/env python3
"""Phase 0.8 — Poll, download, and compute flip-rate + seed metrics.

Waits for all 10 detached Modal gens to land in the volume, then downloads
them locally and invokes the two metric scripts. Run after firing
`phase08_launch_gens_parallel.sh`.

Usage:
  python scripts/phase08_harvest_and_compute.py
  python scripts/phase08_harvest_and_compute.py --no-wait    # don't wait, just process what's there
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import modal

VOLUME_NAME = "debias-vlm-persistent-storage"

CF_FILES = {
    "base":              "phase08_cf/9axis_base_sbbench_gen.jsonl",
    "phase08_2k":        "phase08_cf/9axis_phase08_2k_sbbench_gen.jsonl",
    "corrOnly":          "phase08_cf/9axis_phase08_2k_corrOnly_sbbench_gen.jsonl",
    "biasOnly":          "phase08_cf/9axis_phase08_2k_biasOnly_sbbench_gen.jsonl",
}
SEED_FILES = {
    f"L{L}_s{S}": f"phase08_seeds/9axis_L{L}_s{S}_sbbench_gen.jsonl"
    for L in (13, 17) for S in (1, 2, 4)
}

CF_LOCAL_DIR = Path("Phase0.8/counterfactual_eval/cf_runs")
SEED_LOCAL_DIR = Path("Phase0.8/seed_runs")
CF_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
SEED_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_ROWS_9AXIS = 3747


def _existing(v) -> dict[str, int]:
    """Return {path: size} for all files in /phase08_cf and /phase08_seeds."""
    out = {}
    for d in ("/phase08_cf", "/phase08_seeds"):
        try:
            for e in v.iterdir(d):
                out[str(e.path)] = getattr(e, "size", 0) or 0
        except Exception:
            pass
    return out


def _is_complete(v, remote_path: str) -> bool:
    """A gen file is considered complete if it has ≥ 3747 lines."""
    snap = _existing(v)
    if remote_path not in snap:
        return False
    # crude: size >= 350 bytes/row × 3747 rows = ~1.3MB
    return snap[remote_path] >= 1_200_000


def _download(v, remote_path: str, local_path: Path) -> int:
    """Download a single file from the volume; return byte count."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as fh:
        n = 0
        for chunk in v.read_file(remote_path):
            fh.write(chunk)
            n += len(chunk)
    return n


def _line_count(path: Path) -> int:
    with open(path) as fh:
        return sum(1 for _ in fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wait", action="store_true",
                    help="don't poll; process whatever already exists")
    ap.add_argument("--poll-interval", type=int, default=60,
                    help="seconds between polls")
    ap.add_argument("--max-wait", type=int, default=3600,
                    help="give up after N seconds")
    args = ap.parse_args()

    v = modal.Volume.from_name(VOLUME_NAME)

    all_remote = {**{k: f"/{p}" for k, p in CF_FILES.items()},
                  **{k: f"/{p}" for k, p in SEED_FILES.items()}}
    # iterdir paths don't include leading slash
    iter_remote = {k: p.lstrip("/") for k, p in all_remote.items()}

    if not args.no_wait:
        start = time.time()
        done: set[str] = set()
        while time.time() - start < args.max_wait:
            snap = _existing(v)
            for tag, rp in iter_remote.items():
                if tag in done:
                    continue
                if snap.get(rp, 0) >= 1_200_000:
                    done.add(tag)
            print(f"[poll {int(time.time()-start)}s] complete: {len(done)}/{len(iter_remote)} "
                  f"→ {sorted(done)}")
            if len(done) == len(iter_remote):
                print("✅ all gens complete")
                break
            time.sleep(args.poll_interval)
        else:
            print(f"⚠ timed out after {args.max_wait}s; processing partial set")

    # Download whatever is present
    print("\n=== Downloading ===")
    local_paths: dict[str, Path] = {}
    for tag, rp in iter_remote.items():
        is_cf = tag in CF_FILES
        local = (CF_LOCAL_DIR if is_cf else SEED_LOCAL_DIR) / Path(rp).name
        try:
            n = _download(v, rp, local)
            rows = _line_count(local)
            print(f"  ✓ {tag:18} → {local} ({n} bytes, {rows} rows)")
            local_paths[tag] = local
        except Exception as ex:
            print(f"  ✗ {tag:18} download failed: {ex}")

    # Compute CF flip-rate
    cf_present = {k: local_paths[k] for k in CF_FILES if k in local_paths}
    if "base" in cf_present and len(cf_present) >= 2:
        print("\n=== CF flip-rate ===")
        out_json = Path("Phase0.8/counterfactual_eval/cf_metrics.json")
        gens_args = [f"{k}={p}" for k, p in cf_present.items()]
        cmd = [
            sys.executable,
            "scripts/phase08_compute_flip_rate.py",
            "--pairs", "Phase0.8/counterfactual_eval/cf_pairs.parquet",
            "--gens", *gens_args,
            "--output-json", str(out_json),
        ]
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=True)
        with open(out_json) as fh:
            cf = json.load(fh)
        print(json.dumps(cf.get("summary", cf), indent=2))
    else:
        print("\n⚠ skipping CF flip-rate (need base + ≥1 variant)")

    # Compute seed-triple per layer
    for L in (13, 17):
        seeds_present = {S: local_paths[f"L{L}_s{S}"]
                         for S in (1, 2, 4) if f"L{L}_s{S}" in local_paths}
        if "base" in local_paths and len(seeds_present) >= 2:
            print(f"\n=== Seed triple L{L} ===")
            out_json = SEED_LOCAL_DIR / f"L{L}_aggregate.json"
            cmd = [
                sys.executable,
                "scripts/phase08_seed_aggregate.py",
                "--vanilla", str(local_paths["base"]),
                "--variant", f"L{L}",
                "--seeds", *[str(p) for p in seeds_present.values()],
                "--output-json", str(out_json),
            ]
            print("$", " ".join(cmd))
            subprocess.run(cmd, check=True)
            with open(out_json) as fh:
                agg = json.load(fh)
            print(json.dumps(agg.get("summary", agg), indent=2))
        else:
            print(f"\n⚠ skipping L{L} seed triple (need base + ≥2 seeds, have {len(seeds_present)})")


if __name__ == "__main__":
    main()
