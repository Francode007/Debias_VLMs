#!/usr/bin/env bash
# Phase 0.9.5 E1 — Inference-time bias-subspace erasure launcher.
#
# Spec: Phase0.9_Strategic_Plan.md §3.1
# Motivation: Phase0.9_Critical_Review.md §6
#
# Fires 4 Modal A100-80GB jobs in parallel (detached, disconnect-safe):
#   • E1a (L13 only)         × SB-Bench n=2916 canonical
#   • E1a (L13 only)         × VLBias  n=2000  stratified
#   • E1b (L13,L17,L21,L25)  × SB-Bench n=2916 canonical
#   • E1b (L13,L17,L21,L25)  × VLBias  n=2000  stratified
#
# Each job costs ~$0.25 on A100-80GB (~30 min). Total ≈ $1 for the full quad.
# Set VARIANTS / DATASETS env vars to subset (e.g. just E1a first).
#
# Usage
# ─────
#   bash scripts/phase09_e1_erasure_launch.sh
#   VARIANTS="E1a"           bash scripts/phase09_e1_erasure_launch.sh   # E1a only
#   DATASETS="sb_bench"      bash scripts/phase09_e1_erasure_launch.sh   # SB-Bench only
#   VARIANTS="E1a" DATASETS="sb_bench"  bash scripts/phase09_e1_erasure_launch.sh
#
# After jobs land:
#   mkdir -p Phase0.9/erasure
#   for V in E1a E1b; do
#     for D in sbbench vlbias; do
#       modal volume get debias-vlm-persistent-storage \
#         "/phase09_erasure/${V}_${D}_gen.jsonl" Phase0.9/erasure/ --force
#       modal volume get debias-vlm-persistent-storage \
#         "/phase09_erasure/${V}_${D}_eval_results.json" Phase0.9/erasure/ --force
#     done
#   done

set -uo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source debias_env/bin/activate
fi

VARIANTS="${VARIANTS:-E1a E1b}"
DATASETS="${DATASETS:-sb_bench vlbias}"

# Probe paths (resolved inside the container at /root/debias-vlms/...).
L13_PROBE="Phase0.8/a3_results/base_L13_probe_weights_biasA.npz"
L17_PROBE="Phase0.9/ensemble_bundle_raw/base_L17_probe_weights_biasA.npz"
L21_PROBE="Phase0.9/ensemble_bundle_raw/base_L21_probe_weights_biasA.npz"
L25_PROBE="Phase0.9/ensemble_bundle_raw/base_L25_probe_weights_biasA.npz"

E1A_LAYERS="13"
E1A_PROBES="$L13_PROBE"
E1B_LAYERS="13,17,21,25"
E1B_PROBES="$L13_PROBE,$L17_PROBE,$L21_PROBE,$L25_PROBE"

fire() {
  local variant="$1"; local dataset="$2"; local layers="$3"; local probes="$4"
  local stem; local tag; local out; local log

  if [[ "$dataset" == "sb_bench" ]]; then
    stem="sbbench"
  else
    stem="vlbias"
  fi
  tag="${variant}_${stem}_erasure"
  out="/mnt/data/phase09_erasure/${variant}_${stem}_gen.jsonl"
  log="/tmp/phase09_${variant}_${stem}_erasure.log"

  echo "▶ [${variant}/${stem}] firing → ${out}  (log: ${log})"
  nohup modal run --detach src/run_modal.py::run_inference_erasure \
    --dataset "${dataset}" \
    --output-jsonl "${out}" \
    --layer-indices "${layers}" \
    --probe-paths "${probes}" \
    --tag "${tag}" \
    >"${log}" 2>&1 &
  sleep 2
}

n_fired=0
for VAR in ${VARIANTS}; do
  case "${VAR}" in
    E1a) LAYERS="${E1A_LAYERS}"; PROBES="${E1A_PROBES}" ;;
    E1b) LAYERS="${E1B_LAYERS}"; PROBES="${E1B_PROBES}" ;;
    *)
      echo "⚠  Unknown variant '${VAR}' (expected E1a or E1b); skipping."
      continue
      ;;
  esac
  for DS in ${DATASETS}; do
    case "${DS}" in
      sb_bench|vlbias) ;;
      *) echo "⚠  Unknown dataset '${DS}' (expected sb_bench or vlbias); skipping."; continue ;;
    esac
    fire "${VAR}" "${DS}" "${LAYERS}" "${PROBES}"
    n_fired=$((n_fired + 1))
  done
done

echo ""
echo "✅ Fired ${n_fired} jobs (VARIANTS='${VARIANTS}', DATASETS='${DATASETS}')."
echo ""
echo "Track via:"
echo "    modal app list | head -15"
echo "    tail -f /tmp/phase09_E1*_erasure.log"
echo ""
echo "Disconnect-safe: --detach keeps jobs running after this shell closes."
echo ""
echo "Pull on completion (~30 min each, in parallel):"
echo "    mkdir -p Phase0.9/erasure"
echo "    for V in E1a E1b; do for D in sbbench vlbias; do"
echo "      modal volume get debias-vlm-persistent-storage \\"
echo "        \"/phase09_erasure/\${V}_\${D}_gen.jsonl\" Phase0.9/erasure/ --force"
echo "      modal volume get debias-vlm-persistent-storage \\"
echo "        \"/phase09_erasure/\${V}_\${D}_eval_results.json\" Phase0.9/erasure/ --force"
echo "    done; done"
