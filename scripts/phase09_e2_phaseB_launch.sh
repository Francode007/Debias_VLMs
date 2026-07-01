#!/usr/bin/env bash
# Phase 0.9.5 E2 Phase B — Larger-n probe-direction convergence diagnostic.
#
# Spec: Phase0.9_Strategic_Plan.md §3.2 (E2 escalation when Phase A is ambiguous)
# Phase A verdict: size_cos(L13, n=300 → n=600) = 0.927 ∈ [0.90, 0.95) — ambiguous.
# Phase B settles it by extracting HS on the FULL base_vlbias_gen.jsonl
# (n=3000 → ~2000 usable bias-aligned after dropping ambig records), then
# refitting at multiple n and computing cos(w_600, w_max).
#
# Step 1 (Modal, ~$1, ~30 min): re-run run_layer_probe with max_per_cell=100,
#         producing a (~3000, 37, 2048) HS cache at
#         /mnt/data/phase09_probe_convergence/probe_hs_n3000.npz
#
# Step 2 (after Modal finishes; CPU-only, free): pull the cache and re-run
#         scripts/phase09_probe_convergence.py against it. Verdict comes from
#         the updated diagnostic.json.
#
# Usage
# ─────
#   bash scripts/phase09_e2_phaseB_launch.sh                # fire the Modal job
#   bash scripts/phase09_e2_phaseB_launch.sh pull           # pull cache (after Modal done)
#   bash scripts/phase09_e2_phaseB_launch.sh rerun          # re-run CPU convergence script
#   bash scripts/phase09_e2_phaseB_launch.sh pull rerun     # both

set -uo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source debias_env/bin/activate
fi

STAGES=("$@")
if [[ ${#STAGES[@]} -eq 0 ]]; then
    STAGES=(fire)
fi

VOL="debias-vlm-persistent-storage"
REMOTE_CACHE="/phase09_probe_convergence/probe_hs_n3000.npz"
REMOTE_OUTDIR="/mnt/data/phase09_probe_convergence"
LOCAL_CACHE="Phase0.9/probe_convergence/probe_hs_n3000.npz"

for STAGE in "${STAGES[@]}"; do
    case "${STAGE}" in
        fire)
            LOG="/tmp/phase09_e2_phaseB.log"
            echo "▶ Firing Modal HS extract (max_per_cell=100) → ${REMOTE_OUTDIR}${REMOTE_CACHE}"
            echo "  log: ${LOG}"
            # run_layer_probe re-runs probe_layers.py end-to-end:
            #   * extracts HS at all 37 LM layers on the existing base_vlbias_gen.jsonl
            #     stratified at max_per_cell=100 (→ up to 3000 records before
            #     bias_aligned filter; ~2000 usable after).
            #   * saves the cache npz at --cache-npz.
            #   * also runs the per-layer CV probing as a side effect (cheap CPU);
            #     output JSON ends up at <output-dir>/base_layerwise_probe.json
            #     and is informational only.
            nohup modal run --detach src/run_modal.py::run_layer_probe \
                --variant base \
                --max-per-cell 100 \
                --max-holdout 0 \
                --cache-npz "/mnt/data/phase09_probe_convergence/probe_hs_n3000.npz" \
                --output-dir "${REMOTE_OUTDIR}" \
                --save-plot false \
                >"${LOG}" 2>&1 &
            sleep 2
            echo "✅ Fired (detached). Wait ~30 min, then:"
            echo "    tail -f ${LOG}"
            echo "    bash scripts/phase09_e2_phaseB_launch.sh pull rerun"
            ;;

        pull)
            mkdir -p Phase0.9/probe_convergence
            echo "▶ Pulling ${REMOTE_CACHE} → ${LOCAL_CACHE}"
            modal volume get "${VOL}" "${REMOTE_CACHE}" Phase0.9/probe_convergence/ --force
            echo "✅ Pull complete."
            ls -lh "${LOCAL_CACHE}"
            ;;

        rerun)
            if [[ ! -f "${LOCAL_CACHE}" ]]; then
                echo "❌ Local cache missing: ${LOCAL_CACHE}"
                echo "   Run 'bash scripts/phase09_e2_phaseB_launch.sh pull' first."
                exit 1
            fi
            echo "▶ Re-running convergence diagnostic on ${LOCAL_CACHE}"
            python scripts/phase09_probe_convergence.py \
                --cache "${LOCAL_CACHE}" \
                --out Phase0.9/probe_convergence/diagnostic_phaseB.json
            echo "✅ Phase B verdict in Phase0.9/probe_convergence/diagnostic_phaseB.json"
            ;;

        *)
            echo "⚠  Unknown stage '${STAGE}' (expected: fire | pull | rerun)"
            ;;
    esac
done
