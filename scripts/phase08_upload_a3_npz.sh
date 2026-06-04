#!/usr/bin/env bash
# Phase 0.8 Strategic Plan §3 — Upload per-layer probe weight npzs to the
# Modal volume so phase08_washout_make_heads.sh can read them.
#
# Local source: Phase0.8/a3_results/base_L${N}_probe_weights_biasA.npz
# Volume target: /mnt/data/phase08_a3_results/base_L${N}_probe_weights_biasA.npz
#
# Modal CLI: `modal volume put <volume_name> <local_path> <remote_path>`
# The volume name is `debias-vlm-persistent-storage` (mounted at /mnt/data).
#
# Usage:
#   bash scripts/phase08_upload_a3_npz.sh                 # all 11 layers
#   bash scripts/phase08_upload_a3_npz.sh 13 17           # just those layers
#
# After this completes, run phase08_washout_make_heads.sh.
set -euo pipefail

source debias_env/bin/activate

VOLUME="debias-vlm-persistent-storage"
LOCAL_DIR="Phase0.8/a3_results"
REMOTE_DIR="phase08_a3_results"   # → /mnt/data/phase08_a3_results

DEFAULT_LAYERS=(1 5 9 11 13 17 21 25 29 33 35)
LAYERS=("${@:-${DEFAULT_LAYERS[@]}}")

# Ensure the target dir exists on the volume (modal volume put creates parents
# implicitly when given a file target, so this is just a no-op probe).
echo "▶ Uploading ${#LAYERS[@]} probe-weight npzs to ${VOLUME}:/${REMOTE_DIR}/"

for L in "${LAYERS[@]}"; do
    SRC="${LOCAL_DIR}/base_L${L}_probe_weights_biasA.npz"
    DST="${REMOTE_DIR}/base_L${L}_probe_weights_biasA.npz"
    if [[ ! -f "${SRC}" ]]; then
        echo "  ⚠ missing ${SRC} — skipping L${L}"
        continue
    fi
    echo "  L${L}: ${SRC} → ${VOLUME}:/${DST}"
    modal volume put --force "${VOLUME}" "${SRC}" "${DST}"
done

echo "✅ Upload complete. Verify with: modal volume ls ${VOLUME} ${REMOTE_DIR}"
