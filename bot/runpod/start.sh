#!/usr/bin/env bash
# Train on a RunPod pod, then stop the pod so it stops billing.
#   HOURS=4 bash runpod/start.sh [extra train.py args]
# Checkpoints go to /workspace (the pod's persistent volume), so re-running resumes.
set -euo pipefail
cd "$(dirname "$0")/.."

HOURS="${HOURS:-4}"
CKPT_DIR="${CKPT_DIR:-/workspace/poker-ckpt}"

pip install -q -r requirements.txt
python -m unittest tests.test_poker

status=0
python train.py --ckpt-dir "$CKPT_DIR" --hours "$HOURS" "$@" 2>&1 | tee -a "$CKPT_DIR.train.log" || status=$?
python evaluate.py --ckpt "$CKPT_DIR/latest.pt" --pairs 20000 2>&1 | tee -a "$CKPT_DIR.train.log" || true

if [ -n "${RUNPOD_POD_ID:-}" ] && [ "${STOP_WHEN_DONE:-1}" = "1" ]; then
  echo "stopping pod $RUNPOD_POD_ID"
  runpodctl stop pod "$RUNPOD_POD_ID"
fi
exit $status
