#!/usr/bin/env bash
# Runs the whole classification pipeline (preprocess -> train -> evaluate) on a Google Colab GPU
# through the Colab CLI (https://pypi.org/project/google-colab-cli/), downloads results/ and stops
# the session at the end, even on errors.
#
# usage: colab/run_colab.sh [ARCH ...]        (default: resnet18 simplecnn)
# env:   COLAB_GPU (A100), SEED (42), EPOCHS (60)
set -Eeuo pipefail
cd "$(dirname "$(realpath "$0")")/.."

GPU=${COLAB_GPU:-A100}
ARCHS="${*:-resnet18 simplecnn}"
SESSION="mri-$(date +%m%d-%H%M%S)"
TMP=$(mktemp -d)
mkdir -p results
LOG=results/colab_run.log
: > "$LOG"
log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
balance() { colab usage 2>&1 | sed -n 's/^Current balance: *\([0-9.,]*\).*/\1/p' | tr -d ,; }

tar czf "$TMP/mri.tar.gz" preprocess.py classifier.py train.py evaluate.py requirements.txt classification

CU0=$(balance); log "compute units before: ${CU0:-?}"
T0=$(date +%s)
log "creating session $SESSION ($GPU)"
# --high-mem selects the A100-SXM4-80GB shape; without it Colab hands out the 40 GB one.
colab new --session "$SESSION" --gpu "$GPU" --high-mem >>"$LOG" 2>&1
cleanup() {
  log "stopping session $SESSION"
  colab stop --session "$SESSION" >>"$LOG" 2>&1 || log "WARNING: stop failed, run: colab stop --session $SESSION"
  rm -rf "$TMP"
  CU1=$(balance)
  log "compute units after: ${CU1:-?}; session time $(( $(date +%s) - T0 )) s"
}
trap cleanup EXIT
log "session ready in $(( $(date +%s) - T0 )) s"

colab upload --session "$SESSION" "$TMP/mri.tar.gz" /content/mri.tar.gz >>"$LOG" 2>&1
log "running pipeline for: $ARCHS"
colab exec --session "$SESSION" --timeout 5400 \
  --env ARCHS="$ARCHS" --env SEED="${SEED:-42}" --env EPOCHS="${EPOCHS:-60}" \
  --file colab/remote_train.py 2>&1 | grep -vE 'MissingIDFieldWarning|_validate\(' | tee -a "$LOG"
grep -q REMOTE_DONE "$LOG" || { log "ERROR: remote pipeline did not finish"; exit 1; }

colab download --session "$SESSION" /content/mri/results.tar.gz "$TMP/results.tar.gz" >>"$LOG" 2>&1
tar xzf "$TMP/results.tar.gz"
log "results saved to results/"
