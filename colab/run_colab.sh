#!/usr/bin/env bash
# Runs the classification pipeline on a Google Colab GPU through the Colab CLI
# (https://pypi.org/project/google-colab-cli/), downloads results/ and the checkpoints and stops
# the session at the end, even on errors.
#
# usage: colab/run_colab.sh [ARCH ...]             train + evaluate (default: resnet18 simplecnn)
#        PIPELINE=cv colab/run_colab.sh            cross-validate configs.py, retrain the best one, evaluate
# env:   COLAB_GPU (A100), SEED (42), EPOCHS (60, train only),
#        CONFIGS ("baseline schedule schedule_wd schedule_wd_freeze"), CV_SEEDS ("42 43 44") and
#        CV_ONLY (0; 1 skips the final retraining) for cv,
#        CKPT_DIR (.): where the checkpoints are unpacked, keeping their results/... paths
set -Eeuo pipefail
cd "$(dirname "$(realpath "$0")")/.."

GPU=${COLAB_GPU:-A100}
PIPELINE=${PIPELINE:-train}
ARCHS="${*:-resnet18 simplecnn}"
CONFIGS=${CONFIGS:-baseline schedule schedule_wd schedule_wd_freeze}
CV_SEEDS=${CV_SEEDS:-42 43 44}
CKPT_DIR=${CKPT_DIR:-.}
SESSION="mri-$(date +%m%d-%H%M%S)"
TMP=$(mktemp -d)
if [[ $PIPELINE == cv ]]; then LOG=results/cv/colab_run.log; else LOG=results/colab_run.log; fi
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
balance() { colab usage 2>&1 | sed -n 's/^Current balance: *\([0-9.,]*\).*/\1/p' | tr -d ,; }

tar czf "$TMP/mri.tar.gz" preprocess.py classifier.py configs.py train.py evaluate.py cv.py requirements.txt classification

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
if [[ $PIPELINE == cv ]]; then log "running cv for: $CONFIGS (seeds $CV_SEEDS)"; else log "running pipeline for: $ARCHS"; fi
colab exec --session "$SESSION" --timeout 5400 \
  --env PIPELINE="$PIPELINE" --env ARCHS="$ARCHS" --env CONFIGS="$CONFIGS" --env CV_SEEDS="$CV_SEEDS" \
  --env CV_ONLY="${CV_ONLY:-0}" \
  --env SEED="${SEED:-42}" --env EPOCHS="${EPOCHS:-60}" \
  --file colab/remote_train.py 2>&1 | grep -vE 'MissingIDFieldWarning|_validate\(' | tee -a "$LOG"
grep -q REMOTE_DONE "$LOG" || { log "ERROR: remote pipeline did not finish"; exit 1; }

colab download --session "$SESSION" /content/mri/results.tar.gz "$TMP/results.tar.gz" >>"$LOG" 2>&1
tar xzf "$TMP/results.tar.gz"
log "results saved to results/"
colab download --session "$SESSION" /content/mri/checkpoints.tar "$TMP/checkpoints.tar" >>"$LOG" 2>&1
mkdir -p "$CKPT_DIR"
tar xf "$TMP/checkpoints.tar" -C "$CKPT_DIR"
log "checkpoints ($(du -sh "$TMP/checkpoints.tar" | cut -f1)) saved under $CKPT_DIR/results/"
