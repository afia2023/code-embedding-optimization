#!/usr/bin/env bash
# ============================================================
# Full Training — T5-small d512 (60.5M) or d320 (27.5M)
# Task: Java Code Summarization (CodeXGLUE)
# Run after lr_sweep_all_batchsizes.sh to use the best LR.
#
# Usage:
#   bash scripts/full_training.sh <batch_size> <lr> <model>
#
# Examples:
#   bash scripts/full_training.sh 8  6.25e-5 d512
#   bash scripts/full_training.sh 16 1.25e-4 d320
#   bash scripts/full_training.sh 32 2.5e-4  d512
#   bash scripts/full_training.sh 64 3e-4    d320
#
# GPU: set CUDA_VISIBLE_DEVICES before calling
#   CUDA_VISIBLE_DEVICES=0 bash scripts/full_training.sh 8 6.25e-5 d512
# ============================================================
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: bash scripts/full_training.sh <batch_size> <lr> <model:d512|d320>"
    exit 1
fi

BATCH=$1
LR=$2
MODEL_LABEL=$3

if [ "$MODEL_LABEL" = "d512" ]; then
    HIDDEN_DIM=512
elif [ "$MODEL_LABEL" = "d320" ]; then
    HIDDEN_DIM=320
else
    echo "Error: model must be d512 (T5-small 60.5M) or d320 (compressed 27.5M)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ -f "$PROJECT_DIR/afiabase/bin/activate" ]; then
    source "$PROJECT_DIR/afiabase/bin/activate"
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True

OUTPUT_DIR="$PROJECT_DIR/outputs/full_training_bs${BATCH}_lr${LR}_${MODEL_LABEL}"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "FULL TRAINING"
echo "  Model     : T5-small ${MODEL_LABEL} (d_model=${HIDDEN_DIM})"
echo "  Batch size: ${BATCH}"
echo "  LR        : ${LR}"
echo "  Output    : ${OUTPUT_DIR}"
echo "========================================"

python -m code_tasks.cli \
    --task summarization \
    --language java \
    --model-name-or-path t5-small \
    --dataset codexglue_code_to_text \
    --output-dir "$OUTPUT_DIR" \
    --overwrite-output-dir \
    --do-train \
    --do-eval \
    --do-test \
    --hidden-dim "$HIDDEN_DIM" \
    --init-method random \
    --num-train-epochs 15 \
    --per-device-train-batch-size "$BATCH" \
    --per-device-eval-batch-size "$BATCH" \
    --gradient-accumulation-steps 1 \
    --learning-rate "$LR" \
    --warmup-ratio 0.1 \
    --weight-decay 0.01 \
    --evaluation-strategy epoch \
    --save-strategy epoch \
    --save-total-limit 2 \
    --num-beams 4 \
    --max-input-length 512 \
    --max-target-length 128 \
    --logging-steps 50 \
    --metric-for-best-model bleu \
    --fp16 \
    --seed 42 \
    --report-to none \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "========================================"
echo "TRAINING COMPLETE"
echo "Results: ${OUTPUT_DIR}/test_results.json"
cat "${OUTPUT_DIR}/test_results.json" 2>/dev/null || echo "No test results found."
echo "========================================"
