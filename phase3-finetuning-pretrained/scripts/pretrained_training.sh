#!/usr/bin/env bash
# ============================================================
# Phase 2 Fine-tuning — with pre-trained weights
# Task: Java Code Summarization (CodeXGLUE)
# Identical settings to full_training.sh except:
#   - model loaded from pre-trained checkpoint (not t5-small random)
#   - init-method = pretrained
#
# Usage:
#   bash scripts/pretrained_training.sh <batch_size> <lr> <model:d512|d320>
#
# 4 runs for d512:
#   bash scripts/pretrained_training.sh 8  5e-5  d512
#   bash scripts/pretrained_training.sh 16 1e-4  d512
#   bash scripts/pretrained_training.sh 32 1e-4  d512
#   bash scripts/pretrained_training.sh 64 2e-4  d512
#
# 4 runs for d320:
#   bash scripts/pretrained_training.sh 8  5e-5  d320
#   bash scripts/pretrained_training.sh 16 1e-4  d320
#   bash scripts/pretrained_training.sh 32 1e-4  d320
#   bash scripts/pretrained_training.sh 64 2e-4  d320
#
# GPU: set CUDA_VISIBLE_DEVICES before calling
#   CUDA_VISIBLE_DEVICES=0 bash scripts/pretrained_training.sh 32 1e-4 d512
# ============================================================
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: bash scripts/pretrained_training.sh <batch_size> <lr> <model:d512|d320>"
    exit 1
fi

BATCH=$1
LR=$2
MODEL_LABEL=$3

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PRETRAIN_DIR="/scratch/afia/code-embedding-optimization-pretrainning/outputs"

if [ "$MODEL_LABEL" = "d512" ]; then
    HIDDEN_DIM=512
    PRETRAINED_PATH="${PRETRAIN_DIR}/pretrained_d512/final"
elif [ "$MODEL_LABEL" = "d320" ]; then
    HIDDEN_DIM=320
    PRETRAINED_PATH="${PRETRAIN_DIR}/pretrained_d320/final"
else
    echo "Error: model must be d512 or d320"
    exit 1
fi

if [ ! -d "$PRETRAINED_PATH" ]; then
    echo "Error: pre-trained model not found at $PRETRAINED_PATH"
    exit 1
fi

cd "$PROJECT_DIR"

if [ -f "$PROJECT_DIR/afiabase/bin/activate" ]; then
    source "$PROJECT_DIR/afiabase/bin/activate"
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True

PRECISION_FLAG="--fp16"
if python -c "import torch; assert torch.cuda.is_bf16_supported()" 2>/dev/null; then
    PRECISION_FLAG="--bf16"
fi

OUTPUT_DIR="$PROJECT_DIR/outputs/pretrained_ft_bs${BATCH}_lr${LR}_${MODEL_LABEL}"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "PHASE 2 FINE-TUNING (pre-trained weights)"
echo "  Model      : T5-small ${MODEL_LABEL} (d_model=${HIDDEN_DIM})"
echo "  Checkpoint : ${PRETRAINED_PATH}"
echo "  Batch size : ${BATCH}"
echo "  LR         : ${LR}"
echo "  Precision  : ${PRECISION_FLAG}"
echo "  Output     : ${OUTPUT_DIR}"
echo "========================================"

python -m code_tasks.cli \
    --task summarization \
    --language java \
    --model-name-or-path "$PRETRAINED_PATH" \
    --dataset codexglue_code_to_text \
    --output-dir "$OUTPUT_DIR" \
    --overwrite-output-dir \
    --do-train \
    --do-eval \
    --do-test \
    --hidden-dim "$HIDDEN_DIM" \
    --init-method pretrained \
    --num-train-epochs 15 \
    --per-device-train-batch-size "$BATCH" \
    --per-device-eval-batch-size "$BATCH" \
    --gradient-accumulation-steps 1 \
    --learning-rate "$LR" \
    --warmup-ratio 0.1 \
    --lr-scheduler-type linear \
    --weight-decay 0.01 \
    --evaluation-strategy epoch \
    --save-strategy epoch \
    --save-total-limit 2 \
    --num-beams 4 \
    --max-input-length 512 \
    --max-target-length 128 \
    --logging-steps 50 \
    --metric-for-best-model bleu \
    --dataloader-num-workers 4 \
    $PRECISION_FLAG \
    --seed 42 \
    --report-to none \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "========================================"
echo "TRAINING COMPLETE"
echo "Results: ${OUTPUT_DIR}/test_results.json"
cat "${OUTPUT_DIR}/test_results.json" 2>/dev/null || echo "No test results found."
echo "========================================"
