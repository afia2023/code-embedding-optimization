#!/usr/bin/env bash
# ============================================================
# Experiment 2 — COMPRESSED (T5-small, d_model=320, from scratch)
# Task: Code Summarization  |  GPU: 2  |  Language: Java
# hidden_dim=320 chosen via PCA 90% variance analysis (original result: 322,
# rounded to 320 for clean 8-head attention: 320/8=40 heads each).
# Built from scratch — random init, no pretrained weights, no SVD/PCA on weights.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/afiabase/bin/activate"

export CUDA_VISIBLE_DEVICES=2
export PYTORCH_ALLOC_CONF=expandable_segments:True

OUTPUT_DIR="$PROJECT_DIR/outputs/summarization_compressed_t5small_d320_random"
mkdir -p "$OUTPUT_DIR"

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
    --device cuda:0 \
    --hidden-dim 320 \
    --init-method random \
    --num-train-epochs 15 \
    --per-device-train-batch-size 32 \
    --per-device-eval-batch-size 16 \
    --gradient-accumulation-steps 1 \
    --learning-rate 5e-4 \
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
echo "=== Compressed experiment finished ==="
echo "Timing report: ${OUTPUT_DIR}/timing.json"
cat "${OUTPUT_DIR}/timing.json"
