#!/usr/bin/env bash
# ============================================================
# Experiment 1 — BASELINE (pre-trained CodeT5+ 220M, full d_model=768)
# Task: Code Summarization  |  GPU: 0  |  Language: Java
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR/src"
source "$PROJECT_DIR/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES=0

OUTPUT_DIR="$PROJECT_DIR/outputs/summarization_baseline_d768"
mkdir -p "$OUTPUT_DIR"

python -m code_tasks.cli \
    --task summarization \
    --language java \
    --model-name-or-path Salesforce/codet5p-220m \
    --dataset codexglue_code_to_text \
    --output-dir "$OUTPUT_DIR" \
    --overwrite-output-dir \
    --do-train \
    --do-eval \
    --do-test \
    --device cuda:0 \
    --num-train-epochs 10 \
    --per-device-train-batch-size 16 \
    --per-device-eval-batch-size 16 \
    --gradient-accumulation-steps 1 \
    --learning-rate 5e-5 \
    --warmup-ratio 0.05 \
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
echo "=== Baseline experiment finished ==="
echo "Timing report: ${OUTPUT_DIR}/timing.json"
cat "${OUTPUT_DIR}/timing.json"
