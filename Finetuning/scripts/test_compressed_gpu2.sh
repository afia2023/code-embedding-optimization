#!/usr/bin/env bash
# ============================================================
# SMOKE TEST — Compressed T5-small d_model=320 (100 samples, 1 epoch) on GPU 2
# Purpose: verify pipeline runs end-to-end before full training
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/afiabase/bin/activate"

export CUDA_VISIBLE_DEVICES=2
export PYTORCH_ALLOC_CONF=expandable_segments:True

OUTPUT_DIR="$PROJECT_DIR/outputs/smoketest_compressed_t5small_d320"
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
    --num-train-epochs 1 \
    --max-train-samples 100 \
    --max-eval-samples 50 \
    --max-test-samples 50 \
    --per-device-train-batch-size 32 \
    --per-device-eval-batch-size 16 \
    --gradient-accumulation-steps 1 \
    --learning-rate 5e-4 \
    --warmup-ratio 0.1 \
    --weight-decay 0.01 \
    --evaluation-strategy epoch \
    --save-strategy epoch \
    --save-total-limit 1 \
    --num-beams 2 \
    --max-input-length 512 \
    --max-target-length 128 \
    --logging-steps 10 \
    --metric-for-best-model bleu \
    --fp16 \
    --seed 42 \
    --report-to none \
    2>&1 | tee "${OUTPUT_DIR}/smoketest.log"

echo ""
echo "=== Compressed smoke test PASSED ==="
echo "Timing: ${OUTPUT_DIR}/timing.json"
cat "${OUTPUT_DIR}/timing.json"
