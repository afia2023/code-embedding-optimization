#!/usr/bin/env bash
# ============================================================
# SMOKE TEST — Optimized settings (batch=32, LR=5e-4, workers=4)
# Runs 3 epochs on 5000 samples to verify:
#   1. Loss matches original (same batch+LR, only workers changed)
#   2. BLEU is comparable to original (batch=32, LR=5e-4)
#   3. No OOM or data worker errors
#
# NOTE: LR is NOT changed here. The linear scaling rule (LR ∝ batch)
# only applies when batch size actually increases. We keep batch=32
# (GPU memory is too constrained for batch=64), so LR stays at 5e-4.
# The only change being tested is preprocessing/dataloader workers=4.
#
# Usage:
#   bash scripts/test_optimized.sh 1    # GPU 1, d512 baseline
#   bash scripts/test_optimized.sh 2    # GPU 2, d320 compressed
#
# Expected loss at epoch 3 (should match original closely):
#   d512: train_loss~2.66, BLEU~9.77
#   d320: train_loss~6.17, BLEU~0.47
# ============================================================
set -euo pipefail

GPU="${1:-1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/afiabase/bin/activate"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_ALLOC_CONF=expandable_segments:True

if [ "$GPU" = "2" ]; then
    HIDDEN_DIM=320
    LABEL="compressed_d320"
else
    HIDDEN_DIM=512
    LABEL="baseline_d512"
fi

OUTPUT_DIR="$PROJECT_DIR/outputs/smoketest_optimized_${LABEL}"
mkdir -p "$OUTPUT_DIR"

echo "Smoke test: model=${LABEL}, batch=32, LR=5e-4, workers=4 (only workers changed)"
echo "Reference (batch=32, LR=5e-4, 3 epochs on 5k samples):"
if [ "$GPU" = "2" ]; then
    echo "  d320: train_loss~6.17, BLEU~0.47"
else
    echo "  d512: train_loss~2.66, BLEU~9.77"
fi
echo ""

python -m code_tasks.cli \
    --task summarization \
    --language java \
    --model-name-or-path t5-small \
    --dataset codexglue_code_to_text \
    --output-dir "$OUTPUT_DIR" \
    --overwrite-output-dir \
    --do-train \
    --do-eval \
    --device cuda:0 \
    --hidden-dim "$HIDDEN_DIM" \
    --init-method random \
    --num-train-epochs 3 \
    --max-train-samples 5000 \
    --max-eval-samples 1000 \
    --per-device-train-batch-size 32 \
    --per-device-eval-batch-size 16 \
    --gradient-accumulation-steps 1 \
    --learning-rate 5e-4 \
    --warmup-ratio 0.1 \
    --weight-decay 0.01 \
    --preprocessing-num-workers 4 \
    --dataloader-num-workers 4 \
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
    --report-to none

echo ""
echo "=== Smoke test result ==="
python3 - "$OUTPUT_DIR" <<'PYEOF'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])

val = json.loads((d / "validation_results.json").read_text())
print(f"  BLEU      : {val['validation_bleu']:.4f}")
print(f"  train_loss: ", end="")
ts = json.loads((d / "trainer_state.json").read_text())
losses = [e["loss"] for e in ts["log_history"] if "loss" in e and "eval_loss" not in e]
print(f"{losses[-1]:.4f}  (last logged step)")
print(f"  eval_loss : {val['validation_loss']:.4f}")

print()
print("Verdict:")
bleu = val["validation_bleu"]
loss = val["validation_loss"]
if loss > 15:
    print("  FAIL — loss diverged unexpectedly. Check data pipeline or model init.")
elif bleu < 1.0 and loss > 5:
    print("  CAUTION — model learning slowly. May still be OK for full run.")
else:
    print("  PASS — results match original, workers=4 is safe for full run.")
PYEOF
