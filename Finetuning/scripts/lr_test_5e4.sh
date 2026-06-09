#!/usr/bin/env bash
# Test LR=5e-4 for d512 (GPU1) or d320 (GPU2)
# Usage:  bash scripts/lr_test_5e4.sh 1   (GPU1, d512)
#         bash scripts/lr_test_5e4.sh 2   (GPU2, d320)
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

RUN_DIR="$PROJECT_DIR/outputs/lr_sweep_${LABEL}/lr_5e-4"
mkdir -p "$RUN_DIR"

echo "Testing LR=5e-4 | model=${LABEL} | hidden_dim=${HIDDEN_DIM}"

python -m code_tasks.cli \
    --task summarization \
    --language java \
    --model-name-or-path t5-small \
    --dataset codexglue_code_to_text \
    --output-dir "$RUN_DIR" \
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
    --evaluation-strategy epoch \
    --save-strategy epoch \
    --save-total-limit 1 \
    --num-beams 2 \
    --max-input-length 512 \
    --max-target-length 128 \
    --logging-steps 20 \
    --metric-for-best-model bleu \
    --fp16 \
    --seed 42 \
    --report-to none

echo ""
echo "=== LR comparison for ${LABEL} ==="
printf "%-10s %-10s %-12s\n" "LR" "BLEU" "eval_loss"
printf "%-10s %-10s %-12s\n" "---" "----" "---------"
SWEEP_DIR="$PROJECT_DIR/outputs/lr_sweep_${LABEL}"
for LR in 1e-4 3e-4 5e-4; do
    VFILE="$SWEEP_DIR/lr_${LR}/validation_results.json"
    if [ -f "$VFILE" ]; then
        python3 - "$VFILE" "$LR" <<'PYEOF'
import json, sys
data = json.loads(open(sys.argv[1]).read())
lr = sys.argv[2]
bleu = data.get('validation_bleu', 'N/A')
loss = data.get('validation_loss', 'N/A')
b = f"{bleu:.4f}" if isinstance(bleu, float) else bleu
l = f"{loss:.4f}" if isinstance(loss, float) else loss
print(f"{lr:<10} {b:<10} {l:<12}")
PYEOF
    else
        printf "%-10s %-10s %-12s\n" "$LR" "N/A" "N/A"
    fi
done
