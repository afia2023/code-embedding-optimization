#!/usr/bin/env bash
# ============================================================
# LR Sweep — remaining LRs (3e-4, 5e-4) for both models
# 1e-4 already ran. This completes the sweep.
#
# Usage:
#   bash scripts/lr_sweep_remaining.sh 1    # GPU 1, d_model=512
#   bash scripts/lr_sweep_remaining.sh 2    # GPU 2, d_model=322
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
    HIDDEN_DIM=322
    LABEL="compressed_d322"
else
    HIDDEN_DIM=512
    LABEL="baseline_d512"
fi

SWEEP_DIR="$PROJECT_DIR/outputs/lr_sweep_${LABEL}"
mkdir -p "$SWEEP_DIR"

echo "Running remaining LRs for model=${LABEL}, hidden_dim=${HIDDEN_DIM}"
echo ""

for LR in 3e-4 5e-4; do
    RUN_DIR="$SWEEP_DIR/lr_${LR}"
    mkdir -p "$RUN_DIR"
    echo ">>> Testing LR=${LR} ..."

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
        --learning-rate "$LR" \
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
        --report-to none \
        2>&1 | tee "${RUN_DIR}/sweep.log"

    echo ">>> LR=${LR} done."
    echo ""
done

echo "========================================"
echo "All LRs complete. Summary:"
echo ""
printf "%-12s %-10s %-12s\n" "LR" "BLEU" "eval_loss"
printf "%-12s %-10s %-12s\n" "----" "----" "---------"
for LR in 1e-4 3e-4 5e-4; do
    VFILE="$SWEEP_DIR/lr_${LR}/validation_results.json"
    if [ -f "$VFILE" ]; then
        python3 - "$VFILE" "$LR" <<'PYEOF'
import json, sys
vfile, lr = sys.argv[1], sys.argv[2]
data = json.loads(open(vfile).read())
bleu = data.get('validation_bleu', 'N/A')
loss = data.get('validation_loss', 'N/A')
bleu_str = f"{bleu:.4f}" if isinstance(bleu, float) else str(bleu)
loss_str = f"{loss:.4f}" if isinstance(loss, float) else str(loss)
print(f"{lr:<12} {bleu_str:<10} {loss_str:<12}")
PYEOF
    else
        printf "%-12s %-10s %-12s\n" "$LR" "N/A" "N/A"
    fi
done
echo "========================================"
