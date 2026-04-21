#!/usr/bin/env bash
# ============================================================
# LR Sweep — All Batch Sizes (8, 16, 32, 64)
# Models: T5-small d512 (60.5M) and compressed d320 (27.5M)
# Task: Java Code Summarization (CodeXGLUE)
#
# LR candidates derived from linear scaling rule (Goyal et al., 2017)
# anchored on T5 baseline (Raffel et al., 2020): batch=128, LR=1e-3
# Capped at HuggingFace T5-small recommended range (1e-4 to 3e-4)
# for batch sizes where scaling exceeds the safe range.
#
# Sweep config: 40k train samples, 8 epochs (enough to see convergence)
#
# Usage:
#   bash scripts/lr_sweep_all_batchsizes.sh          # runs all batch sizes
#   bash scripts/lr_sweep_all_batchsizes.sh 16        # runs only batch 16
#
# GPU assignment:
#   Set CUDA_VISIBLE_DEVICES before calling, or pass via env:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/lr_sweep_all_batchsizes.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate virtualenv if present
if [ -f "$PROJECT_DIR/afiabase/bin/activate" ]; then
    source "$PROJECT_DIR/afiabase/bin/activate"
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Which batch sizes to run (default: all)
TARGET_BATCH="${1:-all}"

# ---- LR candidates per batch size ----
# Derived via: LR = 1e-3 * (batch / 128)  [Goyal et al., 2017 + Raffel et al., 2020]
# Capped at 3e-4 for batch 64 per HuggingFace T5-small guidelines
declare -A LR_CANDIDATES
LR_CANDIDATES[8]="3e-5 6.25e-5 1.25e-4"
LR_CANDIDATES[16]="6.25e-5 1.25e-4 2.5e-4"
LR_CANDIDATES[32]="1.25e-4 2.5e-4 5e-4"
LR_CANDIDATES[64]="1.25e-4 2.5e-4 3e-4"

# ---- Models ----
# d512 = T5-small standard (60.5M params)
# d320 = compressed model (27.5M params)
declare -A MODEL_DIMS
MODEL_DIMS[d512]=512
MODEL_DIMS[d320]=320

SWEEP_SAMPLES=40000
SWEEP_EPOCHS=8

run_sweep() {
    local BATCH=$1
    local LR=$2
    local MODEL_LABEL=$3
    local HIDDEN_DIM=$4

    local RUN_DIR="$PROJECT_DIR/outputs/lr_sweep_bs${BATCH}_${MODEL_LABEL}/lr_${LR}"
    mkdir -p "$RUN_DIR"

    echo ""
    echo ">>> batch=${BATCH} | lr=${LR} | model=${MODEL_LABEL} (d_model=${HIDDEN_DIM})"

    python -m code_tasks.cli \
        --task summarization \
        --language java \
        --model-name-or-path t5-small \
        --dataset codexglue_code_to_text \
        --output-dir "$RUN_DIR" \
        --overwrite-output-dir \
        --do-train \
        --do-eval \
        --hidden-dim "$HIDDEN_DIM" \
        --init-method random \
        --num-train-epochs "$SWEEP_EPOCHS" \
        --max-train-samples "$SWEEP_SAMPLES" \
        --max-eval-samples 5000 \
        --per-device-train-batch-size "$BATCH" \
        --per-device-eval-batch-size "$BATCH" \
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
        --logging-steps 50 \
        --metric-for-best-model bleu \
        --fp16 \
        --seed 42 \
        --report-to none \
        2>&1 | tee "${RUN_DIR}/sweep.log"

    echo ">>> Done: batch=${BATCH} | lr=${LR} | model=${MODEL_LABEL}"
}

print_summary() {
    local BATCH=$1
    local MODEL_LABEL=$2
    local SWEEP_DIR="$PROJECT_DIR/outputs/lr_sweep_bs${BATCH}_${MODEL_LABEL}"

    echo ""
    echo "========================================"
    echo "Summary: batch=${BATCH} | model=${MODEL_LABEL}"
    printf "%-12s %-10s %-12s\n" "LR" "BLEU" "eval_loss"
    printf "%-12s %-10s %-12s\n" "---" "----" "---------"

    for LR in ${LR_CANDIDATES[$BATCH]}; do
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
print(f"{lr:<12} {b:<10} {l:<12}")
PYEOF
        else
            printf "%-12s %-10s %-12s\n" "$LR" "N/A" "N/A"
        fi
    done
    echo "========================================"
}

# ---- Main loop ----
for BATCH in 8 16 32 64; do
    if [ "$TARGET_BATCH" != "all" ] && [ "$TARGET_BATCH" != "$BATCH" ]; then
        continue
    fi

    echo ""
    echo "###################################################"
    echo "# BATCH SIZE: $BATCH"
    echo "# LR candidates: ${LR_CANDIDATES[$BATCH]}"
    echo "###################################################"

    for MODEL_LABEL in d512 d320; do
        HIDDEN_DIM="${MODEL_DIMS[$MODEL_LABEL]}"
        for LR in ${LR_CANDIDATES[$BATCH]}; do
            run_sweep "$BATCH" "$LR" "$MODEL_LABEL" "$HIDDEN_DIM"
        done
        print_summary "$BATCH" "$MODEL_LABEL"
    done
done

echo ""
echo "========================================"
echo "ALL SWEEPS COMPLETE"
echo "Results saved under: $PROJECT_DIR/outputs/lr_sweep_bs*"
echo ""
echo "Next step: pick best LR per batch size and run full training:"
echo "  bash scripts/full_training.sh <batch_size> <lr> <model>"
echo "========================================"
