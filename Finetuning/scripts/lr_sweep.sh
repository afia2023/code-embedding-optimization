#!/usr/bin/env bash
# ============================================================
# LR Sweep — find best learning rate before full training
# Tests: 1e-4, 3e-4, 5e-4 on both d_model=512 and d_model=322
# Uses 5k train / 1k eval samples, 3 epochs — fast (~10-15 min each)
# Run on GPU 1 (baseline model) or GPU 2 (compressed model)
#
# Usage:
#   bash scripts/lr_sweep.sh 1      # run on GPU 1, d_model=512
#   bash scripts/lr_sweep.sh 2      # run on GPU 2, d_model=322
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

RESULTS_FILE="$SWEEP_DIR/lr_sweep_results.txt"
echo "LR Sweep: model=${LABEL}, hidden_dim=${HIDDEN_DIM}" | tee "$RESULTS_FILE"
echo "LR          | BLEU  | eval_loss" | tee -a "$RESULTS_FILE"
echo "------------|-------|----------" | tee -a "$RESULTS_FILE"

for LR in 1e-4 3e-4 5e-4; do
    RUN_DIR="$SWEEP_DIR/lr_${LR}"
    echo ""
    echo ">>> Testing LR=${LR}, hidden_dim=${HIDDEN_DIM} ..."

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

    # Extract best BLEU and eval loss from trainer state
    BLEU=$(python3 -c "
import json, pathlib, sys
f = pathlib.Path('${RUN_DIR}/trainer_state.json')
if not f.exists():
    print('N/A')
    sys.exit()
data = json.loads(f.read_text())
best = data.get('best_metric', None)
print(f'{best:.4f}' if best else 'N/A')
" 2>/dev/null || echo "N/A")

    EVAL_LOSS=$(python3 -c "
import json, pathlib, sys
f = pathlib.Path('${RUN_DIR}/validation_results.json')
if not f.exists():
    print('N/A')
    sys.exit()
data = json.loads(f.read_text())
loss = data.get('validation_loss', None)
print(f'{loss:.4f}' if loss else 'N/A')
" 2>/dev/null || echo "N/A")

    printf "%-12s| %-6s| %s\n" "$LR" "$BLEU" "$EVAL_LOSS" | tee -a "$RESULTS_FILE"
done

echo ""
echo "========================================"
echo "LR sweep complete. Results saved to:"
echo "  $RESULTS_FILE"
echo ""
cat "$RESULTS_FILE"
echo ""
echo "Pick the LR with highest BLEU and update run_baseline_gpu1.sh / run_compressed_gpu2.sh before full training."
