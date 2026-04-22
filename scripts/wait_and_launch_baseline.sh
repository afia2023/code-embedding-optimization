#!/usr/bin/env bash
# ============================================================
# Auto-launcher: waits for ANY GPU to be fully empty,
# then launches baseline d512 with batch=64 on that GPU.
#
# batch=64 needs ~20 GB — requires a fully free GPU (no vLLM).
# Checks all 4 GPUs every 2 minutes.
#
# Usage:
#   nohup bash scripts/wait_and_launch_baseline.sh > logs/wait_baseline.log 2>&1 &
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

MIN_FREE_MiB=30000   # need ~30 GB free — batch=64 uses ~20 GB, 10 GB headroom
CHECK_INTERVAL=120   # check every 2 minutes
MAX_WAIT=86400       # give up after 24 hours

echo "=============================================="
echo "Baseline auto-launcher started: $(date)"
echo "Waiting for any GPU with ${MIN_FREE_MiB} MiB free (30 GB)..."
echo "Will check every ${CHECK_INTERVAL}s"
echo "=============================================="

elapsed=0
TARGET_GPU=-1

while true; do
    # Check each GPU for free memory
    for gpu_id in 0 1 2 3; do
        # Skip GPU 2 — d320 compressed is running there
        if [ "$gpu_id" -eq 2 ]; then
            continue
        fi

        free_mib=$(nvidia-smi --id=$gpu_id --query-gpu=memory.free --format=csv,noheader,nounits)

        if [ "$free_mib" -ge "$MIN_FREE_MiB" ]; then
            TARGET_GPU=$gpu_id
            echo ""
            echo "=============================================="
            echo "GPU ${TARGET_GPU} is FREE (${free_mib} MiB free): $(date)"
            echo "Launching baseline d512 with batch=64, LR=1e-3..."
            echo "=============================================="
            break 2
        fi
    done

    # Print current free memory on all GPUs
    free_all=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk '{printf "GPU%s:%sMiB  ", $1, $2}')
    echo "$(date): $free_all — waiting..."

    sleep "$CHECK_INTERVAL"
    elapsed=$((elapsed + CHECK_INTERVAL))

    if [ "$elapsed" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Waited 24 hours, giving up."
        exit 1
    fi
done

# Brief pause for memory to fully release
sleep 10

echo "Launching on GPU ${TARGET_GPU}..."
CUDA_VISIBLE_DEVICES=$TARGET_GPU python -m code_tasks.cli \
    --task summarization \
    --language java \
    --model-name-or-path t5-small \
    --dataset codexglue_code_to_text \
    --output-dir /scratch/afia/code-embedding-optimization-svd/outputs/summarization_baseline_t5small_d512_optimized \
    --overwrite-output-dir \
    --do-train --do-eval --do-test \
    --device cuda:0 \
    --hidden-dim 512 \
    --init-method random \
    --num-train-epochs 15 \
    --per-device-train-batch-size 64 \
    --per-device-eval-batch-size 32 \
    --gradient-accumulation-steps 1 \
    --learning-rate 1e-3 \
    --warmup-ratio 0.1 \
    --weight-decay 0.01 \
    --preprocessing-num-workers 4 \
    --dataloader-num-workers 4 \
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
    2>&1 | tee "$LOG_DIR/baseline_d512_optimized.log"

echo "Baseline DONE: $(date)"
