#!/bin/bash

# Auto-launcher for batch=32 and batch=64 smoke tests
# Waits for free GPUs and launches experiments automatically

cd /scratch/afia/code-embedding-optimization-svd
source afiabase/bin/activate

# Experiments to run: "name|hidden_dim|batch|lr|gpu_assigned"
declare -A LAUNCHED
LAUNCHED[0]=0
LAUNCHED[1]=0
LAUNCHED[2]=0
LAUNCHED[3]=0

EXPERIMENTS=(
    "d512_batch32|512|32|5e-4"
    "d320_batch32|320|32|5e-4"
    "d512_batch64|512|64|7.07e-4"
    "d320_batch64|320|64|7.07e-4"
)

EXP_INDEX=0

is_gpu_free() {
    local gpu=$1
    local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu | tr -d ' ')
    if [ "$mem_used" -lt 2000 ]; then
        return 0  # free
    else
        return 1  # busy
    fi
}

launch_experiment() {
    local gpu=$1
    local exp=$2
    IFS='|' read -r name hidden_dim batch lr <<< "$exp"

    local outdir="outputs/sweep_${name}_lr${lr}_bf16"
    local logfile="logs/sweep_${name}_lr${lr}_bf16.log"

    echo "[$(date '+%H:%M:%S')] Launching $name on GPU $gpu (LR=$lr, batch=$batch)"

    CUDA_VISIBLE_DEVICES=$gpu nohup python -m code_tasks.cli \
        --task summarization --language java \
        --output-dir "$outdir" \
        --hidden-dim "$hidden_dim" --init-method random \
        --do-train --do-eval \
        --per-device-train-batch-size "$batch" --per-device-eval-batch-size 16 \
        --learning-rate "$lr" --num-train-epochs 3 \
        --warmup-ratio 0.2 --weight-decay 0.01 \
        --max-grad-norm 1.0 \
        --max-input-length 512 --max-target-length 128 \
        --num-beams 4 --seed 42 --dataloader-num-workers 4 \
        --max-train-samples 500 --max-eval-samples 200 \
        --bf16 --overwrite-output-dir > "$logfile" 2>&1 &

    echo "[$(date '+%H:%M:%S')] $name PID: $!"
}

echo "=========================================="
echo " Smoke test auto-launcher started"
echo " Waiting for free GPUs..."
echo " Experiments queued: ${#EXPERIMENTS[@]}"
echo "=========================================="

while [ $EXP_INDEX -lt ${#EXPERIMENTS[@]} ]; do
    for gpu in 0 1 2 3; do
        if [ $EXP_INDEX -ge ${#EXPERIMENTS[@]} ]; then
            break
        fi
        if is_gpu_free $gpu; then
            launch_experiment $gpu "${EXPERIMENTS[$EXP_INDEX]}"
            EXP_INDEX=$((EXP_INDEX + 1))
            sleep 10  # small delay before checking next GPU
        fi
    done

    if [ $EXP_INDEX -lt ${#EXPERIMENTS[@]} ]; then
        echo "[$(date '+%H:%M:%S')] ${#EXPERIMENTS[@]} experiments total, $EXP_INDEX launched. Waiting 60s..."
        sleep 60
    fi
done

echo "[$(date '+%H:%M:%S')] All 4 experiments launched. Monitor with:"
echo "tail -f logs/sweep_d512_batch32_lr5e-4_bf16.log logs/sweep_d512_batch64_lr7.07e-4_bf16.log logs/sweep_d320_batch32_lr5e-4_bf16.log logs/sweep_d320_batch64_lr7.07e-4_bf16.log"
