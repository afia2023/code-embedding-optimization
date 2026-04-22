#!/usr/bin/env bash
# ============================================================
# Auto-launcher: waits for vLLM to release all GPUs, then
# launches baseline (GPU 1) and compressed (GPU 2) optimized
# experiments in parallel.
#
# Usage:
#   bash scripts/wait_and_launch.sh
#   # or in background:
#   nohup bash scripts/wait_and_launch.sh > logs/wait_and_launch.log 2>&1 &
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

CHECK_INTERVAL=300   # check every 5 minutes
MAX_WAIT=86400       # give up after 24 hours

echo "=============================================="
echo "Auto-launcher started: $(date)"
echo "Waiting for vLLM to release GPUs..."
echo "Will check every ${CHECK_INTERVAL}s (max 24h)"
echo "Logs will be written to: $LOG_DIR"
echo "=============================================="

elapsed=0
while true; do
    if ! nvidia-smi | grep -q "VLLM"; then
        echo ""
        echo "=============================================="
        echo "GPUs are FREE: $(date)"
        echo "Launching both experiments in parallel..."
        echo "=============================================="
        break
    fi

    # Show current memory usage
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -2 | awk '{sum += $1} END {print sum}')
    echo "$(date): vLLM still running. Free memory across GPU1+GPU2: ${free_mem} MiB. Next check in ${CHECK_INTERVAL}s..."

    sleep "$CHECK_INTERVAL"
    elapsed=$((elapsed + CHECK_INTERVAL))

    if [ "$elapsed" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Waited 24 hours, giving up. Run experiments manually."
        exit 1
    fi
done

# Brief pause to let GPU memory fully release
sleep 10

echo ""
echo "--- Launching: baseline d512 on GPU 1 ---"
nohup bash "$SCRIPT_DIR/run_baseline_gpu1_optimized.sh" \
    > "$LOG_DIR/baseline_d512_optimized.log" 2>&1 &
BASELINE_PID=$!
echo "Baseline PID: $BASELINE_PID  → $LOG_DIR/baseline_d512_optimized.log"

echo "--- Launching: compressed d320 on GPU 2 ---"
nohup bash "$SCRIPT_DIR/run_compressed_gpu2_optimized.sh" \
    > "$LOG_DIR/compressed_d320_optimized.log" 2>&1 &
COMPRESSED_PID=$!
echo "Compressed PID: $COMPRESSED_PID  → $LOG_DIR/compressed_d320_optimized.log"

echo ""
echo "Both experiments launched. Monitor with:"
echo "  tail -f $LOG_DIR/baseline_d512_optimized.log"
echo "  tail -f $LOG_DIR/compressed_d320_optimized.log"
echo ""
echo "Or check GPU usage:"
echo "  watch -n 30 nvidia-smi"

# Wait for both to finish and report
wait "$BASELINE_PID" && echo "Baseline DONE: $(date)" || echo "Baseline FAILED: $(date)"
wait "$COMPRESSED_PID" && echo "Compressed DONE: $(date)" || echo "Compressed FAILED: $(date)"

echo ""
echo "=============================================="
echo "All experiments finished: $(date)"
echo "=============================================="
