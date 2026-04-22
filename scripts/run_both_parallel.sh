#!/usr/bin/env bash
# ============================================================
# Launch BOTH experiments in parallel on separate GPUs
#   GPU 0 → Baseline (pre-trained, d_model=768)
#   GPU 1 → Reduced  (random init, d_model=384)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR/src"

echo "Starting parallel experiments at $(date '+%Y-%m-%d %H:%M:%S')"
echo "  GPU 0: Baseline (d_model=768, pre-trained)"
echo "  GPU 1: Reduced  (d_model=384, random init)"
echo ""

# Create output dirs upfront so tee doesn't fail
mkdir -p "$PROJECT_DIR/outputs/summarization_baseline_d768"
mkdir -p "$PROJECT_DIR/outputs/summarization_reduced_d384_random"

# Launch both in background
bash "$SCRIPT_DIR/run_baseline_gpu0.sh" &
PID_BASELINE=$!

bash "$SCRIPT_DIR/run_reduced_gpu1.sh" &
PID_REDUCED=$!

echo "PIDs: baseline=$PID_BASELINE, reduced=$PID_REDUCED"
echo "Waiting for both to finish..."

# Wait and capture exit codes
FAIL=0
wait $PID_BASELINE || FAIL=1
wait $PID_REDUCED  || FAIL=1

echo ""
echo "============================================"
echo "Both experiments finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# Print timing comparison
echo ""
echo "--- Timing Comparison ---"
if [ -f "$PROJECT_DIR/outputs/summarization_baseline_d768/timing.json" ]; then
    echo "BASELINE (d_model=768):"
    python3 -c "
import json
t = json.load(open('$PROJECT_DIR/outputs/summarization_baseline_d768/timing.json'))
for k, v in t.items():
    if isinstance(v, list):
        v = ', '.join(f'{x:.1f}s' for x in v)
    elif isinstance(v, (int, float)):
        v = f'{v:.1f}s'
    print(f'  {k}: {v}')
"
fi

echo ""
if [ -f "$PROJECT_DIR/outputs/summarization_reduced_d384_random/timing.json" ]; then
    echo "REDUCED (d_model=384, random):"
    python3 -c "
import json
t = json.load(open('$PROJECT_DIR/outputs/summarization_reduced_d384_random/timing.json'))
for k, v in t.items():
    if isinstance(v, list):
        v = ', '.join(f'{x:.1f}s' for x in v)
    elif isinstance(v, (int, float)):
        v = f'{v:.1f}s'
    print(f'  {k}: {v}')
"
fi

exit $FAIL
