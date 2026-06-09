#!/usr/bin/env bash
# ============================================================
# VM Setup Script — Run once after cloning on a fresh VM
# Installs all dependencies and verifies GPU availability
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "========================================"
echo "Setting up environment..."
echo "========================================"

# Create virtualenv
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install the package itself
pip install -e .

echo ""
echo "========================================"
echo "Verifying GPU..."
echo "========================================"
python3 -c "
import torch
print(f'PyTorch version : {torch.__version__}')
print(f'CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU             : {torch.cuda.get_device_name(0)}')
    print(f'VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "========================================"
echo "Setup complete. Run sweep with:"
echo "  CUDA_VISIBLE_DEVICES=0 bash scripts/lr_sweep_all_batchsizes.sh"
echo "========================================"
