#!/bin/bash
set -e

# ────────────────────────────────────────────────────────────────────────────────────
# setup.sh — One-time setup for Piecewise Dynamic Diffusion Regularization (PDDR)
#
# Usage:
#   bash setup.sh /path/to/base_directory
#
# This script will:
#   1. Create the directory structure (datasets, models, experiments)
#   2. Install Python dependencies via pip
#   3. Install the BART MRI toolbox (skipped inside Docker)
#   4. Verify that critical tools are available
# ────────────────────────────────────────────────────────────────────────────────────

BART_VERSION="0.9.00"

# ── Parse arguments ──────────────────────────────────────────
if [ -z "$1" ]; then
    echo "Usage: bash setup.sh <base_path>"
    echo "  <base_path>  Root directory for datasets, models, and experiments."
    exit 1
fi

BASE_PATH="$(realpath "$1")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  PDDR Setup"
echo "============================================"
echo "Base path : $BASE_PATH"
echo "Repo path : $SCRIPT_DIR"
echo ""

# ── 1. Create directory structure ────────────────────────────
echo "[1/4] Creating directory structure under $BASE_PATH ..."

mkdir -p "$BASE_PATH/datasets"
mkdir -p "$BASE_PATH/models"
mkdir -p "$BASE_PATH/experiments"

echo "  Created: $BASE_PATH/datasets"
echo "  Created: $BASE_PATH/models"
echo "  Created: $BASE_PATH/experiments"
echo ""

# ── 2. Install Python requirements ──────────────────────────
echo "[2/4] Installing Python requirements ..."

pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""

# ── 3. Install BART ─────────────────────────────────────────
if [ -f /.dockerenv ] || grep -q 'docker\|containerd' /proc/1/cgroup 2>/dev/null; then
    echo "[3/4] Skipping BART installation (running inside Docker) ..."
    echo ""
else
    echo "[3/4] Installing BART toolbox v${BART_VERSION} ..."

    BART_DIR="$BASE_PATH/bart-${BART_VERSION}"

    if [ -x "$BART_DIR/bart" ]; then
        echo "  BART already installed at $BART_DIR — skipping."
    else
        # Install build dependencies
        sudo apt-get update -qq
        sudo apt-get install -y -qq wget make gcc libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev

        # Download and build
        wget -q -O "$BASE_PATH/bart-v${BART_VERSION}.tar.gz" \
            "https://github.com/mrirecon/bart/archive/v${BART_VERSION}.tar.gz"
        tar xzf "$BASE_PATH/bart-v${BART_VERSION}.tar.gz" -C "$BASE_PATH"
        rm "$BASE_PATH/bart-v${BART_VERSION}.tar.gz"
        make -C "$BART_DIR" -j"$(nproc)"

        echo "  BART built at: $BART_DIR"
    fi

    # Export env vars for this session
    export TOOLBOX_PATH="$BART_DIR"
    export PATH="${TOOLBOX_PATH}:${PATH}"
    export PYTHONPATH="${TOOLBOX_PATH}/python:${PYTHONPATH}"

    echo "  TOOLBOX_PATH=$TOOLBOX_PATH"
    echo ""
    echo "  Add the following to your shell profile (~/.bashrc) to persist:"
    echo "    export TOOLBOX_PATH=\"$BART_DIR\""
    echo "    export PATH=\"\$TOOLBOX_PATH:\$PATH\""
    echo "    export PYTHONPATH=\"\$TOOLBOX_PATH/python:\$PYTHONPATH\""
    echo ""
fi

# ── 4. Verify installation ──────────────────────────────────
echo "[4/4] Verifying installation ..."

python -c "import torch; print(f'  PyTorch {torch.__version__}  (CUDA available: {torch.cuda.is_available()})')"
python -c "import numpy; print(f'  NumPy   {numpy.__version__}')"
python -c "import einops; print(f'  einops  {einops.__version__}')"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Directory layout:"
echo "    $BASE_PATH/datasets     — place preprocessed data here"
echo "    $BASE_PATH/models       — trained model checkpoints"
echo "    $BASE_PATH/experiments  — reconstruction outputs & metrics"
echo ""
echo "  Next steps:"
echo "    1. Download datasets (see README.md for links)"
echo "    2. Preprocess data    (see preprocess/README.md)"
echo "    3. Update config paths in configs/*.yaml to point to the directories above."
echo "============================================"
