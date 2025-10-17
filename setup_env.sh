#!/bin/bash
set -e  # Exit on any error

echo "=== Updating system packages ==="
sudo apt-get update

echo "=== Installing system dependencies ==="
sudo apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    libdmtx0t64 libdmtx-dev \
    libgl1 libglib2.0-0 \
    libglib2.0-0

echo "=== Creating virtual environment ==="
python3.11 -m venv .venv

echo "=== Activating virtual environment ==="
source .venv/bin/activate

echo "=== Upgrading pip, setuptools, wheel ==="
pip install --upgrade pip setuptools wheel

echo "=== Installing Python dependencies ==="
pip install --force-reinstall \
    pylibdmtx \
    pillow \
    pandas \
    opencv-python-headless \
    numpy

echo "=== Verifying installation ==="
python - <<EOF
from pylibdmtx import pylibdmtx
import cv2
import pandas as pd
print("✅ Environment setup complete!")
EOF
