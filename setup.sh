#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Tiger Tracking System Setup ==="

if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

if [ ! -d ".venv" ]; then
    python3 -m venv .venv || {
        echo "Hint: install python3-venv: sudo apt install python3-venv"
        exit 1
    }
fi

source .venv/bin/activate
pip install -r requirements.txt

echo ""
echo "Downloading ML models (MegaDetector + MiewID)..."
python scripts/download_models.py || echo "Warning: model download failed — pipeline will use heuristic fallback"

echo ""
echo "Generating demo data..."
python scripts/seed_demo_data.py

echo ""
echo "Running pipeline..."
python scripts/run_pipeline.py -i data/raw/demo -s data/stations.csv

echo ""
echo "=== Setup Complete ==="
echo "Dashboard:  streamlit run dashboard.py"
echo "API:        uvicorn src.main:app --reload --port 8000"
echo "Smoke test: python scripts/smoke_test.py"
