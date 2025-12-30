#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed"
    exit 1
fi

[ ! -d "$VENV_DIR" ] && python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip --quiet
pip install -r "$SCRIPT_DIR/requirements.txt" --quiet

python "$SCRIPT_DIR/protein_splitter.py"

