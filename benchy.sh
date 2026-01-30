#!/usr/bin/env bash
set -e

# Create a temporary directory for the venv
VENV_DIR=$(mktemp -d)

# Create and activate the venv
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip quietly
pip install --upgrade pip >/dev/null

# Install your local llama-benchy in editable mode
cd ~/repos/others/llama-benchy

pip install -e . >/dev/null
pip install protobuf >/dev/null

# Keep it offline
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Run llama-benchy from source
python -m llama_benchy "$@"

# Deactivate and delete the venv
deactivate
rm -rf "$VENV_DIR"

