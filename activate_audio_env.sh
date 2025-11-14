#!/bin/bash
set -e  # Exit on any error

VENV_PATH="/home/ness/FuseMachines/audio/speech_seperation_pytorch/audio_sep_pytorch"

echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

echo "Upgrading ipykernel..."
pip install --upgrade ipykernel

echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name audio_sep_pytorch --display-name "Audio sep (PyTorch)"

echo "VS Code kernel registered! Restart VS Code or reload window."