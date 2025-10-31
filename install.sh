#!/bin/bash
# Installation script for AKS Keyframe Extraction

echo "Setting up AKS Keyframe Extraction environment..."

# Create virtual environment
echo "Creating virtual environment..."
uv venv

# Activate virtual environment and install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt

echo "Installation complete!"
echo "To activate the environment, run: .venv\Scripts\activate (Windows) or source .venv/bin/activate (Linux/Mac)"
echo ""
echo "To run feature extraction: uv run python feature_extract.py --video_dir ./videos --output_dir ./outscores"
echo "To run keyframe selection: uv run python frame_select.py --video_dir ./videos --scores_dir ./outscores --output_dir ./selected_frames"
