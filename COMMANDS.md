## Quick Commands
- `python -m venv .venv` — create a virtual environment on Kaggle/Unix if `uv venv` isn’t available
- `.venv\Scripts\activate` — activate the venv on Windows PowerShell/CMD
- `source .venv/bin/activate` — activate the venv on WSL/Linux/macOS
- `uv sync` — install all dependencies (requirements + pyproject)
- `uv run python feature_extract.py --video_dir ./videos --output_dir ./outscores --device cpu` — extract BLIP frame scores for every MP4 in `videos/`
- `uv run python frame_select.py --video_dir ./videos --scores_dir ./outscores --output_dir ./selected_frames --max_num_frames 64` — pick keyframes, save JPEGs, and generate plots
