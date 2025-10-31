#!/usr/bin/env python3
"""Extract BLIP ITM scores for MP4 videos."""

import argparse
import json
from pathlib import Path

import cv2
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess


DEFAULT_PROMPT = "ساعت مچی طرح خودرو مردانه"

try:
    RESIZE_RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9.1
    RESIZE_RESAMPLE = Image.BILINEAR

TARGET_SIZE = (256, 256)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract BLIP ITM scores for MP4 videos")
    parser.add_argument("--video_dir", type=str, default="./videos", help="Directory containing MP4 videos")
    parser.add_argument("--output_dir", type=str, default="./outscores", help="Directory to save per-video score JSON files")
    parser.add_argument("--device", type=str, default="auto", help="Device to use: auto, cuda, or cpu")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Text prompt for BLIP image-text matching")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            print("Using CUDA GPU for processing")
            return "cuda"
        print("CUDA not available, using CPU for processing")
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable, falling back to CPU")
        return "cpu"
    return requested


def load_videos(video_dir: Path) -> list[Path]:
    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        print(f"No MP4 files found in {video_dir.resolve()}")
    else:
        print(f"Found {len(videos)} MP4 file(s) in {video_dir.resolve()}")
    return videos


def sample_stride(fps: float) -> int:
    if fps and fps > 0:
        stride = max(int(round(fps)), 1)
    else:
        stride = 30
        print("Warning: FPS reported as 0. Defaulting to 30 FPS for sampling stride")
    return stride


def extract_scores_for_video(
    model,
    vis_processors,
    text_processed,
    device: str,
    video_path: Path,
    stride: int,
) -> tuple[list[int], list[float]]:
    print(f"Processing {video_path.name} (stride={stride})")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    frame_indices: list[int] = []
    scores: list[float] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            if TARGET_SIZE is not None:
                pil_image = pil_image.resize(TARGET_SIZE, RESIZE_RESAMPLE)
            img = vis_processors["eval"](pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model({"image": img, "text_input": text_processed}, match_head="itm")
            blip_scores = torch.nn.functional.softmax(output, dim=1)
            scores.append(float(blip_scores[:, 1].item()))
            frame_indices.append(frame_idx)

        frame_idx += 1

    cap.release()
    print(f"Sampled {len(frame_indices)} frame(s) from {video_path.name}")
    return frame_indices, scores


def save_scores(output_dir: Path, base_name: str, video_name: str, frame_indices: list[int], scores: list[float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{base_name}_scores.json"
    payload = {
        "video_name": video_name,
        "frame_indices": frame_indices,
        "itc_scores": scores,
    }
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    print(f"Saved scores to {output_path}")


def main() -> None:
    args = parse_arguments()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model, vis_processors, text_processors = load_model_and_preprocess(
        "blip_image_text_matching", "large", device=device, is_eval=True
    )
    text_processed = text_processors["eval"](args.prompt)

    videos = load_videos(video_dir)
    for video_path in videos:
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            stride = sample_stride(fps)

            frame_indices, scores = extract_scores_for_video(
                model, vis_processors, text_processed, device, video_path, stride
            )
            save_scores(output_dir, video_path.stem, video_path.name, frame_indices, scores)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to process {video_path.name}: {exc}")


if __name__ == "__main__":
    main()