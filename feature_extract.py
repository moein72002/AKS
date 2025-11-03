#!/usr/bin/env python3
"""Extract BLIP ITM scores for MP4 videos using an exported ONNX model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoTokenizer


DEFAULT_PROMPT = "What is the advertised product?"
DEFAULT_ONNX_MODEL = "./blip_itm_large_onnx/blip_itm_large.onnx"
DEFAULT_METADATA_JSON = "./blip_itm_large_onnx/preprocessing_metadata.json"
DEFAULT_TOKENIZER_DIR = "./blip_itm_large_onnx/tokenizer"

try:
    RESIZE_RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9.1
    RESIZE_RESAMPLE = Image.BILINEAR


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract BLIP ITM scores for MP4 videos (ONNX)")
    parser.add_argument("--video_dir", type=str, default="./videos", help="Directory containing MP4 videos")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outscores",
        help="Directory to save per-video score JSON files",
    )
    parser.add_argument(
        "--onnx_model",
        type=str,
        default=DEFAULT_ONNX_MODEL,
        help="Path to the exported BLIP ITM ONNX model",
    )
    parser.add_argument(
        "--metadata_json",
        type=str,
        default=DEFAULT_METADATA_JSON,
        help="Path to preprocessing metadata JSON saved alongside the ONNX model",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=DEFAULT_TOKENIZER_DIR,
        help="Directory containing the tokenizer assets saved with the ONNX export",
    )
    parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        default=["CPUExecutionProvider"],
        help="ONNX Runtime execution providers to use (default: CPUExecutionProvider)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Text prompt for BLIP image-text matching",
    )
    return parser.parse_args()


def load_videos(video_dir: Path) -> List[Path]:
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


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits - max_logits)
    denom = np.sum(exp, axis=-1, keepdims=True)
    return exp / np.clip(denom, a_min=1e-9, a_max=None)


def _ensure_hw(size: Sequence[int] | None) -> Tuple[int, int] | None:
    if size is None:
        return None
    if len(size) != 2:
        return None
    height, width = int(size[0]), int(size[1])
    return height, width


@dataclass
class PreprocessConfig:
    image_height: int
    image_width: int
    max_txt_len: int
    resize_hw: Tuple[int, int] | None
    center_crop_hw: Tuple[int, int] | None
    normalize_mean: np.ndarray
    normalize_std: np.ndarray


class BlipOnnxPipeline:
    """Utility helper to run BLIP ITM inference with ONNX Runtime."""

    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        tokenizer_dir: Path,
        providers: Sequence[str] | None = None,
        provider_options: Sequence[Dict[str, str]] | None = None,
    ) -> None:
        self.session = ort.InferenceSession(
            str(model_path),
            providers=list(providers) if providers else ["CPUExecutionProvider"],
            provider_options=list(provider_options) if provider_options else None,
        )

        with metadata_path.open("r", encoding="utf-8") as fp:
            metadata = json.load(fp)

        preprocess = metadata.get("preprocess", {})
        resize_hw = _ensure_hw(preprocess.get("resize"))
        center_crop_hw = _ensure_hw(preprocess.get("center_crop"))

        normalize_mean = np.asarray(
            preprocess.get(
                "normalize_mean",
                [0.48145466, 0.4578275, 0.40821073],
            ),
            dtype=np.float32,
        )
        normalize_std = np.asarray(
            preprocess.get(
                "normalize_std",
                [0.26862954, 0.26130258, 0.27577711],
            ),
            dtype=np.float32,
        )

        self.config = PreprocessConfig(
            image_height=int(metadata.get("image_height", 384)),
            image_width=int(metadata.get("image_width", 384)),
            max_txt_len=int(metadata.get("max_txt_len", 32)),
            resize_hw=resize_hw,
            center_crop_hw=center_crop_hw,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB")

        if self.config.resize_hw is not None:
            height, width = self.config.resize_hw
            img = img.resize((width, height), RESIZE_RESAMPLE)

        if self.config.center_crop_hw is not None:
            crop_h, crop_w = self.config.center_crop_hw
            img = self._center_crop(img, crop_h, crop_w)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        mean = self.config.normalize_mean.reshape(3, 1, 1)
        std = self.config.normalize_std.reshape(3, 1, 1)
        arr = (arr - mean) / std
        return np.expand_dims(arr.astype(np.float32), axis=0)

    @staticmethod
    def _center_crop(image: Image.Image, crop_h: int, crop_w: int) -> Image.Image:
        width, height = image.size
        left = max((width - crop_w) // 2, 0)
        top = max((height - crop_h) // 2, 0)
        right = left + crop_w
        bottom = top + crop_h
        return image.crop((left, top, right, bottom))

    def preprocess_text(self, prompt: str) -> Dict[str, np.ndarray]:
        encoded = self.tokenizer(
            [prompt],
            padding="max_length",
            truncation=True,
            max_length=self.config.max_txt_len,
            return_tensors="np",
        )
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _session_inputs(self, pixel_values: np.ndarray, text_inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        inputs: Dict[str, np.ndarray] = {
            "pixel_values": np.ascontiguousarray(pixel_values.astype(np.float32)),
            "input_ids": np.ascontiguousarray(text_inputs["input_ids"].astype(np.int64)),
            "attention_mask": np.ascontiguousarray(text_inputs["attention_mask"].astype(np.int64)),
        }
        return inputs

    def run(self, pixel_values: np.ndarray, text_inputs: Dict[str, np.ndarray]) -> np.ndarray:
        outputs = self.session.run(None, self._session_inputs(pixel_values, text_inputs))
        return outputs[0]

    def predict_score(self, image: Image.Image, text_inputs: Dict[str, np.ndarray]) -> float:
        pixel_values = self.preprocess_image(image)
        logits = self.run(pixel_values, text_inputs)[0]
        probs = softmax(logits)
        return float(probs[1])


def extract_scores_for_video(
    pipeline: BlipOnnxPipeline,
    text_inputs: Dict[str, np.ndarray],
    video_path: Path,
    stride: int,
) -> Tuple[List[int], List[float]]:
    print(f"Processing {video_path.name} (stride={stride})")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    frame_indices: List[int] = []
    scores: List[float] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            score = pipeline.predict_score(pil_image, text_inputs)
            scores.append(score)
            frame_indices.append(frame_idx)

        frame_idx += 1

    cap.release()
    print(f"Sampled {len(frame_indices)} frame(s) from {video_path.name}")
    return frame_indices, scores


def save_scores(output_dir: Path, base_name: str, video_name: str, frame_indices: List[int], scores: List[float]) -> None:
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

    pipeline = BlipOnnxPipeline(
        model_path=Path(args.onnx_model),
        metadata_path=Path(args.metadata_json),
        tokenizer_dir=Path(args.tokenizer_dir),
        providers=args.providers,
    )
    text_inputs = pipeline.preprocess_text(args.prompt)

    videos = load_videos(video_dir)
    for video_path in videos:
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            stride = sample_stride(fps)

            frame_indices, scores = extract_scores_for_video(pipeline, text_inputs, video_path, stride)
            save_scores(output_dir, video_path.stem, video_path.name, frame_indices, scores)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to process {video_path.name}: {exc}")


if __name__ == "__main__":
    main()