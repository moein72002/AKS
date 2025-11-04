"""GPU-resident API for video keyframe extraction using ONNX BLIP."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

from flask import Flask, jsonify, request

from feature_extract import (
    DEFAULT_PROMPT,
    MODEL_VARIANTS,
    BlipOnnxPipeline,
    detect_default_providers,
    extract_scores_for_video,
    sample_stride,
    save_scores,
)
from frame_select import (
    extract_and_save_frames,
    load_frame_images,
    plot_scores,
    select_keyframes,
)


app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

_PIPELINE_REGISTRY: Dict[str, BlipOnnxPipeline] = {}
_PIPELINE_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()


def _resolve_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _get_pipeline(variant: str) -> BlipOnnxPipeline:
    variant = variant.lower()
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model_variant '{variant}'. Expected one of {tuple(MODEL_VARIANTS.keys())}.")

    with _PIPELINE_LOCK:
        pipeline = _PIPELINE_REGISTRY.get(variant)
        if pipeline is None:
            config = MODEL_VARIANTS[variant]
            providers = detect_default_providers()
            pipeline = BlipOnnxPipeline(
                model_path=config["onnx_model"],
                metadata_path=config["metadata"],
                tokenizer_dir=config["tokenizer"],
                providers=providers,
            )
            _PIPELINE_REGISTRY[variant] = pipeline
    return pipeline


def _prepare_selection_args(payload: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        ratio=int(payload.get("ratio", 1)),
        max_num_frames=int(payload.get("max_num_frames", 64)),
        t1=float(payload.get("t1", 0.8)),
        t2=float(payload.get("t2", -100.0)),
        all_depth=int(payload.get("all_depth", 5)),
    )


def _build_summary(
    video_path: Path,
    frame_indices: list[int],
    scores: list[float],
    selected_indices: list[int],
    saved_frames: list[int],
    output_dir: Path,
    elapsed_selection: float,
) -> Tuple[Path, Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    top_pairs = sorted(zip(frame_indices, scores), key=lambda pair: pair[1], reverse=True)[:5]
    top_indices = [int(idx) for idx, _ in top_pairs]
    top_images_map = load_frame_images(video_path, top_indices)
    top_frame_data = [(idx, top_images_map[idx]) for idx in top_indices if idx in top_images_map]

    summary_path = output_dir / "selected_frames.json"
    summary_payload = {
        "video_name": video_path.name,
        "selected_frame_indices": selected_indices,
        "saved_frames": saved_frames,
        "top_score_frame_indices": top_indices,
        "selection_time_seconds": elapsed_selection,
    }
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2)

    plot_path = output_dir / "keyframe_plot.png"
    plot_scores(
        frame_indices,
        scores,
        selected_indices,
        top_frame_data,
        plot_path,
        f"AKS Keyframe Selection for {video_path.name}",
        elapsed_selection,
    )

    return summary_path, {
        "summary_path": str(summary_path.resolve()),
        "plot_path": str(plot_path.resolve()),
        "saved_frames": [str(output_dir / f"frame_{idx:06d}.jpg") for idx in saved_frames],
        "top_score_frame_indices": top_indices,
        "selection_time_seconds": elapsed_selection,
    }


@app.route("/healthz", methods=["GET"])
def healthcheck() -> Any:
    return jsonify({"status": "ok", "loaded_variants": list(_PIPELINE_REGISTRY.keys())})


@app.route("/process", methods=["POST"])
def process_video() -> Any:
    try:
        payload = request.get_json(force=True) or {}
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "message": f"Invalid JSON payload: {exc}"}), 400

    video_path_raw = payload.get("video_path")
    if not video_path_raw:
        return jsonify({"status": "error", "message": "'video_path' is required"}), 400

    video_path = Path(video_path_raw)
    if not video_path.exists():
        return jsonify({"status": "error", "message": f"Video not found: {video_path}"}), 404

    model_variant = payload.get("model_variant", "base").lower()
    prompt = payload.get("prompt", DEFAULT_PROMPT)
    scores_dir = Path(payload.get("scores_dir", "outscores"))
    output_dir = Path(payload.get("output_dir", "selected_frames"))
    save_scores_flag = _resolve_bool(payload.get("save_scores"), True)

    try:
        pipeline = _get_pipeline(model_variant)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    timings: Dict[str, float] = {}

    start_overall = time.perf_counter()

    text_start = time.perf_counter()
    text_inputs = pipeline.preprocess_text(prompt)
    timings["text_preprocess_seconds"] = time.perf_counter() - text_start

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return jsonify({"status": "error", "message": f"Failed to open video: {video_path}"}), 500
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    stride = sample_stride(fps)

    with _RUN_LOCK:
        inference_start = time.perf_counter()
        frame_indices, scores = extract_scores_for_video(pipeline, text_inputs, video_path, stride)
        timings["inference_seconds"] = time.perf_counter() - inference_start

    scores_record: Dict[str, Any] | None = None
    if save_scores_flag:
        scores_dir.mkdir(parents=True, exist_ok=True)
        save_scores(scores_dir, video_path.stem, video_path.name, frame_indices, scores)
        scores_record = {
            "scores_path": str((scores_dir / f"{video_path.stem}_scores.json").resolve()),
            "frame_indices": frame_indices,
        }

    selection_args = _prepare_selection_args(payload)
    selection_start = time.perf_counter()
    selected_indices = select_keyframes(scores, frame_indices, selection_args)
    output_subdir = output_dir / video_path.stem
    output_subdir.mkdir(parents=True, exist_ok=True)
    saved_frames = extract_and_save_frames(video_path, selected_indices, output_subdir)
    timings["selection_seconds"] = time.perf_counter() - selection_start

    summary_path, selection_payload = _build_summary(
        video_path,
        frame_indices,
        scores,
        selected_indices,
        saved_frames,
        output_subdir,
        timings["selection_seconds"],
    )

    timings["total_seconds"] = time.perf_counter() - start_overall

    response: Dict[str, Any] = {
        "status": "ok",
        "video": str(video_path.resolve()),
        "model_variant": model_variant,
        "prompt": prompt,
        "stride": stride,
        "num_frames_scored": len(frame_indices),
        "num_keyframes": len(saved_frames),
        "timings": timings,
        "selection": selection_payload,
    }
    if scores_record:
        response["scores"] = scores_record

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)

