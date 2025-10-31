#!/usr/bin/env python3
"""Select keyframes from BLIP scores and save frames/plots."""

import argparse
import heapq
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select keyframes using the AKS algorithm")
    parser.add_argument("--video_dir", type=str, default="./videos", help="Directory containing original MP4 videos")
    parser.add_argument("--scores_dir", type=str, default="./outscores", help="Directory containing *_scores.json files")
    parser.add_argument("--output_dir", type=str, default="./selected_frames", help="Directory to store extracted frames and plots")
    parser.add_argument("--max_num_frames", type=int, default=64, help="Maximum number of keyframes to select")
    parser.add_argument("--ratio", type=int, default=1, help="Optional down-sampling ratio for scores list")
    parser.add_argument("--t1", type=float, default=0.8, help="AKS threshold t1")
    parser.add_argument("--t2", type=float, default=-100, help="AKS threshold t2")
    parser.add_argument("--all_depth", type=int, default=5, help="AKS recursion depth")
    return parser.parse_args()


def meanstd(len_scores, dic_scores, n, fns, t1, t2, all_depth):
    split_scores = []
    split_fn = []
    no_split_scores = []
    no_split_fn = []
    for dic_score, fn in zip(dic_scores, fns):
        score = dic_score["score"]
        depth = dic_score["depth"]
        mean = float(np.mean(score))
        std = float(np.std(score))

        top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
        top_score = [score[t] for t in top_n]
        mean_diff = float(np.mean(top_score) - mean)
        if mean_diff > t1 and std > t2:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
        elif depth < all_depth:
            mid = len(score) // 2
            score1 = score[:mid]
            score2 = score[mid:]
            fn1 = fn[:mid]
            fn2 = fn[mid:]
            split_scores.append(dict(score=score1, depth=depth + 1))
            split_scores.append(dict(score=score2, depth=depth + 1))
            split_fn.append(fn1)
            split_fn.append(fn2)
        else:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
    if split_scores:
        all_split_score, all_split_fn = meanstd(len_scores, split_scores, n, split_fn, t1, t2, all_depth)
    else:
        all_split_score = []
        all_split_fn = []
    all_split_score = no_split_scores + all_split_score
    all_split_fn = no_split_fn + all_split_fn
    return all_split_score, all_split_fn


def downsample(values: List[float], indices: List[int], ratio: int) -> Tuple[List[float], List[int]]:
    if ratio <= 1:
        return values, indices
    step_count = max(len(values) // ratio, 0)
    sampled_vals = [values[i * ratio] for i in range(step_count)]
    sampled_idx = [indices[i * ratio] for i in range(step_count)]
    return sampled_vals, sampled_idx


def normalize_scores(scores: List[float]) -> List[float]:
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return []
    min_val = float(arr.min())
    max_val = float(arr.max())
    if np.isclose(max_val, min_val):
        return [0.0 for _ in scores]
    normalized = (arr - min_val) / (max_val - min_val)
    return normalized.tolist()


def select_keyframes(scores: List[float], frame_indices: List[int], args: argparse.Namespace) -> List[int]:
    if not scores:
        return []

    sampled_scores, sampled_indices = downsample(scores, frame_indices, args.ratio)

    if len(sampled_scores) >= args.max_num_frames:
        normalized = normalize_scores(sampled_scores)
        selected_segments, selected_frame_lists = meanstd(
            len(sampled_scores),
            [dict(score=normalized, depth=0)],
            args.max_num_frames,
            [sampled_indices],
            args.t1,
            args.t2,
            args.all_depth,
        )

        selected_original_indices: List[int] = []
        for segment, frame_list in zip(selected_segments, selected_frame_lists):
            split_depth = segment["depth"]
            split_scores = segment["score"]
            take_num = max(int(args.max_num_frames / (2 ** split_depth)), 1)
            topk = heapq.nlargest(take_num, range(len(split_scores)), split_scores.__getitem__)
            selected_original_indices.extend(frame_list[idx] for idx in topk)
    else:
        selected_original_indices = list(sampled_indices)

    selected_unique = sorted(set(int(idx) for idx in selected_original_indices))
    return selected_unique


def extract_and_save_frames(video_path: Path, frame_indices: List[int], output_dir: Path) -> List[int]:
    if not frame_indices:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    saved: List[int] = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_file = output_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_file), frame)
        saved.append(frame_idx)

    cap.release()
    return saved


def load_frame_images(video_path: Path, frame_indices: List[int]) -> Dict[int, np.ndarray]:
    images: Dict[int, np.ndarray] = {}
    if not frame_indices:
        return images

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images[int(frame_idx)] = frame_rgb

    cap.release()
    return images


def plot_scores(
    frame_indices: List[int],
    scores: List[float],
    selected_indices: List[int],
    top_frame_data: List[Tuple[int, np.ndarray]],
    output_path: Path,
    title: str,
    elapsed_seconds: float,
) -> None:
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 3], hspace=0.25)

    thumbs_ax = fig.add_subplot(gs[0, 0])
    thumbs_ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(frame_indices, scores, "b-", alpha=0.7, label="Frame scores")

    if selected_indices:
        score_lookup = {idx: score for idx, score in zip(frame_indices, scores)}
        selected_scores = [score_lookup.get(idx, np.nan) for idx in selected_indices]
        ax.scatter(selected_indices, selected_scores, color="red", s=50, zorder=5, label="Selected keyframes")

    if top_frame_data:
        score_lookup = {idx: score for idx, score in zip(frame_indices, scores)}
        top_indices = [idx for idx, _ in top_frame_data]
        top_scores = [score_lookup.get(idx, np.nan) for idx in top_indices]
        ax.scatter(
            top_indices,
            top_scores,
            color="gold",
            edgecolors="black",
            marker="^",
            s=80,
            zorder=6,
            label="Top-5 scores",
        )

        n = len(top_frame_data)
        margin = 0.02
        slot_width = (1.0 - margin * (n + 1)) / max(n, 1)
        slot_height = 0.85
        bottom = (1.0 - slot_height) / 2

        for i, (idx, img) in enumerate(top_frame_data):
            left = margin + i * (slot_width + margin)
            inset = thumbs_ax.inset_axes([left, bottom, slot_width, slot_height])
            inset.imshow(img)
            inset.set_title(f"idx {idx}", fontsize=8)
            inset.axis("off")

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Importance score")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.text(
        0.99,
        0.01,
        f"Selection time: {elapsed_seconds:.2f}s",
        ha="right",
        va="bottom",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_scores_file(score_file: Path, args: argparse.Namespace) -> None:
    with score_file.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    video_name = data.get("video_name")
    frame_indices = data.get("frame_indices", [])
    scores = data.get("itc_scores", [])

    base_name = Path(video_name).stem if video_name else score_file.stem.replace("_scores", "")
    video_path = Path(args.video_dir) / video_name
    if not video_path.exists():
        print(f"Warning: Skipping {score_file.name} because video {video_path} was not found")
        return

    output_dir = Path(args.output_dir) / base_name
    output_dir.mkdir(parents=True, exist_ok=True)

    top_pairs = sorted(zip(frame_indices, scores), key=lambda pair: pair[1], reverse=True)[:5]
    top_score_indices = [int(idx) for idx, _ in top_pairs]
    top_images_map = load_frame_images(video_path, top_score_indices)
    top_frame_data = [(idx, top_images_map[idx]) for idx in top_score_indices if idx in top_images_map]

    start_time = time.perf_counter()
    selected_indices = select_keyframes(scores, frame_indices, args)
    saved_frames = extract_and_save_frames(video_path, selected_indices, output_dir)
    elapsed = time.perf_counter() - start_time

    summary_path = output_dir / "selected_frames.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "video_name": video_name,
                "scores_file": score_file.name,
                "selected_frame_indices": selected_indices,
                "saved_frames": saved_frames,
                "top_score_frame_indices": top_score_indices,
                "selection_time_seconds": elapsed,
            },
            fp,
            indent=2,
        )

    plot_path = output_dir / "keyframe_plot.png"
    plot_scores(
        frame_indices,
        scores,
        selected_indices,
        top_frame_data,
        plot_path,
        f"AKS Keyframe Selection for {video_name}",
        elapsed,
    )

    print(f"Processed {video_name}: selected {len(saved_frames)} frame(s)")
    print(f"Saved frames and plot under {output_dir}")


def main() -> None:
    args = parse_arguments()

    scores_dir = Path(args.scores_dir)
    if not scores_dir.exists():
        raise FileNotFoundError(f"Scores directory does not exist: {scores_dir}")

    score_files = sorted(scores_dir.glob("*_scores.json"))
    if not score_files:
        print(f"No score JSON files found in {scores_dir.resolve()}")
        return

    print(f"Found {len(score_files)} score file(s) in {scores_dir.resolve()}")

    for score_file in score_files:
        try:
            process_scores_file(score_file, args)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to process {score_file.name}: {exc}")


if __name__ == "__main__":
    main()