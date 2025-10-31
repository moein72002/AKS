#!/usr/bin/env python3
"""Extract BLIP ITM scores for MP4 videos."""

import argparse
import json
from pathlib import Path

import cv2
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess


DEFAULT_PROMPT = "What is the product in the video?"


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
def main(args):
    if args.dataset_name =="longvideobench":
       label_path = os.path.join(args.dataset_path,'lvb_val.json')
       video_path = os.path.join(args.dataset_path,'videos')
    elif args.dataset_name =="videomme":
       label_path = os.path.join(args.dataset_path,'videomme.json')
       video_path = os.path.join(args.dataset_path,'data')
    else:
       raise ValueError("dataset_name: longvideobench or videomme")
    
    if os.path.exists(label_path):
        with open(label_path,'r') as f:
            datas = json.load(f)
    else:
        raise OSError("the label file does not exist")
    
    device = args.device
    
    if args.extract_feature_model == 'blip':
        model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)
    elif args.extract_feature_model == 'clip':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model.to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    elif args.extract_feature_model == 'sevila':
        model, vis_processors, text_processors = load_model_and_preprocess(name="sevila", model_type="pretrain_flant5xl", is_eval=True, device=device)
    else:
        raise ValueError("model not support")

    with open(label_path,'r') as f:
        datas = json.load(f)

    if not os.path.exists(os.path.join(args.output_file,args.dataset_name)):
        os.mkdir(os.path.join(args.output_file,args.dataset_name))
    out_score_path = os.path.join(args.output_file,args.dataset_name,args.extract_feature_model)
    if not os.path.exists(out_score_path):
        os.mkdir(out_score_path)
   

    scores = []
    fn = []
    score_path = os.path.join(out_score_path,'scores.json')
    frame_path = os.path.join(out_score_path,'frames.json')

    for data in datas:
        text = data['question']  

        if args.dataset_name == 'longvideobench':
            video = os.path.join(video_path, data["video_path"])
        else:
            video = os.path.join(video_path, data["videoID"]+'.mp4')
            
        duration = data['duration']
        vr = VideoReader(video, ctx=cpu(0), num_threads=1)
        fps = vr.get_avg_fps()
        frame_nums = int(len(vr)/int(fps))

        score = []
        frame_num = []

        if args.extract_feature_model == 'blip':
            txt = text_processors["eval"](text)
            for j in range(frame_nums):
                raw_image = np.array(vr[j*int(fps)])
                raw_image = Image.fromarray(raw_image)
                img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    blip_output = model({"image": img, "text_input": txt}, match_head="itm")               
                blip_scores = torch.nn.functional.softmax(blip_output, dim=1)
                score.append(blip_scores[:, 1].item())
                frame_num.append(j*int(fps))

        elif args.extract_feature_model == 'clip':
            inputs_text = processor(text=text, return_tensors="pt", padding=True,truncation=True).to(device)
            text_features = model.get_text_features(**inputs_text)
            for j in range(frame_nums):
                raw_image = np.array(vr[j*int(fps)])
                raw_image = Image.fromarray(raw_image)
                inputs_image = processor(images=raw_image, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs_image)
                clip_score = torch.nn.CosineSimilarity(dim=-1)(text_features, image_features)
                score.append(clip_score.item())
                frame_num.append(j*int(fps))

        else:
            text = 'Question: ' + data['question'] + ' Candidate: ' 
            if args.dataset_name == 'longvideobench':
                for j,cad in enumerate(data['candidates']):
                    text = text + ". ".join([chr(ord("A")+j), cad]) + ' '
            else:   
                for j in data['options']:
                    text = text + j
            text = text + '. Is this a good frame can answer the question?'
            txt = text_processors["eval"](text)
            for j in range(frame_nums):
                raw_image = np.array(vr[j*int(fps)])
                raw_image = Image.fromarray(raw_image)
                img = vis_processors["eval"](raw_image).unsqueeze(0).unsqueeze(0).to(device)
                samples = {'video':img,'loc_input':txt}
                sevila_score = float(model.generate_score(samples).squeeze(0).squeeze(0))
                score.append(sevila_score)
                frame_num.append(j*int(fps))

        fn.append(frame_num)
        scores.append(score)
        
    with open(frame_path,'w') as f:
        json.dump(fn,f)
    with open(score_path,'w') as f:
        json.dump(scores,f)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)