#!/usr/bin/env python3

import argparse
import base64
import gc
import json
import os
import random
from pathlib import Path
from io import BytesIO

import yaml
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
import hashlib
import sys
import time

from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)
from json_repair import repair_json
from qwen_vl_utils import process_vision_info


def sample_video_frames(
    video_path: str,
    every_n_frames: int,
    max_frames: int | None = None,
):
    cap = cv2.VideoCapture(video_path)
    raw_frames = []
    idx = 0

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % every_n_frames == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            raw_frames.append(img)
            if max_frames and len(raw_frames) >= max_frames:
                break

        idx += 1

    cap.release()

    if len(raw_frames) < 2:
        raise RuntimeError("Not enough frames to satisfy temporal patch size=2")

    # Temporal duplication: [F0,F1] [F1,F2] ...
    expanded = []
    for i in range(len(raw_frames) - 1):
        expanded.append(raw_frames[i])
        expanded.append(raw_frames[i + 1])

    return expanded, {"num_frames": idx, "num_sampled_frames": len(expanded)}


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ----------------------
# Determinism
# ----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------
# Device handling
# ----------------------
def validate_device(device: str):
    if device == "cpu":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if device.startswith("cuda:"):
        idx = int(device.split(":")[1])
        if idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {device}, only {torch.cuda.device_count()} GPUs available"
            )
    elif device != "cuda":
        raise ValueError(f"Invalid device string: {device}")


# ----------------------
# Model loader
# ----------------------
class AutoModel:
    @staticmethod
    def from_pretrained(model_id: str):
        if model_id.startswith("Qwen/Qwen2-VL"):
            loader = Qwen2VLForConditionalGeneration
        elif model_id.startswith("Qwen/Qwen2.5-VL"):
            loader = Qwen2_5_VLForConditionalGeneration
        elif model_id.startswith("Qwen/Qwen3-VL"):
            loader = Qwen3VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model ID: {model_id}")

        return loader.from_pretrained(model_id, device_map=None, torch_dtype="auto")


# ----------------------
# Utilities
# ----------------------
def resize_image(image: Image.Image, target_size: int) -> Image.Image:
    w, h = image.size
    if max(w, h) <= target_size:
        return image
    scale = target_size / max(w, h)
    return image.resize((int(w * scale), int(h * scale)))


def image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def label_to_color(label: str):
    """Deterministic color per label."""
    h = int(hashlib.md5(label.encode("utf-8")).hexdigest()[:6], 16)
    r = (h >> 16) & 0xFF
    g = (h >> 8) & 0xFF
    b = h & 0xFF
    return (r, g, b)


def draw_bboxes(image: Image.Image, detections: list) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    for d in detections:
        xmin, ymin, xmax, ymax = d["bbox_2d"]
        label = d.get("label", "unknown")
        color = label_to_color(label)

        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

        if hasattr(font, "getbbox"):
            bbox = font.getbbox(label)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        else:
            text_w, text_h = font.getsize(label)

        draw.rectangle([xmin, ymin - text_h, xmin + text_w, ymin], fill=color)
        draw.text((xmin, ymin - text_h), label, fill="white", font=font)
    return img


def postprocess_boxes(detections: list, image: Image.Image, model_id: str) -> list:
    scale = not model_id.startswith("Qwen/Qwen2.5-VL")
    if not scale:
        return detections

    x_scale = image.width / 1000
    y_scale = image.height / 1000

    out = []
    for d in detections:
        xmin, ymin, xmax, ymax = d["bbox_2d"]
        d["bbox_2d"] = [
            int(xmin * x_scale),
            int(ymin * y_scale),
            int(xmax * x_scale),
            int(ymax * y_scale),
        ]
        out.append(d)
    return out

# ----------------------
# Check input type
# ----------------------
def is_video(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]


# ----------------------
# Model execution
# ----------------------
def run_model(cfg: dict):
    device = cfg["device"]
    model = AutoModel.from_pretrained(cfg["model_id"])
    processor = AutoProcessor.from_pretrained(cfg["model_id"])
    model.to(device)
    model.eval()

    log(f"Model {cfg['model_id']} loaded on {device}")

    input_path = cfg["input_file"]
    input_type = "video" if is_video(input_path) else "image"
    
    frames = None

    log(f"Input type determined: {input_type}")

    # Prepare message
    if input_type == "image":
        image = Image.open(input_path).convert("RGB")
        if cfg.get("resize_image", True):
            image_proc = resize_image(image, cfg.get("image_target_size", 1024))
        else:
            image_proc = image
        b64 = image_to_base64(image_proc)
        content = [{"type": "image", "image": f"data:image;base64,{b64}"}]

    else:  # video
        if "sample_video" in cfg:
            frames, out_args = sample_video_frames(
                video_path=input_path,
                every_n_frames=cfg["sample_video"]["every_n_frames"],
                max_frames=cfg["sample_video"].get("max_frames"),
            )
            log(f"Sampled {out_args['num_sampled_frames']} out of {out_args['num_frames']} frames from video")

            content = [{"type": "video", "video": frames}]
        else:
            # fallback to native video handling
            content = [{"type": "video", "video": input_path}]

    content += [
        {"type": "text", "text": cfg["system_prompt"]},
        {"type": "text", "text": cfg["user_prompt"]},
    ]

    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    log("Prepared input messages for the model")

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        image_patch_size=16,
        return_video_metadata=True,
    )
    if video_inputs:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        for v in video_inputs:
            log(f"Video input detected with shape: {v.shape}")
    else:
        video_metadatas = None, None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        video_metadata=video_metadatas,
        **video_kwargs,
    ).to(device)

    log("Inputs tokenized and moved to device")

    with torch.no_grad():
        # Qwen3-VL video-native
        out_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.get("max_new_tokens", 512),
        )
    log("Model generation completed")

    # Trim input_ids
    gen_ids = [out[len(inp) :] for inp, out in zip(inputs.input_ids, out_ids)]
    decoded = processor.batch_decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return decoded, {"frames": frames, "video_metadatas": video_metadatas}  # This is now a list of strings per frame (list of lists)


# ----------------------
# Stage-wise pipeline
# ----------------------
def run_pipeline(cfg: dict):
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Starting pipeline for input: {cfg['input_file']}")
    log(f"Using device: {cfg['device']}")
    if cfg["device"].startswith("cuda"):
        mem_alloc = torch.cuda.memory_allocated(cfg["device"]) / 1e9
        mem_reserved = torch.cuda.memory_reserved(cfg["device"]) / 1e9
        log(
            f"GPU memory before inference: {mem_alloc:.2f}GB / reserved: {mem_reserved:.2f}GB"
        )

    is_vid = is_video(cfg["input_file"])
    raw_outputs, run_model_info = run_model(cfg)  # always a list of length 1
    output_text = raw_outputs[0]
    log("Model inference completed")

    # Stage 0: save raw output
    raw_text_path = out_dir / "02_raw_model_output.txt"
    write_text(raw_text_path, output_text)
    log(f"Saved raw model output to {raw_text_path}")

    # Stage 1: repair JSON
    repaired_text = repair_json(output_text)
    write_text(out_dir / "03_repaired_text.txt", repaired_text)
    log("Repaired JSON saved")

    # Stage 2: parse JSON
    try:
        parsed = json.loads(repaired_text)
        with open(out_dir / "04_parsed_json.json", "w") as f:
            json.dump(parsed, f, indent=2)
        log("Parsed JSON saved")
    except Exception as e:
        write_text(out_dir / "ERROR.txt", str(e))
        parsed = []
        log(f"JSON parse error: {e}")

    # Stage 3: postprocess
    if is_vid:
        if cfg.get("last_frame", False):
            # video detections will refer to the last frame
            video = cfg["input_file"]
            # Here we would extract the last frame
            import cv2

            cap = cv2.VideoCapture(video)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError("Failed to read last frame from video")
            frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            # Structure is [{"time": 1.0, "bbox_2d": [x_min, y_min, x_max, y_max], "label": ""}, {"time": 2.0, "bbox_2d": [x_min, y_min, x_max, y_max], "label": ""}, ...].
            # Or [{"frame": 0, "bbox_2d": [...], "label": ""}, ...]
            # We need to group by frame/time
            time_to_dets = {}
            for det in parsed:
                t = det.get("time", det.get("frame"))
                if t not in time_to_dets:
                    time_to_dets[t] = []
                time_to_dets[t].append(det)
            # Sort by time
            parsed = [time_to_dets[t] for t in sorted(time_to_dets.keys())]

            for idx, frame_dets in enumerate(parsed):
                frame_dir = out_dir / f"{idx:05d}"
                frame_dir.mkdir(exist_ok=True)
                post = [
                    {
                        "bbox_2d": [int(c) for c in det["bbox_2d"]],
                        "label": det.get("label", ""),
                    }
                    for det in frame_dets
                ]
                with open(frame_dir / "05_postprocessed.json", "w") as f:
                    json.dump(post, f, indent=2)
                log(f"Frame {idx}: postprocessed JSON saved")
                # No annotated image for video frames (optional)
                if run_model_info["frames"] is not None:
                    temporal_shift = cfg.get("temporal_shift", 0)
                    frame_image = run_model_info["frames"][idx * 2 + temporal_shift]  # due to temporal duplication
                    post = postprocess_boxes(post, frame_image, cfg["model_id"])
                    annotated = draw_bboxes(frame_image, post)
                    annotated.save(frame_dir / "06_annotated.png")
                    log(f"Frame {idx}: annotated image saved")
    else:
        # single image
        frame_image = Image.open(cfg["input_file"]).convert("RGB")

        post = [
            {"bbox_2d": [int(c) for c in det["bbox_2d"]], "label": det.get("label", "")}
            for det in parsed
        ]
        post = postprocess_boxes(post, frame_image, cfg["model_id"])
        frame_dir = out_dir / "00000"
        frame_dir.mkdir(exist_ok=True)
        with open(frame_dir / "05_postprocessed.json", "w") as f:
            json.dump(post, f, indent=2)
        annotated = draw_bboxes(frame_image, post)
        annotated.save(frame_dir / "06_annotated.png")
        log("Annotated image saved for single image input")

    log("Pipeline finished successfully")
    if cfg["device"].startswith("cuda"):
        mem_alloc = torch.cuda.memory_allocated(cfg["device"]) / 1e9
        mem_reserved = torch.cuda.memory_reserved(cfg["device"]) / 1e9
        log(
            f"GPU memory after inference: {mem_alloc:.2f}GB / reserved: {mem_reserved:.2f}GB"
        )


# ----------------------
# CLI
# ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.device:
        cfg["device"] = args.device
    if "device" not in cfg:
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    validate_device(cfg["device"])
    set_seed(cfg.get("seed", 42))

    run_pipeline(cfg)
    print(f"Done. Outputs in {cfg['output_dir']}")


if __name__ == "__main__":
    main()
