import base64
from collections import deque
from io import BytesIO
import os
import subprocess
import cv2
import json
import yaml
import torch
import csv
import gc
import time
from pathlib import Path
from typing import Dict
import importlib
import inspect

from PIL import Image
from ultralytics import YOLO
from json_repair import repair_json
from vista.qwen import get_model
from vista.utils import set_seed, image_to_base64, resize_image, log, IGNORE_CATEGORIES, YOLO_INITIAL_CAPTION, snap_to_vocabulary
from vista.utils import get_emergency_level


os.environ["HF_HOME"] = "/media/nvme/pasquale/HF"


# ============================================================
# Geometry
# ============================================================

def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (areaA + areaB - inter)

# ============================================================
# Drawing
# ============================================================

def draw_tracks(frame, tracks):
    font_size = 1.0
    thickness = 2
    for tid, t in tracks.items():
        # Assign a color based on the emergency level
        color_db = {
            3: (0, 0, 255),    # Red for high emergency
            2: (0, 165, 255),  # Orange for medium emergency
            1: (0, 255, 255),  # Yellow for low emergency
            0: (255, 0, 0),    # Blue for no emergency
        }
        # color = color_db.get(t.get("emergency_level", 0), (255, 0, 0))  # Default to blue if not found
        x1, y1, x2, y2 = map(int, t["bbox"])
        label = t.get("label", "unknown")
        # color = (255, 0, 0)  # Uniform color for all tracks
        color = (0, 0, 255) if "person" in label.lower() else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness+1)
        # draw white rectangle background for text
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - 6 - text_h)), (x1 + text_w, max(0, y1 - 6 + 2)), (0,0,0), -1)
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 255, 255),
            thickness,
        )
    return frame

def draw_qwen_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox_2d"])
        label = det.get("label", "unknown")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
    return frame


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


# ============================================================
# Pipeline
# ============================================================

def _parse_frame(value, fps):
    """Convert a frame spec to an int frame index.
    Accepts an int/float (already a frame number) or a 'min:sec' string."""
    if isinstance(value, str):
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError(f"Expected 'min:sec' format, got: {value!r}")
        minutes, seconds = int(parts[0]), float(parts[1])
        return int((minutes * 60 + seconds) * fps)
    return int(value)


def run_pipeline(cfg: dict):
    video_path = cfg["input"]["video"]
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = cfg["device"]

    log("Loading YOLO tracker")
    yolo_model_path = cfg["yolo"]["model"]
    if not Path(yolo_model_path).exists():
        fallback = cfg["yolo"].get("base_model", "yolo12x.pt")
        log(f"Fine-tuned model not found at {yolo_model_path!r}, falling back to {fallback!r}")
        yolo_model_path = fallback
    yolo = YOLO(yolo_model_path)


    load_prediction_dir = cfg["qwen"].get("load_prediction_dir", None)

    if load_prediction_dir is not None:
        model = None
        log(f"Loading Qwen predictions from {load_prediction_dir}, skipping Qwen inference")
    else:
        log("Loading Qwen processor and model")
        model = get_model(cfg)
        log("Models loaded successfully")

    # If the config specifies a pipeline class, try to import and run it.
    pipeline_class_path = cfg.get("pipeline", {}).get("class")
    if pipeline_class_path:
        try:
            mod_name, cls_name = pipeline_class_path.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            PipelineCls = getattr(mod, cls_name)
        except Exception as e:
            log(f"Failed to import pipeline class {pipeline_class_path}: {e}")
            PipelineCls = None

        if PipelineCls is not None:
            # Try flexible instantiation: (yolo, vlm/model, **kwargs) or fallbacks
            vlm = model
            p_cfg = cfg.get("pipeline", {})
            # pull some commonly used pipeline args if present
            kwargs = {}
            for k in ("caption_stride", "yolo_conf", "history_len", "check_emergency", "iou_threshold", "yolo_conf"):
                if k in p_cfg:
                    kwargs[k] = p_cfg[k]
            # ajoute juste cette ligne après :
            kwargs["cfg"] = cfg

            pipeline = None
            
            try:
                # preferred constructor: (yolo_model, vlm, ...)
                pipeline = PipelineCls(yolo, vlm, **kwargs)
            except TypeError:
                try:
                    pipeline = PipelineCls(yolo, vlm)
                except TypeError:
                    try:
                        pipeline = PipelineCls(yolo)
                    except Exception as e:
                        log(f"Failed to instantiate pipeline {pipeline_class_path}: {e}")
                        pipeline = None

            if pipeline is not None:
                log(f"Running custom pipeline: {pipeline_class_path}")
                # delegate to a generic runner that calls pipeline.forward()
                _run_vista_pipeline(pipeline, cfg, out_dir, video_path)
                return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)

    raw_start = cfg["input"].get("start_frame", 0)
    raw_end = cfg["input"].get("end_frame", None)
    start_frame = _parse_frame(raw_start, fps) if raw_start else 0
    end_frame = _parse_frame(raw_end, fps) if raw_end is not None else None
    if isinstance(raw_start, str) or isinstance(raw_end, str):
        log(f"Time-based range: start_frame={start_frame}, end_frame={end_frame} (fps={fps:.2f})")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_video_path = str(out_dir / "annotated_raw.mp4")
    output_video_path = str(out_dir / "annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(
        raw_video_path,
        fourcc,
        fps,
        (W, H),
    )

    track_db: Dict[int, Dict] = {}

    stride = cfg["qwen"]["every_n_frames"]
    iou_thr = cfg["yolo"]["iou_match_threshold"]
    qwen_N = cfg["qwen"].get("num_frames", 1)  # how many frames to give to Qwen

    (out_dir / "qwen").mkdir(exist_ok=True)
    qwen_history = deque(maxlen=qwen_N)

    log("Starting video loop")
    
    # Seek to start_frame if specified
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
        log(f"Seeking to frame {start_frame}")
    else:
        frame_idx = 0

    join_tracks: Dict[int, Dict] = {}
    qwen_only_tracks: Dict[int, Dict] = {}
    video_id = Path(video_path).stem
    all_detections: list = []
    final_tracks: Dict[int, Dict] = {}
    loop_start_time = time.perf_counter()
    yolo_time_total = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if we've reached end_frame
        if end_frame is not None and frame_idx >= end_frame:
            log(f"Reached end_frame {end_frame}, stopping processing")
            break

        _yolo_t0 = time.perf_counter()
        results = yolo.track(frame, persist=True, verbose=False, conf=cfg["yolo"].get("conf", None))[0]
        yolo_time_total += time.perf_counter() - _yolo_t0
        active_tracks = {}

        if results.boxes.id is not None:
            for box, tid, cls, det_conf in zip(results.boxes.xyxy, results.boxes.id, results.boxes.cls, results.boxes.conf):
                tid = int(tid.item())
                yolo_class = results.names.get(int(cls.item()), "unknown")
                initial_label = YOLO_INITIAL_CAPTION.get(yolo_class, yolo_class)
                active_tracks[tid] = {
                    "bbox": box.cpu().numpy().tolist(),
                    "label": track_db.get(tid, {}).get("label", initial_label),
                    "emergency_level": track_db.get(tid, {}).get("emergency_level", 0),
                    "conf": float(det_conf.item()),
                }
        # Remove inactive tracks from track_db
        inactive_tids = set(track_db.keys()) - set(active_tracks.keys())
        for tid in inactive_tids:
            final_tracks[tid] = track_db[tid]
            del track_db[tid]
        
        # Remove tracks belonging to ignored categories
        active_tracks = {tid: tr for tid, tr in active_tracks.items() if tr["label"] not in IGNORE_CATEGORIES}

        # ----------------------------
        # Update frame memory
        # ----------------------------
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # ----------------------------
        # Qwen inference
        # ----------------------------
        if frame_idx % stride == 0:
            log(f"Qwen inference at frame {frame_idx}")

            # Run Qwen
            if load_prediction_dir is not None:
                raw_path = Path(load_prediction_dir) / "qwen" / f"{frame_idx:06d}_raw.txt"
                raw = raw_path.read_text()
                log(f"Loaded Qwen predictions from {raw_path}")
            else:
                raw = model.generate(frame=pil_frame, history=list(qwen_history))
                
            qwen_history.append((pil_frame, raw))

            qwen_dir = out_dir / "qwen"
            raw_path = qwen_dir / f"{frame_idx:06d}_raw.txt"
            raw_path.write_text(raw)

            repaired = repair_json(raw)
            (qwen_dir / f"{frame_idx:06d}_repaired.json").write_text(repaired)

            try:
                parsed = json.loads(repaired)
                parsed = postprocess_boxes(parsed, pil_frame, cfg["qwen"]["model_id"])
                
                qwen_annotated = draw_qwen_detections(frame.copy(), parsed)
                cv2.imwrite(
                    str(qwen_dir / f"{frame_idx:06d}_annotated.png"),
                    qwen_annotated,
                )
            except Exception as e:
                log(f"Qwen JSON parse error at frame {frame_idx}: {e}")
                parsed = []

            # ----------------------------
            # Match Qwen detections to tracks
            # ----------------------------
            matches = 0
            join_tracks = {}       # reset for this stride frame
            qwen_only_tracks = {}  # reset for this stride frame
            for det in parsed:
                qb = det["bbox_2d"]
                label = snap_to_vocabulary(det.get("label", "unknown"))
                urgency_level = det.get("level", get_emergency_level(label))

                best_tid = None
                best_score = 0.0
                for tid, tr in active_tracks.items():
                    score = iou(qb, tr["bbox"])
                    if score > best_score:
                        best_score = score
                        best_tid = tid

                if best_tid is not None and best_score >= iou_thr:
                    prev = track_db.get(best_tid, {})
                    track_db[best_tid] = {
                        "bbox": active_tracks[best_tid]["bbox"],
                        "label": label,
                        "emergency_level": urgency_level,
                        "frame_start": prev.get("frame_start", frame_idx),
                        "frame_end": frame_idx,
                        "conf": active_tracks[best_tid].get("conf", 1.0),
                    }
                    join_tracks[best_tid] = track_db[best_tid]
                    matches += 1
                else:
                    # Create a new track for Qwen-only detection
                    new_tid = max(track_db.keys(), default=0) + 1
                    qwen_only_tracks[new_tid] = {
                        "bbox": qb,
                        "label": label,
                        "emergency_level": urgency_level,
                        "frame_start": frame_idx,
                        "frame_end": frame_idx,
                        "conf": 1.0,
                    }
                    track_db[new_tid] = qwen_only_tracks[new_tid]
            log(f"YOLO detected {len(active_tracks)} tracks, Qwen detected {len(parsed)} objects, matched {matches}")
            
        merge_method = cfg.get("merge_method", "update")

        # ----------------------------
        # Update track DB
        # ----------------------------
        for tid, tr in active_tracks.items():
            if tid not in track_db:
                track_db[tid] = tr
                track_db[tid]["frame_start"] = frame_idx
            else:
                track_db[tid]["bbox"] = tr["bbox"]
                track_db[tid]["conf"] = tr["conf"]
            track_db[tid]["frame_end"] = frame_idx
                
        if merge_method == "intersection":
            # Keep only tracks that were matched with Qwen detections
            track_db = {tid: tr for tid, tr in track_db.items() if tid in join_tracks}
        elif merge_method == "union":
            # Add Qwen-only tracks to track_db
            for tid, tr in qwen_only_tracks.items():
                track_db[tid] = tr

        annotated = draw_tracks(frame.copy(), track_db)
        if cfg.get("save_annotated_frames_frequency", 0) > 0 and frame_idx % cfg["save_annotated_frames_frequency"] == 0:
            Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).save(
                out_dir / f"annotated_frame_{frame_idx:06d}.png"
            )
        writer.write(annotated)
        for tid, tr in track_db.items():
            x1, y1, x2, y2 = map(int, tr["bbox"])
            all_detections.append((
                video_id, frame_idx, tid,
                x1, y1, x2, y2,
                round(tr.get("conf", 1.0), 4),
                tr.get("label", "unknown"),
            ))
        frame_idx += 1

    cap.release()
    writer.release()

    for tid, tr in track_db.items():
        final_tracks[tid] = tr
    frames_processed = frame_idx - start_frame
    elapsed = time.perf_counter() - loop_start_time
    fps_achieved = frames_processed / elapsed if elapsed > 0 else 0
    yolo_fps = frames_processed / yolo_time_total if yolo_time_total > 0 else 0
    log(f"Processed {frames_processed} frames in {elapsed:.1f}s — total pipeline: {fps_achieved:.1f} FPS")
    log(f"Shallow stage (YOLO tracking only): {yolo_fps:.1f} FPS  (challenge threshold: >=5 FPS)")

    log(f"Converting video to H.264: {output_video_path}")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", raw_video_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-movflags", "+faststart",
            output_video_path,
        ],
        check=True,
    )
    os.remove(raw_video_path)
    log(f"Video saved to {output_video_path}")

    predictions_tracks_path = out_dir / "predictions_tracks.csv"
    with open(predictions_tracks_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["video_id", "track_id", "frame_start", "frame_end", "caption"])
        for tid, tr in final_tracks.items():
            csv_writer.writerow([
                video_id, tid,
                tr.get("frame_start", 0),
                tr.get("frame_end", 0),
                tr.get("label", ""),
            ])
    log(f"Saved {predictions_tracks_path}")

    predictions_mot_path = out_dir / "predictions_mot.csv"
    with open(predictions_mot_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["video_id", "frame_id", "track_id", "x1", "y1", "x2", "y2", "conf", "category"])
        for det in all_detections:
            csv_writer.writerow(det)
    log(f"Saved {predictions_mot_path}")


def _run_vista_pipeline(pipeline, cfg, out_dir: Path, video_path: str):
    """Generic runner for VistaPipeline-like objects (calls forward on each frame).

    Outputs: annotated video, predictions_tracks.csv, predictions_mot.csv (same layout as other runner).
    """
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video for pipeline runner")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_video_path = str(out_dir / "annotated_raw.mp4")
    output_video_path = str(out_dir / "annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(raw_video_path, fourcc, fps, (W, H))

    track_db = {}
    all_detections = []
    final_tracks = {}

    frame_idx = 0
    loop_start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # pipeline.forward returns a FrameResult with detections
        try:
            res = pipeline.forward(pil_frame, frame_idx)
        except Exception as e:
            log(f"Pipeline forward() error at frame {frame_idx}: {e}")
            frame_idx += 1
            continue

        # build track_db from detections
        for det in res.detections:
            tid = getattr(det, "track_id", None)
            bbox = tuple(det.bbox)
            label = getattr(det, "category", "unknown")
            conf = getattr(det, "confidence", 1.0)
            track_db[tid] = {
                "bbox": bbox,
                "label": label,
                "conf": conf,
                "frame_start": track_db.get(tid, {}).get("frame_start", frame_idx),
                "frame_end": frame_idx,
            }
            x1, y1, x2, y2 = map(int, bbox)
            all_detections.append((Path(video_path).stem, frame_idx, tid, x1, y1, x2, y2, round(conf, 4), label))

        # draw tracks
        annotated = frame.copy()
        for tid, tr in track_db.items():
            x1, y1, x2, y2 = map(int, tr["bbox"])
            label = tr.get("label", "")
            color = (0, 0, 255) if "person" in label.lower() else (255, 0, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    for tid, tr in track_db.items():
        final_tracks[tid] = tr

    predictions_tracks_path = out_dir / "predictions_tracks.csv"
    with open(predictions_tracks_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["video_id", "track_id", "frame_start", "frame_end", "caption"])
        for tid, tr in final_tracks.items():
            csv_writer.writerow([Path(video_path).stem, tid, tr.get("frame_start", 0), tr.get("frame_end", 0), tr.get("label", "")])
    log(f"Saved {predictions_tracks_path}")

    predictions_mot_path = out_dir / "predictions_mot.csv"
    with open(predictions_mot_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["video_id", "frame_id", "track_id", "x1", "y1", "x2", "y2", "conf", "category"])
        for det in all_detections:
            csv_writer.writerow(det)
    log(f"Saved {predictions_mot_path}")

    # Try to convert to H.264 like original runner
    try:
        log(f"Converting video to H.264: {output_video_path}")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", raw_video_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-movflags", "+faststart",
            output_video_path,
        ], check=True)
        os.remove(raw_video_path)
        log(f"Video saved to {output_video_path}")
    except Exception as e:
        log(f"ffmpeg conversion failed or not available: {e}")

# ============================================================
# Entry point
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    run_pipeline(cfg)

if __name__ == "__main__":
    main()
