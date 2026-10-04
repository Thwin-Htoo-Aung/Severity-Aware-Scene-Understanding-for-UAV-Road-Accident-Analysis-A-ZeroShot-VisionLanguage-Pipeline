# DroneAmbulanceAI

Part of the **VISTA** project (University of Bari Aldo Moro — Italian Ministry of Infrastructure and Transport). VISTA dispatches an AI-equipped UAV to road accident scenes ahead of first responders, giving paramedics situational awareness before they arrive.

This repository contains two complementary systems built on top of the **CrashVista** dataset:

1. **Static detection** — benchmark of zero-shot and fine-tuned detectors on UAV accident imagery.
2. **Video pipeline** — frame-by-frame tracking and grounded captioning of accident-scene video.

---

## Repository Structure

```
DroneAmbulanceAI/
├── vista/
│   ├── models/             # Detection model wrappers
│   │   ├── yolo.py         # YOLOVista  (YOLO family)
│   │   ├── yoloe.py        # YOLOEVista (YOLOe)
│   │   ├── moondream.py    # MoonDream  (multimodal LM)
│   │   ├── omdetturbo.py   # OmDetTurbo (open-vocabulary)
│   │   ├── rtdetr.py       # RT-DETR
│   │   ├── sam/            # SAM 3 + linear probe
│   │   ├── validator.py    # VISTAValidator / VISTAOutputMixin
│   │   └── __init__.py     # MODEL_ZOO + get_model()
│   ├── pipeline/
│   │   ├── base.py         # VistaPipeline ABC, Detection, FrameResult
│   │   ├── qwen_yolo.py    # QwenYoloPipeline (YOLO tracker + Qwen-VL)
│   │   └── __init__.py
│   ├── qwen.py             # Qwen-VL model loader (HF + vLLM + Unsloth)
│   ├── evaluate.py         # run() — train / val entry point
│   └── stats.py            # YOLO dataset statistics
├── config/                 # Experiment YAML configs
│   ├── VistaSynth/         # Detection benchmark configs
│   ├── qwenyolo/           # Video pipeline configs
│   └── sim18/              # Simulation sequence configs
├── main.py                 # CLI: run | grid | stats
├── qwen_yolo.py            # Standalone video pipeline entry point
├── app.py                  # Streamlit dataset explorer
└── eval.py                 # Quick one-shot evaluation script
```

---

## Installation

Requires **Python 3.10.13**. Dependencies are managed with [uv](https://github.com/astral-sh/uv).

```bash
# install uv if needed
pip install uv

# create the virtual environment and install all dependencies
uv sync
```

A GPU with at least 16 GB VRAM is recommended. The Qwen-VL models require 24–80 GB depending on the variant.

---

## Detection Models

All detection models are registered in `vista/models/__init__.py` and share the Ultralytics `predict` / `val` / `set_classes` interface. Validation automatically writes structured outputs (`metrics_summary.json`, `pr_curves.json`, `speed_summary.json`, etc.) to the run directory via `VISTAOutputMixin`.

| Key | Class | Backend | Notes |
|---|---|---|---|
| `yolo` | `YOLOVista` | Ultralytics YOLO | YOLO 11/12/26 family |
| `yoloe` | `YOLOEVista` | Ultralytics YOLOe | Open-vocabulary; supports `set_classes` |
| `moondream` | `MoonDream` | HuggingFace | Compact multimodal LM |
| `omdetturbo` | `OmDetTurbo` | HuggingFace | Real-time open-vocab transformer |
| `rtdetr` | `RTDETRVista` | Ultralytics RT-DETR | Transformer detector |
| `sam` | `Sam3Model` | Meta SAM 3 | Segmentation → detection; supports linear probe |

### Running an evaluation

```bash
python main.py run --parameters config/VistaSynth/base.yaml
```

Config format:

```yaml
model:
  name: yoloe
  model: yoloe-26s-seg.pt

classes: ["crashed_car", "person", "car"]

val:
  data: data/VistaSynth/data.yaml
  split: test
```

### Running a grid search

```bash
python main.py grid --parameters config/VistaSynth/base.yaml
```

Grid configs use a `parameters` block where each key can be a list of values; the CLI expands all combinations and runs them sequentially (or in parallel with `--parallel`).

---

## Video Pipeline

The `vista/pipeline/` module provides an abstract interface for all video-level systems. Any pipeline must subclass `VistaPipeline` and implement `forward`:

```python
from vista.pipeline import VistaPipeline, FrameResult, Detection
from PIL import Image

class MyPipeline(VistaPipeline):

    def forward(self, frame: Image.Image, frame_idx: int) -> FrameResult:
        ...
        return FrameResult(
            detections=[
                Detection(
                    bbox=(x1, y1, x2, y2),
                    category="person",
                    confidence=0.9,
                    track_id=42,
                    caption="injured, sitting",
                )
            ],
            frame_idx=frame_idx,
        )

    def reset(self) -> None:
        ...  # clear tracker state between videos
```

The harness calls `reset()` once before each video and `pipeline(frame, frame_idx)` for every frame. The convenience method `process_video(path)` wraps this loop and yields `FrameResult` objects:

```python
pipeline = MyPipeline(...)
for result in pipeline.process_video("accident.mp4"):
    for det in result.detections:
        print(det.track_id, det.category, det.caption)
```

### QwenYoloPipeline

The reference implementation fuses YOLO tracking with Qwen-VL captioning:

```python
from ultralytics import YOLO
from vista.pipeline.qwen_yolo import QwenYoloPipeline
from vista.qwen import get_model

cfg = { ... }  # YAML config dict
pipeline = QwenYoloPipeline(
    yolo_model=YOLO("yolo12x.pt"),
    qwen_model=get_model(cfg),
    caption_stride=30,
    iou_threshold=0.3,
)
for result in pipeline.process_video("accident.mp4"):
    ...
```

Or run it directly from a YAML config:

```bash
python qwen_yolo.py --config config/qwenyolo/cfg10.yaml
```

Key config options:

```yaml
input:
  video: data/sequences/accident.mp4
  start_frame: 0          # or "1:30" (min:sec)
  end_frame: 900

output:
  dir: out/my_run

yolo:
  model: yolo12x.pt
  iou_match_threshold: 0.3
  conf: 0.05

qwen:
  model_id: Qwen/Qwen3-VL-8B-Instruct
  every_n_frames: 30
  max_new_tokens: 4096
  system_prompt: >
    You are an operator supervising a drone over an accident scene...
```

The pipeline saves:
- `annotated.mp4` — video with bounding boxes and captions overlaid
- `qwen/<frame>_raw.txt` — raw VLM output per queried frame
- `qwen/<frame>_repaired.json` — JSON-repaired structured detections
- `qwen/<frame>_annotated.png` — per-frame Qwen detection overlay

---

## Dataset Statistics

Compute and save full statistics for any YOLO-format dataset:

```bash
python main.py stats data/VistaSynth/data.yaml --output_dir stats/ --splits train,val,test
```

Outputs: class distribution plots, bounding-box scatter, spatial heatmaps, aspect-ratio histograms, co-occurrence matrix, per-class CSV — all as SVG/PNG.

---

## Dataset Explorer

A Streamlit app for browsing images with annotation overlays and inspecting dataset statistics interactively:

```bash
streamlit run app.py
```

Enter the path to any YOLO `data.yaml` file in the sidebar. The **Statistics** tab shows all the plots above; the **Explorer** tab lets you filter by split, class, and annotation count, and navigate images one by one or in a grid.
