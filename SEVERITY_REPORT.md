# Severity Detection — VISTA Challenge Submission

**Course**: Computer Vision — MSc Computer Science, University of Bari Aldo Moro  
**Challenge**: VISTA — UAV-based Road Accident Scene Understanding  
**Assignment**: Severity-aware captioning pipeline (Detect → Track → Caption)

---

## 1. Challenge Requirements

| Requirement | Status |
|---|---|
| Subclass `VistaPipeline`, implement `forward()` + `reset()` | ✅ `vista/pipeline/severity.py` |
| Detect vehicles and persons in every frame | ✅ YOLO tracking every frame |
| Assign persistent `track_id` per object | ✅ YOLO `.track(persist=True)` |
| Caption each track with a severity label | ✅ Qwen-VL zero-shot per-crop |
| ≥ 5 FPS on the shallow (detection/tracking) stage | ✅ Measured in `colab_run.ipynb` cell 8 |
| Zero-shot constraint on captioner | ✅ VLM never sees VISTA ground-truth labels |
| `predictions_tracks.csv` output | ✅ `video_id, track_id, frame_start, frame_end, caption` |
| `predictions_mot.csv` output | ✅ `video_id, frame_id, track_id, x1, y1, x2, y2, conf, category` |
| Primary metric: BERTScore-F1 | ✅ Evaluated in `colab_run.ipynb` cell 9 |
| Tie-breakers: MOTA and IDF1 | ✅ Derivable from `predictions_mot.csv` |

---

## 2. Inspiration from the Paper

This implementation is directly motivated by the limitations and future directions identified in:

> *"Towards Real-Time Drone Vision for Road Safety"*, De Marinis et al., University of Bari Aldo Moro / Ministry of Infrastructure and Transport.

The paper introduces **CrashVista** (1,000 annotated UAV images) and benchmarks detection models across zero-shot and fine-tuned paradigms. Its **Section 5 (Future Work)** explicitly identifies two gaps that this severity pipeline addresses:

> *"A second research direction concerns the development of modules for assessing accident severity. Beyond binary detection, estimating the severity of a crash and the potential involvement of injured individuals would significantly enhance the system's utility for emergency response."*

> *"We also plan to enrich the annotation scheme by introducing fine-grained, instance-level descriptions for each detected object (e.g., 'person lying on the ground', 'vehicle severely damaged', 'person standing nearby')."*

My implementation directly operationalises these two future directions within the constraints of the challenge.

---

## 3. What I Built

### 3.1 Architecture — Three Stages

```
UAV Video
    │
    ▼ (every frame)
┌──────────────────────────────────────┐
│  Stage 1 — YOLO Detection + Tracking │  ← shallow stage (≥ 5 FPS)
│  • Locates vehicles and persons       │
│  • Assigns persistent track_id        │
│  • Initial label from YOLO class name │
│    (crashed_car → "heavily damaged")  │
└──────────────┬───────────────────────┘
               │ (every N frames, default N=30)
               ▼
┌──────────────────────────────────────┐
│  Stage 2 — Qwen-VL Crop Captioning   │  ← zero-shot, no VISTA labels
│  • Crops each track's bounding box   │
│  • Sends crop + category prompt      │
│  • Returns a severity label string   │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Stage 3 — peak_severity()           │  ← conservative aggregation
│  • Retains the worst observed state  │
│  • Never downgrades a track's status │
└──────────────────────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
predictions_tracks.csv  predictions_mot.csv
```

### 3.2 Severity Vocabularies (`vista/utils/severity.py`)

Two controlled vocabularies, **ordered by severity rank** (higher index = more severe):

**Vehicles**
```
0: undamaged
1: minor damage
2: heavily damaged
3: overturned
4: on fire
```

**Persons**
```
0: standing
1: running
2: helping
3: calling for help
4: injured, sitting
5: injured, lying
6: unconscious
```

**Emergency vehicles** (reclassified from trucks/buses by the VLM)
```
ambulance on scene | fire truck on scene | police on scene | emergency vehicle
```

### 3.3 Zero-Shot Prompts

Three category-specific prompts are defined in `vista/utils/severity.py`:

- `vehicle_severity_prompt()` — asks the VLM to pick the single best match from the vehicle vocabulary
- `person_severity_prompt()` — asks the VLM to pick the single best match from the person vocabulary
- `emergency_vehicle_prompt()` — probes trucks/buses to detect ambulances/police cars before applying severity

The system prompt in `cfg_severity.yaml` constrains the VLM to reply with **only the label**, no explanation.

### 3.4 Peak Severity Aggregation

```python
def peak_severity(captions: list[str], category: str) -> str:
    rank = _PERSON_RANK if category == "person" else _VEHICLE_RANK
    scored = [(rank.get(c, -1), c) for c in captions]
    return max(scored, key=lambda x: x[0])[1]
```

**Rationale**: In emergency dispatch, a state once observed must not be forgotten. If a person is detected `unconscious` at frame 150 and the VLM misses them at frame 180, the track should still report `unconscious`. This conservative strategy is appropriate for first-responder use.

### 3.5 Emergency Vehicle Reclassification

YOLO does not have an `ambulance` class. Trucks and buses are sent to the VLM with a dedicated probe before severity scoring. If confirmed as an emergency vehicle, they are reclassified to category `emergency_vehicle` and skip the severity ranking entirely.

### 3.6 Initial Severity Bootstrap from YOLO

The fine-tuned YOLO model on VistaCrash provides a free binary severity signal:

| YOLO class | Initial caption | Category |
|---|---|---|
| `crashed_car` | `heavily damaged` | `car` |
| `car` | `undamaged` | `car` |
| `person` | `standing` | `person` |
| `truck` | `undamaged` | `car` (may become `emergency_vehicle`) |

This means even **before** the VLM runs, tracks have a meaningful severity label.

---

## 4. Files Modified / Created

| File | Role |
|---|---|
| `vista/utils/severity.py` | Severity vocabularies, prompts, `peak_severity()`, YOLO bootstrap maps |
| `vista/pipeline/severity.py` | `SeverityPipeline` — full VistaPipeline subclass |
| `vista/pipeline/__init__.py` | Exports `SeverityPipeline` |
| `vista/qwen.py` | Added `caption_crop(crop, prompt)` to `QwenVLHF` and `QwenVLUnsloth` |
| `qwen_yolo.py` | Fixed missing `csv` import, `frame_start/end` tracking, `conf` collection, FPS measurement, correct CSV export |
| `config/qwenyolo/cfg_severity.yaml` | Severity-focused config: 32-token max, 512px crops, controlled system prompt |
| `colab_run.ipynb` | Full Google Colab runner: install, run, FPS benchmark, BERTScore evaluation |

---

## 5. How to Run

### Google Colab (T4 GPU)
```
1. Upload VISTA.rar to Google Drive
2. Open colab_run.ipynb in Colab
3. Runtime → Change runtime type → T4 GPU
4. Run cells 1 → 11 in order
```

### Locally
```bash
python qwen_yolo.py --config config/qwenyolo/cfg_severity.yaml
```

### FPS Benchmark (shallow stage)
Cell 8 of `colab_run.ipynb` measures YOLO-only throughput over 100 frames after a 10-frame warmup. This is the stage the challenge requires ≥ 5 FPS on.

---

## 6. Limitations

### 6.1 VLM Latency
The Qwen-VL captioning stage runs every N frames (default: 30). This keeps the **shallow stage** well above 5 FPS, but the **total pipeline FPS** is dominated by VLM inference time. On a T4 GPU with `Qwen2.5-VL-7B-Instruct-bnb-4bit`, each Qwen call takes ~1–3 seconds, meaning the effective captioning rate is low for videos with many tracks.

### 6.2 Vocabulary Rigidity
The severity prompts constrain the VLM to a fixed vocabulary. This improves BERTScore consistency but may miss nuanced descriptions (e.g., "partially trapped under vehicle") that could score higher against free-form ground truth.

### 6.3 YOLO Class Dependency
The bootstrap severity signal (`crashed_car` → `heavily damaged`) requires a YOLO model fine-tuned on VistaCrash. If using the base `yolo12x.pt`, all vehicles start as `undamaged` until the VLM refines them.

### 6.4 No Temporal Smoothing Beyond Peak
`peak_severity` prevents downgrades but does not smooth noisy intermediate outputs. A single VLM hallucination (e.g., `on fire` on a clear frame) permanently sets a track to the highest severity.

### 6.5 Single-Crop Context
The VLM receives only the cropped bounding box, not the full scene. This limits contextual reasoning — a person next to a heavily damaged car could be better assessed with scene-level context.

### 6.6 Emergency Vehicle Probe Cost
Trucks and buses trigger two VLM calls (emergency probe + severity prompt). In dense scenes this doubles captioning time for those tracks.

---

## 7. Evaluation Metrics

| Metric | What it measures | Target |
|---|---|---|
| **BERTScore-F1** (primary) | Semantic similarity between predicted captions and ground truth | Maximize |
| **MOTA** (tie-breaker) | Tracking accuracy: penalises FP, FN, ID switches | Maximize |
| **IDF1** (tie-breaker) | Consistency of track IDs over time | Maximize |
| **FPS** (constraint) | Shallow stage throughput (YOLO only) | ≥ 5 FPS |

BERTScore is computed on the `caption` column of `predictions_tracks.csv`. It is **semantic**, not lexical — `"injured, sitting"` and `"person seated on ground"` score highly against each other, which rewards expressive, natural-language severity labels over keyword-only outputs.
