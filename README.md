# INDIAN LICENCE PLATE RECOGNITION (ANPR)

A lightweight, production-oriented pipeline for **Indian number plate recognition** built on **YOLOv8 (Ultralytics)** for detection, **SORT** for multi-object tracking, and **EasyOCR** for text reading — with domain-specific rules that enforce **Indian plate formats** and post-processing that **interpolates** gaps and **merges** fragmented tracks.

---

## Why this matters

Automatic Number Plate Recognition (ANPR) is critical for **traffic analytics**, **parking automation**, **tolling**, **law enforcement**, and **access control**.  
This project focuses on **India-specific formats**, boosting practical accuracy by:

- Using a **strict Indian plate regex** to filter OCR noise early.  
- **Tracking** vehicles with SORT so the plate is read **once per car** instead of per frame.  
- **Interpolating** missing detections and **merging** fragmented IDs using **plate similarity + IoU**.  
- Favoring sharp, confident frames via a combined score (**OCR score + bbox confidence + sharpness**).

---

## Pipeline overview

```
Video -> YOLOv8 (vehicle detection) -> SORT (tracking)
      -> YOLO (plate detection) -> crop plate
      -> EasyOCR (strict-only reading) -> format clean-up
      -> best-per-car selection (OCR + det + sharpness)
      -> CSV export (strict-only rows)
      -> Interpolation + ID merge (plate similarity + IoU)
      -> Visualization (overlay bboxes + best crop + text)
```

**Key design choices**
- **Strict‐only OCR**: only plates passing `MH12AB1234` / `MH12A1234`–style rules are kept.
- **Per-car best frame**: select using `text_score + 0.15 * det_conf + 0.10 * focus_score`.
- **Robust merging**: car IDs are unified across time with **dynamic plate similarity thresholds** and **IoU**.

---

## Project structure

```
.
├── main.py                 # Orchestrates detection, tracking, OCR, CSV, interpolation, visualization
├── util.py                 # OCR (EasyOCR), Indian plate rules, CSV writer, bbox parser, similarity utils
├── add_missing_data.py     # Interpolate bboxes; merge duplicate car_ids via plate similarity + IoU
├── visualize.py            # Render output.mp4 with overlays (car box, plate box, plate text + crop)
├── sort/                   # SORT tracker (expects sort/sort.py and dependencies)
│   └── sort.py
├── INPD_more_accuracy_n.pt # Your license-plate YOLO weights (place in repo root)
├── requirements.txt
└── README.md
```

**Models used**
- `yolov8n.pt` (downloaded automatically by Ultralytics) — **vehicle** detection.  
- `INPD_more_accuracy_n.pt` — **license plate** detector (place in the repository root or update the path in `main.py`).

**SORT dependency**
You must clone the SORT repository in the same project directory before running `main.py`:

```bash
git clone https://github.com/abewley/sort.git
```

This will create the `sort/` folder that `main.py` depends on.

---

## Installation

> **Python 3.9–3.11 recommended**

```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**GPU (optional):**  
- PyTorch GPU wheels depend on your CUDA version. See notes inside `requirements.txt`.  
- EasyOCR uses PyTorch too. In `util.py` it’s set to `gpu=False` by default for portability; set `gpu=True` if you have CUDA.

**OpenCV headless (server):**  
- If you run on a headless server, consider swapping `opencv-python` → `opencv-python-headless` in `requirements.txt`.

---

## Quick start

Place your input video (e.g., `sample.mp4`) in the repo directory and run:

```bash
python main.py --video_path sample.mp4 --det-conf 0.88 --det-imgsz 736
```

**Outputs**
- `<video_name>_test.csv` — raw, strict-only detections per frame.  
- `<video_name>_interpolated.csv` — after interpolation & ID merge.  
- `<video_name>_output.mp4` — visualization with overlays:
  - Green **vehicle** border (fancy corner style)
  - Red **plate** rectangle
  - Plate **text + score** and the best **plate crop** rendered above the car

---

## CLI options

`main.py` accepts two hyper-parameters for **vehicle** detection (not plate detection):

- `--det-conf` *(float, default 0.88)*  
  YOLO confidence threshold for **vehicle** detection.  
- `--det-imgsz` *(int, default 736)*  
  YOLO input image size for **vehicle** detection. Typical values: `640, 704, 736, 832, 960`.

> The plate detector currently uses its default settings. If you want to tune plate detection as well, add `conf=` and `imgsz=` to the `lpd_model(frame, ...)` call in `main.py`.

---

## How to tune for better accuracy

### 1) Vehicle detection (`--det-conf`, `--det-imgsz`)

- **`--det-conf` (confidence threshold)**
  - **Higher (e.g., 0.85–0.92)** → **fewer false positives** but can miss small/far vehicles.
  - **Lower (e.g., 0.60–0.80)** → **more recall** (detects more vehicles) but may add false positives.
  - Start around **0.85–0.90** for daylight, reduce to **0.75–0.85** for crowded/low-light scenes.

- **`--det-imgsz` (input size)**
  - **Larger (736–960)** → better small-object recall (far plates) but slower.
  - **Smaller (640–704)** → faster but may miss small vehicles/plates.
  - If plates are **far/small**, try **832–960** (with GPU). For CPU, **704–736** is a good compromise.

### 2) Plate detection (optional code tweak)
In `main.py`, the plate model call is currently:
```python
lp_res = lpd_model(frame)[0]
```
You can expose its thresholds too:
```python
lp_res = lpd_model(frame, conf=0.25, imgsz=640)[0]
```
Make these configurable (arguments) if you want full control.

### 3) OCR strictness & post-processing

- **Strict Indian format** is enforced in `util.strict_format()`.  
  If your data includes commercial/temporary formats, tweak it there.
- **Character disambiguation** (e.g., `O↔0`, `I↔1`, `S↔5`) happens in `util.clean_and_correct()`.

### 4) Track/ID merging behavior

- In `add_missing_data.merge_car_ids()` we use **plate similarity + IoU**:
  - **Dynamic plate similarity threshold**:
    - When OCR is **confident** (`score ≥ 0.6`) → **stricter** (`threshold ≈ 0.9`).
    - When OCR is **uncertain** → **looser** (`threshold ≈ 0.7`).
  - **Spatial check**: cars only merge if **IoU > 0.85** by default.

> If you see the same vehicle split into multiple IDs, try **lowering IoU** (e.g., 0.75–0.8) or relaxing the **similarity thresholds** slightly.  
> If different cars are being merged, **increase IoU** (e.g., 0.9) or tighten similarity (e.g., 0.92+).

### 5) “Best frame” scoring weights

In `main.py`, per-car “best” frame uses:
```python
combined = text_score + 0.15 * det_conf + 0.10 * focus_score
```
If you want to prioritize sharpness or detection confidence more, adjust `0.15` / `0.10`.

---

## Data files and CSV schema

**Raw CSV** (`*_test.csv`) — one row per (frame, car):

| Column                          | Type    | Notes                                                  |
|---------------------------------|---------|--------------------------------------------------------|
| frame_nmr                       | int     | Frame index                                            |
| car_id                          | int     | Global ID (unified by plate similarity)                |
| car_bbox                        | json[4] | `[x1,y1,x2,y2]`                                        |
| license_plate_bbox              | json[4] | `[x1,y1,x2,y2]`                                        |
| license_plate_bbox_score        | float   | YOLO score for the plate box                           |
| license_number                  | str     | Strict Indian plate (cleaned & validated)              |
| license_number_score            | float   | EasyOCR mean score of the merged read                  |

**Interpolated CSV** (`*_interpolated.csv`) — includes **interpolated frames** and **merged IDs**.

---

## Tips & troubleshooting

- **No plates detected / Empty CSV**
  - Ensure `INPD_more_accuracy_n.pt` exists at the project root (or update its path).
  - Try **lowering `--det-conf`** or **increasing `--det-imgsz`** (vehicles may be missed otherwise).
  - Your plates may be too small; increase video resolution or camera proximity.

- **Visualization video not saved**
  - Check that your OpenCV build supports the `mp4v` codec.  
    On Linux, installing `ffmpeg` helps. You can also try `fourcc = cv2.VideoWriter_fourcc(*'avc1')`.

- **Slow on CPU**
  - Reduce `--det-imgsz` to **640–704**.
  - Set EasyOCR to **`gpu=True`** if you have CUDA (in `util.py`).

- **Different Indian formats**
  - Adjust `strict_format()` in `util.py` to match your specific plate formats.

- **Headless servers**
  - Use `opencv-python-headless` and avoid any GUI calls (this project doesn’t open windows).

---

## Ethical use

Respect privacy laws and local regulations. Obtain consent where required. Avoid storing personally identifiable information longer than necessary. This project is provided **for research/educational use**; you are responsible for compliance in your deployment.

---

## Acknowledgements

- [Ultralytics YOLOv8] for detection  
- [SORT](https://github.com/abewley/sort) (Simple Online and Realtime Tracking) for multi-object tracking  
- [EasyOCR] for OCR  
- [FilterPy] for the Kalman filter used by SORT

---

## Contributing

Issues and PRs are welcome! If you contribute:
- Keep code **pep8/black** formatted.
- Add comments for any **thresholds** or **heuristics** you change.
- Where possible, keep **defaults practical** for CPU users.

---

## License

Choose a license that fits your use case (e.g., **MIT**, **Apache-2.0**).  
Add a `LICENSE` file at the repository root.

---

## Example commands

```bash
# Default (balanced)
python main.py --video_path sample.mp4 --det-conf 0.88 --det-imgsz 736

# Higher recall (crowded scenes, smaller vehicles)
python main.py --video_path sample.mp4 --det-conf 0.78 --det-imgsz 832

# Faster (CPU)
python main.py --video_path sample.mp4 --det-conf 0.85 --det-imgsz 640
```
