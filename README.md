# Building Entrance Prediction (Project D)
**Author:** Nickolas Vela

## Description
This project explores AI-based approaches for predicting building entrance locations and accessibility points from various data sources. It investigates how machine learning and spatial analysis techniques can improve the accuracy of entrance detection for applications in navigation, delivery routing, and accessibility mapping.

## Goal
Develop and prototype AI methods capable of accurately predicting building entrance locations using available geospatial and visual data.

---

## How It Works

The pipeline combines three data sources to triangulate entrance locations:

1. **Overture Maps (AWS S3)** — building footprint polygons and place metadata
2. **Mapillary Graph API** — geolocated street-level photos with compass bearings
3. **YOLOv8 entrance detector** — deep-learning model that finds door bounding boxes in images

For each detected door:
- A camera ray is constructed using the image compass angle and the true per-camera horizontal FOV (derived from Mapillary `camera_parameters`; falls back to 70° for most street cameras)
- The ray is intersected against the nearest building wall segments (AEQD flat-earth projection) with a ±3° spread to account for compass calibration error
- The hit point is snapped to the wall and offset 0.5 m outward to mark the entrance

Detections from multiple images of the same building are clustered (5 m radius) and merged via **confidence-weighted mean positioning**.

---

## Improvements Over the Reference Implementation

| Feature | mapillary-entrances | nick-locational-predictions |
|---|---|---|
| Input modes | Single lat/lon only | Single lat/lon **or** batch from parquet |
| Entrances per building | 1 (first cluster only) | **All clusters** returned |
| Confidence scoring | None | **YOLO conf × geometric quality** propagated to GeoJSON |
| Cluster merging | Simple arithmetic mean | **Confidence-weighted mean** |
| Map marker colours | Always red | **Red → green gradient** by confidence |
| GeoJSON properties | Minimal | Name, category, address, confidence, n_detections |
| Batch output | None | Summary `predicted_entrances.parquet` |
| Category priors | 7 categories | **23 categories** tuned to this dataset |
| Camera FOV | Hardcoded 45° | **True FOV from `camera_parameters`** (70° fallback) |
| Image sharpness filter | Strict (100 Laplacian) | **Relaxed (50)** — fewer false rejections |
| Ray-cast spread | 1° | **3°** — handles compass calibration error |
| Image resolution | 1024 px | **2048 px** thumbnails |
| Images per run | 20 | **50** default |
| Output organisation | Flat `outputs/` folder | **Timestamped run folders** |
| Visualiser | Hover panel | **Click-to-pin panel**, CartoDB tiles, zoom 21, layer toggles, name labels, full-group hover fade |
| Debug mode | None | `--debug` saves annotated YOLO images |

---

## Project Structure

```
nick-locational-predictions/
├── pipeline.py                    # Entry point
├── visualize.py                   # Interactive HTML map generator
├── project_d_samples.parquet      # 3 425 pre-labelled Overture place records
├── requirements.txt
├── .env.example
├── data/                          # Downloaded Overture parquet files (generated)
├── outputs/
│   └── 2024-01-15_14-30-00/       # One folder per pipeline run (auto-named)
│       ├── candidates/            # Downloaded Mapillary thumbnails
│       ├── geojsons/              # Per-point GeoJSON verification files
│       ├── visualizations/        # YOLO-annotated debug images (--debug only)
│       └── predicted_entrances.parquet  # Batch summary (generated)
└── src/
    ├── api.py                     # Orchestrates buildings + imagery gathering
    ├── download.py                # Downloads Overture data from S3
    ├── imagery.py                 # Fetches and saves Mapillary images
    ├── inference.py               # YOLO + ray-casting + clustering
    └── utils/
        ├── constants.py           # Overture release, category priors, thresholds
        ├── duck_db_utils.py       # DuckDB spatial queries
        ├── geo_utils.py           # Haversine, bbox helpers
        ├── inference_utils.py     # AEQD projection, ray-segment math, YOLO
        ├── io_utils.py            # GeoJSON + parquet output
        ├── mapillary_utils.py     # Mapillary API calls
        ├── matching_utils.py      # Building-place scoring
        └── polygon_utils.py       # WKT → vertex lists
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Mapillary token
cp .env.example .env
# edit .env and set MAPILLARY_ACCESS_TOKEN=<your token>
```

---

## Usage

### Single point
```bash
PYTHONPATH=. python3 pipeline.py \
    --input_point="37.780,-122.4092" \
    --search_radius=150 \
    --model=yolo_weights_750_image_set.pt \
    --conf=0.35 --prefer_360
```

### Single point with debug visualizations
```bash
PYTHONPATH=. python3 pipeline.py \
    --input_point="37.780,-122.4092" \
    --search_radius=150 \
    --model=yolo_weights_750_image_set.pt \
    --conf=0.35 --prefer_360 --debug
```
`--debug` saves a YOLO-annotated image for every input photo into `visualizations/`, so you can inspect exactly what the detector saw even for images with no detections.

### Batch — process places from the parquet dataset
```bash
# First 20 places (good for testing)
PYTHONPATH=. python3 pipeline.py \
    --from_parquet=project_d_samples.parquet \
    --limit=20 \
    --search_radius=150 \
    --model=yolo_weights_750_image_set.pt \
    --conf=0.35

# First 20 places with YOLO debug images saved
PYTHONPATH=. python3 pipeline.py \
    --from_parquet=project_d_samples.parquet \
    --limit=20 \
    --search_radius=150 \
    --model=yolo_weights_750_image_set.pt \
    --conf=0.35 --debug

# Hotels only
PYTHONPATH=. python3 pipeline.py \
    --from_parquet=project_d_samples.parquet \
    --category=hotel \
    --limit=10 \
    --model=yolo_weights_750_image_set.pt \
    --conf=0.35 --prefer_360
```

Each batch run creates a new timestamped folder, e.g. `outputs/2024-01-15_14-30-00/`, and produces:
- **Per-place GeoJSONs** in `geojsons/` — one file per location for per-location debugging
- **`batch_combined.geojson`** — all locations merged into one file
- **`batch_combined.html`** — the same interactive visualize.py map as single-point mode, opened automatically in the browser with all detected entrances across every location
- **`predicted_entrances.parquet`** — summary table of all detections

### Visualise results
```bash
# Auto-opens the most recent run
python3 visualize.py

# Visualise a specific GeoJSON (single-point or batch combined)
python3 visualize.py outputs/2024-01-15_14-30-00/geojsons/batch_combined.geojson
```

The map opens in your browser. **Click** any pin to see the photo and metadata. **Hover** over a pin to highlight the entire group — hovering a place pin highlights all its predicted entrances, and hovering an entrance highlights the place pin plus all sibling entrances for that building. All unrelated pins fade to 25% opacity.

---

## Key CLI Arguments

### pipeline.py

| Argument | Default | Description |
|---|---|---|
| `--input_point` | — | Single-point mode: `lat,lon` e.g. `37.780,-122.409` |
| `--from_parquet` | — | Batch mode: path to a `.parquet` places file |
| `--category` | — | *(batch)* Filter to a specific Overture place category, e.g. `hotel` |
| `--limit` | — | *(batch)* Maximum number of places to process |
| `--search_radius` | 120 m | Radius around each point for building + imagery search |
| `--place_radius` | 120 m | Radius for joining buildings to places |
| `--max_images` | 50 | Maximum Mapillary images downloaded per point |
| `--prefer_360` | off | Prefer 360° panoramic images when available |
| `--min_capture_date` | — | Minimum image capture date `YYYY-MM-DD` |
| `--model` | *(required)* | Path to YOLOv8 weights file |
| `--device` | `cpu` | Inference device (`cpu` or `cuda`) |
| `--conf` | 0.35 | YOLO confidence threshold |
| `--iou` | 0.5 | YOLO IOU threshold for NMS |
| `--save` | `outputs` | Root output directory (timestamped sub-folder created per run) |
| `--open_browser` | on | Open the interactive visualize.py map in the browser after a single-point run |
| `--debug` | off | Save a YOLO-annotated image for every input photo into `visualizations/` |

### visualize.py

| Argument | Default | Description |
|---|---|---|
| `geojson` *(positional)* | most recent | Path to a `.geojson` file; auto-discovers the latest run if omitted |
| `--outputs` | `outputs` | Root directory to search for runs when no explicit path is given |

---

## Ideas for Further Improvement

1. **Multi-ray triangulation** — Instead of accepting the first good ray-wall intersection, fire rays from every available camera that can see the building and solve for the intersection point with least-squares. More cameras = tighter estimate.

2. **Semantic segmentation** — Replace YOLO bounding boxes with a segmentation mask (SAM or Mask-RCNN) to get a per-pixel door boundary, then use the mask centroid as the anchor pixel instead of the bbox bottom-centre.

3. **Address-side prior** — The `addresses` field in the parquet contains street names. Geocoding the street and computing which building wall faces it would let the pipeline bias ray-casting toward the correct facade before YOLO runs.

4. **Temporal consistency** — Mapillary has multiple capture dates. Requiring at least two independent dates to confirm an entrance location would filter transient occlusions (parked trucks, scaffolding).

5. **Accuracy evaluation against the parquet dataset** — The `geometry` column already records where Overture thinks the place is. Computing the distribution of `distance(predicted_entrance, overture_centroid)` across the 3 425 places gives a direct precision metric without needing ground-truth surveys.

6. **Fine-tune YOLOv8 on hard negatives** — The current model was trained on 750 images. Collecting false positives from this dataset (garages, windows, signs) and retraining with `yolo train` would reduce noise in the ray-casting stage.
