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
- A camera ray is constructed using the image compass angle and a pinhole camera model
- The ray is intersected against the nearest building wall segments (AEQD flat-earth projection)
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

---

## Project Structure

```
nick-locational-predictions/
├── pipeline.py                    # Entry point
├── project_d_samples.parquet      # 3 425 pre-labelled Overture place records
├── requirements.txt
├── .env.example
├── data/                          # Downloaded Overture parquet files (generated)
├── outputs/
│   ├── candidates/                # Downloaded Mapillary thumbnails
│   ├── geojsons/                  # Per-point GeoJSON verification files
│   └── predicted_entrances.parquet  # Batch summary (generated)
└── src/
    ├── api.py                     # Orchestrates buildings + imagery gathering
    ├── download.py                # Downloads Overture data from S3
    ├── imagery.py                 # Fetches and saves Mapillary images
    ├── inference.py               # YOLO + ray-casting + clustering
    └── utils/
        ├── constants.py           # Overture release, category priors
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

### Single point (same as mapillary-entrances)
```bash
PYTHONPATH=. python pipeline.py \
    --input_point="37.780,-122.4092" \
    --search_radius=100 \
    --model=yolo_weights_750_image_set.pt \
    --conf=0.55 --iou=0.50 --prefer_360
```

### Batch — process places from the parquet dataset
```bash
# All 3 425 places (slow — use --limit for testing)
PYTHONPATH=. python pipeline.py \
    --from_parquet=project_d_samples.parquet \
    --limit=20 \
    --search_radius=120 \
    --model=yolo_weights_750_image_set.pt

# Hotels only
PYTHONPATH=. python pipeline.py \
    --from_parquet=project_d_samples.parquet \
    --category=hotel \
    --limit=10 \
    --model=yolo_weights_750_image_set.pt
```

---

## Ideas for Further Improvement

1. **Actual camera FOV from Mapillary metadata** — Mapillary exposes `camera_parameters` for some sequences; using true focal length instead of the assumed 45° would improve angular accuracy.

2. **Multi-ray triangulation** — Instead of accepting the first good ray-wall intersection, fire rays from every available camera that can see the building and solve for the intersection point with least-squares. More cameras = tighter estimate.

3. **Semantic segmentation** — Replace YOLO bounding boxes with a segmentation mask (SAM or Mask-RCNN) to get a per-pixel door boundary, then use the mask centroid as the anchor pixel instead of the bbox bottom-centre.

4. **Address-side prior** — The `addresses` field in the parquet contains street names. Geocoding the street and computing which building wall faces it would let the pipeline bias ray-casting toward the correct facade before YOLO runs.

5. **Temporal consistency** — Mapillary has multiple capture dates. Requiring at least two independent dates to confirm an entrance location would filter transient occlusions (parked trucks, scaffolding).

6. **Accuracy evaluation against the parquet dataset** — The `geometry` column already records where Overture thinks the place is. Computing the distribution of `distance(predicted_entrance, overture_centroid)` across the 3 425 places gives a direct precision metric without needing ground-truth surveys.

7. **Fine-tune YOLOv8 on hard negatives** — The current model was trained on 750 images. Collecting false positives from this dataset (garages, windows, signs) and retraining would reduce noise in the ray-casting stage.
