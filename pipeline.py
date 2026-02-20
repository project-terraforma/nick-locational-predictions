"""
pipeline.py — Building Entrance Prediction Pipeline
Nick Locational Predictions

Usage examples
──────────────
# Process a single lat/lon (same as mapillary-entrances):
PYTHONPATH=. python pipeline.py \
    --input_point="37.780,-122.4092" \
    --search_radius=100 \
    --model=yolo_weights_750_image_set.pt \
    --conf=0.55 --iou=0.50

# NEW: batch-process places from the Overture parquet dataset:
PYTHONPATH=. python pipeline.py \
    --from_parquet=project_d_samples.parquet \
    --limit=20 \
    --search_radius=120 \
    --model=yolo_weights_750_image_set.pt \
    --conf=0.55 --iou=0.50

# Filter to a specific category in batch mode:
PYTHONPATH=. python pipeline.py \
    --from_parquet=project_d_samples.parquet \
    --category=hotel \
    --limit=10 \
    --model=yolo_weights_750_image_set.pt
"""

import argparse
import struct
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import duckdb
import dotenv; dotenv.load_dotenv()

import visualize as _vis

from src.api import get_buildings_and_imagery_in_radius
from src.download import download_overture_radius
from src.inference import run_inference
from src.utils.io_utils import (
    write_geojson_for_verification,
    write_results_parquet,
    tlog,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_wkb_point(wkb_bytes: bytes):
    """Decode a WKB Point to (lon, lat).  Handles little-endian WKB only."""
    # byte order flag (1 byte) + type (4 bytes) + x (8 bytes) + y (8 bytes)
    if len(wkb_bytes) < 21:
        return None, None
    byte_order = wkb_bytes[0]
    if byte_order != 1:  # 1 = little-endian
        return None, None
    x = struct.unpack_from("<d", wkb_bytes, 5)[0]
    y = struct.unpack_from("<d", wkb_bytes, 13)[0]
    return x, y   # lon, lat


def load_places_from_parquet(parquet_path: str, category: str = None, limit: int = None):
    """
    Load place records from project_d_samples.parquet via DuckDB.

    Returns a list of dicts with keys:
        id, name, category, address, lon, lat, overture_confidence
    """
    con = duckdb.connect(":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")

    where_clauses = ["geometry IS NOT NULL"]
    if category:
        where_clauses.append(f"categories->>'primary' = '{category}'")

    where_sql = " AND ".join(where_clauses)
    limit_sql  = f"LIMIT {limit}" if limit else ""

    rows = con.execute(f"""
        SELECT
            id,
            geometry,
            names->>'primary'        AS name,
            categories->>'primary'   AS category,
            confidence,
            addresses
        FROM read_parquet('{parquet_path}')
        WHERE {where_sql}
        {limit_sql}
    """).fetchall()
    con.close()

    places = []
    for row in rows:
        place_id, geom_bytes, name, category_val, conf, addresses = row
        lon, lat = _parse_wkb_point(bytes(geom_bytes))
        if lon is None:
            continue

        # Extract first address freeform
        address = None
        if addresses:
            try:
                if isinstance(addresses, list) and addresses:
                    addr = addresses[0]
                    if isinstance(addr, dict):
                        parts = [addr.get("freeform"), addr.get("locality"), addr.get("region")]
                        address = ", ".join(p for p in parts if p) or None
            except Exception:
                pass

        places.append({
            "id":                  place_id,
            "name":                name or "",
            "category":            category_val or "",
            "address":             address or "",
            "lon":                 lon,
            "lat":                 lat,
            "overture_confidence": float(conf) if conf else None,
        })

    return places


# ── Single-point run ──────────────────────────────────────────────────────────

def run_single(lat: float, lon: float, args, output_dir: Path, candidates_dir: Path):
    """Process a single lat/lon — mirrors the mapillary-entrances pipeline."""
    t0 = time.perf_counter()
    print(f"\n{'='*60}")
    print(f"Point ({lat:.6f}, {lon:.6f})")
    print(f"{'='*60}")

    print("[1/3] Downloading Overture data...")
    download_overture_radius(
        lat, lon, args.search_radius,
        out_buildings="data/buildings_local.parquet",
        out_places="data/places_local.parquet",
    )

    print("[2/3] Gathering imagery...")
    data = get_buildings_and_imagery_in_radius(
        lat, lon,
        search_radius_m=args.search_radius,
        place_radius_m=args.place_radius,
        max_images_total=args.max_images,
        min_capture_date=args.min_capture_date,
        prefer_360=args.prefer_360,
        candidates_dir=candidates_dir,
    )

    if not data["image_dicts"]:
        print("[WARN] No imagery found — skipping inference.")
        return []

    print("[3/3] Running inference...")
    save_vis = str(output_dir) if getattr(args, "debug", False) else None
    entrances, buildings_lat_lon, place_names = run_inference(
        data, args.model, args.conf, args.iou, args.device, save_vis,
    )

    if not entrances:
        print("  No entrances detected.")
    else:
        for e in entrances:
            elon, elat = e["entrance"]
            conf = e.get("detection_confidence", 0.0)
            print(f"  Entrance lat={elat:.7f} lon={elon:.7f}  conf={conf:.3f}  "
                  f"n_dets={e.get('num_detections',1)}  img={Path(e['image_path']).name}")

    out_path = write_geojson_for_verification(
        entrances, buildings_lat_lon, place_names,
        output_dir=output_dir,
        output_name=f"{lat:.5f}_{lon:.5f}.geojson",
        open_browser=False,
    )
    if getattr(args, "open_browser", True):
        vis_dir = output_dir / "visualizations"
        html = _vis.build_html(out_path, candidates_dir, vis_dir if vis_dir.exists() else None)
        out_html = out_path.with_suffix(".html")
        out_html.write_text(html)
        print(f"[vis] Map: {out_html.resolve()}")
        webbrowser.open(out_html.resolve().as_uri())
    tlog("Point done", t0)
    return entrances


# ── Batch run (--from_parquet) ────────────────────────────────────────────────

def run_batch(args, output_dir: Path, candidates_dir: Path):
    """
    New mode: iterate over places in project_d_samples.parquet and run the
    entrance-prediction pipeline for each one.

    Results are accumulated into a summary parquet written at the end.
    """
    print(f"[batch] Loading places from {args.from_parquet}")
    places = load_places_from_parquet(
        args.from_parquet,
        category=args.category if args.category else None,
        limit=args.limit,
    )
    print(f"[batch] {len(places)} places to process")

    all_results = []

    for i, place in enumerate(places, 1):
        lat = place["lat"]
        lon = place["lon"]
        print(f"\n[{i}/{len(places)}] {place['name'] or place['id']}  "
              f"({lat:.6f}, {lon:.6f})  category={place['category']}")

        try:
            download_overture_radius(
                lat, lon, args.search_radius,
                out_buildings="data/buildings_local.parquet",
                out_places="data/places_local.parquet",
            )

            data = get_buildings_and_imagery_in_radius(
                lat, lon,
                search_radius_m=args.search_radius,
                place_radius_m=args.place_radius,
                max_images_total=args.max_images,
                min_capture_date=args.min_capture_date,
                prefer_360=args.prefer_360,
                candidates_dir=candidates_dir,
            )

            if not data["image_dicts"]:
                print("  No imagery — skipping.")
                continue

            save_vis = str(output_dir) if getattr(args, "debug", False) else None
            entrances, buildings_lat_lon, place_names = run_inference(
                data, args.model, args.conf, args.iou, args.device, save_vis,
            )

            # Write per-place GeoJSON
            write_geojson_for_verification(
                entrances, buildings_lat_lon, place_names,
                output_dir=output_dir,
                output_name=f"{lat:.5f}_{lon:.5f}.geojson",
                open_browser=False,  # don't open browser for every place in batch
            )

            # Accumulate results for the summary parquet
            for e in entrances:
                elon, elat = e["entrance"]
                all_results.append({
                    "place_id":             place["id"],
                    "place_name":           place["name"],
                    "category":             place["category"],
                    "address":              place["address"],
                    "overture_lon":         lon,
                    "overture_lat":         lat,
                    "entrance_lon":         elon,
                    "entrance_lat":         elat,
                    "detection_confidence": e.get("detection_confidence"),
                    "num_detections":       e.get("num_detections"),
                    "building_id":          e.get("bid"),
                })

        except Exception as exc:
            print(f"  [ERROR] {exc}")
            continue

    # Write summary
    if all_results:
        write_results_parquet(all_results, out_path=str(output_dir / "predicted_entrances.parquet"))
        print(f"\n[batch] Done. {len(all_results)} entrances found across {len(places)} places.")
    else:
        print("\n[batch] Done. No entrances detected.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Building entrance prediction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input — one of these two modes
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--input_point", type=str,
                      help="Single point mode: lat,lon  e.g. 37.780,-122.409")
    mode.add_argument("--from_parquet", type=str,
                      help="Batch mode: path to project_d_samples.parquet")

    # Batch-mode filters
    p.add_argument("--category", type=str, default=None,
                   help="(batch) Filter to a specific Overture place category e.g. hotel")
    p.add_argument("--limit", type=int, default=None,
                   help="(batch) Maximum number of places to process")

    # Search parameters
    p.add_argument("--search_radius", type=int, default=120,
                   help="Radius in metres around each point for buildings + imagery")
    p.add_argument("--place_radius", type=int, default=120,
                   help="Radius in metres for joining buildings to places")
    p.add_argument("--max_images", type=int, default=50,
                   help="Maximum Mapillary images to download per point")
    p.add_argument("--prefer_360", action="store_true",
                   help="Prefer 360° panoramic images when available")
    p.add_argument("--min_capture_date", type=str, default=None,
                   help="Minimum image capture date YYYY-MM-DD")

    # Inference parameters
    p.add_argument("--model", type=str, required=True,
                   help="Path to YOLOv8 weights file (auto-downloaded if missing)")
    p.add_argument("--device", type=str, default="cpu",
                   help="Inference device: cpu or cuda")
    p.add_argument("--conf", type=float, default=0.35,
                   help="YOLO confidence threshold")
    p.add_argument("--iou", type=float, default=0.5,
                   help="YOLO IOU threshold for NMS")

    # Output
    p.add_argument("--save", type=str, default="outputs",
                   help="Output directory for GeoJSON and candidate images")
    p.add_argument("--open_browser", action="store_true", default=True,
                   help="Open the interactive visualize.py map in the browser after a single-point run")
    p.add_argument("--debug", action="store_true",
                   help="Save YOLO visualizations for every image (not just hits) "
                        "so you can see what the detector saw")

    return p


def main():
    args = build_parser().parse_args()

    # Each run gets its own timestamped folder so outputs never overwrite each other
    run_dir = Path(args.save) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] Output directory: {run_dir.resolve()}")

    if args.from_parquet:
        run_batch(args, run_dir, candidates_dir)
    elif args.input_point:
        lat_str, lon_str = args.input_point.split(",")
        run_single(float(lat_str), float(lon_str), args, run_dir, candidates_dir)
    else:
        build_parser().error("Provide --input_point or --from_parquet")


if __name__ == "__main__":
    main()
