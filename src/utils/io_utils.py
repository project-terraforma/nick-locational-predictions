"""
io_utils.py

Improvements over mapillary-entrances:
  - Entrance GeoJSON features include detection_confidence and num_detections.
  - Place features include name, primary category, and street address.
  - Entrance markers are colour-graded from red (low confidence) to green
    (high confidence) so map reviewers can triage results at a glance.
  - write_results_parquet() lets the batch pipeline save all results as a
    single queryable file alongside the source project_d_samples.parquet.
"""

import json
import time
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def tlog(label: str, t0: float) -> float:
    t1 = time.perf_counter()
    print(f"[{t1 - t0:6.2f}s] {label}", flush=True)
    return t1


def _confidence_to_hex(conf: float) -> str:
    """
    Linearly interpolate from red (#ff0000) at conf=0 to green (#00cc00) at conf=1.
    Gives map reviewers an instant visual signal about detection quality.
    """
    conf = max(0.0, min(1.0, float(conf)))
    r = int(255 * (1.0 - conf))
    g = int(204 * conf)
    return f"#{r:02x}{g:02x}00"


def _extract_address(addresses) -> Optional[str]:
    """Pull a readable address string from the Overture addresses list."""
    if not addresses:
        return None
    if isinstance(addresses, list) and addresses:
        addr = addresses[0]
        if isinstance(addr, dict):
            parts = [addr.get("freeform"), addr.get("locality"), addr.get("region")]
            return ", ".join(p for p in parts if p)
    return None


def write_geojson_for_verification(
    building_entrances: List[Dict[str, Any]],
    buildings_lat_lon: Dict[str, List[List[float]]],
    place_names: Dict[str, Dict[str, Any]],
    output_dir: Path,
    output_name: str,
    open_browser: bool = True,
) -> Path:
    """
    Generate a GeoJSON FeatureCollection and optionally open it in geojson.io.

    Layer contents
    ──────────────
    Blue polygons  : building footprints
    Confidence-coloured stars : predicted entrances
      colour ranges from red (low confidence) → green (high confidence)
    Green circles  : Overture place centroids (original map pin locations)
    """
    features = []

    # ── Building polygons ──────────────────────────────────────────────────
    for bid, polygon in buildings_lat_lon.items():
        if polygon and polygon[0] != polygon[-1]:
            polygon = polygon + [polygon[0]]
        features.append({
            "type": "Feature",
            "properties": {
                "name": f"Building {bid}",
                "stroke": "#1f77b4",
                "stroke-width": 2,
                "fill": "#1f77b4",
                "fill-opacity": 0.08,
            },
            "geometry": {"type": "Polygon", "coordinates": [polygon]},
        })

    # ── Predicted entrance points ──────────────────────────────────────────
    for dic in building_entrances:
        bid      = dic.get("bid")
        entrance = dic.get("entrance")
        if not bid or not entrance or len(entrance) != 2:
            continue

        lon, lat     = float(entrance[0]), float(entrance[1])
        conf         = dic.get("detection_confidence", 0.0)
        n_dets       = dic.get("num_detections", 1)
        colour       = _confidence_to_hex(conf)
        place        = place_names.get(bid) or {}
        place_name   = place.get("name") or ""

        features.append({
            "type": "Feature",
            "properties": {
                "name":                 f"Entrance — {place_name or bid}",
                "marker-color":         colour,
                "marker-symbol":        "star",
                "marker-size":          "medium",
                "detection_confidence": round(conf, 4),
                "num_detections":       n_dets,
                "building_id":          bid,
                "place_name":           place_name,
                "source_image":         Path(dic.get("image_path", "")).name,
            },
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
        })

    # ── Original place centroids ───────────────────────────────────────────
    for bid, place in place_names.items():
        if not isinstance(place, dict):
            continue
        plon = place.get("lon")
        plat = place.get("lat")
        if plon is None or plat is None:
            continue

        categories = place.get("categories") or {}
        primary_cat = (
            categories.get("primary", "") if isinstance(categories, dict) else ""
        )
        address = _extract_address(place.get("addresses"))

        features.append({
            "type": "Feature",
            "properties": {
                "name":         place.get("name", ""),
                "marker-color": "#00cc00",
                "marker-symbol": "circle",
                "marker-size":  "small",
                "category":     primary_cat,
                "address":      address or "",
                "overture_confidence": place.get("confidence", ""),
            },
            "geometry": {"type": "Point", "coordinates": [float(plon), float(plat)]},
        })

    geojson_data = {"type": "FeatureCollection", "features": features}

    out_path = Path(output_dir) / "geojsons" / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(geojson_data, f, indent=2)
    print(f"[OK] GeoJSON written: {out_path.resolve()}")

    if open_browser:
        encoded = urllib.parse.quote(json.dumps(geojson_data))
        webbrowser.open(f"https://geojson.io/#data=data:application/json,{encoded}")

    return out_path


def write_results_parquet(
    results: List[Dict[str, Any]],
    out_path: str = "outputs/predicted_entrances.parquet",
) -> None:
    """
    Save all predicted entrances from a batch run as a single parquet file.

    Columns: place_id, place_name, category, address, overture_lon, overture_lat,
             entrance_lon, entrance_lat, detection_confidence, num_detections, building_id
    """
    if not results:
        print("[WARN] No results to write.")
        return

    rows = []
    for r in results:
        rows.append({
            "place_id":             r.get("place_id"),
            "place_name":           r.get("place_name"),
            "category":             r.get("category"),
            "address":              r.get("address"),
            "overture_lon":         r.get("overture_lon"),
            "overture_lat":         r.get("overture_lat"),
            "entrance_lon":         r.get("entrance_lon"),
            "entrance_lat":         r.get("entrance_lat"),
            "detection_confidence": r.get("detection_confidence"),
            "num_detections":       r.get("num_detections"),
            "building_id":          r.get("building_id"),
        })

    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[OK] Results parquet written: {Path(out_path).resolve()} ({len(df)} rows)")
