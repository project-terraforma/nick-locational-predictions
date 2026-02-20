"""
inference.py

Improvements over mapillary-entrances:
  1. All entrance clusters per building are returned (not just one).
  2. Each cluster carries a `detection_confidence` score propagated from
     YOLO confidence × geometric quality (lateral-error ratio).
  3. Weighted-mean clustering: cluster centres are the confidence-weighted
     mean of member positions, giving more accurate final coordinates.
  4. `yolo_conf` is attached to every candidate before geometry so it flows
     through clamp_entrance_to_hit_segment into the final record.
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from src.utils.inference_utils import (
    make_local_proj,
    to_local_xy,
    filter_images_by_quality,
    load_yolo_model,
    detect_entrances_in_image,
    extract_bbox_coordinates,
    get_fov_half_angle,
    match_entrance_to_building,
    select_exterior_seg,
    clamp_entrance_to_hit_segment,
)
from src.utils.geo_utils import _haversine

try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except Exception:
    _HAS_ULTRALYTICS = False


def run_inference(
    data: Dict[str, Any],
    yolo_weights: str,
    conf: float,
    iou: float,
    device: str,
    save_vis,
) -> Tuple[List[Dict], Dict, Dict]:
    """
    Run the full entrance-detection pipeline on the gathered data dict.

    Returns
    -------
    building_entrances : list of dicts, one per detected cluster.
        Each dict has keys: bid, entrance (lon,lat), detection_confidence,
        num_detections, snapped, image_path, hit, wall_segment.
    buildings_lat_lon  : {building_id: [[lon,lat], ...]}
    place_names        : {building_id: place_info_dict or None}
    """
    if not _HAS_ULTRALYTICS:
        raise SystemExit("ERROR: ultralytics not installed. pip install ultralytics")

    centroid_lon = data["input_coordinates"][0]
    centroid_lat = data["input_coordinates"][1]
    all_images   = data["image_dicts"]
    buildings_lat_lon = data["building_polygons"]
    place_names  = data["places"]

    # Quality filter
    print(f"  Images before quality filter: {len(all_images)}")
    all_images = filter_images_by_quality(all_images)
    print(f"  Images after  quality filter: {len(all_images)}")

    if not all_images:
        return [], buildings_lat_lon, place_names

    # Local flat-earth projection centred on the search area
    proj_local = make_local_proj(centroid_lat, centroid_lon)

    # Convert building polygons to local (x,y)
    buildings_xy: Dict[str, List] = {}
    for bid, polygon in buildings_lat_lon.items():
        buildings_xy[bid] = [to_local_xy(v[0], v[1], proj_local) for v in polygon]

    # Load YOLO once
    model = load_yolo_model(yolo_weights, device=device)

    raw_entrances: List[Dict] = []

    for img in all_images:
        path = img["image_path"]
        dets = detect_entrances_in_image(path, model, conf_thr=conf, iou_thr=iou,
                                         device=device, save_dir=save_vis)
        if not dets:
            continue

        if len(dets) > 1:
            print(f"  {len(dets)} detections in {path}")

        for det in dets:
            yolo_conf = det["conf"]
            C, dir_xy = extract_bbox_coordinates(img, det["bbox"], proj_local, get_fov_half_angle(img))

            bid, candidates = match_entrance_to_building((C, dir_xy), buildings_xy)
            if bid is None or not candidates:
                continue

            best = select_exterior_seg(candidates, C)
            if best is None:
                continue

            raw_entrances.append({
                "camera_xy":    C,
                "bid":          bid,
                "image_path":   path,
                "hit":          best["hit"],
                "wall_segment": (best["segment"][0].tolist(), best["segment"][1].tolist()),
                "yolo_conf":    yolo_conf,   # ← NEW: carry YOLO confidence forward
            })

    print(f"  Raw entrance candidates: {len(raw_entrances)}")

    if not raw_entrances:
        return [], buildings_lat_lon, place_names

    # Snap each hit onto its wall segment and compute detection_confidence
    raw_entrances = clamp_entrance_to_hit_segment(proj_local, raw_entrances)
    print(f"  After lateral-error filter: {len(raw_entrances)}")

    # ── Weighted-mean clustering per building ─────────────────────────────────
    #
    # Improvement: instead of a simple mean, we weight each detection's position
    # by its detection_confidence when computing the cluster centre.  This reduces
    # the influence of low-quality detections on the final coordinate.
    #
    # All clusters are preserved (mapillary-entrances only kept 1 per building).

    groups: Dict[str, List] = defaultdict(list)
    for ent in raw_entrances:
        if ent.get("bid") and ent.get("entrance"):
            groups[ent["bid"]].append(ent)

    building_entrances: List[Dict] = []
    for bid, items in groups.items():
        clusters: List[List] = []

        for it in items:
            lon, lat = it["entrance"]
            placed = False
            for cl in clusters:
                rep_lon, rep_lat = cl[0]["entrance"]
                if _haversine(lat, lon, rep_lat, rep_lon) < 5.0:
                    cl.append(it)
                    placed = True
                    break
            if not placed:
                clusters.append([it])

        for cl in clusters:
            weights = np.array([max(1e-6, it.get("detection_confidence", 1.0)) for it in cl])
            lons = np.array([it["entrance"][0] for it in cl])
            lats = np.array([it["entrance"][1] for it in cl])

            # Confidence-weighted mean position
            w_sum  = weights.sum()
            lon_wm = float((lons * weights).sum() / w_sum)
            lat_wm = float((lats * weights).sum() / w_sum)

            # Cluster confidence = mean of member confidences (simple mean — the
            # weighting already determined position; a simple mean is more honest
            # for the confidence score itself)
            cluster_conf = float(weights.mean())

            rep = cl[0]
            building_entrances.append({
                "bid":                  bid,
                "entrance":             (lon_wm, lat_wm),
                "detection_confidence": round(cluster_conf, 4),
                "num_detections":       len(cl),
                "snapped":              any(it.get("snapped", False) for it in cl),
                "image_path":           rep.get("image_path"),
                "hit":                  rep.get("hit"),
                "wall_segment":         rep.get("wall_segment"),
            })

    print(f"  Final entrance clusters: {len(building_entrances)}")
    return building_entrances, buildings_lat_lon, place_names
