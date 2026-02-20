"""
inference_utils.py

Core geometric algorithms for entrance detection and map-pin alignment.

Improvements over mapillary-entrances:
  - Every ray-cast hit carries a `detection_confidence` score
    (YOLO confidence × geometric quality) that propagates all the way
    through clustering and into the final GeoJSON output.
  - Geometric quality is defined as 1 - (lateral_error / MAX_LATERAL_ERROR_M),
    so detections with a clean ray-wall intersection score highest.
  - Weighted-mean clustering: cluster centres are the confidence-weighted
    mean of member positions rather than a simple arithmetic mean.
"""

import math
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

import cv2
import numpy as np
from pyproj import CRS, Transformer
from huggingface_hub import hf_hub_download

from .constants import (
    MAX_LATERAL_ERROR_M, ENTRANCE_OFFSET_M, MIN_RAY_DISTANCE_M, MAX_RAY_DISTANCE_M,
    SHARPNESS_THRESH, DEFAULT_HFOV_DEG,
)

try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except Exception:
    _HAS_ULTRALYTICS = False


# ── Coordinate projection ─────────────────────────────────────────────────────

def _is_360(img: dict) -> bool:
    ct = (img.get("camera_type") or "").lower()
    return ct in {"spherical", "equirectangular", "panorama", "panoramic", "360"}


def make_local_proj(lat0, lon0):
    """Create an Azimuthal Equidistant projection centred on (lat0, lon0).
    All coordinates within the search radius are treated as flat 2-D (x, y) in metres.
    """
    return CRS.from_user_input(
        f"+proj=aeqd +lon_0={float(lon0)} +lat_0={float(lat0)} +ellps=WGS84 +units=m +no_defs"
    )


def to_local_xy(lon, lat, crs_local) -> np.ndarray:
    t = Transformer.from_crs("EPSG:4326", crs_local, always_xy=True)
    x, y = t.transform(float(lon), float(lat))
    return np.array([x, y])


def to_lonlat_xy(xy, crs_local) -> Tuple[float, float]:
    x, y = map(float, xy)
    t = Transformer.from_crs(crs_local, "EPSG:4326", always_xy=True)
    lon, lat = t.transform(x, y)
    return float(lon), float(lat)


# ── Image quality filters ─────────────────────────────────────────────────────

def is_sharp(img: np.ndarray, thresh: float = SHARPNESS_THRESH) -> bool:
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(img, cv2.CV_64F).var() > thresh


def is_well_exposed(img: np.ndarray, dark_thresh: float = 0.05, bright_thresh: float = 0.95) -> bool:
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flat = img.flatten() / 255.0
    return (flat < dark_thresh).mean() < 0.5 and (flat > bright_thresh).mean() < 0.5


def filter_images_by_quality(
    images: List,
    sharpness_thresh: float = 100.0,
    dark_thresh: float = 0.05,
    bright_thresh: float = 0.95,
) -> List:
    good = []
    for img_meta in images:
        img = cv2.imread(img_meta["image_path"])
        if img is None:
            print(f"  [WARN] could not read {img_meta['image_path']}")
            continue
        if is_sharp(img, sharpness_thresh) and is_well_exposed(img, dark_thresh, bright_thresh):
            good.append(img_meta)
    return good


# ── YOLO model ────────────────────────────────────────────────────────────────

def load_yolo_model(model_path: str, device: Optional[str] = None):
    """Load YOLOv8, auto-downloading weights from Hugging Face if missing."""
    FILE = Path(model_path).name
    LOCAL = Path("./") / FILE

    if not LOCAL.exists():
        print("Downloading YOLO weights from Hugging Face...")
        downloaded = hf_hub_download(
            repo_id="erantala1/yolov8s-entrance-detector",
            filename=FILE,
        )
        LOCAL.write_bytes(Path(downloaded).read_bytes())
        print(f"Saved to {LOCAL.resolve()}")

    if not _HAS_ULTRALYTICS:
        raise RuntimeError("ultralytics not installed. `pip install ultralytics`")

    model = YOLO(str(LOCAL))
    if device is not None:
        model.overrides = model.overrides or {}
        model.overrides["device"] = device
    return model


def run_yolo_on_image(
    model,
    img: np.ndarray,
    conf_thr: float = 0.35,
    iou_thr: float = 0.5,
    device: Optional[str] = None,
) -> List[Dict]:
    """Run YOLO and return [{'conf', 'bbox', 'cls_id', 'cls_name'}, ...]."""
    results = model.predict(source=img, conf=conf_thr, iou=iou_thr, verbose=False, device=device)
    dets: List[Dict] = []
    if not results:
        return dets
    res = results[0]
    if res.boxes is None:
        return dets
    names = res.names
    for b in res.boxes:
        xyxy = b.xyxy[0].cpu().numpy().astype(int)
        conf = float(b.conf[0].cpu().numpy())
        cls_id = int(b.cls[0].cpu().numpy()) if b.cls is not None else -1
        dets.append({
            "conf": conf,
            "bbox": tuple(xyxy.tolist()),
            "cls_id": cls_id,
            "cls_name": names.get(cls_id, str(cls_id)),
        })
    return dets


def _draw_dets(img: np.ndarray, dets: List[Dict], color=(0, 200, 0)) -> np.ndarray:
    vis = img.copy()
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{d.get('cls_name','obj')} {d['conf']:.2f}",
                    (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


def detect_entrances_in_image(
    image_path: str,
    model,
    conf_thr: float = 0.5,
    iou_thr: float = 0.5,
    device: Optional[str] = None,
    save_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Run YOLO on a single image and return a list of detection dicts that
    include the YOLO confidence score for downstream propagation.
    Returns [] when nothing is detected.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [WARN] could not read {image_path}")
        return []

    dets = run_yolo_on_image(model, img, conf_thr=conf_thr, iou_thr=iou_thr, device=device)

    if dets and save_dir:
        vis_dir = Path(save_dir) / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis = _draw_dets(img, dets)
        out = vis_dir / f"{Path(image_path).stem}_vis.jpg"
        cv2.imwrite(str(out), vis)

    return dets


# ── Camera model ──────────────────────────────────────────────────────────────

def get_fov_half_angle(img_dict: dict) -> float:
    """Return the full horizontal FOV in degrees for this image's camera.

    For 360° equirectangular images this is always 360° (90° returned as the
    half-angle convention used by horizontal_fov_to_fx is not applied here —
    the caller passes this directly as hfov_deg).

    For perspective cameras we derive the true FOV from the Mapillary
    camera_parameters field when available.  camera_parameters[0] is the
    normalized focal length fx / max(W, H).  Falling back to DEFAULT_HFOV_DEG
    (70°) is much more realistic than the previous hardcoded 45°.
    """
    if _is_360(img_dict) or img_dict.get("is_360", False):
        return 90.0

    params = img_dict.get("camera_parameters")
    W = img_dict.get("width")
    H = img_dict.get("height")
    if params and W and H and len(params) >= 1:
        fx_norm = params[0]
        fx_px = float(fx_norm) * max(int(W), int(H))
        if fx_px > 0:
            # Full horizontal FOV = 2 * atan(W/2 / fx_px)
            return math.degrees(2.0 * math.atan(int(W) / (2.0 * fx_px)))

    return DEFAULT_HFOV_DEG  # 70° — realistic default for street cameras


def horizontal_fov_to_fx(img_w: int, hfov_deg: float) -> float:
    """Pinhole focal length from image width and horizontal FOV."""
    return (img_w * 0.5) / math.tan(math.radians(hfov_deg * 0.5))


def extract_bbox_coordinates(
    image_dict: Dict,
    bbox,
    proj_local,
    hfov_deg: float = 45.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the 2-D camera ray (origin C, unit direction d) in local AEQD coordinates
    for a detection bounding box.

    Uses bottom-centre of the bbox as the pixel anchor (door threshold).
    """
    cam_lon = image_dict["coordinates"][0]
    cam_lat = image_dict["coordinates"][1]
    C = to_local_xy(cam_lon, cam_lat, proj_local)

    img = cv2.imread(str(image_dict["image_path"]), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]

    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        u = 0.5 * (x1 + x2)
    else:
        xc, yc, bw, bh = bbox
        u = float(xc)

    if _is_360(image_dict):
        yaw_offset_deg = (u / W) * 360.0 - 180.0
    else:
        fx = horizontal_fov_to_fx(W, hfov_deg)
        yaw_offset_deg = math.degrees(math.atan2(u - 0.5 * W, fx))

    compass_deg = float(image_dict.get("compass_angle") or 0.0)
    bearing_deg = (compass_deg + yaw_offset_deg) % 360.0

    theta = math.radians(bearing_deg)
    d = np.array([math.sin(theta), math.cos(theta)], dtype=float)
    d /= np.linalg.norm(d)

    return C, d


# ── Ray-segment intersection ──────────────────────────────────────────────────

def _ray_segment_intersection(
    C: np.ndarray,
    d: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    t_min: float = MIN_RAY_DISTANCE_M,
) -> Tuple[Optional[np.ndarray], Optional[float], bool]:
    """
    Solve C + t*d = A + u*(B-A).
    Returns (hit_xy, t, ok).  ok=True iff t >= t_min and u in [0,1].
    """
    AB = B - A
    M = np.stack([d, -AB], axis=1)
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) < 1e-12:
        return None, None, False
    inv = np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]], dtype=float) / det
    t, u = inv @ (A - C)
    if t >= t_min and 0.0 <= u <= 1.0:
        return C + t * d, float(t), True
    return None, None, False


def match_entrance_to_building(
    ray: Tuple[np.ndarray, np.ndarray],
    buildings_xy: Dict,
    max_range_m: float = MAX_RAY_DISTANCE_M,
    spread_deg: float = 3.0,
) -> Tuple[Optional[str], Optional[List]]:
    """
    Cast three slightly-spread rays and intersect against all building wall segments.
    Returns the building ID with the best-scoring intersection and its candidate list.

    The ±spread_deg fan accounts for minor compass calibration errors.
    """
    C, d = ray
    base_bearing = math.atan2(d[0], d[1])
    best_bid = None
    best_score = float("inf")
    building_candidates: Dict[str, List] = {}

    for delta in (-spread_deg, 0.0, spread_deg):
        bearing = base_bearing + math.radians(delta)
        d_rot = np.array([math.sin(bearing), math.cos(bearing)], dtype=float)

        for bid, poly in buildings_xy.items():
            poly = np.asarray(poly, float)
            centroid = poly.mean(axis=0)

            if np.dot(centroid - C, d_rot) <= 0:
                continue  # building is behind camera

            for i in range(len(poly)):
                A = poly[i]
                B = poly[(i + 1) % len(poly)]

                hit, t, ok = _ray_segment_intersection(C, d_rot, A, B)
                if not ok or t > max_range_m:
                    continue

                # Only accept front-facing walls (camera on exterior side)
                wall = B - A
                norm = np.linalg.norm(wall)
                if norm < 1e-9:
                    continue
                wall /= norm
                n = np.array([wall[1], -wall[0]])
                if np.dot(n, C - A) < 0:
                    n = -n
                if np.dot(d_rot, n) >= 0:
                    continue  # back-facing — skip

                lateral = abs(np.cross(d_rot, hit - C)) / np.linalg.norm(hit - C)
                score = t + 2.0 * lateral
                building_candidates.setdefault(bid, []).append({
                    "hit": hit,
                    "segment": (A.copy(), B.copy()),
                    "t": t,
                    "lateral": lateral,
                    "score": score,
                })
                if score < best_score:
                    best_score = score
                    best_bid = bid

    if best_bid is None:
        return None, None
    return best_bid, building_candidates[best_bid]


def select_exterior_seg(candidates: List[Dict], camera_xy: np.ndarray) -> Optional[Dict]:
    """Select the exterior-facing wall segment with the best combined score."""
    best = None
    best_score = float("inf")
    for c in candidates:
        A, B = np.asarray(c["segment"][0], float), np.asarray(c["segment"][1], float)
        mid = 0.5 * (A + B)
        wall = B - A
        wall /= np.linalg.norm(wall)
        n = np.array([wall[1], -wall[0]])
        if np.dot(n, camera_xy - mid) < 0:
            n = -n
        if np.dot(n, camera_xy - mid) <= 0:
            continue  # camera is on wrong side
        dist_cam = np.linalg.norm(mid - camera_xy)
        score = c["t"] + 1.5 * c["lateral"] + 0.05 * dist_cam
        if score < best_score:
            best_score = score
            best = c
    return best


# ── Point snapping ────────────────────────────────────────────────────────────

def snap_point_to_segment(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    AP, AB = P - A, B - A
    t = float(np.dot(AP, AB) / np.dot(AB, AB))
    return A + max(0.0, min(1.0, t)) * AB


def point_line_distance_segment(P, A, B) -> float:
    P, A, B = (np.asarray(x, float).reshape(2) for x in (P, A, B))
    AB = B - A
    d2 = np.dot(AB, AB)
    if d2 < 1e-12:
        return float(np.linalg.norm(P - A))
    t = max(0.0, min(1.0, float(np.dot(P - A, AB) / d2)))
    return float(np.linalg.norm(P - (A + t * AB)))


def outward_normal(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    wall = (B - A) / np.linalg.norm(B - A)
    n = np.array([wall[1], -wall[0]])
    return n if np.dot(n, C - A) > 0 else -n


def clamp_entrance_to_hit_segment(
    proj_local,
    building_entrances: List[Dict],
    outward_offset_m: float = ENTRANCE_OFFSET_M,
    max_lateral_error_m: float = MAX_LATERAL_ERROR_M,
) -> List[Dict]:
    """
    Project each candidate hit onto the wall segment, apply the outward offset,
    convert back to WGS-84, and compute a geometric quality score.

    Improvement over mapillary-entrances:
    Each result carries a `geo_quality` value in [0, 1] derived from how cleanly
    the ray intersected the wall.  This is later multiplied by the YOLO confidence
    to give a final `detection_confidence` score for the entrance.
    """
    clamped = []
    for ent in building_entrances:
        P = np.asarray(ent["hit"], float)
        C = np.asarray(ent["camera_xy"], float)
        A = np.asarray(ent["wall_segment"][0], float)
        B = np.asarray(ent["wall_segment"][1], float)

        lateral = point_line_distance_segment(P, A, B)
        if lateral > max_lateral_error_m:
            continue  # too far off the wall

        P_wall = snap_point_to_segment(P, A, B)
        n = outward_normal(A, B, C)
        P_out = P_wall + outward_offset_m * n

        lon, lat = to_lonlat_xy(P_out, proj_local)

        # Geometric quality: 1.0 = ray hits wall perfectly on-centre, 0.0 = at the tolerance limit
        geo_quality = max(0.0, 1.0 - lateral / max_lateral_error_m)

        ent_out = dict(ent)
        ent_out["entrance"]    = (lon, lat)
        ent_out["entrance_xy"] = P_out
        ent_out["snapped"]     = True
        ent_out["geo_quality"] = geo_quality
        # Combined detection confidence = YOLO confidence × geometric quality
        ent_out["detection_confidence"] = float(ent.get("yolo_conf", 1.0)) * geo_quality

        clamped.append(ent_out)

    return clamped
