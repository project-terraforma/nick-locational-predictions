import os
import time
from pathlib import Path
from typing import Optional, List, Dict
from src.utils.io_utils import _ensure_dir, tlog
from src.utils.mapillary_utils import fetch_images, download_image, _is_360


def fetch_and_slice_for_building(
    building_row,
    radius_m: int,
    min_capture_date: Optional[str],
    max_images_per_building: int,
    prefer_360: bool,
) -> List[Dict]:
    """
    Fetch Mapillary images near a building centroid, download thumbnails,
    and return a list of metadata dicts.
    """
    lat = building_row["lat"]
    lon = building_row["lon"]
    building_id = building_row["id"]

    token = os.getenv("MAPILLARY_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("MAPILLARY_ACCESS_TOKEN missing from environment.")

    out_dir = Path("outputs/candidates/")
    _ensure_dir(out_dir)

    t0 = time.perf_counter()
    print(f"  Building {building_id} ({lat:.6f}, {lon:.6f}), radius {radius_m} m")

    imgs = fetch_images(
        token=token,
        lat=lat,
        lon=lon,
        radius_m=radius_m,
        fields=["id", "computed_geometry", "captured_at", "compass_angle",
                "thumb_1024_url", "camera_type"],
        prefer_360=prefer_360,
        min_capture_date_filter=min_capture_date,
    )

    if not imgs:
        print("    no images nearby")
        return []

    imgs = imgs[:max_images_per_building]

    saved = []
    for img in imgs:
        img_id = img.get("id")
        img_path = out_dir / f"{img_id}.jpg"
        download_image(img.get("thumb_1024_url"), img_path)
        saved.append({
            "id": img_id,
            "path": str(img_path),
            "coordinates": img.get("computed_geometry", {}).get("coordinates"),
            "compass_angle": img.get("compass_angle"),
            "camera_type": img.get("camera_type"),
            "is_360": _is_360(img),
            "captured_at": img.get("captured_at"),
        })

    tlog("Building done", t0)
    return saved
