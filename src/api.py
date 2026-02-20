from typing import Optional, Dict, Any, List

from src.utils.geo_utils import simple_radius_to_bbox
from src.utils.matching_utils import resolve_sources, select_best_place_for_building
from src.utils.duck_db_utils import get_buildings, join_buildings_places
from src.utils.polygon_utils import polygon_vertices_from_wkt
from src.utils.mapillary_utils import _extract_lon_lat
from src.imagery import fetch_and_slice_for_building


def get_buildings_and_imagery_in_radius(
    lat: float,
    lon: float,
    search_radius_m: int,
    place_radius_m: int,
    max_images_total: int,
    min_capture_date: Optional[str],
    prefer_360: bool,
) -> Dict[str, Any]:
    """
    Find all buildings within search_radius_m, join with nearby places,
    fetch a shared set of Mapillary images, and return a unified dict
    ready for run_inference().

    Returns
    -------
    {
        "input_coordinates": [lon, lat],
        "building_polygons": {bid: [[lon,lat], ...]},
        "building_walls":    {bid: [[[lon,lat],[lon,lat]], ...]},
        "places":            {bid: place_dict or None},
        "image_dicts":       [{...}, ...]
    }
    """
    bbox = simple_radius_to_bbox(lon, lat, meters=search_radius_m)
    b_src, p_src = resolve_sources(bbox)
    bdf = get_buildings(bbox, b_src, limit_hint=200)

    if bdf is None or len(bdf) == 0:
        print("[WARN] No buildings found in radius.")
        return {
            "input_coordinates": [lon, lat],
            "building_polygons": {},
            "building_walls": {},
            "places": {},
            "image_dicts": [],
        }

    print(f"[api] {len(bdf)} buildings within {search_radius_m} m")

    links = join_buildings_places(bdf, bbox, p_src, radius_m=place_radius_m)

    building_polygons: Dict[str, List] = {}
    building_walls:    Dict[str, List] = {}
    building_places:   Dict[str, Any]  = {}

    for _, b in bdf.iterrows():
        bid = b["id"]
        polygon = polygon_vertices_from_wkt(b["wkt"])
        building_polygons[bid] = polygon

        walls = []
        if len(polygon) >= 2:
            for i in range(len(polygon)):
                walls.append([polygon[i], polygon[(i + 1) % len(polygon)]])
        building_walls[bid] = walls

        best_place = None
        if "building_id" in links.columns:
            subset = links[links["building_id"] == bid]
            if len(subset) > 0:
                best_place = select_best_place_for_building(
                    subset, building_id=bid, max_dist_m=place_radius_m
                )
        building_places[bid] = best_place

    # Fetch imagery once for the whole search area
    print(f"[api] Fetching imagery around ({lat:.6f}, {lon:.6f}) r={search_radius_m} m")
    temp_building = {"id": "shared_area", "lat": lat, "lon": lon, "wkt": None}
    saved = fetch_and_slice_for_building(
        temp_building,
        radius_m=search_radius_m,
        min_capture_date=min_capture_date,
        max_images_per_building=max_images_total,
        prefer_360=prefer_360,
    )

    image_data: List[Dict[str, Any]] = []
    for rec in (saved or []):
        lo, la = _extract_lon_lat(rec, lon, lat)
        image_data.append({
            "image_path":   rec.get("path") or rec.get("jpg_path"),
            "compass_angle": rec.get("compass_angle"),
            "coordinates":  [lo, la],
            "is_360":       rec.get("is_360", False),
            "camera_type":  rec.get("camera_type"),
        })

    return {
        "input_coordinates": [lon, lat],
        "building_polygons": building_polygons,
        "building_walls":    building_walls,
        "places":            building_places,
        "image_dicts":       image_data,
    }
