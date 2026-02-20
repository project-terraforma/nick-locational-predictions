from pathlib import Path
import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config

DEFAULT_RELEASE = "2025-12-17.0"

def get_latest_overture_release(bucket="overturemaps-us-west-2", prefix="release/") -> str:
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/")
    releases = [
        cp["Prefix"].replace(prefix, "").strip("/")
        for page in pages
        for cp in page.get("CommonPrefixes", [])
    ]
    if not releases:
        raise RuntimeError("No Overture releases found")
    return sorted(releases)[-1]

def resolve_overture_release() -> str:
    try:
        return get_latest_overture_release()
    except Exception as e:
        print(f"[WARN] Could not fetch latest Overture release ({e}). Falling back to {DEFAULT_RELEASE}.")
        return DEFAULT_RELEASE

RELEASE = resolve_overture_release()

EARTH_RADIUS_M = 6378137

LOCAL_BUILDINGS = Path("data/buildings_local.parquet")
LOCAL_PLACES    = Path("data/places_local.parquet")

S3_BUILDINGS = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=buildings/type=building/*"
S3_PLACES    = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=places/type=place/*"

# Expanded category priors tuned for the project_d_samples.parquet dataset.
# Values > 1.0 mean the category tends to have a clearly visible, prominent entrance.
# Values < 1.0 mean the category is less likely to have a street-facing formal entrance.
_CAT_PRIOR = {
    # Cultural / civic
    "art_museum": 1.2,
    "museum": 1.1,
    "library": 1.1,
    "city_hall": 1.2,
    "government": 1.15,
    "place_of_worship": 1.1,
    # Education
    "school": 1.1,
    "university": 1.1,
    "day_care_preschool": 1.05,
    # Hospitality — hotels tend to have prominent, camera-visible entrances
    "hotel": 1.2,
    "resort": 1.15,
    "accommodation": 1.1,
    "campground": 0.8,   # typically no formal street entrance
    # Food & drink
    "restaurant": 1.05,
    "fast_food_restaurant": 1.0,
    "pizza_restaurant": 1.0,
    "mexican_restaurant": 1.0,
    "coffee_shop": 1.0,
    # Retail / services
    "convenience_store": 1.05,
    "pharmacy": 1.1,
    "advertising_agency": 1.0,
    "professional_services": 1.0,
    "lawyer": 1.0,
    "event_planning": 1.0,
    "self_storage_facility": 0.9,  # usually rolling doors, not pedestrian
    # Health
    "health_and_medical": 1.1,
    "laboratory_testing": 1.0,
    "prenatal_perinatal_care": 1.0,
}

# Cluster radius for deduplicating detections of the same physical entrance
CLUSTER_RADIUS_M = 5.0

# Metres the final entrance point is pushed outward from the building wall
ENTRANCE_OFFSET_M = 0.5

# Ray-casting sanity limits
MAX_LATERAL_ERROR_M = 5.0   # raised from 3.0 — allows valid hits slightly off the wall
MIN_RAY_DISTANCE_M  = 0.1
MAX_RAY_DISTANCE_M  = 60.0

# Image quality filter — Laplacian variance below this is considered blurry
# Lowered from 100.0: many valid street-level images fail the stricter threshold
SHARPNESS_THRESH = 50.0

# Default horizontal FOV (degrees) for perspective (non-360) cameras when
# camera_parameters are not available from the Mapillary API.
# Most smartphones and dashcams used for street mapping are 65–80°.
DEFAULT_HFOV_DEG = 70.0
