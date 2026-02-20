import time
from pathlib import Path
from src.utils.duck_db_utils import open_duckdb
from src.utils.geo_utils import robust_radius_to_bbox
from src.utils.constants import S3_BUILDINGS, S3_PLACES


def download_overture_radius(
    lat: float,
    lon: float,
    radius_m: float,
    out_buildings: str = "data/buildings_local.parquet",
    out_places: str = "data/places_local.parquet",
) -> None:
    """
    Download Overture buildings and places within radius_m of (lat, lon)
    from AWS S3 and save as local parquet files.
    """
    Path(out_buildings).parent.mkdir(parents=True, exist_ok=True)

    xmin, ymin, xmax, ymax = robust_radius_to_bbox(lat, lon, radius_m)
    con = open_duckdb()

    # ── Buildings ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    print(f"[download] buildings from {S3_BUILDINGS}")
    con.execute(f"""
        COPY (
            SELECT
                id,
                ST_AsText(geometry) AS wkt,
                (bbox.xmin + bbox.xmax) / 2 AS lon,
                (bbox.ymin + bbox.ymax) / 2 AS lat
            FROM read_parquet('{S3_BUILDINGS}', hive_partitioning=1)
            WHERE bbox.xmax >= {xmin}
              AND bbox.xmin <= {xmax}
              AND bbox.ymax >= {ymin}
              AND bbox.ymin <= {ymax}
        ) TO '{out_buildings}' (FORMAT PARQUET);
    """)
    print(f"[download] buildings saved to {out_buildings} [{time.perf_counter() - t0:.1f}s]")

    # ── Places ───────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    print(f"[download] places from {S3_PLACES}")
    con.execute(f"""
        COPY (
            SELECT id, geometry, names, categories, bbox
            FROM read_parquet('{S3_PLACES}', hive_partitioning=1)
            WHERE bbox.xmax >= {xmin}
              AND bbox.xmin <= {xmax}
              AND bbox.ymax >= {ymin}
              AND bbox.ymin <= {ymax}
        ) TO '{out_places}' (FORMAT PARQUET);
    """)
    print(f"[download] places saved to {out_places} [{time.perf_counter() - t0:.1f}s]")

    con.close()
