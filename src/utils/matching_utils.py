from typing import Any, Optional, Tuple, Dict
from .constants import _CAT_PRIOR, LOCAL_BUILDINGS, LOCAL_PLACES


def resolve_sources(_: Dict[str, float]) -> Tuple[str, str]:
    """Return paths to the locally-downloaded building and place parquet files."""
    lb = LOCAL_BUILDINGS if LOCAL_BUILDINGS.exists() else None
    lp = LOCAL_PLACES    if LOCAL_PLACES.exists()    else None
    if not (lb and lp):
        raise RuntimeError(
            f"Local parquet files not found. Expected:\n"
            f"  {LOCAL_BUILDINGS.resolve()}\n"
            f"  {LOCAL_PLACES.resolve()}\n"
            "Run the download step first."
        )
    return str(lb), str(lp)


def _cat_weight(categories: Any) -> float:
    """Return the category prior weight for a place."""
    try:
        primary = None
        if isinstance(categories, dict):
            primary = categories.get("primary") or None
        if primary and isinstance(primary, str):
            return float(_CAT_PRIOR.get(primary, 1.0))
    except Exception:
        pass
    return 1.0


def _extract_place_name(names_obj) -> Optional[str]:
    """Extract a human-readable name from the Overture 'names' field."""
    if not isinstance(names_obj, dict):
        return None
    primary = names_obj.get("primary")
    if isinstance(primary, str):
        return primary
    if isinstance(primary, dict) and primary:
        return primary.get("en") or next(iter(primary.values()), None)
    alt = names_obj.get("alternate")
    if isinstance(alt, dict) and alt:
        return alt.get("en") or next(iter(alt.values()), None)
    return None


def _geom_weight(row) -> float:
    w = 0.0
    if "inside" in row and row["inside"]:
        w += 0.6
    if "dist_m" in row and row["dist_m"] is not None:
        w += max(0.0, 0.4 - 0.4 * min(1.0, float(row["dist_m"]) / 10.0))
    return w


def select_best_place_for_building(
    links_df,
    building_id,
    max_dist_m: float = 60.0,
    hard_max_dist_m: Optional[float] = None,
) -> Optional[Dict]:
    """
    Pick the highest-scoring place for a given building.
    Returns a dict with name extracted, or None if nothing qualifies.
    """
    if links_df is None or len(links_df) == 0:
        return None

    df = links_df[links_df["building_id"] == building_id].copy()
    if df.empty:
        return None

    cap = hard_max_dist_m if hard_max_dist_m is not None else max_dist_m
    if "dist_m" in df.columns:
        df = df[df["dist_m"].notna() & (df["dist_m"] <= cap)]
    if df.empty:
        return None

    scores = [_geom_weight(r) + _cat_weight(r.get("categories", {}) or {}) for _, r in df.iterrows()]
    df = df.assign(_score=scores).sort_values("_score", ascending=False)
    top = df.iloc[0].to_dict()

    return {
        "place_id":   top.get("place_id"),
        "name":       _extract_place_name(top.get("names")),
        "categories": top.get("categories"),
        "lon":        top.get("place_lon"),
        "lat":        top.get("place_lat"),
        "inside":     bool(top.get("inside", False)),
        "dist_m":     float(top["dist_m"]) if top.get("dist_m") is not None else None,
    }
