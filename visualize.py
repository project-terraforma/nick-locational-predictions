"""
visualize.py — Interactive entrance prediction map

Usage
-----
# Visualize the most recently generated GeoJSON (auto-discovers latest run):
python3 visualize.py

# Visualize a specific GeoJSON:
python3 visualize.py outputs/2024-01-15_14-30-00/geojsons/37.78000_-122.40920.geojson

# Override the outputs root:
python3 visualize.py --outputs=my_outputs/

Features
--------
- Layer toggles: show/hide building footprints, Overture place pins, predicted entrances,
  and business name labels independently
- Google Maps-style teardrop pins for both entrance and place markers
- Click any pin to open a pinned info panel (click × or map to close)
- Entrance panel shows the YOLO-annotated visualization image (falls back to raw thumbnail)
- Click the photo in the panel to open a full-screen lightbox
  — scroll wheel to zoom, drag to pan, Esc or click outside to close
- Hover a place pin → highlights ALL predicted entrances for that building + fades everything else
- Hover an entrance pin → highlights the place pin + ALL sibling entrances + fades everything else
- All non-highlighted pins fade to 25% opacity on hover for maximum focus
- CartoDB Positron tile layer — clean white background for maximum pin clarity
- Zoom levels up to 21 supported
"""

import argparse
import base64
import json
import webbrowser
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _img_to_base64(path: Path) -> str:
    if not path or not path.exists():
        return ""
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def _find_image(source_image_name: str, candidates_dir: Path, vis_dir: Path) -> str:
    """Return base64-encoded image data.

    Prefers the YOLO-annotated debug image (dbg_<name>) when it exists so
    the info panel shows exactly which detection drove the entrance prediction.
    Falls back to the raw Mapillary candidate thumbnail.
    """
    if not source_image_name:
        return ""
    if vis_dir:
        dbg = vis_dir / f"dbg_{source_image_name}"
        if dbg.exists():
            return _img_to_base64(dbg)
    return _img_to_base64(candidates_dir / source_image_name)


def _latest_geojson(outputs_dir: Path) -> Path:
    """Find the most recently modified GeoJSON across all timestamped run folders."""
    files = sorted(
        outputs_dir.glob("*/geojsons/*.geojson"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        files = sorted(
            outputs_dir.glob("geojsons/*.geojson"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if not files:
        raise FileNotFoundError(f"No GeoJSON files found under {outputs_dir}")
    return files[0]


# ── HTML generation ───────────────────────────────────────────────────────────

def build_html(geojson_path: Path, candidates_dir: Path, vis_dir: Path) -> str:
    with open(geojson_path) as f:
        gj = json.load(f)

    features  = gj["features"]
    buildings = [f for f in features if f["geometry"]["type"] == "Polygon"]
    entrances = [f for f in features if f["properties"].get("marker-symbol") == "star"]
    places    = [f for f in features if f["properties"].get("marker-symbol") == "circle"]

    all_lons, all_lats = [], []
    for feat in features:
        geom = feat["geometry"]
        if geom["type"] == "Point":
            all_lons.append(geom["coordinates"][0])
            all_lats.append(geom["coordinates"][1])
        elif geom["type"] == "Polygon":
            for ring in geom["coordinates"]:
                for c in ring:
                    all_lons.append(c[0])
                    all_lats.append(c[1])

    center_lat = sum(all_lats) / len(all_lats) if all_lats else 0
    center_lon = sum(all_lons) / len(all_lons) if all_lons else 0

    buildings_js = json.dumps([
        {"coords": [[c[1], c[0]] for c in feat["geometry"]["coordinates"][0]],
         "name":   feat["properties"].get("name", "Building"),
         "id":     feat["properties"].get("building_id", "")}
        for feat in buildings
    ])

    places_js = json.dumps([
        {
            "lat":        feat["geometry"]["coordinates"][1],
            "lon":        feat["geometry"]["coordinates"][0],
            "bid":        feat["properties"].get("building_id", ""),
            "name":       feat["properties"].get("name", ""),
            "category":   feat["properties"].get("category", ""),
            "address":    feat["properties"].get("address", ""),
            "confidence": feat["properties"].get("overture_confidence", ""),
        }
        for feat in places
    ])

    entrance_records = []
    for feat in entrances:
        props = feat["properties"]
        src   = props.get("source_image", "")
        img   = _find_image(src, candidates_dir, vis_dir)
        entrance_records.append({
            "lat":        feat["geometry"]["coordinates"][1],
            "lon":        feat["geometry"]["coordinates"][0],
            "bid":        props.get("building_id", ""),
            "place_name": props.get("place_name", ""),
            "conf":       props.get("detection_confidence", 0.0),
            "n_dets":     props.get("num_detections", 1),
            "colour":     props.get("marker-color", "#e74c3c"),
            "source_img": src,
            "img_data":   img,
        })
    entrances_js = json.dumps(entrance_records)

    # ── HTML template ──────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Entrance Predictions — {geojson_path.name}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    #map {{ height: 100vh; width: 100%; }}

    /* ── Layer control overrides ── */
    .leaflet-control-layers {{
      border-radius: 10px !important;
      box-shadow: 0 2px 14px rgba(0,0,0,0.13) !important;
      border: none !important;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .leaflet-control-layers-expanded {{ padding: 10px 14px !important; }}
    .leaflet-control-layers-list label {{
      font-size: 12.5px; cursor: pointer; margin: 4px 0; display: flex;
      align-items: center; gap: 6px; color: #333;
    }}
    .leaflet-control-layers-list input {{ cursor: pointer; }}
    .leaflet-control-layers-separator {{ margin: 6px 0 !important; }}

    /* ── Place name labels (tooltip nametag above pin) ── */
    .name-label {{
      background: rgba(255,255,255,0.92) !important;
      border: 1px solid #ccc !important;
      border-radius: 5px !important;
      padding: 2px 8px !important;
      font-size: 11px !important;
      font-weight: 600 !important;
      color: #222 !important;
      box-shadow: 0 1px 6px rgba(0,0,0,0.12) !important;
      white-space: nowrap !important;
    }}
    .name-label::before {{ display: none !important; }}

    /* ── Building name map labels (integrated text, no card) ── */
    .map-label {{
      transform: translate(-50%, -100%);
      color: #445;
      font-size: 10.5px;
      font-weight: 700;
      text-shadow: 1px 1px 0 white, -1px 1px 0 white,
                   1px -1px 0 white, -1px -1px 0 white,
                   2px 0 0 white, -2px 0 0 white,
                   0 2px 0 white,  0 -2px 0 white;
      white-space: nowrap;
      pointer-events: none;
      letter-spacing: 0.35px;
      text-transform: uppercase;
    }}

    /* ── Info panel ── */
    #info-panel {{
      position: fixed; top: 14px; right: 14px; z-index: 1000;
      background: #fff; border-radius: 12px;
      box-shadow: 0 6px 28px rgba(0,0,0,0.16);
      padding: 16px 18px 14px; width: 360px;
      display: none;
    }}
    .ip-header {{
      display: flex; align-items: flex-start; gap: 8px; margin-bottom: 8px;
    }}
    #ip-title {{
      font-size: 14px; font-weight: 600; color: #111; flex: 1; line-height: 1.35;
    }}
    #ip-close {{
      background: none; border: none; cursor: pointer; font-size: 20px;
      line-height: 1; color: #bbb; padding: 0 2px; flex-shrink: 0;
    }}
    #ip-close:hover {{ color: #555; }}
    #ip-body p {{ font-size: 12px; color: #555; margin: 3px 0; line-height: 1.55; }}
    #ip-body img {{
      width: 100%; border-radius: 8px; margin-top: 10px;
      border: 1px solid #eee; cursor: zoom-in;
      transition: opacity 0.15s;
    }}
    #ip-body img:hover {{ opacity: 0.9; }}
    .badge {{
      display: inline-block; padding: 2px 9px; border-radius: 999px;
      font-size: 11px; font-weight: 700; color: #fff; margin-top: 6px;
      letter-spacing: 0.3px;
    }}
    .no-img {{
      margin-top: 10px; padding: 12px; background: #f8f8f8; border-radius: 8px;
      font-size: 12px; color: #aaa; font-style: italic; text-align: center;
    }}

    /* ── Lightbox ── */
    #lightbox {{
      display: none; position: fixed; inset: 0; z-index: 9999;
      background: rgba(0,0,0,0.9);
      align-items: center; justify-content: center;
    }}
    #lightbox.lb-open {{ display: flex; }}
    #lb-img {{
      max-width: 90vw; max-height: 90vh;
      object-fit: contain; border-radius: 4px;
      transform-origin: center center;
      user-select: none; -webkit-user-drag: none;
      cursor: default;
    }}
    #lb-close {{
      position: absolute; top: 16px; right: 20px;
      background: rgba(255,255,255,0.12);
      border: 1px solid rgba(255,255,255,0.2);
      color: #fff; cursor: pointer; font-size: 22px;
      width: 42px; height: 42px; border-radius: 50%;
      line-height: 40px; text-align: center;
      transition: background 0.2s;
    }}
    #lb-close:hover {{ background: rgba(255,255,255,0.25); }}
    #lb-hint {{
      position: absolute; bottom: 18px; left: 50%; transform: translateX(-50%);
      color: rgba(255,255,255,0.4); font-size: 12px; pointer-events: none;
      white-space: nowrap;
    }}

    /* ── Legend ── */
    #legend {{
      position: fixed; bottom: 58px; left: 14px; z-index: 1000;
      background: #fff; border-radius: 10px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.12);
      padding: 10px 15px; font-size: 12px; color: #444;
    }}
    #legend b {{ font-size: 13px; display: block; margin-bottom: 5px; }}
    .legend-row {{ display: flex; align-items: center; gap: 7px; margin: 4px 0; }}
    .legend-pin {{ width: 16px; height: 22px; flex-shrink: 0; }}
    #legend .hint {{ font-size: 11px; color: #aaa; margin-top: 6px; }}
  </style>
</head>
<body>

<div id="map"></div>

<!-- Info panel -->
<div id="info-panel">
  <div class="ip-header">
    <div id="ip-title"></div>
    <button id="ip-close" title="Close">&#x2715;</button>
  </div>
  <div id="ip-body"></div>
</div>

<!-- Lightbox -->
<div id="lightbox">
  <img id="lb-img" src="" alt="Enlarged photo" />
  <button id="lb-close" title="Close (Esc)">&#x2715;</button>
  <p id="lb-hint">Scroll to zoom &nbsp;·&nbsp; Drag to pan &nbsp;·&nbsp; Esc to close</p>
</div>

<!-- Legend -->
<div id="legend">
  <b>Legend</b>
  <div class="legend-row">
    <svg class="legend-pin" viewBox="0 0 16 22"><path d="M8 0C3.6 0 0 3.6 0 8C0 14 8 22 8 22C8 22 16 14 16 8C16 3.6 12.4 0 8 0Z" fill="#4a90d9" opacity="0.55"/></svg>
    Building footprint
  </div>
  <div class="legend-row">
    <svg class="legend-pin" viewBox="0 0 16 22"><path d="M8 0C3.6 0 0 3.6 0 8C0 14 8 22 8 22C8 22 16 14 16 8C16 3.6 12.4 0 8 0Z" fill="#27ae60"/><circle cx="8" cy="8" r="3" fill="white" opacity="0.9"/></svg>
    Overture place pin
  </div>
  <div class="legend-row">
    <svg class="legend-pin" viewBox="0 0 16 22"><path d="M8 0C3.6 0 0 3.6 0 8C0 14 8 22 8 22C8 22 16 14 16 8C16 3.6 12.4 0 8 0Z" fill="#e74c3c"/><text x="8" y="12" text-anchor="middle" font-size="9" fill="white" font-family="sans-serif">★</text></svg>
    Predicted entrance
  </div>
  <div class="hint">red&thinsp;→&thinsp;green = confidence &nbsp;·&nbsp; click pin to inspect</div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
// ── Embedded data ──────────────────────────────────────────────────────────
var BUILDINGS = {buildings_js};
var PLACES    = {places_js};
var ENTRANCES = {entrances_js};

// ── Map ────────────────────────────────────────────────────────────────────
var map = L.map('map', {{ zoomControl: false }}).setView([{center_lat}, {center_lon}], 20);
L.control.zoom({{ position: 'bottomright' }}).addTo(map);

// ── Base tile layers (radio — only one active at a time) ───────────────────
var cartoLight = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
  subdomains: 'abcd',
  maxZoom: 21,
}}).addTo(map);

var satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
  attribution: '&copy; Esri &mdash; Esri, Maxar, GeoEye, Earthstar Geographics, CNES/Airbus DS, USDA, USGS, AeroGRID, IGN',
  maxZoom: 19,
}});

// ── Layer groups ───────────────────────────────────────────────────────────
var buildingGroup  = L.layerGroup().addTo(map);
var placeGroup     = L.layerGroup().addTo(map);
var entranceGroup  = L.layerGroup().addTo(map);
var namesGroup     = L.layerGroup().addTo(map);

L.control.layers(
  {{ 'Map': cartoLight, 'Satellite': satellite }},
  {{
    'Building footprints':       buildingGroup,
    'Overture place pins':       placeGroup,
    'Predicted entrances':       entranceGroup,
    'Business / building names': namesGroup,
  }},
  {{ collapsed: false, position: 'topleft' }}
).addTo(map);

// ── Info panel ─────────────────────────────────────────────────────────────
var panel = document.getElementById('info-panel');

function showPanel(title, bodyHtml) {{
  document.getElementById('ip-title').textContent = title;
  document.getElementById('ip-body').innerHTML    = bodyHtml;
  panel.style.display = 'block';
}}
function closePanel() {{
  panel.style.display = 'none';
}}
document.getElementById('ip-close').addEventListener('click', closePanel);
map.on('click', closePanel);

// ── Lightbox ───────────────────────────────────────────────────────────────
var lb = {{ scale: 1, tx: 0, ty: 0, dragging: false, sx: 0, sy: 0, bx: 0, by: 0 }};
var lbBox = document.getElementById('lightbox');
var lbImg = document.getElementById('lb-img');

function lbSetTransform() {{
  lbImg.style.transform =
    'translate(' + lb.tx + 'px,' + lb.ty + 'px) scale(' + lb.scale + ')';
}}

function openLightbox(src) {{
  lbImg.src = src;
  lb.scale = 1; lb.tx = 0; lb.ty = 0;
  lbSetTransform();
  lbImg.style.cursor = 'default';
  lbBox.classList.add('lb-open');
}}

function closeLightbox() {{
  lbBox.classList.remove('lb-open');
  lbImg.src = '';
}}

document.getElementById('lb-close').addEventListener('click', closeLightbox);
lbBox.addEventListener('click', function(e) {{
  if (e.target === lbBox) closeLightbox();
}});
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') closeLightbox();
}});

// Scroll to zoom
lbBox.addEventListener('wheel', function(e) {{
  e.preventDefault();
  var delta = e.deltaY < 0 ? 0.18 : -0.18;
  lb.scale = Math.max(1, Math.min(6, lb.scale + delta));
  lbSetTransform();
  lbImg.style.cursor = lb.scale > 1 ? 'grab' : 'default';
}}, {{ passive: false }});

// Drag to pan (when zoomed)
lbImg.addEventListener('mousedown', function(e) {{
  if (lb.scale <= 1) return;
  lb.dragging = true;
  lb.sx = e.clientX; lb.sy = e.clientY;
  lb.bx = lb.tx;     lb.by = lb.ty;
  lbImg.style.cursor = 'grabbing';
  e.preventDefault();
}});
document.addEventListener('mousemove', function(e) {{
  if (!lb.dragging) return;
  lb.tx = lb.bx + (e.clientX - lb.sx);
  lb.ty = lb.by + (e.clientY - lb.sy);
  lbSetTransform();
}});
document.addEventListener('mouseup', function() {{
  if (!lb.dragging) return;
  lb.dragging = false;
  lbImg.style.cursor = lb.scale > 1 ? 'grab' : 'default';
}});

// ── Marker registry & hover helpers ───────────────────────────────────────
// allMarkers   : flat list of every pin marker (used by restoreAll)
// registry[bid]: {{ place, entrances[] }}  — groups by building ID
// nameRegistry[place_name]: [entrance markers] — secondary grouping by name
//   catches "outlier" entrances attributed to a different building polygon
//   but still belonging to the same place.
var allMarkers   = [];
var registry     = {{}};
var nameRegistry = {{}};

function getOrCreate(bid) {{
  if (!registry[bid]) registry[bid] = {{ place: null, entrances: [] }};
  return registry[bid];
}}

// Highlight one marker with a glow
function highlightLayer(layer) {{
  if (!layer) return;
  var el = layer.getElement ? layer.getElement() : null;
  if (el) el.style.filter = 'drop-shadow(0 0 8px rgba(231,76,60,0.95)) brightness(1.25)';
}}

// Restore every marker to default appearance
function restoreAll() {{
  allMarkers.forEach(function(m) {{
    var el = m.getElement ? m.getElement() : null;
    if (el) {{ el.style.filter = ''; el.style.opacity = ''; }}
  }});
}}

// ── SVG pin factory ────────────────────────────────────────────────────────
function makePinIcon(fillColour, innerSvg, w, h) {{
  var svg =
    '<svg width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + ' ' + h + '" ' +
    'xmlns="http://www.w3.org/2000/svg">' +
    '<path d="M' + (w/2) + ' 0C' + (w*0.225) + ' 0 0 ' + (w*0.225) + ' 0 ' + (w/2) +
    'C0 ' + (h*0.6) + ' ' + (w/2) + ' ' + h + ' ' + (w/2) + ' ' + h +
    'C' + (w/2) + ' ' + h + ' ' + w + ' ' + (h*0.6) + ' ' + w + ' ' + (w/2) +
    'C' + w + ' ' + (w*0.225) + ' ' + (w*0.775) + ' 0 ' + (w/2) + ' 0Z" ' +
    'fill="' + fillColour + '" stroke="rgba(0,0,0,0.22)" stroke-width="1.2"/>' +
    innerSvg +
    '</svg>';
  return L.divIcon({{
    html:       svg,
    className:  '',
    iconSize:   [w, h],
    iconAnchor: [w/2, h],
  }});
}}

// ── Building polygons + centroid lookup ───────────────────────────────────
// interactive: false disables all mouse events (no click square, no hover cursor)
// buildingCentroid[bid] is used later to anchor name labels inside footprints.
var buildingCentroid = {{}};
BUILDINGS.forEach(function(b) {{
  L.polygon(b.coords, {{
    color: '#4a90d9', weight: 1.5,
    fillColor: '#4a90d9', fillOpacity: 0.07,
    interactive: false,
  }}).addTo(buildingGroup);
  if (b.id) {{
    var n = b.coords.length;
    buildingCentroid[b.id] = b.coords.reduce(
      function(acc, c) {{ return [acc[0] + c[0] / n, acc[1] + c[1] / n]; }},
      [0, 0]
    );
  }}
}});

// ── Place pins ─────────────────────────────────────────────────────────────
PLACES.forEach(function(p) {{
  var icon = makePinIcon(
    '#27ae60',
    '<circle cx="12" cy="12" r="4.5" fill="white" opacity="0.92"/>',
    24, 36
  );
  var marker = L.marker([p.lat, p.lon], {{ icon: icon }});

  marker.on('mouseover', function() {{
    // Active set: this place pin + entrances by bid + entrances by place_name
    var active = [marker];
    if (p.bid) {{
      var reg = registry[p.bid] || {{ entrances: [] }};
      reg.entrances.forEach(function(m) {{ if (active.indexOf(m) === -1) active.push(m); }});
    }}
    if (p.name && nameRegistry[p.name]) {{
      nameRegistry[p.name].forEach(function(m) {{ if (active.indexOf(m) === -1) active.push(m); }});
    }}
    active.forEach(highlightLayer);
  }});
  marker.on('mouseout', restoreAll);
  marker.on('click', function(ev) {{
    L.DomEvent.stopPropagation(ev);
    // Count all entrances for this place (by name covers outliers with a different bid)
    var entCount = (p.name && nameRegistry[p.name]) ? nameRegistry[p.name].length
                 : (p.bid && registry[p.bid] ? registry[p.bid].entrances.length : 0);
    showPanel(
      p.name || 'Place',
      '<p><b>Category:</b> '             + (p.category   || '—') + '</p>' +
      '<p><b>Address:</b> '              + (p.address    || '—') + '</p>' +
      '<p><b>Overture confidence:</b> '  + (p.confidence || '—') + '</p>' +
      '<p><b>Predicted entrances found:</b> ' + entCount + '</p>' +
      '<p style="color:#bbb;margin-top:8px;font-size:11px">Original Overture map pin</p>'
    );
  }});

  marker.addTo(placeGroup);
  allMarkers.push(marker);
  if (p.bid) getOrCreate(p.bid).place = marker;
}});

// ── Entrance markers ───────────────────────────────────────────────────────
ENTRANCES.forEach(function(e) {{
  var cx = 14, cy = 13;
  var icon = makePinIcon(
    e.colour,
    '<text x="' + cx + '" y="' + (cy + 5) + '" text-anchor="middle" ' +
    'font-size="13" fill="white" font-family="Arial,sans-serif" ' +
    'font-weight="bold">&#9733;</text>',
    28, 40
  );
  var marker = L.marker([e.lat, e.lon], {{ icon: icon }});

  var badgeColour = e.conf > 0.7 ? '#27ae60' : e.conf > 0.4 ? '#e67e22' : '#c0392b';
  var confPct     = Math.round(e.conf * 100);
  var imgHtml     = e.img_data
    ? '<img src="' + e.img_data + '" alt="Source image" ' +
      'title="Click to enlarge" onclick="openLightbox(this.src)" />'
    : '<div class="no-img">Image not found in candidates/ or visualizations/</div>';

  marker.on('mouseover', function() {{
    // Active set: this entrance + place pin + all siblings by bid + all siblings by place_name
    // The name-based lookup catches outliers attributed to a different building polygon.
    var active = [marker];
    if (e.bid) {{
      var reg = registry[e.bid] || {{ place: null, entrances: [] }};
      if (reg.place) active.push(reg.place);
      reg.entrances.forEach(function(m) {{ if (active.indexOf(m) === -1) active.push(m); }});
    }}
    if (e.place_name && nameRegistry[e.place_name]) {{
      nameRegistry[e.place_name].forEach(function(m) {{ if (active.indexOf(m) === -1) active.push(m); }});
    }}
    active.forEach(highlightLayer);
  }});
  marker.on('mouseout', restoreAll);
  marker.on('click', function(ev) {{
    L.DomEvent.stopPropagation(ev);
    showPanel(
      e.place_name || 'Predicted Entrance',
      '<span class="badge" style="background:' + badgeColour + '">Confidence ' + confPct + '%</span>' +
      ' <span class="badge" style="background:#666">' + e.n_dets +
      ' detection' + (e.n_dets !== 1 ? 's' : '') + '</span>' +
      '<p style="margin-top:8px"><b>Source image:</b> ' + (e.source_img || '—') + '</p>' +
      imgHtml
    );
  }});

  marker.addTo(entranceGroup);
  allMarkers.push(marker);
  if (e.bid) getOrCreate(e.bid).entrances.push(marker);
  // Secondary index by place_name — covers outliers that share a name but not a bid
  if (e.place_name) {{
    if (!nameRegistry[e.place_name]) nameRegistry[e.place_name] = [];
    nameRegistry[e.place_name].push(marker);
  }}
}});

// ── Name labels (toggleable via namesGroup) ────────────────────────────────
// Place names: tooltip nametag above the place pin (card style, directional).
PLACES.forEach(function(p) {{
  if (!p.name) return;
  L.marker([p.lat, p.lon], {{
    icon: L.divIcon({{ html: '', className: '', iconSize: [0, 0] }}),
    interactive: false,
  }})
  .bindTooltip(p.name, {{
    permanent: true, direction: 'top', offset: [0, -40],
    className: 'name-label',
  }})
  .addTo(namesGroup);
}});

// Building names: integrated map-text label centred on the polygon footprint.
// Skips generic "Building <uuid>" entries.
BUILDINGS.forEach(function(b) {{
  if (!b.name || b.name.startsWith('Building ')) return;
  var centroid = buildingCentroid[b.id];
  if (!centroid) return;
  L.marker(centroid, {{
    icon: L.divIcon({{
      html: '<div class="map-label">' + b.name + '</div>',
      className: '',
      iconSize:   [0, 0],
      iconAnchor: [0, 0],
    }}),
    interactive: false,
  }}).addTo(namesGroup);
}});

// ── Zoom-based name label visibility ──────────────────────────────────────
// Labels are hidden below zoom 19 to avoid clutter at city / block scale.
var NAME_ZOOM_THRESHOLD = 19;
function updateNamesVisibility() {{
  var visible = map.getZoom() >= NAME_ZOOM_THRESHOLD;
  namesGroup.eachLayer(function(layer) {{
    // divIcon markers (building names)
    var el = layer.getElement ? layer.getElement() : null;
    if (el) el.style.visibility = visible ? '' : 'hidden';
    // Permanent tooltip markers (place names)
    var tt = layer.getTooltip ? layer.getTooltip() : null;
    if (tt) tt.setOpacity(visible ? 1 : 0);
  }});
}}
map.on('zoomend', updateNamesVisibility);
updateNamesVisibility();  // apply on first load
</script>
</body>
</html>"""
    return html


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize entrance prediction GeoJSON results")
    parser.add_argument("geojson", nargs="?", help="Path to .geojson file (default: most recent)")
    parser.add_argument("--outputs", default="outputs", help="Outputs root directory")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs)

    if args.geojson:
        geojson_path = Path(args.geojson)
    else:
        geojson_path = _latest_geojson(outputs_dir)

    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")

    run_dir        = geojson_path.parent.parent
    candidates_dir = run_dir / "candidates"
    vis_dir        = run_dir / "visualizations"

    if not candidates_dir.exists():
        candidates_dir = outputs_dir / "candidates"

    print(f"Visualizing:    {geojson_path}")
    print(f"Candidates dir: {candidates_dir}")
    print(f"Debug vis dir:  {vis_dir if vis_dir.exists() else '(not found — using raw candidates)'}")

    html = build_html(geojson_path, candidates_dir, vis_dir if vis_dir.exists() else None)

    out_html = geojson_path.with_suffix(".html")
    with open(out_html, "w") as f:
        f.write(html)

    print(f"Map written to: {out_html.resolve()}")
    webbrowser.open(out_html.resolve().as_uri())


if __name__ == "__main__":
    main()
