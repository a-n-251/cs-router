from typing import List, Dict, Any
from shapely.geometry import Point, LineString
from shapely.ops import unary_union

from app.services.osm import OSMBundle

def extract_amenities(osm: OSMBundle, route_line_wgs84: LineString, buffer_m: float = 200.0) -> List[Dict[str, Any]]:
    # Buffer in degrees is wrong; MVP approximation: only filter by bbox expanded a bit.
    # Next version: project to 3857 and buffer properly.
    minx, miny, maxx, maxy = route_line_wgs84.bounds
    pad = 0.003  # ~300m-ish in lat; rough MVP
    bbox = (minx-pad, miny-pad, maxx+pad, maxy+pad)

    out = []
    for el in osm.raw.get("elements", []):
        if el.get("type") == "node":
            tags = el.get("tags", {})
            if tags.get("amenity") in {"drinking_water", "toilets"}:
                lon = float(el["lon"]); lat = float(el["lat"])
                if bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]:
                    out.append({
                        "type": tags.get("amenity"),
                        "name": tags.get("name"),
                        "lon": lon,
                        "lat": lat
                    })
    return out
