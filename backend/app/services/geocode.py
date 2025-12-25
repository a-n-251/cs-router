from dataclasses import dataclass
from shapely.geometry import Point
import requests

@dataclass(frozen=True)
class GeoResult:
    point_wgs84: Point
    display_name: str

def geocode_address(address: str) -> GeoResult:
    if not address:
        raise ValueError("Empty address")

    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": address, "format": "json", "limit": 1},
        headers={"User-Agent": "CityStridesRouterMVP/1.0"}
    )
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError(f"Geocoding failed: {address}")
    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    return GeoResult(point_wgs84=Point(lon, lat), display_name=data[0].get("display_name", address))
