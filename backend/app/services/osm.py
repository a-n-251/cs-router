from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import requests


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


@dataclass
class OSMBundle:
    raw: Dict[str, Any]


def fetch_osm(
    center_lat: float,
    center_lon: float,
    radius_m: int,
) -> OSMBundle:
    """
    Fetch OSM data around a center point using Overpass API.
    Returns raw elements suitable for graph construction.
    """

    query = f"""
    [out:json][timeout:25];
    (
      way(around:{radius_m},{center_lat},{center_lon});
      >;
    );
    out body;
    """

    resp = requests.post(
        OVERPASS_URL,
        data=query.encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=60,
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Overpass API error {resp.status_code}: {resp.text[:200]}"
        )

    data = resp.json()

    if "elements" not in data:
        raise RuntimeError("Invalid Overpass response (no elements)")

    return OSMBundle(raw=data)
