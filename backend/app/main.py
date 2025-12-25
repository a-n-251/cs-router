from __future__ import annotations

import json
import cv2
import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from shapely.geometry import Polygon

from app.models import PlanRequest
from app.services.georef import compute_georef, pix_to_lonlat
from app.services.osm import fetch_osm
from app.services.graph import build_graph_bundle
from app.services.completion import classify_required_segments
from app.services.solver import plan_route


api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.get("/health")
def health():
    return {"ok": True}


def _geocode(addr: str):
    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": addr, "format": "json", "limit": 1},
        headers={"User-Agent": "citystrides-router"},
        timeout=20,
    )
    if r.status_code != 200 or not r.json():
        raise HTTPException(400, f"Address not found: {addr}")
    return float(r.json()[0]["lat"]), float(r.json()[0]["lon"])


def _extract_completed_mask(completion, n_segments: int):
    """
    Completion is index-based. Try known representations and
    return a boolean mask aligned with required_segments.
    """
    if hasattr(completion, "completed_mask"):
        return completion.completed_mask

    if hasattr(completion, "completed"):
        return completion.completed

    if hasattr(completion, "segment_completed"):
        return completion.segment_completed

    # Fallback: derive from debug summaries if present
    dbg = getattr(completion, "debug", {})
    summaries = dbg.get("segment_summaries")
    if summaries:
        mask = [False] * n_segments
        for s in summaries:
            if s.get("completed"):
                mask[s["seg_index"]] = True
        return mask

    raise HTTPException(
        500,
        "CompletionResult does not expose a completed segment mask. "
        "Expected one of: completed_mask, completed, segment_completed, "
        "or debug.segment_summaries[].completed",
    )


async def _handle_plan(screenshot: UploadFile, payload: str):
    try:
        req = PlanRequest(**json.loads(payload))
    except Exception as e:
        raise HTTPException(400, f"Invalid payload: {e}")

    img_bytes = await screenshot.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")

    h, w = img.shape[:2]

    # --- start location ---
    start_lat, start_lon = _geocode(req.start_address)

    # --- georeference ---
    georef = compute_georef(req.control_points, w, h)

    # --- clip polygon from screenshot ---
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    ll = [pix_to_lonlat(georef, x, y) for x, y in corners]
    clip_poly = Polygon(ll)

    # --- OSM ---
    osm = fetch_osm(start_lat, start_lon, radius_m=1500)

    # --- graph ---
    bundle = build_graph_bundle(
        osm=osm,
        clip_poly_wgs84=clip_poly,
        allow_private=req.allow_private,
        max_required_edge_len_m=80.0,
    )

    # --- completion detection ---
    completion = classify_required_segments(
        img_bytes=img_bytes,
        georef=georef,
        required_segments=bundle.required_segments,
        proximity_m=req.proximity_m,
        max_angle_deg=req.max_angle_deg,
        completion_ratio=req.completion_ratio,
    )

    # --- uncompleted required segments (INDEX-BASED) ---
    completed_mask = _extract_completed_mask(
        completion,
        n_segments=len(bundle.required_segments),
    )

    uncompleted_segments = [
        seg
        for i, seg in enumerate(bundle.required_segments)
        if not completed_mask[i]
    ]

    # --- end location (optional) ---
    end_lon = None
    end_lat = None
    if not req.loop and req.end_address:
        end_lat, end_lon = _geocode(req.end_address)

    # --- route planning ---
    result = plan_route(
        bundle=bundle,
        start_lon=start_lon,
        start_lat=start_lat,
        loop=req.loop,
        end_lon=end_lon,
        end_lat=end_lat,
        uncompleted_required_segments=uncompleted_segments,
        max_distance_m=req.max_distance_m,
    )

    # --- response ---
    resp = {
        "warnings": result.warnings or [],
        "stats": {
            "distance_m_est": result.stats.get("distance_m_est", 0),
            "covered_required_len_m_est": result.stats.get("covered_required_len_m_est", 0),
            "required_segments_targeted": result.stats.get("required_segments_targeted", 0),
            "required_segments_total_uncompleted": result.stats.get(
                "required_segments_total_uncompleted", 0
            ),
        },
        "directions": result.directions or [],
        "route": (
            result.route_geojson
            if result.route_geojson and result.route_geojson.get("features")
            else None
        ),
        "completion_debug": completion.debug,
    }

    if resp["route"] is None:
        resp["warnings"].append("No drawable route produced")

    return resp


@api.post("/plan")
async def plan(screenshot: UploadFile = File(...), payload: str = Form(...)):
    return await _handle_plan(screenshot, payload)


@api.post("/plan_multipart")
async def plan_multipart(screenshot: UploadFile = File(...), payload: str = Form(...)):
    return await _handle_plan(screenshot, payload)
