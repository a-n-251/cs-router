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


def _get_result_value(result, key, default=None):
    """
    Safely extract a value from either an object with attributes or a dictionary.
    Falls back to `default` if the key/attribute is missing.
    """

    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


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


def _normalize_bool_mask(mask, n_segments: int):
    """
    Normalize a truthy/falsey mask to exactly `n_segments` entries.
    Truncates extra entries and pads missing entries with False.
    """
    normalized = [bool(v) for v in list(mask or [])]
    if len(normalized) < n_segments:
        normalized.extend([False] * (n_segments - len(normalized)))
    return normalized[:n_segments]


def _extract_completed_mask(completion, required_segments):
    """
    Completion is index-based. Try known representations and
    return a boolean mask aligned with required_segments.
    """
    n_segments = len(required_segments)

    if hasattr(completion, "completed_mask"):
        return _normalize_bool_mask(completion.completed_mask, n_segments)

    if hasattr(completion, "completed"):
        completed_attr = completion.completed

        # Already a bool mask? Normalize length.
        if completed_attr and all(isinstance(v, bool) for v in completed_attr):
            return _normalize_bool_mask(completed_attr, n_segments)

        # Otherwise assume it's an iterable of segment objects.
        bool_mask = [False] * n_segments
        required_by_id = {id(seg): i for i, seg in enumerate(required_segments)}
        required_by_seg_id = {
            getattr(seg, "seg_id", None): i
            for i, seg in enumerate(required_segments)
            if getattr(seg, "seg_id", None) is not None
        }

        for seg in completed_attr:
            idx = required_by_id.get(id(seg))
            if idx is None:
                seg_id = getattr(seg, "seg_id", None)
                idx = required_by_seg_id.get(seg_id)
            if idx is not None and 0 <= idx < n_segments:
                bool_mask[idx] = True

        return bool_mask

    if hasattr(completion, "segment_completed"):
        return _normalize_bool_mask(completion.segment_completed, n_segments)

    # Fallback: derive from debug summaries if present
    dbg = getattr(completion, "debug", {})
    summaries = dbg.get("segment_summaries")
    if summaries:
        mask = [False] * n_segments
        for s in summaries:
            if s.get("completed"):
                idx = s.get("seg_index")
                if isinstance(idx, int) and 0 <= idx < n_segments:
                    mask[idx] = True
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
    completed_mask = _extract_completed_mask(completion, bundle.required_segments)

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

    warnings = _get_result_value(result, "warnings", []) or []
    directions = _get_result_value(result, "directions", []) or []

    stats = _get_result_value(result, "stats", None)
    if not isinstance(stats, dict):
        stats = {}
    stats = {
        "distance_m_est": stats.get("distance_m_est", _get_result_value(result, "distance_m_est", 0) or 0),
        "covered_required_len_m_est": stats.get(
            "covered_required_len_m_est", _get_result_value(result, "covered_required_len_m_est", 0) or 0
        ),
        "required_segments_targeted": stats.get(
            "required_segments_targeted", _get_result_value(result, "required_segments_targeted", 0) or 0
        ),
        "required_segments_total_uncompleted": stats.get(
            "required_segments_total_uncompleted",
            _get_result_value(result, "required_segments_total_uncompleted", 0) or 0,
        ),
    }

    route_geojson = _get_result_value(result, "route_geojson", None)
    if route_geojson is None:
        route_geojson = _get_result_value(result, "route", None)

    unreachable_segments = _get_result_value(result, "unreachable_segments", None)

    # --- response ---
    resp = {
        "warnings": warnings,
        "stats": stats,
        "directions": directions,
        "route": (
            route_geojson
            if route_geojson and route_geojson.get("features")
            else None
        ),
        "completion_debug": completion.debug,
        "unreachable_segments": unreachable_segments,
        "is_loop": bool(req.loop),
    }

    if resp["route"] is None and "No drawable route produced" not in resp["warnings"]:
        resp["warnings"].append("No drawable route produced")

    return resp


@api.post("/plan")
async def plan(screenshot: UploadFile = File(...), payload: str = Form(...)):
    return await _handle_plan(screenshot, payload)


@api.post("/plan_multipart")
async def plan_multipart(screenshot: UploadFile = File(...), payload: str = Form(...)):
    return await _handle_plan(screenshot, payload)
