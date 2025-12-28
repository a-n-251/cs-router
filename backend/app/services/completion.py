from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import cv2
from shapely.geometry import LineString
from pyproj import Transformer

from app.services.georef import lonlat_to_pix


@dataclass
class CompletionResult:
    completed: List
    uncompleted: List
    debug: Dict[str, Any]


def classify_required_segments(
    img_bytes: bytes,
    georef,
    required_segments,
    proximity_m: float,
    max_angle_deg: float,
    completion_ratio: float,
):
    """
    REQUIRED SEGMENT GEOM: EPSG:3857 meters (Segment.geom_3857)
    GEOREF: maps between lon/lat and pixel coords
    Therefore pipeline must be: 3857 -> lon/lat -> pixel

    This function returns debug info in .debug for display in API response.
    """

    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid screenshot")

    h, w = img.shape[:2]

    # Wide purple/blue-purple mask for CityStrides overlay.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    purple_mask = cv2.inRange(
        hsv,
        np.array([95, 18, 18], dtype=np.uint8),
        np.array([180, 255, 255], dtype=np.uint8),
    )

    # Slightly dilate to be forgiving for parallel/offset paths (e.g., footpaths).
    DILATION_RADIUS_PX = 3
    if DILATION_RADIUS_PX > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (DILATION_RADIUS_PX * 2 + 1, DILATION_RADIUS_PX * 2 + 1),
        )
        expanded_mask = cv2.dilate(purple_mask, kernel, iterations=1)
    else:
        expanded_mask = purple_mask

    # Distance (in px) to the nearest purple pixel after dilation.
    purple_distance = cv2.distanceTransform(
        255 - expanded_mask, cv2.DIST_L2, 3
    )

    purple_px = int(np.count_nonzero(purple_mask))
    total_px = int(w * h)

    # IMPORTANT: segment points are EPSG:3857. Convert to lon/lat before lonlat_to_pix.
    tf_3857_to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    SEARCH_RADIUS_PX = 14
    NEARBY_DISTANCE_PX = 26  # Allow “close but parallel” lines to count as a hit.
    SAMPLES_PER_SEGMENT = 60

    completed = []
    uncompleted = []

    # Debug aggregates
    segs_considered = 0
    segs_ignored_outside = 0
    segs_completed = 0
    segs_uncompleted = 0

    total_samples = 0
    in_bounds_samples = 0
    purple_hit_samples = 0
    nearby_only_hits = 0

    # keep a few examples so you can see coordinate sanity
    example_points: List[Dict[str, Any]] = []
    example_segs: List[Dict[str, Any]] = []

    # Treat very short segments as incomplete unless we see purple near the middle.
    SHORT_SEGMENT_LEN_M = 100.0

    for i, seg in enumerate(required_segments):
        ls: LineString = seg.geom_3857
        if ls is None or ls.is_empty:
            continue

        segs_considered += 1

        hits = 0
        total = 0

        # sample along the 3857 geometry
        center_hits = 0

        for d in np.linspace(0, ls.length, SAMPLES_PER_SEGMENT):
            pt = ls.interpolate(float(d))

            # 3857 -> lon/lat
            lon, lat = tf_3857_to_4326.transform(pt.x, pt.y)

            # lon/lat -> pixel via georef
            x, y = lonlat_to_pix(georef, lon, lat)
            ix, iy = int(round(x)), int(round(y))

            total_samples += 1

            if 0 <= ix < w and 0 <= iy < h:
                total += 1
                in_bounds_samples += 1

                x0 = max(ix - SEARCH_RADIUS_PX, 0)
                x1 = min(ix + SEARCH_RADIUS_PX, w - 1)
                y0 = max(iy - SEARCH_RADIUS_PX, 0)
                y1 = min(iy + SEARCH_RADIUS_PX, h - 1)

                has_direct_hit = np.any(expanded_mask[y0:y1 + 1, x0:x1 + 1])
                is_near_hit = purple_distance[iy, ix] <= NEARBY_DISTANCE_PX

                if has_direct_hit or is_near_hit:
                    hits += 1
                    purple_hit_samples += 1
                    if is_near_hit and not has_direct_hit:
                        nearby_only_hits += 1

                    # Track hits that land away from the segment ends.
                    if ls.length > 0:
                        t = d / ls.length
                        if 0.2 <= t <= 0.8:
                            center_hits += 1

            # collect a few sample mappings for sanity
            if len(example_points) < 12 and i % 50 == 0:
                example_points.append(
                    {
                        "seg_index": i,
                        "seg_id": getattr(seg, "seg_id", None),
                        "pt_3857": [float(pt.x), float(pt.y)],
                        "lonlat": [float(lon), float(lat)],
                        "pix": [float(x), float(y)],
                        "pix_rounded": [ix, iy],
                        "in_bounds": bool(0 <= ix < w and 0 <= iy < h),
                    }
                )

        if total == 0:
            segs_ignored_outside += 1
            continue

        ratio = hits / total

        # If only the endpoints of a very short street are purple (e.g., nodes at
        # the ends when a perpendicular street was completed), require at least one
        # hit near the middle before considering the street complete.
        if ls.length <= SHORT_SEGMENT_LEN_M and hits > 0 and center_hits == 0:
            ratio = 0.0

        if len(example_segs) < 20 and i % 25 == 0:
            example_segs.append(
                {
                    "seg_index": i,
                    "seg_id": getattr(seg, "seg_id", None),
                    "name": seg.tags.get("name") if hasattr(seg, "tags") else None,
                    "highway": seg.tags.get("highway") if hasattr(seg, "tags") else None,
                    "seg_len_m": float(getattr(seg, "length_m", 0.0)),
                    "samples_in_bounds": int(total),
                    "purple_hits": int(hits),
                    "hit_ratio": float(ratio),
                    "classified_completed": bool(ratio >= completion_ratio),
                }
            )

        if ratio >= completion_ratio:
            completed.append(seg)
            segs_completed += 1
        else:
            uncompleted.append(seg)
            segs_uncompleted += 1

    debug = {
        "image": {"w": w, "h": h},
        "purple_mask": {
            "purple_px": purple_px,
            "total_px": total_px,
            "purple_fraction": (purple_px / total_px) if total_px else 0.0,
            "hsv_range": {"low": [95, 18, 18], "high": [180, 255, 255]},
            "search_radius_px": SEARCH_RADIUS_PX,
            "dilation_radius_px": DILATION_RADIUS_PX,
        },
        "sampling": {
            "samples_per_segment": SAMPLES_PER_SEGMENT,
            "total_samples": total_samples,
            "in_bounds_samples": in_bounds_samples,
            "purple_hit_samples": purple_hit_samples,
            "nearby_only_hit_samples": nearby_only_hits,
            "in_bounds_fraction": (in_bounds_samples / total_samples) if total_samples else 0.0,
            "purple_hit_fraction_of_in_bounds": (purple_hit_samples / in_bounds_samples) if in_bounds_samples else 0.0,
            "completion_ratio_threshold": float(completion_ratio),
            "nearby_hit_distance_px": NEARBY_DISTANCE_PX,
        },
        "segments": {
            "segs_input": len(required_segments),
            "segs_considered": segs_considered,
            "segs_ignored_outside_screenshot": segs_ignored_outside,
            "segs_completed": segs_completed,
            "segs_uncompleted": segs_uncompleted,
        },
        "examples": {
            "point_mappings": example_points,
            "segment_summaries": example_segs,
        },
    }

    return CompletionResult(completed=completed, uncompleted=uncompleted, debug=debug)
