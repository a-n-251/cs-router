from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2


@dataclass
class GeoRef:
    H: np.ndarray
    H_inv: np.ndarray
    img_w: int
    img_h: int


def _get_pixel_xy(cp) -> Tuple[float, float]:
    if hasattr(cp, "x") and hasattr(cp, "y"):
        return float(cp.x), float(cp.y)
    if hasattr(cp, "pixel_x") and hasattr(cp, "pixel_y"):
        return float(cp.pixel_x), float(cp.pixel_y)
    if hasattr(cp, "px") and hasattr(cp, "py"):
        return float(cp.px), float(cp.py)
    raise AttributeError("ControlPoint missing pixel coordinates")


def compute_georef(
    control_points: List[object],
    img_w: int,
    img_h: int,
) -> GeoRef:

    if len(control_points) < 3:
        raise ValueError("At least 3 control points are required")

    src = []
    dst = []

    for cp in control_points:
        px, py = _get_pixel_xy(cp)
        src.append([px, py])
        dst.append([float(cp.lon), float(cp.lat)])

    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)

    # --- Choose correct transform ---
    if len(control_points) == 3:
        # Affine transform
        A = cv2.getAffineTransform(src, dst)  # 2x3
        H = np.vstack([A, [0, 0, 1]])
    else:
        # Homography
        H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC)
        if H is None:
            raise ValueError("Homography computation failed")

    if not np.isfinite(H).all():
        raise ValueError("Invalid transform matrix")

    H_inv = np.linalg.inv(H)

    return GeoRef(
        H=H,
        H_inv=H_inv,
        img_w=img_w,
        img_h=img_h,
    )


def lonlat_to_pix(georef: GeoRef, lon: float, lat: float) -> Tuple[float, float]:
    p = np.array([lon, lat, 1.0], dtype=np.float64)
    q = georef.H_inv @ p
    if q[2] == 0:
        raise ValueError("Invalid inverse transform")
    return float(q[0] / q[2]), float(q[1] / q[2])


def pix_to_lonlat(georef: GeoRef, x: float, y: float) -> Tuple[float, float]:
    p = np.array([x, y, 1.0], dtype=np.float64)
    q = georef.H @ p
    if q[2] == 0:
        raise ValueError("Invalid transform")
    return float(q[0] / q[2]), float(q[1] / q[2])


def xy3857_to_pix(georef: GeoRef, x: float, y: float) -> Tuple[float, float]:
    return lonlat_to_pix(georef, x, y)
