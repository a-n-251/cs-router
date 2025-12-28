from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import networkx as nx
from shapely.geometry import LineString, Point

from app.services.graph import GraphBundle, Segment


@dataclass
class Step:
    instruction: str
    distance_m: int


def _bearing(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    # bearing in degrees (0..360), in projected coords; good enough for turn classification
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    ang = math.degrees(math.atan2(dx, dy))  # note swap for "north-up" style
    return (ang + 360.0) % 360.0


def _turn_word(delta: float) -> str:
    # delta in degrees (-180..180)
    ad = abs(delta)
    if ad < 15:
        return "Continue"
    if ad < 40:
        return "Slight " + ("right" if delta > 0 else "left")
    if ad < 110:
        return ("Right" if delta > 0 else "Left")
    if ad < 160:
        return "Sharp " + ("right" if delta > 0 else "left")
    return "U-turn"


def _delta_angle(a: float, b: float) -> float:
    # smallest signed difference b-a in degrees (-180..180)
    d = (b - a + 540.0) % 360.0 - 180.0
    return d


def _nearest_node(G: nx.Graph, x: float, y: float) -> Optional[Tuple[int, int]]:
    if G.number_of_nodes() == 0:
        return None

    best = None
    bestd = float("inf")
    for n, data in G.nodes(data=True):
        nxv = float(data.get("x", n[0]))
        nyv = float(data.get("y", n[1]))
        d = (nxv - x) * (nxv - x) + (nyv - y) * (nyv - y)
        if d < bestd:
            bestd = d
            best = n
    return best


def _snap_segment_endpoints(bundle: GraphBundle, seg: Segment) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]], Tuple[float, float]]:
    G = bundle.G
    a = seg.geom_3857.coords[0]
    b = seg.geom_3857.coords[-1]
    u = _nearest_node(G, a[0], a[1])
    v = _nearest_node(G, b[0], b[1])
    mid = seg.geom_3857.interpolate(0.5, normalized=True)
    return u, v, (mid.x, mid.y)


def _shortest_path_nodes(G: nx.Graph, src: Tuple[int, int], dst: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    try:
        return nx.shortest_path(G, src, dst, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _path_length_m(G: nx.Graph, path: List[Tuple[int, int]]) -> float:
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path, path[1:]):
        ed = G[a][b]
        total += float(ed.get("distance_m", ed.get("weight", 0.0)))
    return total


def _path_to_geojson(bundle: GraphBundle, node_path: List[Tuple[int, int]]) -> Dict[str, Any]:
    # node x/y are EPSG:3857 (meters). Convert to lon/lat for frontend.
    coords = []
    for n in node_path:
        data = bundle.G.nodes[n]
        x = float(data.get("x", n[0]))
        y = float(data.get("y", n[1]))
        lon, lat = bundle.to_wgs84(x, y)
        coords.append([float(lon), float(lat)])

    return {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": coords}}],
    }


def _build_directions(bundle: GraphBundle, node_path: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
    G = bundle.G
    if not node_path or len(node_path) < 2:
        return []

    # Build per-edge records with name + length + bearing
    edges = []
    for a, b in zip(node_path, node_path[1:]):
        ed = G[a][b]
        ax = float(G.nodes[a].get("x", a[0]))
        ay = float(G.nodes[a].get("y", a[1]))
        bx = float(G.nodes[b].get("x", b[0]))
        by = float(G.nodes[b].get("y", b[1]))
        name = ed.get("name") or "unnamed road"
        length = float(ed.get("distance_m", ed.get("weight", 0.0)))
        brg = _bearing((ax, ay), (bx, by))
        edges.append((name, length, brg))

    # Collapse consecutive edges with same street name
    collapsed = []
    cur_name, cur_len, cur_brg = edges[0][0], edges[0][1], edges[0][2]
    for name, length, brg in edges[1:]:
        if name == cur_name:
            cur_len += length
        else:
            collapsed.append((cur_name, cur_len, cur_brg))
            cur_name, cur_len, cur_brg = name, length, brg
    collapsed.append((cur_name, cur_len, cur_brg))

    # Turn classification based on bearing change between collapsed legs
    steps: List[Dict[str, Any]] = []
    prev_brg = None
    for i, (name, dist, brg) in enumerate(collapsed):
        if i == 0:
            steps.append({"instruction": f"Start on {name}", "distance_m": int(round(dist))})
            prev_brg = brg
            continue
        delta = _delta_angle(prev_brg, brg)
        tw = _turn_word(delta)
        if tw == "Continue":
            instr = f"Continue on {name}"
        elif tw == "U-turn":
            instr = f"Make a U-turn onto {name}"
        else:
            instr = f"{tw} onto {name}"
        steps.append({"instruction": instr, "distance_m": int(round(dist))})
        prev_brg = brg

    return steps


def plan_route(
    bundle: GraphBundle,
    start_lon: float,
    start_lat: float,
    loop: bool,
    end_lon: Optional[float],
    end_lat: Optional[float],
    uncompleted_required_segments: List[Segment],
    max_distance_m: Optional[int],
) -> Dict[str, Any]:
    G = bundle.G
    sx, sy = bundle.to_3857(start_lon, start_lat)
    start_node = _nearest_node(G, sx, sy)

    if start_node is None:
        return {
            "route": None,
            "directions": [],
            "distance_m_est": 0,
            "covered_required_len_m_est": 0,
            "required_segments_targeted": 0,
            "required_segments_total_uncompleted": len(uncompleted_required_segments),
            "warnings": ["No walkable graph nodes in this area (OSM graph empty)."],
            "routing_debug": {"graph_nodes": 0, "graph_edges": 0},
        }

    # Determine end node if not looping and end provided
    end_node = None
    if (not loop) and end_lon is not None and end_lat is not None:
        ex, ey = bundle.to_3857(end_lon, end_lat)
        end_node = _nearest_node(G, ex, ey)

    # If looping, reserve distance to return to start.
    budget = float(max_distance_m) if max_distance_m else None

    # Snap required segments to graph and filter reachable by CC of start
    start_cc = None
    try:
        start_cc = nx.node_connected_component(G, start_node)
    except Exception:
        start_cc = set(G.nodes)

    snapped = []
    unreachable = []
    for seg in uncompleted_required_segments:
        u, v, mid = _snap_segment_endpoints(bundle, seg)
        if u is None or v is None:
            unreachable.append(seg)
            continue
        if (u not in start_cc) and (v not in start_cc):
            unreachable.append(seg)
            continue
        snapped.append((seg, u, v, mid))

    warnings = []
    if unreachable:
        warnings.append(
            f"{len(unreachable)} required segments are unreachable using OSM connectivity and were ignored. Consider adjusting the screenshot bounds."
        )

    # Greedy cover: repeatedly go to nearest uncovered segment endpoint (u or v),
    # then traverse "through" that segment by visiting its far endpoint if possible.
    current = start_node
    full_path_nodes: List[Tuple[int, int]] = [current]
    covered_len = 0.0
    remaining = snapped[:]
    targeted = []

    def append_path(p: List[Tuple[int, int]]) -> None:
        nonlocal full_path_nodes
        if not p or len(p) < 2:
            return
        if full_path_nodes and p[0] == full_path_nodes[-1]:
            full_path_nodes.extend(p[1:])
        else:
            full_path_nodes.extend(p)

    # Precompute simple "return-to-start" cost when looping
    def return_cost_from(node: Tuple[int, int]) -> float:
        if not loop:
            return 0.0
        p = _shortest_path_nodes(G, node, start_node)
        if not p:
            return float("inf")
        return _path_length_m(G, p)

    # Main greedy loop
    while remaining:
        # For each remaining segment, compute cost to reach nearest endpoint
        best = None
        best_cost = float("inf")
        best_path = None
        best_entry = None  # which endpoint entered

        for seg, u, v, mid in remaining:
            for entry, other in ((u, v), (v, u)):
                p = _shortest_path_nodes(G, current, entry)
                if not p:
                    continue
                cost = _path_length_m(G, p)

                # Optionally include traversing to the other endpoint (to "cover" the segment)
                p2 = _shortest_path_nodes(G, entry, other)
                trav = _path_length_m(G, p2) if p2 else 0.0

                total = cost + trav

                # Enforce budget (including return if looping)
                if budget is not None:
                    projected_len = _path_length_m(G, full_path_nodes) + total
                    if loop:
                        projected_len += return_cost_from(other if p2 else entry)
                    else:
                        if end_node is not None:
                            pend = _shortest_path_nodes(G, other if p2 else entry, end_node)
                            if pend:
                                projected_len += _path_length_m(G, pend)
                    if projected_len > budget:
                        continue

                if total < best_cost:
                    best_cost = total
                    best = (seg, u, v, mid)
                    best_path = (p, p2)
                    best_entry = (entry, other)

        if best is None or best_path is None:
            # Can't reach any more segments under constraints.
            break

        seg, u, v, mid = best
        p_to_entry, p_entry_to_other = best_path
        entry, other = best_entry

        # Move to entry
        append_path(p_to_entry)
        current = entry

        # Move through segment if possible (helps ensure coverage)
        if p_entry_to_other:
            append_path(p_entry_to_other)
            current = other

        targeted.append(seg.seg_id)
        covered_len += float(seg.length_m)

        remaining = [t for t in remaining if t[0].seg_id != seg.seg_id]

    # Finish: loop back to start, or go to end if provided
    if loop:
        back = _shortest_path_nodes(G, current, start_node)
        if back:
            append_path(back)
    else:
        if end_node is not None:
            pend = _shortest_path_nodes(G, current, end_node)
            if pend:
                append_path(pend)

    distance_est = _path_length_m(G, full_path_nodes)

    # Directions
    directions = _build_directions(bundle, full_path_nodes)

    route_geojson = None
    if len(full_path_nodes) >= 2:
        route_geojson = _path_to_geojson(bundle, full_path_nodes)

    routing_debug = {
        "graph_nodes": int(G.number_of_nodes()),
        "graph_edges": int(G.number_of_edges()),
        "start_node": list(start_node),
        "start_cc_nodes": int(len(start_cc)) if start_cc is not None else None,
        "uncompleted_input": int(len(uncompleted_required_segments)),
        "reachable_after_cc_filter": int(len(snapped)),
        "unreachable_count": int(len(unreachable)),
        "targeted_count": int(len(targeted)),
        "targeted_seg_ids_sample": targeted[:10],
    }

    return {
        "route": route_geojson,
        "directions": directions,
        "distance_m_est": int(round(distance_est)),
        "covered_required_len_m_est": int(round(covered_len)),
        "required_segments_targeted": int(len(targeted)),
        "required_segments_total_uncompleted": int(len(uncompleted_required_segments)),
        "warnings": warnings + ([] if route_geojson else ["No drawable route produced"]),
        "routing_debug": routing_debug,
    }
