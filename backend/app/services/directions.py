from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

import networkx as nx


@dataclass
class DirectionStep:
    instruction: str
    distance_m: int


def _bearing_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    ang = math.degrees(math.atan2(dy, dx))
    return (ang + 360.0) % 360.0


def _turn_phrase(delta: float) -> str:
    # delta in degrees, (-180..180], positive = left turn
    ad = abs(delta)
    if ad < 15:
        return "Continue"
    if ad < 35:
        return "Slight " + ("left" if delta > 0 else "right")
    if ad < 100:
        return "Turn " + ("left" if delta > 0 else "right")
    if ad < 160:
        return "Sharp " + ("left" if delta > 0 else "right")
    return "U-turn"


def _normalize_turn(prev_b: float, next_b: float) -> float:
    d = (next_b - prev_b + 540.0) % 360.0 - 180.0
    # in this convention, positive means "left" if y-axis is north; with 3857, y is north, ok.
    return -d  # invert to make positive => left (user preference)


def _edge_street_id(attrs: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    # Prefer name, then ref, else None
    name = attrs.get("name")
    ref = attrs.get("ref")
    way_id = attrs.get("way_id")
    return (name, ref, way_id)


def _street_label(attrs: Dict[str, Any]) -> str:
    name = attrs.get("name")
    ref = attrs.get("ref")
    if name and ref:
        return f"{name} ({ref})"
    if name:
        return str(name)
    if ref:
        return str(ref)
    return "unnamed way"


def build_directions(route: Any, prefer_lr: bool = True) -> List[Dict[str, Any]]:
    """
    Expects route to have:
      - route.node_path_xy3857: List[(x,y)] or route.node_path: List[node_id]
      - route.edge_path: List[(u,v,key?)] OR route.edge_attrs: List[dict]
      - route.G: networkx graph (optional) OR route.graph passed in route.graph
    """
    G: nx.Graph = getattr(route, "G", None) or getattr(route, "graph", None)
    node_path = getattr(route, "node_path", None)
    node_xy = getattr(route, "node_path_xy3857", None)

    # If we only have node ids, fetch xy from graph.
    if node_xy is None and node_path is not None and G is not None:
        node_xy = [(float(G.nodes[n]["x"]), float(G.nodes[n]["y"])) for n in node_path]

    if not node_xy or len(node_xy) < 2:
        return []

    # Build a list of edge attribute dicts aligned to steps between consecutive nodes.
    edge_attrs: List[Dict[str, Any]] = []
    if getattr(route, "edge_attrs", None) is not None:
        edge_attrs = list(route.edge_attrs)
    elif getattr(route, "edge_path", None) is not None and G is not None:
        for e in route.edge_path:
            if len(e) == 3 and G.is_multigraph():
                u, v, k = e
                edge_attrs.append(G.get_edge_data(u, v, k) or {})
            else:
                u, v = e[0], e[1]
                # choose best edge if multigraph
                if G.is_multigraph():
                    data = min(G.get_edge_data(u, v).values(), key=lambda d: d.get("weight", 1e9))
                    edge_attrs.append(data or {})
                else:
                    edge_attrs.append(G.get_edge_data(u, v) or {})
    else:
        # Fallback: empty attrs, still aggregate by turns
        edge_attrs = [{} for _ in range(len(node_xy) - 1)]

    # Compute per-edge distance and bearing
    segs = []
    for i in range(len(node_xy) - 1):
        a = node_xy[i]
        b = node_xy[i + 1]
        dist = float(math.hypot(b[0] - a[0], b[1] - a[1]))
        bearing = _bearing_deg(a, b)
        segs.append((dist, bearing, edge_attrs[i]))

    # Aggregate: group consecutive segments by same street id unless a big turn happens.
    steps: List[DirectionStep] = []
    cur_dist = 0.0
    cur_street = _edge_street_id(segs[0][2])
    cur_label = _street_label(segs[0][2])
    cur_phrase = "Start"

    # Helper to flush
    def flush(phrase: str, label: str, dist_m: float):
        dm = int(round(dist_m))
        if dm <= 0:
            return
        if phrase == "Continue":
            instr = f"Continue on {label}"
        elif phrase == "Start":
            instr = f"Start on {label}"
        elif phrase == "U-turn":
            instr = f"Make a U-turn on {label}"
        else:
            instr = f"{phrase} onto {label}"
        steps.append(DirectionStep(instr, dm))

    # Detect cul-de-sac U-turns by pattern: traverse into degree-1 node and return back along same street
    # We can only do this if we have node ids and graph.
    dead_end_nodes = set()
    if G is not None:
        for n in G.nodes:
            if G.degree(n) == 1:
                dead_end_nodes.add(n)

    prev_bearing = segs[0][1]
    cur_dist += segs[0][0]

    for i in range(1, len(segs)):
        dist, bearing, attrs = segs[i]
        street = _edge_street_id(attrs)
        label = _street_label(attrs)
        delta = _normalize_turn(prev_bearing, bearing)
        phrase = _turn_phrase(delta)

        big_turn = abs(delta) >= 25
        street_change = street != cur_street

        if big_turn or street_change:
            flush(cur_phrase if cur_phrase != "Start" else "Continue", cur_label, cur_dist)
            cur_dist = 0.0
            cur_phrase = phrase
            cur_street = street
            cur_label = label

        cur_dist += dist
        prev_bearing = bearing

    flush("Continue" if cur_phrase == "Start" else cur_phrase, cur_label, cur_dist)

    # Merge tiny consecutive "Continue on same street" instructions
    merged: List[DirectionStep] = []
    for s in steps:
        if merged and s.instruction.startswith("Continue on ") and merged[-1].instruction == s.instruction:
            merged[-1].distance_m += s.distance_m
        else:
            merged.append(s)

    return [{"instruction": s.instruction, "distance_m": s.distance_m} for s in merged]
