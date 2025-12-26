from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
import math
import networkx as nx
from shapely.geometry import LineString, Polygon, Point
from pyproj import Transformer

from app.services.osm import OSMBundle


@dataclass
class Segment:
    seg_id: str
    geom_3857: LineString
    tags: Dict[str, Any]
    length_m: float


@dataclass
class GraphBundle:
    G: nx.Graph
    required_segments: List[Segment]
    tf: Transformer
    tf_inv: Transformer
    clip_poly_3857: Polygon

    def to_3857(self, lon: Union[float, Point], lat: Optional[float] = None) -> Tuple[float, float]:
        """
        Accept either:
          - to_3857(lon, lat)
          - to_3857(Point(lon, lat))
        Returns (x, y) in EPSG:3857.
        """
        if isinstance(lon, Point):
            return self.tf.transform(float(lon.x), float(lon.y))
        if lat is None:
            raise TypeError("to_3857() requires (lon, lat) or Point(lon, lat)")
        return self.tf.transform(float(lon), float(lat))

    def to_wgs84(self, x: float, y: float) -> Tuple[float, float]:
        """
        Transform projected EPSG:3857 coordinates back to lon/lat (EPSG:4326).
        """
        return self.tf_inv.transform(float(x), float(y))


def _parse_osm(raw: dict) -> Tuple[Dict[int, Tuple[float, float, Dict[str, Any]]], List[Dict[str, Any]]]:
    nodes: Dict[int, Tuple[float, float, Dict[str, Any]]] = {}
    ways: List[Dict[str, Any]] = []
    for el in raw.get("elements", []):
        if el.get("type") == "node":
            nodes[int(el["id"])] = (float(el["lon"]), float(el["lat"]), el.get("tags", {}))
        elif el.get("type") == "way":
            ways.append(
                {
                    "id": int(el["id"]),
                    "nodes": [int(n) for n in el.get("nodes", [])],
                    "tags": el.get("tags", {}),
                }
            )
    return nodes, ways


def _way_linestring_3857(
    way: Dict[str, Any],
    nodes: Dict[int, Tuple[float, float, Dict[str, Any]]],
    tf: Transformer,
) -> Optional[LineString]:
    coords = []
    for nid in way["nodes"]:
        if nid not in nodes:
            continue
        lon, lat, _ = nodes[nid]
        x, y = tf.transform(lon, lat)
        coords.append((x, y))
    if len(coords) < 2:
        return None
    return LineString(coords)


def _substring(ls: LineString, a: float, b: float) -> LineString:
    a = max(0.0, a)
    b = min(ls.length, b)
    step = 10.0
    pts = [ls.interpolate(a)]
    d = a
    while d + step < b:
        d += step
        pts.append(ls.interpolate(d))
    pts.append(ls.interpolate(b))
    return LineString([(p.x, p.y) for p in pts])


def _split_to_maxlen(ls: LineString, maxlen: float) -> List[LineString]:
    if ls.length <= maxlen:
        return [ls]
    n = int(math.ceil(ls.length / maxlen))
    out = []
    for i in range(n):
        a = (i / n) * ls.length
        b = ((i + 1) / n) * ls.length
        out.append(_substring(ls, a, b))
    return [g for g in out if g.length > 1e-2]


def _is_walkable(tags: Dict[str, Any], allow_private: bool) -> bool:
    highway = tags.get("highway")
    if highway is None:
        return False

    # exclude motorways/trunks for walking connectivity (keep conservative)
    if highway in {"motorway", "motorway_link", "trunk", "trunk_link", "expressway"}:
        return False

    if tags.get("foot") == "no":
        return False

    access = tags.get("access")
    if access == "private" and not allow_private:
        return False

    if tags.get("fee") == "yes":
        # treat fee=yes as non-walkable for MVP (avoid paywalled)
        return False

    return True


def _is_required_candidate(tags: Dict[str, Any]) -> bool:
    highway = tags.get("highway")
    if highway is None:
        return False

    if highway in {"motorway", "motorway_link", "expressway"}:
        return False
    if tags.get("foot") == "no":
        return False
    if tags.get("fee") == "yes":
        return False

    # exclude unnamed service from required
    if highway == "service" and not tags.get("name"):
        return False

    # do not require pure paths/footways
    if highway in {"path", "footway", "cycleway", "steps"}:
        return False

    return True


def build_graph_bundle(
    osm: OSMBundle,
    clip_poly_wgs84: Polygon,
    allow_private: bool,
    max_required_edge_len_m: float,
) -> GraphBundle:
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    tf_inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    clip_coords = [tf.transform(x, y) for x, y in clip_poly_wgs84.exterior.coords]
    clip3857 = Polygon(clip_coords)

    nodes, ways = _parse_osm(osm.raw)

    G = nx.Graph()

    # Round to reduce accidental duplicates; store x/y as floats in node attrs.
    def node_key(x: float, y: float) -> Tuple[int, int]:
        return (int(round(x)), int(round(y)))

    # Walk graph edges: keep weight + name/highway/way_id for directions
    for w in ways:
        tags = w["tags"]
        if not _is_walkable(tags, allow_private):
            continue

        ls = _way_linestring_3857(w, nodes, tf)
        if ls is None:
            continue

        clipped = ls.intersection(clip3857.buffer(30))
        if clipped.is_empty:
            continue

        geoms = [clipped] if isinstance(clipped, LineString) else list(getattr(clipped, "geoms", []))
        way_name = tags.get("name") or tags.get("ref") or tags.get("highway") or "way"
        highway = tags.get("highway")

        for g in geoms:
            coords = list(g.coords)
            for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
                dist = math.hypot(x2 - x1, y2 - y1)
                if dist < 0.5:
                    continue

                k1 = node_key(x1, y1)
                k2 = node_key(x2, y2)

                # store node coordinates too
                if k1 not in G:
                    G.add_node(k1, x=float(k1[0]), y=float(k1[1]))
                if k2 not in G:
                    G.add_node(k2, x=float(k2[0]), y=float(k2[1]))

                # If multiple edges collapse into one undirected edge, keep the shortest weight,
                # but keep the "best" name (prefer real name over generic).
                if G.has_edge(k1, k2):
                    if dist < float(G[k1][k2]["weight"]):
                        G[k1][k2]["weight"] = float(dist)
                        G[k1][k2]["name"] = way_name
                        G[k1][k2]["highway"] = highway
                        G[k1][k2]["way_id"] = int(w["id"])
                else:
                    G.add_edge(
                        k1,
                        k2,
                        weight=float(dist),
                        name=way_name,
                        highway=highway,
                        way_id=int(w["id"]),
                    )

    required: List[Segment] = []
    for w in ways:
        tags = w["tags"]
        if not _is_required_candidate(tags):
            continue

        ls = _way_linestring_3857(w, nodes, tf)
        if ls is None:
            continue

        clipped = ls.intersection(clip3857)
        if clipped.is_empty:
            continue

        geoms = [clipped] if isinstance(clipped, LineString) else list(getattr(clipped, "geoms", []))
        for g in geoms:
            for piece in _split_to_maxlen(g, max_required_edge_len_m):
                seg_id = f"{w['id']}:{hash(piece.wkt)}"
                required.append(
                    Segment(
                        seg_id=seg_id,
                        geom_3857=piece,
                        tags=tags,
                        length_m=float(piece.length),
                    )
                )

    return GraphBundle(
        G=G,
        required_segments=required,
        tf=tf,
        tf_inv=tf_inv,
        clip_poly_3857=clip3857,
    )
