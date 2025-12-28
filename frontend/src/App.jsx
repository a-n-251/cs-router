// frontend/src/App.jsx
import React, { useEffect, useRef, useState } from "react";
import L from "leaflet";

const BACKEND = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [imgURL, setImgURL] = useState(null);

  const [startAddress, setStartAddress] = useState("Hornsby NSW");
  const [loop, setLoop] = useState(true);
  const [endAddress, setEndAddress] = useState("");
  const [maxDist, setMaxDist] = useState("");

  const [allowPrivate, setAllowPrivate] = useState(true);

  const [controlPoints, setControlPoints] = useState([]);
  const [pendingPx, setPendingPx] = useState(null);

  const [mapSearch, setMapSearch] = useState("");
  const [mapSearchMsg, setMapSearchMsg] = useState("");

  const [planResult, setPlanResult] = useState(null);
  const [error, setError] = useState("");

  const imgRef = useRef(null);
  const mapRef = useRef(null);
  const routeMapRef = useRef(null);
  const mapObjRef = useRef(null);
  const routeMapObjRef = useRef(null);
  const routeLayerRef = useRef(null);

  useEffect(() => {
    if (!mapObjRef.current) {
      const map = L.map(mapRef.current).setView([-33.86, 151.20], 11);
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: "© OpenStreetMap",
      }).addTo(map);
      mapObjRef.current = map;
    }

    if (!routeMapObjRef.current) {
      const map = L.map(routeMapRef.current).setView([-33.86, 151.20], 11);
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: "© OpenStreetMap",
      }).addTo(map);
      routeMapObjRef.current = map;
    }
  }, []);

  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setImgURL(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const onImgClick = (e) => {
    if (!imgRef.current) return;
    const rect = imgRef.current.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    const scaleX = imgRef.current.naturalWidth / rect.width;
    const scaleY = imgRef.current.naturalHeight / rect.height;

    setPendingPx({ px: px * scaleX, py: py * scaleY });
  };

  const removeCP = (idx) => {
    setControlPoints((prev) => prev.filter((_, i) => i !== idx));
  };

const recenterMapToAddress = async () => {
  setMapSearchMsg("");
  const q = mapSearch.trim();
  if (q.length < 3) return;

  try {
    const resp = await fetch(
      `https://nominatim.openstreetmap.org/search?` +
        new URLSearchParams({
          q,
          format: "json",
          limit: "1",
        }),
      {
        headers: {
          "Accept": "application/json",
        },
      }
    );

    const data = await resp.json();
    if (!Array.isArray(data) || data.length === 0) {
      throw new Error("Not found");
    }

    const lat = parseFloat(data[0].lat);
    const lon = parseFloat(data[0].lon);

    const map = mapObjRef.current;
    if (!map) return;

    map.setView([lat, lon], Math.max(map.getZoom(), 16));
    setMapSearchMsg(data[0].display_name || "Centered");
  } catch (e) {
    setMapSearchMsg("Not found");
  }
};

  const submit = async () => {
    setError("");
    setPlanResult(null);

    if (!file) {
      setError("Upload a screenshot first.");
      return;
    }
    if (controlPoints.length < 2) {
      setError("Add at least 2 control points (3–4 recommended).");
      return;
    }

    const payload = {
      start_address: startAddress,
      loop,
      end_address: !loop && endAddress.trim() ? endAddress.trim() : null,
      max_distance_m: maxDist.trim() ? parseInt(maxDist.trim(), 10) : null,
      allow_private: allowPrivate,
      control_points: controlPoints,
    };

    const form = new FormData();
    form.append("screenshot", file);
    form.append("payload", JSON.stringify(payload));

    try {
      const resp = await fetch(`${BACKEND}/plan_multipart`, {
        method: "POST",
        body: form,
      });
      const data = await resp.json();
      if (!resp.ok) {
        const detail =
          (data && (data.detail || data.message)) ||
          (typeof data === "string" ? data : null) ||
          "Request failed";
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }

      setPlanResult(data);
      renderRoute(data);
    } catch (err) {
      // Make errors readable instead of "[object Object]"
      if (err instanceof Error) {
        setError(err.message);
      } else {
        try {
          setError(JSON.stringify(err, null, 2));
        } catch {
          setError(String(err));
        }
      }
    }
  };

  const renderRoute = (data) => {
    const map = routeMapObjRef.current;
    if (!map) return;

    if (routeLayerRef.current) {
      routeLayerRef.current.remove();
      routeLayerRef.current = null;
    }

    const bearingDeg = (lat1, lon1, lat2, lon2) => {
      const phi1 = (lat1 * Math.PI) / 180;
      const phi2 = (lat2 * Math.PI) / 180;
      const dLon = ((lon2 - lon1) * Math.PI) / 180;
      const y = Math.sin(dLon) * Math.cos(phi2);
      const x =
        Math.cos(phi1) * Math.sin(phi2) -
        Math.sin(phi1) * Math.cos(phi2) * Math.cos(dLon);
      const brng = Math.atan2(y, x);
      return ((brng * 180) / Math.PI + 360) % 360;
    };

    const makeArrowMarkers = (latlngs, stepMeters = 250) => {
      if (!latlngs || latlngs.length < 2) return [];
      const arrows = [];
      let traversed = 0;
      let target = stepMeters;

      for (let i = 1; i < latlngs.length; i++) {
        const prev = L.latLng(latlngs[i - 1]);
        const curr = L.latLng(latlngs[i]);
        const segLen = prev.distanceTo(curr);
        const bearing = bearingDeg(prev.lat, prev.lng, curr.lat, curr.lng);

        while (segLen > 0 && target <= traversed + segLen) {
          const t = (target - traversed) / segLen;
          const lat = prev.lat + (curr.lat - prev.lat) * t;
          const lng = prev.lng + (curr.lng - prev.lng) * t;

          const icon = L.divIcon({
            className: "",
            html: `<div class="route-arrow" style="transform: rotate(${bearing}deg);"></div>`,
            iconSize: [14, 14],
            iconAnchor: [7, 7],
          });
          arrows.push(
            L.marker([lat, lng], {
              icon,
              interactive: false,
              keyboard: false,
            })
          );
          target += stepMeters;
        }

        traversed += segLen;
      }
      return arrows;
    };

    // ✅ Compatibility: backend now returns "route" (FeatureCollection).
    // Keep supporting older "route_geojson" if present.
    const routeFC = data?.route_geojson ?? data?.route;

    // Support both:
    // - FeatureCollection with features[0].geometry.coordinates (current backend)
    // - a bare GeoJSON geometry / Feature
    let coords =
      routeFC?.features?.[0]?.geometry?.coordinates ??
      routeFC?.geometry?.coordinates ??
      null;

    const latlngs = coords && coords.length >= 2 ? coords.map(([lon, lat]) => [lat, lon]) : [];
    const routeGroup = L.layerGroup();
    const bounds = L.latLngBounds([]);
    const isLoop = Boolean(data?.is_loop);

    if (latlngs.length >= 2) {
      const poly = L.polyline(latlngs, { weight: 5 });
      routeGroup.addLayer(poly);
      bounds.extend(poly.getBounds());

      // Start marker (neon green circle)
      const startLatLng = latlngs[0];
      routeGroup.addLayer(
        L.circleMarker(startLatLng, {
          radius: 8,
          color: "#39ff14",
          fillColor: "#39ff14",
          fillOpacity: 0.9,
          weight: 3,
          interactive: false,
        })
      );

      // End marker (bright red hexagon) only for non-loop routes
      if (!isLoop && latlngs.length > 1) {
        const endLatLng = latlngs[latlngs.length - 1];
        const endIcon = L.divIcon({
          className: "",
          html: `<div class="route-end-icon"></div>`,
          iconSize: [18, 18],
          iconAnchor: [9, 9],
        });
        routeGroup.addLayer(
          L.marker(endLatLng, { icon: endIcon, interactive: false, keyboard: false })
        );
      }

      makeArrowMarkers(latlngs, 300).forEach((m) => routeGroup.addLayer(m));
    }

    // Unreachable segments overlay (orange dashed)
    const unreachableFC = data?.unreachable_segments;
    const unreachableIcon = L.divIcon({
      className: "",
      html: `<div class="unreachable-marker"></div>`,
      iconSize: [18, 18],
      iconAnchor: [9, 9],
    });

    if (unreachableFC?.features?.length) {
      unreachableFC.features.forEach((feat) => {
        const uc = feat?.geometry?.coordinates;
        if (!uc || uc.length < 2) return;
        const ll = uc.map(([lon, lat]) => [lat, lon]);
        const line = L.polyline(ll, {
          color: "#ff8c00",
          weight: 4,
          dashArray: "8 6",
          opacity: 0.9,
        });
        bounds.extend(line.getBounds());
        if (feat.properties?.name) {
          line.bindTooltip(`Unreachable: ${feat.properties.name}`, {
            permanent: false,
            direction: "top",
            offset: [0, -4],
          });
        }
        routeGroup.addLayer(line);

        const centroid = line.getBounds().getCenter();
        const marker = L.marker(centroid, {
          icon: unreachableIcon,
          interactive: true,
          keyboard: false,
        });
        marker.bindTooltip(
          feat.properties?.name
            ? `Unreachable: ${feat.properties.name}`
            : "Unreachable segment",
          {
            permanent: false,
            direction: "top",
            offset: [0, -6],
          }
        );
        routeGroup.addLayer(marker);
      });
    }

    routeGroup.addTo(map);
    routeLayerRef.current = routeGroup;
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [20, 20] });
    }

    (data.amenities || []).forEach((a) => {
      L.circleMarker([a.lat, a.lon], { radius: 6 })
        .addTo(map)
        .bindPopup(`${a.type}${a.name ? `: ${a.name}` : ""}`);
    });
  };

  return (
    <div>
      <div className="row">
        <div className="col panel">
          <h3>1) Upload screenshot + control points</h3>

          <label>Screenshot (CityStrides completed streets view)</label>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />

          <div style={{ height: 10 }} />

          {imgURL && (
            <div className="imgwrap">
              <img ref={imgRef} src={imgURL} onClick={onImgClick} alt="screenshot" />
            </div>
          )}

          <div className="small" style={{ marginTop: 8 }}>
            Click the screenshot to select a point, then pan/zoom the map so the matching
            point is under the crosshair and use “Use map center…”. Add 3–4 points spread
            across the image for best results.
          </div>

          {pendingPx && (
            <div className="small" style={{ marginTop: 8 }}>
              Pending screenshot point: ({pendingPx.px.toFixed(0)},{" "}
              {pendingPx.py.toFixed(0)})
            </div>
          )}

          <div style={{ marginTop: 10 }}>
            <div className="small">Control points:</div>
            <div className="list">
              {controlPoints.map((cp, idx) => (
                <div className="cpRow" key={idx}>
                  <div className="small">
                    #{idx + 1} px=({cp.px.toFixed(0)},{cp.py.toFixed(0)}) ↔ latlon=(
                    {cp.lat.toFixed(6)},{cp.lon.toFixed(6)})
                  </div>
                  <button onClick={() => removeCP(idx)}>Remove</button>
                </div>
              ))}
            </div>
          </div>

          <div style={{ height: 12 }} />

          <label>Find on map (address/place)</label>
          <div style={{ display: "flex", gap: 8 }}>
            <input
              value={mapSearch}
              onChange={(e) => setMapSearch(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") recenterMapToAddress();
              }}
              placeholder="e.g., Hornsby NSW or a park name"
            />
            <button style={{ width: 140 }} onClick={recenterMapToAddress}>
              Search
            </button>
          </div>
          {mapSearchMsg && (
            <div className="small" style={{ marginTop: 6 }}>
              {mapSearchMsg}
            </div>
          )}

          <div style={{ height: 8 }} />

          <div className="mapWrap">
            <div id="map" ref={mapRef}></div>
            <div className="crosshair"></div>
          </div>

          <div style={{ marginTop: 8, display: "flex", gap: 8 }}>
            <button
              onClick={() => {
                const map = mapObjRef.current;
                if (!map) return;

                if (!pendingPx) {
                  setError("Click the screenshot first to set a pending point.");
                  return;
                }

                const c = map.getCenter();
                const cp = { px: pendingPx.px, py: pendingPx.py, lat: c.lat, lon: c.lng };

                setControlPoints((prev) => [...prev, cp]);
                setPendingPx(null);
                setError("");
              }}
            >
              Use map center for pending point
            </button>

            <button
              onClick={() => {
                setPendingPx(null);
                setError("");
              }}
            >
              Cancel pending point
            </button>
          </div>

          <div className="small" style={{ marginTop: 6 }}>
            Pan/zoom so the target point sits under the crosshair, then click “Use map
            center…”.
          </div>
        </div>

        <div className="col panel">
          <h3>2) Plan route</h3>

          <label>Start address</label>
          <input value={startAddress} onChange={(e) => setStartAddress(e.target.value)} />

          <label style={{ marginTop: 10 }}>
            <input
              type="checkbox"
              checked={loop}
              onChange={(e) => setLoop(e.target.checked)}
            />{" "}
            Loop (end where you start)
          </label>

          {!loop && (
            <>
              <label>End address (optional)</label>
              <input
                value={endAddress}
                onChange={(e) => setEndAddress(e.target.value)}
                placeholder="Leave blank to end anywhere"
              />
            </>
          )}

          <label>Max distance (meters, optional)</label>
          <input value={maxDist} onChange={(e) => setMaxDist(e.target.value)} placeholder="e.g., 25000" />

          <label style={{ marginTop: 10 }}>
            <input
              type="checkbox"
              checked={allowPrivate}
              onChange={(e) => setAllowPrivate(e.target.checked)}
            />{" "}
            Allow private roads (default on)
          </label>

          <div style={{ height: 10 }} />
          <button onClick={submit}>Plan</button>

          {error && <div style={{ color: "crimson", marginTop: 10 }}>{error}</div>}

          {planResult && (
            <>
              <div style={{ marginTop: 10 }} className="small">
                Distance est: {planResult.stats?.distance_m_est} m
                <br />
                Covered required length est: {planResult.stats?.covered_required_len_m_est} m
                <br />
                Uncompleted segments targeted: {planResult.stats?.required_segments_targeted}/
                {planResult.stats?.required_segments_total_uncompleted}
              </div>

              {(planResult.warnings || []).length > 0 && (
                <div style={{ marginTop: 10 }}>
                  <div className="small">
                    <b>Warnings</b>
                  </div>
                  <ul className="small">
                    {planResult.warnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}

              <div style={{ marginTop: 10 }}>
                <div className="small">
                  <b>Directions</b>
                </div>
                <ol className="small list">
                  {(planResult.directions || []).slice(0, 200).map((d, i) => (
                    <li key={i}>
                      {d.instruction} (after ~{d.distance_m} m)
                    </li>
                  ))}
                </ol>
              </div>
            </>
          )}
        </div>

        <div className="col panel">
          <h3>3) Route map</h3>
          <div id="routeMap" ref={routeMapRef}></div>
          <div className="small" style={{ marginTop: 8 }}>
            Amenities shown: drinking water + toilets within ~200m (MVP bbox approximation).
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
