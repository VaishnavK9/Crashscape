"""Accident hotspot detection and maps.

Approach: grid binning. Every accident is snapped to a ~1 km grid cell
(0.01° ≈ 1.1 km of latitude). Counting cells over all 7.7M points is a single
vectorised groupby, whereas DBSCAN's neighbourhood search would be far too slow
and memory-hungry at this scale. Adjacent cells of the same hotspot are
merged away with greedy non-maximum suppression, so the top-20 list contains
20 *distinct* places.

Outputs:
    reports/maps/hotspot_heatmap.html        Folium HeatMap (+ top-20 hotspot layer)
    reports/maps/la_cluster_map.html         Folium clustered markers, Los Angeles
    reports/maps/hotspot_density_plotly.html Plotly density map
    reports/tables/top20_hotspots.csv        top-20 hotspot cells
    reports/figures/17_us_hotspot_density.png, 18_top20_hotspots.png

Usage::

    python -m src.hotspots
"""

from __future__ import annotations

from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from folium import plugins
from matplotlib.colors import LogNorm

from src import config, viz

HOTSPOT_COLUMNS: list[str] = [
    "Start_Lat", "Start_Lng", "Severity", "high_severity", "City", "State",
    "Street", "hour", "Junction", "Traffic_Signal", "year",
]

US_CENTER: tuple[float, float] = (39.5, -98.35)
# Keyless light-gray basemap (CARTO's public tiles now require an API key).
BASEMAP_TILES = ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
                 "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}")
BASEMAP_ATTR = "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ"

# Heat gradient: the sequential blue ramp, skipping the near-surface steps so
# low-density areas stay visible over the light basemap.
HEAT_GRADIENT: dict[float, str] = {0.1: "#b7d3f6", 0.3: "#6da7ec", 0.55: "#2a78d6",
                                   0.8: "#184f95", 1.0: "#0d366b"}


# --------------------------------------------------------------------------- #
# Grid binning
# --------------------------------------------------------------------------- #
def grid_cells(df: pd.DataFrame, cell_deg: float = 0.01) -> pd.DataFrame:
    """Snap accidents to a regular lat/lng grid and aggregate each cell.

    Returns one row per non-empty cell with integer cell indices, the mean
    point location, accident count, mean Severity and high-severity share.
    """
    li = np.floor(df["Start_Lat"].to_numpy() / cell_deg).astype(np.int32)
    gi = np.floor(df["Start_Lng"].to_numpy() / cell_deg).astype(np.int32)
    cells = (
        pd.DataFrame({"li": li, "gi": gi, "lat": df["Start_Lat"].to_numpy(),
                      "lng": df["Start_Lng"].to_numpy(), "sev": df["Severity"].to_numpy(),
                      "hs": df["high_severity"].to_numpy()})
          .groupby(["li", "gi"], sort=False)
          .agg(count=("sev", "size"), lat=("lat", "mean"), lng=("lng", "mean"),
               avg_severity=("sev", "mean"), high_severity_pct=("hs", "mean"))
          .reset_index()
    )
    cells["high_severity_pct"] *= 100
    return cells.sort_values("count", ascending=False, ignore_index=True)


def _non_max_suppression(cells: pd.DataFrame, n: int, min_sep_cells: int) -> pd.DataFrame:
    """Greedily keep the densest cells that are > ``min_sep_cells`` apart (Chebyshev)."""
    kept: list[int] = []
    li, gi = cells["li"].to_numpy(), cells["gi"].to_numpy()
    for i in range(len(cells)):
        if all(max(abs(li[i] - li[k]), abs(gi[i] - gi[k])) > min_sep_cells for k in kept):
            kept.append(i)
            if len(kept) == n:
                break
    return cells.iloc[kept].reset_index(drop=True)


def _mode(s: pd.Series) -> str:
    vc = s.value_counts()
    return str(vc.index[0]).strip() if len(vc) else ""


def top_hotspots(df: pd.DataFrame, n: int = 20, cell_deg: float = 0.01,
                 min_sep_cells: int = 2) -> pd.DataFrame:
    """Top-``n`` distinct ~1 km hotspot cells with descriptive context.

    Args:
        df: Cleaned accidents (needs ``HOTSPOT_COLUMNS``).
        n: Number of hotspots to return.
        cell_deg: Grid cell size in degrees.
        min_sep_cells: Minimum separation between kept cells, in cells.
    """
    cells = grid_cells(df, cell_deg)
    top = _non_max_suppression(cells.head(n * 20), n, min_sep_cells)

    li = np.floor(df["Start_Lat"].to_numpy() / cell_deg).astype(np.int32)
    gi = np.floor(df["Start_Lng"].to_numpy() / cell_deg).astype(np.int32)
    rows = []
    for rank, c in enumerate(top.itertuples(), start=1):
        sub = df[(li == c.li) & (gi == c.gi)]
        rows.append({
            "rank": rank,
            "lat": round(c.lat, 5),
            "lng": round(c.lng, 5),
            "count": int(c.count),
            "avg_severity": round(c.avg_severity, 3),
            "high_severity_pct": round(c.high_severity_pct, 2),
            "city": _mode(sub["City"]),
            "state": _mode(sub["State"]),
            "top_street": _mode(sub["Street"]),
            "peak_hour": int(sub["hour"].mode().iloc[0]),
            "junction_pct": round(sub["Junction"].mean() * 100, 1),
            "signal_pct": round(sub["Traffic_Signal"].mean() * 100, 1),
            "first_year": int(sub["year"].min()),
            "last_year": int(sub["year"].max()),
        })
    out = pd.DataFrame(rows)
    out["label"] = out["top_street"] + " · " + out["city"] + ", " + out["state"]
    return out


# --------------------------------------------------------------------------- #
# Folium maps
# --------------------------------------------------------------------------- #
def _hotspot_popup(r: pd.Series) -> str:
    return (f"<b>#{r['rank']} {r['top_street']}</b><br>{r['city']}, {r['state']}<br>"
            f"Accidents: <b>{r['count']:,}</b><br>"
            f"Avg severity: {r['avg_severity']:.2f} · Severity ≥3: {r['high_severity_pct']:.0f}%<br>"
            f"Peak hour: {r['peak_hour']}:00 · At junction: {r['junction_pct']:.0f}%")


def _add_hotspot_layer(m: folium.Map, hotspots: pd.DataFrame, name: str,
                       hollow: bool = False) -> None:
    """Add sized circle markers for the hotspots as a toggleable layer.

    ``hollow`` draws rings instead of discs, so hotspots stay distinct from the
    filled bubbles of a marker-cluster layer.
    """
    layer = folium.FeatureGroup(name=name, show=True)
    max_count = hotspots["count"].max()
    for _, r in hotspots.iterrows():
        folium.CircleMarker(
            location=(r["lat"], r["lng"]),
            radius=(10 + 10 * np.sqrt(r["count"] / max_count)) if hollow
                   else (4 + 7 * np.sqrt(r["count"] / max_count)),
            color=viz.ORANGE if hollow else "#ffffff", weight=3 if hollow else 2,
            fill=True, fill_color=viz.ORANGE, fill_opacity=0.12 if hollow else 0.9,
            tooltip=f"#{r['rank']} {r['label']} ({r['count']:,})",
            popup=folium.Popup(_hotspot_popup(r), max_width=280),
        ).add_to(layer)
    layer.add_to(m)


def heatmap(df: pd.DataFrame, hotspots: pd.DataFrame, cell_deg: float = 0.04,
            out: Path = config.MAPS_DIR / "hotspot_heatmap.html") -> Path:
    """Nationwide Folium HeatMap built from grid-aggregated weights.

    Aggregating to ~4 km cells first lets the map use *every* accident while
    keeping the HTML to a few MB. Weights are capped at the 99.5th
    percentile and square-rooted, so metros saturate, interstates stay visible
    and the country view doesn't blur into one blob.
    """
    cells = grid_cells(df, cell_deg)
    c = cells["count"].to_numpy()
    w = np.sqrt(np.clip(c / np.percentile(c, 99.5), 0, 1))
    data = np.column_stack([cells["lat"].round(3), cells["lng"].round(3),
                            np.round(w, 3)]).tolist()

    m = _base_map(US_CENTER, 4)
    plugins.HeatMap(data, name="Accident density", radius=8, blur=7, min_opacity=0.12,
                    max_zoom=6, gradient=HEAT_GRADIENT).add_to(m)
    _add_hotspot_layer(m, hotspots, "Top 20 hotspots (~1 km cells)")
    folium.LayerControl(collapsed=False).add_to(m)
    _add_title(m, "Crashscape · US accident density 2016–Mar 2023",
               f"{len(df):,} accidents · orange circles = top-20 hotspots")
    m.save(out)
    return out


def city_cluster_map(df: pd.DataFrame, city: str = config.FOCUS_CITY[0],
                     state: str = config.FOCUS_CITY[1], max_points: int = 30_000,
                     out: Path = config.MAPS_DIR / "la_cluster_map.html") -> Path:
    """Clustered-marker map for one city, plus its top-10 local hotspots.

    Uses ``FastMarkerCluster`` (markers built client-side from a compact array)
    and samples at most ``max_points`` accidents so the page stays responsive.
    """
    city_df = df[(df["City"] == city) & (df["State"] == state)]
    pts = city_df.sample(min(max_points, len(city_df)), random_state=config.RANDOM_STATE)
    center = (float(city_df["Start_Lat"].median()), float(city_df["Start_Lng"].median()))

    m = _base_map(center, 10)
    callback = """
    function (row) {
        var colors = {1: '#86b6ef', 2: '#3987e5', 3: '#1c5cab', 4: '#0d366b'};
        var marker = L.circleMarker(new L.LatLng(row[0], row[1]),
            {radius: 5, color: '#ffffff', weight: 1, fillColor: colors[row[2]], fillOpacity: 0.9});
        marker.bindTooltip('Severity ' + row[2]);
        return marker;
    };
    """
    data = np.column_stack([pts["Start_Lat"].round(5), pts["Start_Lng"].round(5),
                            pts["Severity"]]).tolist()
    plugins.FastMarkerCluster(data, callback=callback,
                              name=f"Accidents (sample of {len(pts):,})").add_to(m)
    local = top_hotspots(city_df, n=10)
    _add_hotspot_layer(m, local, f"Top 10 {city} hotspots (rings)", hollow=True)
    folium.LayerControl(collapsed=False).add_to(m)
    _add_title(m, f"Crashscape · {city}, {state}",
               f"{len(city_df):,} accidents · clusters show a {len(pts):,}-point sample · "
               "orange rings = top-10 local hotspots")
    m.save(out)
    local.to_csv(config.TABLES_DIR / "la_top10_hotspots.csv", index=False)
    return out


def _base_map(location: tuple[float, float], zoom: int) -> folium.Map:
    """Folium map with the light-gray basemap as a named layer."""
    m = folium.Map(location=location, zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer(BASEMAP_TILES, attr=BASEMAP_ATTR, name="Light gray basemap",
                     max_zoom=16).add_to(m)
    return m


def _add_title(m: folium.Map, title: str, subtitle: str) -> None:
    """Overlay a small title card in the top-left corner of a Folium map."""
    html = f"""
    <div style="position:fixed;top:12px;left:56px;z-index:9999;background:rgba(252,252,251,.94);
         padding:8px 12px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.15);
         font-family:system-ui,-apple-system,'Segoe UI',sans-serif;max-width:min(520px,70vw)">
      <div style="font-weight:600;font-size:14px;color:#0b0b0b">{title}</div>
      <div style="font-size:12px;color:#52514e">{subtitle}</div>
    </div>"""
    m.get_root().html.add_child(folium.Element(html))


# --------------------------------------------------------------------------- #
# Plotly density map (also used by the Streamlit dashboard)
# --------------------------------------------------------------------------- #
PLOTLY_SCALE = [[0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"],
                [0.75, "#1c5cab"], [1.0, "#0d366b"]]


def density_figure(df: pd.DataFrame, cell_deg: float | None = None,
                   radius: int | None = None, height: int = 520) -> go.Figure:
    """Plotly density map of accidents, aggregated to a grid for speed.

    The cell size and zoom adapt to the spatial extent of ``df`` so the same
    function works for the whole US and for a single city.
    """
    lat, lng = df["Start_Lat"], df["Start_Lng"]
    span = max(float(lat.max() - lat.min()), float(lng.max() - lng.min()), 0.05)
    if cell_deg is None:
        cell_deg = float(np.clip(span / 250, 0.002, 0.08))
    cells = grid_cells(df, cell_deg)
    cells["weight"] = np.sqrt(cells["count"])
    zoom = float(np.clip(np.log2(360 / span) - 0.6, 3, 13))
    fig = px.density_map(
        cells, lat="lat", lon="lng", z="weight",
        radius=radius or int(np.clip(3 + zoom * 0.6, 5, 11)),
        range_color=(0, float(cells["weight"].quantile(0.995))),
        center={"lat": float(lat.median()), "lon": float(lng.median())}, zoom=zoom,
        map_style="carto-positron", color_continuous_scale=PLOTLY_SCALE,
        hover_data={"count": ":,", "avg_severity": ":.2f", "lat": False, "lng": False,
                    "weight": False},
        height=height,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False)
    return fig


def plotly_density_html(df: pd.DataFrame,
                        out: Path = config.MAPS_DIR / "hotspot_density_plotly.html") -> Path:
    """Write the nationwide Plotly density map (Plotly.js loaded from CDN)."""
    fig = density_figure(df, cell_deg=0.08, height=700)
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"scrollZoom": True, "displaylogo": False})
    return out


# --------------------------------------------------------------------------- #
# Static figures for the README
# --------------------------------------------------------------------------- #
def plot_us_density(df: pd.DataFrame, hotspots: pd.DataFrame, cell_deg: float = 0.05) -> Path:
    """Static US accident-density map (grid cells, log colour) with top-5 labels."""
    viz.set_style()
    cells = grid_cells(df, cell_deg).sort_values("count")
    fig, ax = plt.subplots(figsize=(13, 6.6))
    sc = ax.scatter(cells["lng"], cells["lat"], c=cells["count"], s=1.6, marker="s",
                    cmap=viz.SEQ_CMAP, norm=LogNorm(vmin=1, vmax=cells["count"].max()),
                    linewidths=0)
    placed: list[tuple[float, float]] = []
    for _, r in hotspots.head(5).iterrows():
        ax.scatter(r["lng"], r["lat"], s=70, color=viz.ORANGE, edgecolor=viz.SURFACE,
                   linewidth=2, zorder=3)
        # Stack labels of hotspots in the same metro instead of overprinting them.
        stack = sum(abs(r["lat"] - a) < 1 and abs(r["lng"] - b) < 1 for a, b in placed)
        ax.annotate(f"#{r['rank']} {r['city']}", (r["lng"], r["lat"]),
                    xytext=(9, 4 - 13 * stack), textcoords="offset points", fontsize=9,
                    color=viz.INK, weight="bold")
        placed.append((r["lat"], r["lng"]))
    ax.set_aspect(1 / np.cos(np.radians(38)))
    ax.set_xlim(-125, -66.5)
    ax.set_ylim(24, 49.6)
    ax.axis("off")
    cb = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.01, format=viz.COMPACT)
    cb.set_label("Accidents per ~5 km cell (log scale)", color=viz.INK_2)
    cb.outline.set_visible(False)
    ax.set_title("Accidents concentrate along metro corridors and interstates")
    viz._subtitle(ax, f"{len(df):,} accidents on a 0.05° grid · orange = top-5 ~1 km hotspots")
    viz._source(fig)
    return viz.save(fig, "17_us_hotspot_density")


def plot_top_hotspots(hotspots: pd.DataFrame) -> Path:
    """Bar chart of the top-20 hotspot cells."""
    viz.set_style()
    fig, ax = plt.subplots(figsize=(10, 7))
    viz._hbar(ax, [f"#{r} {l}" for r, l in zip(hotspots["rank"], hotspots["label"])],
              hotspots["count"])
    ax.set_title("The 20 most accident-dense ~1 km spots in the US")
    viz._subtitle(ax, "Accidents per hotspot cell, labelled by most frequent street")
    viz._source(fig, "grid binning, 0.01° cells")
    return viz.save(fig, "18_top20_hotspots")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def main() -> None:
    """Build hotspot table, maps and static figures from the full clean data."""
    from src.data_loader import load_parquet

    config.ensure_dirs()
    df = load_parquet(config.CLEAN_PARQUET, columns=HOTSPOT_COLUMNS)
    hs = top_hotspots(df, n=20)
    hs.to_csv(config.TABLES_DIR / "top20_hotspots.csv", index=False)
    print(hs[["rank", "label", "count", "avg_severity", "high_severity_pct", "peak_hour"]]
          .to_string(index=False))
    for path in (heatmap(df, hs), city_cluster_map(df), plotly_density_html(df),
                 plot_us_density(df, hs), plot_top_hotspots(hs)):
        print(f"✔ {path.relative_to(config.ROOT_DIR)} ({path.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
