"""Crashscape: interactive US traffic-accident dashboard (Streamlit).

Runs on the 500k stratified sample (``data/processed/accidents_sample.parquet``).
If that file is missing (e.g. a fresh clone without the raw data) it falls back
to the committed 50k mini sample, so the app always starts.

    streamlit run app.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.hotspots import density_figure, top_hotspots

FULL_DATASET_ROWS = 7_728_394

# Palette (same validated slots as the static charts in src/viz.py)
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#c3c2b7"
SEV_COLORS = {1: "#86b6ef", 2: "#3987e5", 3: "#1c5cab", 4: "#0d366b"}
SEQ_SCALE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

APP_COLUMNS = [
    "Severity", "high_severity", "Start_Lat", "Start_Lng", "City", "State", "Street",
    "year", "month_name", "day_of_week", "dow", "hour", "is_weekend", "day_night",
    "weather_bucket", "road_type", "Junction", "Traffic_Signal", *config.ROAD_FEATURES,
]

st.set_page_config(page_title="Crashscape · US Accident Patterns", page_icon="🚦",
                   layout="wide", initial_sidebar_state="expanded")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading accident sample …")
def load_data() -> tuple[pd.DataFrame, str]:
    """Load the sample (or the mini fallback) with only the columns the app uses."""
    path = config.SAMPLE_PARQUET if config.SAMPLE_PARQUET.exists() else config.MINI_PARQUET
    cols = list(dict.fromkeys(APP_COLUMNS))
    df = pd.read_parquet(path, columns=cols)
    for c in ("City", "State", "Street"):
        df[c] = df[c].astype("category")
    # Year baseline for year-adjusted severity rates (see src/analysis.py).
    df["_year_base"] = df.groupby("year")["high_severity"].transform("mean").astype("float32")
    return df, path.name


@st.cache_data
def load_factor_tables() -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict | None]:
    """Model outputs written by ``python -m src.factors`` (optional)."""
    imp_p = config.TABLES_DIR / "factor_importance.csv"
    lvl_p = config.TABLES_DIR / "factor_level_effects.csv"
    met_p = config.TABLES_DIR / "model_metrics.json"
    imp = pd.read_csv(imp_p) if imp_p.exists() else None
    lvl = pd.read_csv(lvl_p) if lvl_p.exists() else None
    met = json.loads(met_p.read_text()) if met_p.exists() else None
    return imp, lvl, met


df_all, source_name = load_data()
SCALE = FULL_DATASET_ROWS / len(df_all)
OVERALL_HS = float(df_all["high_severity"].mean())


# --------------------------------------------------------------------------- #
# Plotly styling
# --------------------------------------------------------------------------- #
def style(fig: go.Figure, height: int = 340, legend: bool = False) -> go.Figure:
    """Quiet, consistent chart chrome."""
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE, font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif",
                                        size=12, color=INK_2),
        showlegend=legend, hoverlabel=dict(bgcolor="white", font_color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        bargap=0.35,
    )
    fig.update_xaxes(showgrid=False, linecolor=NEUTRAL, ticks="", title_font_color=INK_2)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=NEUTRAL, ticks="",
                     title_font_color=INK_2)
    return fig


def adjusted_rate(d: pd.DataFrame, by: str) -> pd.DataFrame:
    """Accidents, raw and year-adjusted share of Severity ≥ 3 per group."""
    g = d.groupby(by, observed=True).agg(n=("high_severity", "size"),
                                         hs=("high_severity", "mean"),
                                         base=("_year_base", "mean"))
    g["raw_pct"] = g["hs"] * 100
    g["adj_pct"] = g["hs"] / g["base"] * OVERALL_HS * 100
    return g.reset_index()


def fmt_count(n: float) -> str:
    """Compact count: 1.2K / 3.4M."""
    for div, suf in ((1e6, "M"), (1e3, "K")):
        if n >= div:
            return f"{n / div:.1f}{suf}"
    return f"{n:,.0f}"


# --------------------------------------------------------------------------- #
# Sidebar filters
# --------------------------------------------------------------------------- #
st.sidebar.title("🚦 Crashscape")
st.sidebar.caption("US traffic accidents, 2016 – Mar 2023")

state_counts = df_all["State"].value_counts()
states = st.sidebar.multiselect("State", options=list(state_counts.index),
                                placeholder="All states")

city_pool = df_all if not states else df_all[df_all["State"].isin(states)]
city_counts = city_pool.groupby(["City", "State"], observed=True).size().nlargest(300)
city_labels = [f"{c}, {s}" for c, s in city_counts.index]
# A fixed key keeps the selection when the option list changes with the state filter.
cities = st.sidebar.multiselect("City", options=city_labels, placeholder="All cities",
                                key="city_filter",
                                help="Top 300 cities by accidents within the chosen states")

yr_min, yr_max = int(df_all["year"].min()), int(df_all["year"].max())
years = st.sidebar.slider("Year", yr_min, yr_max, (yr_min, yr_max))

weather = st.sidebar.multiselect("Weather", options=config.WEATHER_BUCKETS,
                                 placeholder="All weather")
light = st.sidebar.segmented_control("Light condition", ["All", "Day", "Night"],
                                     default="All")
road_types = st.sidebar.multiselect("Road type", ["Interstate", "Highway", "Local road"],
                                    placeholder="All road types")


def apply_filters(d: pd.DataFrame) -> pd.DataFrame:
    """Vectorised boolean mask over the cached frame (fast on 500k rows)."""
    # .copy(): with pandas copy-on-write, to_numpy() can return a read-only view.
    m = d["year"].between(*years).to_numpy().copy()
    if states:
        m &= d["State"].isin(states).to_numpy()
    if cities:
        keys = set(cities)
        lab = d["City"].astype(str) + ", " + d["State"].astype(str)
        m &= lab.isin(keys).to_numpy() if len(d) else m
    if weather:
        m &= d["weather_bucket"].isin(weather).to_numpy()
    if light in ("Day", "Night"):
        m &= (d["day_night"] == light).to_numpy()
    if road_types:
        m &= d["road_type"].isin(road_types).to_numpy()
    return d[m]


df = apply_filters(df_all)

st.sidebar.divider()
st.sidebar.caption(
    f"Data: `{source_name}` · {len(df_all):,} sampled rows "
    f"(stratified by State × Severity). Counts are scaled ×{SCALE:.1f} to estimate the "
    f"full {FULL_DATASET_ROWS:,}-row dataset."
)

# --------------------------------------------------------------------------- #
# Header + KPIs
# --------------------------------------------------------------------------- #
st.title("Crashscape · US accident patterns & hotspots")
st.caption("How road conditions, weather and time of day shape where accidents happen "
           "and how severe they are. Severity measures traffic impact (1 = minor delay, "
           "4 = major), not injuries.")

if df.empty:
    st.warning("No accidents match these filters. Widen the selection in the sidebar.")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Accidents", fmt_count(len(df) * SCALE),
          help=f"Estimated for the full dataset: {len(df):,} sampled accidents × {SCALE:.1f}")
sev_gap = (df["high_severity"].mean() - OVERALL_HS) * 100
k2.metric("Severe share", f"{df['high_severity'].mean() * 100:.1f}%",
          delta=f"{sev_gap:+.1f} pts vs all" if abs(sev_gap) >= 0.05 else None,
          delta_color="inverse", help="Share of accidents with Severity ≥ 3 (unadjusted)")
k3.metric("At night", f"{(df['day_night'] == 'Night').mean() * 100:.0f}%")
k4.metric("On weekends", f"{df['is_weekend'].mean() * 100:.0f}%")
top_city = df.groupby(["City", "State"], observed=True).size().idxmax()
k5.metric("Top city", str(top_city[0]), help=f"{top_city[0]}, {top_city[1]}")

# A view switcher instead of st.tabs: tabs render every section on each rerun,
# this renders only the visible one, which keeps filter changes snappy.
VIEWS = ["⏱ Time", "🌦 Weather", "🛣 Road conditions", "📍 Hotspots", "🧠 Contributing factors"]
view = st.segmented_control("View", VIEWS, default=VIEWS[0], key="view",
                            label_visibility="collapsed") or VIEWS[0]

# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #
if view == VIEWS[0]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Hour of day: weekday vs weekend")
        h = (df.assign(part=np.where(df["is_weekend"], "Weekend", "Weekday"))
               .groupby(["part", "hour"], observed=True).size().rename("n").reset_index())
        h["share"] = h["n"] / h.groupby("part")["n"].transform("sum") * 100
        fig = px.line(h, x="hour", y="share", color="part",
                      color_discrete_map={"Weekday": BLUE, "Weekend": ORANGE},
                      labels={"hour": "Hour (local)", "share": "% of week-part's accidents",
                              "part": ""})
        fig.update_traces(line_width=2.5, hovertemplate="%{x}:00 · %{y:.1f}%<extra></extra>")
        st.plotly_chart(style(fig, legend=True), width="stretch")
    with c2:
        st.subheader("Hour × day of week")
        hm = (df.groupby(["day_of_week", "hour"], observed=False).size()
                .unstack("hour").reindex(config.DAY_ORDER) * SCALE)
        fig = px.imshow(hm, color_continuous_scale=SEQ_SCALE, aspect="auto",
                        labels={"x": "Hour (local)", "y": "", "color": "Accidents (est.)"})
        fig.update_traces(hovertemplate="%{y} %{x}:00 · %{z:,.0f}<extra></extra>", xgap=1, ygap=1)
        fig = style(fig)
        fig.update_layout(coloraxis_colorbar=dict(thickness=10, title=None))
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("By month")
        mo = df.groupby("month_name", observed=False).size().mul(SCALE).rename("n").reset_index()
        fig = px.bar(mo, x="month_name", y="n", labels={"month_name": "", "n": "Accidents (est.)"},
                     color_discrete_sequence=[BLUE])
        fig.update_traces(hovertemplate="%{x} · %{y:,.0f}<extra></extra>")
        st.plotly_chart(style(fig), width="stretch")
        st.caption("Includes Jan–Mar 2023 (partial year), which slightly lifts those months.")
    with c4:
        st.subheader("By year")
        yr = df.groupby("year").size().mul(SCALE).rename("n").reset_index()
        yr["kind"] = np.where(yr["year"] == 2023, "Partial (Jan–Mar)", "Full year")
        fig = px.bar(yr, x="year", y="n", color="kind",
                     color_discrete_map={"Full year": BLUE, "Partial (Jan–Mar)": NEUTRAL},
                     labels={"year": "", "n": "Accidents (est.)", "kind": ""})
        fig.update_traces(hovertemplate="%{x} · %{y:,.0f}<extra></extra>")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(style(fig, legend=True), width="stretch")

# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #
if view == VIEWS[1]:
    w = adjusted_rate(df, "weather_bucket").sort_values("n")
    w["est"] = w["n"] * SCALE
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Accidents by weather")
        fig = px.bar(w, x="est", y="weather_bucket", orientation="h",
                     color_discrete_sequence=[BLUE],
                     labels={"est": "Accidents (est.)", "weather_bucket": ""})
        fig.update_traces(hovertemplate="%{y} · %{x:,.0f}<extra></extra>")
        st.plotly_chart(style(fig), width="stretch")
    with c2:
        st.subheader("Severe share by weather (year-adjusted)")
        w_ok = w[w["n"] >= 30]
        fig = px.bar(w_ok, x="adj_pct", y="weather_bucket", orientation="h",
                     color=np.where(w_ok["adj_pct"] > OVERALL_HS * 100 + 1, "Above average",
                                    "At/below average"),
                     color_discrete_map={"Above average": BLUE, "At/below average": NEUTRAL},
                     custom_data=["raw_pct", "n"],
                     labels={"adj_pct": "% Severity ≥ 3", "weather_bucket": "", "color": ""})
        fig.add_vline(x=OVERALL_HS * 100, line_color=INK_2, line_width=1)
        fig.update_traces(hovertemplate="%{y}: %{x:.1f}% adjusted (raw %{customdata[0]:.1f}%, "
                                        "n=%{customdata[1]:,})<extra></extra>")
        st.plotly_chart(style(fig, legend=True), width="stretch")
    st.caption("Year-adjusted: severity labelling changed over 2016–2023, so each group's rate "
               "is compared with what its mix of years would predict. Groups with < 30 sampled "
               "accidents are hidden.")

    st.subheader("Severity mix within each weather condition")
    mix = (pd.crosstab(df["weather_bucket"], df["Severity"], normalize="index") * 100)
    mix = mix.reset_index().melt(id_vars="weather_bucket", var_name="Severity", value_name="pct")
    fig = px.bar(mix, x="pct", y="weather_bucket", color="Severity", orientation="h",
                 color_discrete_map={s: c for s, c in SEV_COLORS.items()},
                 category_orders={"Severity": [1, 2, 3, 4]},
                 labels={"pct": "% of accidents", "weather_bucket": ""})
    fig.update_traces(hovertemplate="%{y} · Severity %{fullData.name}: %{x:.1f}%<extra></extra>",
                      marker_line_color=SURFACE, marker_line_width=1)
    st.plotly_chart(style(fig, height=320, legend=True), width="stretch")

# --------------------------------------------------------------------------- #
# Road conditions
# --------------------------------------------------------------------------- #
if view == VIEWS[2]:
    c1, c2 = st.columns(2)
    rows = []
    for f in config.ROAD_FEATURES:
        mask = df[f].to_numpy()
        if mask.sum() < 30:
            continue
        a, b = df[mask], df[~mask]
        adj = lambda x: x["high_severity"].mean() / x["_year_base"].mean() * OVERALL_HS * 100
        rows.append({"feature": f.replace("_", " "), "share": mask.mean() * 100,
                     "with": adj(a), "without": adj(b), "n": int(mask.sum())})
    rf = pd.DataFrame(rows)
    with c1:
        st.subheader("Share of accidents at each road feature")
        if not rf.empty:
            r1 = rf.sort_values("share")
            fig = px.bar(r1, x="share", y="feature", orientation="h",
                         color_discrete_sequence=[BLUE],
                         labels={"share": "% of accidents", "feature": ""})
            fig.update_traces(hovertemplate="%{y}: %{x:.1f}%<extra></extra>")
            st.plotly_chart(style(fig, height=380), width="stretch")
    with c2:
        st.subheader("Severe share with vs without (year-adjusted)")
        if not rf.empty:
            r2 = rf.assign(lift=rf["with"] - rf["without"]).sort_values("lift")
            fig = go.Figure()
            for _, r in r2.iterrows():
                fig.add_trace(go.Scatter(x=[r["without"], r["with"]], y=[r["feature"]] * 2,
                                         mode="lines", line=dict(color=GRID, width=4),
                                         hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=r2["without"], y=r2["feature"], mode="markers",
                                     name="Without", marker=dict(size=10, color=NEUTRAL,
                                     line=dict(color=SURFACE, width=2)),
                                     hovertemplate="%{y} absent: %{x:.1f}%<extra></extra>"))
            fig.add_trace(go.Scatter(x=r2["with"], y=r2["feature"], mode="markers",
                                     name="With", marker=dict(size=10, color=BLUE,
                                     line=dict(color=SURFACE, width=2)),
                                     hovertemplate="%{y} present: %{x:.1f}%<extra></extra>"))
            fig.update_xaxes(title="% Severity ≥ 3", showgrid=True, gridcolor=GRID)
            st.plotly_chart(style(fig, height=380, legend=True), width="stretch")

    st.subheader("Road type")
    rt = adjusted_rate(df, "road_type")
    c3, c4 = st.columns(2)
    with c3:
        fig = px.bar(rt, x="road_type", y=rt["n"] * SCALE, color_discrete_sequence=[BLUE],
                     labels={"road_type": "", "y": "Accidents (est.)"})
        fig.update_traces(hovertemplate="%{x} · %{y:,.0f}<extra></extra>")
        st.plotly_chart(style(fig, height=280), width="stretch")
    with c4:
        fig = px.bar(rt, x="road_type", y="adj_pct", color_discrete_sequence=[BLUE],
                     labels={"road_type": "", "adj_pct": "% Severity ≥ 3 (year-adj.)"})
        fig.update_traces(hovertemplate="%{x} · %{y:.1f}%<extra></extra>")
        st.plotly_chart(style(fig, height=280), width="stretch")
    st.caption("Road type is derived from the street name: Interstate (I-xx), Highway "
               "(US/State routes, freeways, parkways), otherwise Local road.")

# --------------------------------------------------------------------------- #
# Hotspots
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Finding hotspots …", max_entries=64)
def cached_hotspots(key: tuple, _d: pd.DataFrame) -> pd.DataFrame:
    """Top-10 grid hotspots for the current filter selection (cached per filter key)."""
    cell = 0.01 if len(_d) > 2_000 else 0.02
    return top_hotspots(_d, n=10, cell_deg=cell)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_density(key: tuple, _d: pd.DataFrame) -> go.Figure:
    """Density map for the current filter selection (cached per filter key)."""
    return density_figure(_d, height=560)


if view == VIEWS[3]:
    filter_key = (tuple(states), tuple(cities), years, tuple(weather), light, tuple(road_types))
    hs = cached_hotspots(filter_key, df)
    fig = go.Figure(cached_density(filter_key, df))
    fig.add_trace(go.Scattermap(
        lat=hs["lat"], lon=hs["lng"], mode="markers+text", text=[f"#{r}" for r in hs["rank"]],
        textposition="top right", textfont=dict(color=INK, size=12),
        marker=dict(size=13, color=ORANGE), name="Top-10 hotspots",
        customdata=hs[["label", "count", "high_severity_pct"]],
        hovertemplate="%{customdata[0]}<br>%{customdata[1]:,} sampled accidents"
                      "<br>Severity ≥ 3: %{customdata[2]:.0f}%<extra></extra>"))
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})
    st.caption("Density of the filtered sample on an adaptive grid; orange = top-10 ~1 km "
               "hotspot cells. Full-data maps: `reports/maps/`.")
    show = hs[["rank", "label", "count", "avg_severity", "high_severity_pct", "peak_hour"]].rename(
        columns={"rank": "#", "label": "Hotspot (most frequent street)",
                 "count": "Sampled accidents", "avg_severity": "Avg severity",
                 "high_severity_pct": "% Severity ≥ 3", "peak_hour": "Peak hour"})
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"Avg severity": st.column_config.NumberColumn(format="%.2f"),
                                "% Severity ≥ 3": st.column_config.NumberColumn(format="%.0f%%")})

# --------------------------------------------------------------------------- #
# Contributing factors
# --------------------------------------------------------------------------- #
if view == VIEWS[4]:
    imp, lvl, met = load_factor_tables()
    if imp is None:
        st.info("Run `python -m src.factors` to train the model and populate this tab.")
    else:
        if met:
            m = met["hgb"]
            a, b, c, d = st.columns(4)
            a.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
            b.metric("PR-AUC", f"{m['pr_auc']:.3f}", help=f"Random baseline {m['base_rate']:.3f}")
            c.metric("Recall (severe)", f"{m['recall']:.0%}")
            d.metric("Conditions-only ROC-AUC", f"{met['hgb_conditions_only']['roc_auc']:.3f}",
                     help="Same model without year and state")
        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.subheader("Top factors (share of mean |SHAP|)")
            top = imp.head(15).iloc[::-1]
            fig = px.bar(top, x="share_pct", y="label", orientation="h",
                         color=np.where(top["family"] == "Reporting context",
                                        "Reporting context (year, state)", "Road / time / weather"),
                         color_discrete_map={"Reporting context (year, state)": NEUTRAL,
                                             "Road / time / weather": BLUE},
                         custom_data=["direction"],
                         labels={"share_pct": "% of importance", "label": "", "color": ""})
            fig.update_traces(hovertemplate="%{y}: %{x:.1f}%<br>%{customdata[0]}<extra></extra>")
            st.plotly_chart(style(fig, height=470, legend=True), width="stretch")
        with c2:
            st.subheader("Odds multipliers by condition")
            if lvl is not None:
                groups = {"road_type": "Road type", "time_of_day": "Time of day",
                          "precipitation": "Precipitation", "weather_bucket": "Weather",
                          "day_of_week": "Day of week", "road_feature": "Road feature",
                          "temperature": "Temperature"}
                pick = st.selectbox("Factor", list(groups), format_func=groups.get)
                g = lvl[lvl["group"] == pick].sort_values("odds_x")
                g = g.assign(level=g["level"].str.replace("_", " "), delta=g["odds_x"] - 1, one=1.0)
                # Bars grow from 1.0 (no effect) so raising and lowering read symmetrically.
                fig = px.bar(g, x="delta", y="level", orientation="h", base="one",
                             color=np.where(g["odds_x"] >= 1, "Raises odds", "Lowers odds"),
                             color_discrete_map={"Raises odds": ORANGE, "Lowers odds": BLUE},
                             custom_data=["odds_x", "n"],
                             labels={"delta": "Odds multiplier vs average", "level": "",
                                     "color": ""})
                fig.update_traces(hovertemplate="%{y}: ×%{customdata[0]:.2f} "
                                                "(n=%{customdata[1]:,})<extra></extra>")
                fig.add_vline(x=1, line_color=INK_2, line_width=1)
                st.plotly_chart(style(fig, height=380, legend=True), width="stretch")
        st.caption("Model: HistGradientBoosting on the 500k sample predicting Severity ≥ 3; "
                   "SHAP values on 5k test rows. These are model-wide results (not filtered). "
                   "Full write-up: `reports/factors_explained.md`.")
