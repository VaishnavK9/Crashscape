"""Contributing-factors model: what predicts a high-severity (Severity ≥ 3) accident?

Model: scikit-learn ``HistGradientBoostingClassifier``, a LightGBM-style
histogram gradient-boosting model that needs no system OpenMP library. It's
trained on the 500k stratified sample, with a RandomForest baseline for
comparison. SHAP (TreeExplainer) attributes each prediction to the features;
one-hot columns are summed back to their original feature.

Design choices:
* **No leakage.** ``Distance(mi)`` and ``duration_min`` are measured *after*
  the accident and Severity is defined by traffic delay, so both are excluded.
* **Reporting drift is modelled, not hidden.** ``year`` and ``State`` are
  included as context features. They absorb changes in how severity was
  labelled across years and data sources, so the weather/road/time effects are
  estimated *net* of that drift. A second "conditions-only" model without them
  shows how much signal the conditions carry on their own.

Usage::

    python -m src.factors
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from src import config, viz

# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
TIME_FEATURES = ["hour", "dow", "month", "is_weekend", "is_night"]
WEATHER_FEATURES = [*config.WEATHER_NUMERIC, "precip_missing", "weather_bucket"]
ROAD_FEATURES = ["road_type", *config.ROAD_FEATURES, "road_feature_count"]
CONTEXT_FEATURES = ["year", "State"]
CATEGORICAL = ["weather_bucket", "road_type", "State"]

FAMILY: dict[str, str] = {
    **{f: "Time" for f in TIME_FEATURES},
    **{f: "Weather" for f in WEATHER_FEATURES},
    **{f: "Road" for f in ROAD_FEATURES},
    "year": "Reporting context", "State": "Reporting context",
}

PRETTY: dict[str, str] = {
    "hour": "Hour of day", "dow": "Day of week", "month": "Month",
    "is_weekend": "Weekend", "is_night": "Night-time",
    "Temperature(F)": "Temperature", "Humidity(%)": "Humidity", "Pressure(in)": "Air pressure",
    "Visibility(mi)": "Visibility", "Wind_Speed(mph)": "Wind speed",
    "Precipitation(in)": "Precipitation", "precip_missing": "Precipitation not reported",
    "weather_bucket": "Weather condition", "road_type": "Road type",
    "road_feature_count": "No. of road features", "year": "Year", "State": "State",
    **{f: f.replace("_", " ") for f in config.ROAD_FEATURES},
}

LOAD_COLUMNS = ["high_severity", "Severity", "day_night", *TIME_FEATURES[:-1],
                *WEATHER_FEATURES, *ROAD_FEATURES, *CONTEXT_FEATURES]

MODEL_PATH: Path = config.PROCESSED_DIR / "severity_model.joblib"
SHAP_ROWS: int = 5_000


def build_features(df: pd.DataFrame, include_context: bool = True) -> pd.DataFrame:
    """Model matrix: numeric/boolean features plus one-hot encoded categoricals."""
    df = df.assign(is_night=df["day_night"].astype(str).eq("Night"))
    cols = TIME_FEATURES + WEATHER_FEATURES + ROAD_FEATURES
    if include_context:
        cols = cols + CONTEXT_FEATURES
    X = df[cols].copy()
    cats = [c for c in CATEGORICAL if c in X.columns]
    X = pd.get_dummies(X, columns=cats, prefix_sep="=", dtype=np.uint8)
    return X.astype(np.float32)


def base_feature(col: str) -> str:
    """Map a one-hot column like ``weather_bucket=Rain`` back to ``weather_bucket``."""
    return col.split("=", 1)[0]


# --------------------------------------------------------------------------- #
# Training & evaluation
# --------------------------------------------------------------------------- #
def make_model() -> HistGradientBoostingClassifier:
    """Gradient-boosting classifier with class weighting and early stopping."""
    return HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=600, max_leaf_nodes=63, min_samples_leaf=80,
        l2_regularization=1.0, class_weight="balanced", early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=30, random_state=config.RANDOM_STATE,
    )


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """ROC-AUC, PR-AUC and precision/recall/F1 at the 0.5 threshold."""
    p = model.predict_proba(X)[:, 1]
    pred = p >= 0.5
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "base_rate": float(y.mean()),
    }


def train(df: pd.DataFrame) -> dict:
    """Train the main model, a conditions-only model and a RandomForest baseline."""
    y = df["high_severity"].astype(int)
    X = build_features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=config.RANDOM_STATE)
    results: dict = {"n_train": len(X_tr), "n_test": len(X_te), "n_features": X.shape[1]}

    t0 = time.perf_counter()
    model = make_model().fit(X_tr, y_tr)
    results["hgb"] = {**evaluate(model, X_te, y_te), "iterations": int(model.n_iter_),
                      "fit_seconds": round(time.perf_counter() - t0, 1)}
    print(f"  HistGradientBoosting  {results['hgb']}")

    Xc = build_features(df, include_context=False)
    Xc_tr, Xc_te = Xc.loc[X_tr.index], Xc.loc[X_te.index]
    cond = make_model().fit(Xc_tr, y_tr)
    results["hgb_conditions_only"] = evaluate(cond, Xc_te, y_te)
    print(f"  Conditions only       {results['hgb_conditions_only']}")

    t0 = time.perf_counter()
    rf_idx = X_tr.sample(min(200_000, len(X_tr)), random_state=config.RANDOM_STATE).index
    rf = RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=20,
                                class_weight="balanced_subsample", n_jobs=-1,
                                random_state=config.RANDOM_STATE)
    rf.fit(X_tr.loc[rf_idx], y_tr.loc[rf_idx])
    results["random_forest"] = {**evaluate(rf, X_te, y_te),
                                "fit_seconds": round(time.perf_counter() - t0, 1)}
    print(f"  RandomForest baseline {results['random_forest']}")

    return {"model": model, "X_test": X_te, "y_test": y_te, "metrics": results}


# --------------------------------------------------------------------------- #
# SHAP explanations
# --------------------------------------------------------------------------- #
def shap_values(model, X: pd.DataFrame, n: int = SHAP_ROWS) -> tuple[np.ndarray, pd.DataFrame]:
    """SHAP values (log-odds) for a random subset of ``X``."""
    Xs = X.sample(min(n, len(X)), random_state=config.RANDOM_STATE)
    sv = shap.TreeExplainer(model).shap_values(Xs)
    if isinstance(sv, list):  # older SHAP returns one array per class
        sv = sv[1]
    if sv.ndim == 3:
        sv = sv[..., 1]
    return sv, Xs


def aggregate_shap(sv: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """Sum SHAP over one-hot columns → one column per original feature."""
    frame = pd.DataFrame(sv, columns=X.columns, index=X.index)
    return frame.T.groupby(base_feature).sum().T


def _direction(feature: str, s: pd.Series, X: pd.DataFrame) -> str:
    """Plain-English direction of a feature's effect, derived from its SHAP values."""
    onehot = [c for c in X.columns if c.startswith(feature + "=")]
    if onehot:
        level = pd.Series({c.split("=", 1)[1]: s[X[c] == 1].mean() for c in onehot}).dropna()
        k = max(1, min(3, len(level) // 2))
        up, down = level.nlargest(k), level.nsmallest(k)
        return (f"raises risk most for {', '.join(up.index)}; "
                f"lowers it most for {', '.join(down.index)}")
    x = X[feature]
    if set(np.unique(x)) <= {0.0, 1.0}:
        diff = s[x == 1].mean() - s[x == 0].mean()
        return f"{'raises' if diff > 0 else 'lowers'} risk when present ({diff:+.2f} log-odds)"
    # Numeric: report where along the range the average SHAP is highest / lowest.
    bins = pd.qcut(x, q=10, duplicates="drop")
    m = s.groupby(bins, observed=True).mean()
    if m.max() - m.min() < 0.05:
        return "little effect across its range"
    fmt = lambda iv: f"{max(iv.left, x.min()):.4g}–{iv.right:.4g}"
    return f"highest risk at {fmt(m.idxmax())}, lowest at {fmt(m.idxmin())}"


def importance_table(agg: pd.DataFrame, Xs: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per original feature with family, share and direction."""
    imp = agg.abs().mean().sort_values(ascending=False)
    t = pd.DataFrame({"feature": imp.index, "mean_abs_shap": imp.values})
    t["label"] = t["feature"].map(PRETTY).fillna(t["feature"])
    t["family"] = t["feature"].map(FAMILY)
    t["share_pct"] = t["mean_abs_shap"] / t["mean_abs_shap"].sum() * 100
    t["direction"] = [_direction(f, agg[f], Xs) for f in t["feature"]]
    return t


def level_effects(agg: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Average SHAP effect for interpretable levels/ranges of key features.

    ``odds_x`` = exp(mean SHAP): how much that level multiplies the odds of a
    Severity ≥ 3 accident relative to the model's average prediction.
    """
    groups: dict[str, dict[str, pd.Series]] = {}
    for feat in ("road_type", "weather_bucket"):
        cols = [c for c in X.columns if c.startswith(feat + "=")]
        groups[feat] = {c.split("=", 1)[1]: X[c] == 1 for c in cols}
    groups["road_feature"] = {f: X[f] == 1 for f in config.ROAD_FEATURES if X[f].sum() >= 30}
    groups["day_of_week"] = {d: X["dow"] == i for i, d in enumerate(config.DAY_ORDER)}
    hour_bands = {"Night (0-5)": (0, 5), "Morning rush (6-9)": (6, 9), "Midday (10-15)": (10, 15),
                  "Evening rush (16-19)": (16, 19), "Late evening (20-23)": (20, 23)}
    groups["time_of_day"] = {k: X["hour"].between(a, b) for k, (a, b) in hour_bands.items()}
    pr, reported = X["Precipitation(in)"], X["precip_missing"] == 0
    groups["precipitation"] = {"None": reported & (pr == 0),
                               "Light (≤0.05 in/hr)": reported & pr.between(0.001, 0.05),
                               "Heavy (>0.05 in/hr)": reported & (pr > 0.05)}
    t = X["Temperature(F)"]
    groups["temperature"] = {"Below 32°F": t < 32, "32–50°F": t.between(32, 50),
                             "50–80°F": t.between(50.01, 80), "Above 80°F": t > 80}
    feature_of = {"road_type": "road_type", "weather_bucket": "weather_bucket",
                  "day_of_week": "dow", "time_of_day": "hour",
                  "precipitation": "Precipitation(in)", "temperature": "Temperature(F)"}

    rows = []
    for group, levels in groups.items():
        for level, mask in levels.items():
            if mask.sum() < 30:
                continue
            col = level if group == "road_feature" else feature_of[group]
            mean = float(agg.loc[mask, col].mean())
            rows.append({"group": group, "level": level, "n": int(mask.sum()),
                         "mean_shap": mean, "odds_x": float(np.exp(mean))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
FAMILY_ORDER = ["Road", "Reporting context", "Weather", "Time"]
DIVERGING = LinearSegmentedColormap.from_list("crash_div", ["#2a78d6", "#d9d8d2", "#e34948"])


def plot_families(imp: pd.DataFrame) -> Path:
    """Share of total SHAP importance by feature family."""
    viz.set_style()
    fam = imp.groupby("family")["share_pct"].sum().reindex(FAMILY_ORDER).dropna()
    fam = fam.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    viz._hbar(ax, fam.index, fam.values, fmt=lambda v: f"{v:.0f}%",
              highlight=lambda l: l != "Reporting context")
    ax.set_title("Where an accident happens matters more than the weather or the clock")
    viz._subtitle(ax, "Share of the model's total SHAP importance by feature family "
                      "(gray = reporting drift: year and state)")
    viz._source(fig, "HistGradientBoosting on the 500k sample")
    return viz.save(fig, "19_factor_families")


def plot_top_factors(imp: pd.DataFrame, n: int = 15) -> Path:
    """Top-``n`` features by mean |SHAP|."""
    viz.set_style()
    top = imp.head(n)
    fig, ax = plt.subplots(figsize=(9, 6))
    viz._hbar(ax, top["label"], top["share_pct"], fmt=lambda v: f"{v:.1f}%",
              highlight=lambda l: top.set_index("label").loc[l, "family"] != "Reporting context")
    ax.set_title("After reporting drift, road type is the strongest predictor")
    viz._subtitle(ax, f"Top {n} features, share of mean |SHAP| (gray = reporting context)")
    viz._source(fig, "target: Severity ≥ 3")
    return viz.save(fig, "20_top_factors")


def plot_shap_summary(sv: np.ndarray, Xs: pd.DataFrame, imp: pd.DataFrame, n: int = 15) -> Path:
    """SHAP beeswarm for the top non-one-hot features (blue = low value, red = high)."""
    viz.set_style()
    plain = [f for f in imp["feature"] if f in Xs.columns][:n]
    idx = [Xs.columns.get_loc(f) for f in plain]
    plt.figure(figsize=(9, 6.5))
    shap.summary_plot(sv[:, idx], Xs[plain], feature_names=[PRETTY.get(f, f) for f in plain],
                      cmap=DIVERGING, show=False, plot_size=None, alpha=0.6)
    fig, ax = plt.gcf(), plt.gca()
    ax.set_xlabel("SHAP value (impact on log-odds of Severity ≥ 3)", color=viz.INK_2)
    ax.set_title("How each numeric/flag feature pushes the prediction")
    viz._subtitle(ax, f"Each dot is one accident (n={len(Xs):,}); colour = feature value")
    viz._source(fig)
    return viz.save(fig, "21_shap_summary")


def plot_dependence(agg: pd.DataFrame, Xs: pd.DataFrame) -> Path:
    """Mean SHAP across the range of four interpretable features."""
    viz.set_style()
    specs = [("hour", np.arange(25) - 0.5), ("Temperature(F)", np.arange(-10, 111, 10)),
             ("Visibility(mi)", [0, 0.5, 1, 2, 3, 5, 7, 9.9, 30]),
             ("Precipitation(in)", [-0.001, 0.0005, 0.02, 0.05, 0.1, 0.2, 5])]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, (f, bins) in zip(axes, specs):
        b = pd.cut(Xs[f], bins=bins)
        g = pd.DataFrame({"x": Xs[f], "s": agg[f]}).groupby(b, observed=True)
        m = g.agg(x=("x", "median"), s=("s", "mean"), n=("s", "size"))
        m = m[m["n"] >= 30]
        ax.axhline(0, color=viz.AXIS, linewidth=1)
        ax.plot(m["x"], m["s"], color=viz.BLUE, marker="o", markersize=5,
                markeredgecolor=viz.SURFACE, markeredgewidth=1.5)
        ax.set_xlabel(PRETTY[f])
        if f == "Precipitation(in)":
            ax.set_xscale("symlog", linthresh=0.01)
    axes[0].set_ylabel("Mean SHAP (log-odds)")
    axes[0].set_xticks([0, 6, 12, 18, 23])
    axes[0].set_title("Evenings and heavy rain raise severity; cold and low visibility add little "
                      "once place and season are known")
    viz._subtitle(axes[0], "Average SHAP contribution across each feature's range "
                           "(above 0 = raises risk of Severity ≥ 3)")
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.2, wspace=0.12)
    viz._source(fig)
    return viz.save(fig, "22_shap_dependence")


# --------------------------------------------------------------------------- #
# Plain-English report
# --------------------------------------------------------------------------- #
def _plain_english(levels: pd.DataFrame, fam: pd.Series, imp: pd.DataFrame) -> list[str]:
    """Bullet points in plain English, with numbers taken from ``levels``."""
    L = levels.set_index(["group", "level"])["odds_x"]

    def x(group: str, level: str) -> str:
        return f"×{L[(group, level)]:.2f}" if (group, level) in L.index else "n/a"

    rf = levels[levels["group"] == "road_feature"].sort_values("odds_x")
    calm = ", ".join(rf.head(4)["level"].str.replace("_", " "))
    tod = levels[levels["group"] == "time_of_day"].sort_values("odds_x")
    wx = levels[(levels["group"] == "weather_bucket") & (levels["level"] != "Other")]
    wx = wx.sort_values("odds_x", ascending=False)  # "Other" is mostly missing weather
    vis_share = imp.set_index("feature").loc["Visibility(mi)", "share_pct"]
    return [
        f"1. **Road type is the biggest real-world factor.** On an interstate the odds of a "
        f"severe accident are {x('road_type', 'Interstate')} the average; on a local road "
        f"{x('road_type', 'Local road')}. Highway crashes block lanes for longer, and lane "
        "blockage is exactly what Severity measures.",
        f"2. **Controlled, slow-speed spots produce milder accidents.** {calm} all push the "
        f"odds *down* (e.g. crossing {x('road_feature', 'Crossing')}, traffic signal "
        f"{x('road_feature', 'Traffic_Signal')}). These are frequent fender-benders at low speed.",
        f"3. **Junctions are the exception** ({x('road_feature', 'Junction')}). In this data a "
        "junction is often a highway exit, where merging traffic meets higher speeds.",
        f"4. **Time of day is non-linear.** Odds are highest in the "
        f"{tod.iloc[-1]['level'].lower()} ({tod.iloc[-1]['odds_x']:.2f}×) and lowest in the "
        f"{tod.iloc[0]['level'].lower()} ({tod.iloc[0]['odds_x']:.2f}×): rush hour brings "
        "*many* minor crashes, while evening crashes are fewer but more serious.",
        f"5. **Weekends are worse per accident.** Saturday {x('day_of_week', 'Saturday')} and "
        f"Sunday {x('day_of_week', 'Sunday')} vs Wednesday {x('day_of_week', 'Wednesday')}.",
        f"6. **Heavy rain matters, light rain barely.** Precipitation above 0.05 in/hr: "
        f"{x('precipitation', 'Heavy (>0.05 in/hr)')}; light: "
        f"{x('precipitation', 'Light (≤0.05 in/hr)')}. Weather condition ranks highest for "
        f"{', '.join(wx.head(2)['level'])} and lowest for {', '.join(wx.tail(2)['level'])}.",
        f"7. **Cold and low visibility add little on their own.** Below freezing: "
        f"{x('temperature', 'Below 32°F')}; visibility carries only {vis_share:.1f}% of the "
        "model's importance. Raw data shows more severe crashes in the cold, but "
        "the model attributes that to *where and when* it's cold (northern states, winter "
        "months), not the temperature itself.",
        f"8. **Caveat: reporting drift.** Year and State account for {fam.get('Reporting context', 0):.0f}% "
        "of the model's attributions. Severity labelling changed across years and data "
        "sources, so these are controls rather than real-world risk factors. Weather/road/time "
        "effects above are estimated *net* of them.",
    ]


def write_report(imp: pd.DataFrame, metrics: dict, levels: pd.DataFrame, agg: pd.DataFrame,
                 Xs: pd.DataFrame, path: Path = config.REPORTS_DIR / "factors_explained.md") -> Path:
    """Write an auto-generated plain-English summary of the model and top factors."""
    m, c, rf = metrics["hgb"], metrics["hgb_conditions_only"], metrics["random_forest"]
    fam = imp.groupby("family")["share_pct"].sum().sort_values(ascending=False)
    lines = [
        "# Contributing factors: what predicts a severe accident?",
        "",
        "_Auto-generated by `python -m src.factors`._",
        "",
        "**Target:** Severity ≥ 3 (major traffic impact), "
        f"{m['base_rate'] * 100:.1f}% of accidents. **Data:** 500k stratified sample, "
        f"80/20 stratified split ({metrics['n_train']:,} train / {metrics['n_test']:,} test), "
        f"{metrics['n_features']} model columns.",
        "",
        "## Model performance (test set)",
        "",
        "| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|",
        f"| HistGradientBoosting (all features) | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | "
        f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |",
        f"| HistGradientBoosting (conditions only, no year/state) | {c['roc_auc']:.3f} | "
        f"{c['pr_auc']:.3f} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} |",
        f"| RandomForest baseline | {rf['roc_auc']:.3f} | {rf['pr_auc']:.3f} | "
        f"{rf['precision']:.3f} | {rf['recall']:.3f} | {rf['f1']:.3f} |",
        "",
        f"A random guess scores ROC-AUC 0.5 and PR-AUC {m['base_rate']:.3f}.",
        "",
        "## Importance by feature family (share of mean |SHAP|)",
        "",
        *[f"- **{k}**: {v:.0f}%" for k, v in fam.items()],
        "",
        "## Top 10 factors",
        "",
        "| # | Factor | Family | Share | Effect |",
        "|---|---|---|---|---|",
        *[f"| {i} | {r.label} | {r.family} | {r.share_pct:.1f}% | {r.direction} |"
          for i, r in enumerate(imp.head(10).itertuples(), start=1)],
        "",
        "## In plain English",
        "",
        "Odds multipliers below are exp(mean SHAP) for that level: how much it multiplies the "
        "odds of a Severity ≥ 3 accident relative to the model's average prediction.",
        "",
        *_plain_english(levels, fam, imp),
        "",
    ]
    path.write_text("\n".join(lines))
    return path


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def main(explain_only: bool = False) -> None:
    """Train (unless ``explain_only``), explain and chart the contributing-factors model."""
    from src.data_loader import load_parquet

    df = load_parquet(config.SAMPLE_PARQUET, columns=list(dict.fromkeys(LOAD_COLUMNS)))
    metrics_path = config.TABLES_DIR / "model_metrics.json"
    if explain_only and MODEL_PATH.exists() and metrics_path.exists():
        print("Re-using saved model ...")
        y = df["high_severity"].astype(int)
        X = build_features(df)
        _, X_test, _, _ = train_test_split(X, y, test_size=0.2, stratify=y,
                                           random_state=config.RANDOM_STATE)
        out = {"model": joblib.load(MODEL_PATH), "X_test": X_test,
               "metrics": json.loads(metrics_path.read_text())}
    else:
        print("Training ...")
        out = train(df)
        joblib.dump(out["model"], MODEL_PATH)
        metrics_path.write_text(json.dumps(out["metrics"], indent=2, default=float))

    print("Explaining with SHAP ...")
    sv, Xs = shap_values(out["model"], out["X_test"])
    agg = aggregate_shap(sv, Xs)
    imp = importance_table(agg, Xs)
    imp.round(5).to_csv(config.TABLES_DIR / "factor_importance.csv", index=False)
    levels = level_effects(agg, Xs)
    levels.round(4).to_csv(config.TABLES_DIR / "factor_level_effects.csv", index=False)

    for p in (plot_families(imp), plot_top_factors(imp), plot_shap_summary(sv, Xs, imp),
              plot_dependence(agg, Xs), write_report(imp, out["metrics"], levels, agg, Xs)):
        print(f"✔ {p.relative_to(config.ROOT_DIR)}")
    print(imp.head(12)[["label", "family", "share_pct", "direction"]].to_string(index=False))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Contributing-factors model.")
    ap.add_argument("--explain-only", action="store_true",
                    help="Skip training; re-use the saved model to redo SHAP + charts.")
    main(explain_only=ap.parse_args().explain_only)
