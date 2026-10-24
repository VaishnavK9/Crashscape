# 🚦 Crashscape: US Traffic Accident Patterns & Hotspots

> **Prodigy InfoTech Data Science Internship – Task 05**
> *Analyze traffic accident data to identify patterns related to road conditions, weather and time of day. Visualize accident hotspots and contributing factors.*

![Python](https://img.shields.io/badge/Python-3.10%2B-2a78d6) ![pandas](https://img.shields.io/badge/pandas-3.x-2a78d6) ![scikit-learn](https://img.shields.io/badge/scikit--learn-GBDT%20%2B%20SHAP-2a78d6) ![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-eb6834) ![License](https://img.shields.io/badge/license-MIT-898781)

Crashscape analyzes **7.7 million US traffic accidents (2016 – Mar 2023)**:
- **when** they happen
- **what weather and road conditions** they happen in
- **where** the ~1 km hotspots are
- **which factors** make an accident severe

It covers a memory-efficient pipeline for a 3 GB CSV, 22 publication-quality charts, three interactive maps, a gradient-boosting + SHAP factors model, a storytelling notebook and a fast Streamlit dashboard.

![Dashboard](reports/screenshots/dashboard_time.png)

---

## 📌 Key insights

| | Finding |
|---|---|
| ⏱ **Time** | Weekdays follow the commute: **8.4%** of weekday crashes happen at 7–8 AM, with a second peak at 4–5 PM. Weekends peak at 2 PM with about half the volume, but their crashes are more often severe (odds **×1.25**). |
| 🛣 **Road type** | **Interstates** carry 25% of accidents, and **37.6%** of those are severe (year-adjusted), versus **10.9%** on local roads. Road type is the single strongest real-world predictor of severity. |
| 🚦 **Road features** | Signals (15%), crossings (11%) and junctions (7%) are the top accident sites. Controlled spots produce **milder** crashes (−12 to −15 pts severe); **junctions (often highway exits) are the exception (+6 pts)**. |
| 🌧 **Weather** | **84%** of crashes happen in clear or cloudy weather. When bad weather hits, severity rises: thunderstorms **25%**, snow **24%** and rain **23%** severe, versus clear **18%**. Precipitation shows a clean dose-response, from **19% → 27%**. |
| 🌙 **Day vs night** | 69% of crashes happen in daylight and the Severity ≥ 3 share is about equal, but **Severity 4 is more common at night** (3.5% vs 2.3%). Late evening (8 PM–midnight) is the most severe time band. |
| 📍 **Hotspots** | #1 is **I-605 S, Downey CA** (3,704 accidents in ~1 km), then the **Alexander Hamilton Bridge (Bronx)** and **CA-110 (LA)**. Freeway hotspots are 23–68% severe. Miami/Orlando surface-street hotspots are ~0% severe, which points to a reporting artifact. |
| 🧠 **Model** | HistGradientBoosting reaches **ROC-AUC 0.858** (RandomForest 0.840). SHAP shows road > time > weather, and overturns a raw-data intuition: **cold itself isn't the danger** once place and season are known. |
| ⚠️ **Caveat** | Severity measures *traffic impact*, not injuries, and its labelling **drifted** from 35% → 7% severe between 2016 and 2022 as data sources changed. All severity comparisons are **year-adjusted**. |

Full story and **10 findings & safety recommendations**: [`notebooks/Crashscape_analysis.ipynb`](notebooks/Crashscape_analysis.ipynb).

---

## 📊 Dataset

**US Accidents (2016 – 2023)** by Sobhan Moosavi on Kaggle: <https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents>
- ~7.7M rows × 46 columns, about 3 GB unzipped (`US_Accidents_March23.csv`)
- Covers 49 states. Accidents were collected from traffic APIs and enriched with weather and road-feature (POI) annotations.
- Licensed **CC BY-NC-SA 4.0** (non-commercial use)

### Download

**Option A: Kaggle CLI** (needs an API token from <https://www.kaggle.com/settings> → *Create New Token*):

```bash
pip install kaggle
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d sobhanmoosavi/us-accidents -p data/raw --unzip
```

**Option B: manual.** Download the zip from the Kaggle page and put `US_Accidents_March23.csv.zip` (or the unzipped CSV) in `data/raw/`. The loader reads straight from the zip, so unzipping is optional.

### Citation (requested by the dataset author)

> Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. **"A Countrywide Traffic Accident Dataset."**, 2019. [arXiv:1906.05409](https://arxiv.org/abs/1906.05409)
>
> Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. **"Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights."** In *Proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems*, ACM, 2019.

---

## ⚙️ Pipeline

```
data/raw/US_Accidents_March23.csv.zip   (653 MB zip / 3 GB CSV, never fully loaded)
        │  src/data_loader.py   chunked read · usecols · compact dtypes → Parquet (238 MB, 69 s)
        ▼
data/processed/accidents.parquet  ──►  stratified 500k sample (State × Severity, seed 42)
        │  src/cleaning.py      mixed-format timestamps · time features · 8 weather buckets ·
        │                       outlier nulling + State×month imputation · road features + road type
        ▼
accidents_clean.parquet (7.7M) · accidents_sample.parquet (500k) · accidents_mini.parquet (50k, committed)
        │
        ├─ src/analysis.py + src/viz.py   aggregate tables (year-adjusted severity) → 16 charts
        ├─ src/hotspots.py                 ~1 km grid binning + NMS → top-20 CSV, Folium/Plotly maps
        ├─ src/factors.py                  HistGradientBoosting + RF baseline + SHAP → 4 charts + report
        └─ app.py                          Streamlit dashboard on the sample
```

**Memory efficiency**
- Only 33 of 46 columns are read, using `float32` / `int8` / `boolean` / `category` dtypes and 1M-row chunks.
- The full cleaned table fits in **~860 MB** of RAM. Column pruning drops it to 170–400 MB for each analysis step.
- Memory usage is printed after every load.

**Why year-adjusted severity?** The share of severe accidents falls from ~35% (2016–18) to ~7% (2022) as data sources changed. Comparing raw rates would confuse *which year* with *which condition*. Each group's rate is therefore compared with what its mix of years predicts (direct standardisation).

**Why grid binning, not DBSCAN?** Snapping 7.7M points to 0.01° cells is one vectorised group-by that takes seconds. DBSCAN's neighbourhood search would need far more time and memory at this scale. Non-maximum suppression then keeps 20 *distinct* hotspots.

**Why HistGradientBoosting?** It's the same histogram-GBDT algorithm as LightGBM but ships inside scikit-learn, so there's no system `libomp` dependency (which LightGBM needs on macOS). If you want LightGBM: `brew install libomp && pip install lightgbm`.

---

## 🖼 Gallery

### Time
| | |
|---|---|
| ![Hourly](reports/figures/01_hourly_weekday_vs_weekend.png) | ![Heatmap](reports/figures/04_hour_dow_heatmap.png) |
| ![Day of week](reports/figures/02_day_of_week.png) | ![Year trend](reports/figures/05_year_trend.png) |

### Weather
| | |
|---|---|
| ![Weather](reports/figures/06_weather_count_and_severity.png) | ![Weather readings](reports/figures/07_weather_numeric_vs_severity.png) |

### Road conditions, day/night & severity
| | |
|---|---|
| ![Road share](reports/figures/09_road_feature_share.png) | ![Road severity](reports/figures/10_road_feature_severity.png) |
| ![Day vs night](reports/figures/11_day_vs_night.png) | ![Severity shift](reports/figures/16_severity_shift.png) |

### Where
| | |
|---|---|
| ![Top cities](reports/figures/13_top_cities.png) | ![Per capita](reports/figures/14_state_per_capita.png) |
| ![US density](reports/figures/17_us_hotspot_density.png) | ![Top 20 hotspots](reports/figures/18_top20_hotspots.png) |

### Interactive maps ([`reports/maps/`](reports/maps/), open in a browser)
| Folium HeatMap + top-20 hotspots (zoomed: eastern US) | Los Angeles clustered markers + local hotspots |
|---|---|
| ![Heatmap](reports/screenshots/map_heatmap_east.png) | ![LA clusters](reports/screenshots/map_la_clusters.png) |

### Contributing factors
| | |
|---|---|
| ![Families](reports/figures/19_factor_families.png) | ![Top factors](reports/figures/20_top_factors.png) |
| ![SHAP summary](reports/figures/21_shap_summary.png) | ![Dependence](reports/figures/22_shap_dependence.png) |

### Dashboard
| Hotspots view | Contributing factors view |
|---|---|
| ![Dashboard hotspots](reports/screenshots/dashboard_hotspots.png) | ![Dashboard factors](reports/screenshots/dashboard_factors.png) |

---

## 🚀 How to run

```bash
git clone https://github.com/VaishnavK9/Crashscape.git && cd Crashscape
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Just the dashboard (no download needed).** It runs on the committed 50k mini-sample:

```bash
streamlit run app.py
```

**Full pipeline** (after putting the dataset in `data/raw/`; about 7 minutes in total on a laptop):

```bash
python -m src.data_loader     # CSV → Parquet + 500k raw sample        (~1.5 min)
python -m src.cleaning        # clean + features → clean/sample/mini   (~1.2 min)
python -m src.viz             # 16 analysis charts + tables            (~25 s)
python -m src.hotspots        # top-20 hotspots, maps, 2 charts        (~25 s)
python -m src.factors         # model + SHAP + 4 charts + report       (~3.5 min)
streamlit run app.py          # dashboard on the 500k sample
jupyter notebook notebooks/Crashscape_analysis.ipynb
```

`python -m src.factors --explain-only` redraws the SHAP charts from the saved model without retraining.

---

## 🗂 Project structure

```
Crashscape/
├── app.py                         Streamlit dashboard
├── src/
│   ├── config.py                  paths, sample sizes, column lists, dtypes
│   ├── data_loader.py             chunked CSV → Parquet, stratified sampling, memory report
│   ├── cleaning.py                time/weather/road features, imputation, road type
│   ├── analysis.py                aggregation tables (year-adjusted severity)
│   ├── viz.py                     shared chart style + 16 charts
│   ├── hotspots.py                grid hotspots, Folium & Plotly maps
│   └── factors.py                 GBDT + RF + SHAP, plain-English report
├── notebooks/Crashscape_analysis.ipynb
├── reports/
│   ├── figures/                   22 PNG charts
│   ├── maps/                      hotspot_heatmap.html · la_cluster_map.html · hotspot_density_plotly.html
│   ├── tables/                    analysis tables, top20_hotspots.csv, model metrics & importances
│   ├── screenshots/               map & dashboard screenshots
│   └── factors_explained.md       auto-generated model write-up
├── data/
│   ├── raw/                       (gitignored) dataset zip/CSV
│   └── processed/                 (gitignored) parquet files, except accidents_mini.parquet (4.5 MB)
├── .streamlit/config.toml
├── requirements.txt · LICENSE · README.md
```

---

## 🧰 Tech stack

Python 3.10+ (developed on 3.14) · pandas 3 · NumPy · PyArrow · Matplotlib · Seaborn · Plotly · Folium · scikit-learn · SHAP · Streamlit · Jupyter

---

## 👤 Author

**Vaishnav Kumbhar** · GitHub [@VaishnavK9](https://github.com/VaishnavK9)

Built for the **Prodigy InfoTech Data Science Internship – Task 05**.

## 📄 License

Code: [MIT](LICENSE). Dataset: CC BY-NC-SA 4.0 (Moosavi et al.); see the citation above.
