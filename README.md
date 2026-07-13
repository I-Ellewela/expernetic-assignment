# Data Engineer Intern Assignment — Bangkok Airbnb Market Intelligence

**Candidate:** Ishini Ellewela
**Role Applied For:** Data Engineer Intern
**Company:** Expernetic (Pvt) Ltd
**Dataset:** [Inside Airbnb](https://insideairbnb.com/) — Bangkok, Thailand (listings, calendar, reviews, neighbourhoods)

---

## 📄 Start Here

Read the notebooks in order (table below) — each one documents its own methodology, decisions, and business interpretation inline. Together they cover dataset familiarization, data engineering, exploratory data analysis, and price-prediction modelling.

---

## 📁 Repository Structure

```
.
├── README.md                                        ← you are here
├── notebooks/
│   ├── 01_section2_dataset_familiarization_ipynb.ipynb
│   ├── 02_section3_data_engineering_ipynb.ipynb
│   ├── 03_section4_eda.ipynb
│   └── 04_section6_price_prediction_ipynb.ipynb
├── charts/                                          ← exported figures from the notebooks
└── data/                                            ← not included, see "Data Setup" below
```

## 📓 Notebooks — What Each One Does and In What Order to Review Them

| Order | Notebook | Assignment Section | What It Covers |
|---|---|---|---|
| 1 | `01_section2_dataset_familiarization_ipynb.ipynb` | §2 – Dataset Familiarization | Loads all four source files, profiles schema (columns, types, missing %), documents file relationships and dataset limitations. |
| 2 | `02_section3_data_engineering_ipynb.ipynb` | §3.1–3.3 – Data Engineering | Ingestion, profiling, cleaning/standardization (price, dates, booleans), enrichment/joining into `enriched_listings.csv`, data quality checks, and a SQLite analytical layer (5 SQL queries). |
| 3 | `03_section4_eda.ipynb` | §4 – Exploratory Data Analysis | Price distributions, geographic/neighbourhood analysis, review scores & host concentration, availability/occupancy, and temporal trends (2015–2026). |
| 4 | `04_section6_price_prediction_ipynb.ipynb` | §6.1 – Price Prediction | Feature engineering, comparison of Linear Regression / Random Forest / Gradient Boosting, feature importance, and residual analysis. |

## ⚠️ Scope Note

Per the assignment's own design philosophy ("a candidate who completes two sections with exceptional depth... will outperform one who attempts all sections superficially"), this submission prioritizes **depth on Sections 2, 3 (core), 4, and 6.1** over breadth across every optional section.

**Not attempted:** §3.4–3.6 (dimensional modelling, pipeline automation, cloud-native topics), §5 (formal hypothesis testing), §6.2–6.4 (forecasting, clustering, bias testing), §7 (NLP/LLM/agentic work), §8 (Open Innovation Challenge).

Rationale and trade-offs for each are documented inline in the relevant notebooks.

## 🛠️ Setup & Reproducibility

### Requirements
```
Python 3.10+
pandas
numpy
matplotlib
seaborn
scikit-learn
sqlite3 (standard library — no install needed)
```

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Data Setup

Raw data is **not included** in this repo (per Inside Airbnb's terms and to keep the repo lightweight). To reproduce:

1. Download the Bangkok dataset from [insideairbnb.com/get-the-data](https://insideairbnb.com/get-the-data/):
   - `listings.csv.gz`
   - `calendar.csv.gz`
   - `reviews.csv.gz`
   - `neighbourhoods.csv`
2. Place all four files in a local `data/` folder.
3. Update the `data_path` variable at the top of each notebook to point to that folder.

### Execution Order

Run the notebooks in the order listed in the table above — `02_section3_data_engineering_ipynb.ipynb` produces `enriched_listings.csv`, which downstream analysis references.

```bash
jupyter notebook
```
or, if the `jupyter` command isn't on your PATH:
```bash
python -m notebook
```

## 📊 Headline Findings

- **28,204** cleaned listings analyzed (from 31,069 raw)
- Median nightly price: **฿1,549** (mean ฿2,031 — right-skewed)
- Estimated market-wide occupancy: **~1.6%**, suggesting a large share of dormant/"ghost" listings
- Best price-prediction model: **Random Forest**, R² = **0.664**, MAE = **฿551**
- Superhosts (35.6% of listings) command a measurable price premium and materially higher review volume/scores

See each notebook's summary cells for the full breakdown.
