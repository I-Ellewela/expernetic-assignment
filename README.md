# Expernetic Data Engineer Intern — Technical Assessment

**Candidate:** Ishini Ellewela  
**Dataset:** Inside Airbnb — Bangkok, Thailand  
**Submission Date:** 13 July 2026  
**Contact:** ishiniellewela@gmail.com

---

## Overview

This repository contains my complete submission for the Expernetic Data Engineer Intern technical assessment. 

I performed **end-to-end data engineering and analytics** on the Inside Airbnb Bangkok dataset (31,069 listings), covering data ingestion, cleaning, enrichment, exploratory analysis, SQL analytics, and machine learning (price prediction).

**Completed Sections:**
- ✅ Section 02 — Dataset Familiarization
- ✅ Section 03 — Data Engineering Challenges
- ✅ Section 04 — Exploratory Data Analysis
- ✅ Section 06 — Data Science Challenges (Price Prediction)

---

## Key Findings

- **Median nightly price**: ฿1,549 (right-skewed distribution)
- **Most active neighbourhood**: Vadhana (2,971 listings)
- **Most expensive neighbourhood**: Parthum Wan (median ฿2,547)
- **Superhost premium**: Superhosts charge ~฿192 more per night and have higher ratings (4.86 vs 4.55)
- **Post-COVID recovery**: 2025 review volume is nearly 3× higher than pre-COVID 2019 levels
- **Best price predictor**: Number of bedrooms (strongest feature in Random Forest model)

---

## Repository Structure
Expernetic_Assignment/
├── notebooks/
│   ├── 01_section2_dataset_familiarization.ipynb
│   ├── 02_section3_data_engineering.ipynb
│   ├── 03_section4_eda.ipynb
│   └── 04_section6_price_prediction.ipynb
├── charts/                    # All generated visualizations
├── data/                      # Raw & processed datasets
├── reports/
│   └── final_report.pdf       # Full written report
├── pipeline.py                # Main data processing pipeline (optional)
├── requirements.txt
└── README.md
text---

## How to Reproduce

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter

Download the latest Bangkok dataset from:
https://insideairbnb.com/get-the-data/
Place the files (listings.csv.gz, reviews.csv.gz, calendar.csv.gz, neighbourhoods.csv) in the data/ folder.
Open the notebooks in this order and run them:
01_section2_dataset_familiarization.ipynb
02_section3_data_engineering.ipynb
03_section4_eda.ipynb
04_section6_price_prediction.ipynb



Tech Stack

Language: Python 3
Data Processing: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Machine Learning: Scikit-learn (Linear Regression, Random Forest, Gradient Boosting)
Database: SQLite (in-memory)
Environment: Jupyter Notebook


AI Tools Disclosure
In accordance with Section 10 of the assignment, I used Claude (Anthropic) as an AI coding assistant for:

Suggesting notebook structure and code organization
Debugging errors and suggesting improvements
Refining business interpretations

All code was manually reviewed, executed, tested, and validated by me. All findings, insights, and final outputs are my own work.

Future Improvements (if more time)

Add geospatial visualization using Folium
Implement SHAP explainability for the price model
Containerize the pipeline with Docker
Add seasonal features and dynamic pricing simulation


Thank you for reviewing my submission.
I am available to start from 3rd August 2026 and am flexible for onsite work in Colombo.
<<<<<<< HEAD
Looking forward to discussing this project further!
=======
Looking forward to discussing this project further!
>>>>>>> f4cd7aed5b72fff71deab39d099f486263f37ea2
