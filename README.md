# Delhi Air Quality Analysis & AQI Prediction

## Overview

This project analyzes Delhi's air quality using a multi-year dataset of hourly measurements and builds machine learning models to predict the Air Quality Index (AQI) from pollutant concentrations.

The work covers the full workflow:
- Loading and cleaning real-world environmental data
- Aggregating noisy hourly readings into daily summaries
- Exploratory data analysis (EDA) with visualizations
- Building and evaluating regression models in scikit-learn

The analysis and models are implemented in Python inside a single Jupyter notebook.

---

## Dataset

- **Source:** Air quality dataset for Indian cities from Kaggle (e.g. “Air Quality in India”).
- **Scope used here:** Filtered to **Delhi** only.
- **Raw records (Delhi):** 122,752 hourly/point observations  
- **Aggregated records (Delhi):** 2,192 daily observations  
- **Date range (Delhi):** 2020‑01‑01 to 2025‑12‑31  

Key columns used:

- `datetime`, `date`, `year`, `month`, `day`, `hour`
- `city`, `station`, `latitude`, `longitude`
- Pollutants:  
  - `pm25` (PM2.5 concentration)  
  - `pm10` (PM10 concentration)  
  - `no2` (Nitrogen Dioxide)  
  - `so2` (Sulfur Dioxide)  
  - `co`  (Carbon Monoxide)  
  - `o3`  (Ozone)
- Weather / other:  
  - `temperature`, `humidity`, `wind_speed`, `visibility`
- Target variable:  
  - `aqi` (Air Quality Index)  
  - `aqi_category` (categorical bucket)

> **Note:** The dataset itself is not included in this repository.  
> Please download it directly from Kaggle and place the CSV file (e.g. `city_day.csv`) in the same folder as the notebook.

---

## Project Goals

1. Clean and preprocess Delhi air-quality data with Python and Pandas.
2. Aggregate noisy hourly measurements into daily averages.
3. Explore trends and patterns in pollutants and AQI over time.
4. Understand relationships between pollutants and overall AQI.
5. Build regression models to predict AQI from pollutant levels.
6. Evaluate model performance using standard regression metrics.

---

## Methods and Workflow

All steps are implemented in the notebook `delhi_air_quality.ipynb`:

### 1. Data Loading and Filtering

- Load the full Kaggle CSV using `pandas.read_csv`.
- Convert the `date` column to a proper datetime type.
- Filter the data to `city == "Delhi"`.
- Sort records by date for consistent time-series analysis.

### 2. Cleaning and Aggregation

- Inspect missing values across pollutants and AQI.
- Drop rows with missing values in key columns (`pm25`, `pm10`, `no2`, `so2`, `co`, `o3`, `aqi`).
- Forward-fill less critical missing fields.
- Group by `date` and compute **daily averages** for the main pollutants and AQI, resulting in 2,192 daily records.

### 3. Exploratory Data Analysis (EDA)

Using **Pandas**, **Matplotlib**, and **Seaborn**:

- Time-series plots of daily average **PM2.5** and **AQI** across the full time period.
- Distribution (histogram + KDE) of daily PM2.5 levels.
- Seasonal analysis: boxplots of PM2.5 by `season`.
- Correlation heatmap of daily pollutants and AQI (`pm25`, `pm10`, `no2`, `so2`, `co`, `o3`, `aqi`).

Key observations:
- Strong positive correlations between PM2.5, PM10, NO₂ and overall AQI.
- Clear seasonal variation in PM2.5 and AQI (e.g. higher levels in certain seasons).

### 4. AQI Prediction Models

Using **scikit-learn**:

- Features (X): `[pm25, pm10, no2, so2, co, o3]`
- Target (y): `aqi` (daily average)

Workflow:
- Train–test split (80% training, 20% test).
- Model 1: **Linear Regression**
- Model 2: **Random Forest Regressor**

Evaluation metrics:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

---

## Results

### Linear Regression

- **MAE:** ~77.78 AQI points  
- **RMSE:** ~91.92  
- **R²:** ~0.70  

The linear model captures a substantial portion of the variation in AQI using a simple linear relationship between pollutants and AQI.

### Random Forest Regressor

- **MAE:** ~1.28 AQI points  
- **RMSE:** ~2.06  
- **R²:** ~0.9998  

The Random Forest model achieves extremely high accuracy, with predictions typically within about 1–2 AQI points of the true value and explaining almost all variance in AQI.

This performance is reasonable because AQI is computed from pollutant concentrations via a deterministic formula. A flexible non-linear model like Random Forest can closely approximate that mapping when given enough examples.

---

## Technologies Used

- **Language:** Python
- **Libraries:**
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`
- **Environment:** Jupyter Notebook (run via VS Code)

---

## How to Run the Notebook

1. **Clone or download** this repository:

   ```bash
   git clone https://github.com/Janya-prakash/delhi-air-quality-ml.git
   cd delhi-air-quality-ml
