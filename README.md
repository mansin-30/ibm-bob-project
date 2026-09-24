# 🏠 House Price Prediction Using Data Analytics and Machine Learning

**B.Tech CSE College Project | Academic Year 2024–25**

---
## project dataset link
https://www.kaggle.com/datasets/prokshitha/home-value-insights?resource=download

## 📌 Project Overview

This project builds an end-to-end machine learning pipeline to predict house prices from structured real estate data. It covers complete data analysis, exploratory data analysis (EDA), model training, evaluation, comparison, and final prediction — implemented in a clean, beginner-friendly Jupyter Notebook using Python.

---

## 🎯 Problem Statement

Given a set of house attributes — square footage, number of bedrooms, number of bathrooms, year built, lot size, garage size, and neighbourhood quality — predict the **sale price of a house** as accurately as possible.

This is a **supervised regression** problem. The target variable is the continuous numeric value `House_Price`.

---

## 📋 Objectives

- Inspect and understand the dataset thoroughly
- Perform data cleaning and validate data quality
- Conduct Exploratory Data Analysis (EDA) to uncover patterns
- Train and compare multiple regression models
- Evaluate models using MAE, RMSE, and R² Score
- Identify the best model and predict house prices

---

## 📁 Dataset Information

| Property | Value |
|---|---|
| **File** | `house_price_regression_dataset.csv` |
| **Rows** | 1,000 |
| **Columns** | 8 (7 features + 1 target) |
| **Missing Values** | None |
| **Duplicates** | None |
| **Dataset Source** | Provided locally (no external URL) |

### Feature Descriptions

| Feature | Type | Range | Description |
|---|---|---|---|
| `Square_Footage` | int | 503 – 4999 | Total living area in sq ft |
| `Num_Bedrooms` | int | 1 – 5 | Number of bedrooms |
| `Num_Bathrooms` | int | 1 – 3 | Number of bathrooms |
| `Year_Built` | int | 1950 – 2022 | Year the house was built |
| `Lot_Size` | float | 0.51 – 4.99 acres | Property lot size |
| `Garage_Size` | int | 0, 1, or 2 | Garage capacity (cars) |
| `Neighborhood_Quality` | int | 1 – 10 | Neighbourhood quality rating |
| `House_Price` ⭐ | float | $111,627 – $1,108,237 | **Target — sale price (USD)** |

---

## 🛠️ Technologies Used

| Tool / Library | Purpose |
|---|---|
| Python 3 | Programming language |
| pandas | Data loading and manipulation |
| numpy | Numerical computations |
| matplotlib | Data visualisation |
| seaborn | Statistical visualisation |
| scikit-learn | Machine learning models and metrics |
| Jupyter Notebook | Interactive development environment |

---

## 🔄 Project Workflow

```
1. Load Dataset
       ↓
2. Data Inspection (shape, dtypes, missing values, duplicates, statistics)
       ↓
3. Data Cleaning (verified clean — no action required)
       ↓
4. Exploratory Data Analysis (distributions, correlations, scatter plots, bar charts)
       ↓
5. Feature & Target Preparation (X = features, y = House_Price)
       ↓
6. Train-Test Split (80% train / 20% test, random_state=42)
       ↓
7. Model Training (4 models)
       ↓
8. Model Evaluation (MAE, RMSE, R²)
       ↓
9. Model Comparison & Best Model Selection
       ↓
10. Final Prediction using Best Model
```

---

## 🤖 Models Used

| Model | Type |
|---|---|
| Linear Regression | Parametric linear model |
| Decision Tree Regressor | Non-parametric tree model |
| Random Forest Regressor | Ensemble of 100 trees (bagging) |
| Gradient Boosting Regressor | Sequential ensemble (boosting) |

---

## 📊 Results

| Model | MAE ($) | RMSE ($) | R² Score |
|---|---|---|---|
| **Linear Regression** ✅ | **8,174.58** | **10,071.48** | **0.9984** |
| Gradient Boosting | 12,305.22 | 14,998.85 | 0.9965 |
| Random Forest | 16,114.29 | 19,852.75 | 0.9939 |
| Decision Tree | 24,084.46 | 31,138.40 | 0.9850 |

**🏆 Best Model: Linear Regression** — R² Score of **0.9984** (explains 99.84% of variance in house prices)

### Feature Importance (Random Forest)

| Feature | Importance |
|---|---|
| Square_Footage | 98.61% |
| Year_Built | 0.54% |
| Lot_Size | 0.51% |
| Num_Bedrooms | 0.17% |
| Neighborhood_Quality | 0.07% |
| Num_Bathrooms | 0.05% |
| Garage_Size | 0.04% |

---

## ⚙️ Installation Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1 — Clone or Download the Project

Place all project files in a single folder.

### Step 2 — Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 3 — Launch Jupyter Notebook

```bash
jupyter notebook
```

---

## ▶️ How to Run the Notebook

1. Ensure `house_price_regression_dataset.csv` is in the **same folder** as `House_Price_Prediction.ipynb`.
2. Open a terminal in the project folder.
3. Run: `jupyter notebook`
4. In the browser, open `House_Price_Prediction.ipynb`.
5. Click **Kernel → Restart & Run All** to execute all cells.

---

## 📂 Project Structure

```
house_price_prediction/
│
├── house_price_regression_dataset.csv          # Dataset
├── House_Price_Prediction.ipynb                # Jupyter Notebook (main project file)
├── requirements.txt                            # Python dependencies
├── README.md                                   # This file
├── House_Price_Prediction_Project_Report.docx  # Word project report
│
├── plot_house_price_distribution.png           # EDA plot
├── plot_sqft_vs_price.png                      # EDA plot
├── plot_correlation_heatmap.png                # EDA plot
├── plot_price_by_bed_bath.png                  # EDA plot
├── plot_price_by_garage_neighborhood.png       # EDA plot
├── plot_year_vs_price.png                      # EDA plot
├── plot_feature_distributions.png              # EDA plot
├── plot_model_comparison.png                   # Model comparison plot
├── plot_actual_vs_predicted.png                # Best model plot
├── plot_residuals.png                          # Best model plot
└── plot_feature_importance.png                 # Feature importance plot
```

---

## 🔍 Key Findings

1. **Square_Footage** is the overwhelmingly dominant predictor — Pearson correlation of **0.991** with House_Price and **98.61%** feature importance.
2. The dataset is **perfectly clean** — no missing values, no duplicates, all values within plausible ranges.
3. House prices are **approximately normally distributed** ($112K – $1.1M, mean $619K).
4. **Linear Regression outperforms** all ensemble models because the data has a near-perfect linear structure.
5. All four models achieve **R² > 0.98**, confirming the features are highly predictive.
6. Neighbourhood_Quality, Bathrooms, and Garage_Size contribute negligibly to individual price predictions.

---

## ⚠️ Limitations

- No geographic/location data (city, zip code), which is critical in real-world house pricing.
- Only 1,000 records — a larger, more diverse dataset would improve generalisation.
- The near-perfect dominance of Square_Footage suggests the dataset may be synthetic.
- External economic factors (interest rates, market trends) are not captured.
- No hyperparameter tuning was applied to tree-based models.

---

## 🚀 Future Scope

- Include real-world datasets with location data (e.g., city, zip code, school ratings).
- Apply hyperparameter tuning (GridSearchCV / RandomizedSearchCV).
- Experiment with advanced models: XGBoost, LightGBM, deep learning.
- Deploy the best model as a web application using **Flask** or **Streamlit**.
- Incorporate time-series data to predict price trends over time.

---

## 📜 License

This project is created for educational purposes .

