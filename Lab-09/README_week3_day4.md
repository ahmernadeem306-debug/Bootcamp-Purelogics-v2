# 🎯 Week 3, Day 4 — Outlier Detection & Treatment

A deep-dive notebook on detecting, visualizing, and handling outliers using univariate and multivariate statistical methods, followed by a hands-on exercise applying these techniques to a real-world housing dataset.

## 📌 Overview

This notebook (`Week_3_Day_4.ipynb`) explores outlier detection from first principles using a synthetic dataset, then extends into advanced multivariate methods (Mahalanobis distance, Isolation Forest, Local Outlier Factor) and measures the real impact of outliers on correlation, scaling, and regression models. A guided exercise section applies the same toolkit to the **California Housing dataset**.

## 📂 Datasets Used

- **Synthetic dataset** — `height`, `weight`, and `salary` columns generated with `numpy.random`, with deliberately injected outliers (e.g., `salary` values up to 1,000,000)
- **California Housing dataset** — loaded via `sklearn.datasets.fetch_california_housing`

## 🧠 Topics Covered

### 1. Dataset Setup & Initial Exploration
- Synthetic data generation with realistic distributions (`normal`, `lognormal`) plus intentional outliers
- Summary statistics (`df.describe()`)
- Distribution visualization: histograms + KDE, and boxplots for `height`, `weight`, `salary`
- Scatter plots between feature pairs to visually spot patterns/outliers

### 2. Univariate Outlier Detection
- **Z-score method** — `detect_outliers_zscore()`, flags points with `|z| > 3`
- **IQR method** — `detect_outliers_iqr()`, flags points outside `Q1 - 1.5×IQR` / `Q3 + 1.5×IQR`
- Side-by-side boxplot comparison highlighting outliers found by each method

### 3. Impact of Outliers on Statistics
- Comparing mean/std **with vs. without** outliers (outliers removed via IQR)
- Demonstrates how much outliers can distort summary statistics, especially for `salary`

### 4. Outlier Treatment Techniques
- **Removal** — dropping rows flagged by IQR
- **Winsorization (capping)** — `winsorize()` clips extreme values to the 5th/95th percentile
- **Log transformation** — `np.log()` applied to `salary` to reduce right-skew
- Boxplot comparison: original vs. winsorized vs. log-transformed

### 5. Correlation Analysis
- Pearson correlation heatmap **with** outliers vs. **without** outliers
- **Spearman (rank-based) correlation** as a more outlier-robust alternative

### 6. Feature Profiling
- `profile_feature()` — builds a full profile per feature: count, mean, median, std, min/max, range, IQR, skewness, kurtosis, missing values, and outlier counts (both Z-score and IQR) as percentages

### 7. Multivariate Outlier Detection (Mahalanobis Distance)
- `detect_multivariate_outliers()` — manual implementation of Mahalanobis distance using the covariance matrix and a chi-squared threshold
- Visualizations: pairplot of all features, 2D scatter (height vs. weight) with multivariate outliers highlighted, and a **3D scatter plot** (height, weight, salary) marking outliers

### 8. Scaling Methods vs. Outliers
- `StandardScaler` vs. `RobustScaler` — comparing how each handles outlier-heavy data like `salary`

### 9. Outliers' Impact on Linear Regression
- A simple synthetic linear dataset with one injected outlier
- Regression fit **with** vs. **without** the outlier — shows how much a single extreme point can shift the slope/intercept

## 🎯 Interactive Exercise: Advanced Outlier Detection on Real Data

Using the **California Housing** dataset (`MedInc`, `HouseAge`, `AveRooms`, `MedHouseVal`, etc.):

- **Exercise 1 — EDA for Outliers:** histograms + boxplots for `MedInc`, `HouseAge`, `AveRooms`
- **Exercise 2 — Z-score & IQR on `MedHouseVal`:** side-by-side scatter plots comparing outliers flagged by each method
- **Exercise 3 — Isolation Forest:** unsupervised outlier detection (`contamination=0.05`) on `MedInc` vs. `MedHouseVal`
- **Exercise 4 — Impact on Linear Regression:** comparing `LinearRegression` (`MedInc` → `MedHouseVal`) R² scores across three cleaned datasets — all data, Z-score cleaned, and Isolation Forest cleaned
- **Exercise 5 — Custom Consensus Workflow:** `detect_consensus_outliers()` combines IQR **and** Z-score, returning only outliers flagged by **both** methods, with a boxplot visualization
- **Exercise 6 — Robust Feature Engineering:** log transformation (`np.log1p`) applied to the skewed `AveRooms` feature, with before/after distribution comparison
- **Exercise 7 — Bonus: Multivariate Outlier Detection:** Mahalanobis distance + chi-squared threshold on `MedInc` and `MedHouseVal` jointly, visualized via a labeled pairplot ("Normal" vs. "Outlier")

## 🛠️ Requirements

```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
```

Install with:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

## ▶️ How to Run

1. Open the notebook in Jupyter or Google Colab.
2. Run all cells sequentially — the synthetic dataset section runs first, followed by the California Housing exercise section (which reloads `df` from `fetch_california_housing`, overwriting the synthetic one).
3. Review printed outlier indices/counts and compare the various visualizations to build intuition for how each method behaves.

## 📊 Key Takeaways

- Z-score and IQR frequently disagree on *which* points are outliers, especially for skewed distributions like `salary` — IQR tends to be more robust in those cases.
- Outliers can substantially distort means, standard deviations, Pearson correlations, and regression coefficients — a single extreme point can visibly tilt a regression line.
- Treatment method matters: **removal** loses data, **winsorization** caps extremes without discarding rows, and **log transformation** is especially effective for right-skewed monetary/count data.
- Multivariate methods (Mahalanobis distance, Isolation Forest) catch outliers that look "normal" on any single variable but are unusual in combination — something univariate Z-score/IQR checks miss entirely.
- `RobustScaler` (median/IQR-based) is far less distorted by outliers than `StandardScaler` (mean/std-based).
- Combining multiple detection methods (consensus approach) can reduce false positives compared to relying on just one technique.

## 📁 Suggested File Structure

```
├── Week_3_Day_4.ipynb   # Main notebook
└── README.md            # This file
```

---
*Part of a data science / ML learning series — Week 3, Day 4.*
