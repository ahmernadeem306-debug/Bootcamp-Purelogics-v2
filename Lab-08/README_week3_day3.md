# 📊 Week 3, Day 3 — Exploratory Data Analysis (EDA) Assignment

An in-depth Exploratory Data Analysis notebook on the Titanic dataset, covering univariate, bivariate, and multivariate analysis, outlier detection, and a focused mini-challenge on **passenger titles vs. survival**.

## 📌 Overview

This notebook (`Week_3_Day_3_Assignment.ipynb`) walks through a complete EDA workflow — from basic dataset exploration to statistical outlier detection — followed by a guided exercise investigating how a passenger's **title** (Mr, Mrs, Miss, Master, etc.) relates to their chances of survival.

## 📂 Dataset

The Titanic dataset is loaded directly from GitHub:

```
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
```

Target column: `Survived` (0 = did not survive, 1 = survived)

## 🧠 Topics Covered

### 1. Initial Data Exploration
- Dataset shape, structure, and dtypes (`df.info()`)
- Summary statistics (`df.describe()`)
- Missing value counts per column
- Unique value counts for categorical columns

### 2. Univariate Analysis
- Summary statistics table: **mean, median, std dev, skewness, kurtosis** for numeric columns
- Distribution plots (histogram + KDE, boxplot) for `Age` and `Fare`
- Count plots for categorical variables: `Pclass`, `Sex`, `Embarked`, `Survived`

### 3. Bivariate Analysis
- `Age` distribution split by survival status (stacked histogram + boxplot)
- `Fare` distribution split by survival status (stacked histogram + boxplot)
- Survival rate bar charts by:
  - Passenger Class (`Pclass`)
  - Sex
  - Port of Embarkation (`Embarked`)

### 4. Correlation & Multivariate Analysis
- Correlation matrix (heatmap, lower-triangle masked) for numeric features
- Pairplot of `Survived`, `Pclass`, `Age`, `SibSp`, `Parch`, `Fare` colored by survival
- 3-variable scatter plot: **Age vs. Fare vs. Survival**, with point size scaled by `Pclass`

### 5. Outlier Detection
- **Z-score method** (`|z| > 3`) applied to `Fare`, with a scatter plot flagging outliers in red
- **IQR method** (1.5 × IQR rule) applied to `Fare`, with a similar visual comparison

### 6. Feature Engineering for Deeper Insights
- `AgeGroup` — binned into Child / Teen / Young Adult / Adult / Elderly, with survival rate by group
- `FareCategory` — quartile-based binning (Low / Medium / High / Very High), with survival rate by group
- `FamilySize` = `SibSp + Parch + 1`, with survival rate by family size

## 🎯 Exercise: Does Title Affect Survival?

The guided challenge investigates whether a passenger's **title** relates to survival outcomes:

1. **Title extraction** from the `Name` column using a regex (`' ([A-Za-z]+)\.'`)
2. **Title grouping** — rare/noble titles (`Lady`, `Countess`, `Capt`, `Col`, `Don`, `Dr`, `Major`, `Rev`, `Sir`, `Jonkheer`, `Dona`) grouped into `Rare`; `Mlle`/`Ms` → `Miss`; `Mme` → `Mrs`
3. **Visualizations:**
   - Count plot of passengers per title, split by survival
   - Bar plot of survival rate per title
   - Bonus: strip plot of `Age` vs. `Title`, colored by survival
4. **Statistics:** grouped survival counts, totals, and survival rate (%) per title, sorted descending
5. **Bonus multi-factor analysis:** survival rate grouped by both `Pclass` and `Title` together

### 📈 Findings

| Rank | Group | Approx. Survival Rate | Insight |
|------|-------|----------------------|---------|
| 1 | Mrs | ~70%+ | Married women had the highest survival rate |
| 2 | Miss | ~70%+ | Unmarried women/girls also survived at high rates |
| 3 | Master | ~57% | Young boys were prioritized over adult men |
| 4 | Rare | mid-range | Nobility/professional titles — mixed outcomes |
| 5 | Mr | ~16% | Adult men had by far the lowest survival rate |

**Conclusion:** The data strongly reflects the historical **"women and children first"** evacuation protocol — survival was heavily influenced by title (a proxy for sex, age, and social status), with adult men (`Mr`) surviving at the lowest rate despite being the largest passenger group.

## 🛠️ Requirements

```
numpy
pandas
matplotlib
seaborn
scipy
```

Install with:
```bash
pip install numpy pandas matplotlib seaborn scipy
```

## ▶️ How to Run

1. Open the notebook in Jupyter or Google Colab.
2. Run all cells sequentially — plots and statistics build progressively on the cleaned/engineered dataframe.
3. Review the printed statistical summaries and the plots to follow the EDA narrative.
4. The exercise section (title vs. survival) is fully solved — read through the code and commented conclusions, and try extending the bonus multi-factor (`Pclass` × `Title`) analysis further if desired.

## 📊 Key Takeaways

- Skewness/kurtosis stats quickly flag that `Fare` is heavily right-skewed — visualizations confirm this with a long tail of high-fare outliers.
- Z-score and IQR outlier detection can disagree in count — IQR is generally more robust for skewed distributions like `Fare`.
- Simple feature engineering (age/fare binning, family size, and especially **title extraction**) can reveal strong, interpretable survival patterns that raw columns obscure.
- Combining features (e.g., `Pclass` + `Title`) often uncovers richer, more nuanced relationships than looking at any single variable alone.

## 📁 Suggested File Structure

```
├── Week_3_Day_3_Assignment.ipynb   # Main notebook
├── titanic.csv                     # (optional) local copy if not loading from URL
└── README.md                       # This file
```

---
*Part of a data science / ML learning series — Week 3, Day 3.*
