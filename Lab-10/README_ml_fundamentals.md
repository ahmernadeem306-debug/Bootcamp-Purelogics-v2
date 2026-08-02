# 🤖 Machine Learning Fundamentals — Exercise Collection

A collection of 5 hands-on notebooks/scripts covering core supervised learning algorithms and model evaluation techniques: **Linear Regression** (from scratch and with scikit-learn), **Decision Trees**, **Support Vector Machines**, and **K-Fold Cross-Validation**.

## 📌 Overview

This collection is a set of independent lab exercises, each focused on a specific ML algorithm or evaluation technique. They're meant to be worked through individually, though together they form a solid tour of classic supervised learning workflows in `scikit-learn`.

| # | File | Topic | Dataset(s) |
|---|------|-------|-----------|
| 1 | `Linear_Regression_Exercise.ipynb` | Linear regression **from scratch** (cost function, gradient descent) | Restaurant profit vs. city population |
| 2 | `Linear_Regression.ipynb` | Linear regression **with scikit-learn** (multi-feature + one-hot encoding) | Car prices, hiring/salary data |
| 3 | `Copy_of_Decision_Tree_Classifier.ipynb` | Decision Tree Classifier | Titanic survival |
| 4 | `Copy_of_Support_Vector_Machine.ipynb` | Support Vector Classifier (SVC) | `sklearn` digits dataset |
| 5 | `Copy_of_K_Fold(Iris).ipynb` | K-Fold Cross-Validation across 3 models | Iris dataset |

## 🧠 What Each Notebook Covers

### 1. Linear Regression Exercise (from scratch)
A guided implementation of univariate linear regression **without** scikit-learn, to build intuition for the underlying math:
- Loading and visualizing city population vs. restaurant profit data (scatter plot)
- **`compute_cost(x, y, w, b)`** — implements the mean squared error cost function $J(w,b) = \frac{1}{2m}\sum (f_{w,b}(x^{(i)}) - y^{(i)})^2$
- **`compute_gradient(x, y, w, b)`** — computes the partial derivatives $\partial J/\partial w$ and $\partial J/\partial b$
- **`gradient_descent(...)`** — runs batch gradient descent for a set number of iterations, tracking cost history
- Plotting the final fitted line over the data
- Making profit predictions for cities of a given population (e.g., 35,000 and 70,000 people)
- Includes expected output tables to self-check correctness at each step

### 2. Linear Regression (scikit-learn)
Practical multi-feature linear regression using `sklearn.linear_model.LinearRegression`:
- **Car price prediction** from `cylinders`, `mileage`, and `year`
- Handling missing values (mean/floor imputation for `year`)
- **Hiring/salary prediction** from `experience`, `test_score`, `interview_score`
- Saving and reloading a trained model with **`pickle`**
- **One-hot encoding** of a categorical `Car_Model` column using `pd.get_dummies`
- Predicting car sell price for specific mileage/age/model combinations (e.g., a Mercedes C-Class vs. an Audi A5)

### 3. Decision Tree Classifier
Predicting Titanic survival using `sklearn.tree.DecisionTreeClassifier`:
- Dropping irrelevant columns (`PassengerId`, `Name`, `Ticket`, `Cabin`, etc.)
- Imputing missing `Age` values with the (floored) mean
- Label-encoding `Sex` (and `Embarked` in the extended version)
- Train/test split and model training/scoring
- **Exercise extension:** retrains the tree using *more* feature columns (`SibSp`, `Parch`, `Embarked`) plus tuned hyperparameters (`criterion='entropy'`, `max_depth=5`) to compare accuracy against the simpler baseline model

### 4. Support Vector Machine (SVC)
Digit classification with `sklearn.svm.SVC` on the classic 8×8 digits dataset:
- Exploring and converting the `digits` dataset into a `pandas` DataFrame
- Train/test split, model training, and scoring
- Visualizing predicted digit images with `plt.matshow`
- **Exercise extension:** loops over 8 different `test_size` values (0.1 → 0.8), retrains an SVC each time, and plots how test/train split ratio affects accuracy — a practical demonstration of the bias-variance/data-availability tradeoff

### 5. K-Fold Cross-Validation (Iris)
Comparing three classifiers on the Iris dataset using `cross_val_score` with 5-fold CV:
- `RandomForestClassifier` (`n_estimators=35`)
- `SVC`
- `LogisticRegression` (`max_iter=150`)
- Includes a written explainer on what cross-validation is, how K-Fold splitting works, and how to interpret fold-wise scores (mean + variability)
- Concludes by comparing the average scores across all three models

## 🛠️ Requirements

```
numpy
pandas
matplotlib
scikit-learn
```

Install with:
```bash
pip install numpy pandas matplotlib scikit-learn
```

> ⚠️ Notebook #1 (`Linear_Regression_Exercise`) additionally requires a local **`utils.py`** helper file (providing `load_data()` and test functions like `compute_cost_test`, `compute_gradient_test`) — this is a course-provided helper, not a public package, so make sure it's in the same directory before running.

## 📂 Required Data Files

Several notebooks expect CSV files in the working directory (not downloaded automatically):

| File | Used by |
|------|---------|
| `titanic.csv` | Decision Tree Classifier |
| `Cars_price.csv` | Linear Regression (scikit-learn) |
| `hiring.csv` | Linear Regression (scikit-learn) |
| `carprices.csv` | Linear Regression (scikit-learn) |

The SVM and K-Fold notebooks use built-in `sklearn.datasets` (`load_digits`, `load_iris`) and need no external files.

## ▶️ How to Run

1. Open each notebook individually in Jupyter or Google Colab.
2. Make sure any required CSV/helper files (see table above) are uploaded to the same directory/session.
3. Run all cells top to bottom.
4. For the "exercise" sections at the end of each notebook, review the completed code and compare the printed accuracy/results against the baseline model earlier in the same notebook.

## 📊 Key Takeaways

- **Linear Regression:** Implementing gradient descent manually builds intuition for *why* scikit-learn's `.fit()` works the way it does — cost should monotonically decrease as parameters converge.
- **Categorical data** (car models, `Sex`, `Embarked`) must be encoded (via `LabelEncoder` or `pd.get_dummies`) before being used in any of these models — none of them accept raw text/category columns directly.
- **Decision Trees:** Adding more relevant features and tuning `max_depth`/`criterion` can meaningfully change accuracy — but more features isn't always better without proper preprocessing.
- **SVM:** Test/train split ratio directly affects measured accuracy — too little training data underfits, too little test data gives a noisy accuracy estimate.
- **Cross-validation** gives a far more reliable performance estimate than a single train/test split, since it averages performance across multiple different data partitions and reveals variance across folds.

## 📁 Suggested File Structure

```
├── Linear_Regression_Exercise.ipynb
├── utils.py                          # required helper for Exercise 1
├── Linear_Regression.ipynb
├── Cars_price.csv
├── hiring.csv
├── carprices.csv
├── Copy_of_Decision_Tree_Classifier.ipynb
├── titanic.csv
├── Copy_of_Support_Vector_Machine.ipynb
├── Copy_of_K_Fold(Iris).ipynb
└── README.md                         # This file
```

---
*Part of a data science / ML learning series — Supervised Learning Fundamentals.*
