🚢 Titanic Data Preprocessing & Modeling — Week 3, Day 1

A hands-on notebook covering the full preprocessing pipeline for the classic Titanic dataset — from handling missing values to building tuned ML models with scikit-learn.

📌 Overview

This notebook (Week_3_Day_1.ipynb) walks through the essential data preprocessing steps needed before training a machine learning model, using the Titanic survival dataset as a running example. It includes both a guided walkthrough and a practice exercise section where the same concepts are re-implemented.

📂 Dataset

The Titanic dataset is loaded directly from GitHub:

https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

Target column: Survived (0 = did not survive, 1 = survived)

🧠 Topics Covered
1. Missing Value Exploration
Checking null counts per column (isnull().sum())
Visualizing missing values with a heatmap (seaborn)
Plotting percentage of missing values per column
2. Handling Missing Values
Dropping less useful / highly missing columns (Cabin, Ticket)
Group-based median imputation for Age (grouped by Pclass)
Creating a binary "missing indicator" column (Age_Missing)
Mode imputation for the categorical Embarked column
3. Feature Scaling
Standardization using StandardScaler (z-score)
Normalization using MinMaxScaler
Visual comparison of original vs. standardized vs. normalized distributions (Fare column)
⚠️ Scalers are fit only on training data and applied (transform) to test data — avoiding data leakage
4. Encoding Categorical Variables
Label Encoding for the binary Sex column
One-Hot Encoding for the multi-category Embarked column (with drop='first' to avoid the dummy variable trap)
5. Building Preprocessing Pipelines
Using ColumnTransformer to combine numeric and categorical preprocessing
Wrapping preprocessing + model into a single Pipeline
Training and evaluating a LogisticRegression model
Metrics: Accuracy and Classification Report (precision, recall, F1)
6. Hyperparameter Tuning
Swapping in a RandomForestClassifier
Searching over n_estimators, max_depth, and imputer strategy using GridSearchCV (5-fold CV)
Reporting best parameters, best CV score, and test set accuracy
7. Feature Importance
Extracting feature importances from the tuned Random Forest
Visualizing the top features with a horizontal bar chart