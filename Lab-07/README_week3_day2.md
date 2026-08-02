# 🧪 Week 3, Day 2 — Feature Engineering & Data Augmentation Assignment

A comprehensive notebook covering **advanced feature engineering** on tabular data (Titanic) and **data augmentation techniques** for images and text — including their effect on model performance.

## 📌 Overview

This notebook (`Week_3_Day_2_Assignment_task.ipynb`) is split into two major parts:

1. **Tabular Feature Engineering & Selection** — using the Titanic dataset
2. **Data Augmentation** — for images (MNIST) and text, including how augmentation affects a CNN's performance on imbalanced classes

## 📂 Datasets Used

- **Titanic dataset** (tabular): `https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`
- **MNIST dataset** (image): loaded via `tensorflow.keras.datasets.mnist`
- **Sample sentences** (text): small hard-coded list for augmentation demos

## 🧠 Topics Covered

### Part 1: Feature Engineering for Tabular Data

- Basic exploration and missing value handling (`Age` → median, `Embarked` → mode, `Cabin` → `'Unknown'`)
- **Engineered features:**
  - `Title` extracted from passenger `Name`, with rare titles grouped together
  - `FamilySize` = `SibSp + Parch + 1`
  - `IsAlone` binary flag
  - `FarePerPerson` = `Fare / FamilySize`
  - `AgeBin` — binned into Child / Teenager / Adult / Elderly
  - `FareBin` — binned into Low / Medium / High / Very High (quartiles)
  - `Deck` extracted from the first letter of `Cabin`
  - `Family_Survival` interaction feature
- One-hot encoding of categorical features with `pd.get_dummies`
- Feature scaling comparison: `StandardScaler`, `MinMaxScaler`, `RobustScaler`

### Part 2: Feature Selection & Modeling

- `SelectKBest` with `f_classif` to select the top 15 features
- Train/test split (75/25) and a `RandomForestClassifier`
- Evaluation via accuracy and classification report

### Part 3: Advanced Feature Engineering

- **Polynomial Features** (`PolynomialFeatures`, degree 2) on `Age` and `Fare`
- **PCA** for dimensionality reduction (10 components), with explained variance ratio bar chart and cumulative variance

### Part 4: Image Data Augmentation (MNIST)

Four different augmentation approaches are implemented and visually compared:

1. **Keras `ImageDataGenerator`** — rotation, shifts, zoom, shear, horizontal flip
2. **`tf.image`** — random rotation (90° multiples), brightness, and contrast
3. **OpenCV + `scipy.ndimage`** — random choice of rotate / Gaussian blur / zoom
4. **Albumentations** — `RandomBrightnessContrast`, `RandomGamma`, `ElasticTransform`, `GridDistortion`, `OpticalDistortion`

### Part 5: Text Data Augmentation

Simple, dependency-free text augmentation functions:
- `swap_words(text)` — swaps two random words
- `delete_random_word(text)` — removes one random word
- `insert_random_word(text)` — inserts a random word from a small vocabulary

### Part 6: Class Balancing via Augmentation

- Computing class distribution of MNIST digits
- Creating an artificially imbalanced subset (e.g., very few samples of digit `0` or `8`)
- `augment_minority_class()` — uses `ImageDataGenerator.flow` to generate synthetic samples until the minority class reaches a target count

### Part 7: Effect of Augmentation on Model Performance

- A simple CNN (`Conv2D` → `MaxPooling2D` → `Conv2D` → `MaxPooling2D` → `Dense`) is defined via `create_model()`
- The model is trained **twice**:
  - Once on the **imbalanced** dataset
  - Once on the **balanced (augmented)** dataset
- Training/validation **accuracy and loss curves** are plotted side-by-side for comparison
- Final **test accuracy** is reported for both versions, showing the impact of augmentation on handling class imbalance

## 🛠️ Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
opencv-python
scipy
albumentations
```

Install with:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python scipy albumentations
```

> ⚠️ Note: `tensorflow`, `opencv-python`, and `albumentations` are heavier dependencies — running this notebook may require a GPU-enabled runtime (e.g., Google Colab) for reasonable training speed on the CNN sections.

## ▶️ How to Run

1. Open the notebook in Google Colab (recommended, given the TensorFlow/image dependencies) or Jupyter.
2. Run all cells top to bottom — later sections (CNN training) depend on data loaded/prepared earlier.
3. The `# TODO` sections have already been completed with working code — review the outputs (plots, printed reports, accuracy comparisons) as you go.

## 📊 Key Takeaways

- Thoughtful feature engineering (titles, family size, binning, interaction terms) can surface signal that raw columns don't capture directly.
- `SelectKBest` + PCA offer two different ways to manage high-dimensional feature spaces — one keeps original features (interpretable), the other creates new uncorrelated components (compresses variance).
- Different scalers (`Standard`, `MinMax`, `Robust`) behave differently depending on outliers and feature distribution — `RobustScaler` is more resistant to outliers like extreme `Fare` values.
- Image augmentation can be done at multiple levels of sophistication — from simple Keras generators to fine-grained pixel-level libraries like Albumentations.
- Augmenting a minority class is an effective way to combat class imbalance, and the final CNN comparison quantifies the real accuracy/loss benefit of doing so.
- Even lightweight text augmentation (word swap/delete/insert) can be a quick way to expand small NLP datasets.

## 📁 Suggested File Structure

```
├── Week_3_Day_2_Assignment.ipynb   # Main notebook
├── titanic.csv                     # (optional) local copy if not loading from URL
└── README.md                       # This file
```

---
*Part of a data science / ML learning series — Week 3, Day 2.*
