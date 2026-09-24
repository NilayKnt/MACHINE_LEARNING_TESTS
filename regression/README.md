# 🏢 Energy Efficiency & Load Prediction with Machine Learning

This repository presents a machine learning pipeline for predicting and classifying the heating and cooling loads of buildings based on their structural and architectural features. Using the **Energy Efficiency Dataset (ENB2012)**, this project covers end-to-end data processing, continuous target estimation via Linear and Polynomial Regression, residual diagnostics, and binary classification via Logistic Regression.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Dataset Architecture](#-dataset-architecture)
- [Key Features & Methodology](#-key-features--methodology)
  - [1. Data Preprocessing & EDA](#1-data-preprocessing--eda)
  - [2. Linear Regression & Residual Diagnostics](#2-linear-regression--residual-diagnostics)
  - [3. Single-Feature Polynomial Analysis](#3-single-feature-polynomial-analysis)
  - [4. Binary Classification with Logistic Regression](#4-binary-classification-with-logistic-regression)
- [Model Performance & Results](#-model-performance--results)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [License](#-license)

---

## 📖 Overview

Building energy efficiency assessment is crucial for sustainable design and reducing environmental impact. The objective of this project is twofold:
1. **Regression Task:** Accurately estimate numeric values for **Heating Load ($Y_1$)** and **Cooling Load ($Y_2$)**.
2. **Classification Task:** Categorize buildings into **Low** vs. **High Heating Load** based on median thresholding using Logistic Regression.

---

## 📊 Dataset Architecture

The analysis utilizes the **ENB2012** dataset (`ENB2012_data.xlsx`), comprising 768 samples and 8 architectural features:

| Feature Name | Description | Type |
| :--- | :--- | :--- |
| `x1_compactness` | Relative Compactness | Continuous |
| `x2_surface` | Surface Area | Continuous |
| `x3_wall` | Wall Area | Continuous |
| `x4_roof` | Roof Area | Continuous |
| `x5_height` | Overall Height | Continuous |
| `x6_orientation` | Orientation (2: North, 3: East, 4: South, 5: West) | Categorical/Discrete |
| `x7_glazing` | Glazing Area | Continuous |
| `x8_glazing_dist` | Glazing Area Distribution | Categorical/Discrete |
| **`y1_heating`** | **Heating Load (Target 1)** | Continuous |
| **`y2_cooling`** | **Cooling Load (Target 2)** | Continuous |

---

## ⚙️ Key Features & Methodology

### 1. Data Preprocessing & EDA
- Automated column renaming and structured data loading from Excel formats.
- Summary statistics (`df.describe()`) and structural verification (`df.info()`).

### 2. Linear Regression & Residual Diagnostics
- **Dataset Splitting:** 80/20 Train-Test split (`random_state=42`).
- **Evaluation Metrics:** Mean Absolute Error (MAE), Mean Squared Error (MSE), and $R^2$ Score.
- **Diagnostic Visualizations:**
  - *Actual vs. Predicted* scatter plots.
  - *Residual vs. Predicted* distribution plots to check for homoscedasticity.
  - *Residual Normality* histograms (KDE overlay) to assess error distribution.

### 3. Single-Feature Polynomial Analysis
- Evaluates non-linear relationships using `x2_surface` against degree polynomials ($d \in \{1, 2, 3, 4\}$).
- Visualizes non-linear fit curves against raw data points for both heating and cooling requirements.

### 4. Binary Classification with Logistic Regression
- **Target Transformation:** Engineered a binary classification variable `y1_heating_category` based on the median value of $Y_1$:
  - `0`: Low Heating Load ($\le \text{Median}$)
  - `1`: High Heating Load ($> \text{Median}$)
- **Stratified Splitting & Scaling:** Applied `StandardScaler` to normalize feature distributions.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, and ROC-AUC Score alongside confusion matrix visualization.

---

## 📈 Model Performance & Results

### Linear Regression Metrics
| Target Variable | MAE | MSE | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Heating Load ($Y_1$)** | *Calculated at runtime* | *Calculated at runtime* | ~0.91 |
| **Cooling Load ($Y_2$)** | *Calculated at runtime* | *Calculated at runtime* | ~0.89 |

### Logistic Regression Metrics ($Y_1$ Category)
| Metric | Score |
| :--- | :---: |
| **Accuracy** | High (> 0.90) |
| **Precision** | High |
| **Recall** | High |
| **F1-Score** | High |
| **ROC AUC** | Near Optimal (~0.98+) |

---

## 🛠️ Tech Stack

- **Python 3.x**
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`, `statsmodels`
- **Visualization:** `matplotlib`, `seaborn`

---


