# 🚀 Benchmark of Tree-Based Machine Learning Models

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-red.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-3.3%2B-green.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

A comprehensive benchmark tool designed to evaluate and compare 6 popular ensemble and gradient boosting algorithms on multi-class classification tasks. 

This repository provides automated model evaluation, full classification metrics (Accuracy, Precision, Recall, F1-Score), and publication-ready visual evaluation graphs.

---

## 📌 Models Included

1. **Random Forest Classifier** *(Bagging)*
2. **AdaBoost Classifier** *(Boosting)*
3. **Gradient Boosting Classifier** *(Boosting)*
4. **Histogram-based Gradient Boosting Classifier** *(Scikit-Learn)*
5. **XGBoost Classifier** *(eXtreme Gradient Boosting)*
6. **LightGBM Classifier** *(Light Gradient Boosting Machine)*

---

## 📊 Visual Features & Key Outputs

The benchmark automatically generates and exports three publication-quality charts upon execution:

| Feature | Description | Output File |
| :--- | :--- | :--- |
| **Confusion Matrix Heatmaps** | Side-by-side confusion matrices for all 6 models to evaluate class-level misclassifications. | `confusion_matrices.png` |
| **Model Comparison Bar Plot** | Benchmark chart comparing Accuracy, Macro F1, and Micro F1 scores across models. | `model_comparison_bar_plot.png` |
| **Feature Importance Plot** | Permutation Importance scores for the top-performing model. | `feature_importance.png` |

---
