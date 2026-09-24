# Heart Disease Classification with Keras & PyTorch

This repository contains deep learning implementations for binary classification of heart disease based on patient medical records (`heart.csv`). The project demonstrates and compares Artificial Neural Network (ANN) models built using both **TensorFlow / Keras** and **PyTorch** frameworks on the same dataset.

---

## 📌 Project Overview

- **Objective:** Predict the presence of heart disease (1 or 0) in a patient using clinical features.
- **Key Features:**
  - Data preprocessing (One-Hot Encoding, Feature Scaling)
  - Sequential Artificial Neural Network (ANN) using TensorFlow / Keras
  - Object-oriented Neural Network architecture with custom training loops using PyTorch
  - Regularization techniques including Dropout and Early Stopping

---

## 🛠️ Tech Stack & Requirements

- **Language:** Python 3.x
- **Data Analysis:** `pandas`, `numpy`
- **Preprocessing:** `scikit-learn` (`StandardScaler`, `train_test_split`)
- **Deep Learning Frameworks:**
  - `tensorflow.keras`
  - `torch` (PyTorch)

---

## 📂 Dataset & Preprocessing Pipeline

1. **Data Loading:** Imports the `heart.csv` dataset.
2. **Categorical Encoding:** Categorical features (`cp`, `restecg`, `slope`, `thal`, `ca`) are transformed into dummy variables using `pd.get_dummies()`.
3. **Dataset Splitting:**
   - Split into **80% Training** and **20% Testing** sets.
   - Stratified sampling (`stratify=y`) is applied to maintain target class ratios.
   - Random seeds are fixed (`SEED = 42`) for full reproducibility.
4. **Feature Scaling:** Inputs are normalized using `StandardScaler` (Mean = 0, Variance = 1).

---

## 🧠 Model Architectures

### 1. Keras Model (TensorFlow)

Built using the Sequential API:

- **Input Layer:** Dynamically shaped based on feature count
- **Hidden Layer 1:** 64 Neurons + ReLU + Dropout (0.2)
- **Hidden Layer 2:** 8 Neurons + ReLU + Dropout (0.2)
- **Output Layer:** 1 Neuron + Sigmoid activation
- **Optimizer:** Adam (Learning Rate = 0.0003)
- **Loss Function:** Binary Cross-Entropy
- **Callback:** `EarlyStopping` (patience = 5)

### 2. PyTorch Model

Built with an `nn.Module` object-oriented structure:

- **Architecture:** Linear(n_features, 16) ➔ ReLU ➔ Dropout(0.2) ➔ Linear(16, 1) ➔ Dropout(0.2)
- **Optimizer:** Adam (Learning Rate = 0.001)
- **Loss Function:** `BCEWithLogitsLoss` (Numerically stable Binary Cross-Entropy with integrated Sigmoid)
- **Data Handling:** Batched loading via `TensorDataset` and `DataLoader` (Batch Size = 32)

---

