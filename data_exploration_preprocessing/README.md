
## 📊 Project Architecture

### 1. 🔍 Data Exploration & Preprocessing
- **Dataset Loading:** Exploration of customer segmentation data (`Mall_Customers.csv`).
- **Data Hygiene:** Checking structural integrity, summary statistics, and missing values.
- **Feature Scaling:** Standardizing numerical values to meet model requirements.


### 2. 🧩 Unsupervised Learning: Clustering Algorithms
Applied to customer segmentation (`Mall_Customers.csv`) to discover underlying patterns without pre-existing labels.

| Algorithm | Key Techniques & Evaluation | Visualizations |
| :--- | :--- | :--- |
| **K-Means** | Optimal $K$ selection via **Elbow Method** & **Silhouette Analysis** | 2D/3D Cluster Scatter Plots |
| **Hierarchical** | Structural analysis via **Dendrograms**, Agglomerative implementation | Tree Hierarchy & Cluster Maps |
| **DBSCAN** | Density parameter tuning via **$k$-distance Graph** | Density & Noise Scatter Plots |

---

### 3. 🎯 Supervised Learning: Classification Models
Implemented using the **Breast Cancer Dataset** for predictive diagnosis and tree-based decision analysis.

* **🌳 Decision Tree Classifier**
  * Model training & validation split
  * Hyperparameter tuning (max depth, min samples split)
  * Tree structure visualization for interpretable logic
* **🌲 Random Forest Classifier**
  * Ensemble model training & performance metrics
  * Hyperparameter tuning across decision trees
  * Feature importance ranking & visualization

---

## 🛠️ Tech Stack & Dependencies

- **Language:** Python
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`
- **Data Visualization:** `matplotlib`, `seaborn`

---

