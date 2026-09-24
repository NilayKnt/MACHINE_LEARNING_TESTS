# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import statsmodels.api as sm
# %matplotlib inline
sns.set_style("whitegrid")
sns.set_context("poster")
import warnings
warnings.filterwarnings("ignore")
#

column_names = [
    "x1_compactness",
    "x2_surface",
    "x3_wall",
    "x4_roof",
    "x5_height",
    "x6_orientation",
    "x7_glazing",
    "x8_glazing_dist",
    "y1_heating",
    "y2_cooling",
]

df = pd.read_excel('ENB2012_data.xlsx', skiprows=1, header=None, names=column_names)
df.head(2)

df.info()

df.describe().T

"""## Regression Analysis"""

X = df.drop(columns=['y1_heating', 'y2_cooling'])
y1 = df['y1_heating'] # Heating Load
y2 = df['y2_cooling'] # Cooling Load

"""### Split Data into Training and Testing Sets"""

# Split the data for Heating Load
X_train_y1, X_test_y1, y_train_y1, y_test_y1 = train_test_split(X, y1, test_size=0.2, random_state=42)

# Split the data for Cooling Load
X_train_y2, X_test_y2, y_train_y2, y_test_y2 = train_test_split(X, y2, test_size=0.2, random_state=42)

"""### Train and Evaluate Linear Regression Models"""

# Initialize and train Linear Regression model for Heating Load
model_y1 = LinearRegression()
model_y1.fit(X_train_y1, y_train_y1)

# Make predictions for Heating Load
y_pred_y1 = model_y1.predict(X_test_y1)

# Evaluate the model for Heating Load
mae_y1 = mean_absolute_error(y_test_y1, y_pred_y1)
mse_y1 = mean_squared_error(y_test_y1, y_pred_y1)
r2_y1 = r2_score(y_test_y1, y_pred_y1)

print("--- Heating Load (y1) Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae_y1:.2f}")
print(f"Mean Squared Error (MSE): {mse_y1:.2f}")
print(f"R-squared (R2): {r2_y1:.2f}")

# Initialize and train Linear Regression model for Cooling Load
model_y2 = LinearRegression()
model_y2.fit(X_train_y2, y_train_y2)

# Make predictions for Cooling Load
y_pred_y2 = model_y2.predict(X_test_y2)

# Evaluate the model for Cooling Load
mae_y2 = mean_absolute_error(y_test_y2, y_pred_y2)
mse_y2 = mean_squared_error(y_test_y2, y_pred_y2)
r2_y2 = r2_score(y_test_y2, y_pred_y2)

print("\n--- Cooling Load (y2) Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae_y2:.2f}")
print(f"Mean Squared Error (MSE): {mse_y2:.2f}")
print(f"R-squared (R2): {r2_y2:.2f}")

"""## Residual Analysis for Heating Load (Y1) with Linear Regression"""

# Calculate residuals for Heating Load (Linear Regression)
residual_y1_linear = y_test_y1 - y_pred_y1

# Create subplots for Heating Load (Linear Regression) residuals
fig_linear_y1, axes_linear_y1 = plt.subplots(nrows=1, ncols=3, figsize=(21, 6))

# Plot 1: Actual vs. Predicted for Heating Load
axes_linear_y1[0].scatter(y_test_y1, y_pred_y1, alpha=0.4, color="teal")
axes_linear_y1[0].plot([y_test_y1.min(), y_test_y1.max()], [y_test_y1.min(), y_test_y1.max()], 'r--')
axes_linear_y1[0].set_xlabel("Gerçek Değerler (Isıtma Yükü)")
axes_linear_y1[0].set_ylabel("Tahmin Edilen Değerler (Isıtma Yükü)")
axes_linear_y1[0].set_title("Gerçek vs Tahmin Edilen (Lineer Regresyon)")

# Plot 2: Scatter plot of predicted vs residuals for Heating Load
axes_linear_y1[1].scatter(y_pred_y1, residual_y1_linear, alpha=0.4, color="teal")
axes_linear_y1[1].axhline(y=0, color="r", linestyle="-")
axes_linear_y1[1].set_xlabel("Tahmin Edilen Değerler (Isıtma Yükü)")
axes_linear_y1[1].set_ylabel("Kalıntılar")
axes_linear_y1[1].set_title("Isıtma Yükü Kalıntı Dağılımı (Lineer Regresyon)")

# Plot 3: Histogram of residuals for Heating Load
sns.histplot(residual_y1_linear, kde=True, ax=axes_linear_y1[2], color="teal")
axes_linear_y1[2].set_title("Isıtma Yükü Kalıntı Normalliği (Lineer Regresyon)")
axes_linear_y1[2].set_xlabel("Kalıntı Değeri")
axes_linear_y1[2].set_ylabel("Frekans")

plt.tight_layout()
plt.show()

"""## Residual Analysis for Cooling Load (Y2) with Linear Regression"""

# Calculate residuals for Cooling Load (Linear Regression)
residual_y2_linear = y_test_y2 - y_pred_y2

# Create subplots for Cooling Load (Linear Regression) residuals
fig_linear_y2, axes_linear_y2 = plt.subplots(nrows=1, ncols=3, figsize=(21, 6))

# Plot 1: Actual vs. Predicted for Cooling Load
axes_linear_y2[0].scatter(y_test_y2, y_pred_y2, alpha=0.4, color="purple")
axes_linear_y2[0].plot([y_test_y2.min(), y_test_y2.max()], [y_test_y2.min(), y_test_y2.max()], 'r--')
axes_linear_y2[0].set_xlabel("Gerçek Değerler (Soğutma Yükü)")
axes_linear_y2[0].set_ylabel("Tahmin Edilen Değerler (Soğutma Yükü)")
axes_linear_y2[0].set_title("Gerçek vs Tahmin Edilen (Lineer Regresyon)")

# Plot 2: Scatter plot of predicted vs residuals for Cooling Load
axes_linear_y2[1].scatter(y_pred_y2, residual_y2_linear, alpha=0.4, color="purple")
axes_linear_y2[1].axhline(y=0, color="r", linestyle="-")
axes_linear_y2[1].set_xlabel("Tahmin Edilen Değerler (Soğutma Yükü)")
axes_linear_y2[1].set_ylabel("Kalıntılar")
axes_linear_y2[1].set_title("Soğutma Yükü Kalıntı Dağılımı (Lineer Regresyon)")

# Plot 3: Histogram of residuals for Cooling Load
sns.histplot(residual_y2_linear, kde=True, ax=axes_linear_y2[2], color="purple")
axes_linear_y2[2].set_title("Soğutma Yükü Kalıntı Normalliği (Lineer Regresyon)")
axes_linear_y2[2].set_xlabel("Kalıntı Değeri")
axes_linear_y2[2].set_ylabel("Frekans")

plt.tight_layout()
plt.show()

"""## Polynomial Regression Visualization (Single Feature: x2_surface)"""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

degrees=[1,2,3,4] # Kullanıcının sağladığı dereceler
poly_results_y1 = {}

# 'x2_surface' özelliğini seç
X_one_feature_train_y1 = X_train_y1[['x2_surface']].values
X_one_feature_test_y1 = X_test_y1[['x2_surface']].values

plt.figure(figsize=(10,6))

# Gerçek değerlerin dağılımını scatter plot olarak çiz
plt.scatter(X_one_feature_test_y1, y_test_y1, alpha=0.4, color="teal", label="Gerçek Değerler")

# Çizim için sıralanmış X değerlerini elde et
x_sorted_y1 = np.sort(X_one_feature_test_y1.flatten())

for degree in degrees:
    model_y1_poly_single = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model_y1_poly_single.fit(X_one_feature_train_y1, y_train_y1)
    poly_pred_y1_single = model_y1_poly_single.predict(X_one_feature_test_y1)
    poly_results_y1[degree] = {
        "r2": r2_score(y_test_y1, poly_pred_y1_single),
        "RMSE": np.sqrt(mean_squared_error(y_test_y1, poly_pred_y1_single))
    }

    # Tahmin çizgisini çizmek için sıralanmış x değerleri üzerinde tahmin yap
    y_line_y1 = model_y1_poly_single.predict(x_sorted_y1.reshape(-1,1))
    plt.plot(x_sorted_y1, y_line_y1, linewidth=2, label=f"Derece {degree}")

plt.xlabel("Yüzey Alanı (x2_surface)")
plt.ylabel("Isıtma Yükü (y1_heating)")
plt.title("Yüzey Alanı ile Polinom Regresyonu (Isıtma Yükü)")
plt.legend()
plt.show()

display(pd.DataFrame(poly_results_y1).T.rename_axis("Derece").rename(columns={"r2":"R2","RMSE":"MAE"}))

print("\n--- Soğutma Yükü (y2) için Polinom Regresyonu ---")
poly_results_y2 = {}

# 'x2_surface' özelliğini seç
X_one_feature_train_y2 = X_train_y2[['x2_surface']].values
X_one_feature_test_y2 = X_test_y2[['x2_surface']].values

plt.figure(figsize=(10,6))

# Gerçek değerlerin dağılımını scatter plot olarak çiz
plt.scatter(X_one_feature_test_y2, y_test_y2, alpha=0.4, color="purple", label="Gerçek Değerler")

# Çizim için sıralanmış X değerlerini elde et
x_sorted_y2 = np.sort(X_one_feature_test_y2.flatten())

for degree in degrees:
    model_y2_poly_single = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model_y2_poly_single.fit(X_one_feature_train_y2, y_train_y2)
    poly_pred_y2_single = model_y2_poly_single.predict(X_one_feature_test_y2)
    poly_results_y2[degree] = {
        "r2": r2_score(y_test_y2, poly_pred_y2_single),
        "RMSE": np.sqrt(mean_squared_error(y_test_y2, poly_pred_y2_single))
    }

    # Tahmin çizgisini çizmek için sıralanmış x değerleri üzerinde tahmin yap
    y_line_y2 = model_y2_poly_single.predict(x_sorted_y2.reshape(-1,1))
    plt.plot(x_sorted_y2, y_line_y2, linewidth=2, label=f"Derece {degree}")

plt.xlabel("Yüzey Alanı (x2_surface)")
plt.ylabel("Soğutma Yükü (y2_cooling)")
plt.title("Yüzey Alanı ile Polinom Regresyonu (Soğutma Yükü)")
plt.legend()
plt.show()

display(pd.DataFrame(poly_results_y2).T.rename_axis("Derece").rename(columns={"r2":"R2","RMSE":"MAE"}))

"""## Logistic Regression: Creating a Binary Target Variable
"""

median_y1 = df['y1_heating'].median()
print(f"Median of y1_heating: {median_y1:.2f}")

df['y1_heating_category'] = (df['y1_heating'] > median_y1).astype(int)

print("\nDistribution of the new binary target variable (0: Low Heating Load, 1: High Heating Load):")
display(df['y1_heating_category'].value_counts())

"""## Logistic Regression Classification"""

# Define features (X) and the new binary target (y)
X_classification = df.drop(columns=['y1_heating', 'y2_cooling', 'y1_heating_category'])
y_classification = df['y1_heating_category']

# Split data into training and testing sets
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_classification, y_classification, test_size=0.2, random_state=42, stratify=y_classification
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_cls)
X_test_scaled = scaler.transform(X_test_cls)

print("Data split and scaled successfully.")

"""### Train and Evaluate Logistic Regression Model"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# Initialize and train the Logistic Regression model
log_reg_model = LogisticRegression(random_state=42, solver='liblinear') # 'liblinear' is good for small datasets and binary classification
log_reg_model.fit(X_train_scaled, y_train_cls)

# Make predictions on the scaled test set
y_pred_cls = log_reg_model.predict(X_test_scaled)
y_pred_proba = log_reg_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test_cls, y_pred_cls)
precision = precision_score(y_test_cls, y_pred_cls)
recall = recall_score(y_test_cls, y_pred_cls)
f1 = f1_score(y_test_cls, y_pred_cls)
auc_score = roc_auc_score(y_test_cls, y_pred_proba)

print("--- Logistic Regression Model Evaluation (y1_heating_category) ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC Score: {auc_score:.4f}")

"""### Confusion Matrix"""

# Plot the Confusion Matrix
cm = confusion_matrix(y_test_cls, y_pred_cls)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low Heating', 'High Heating'])

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title('Confusion Matrix for Heating Load Classification')
plt.show()

"""## Actual vs. Predicted Probabilities Table"""

# Get predicted probabilities
y_pred_proba = log_reg_model.predict_proba(X_test_scaled)

# Create a DataFrame to display actual vs predicted probabilities
probability_df = pd.DataFrame({
    'Actual_y1_category': y_test_cls,
    'Predicted_Prob_LowHeating': y_pred_proba[:, 0],
    'Predicted_Prob_HighHeating': y_pred_proba[:, 1]
})

# Display the first few rows of the probability table
display(probability_df.head())


