import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('manufacturing_dataset_1000_samples.csv')

# Drop the 'Timestamp' column as it's not a feature for the model
df = df.drop('Timestamp', axis=1)

# --- Feature Engineering (numerical derived features) ---
df['Temperature_Pressure_Ratio'] = df['Injection_Temperature'] / df['Injection_Pressure']
df['Total_Cycle_Time'] = df['Cycle_Time'] + df['Cooling_Time']
df['Efficiency_Score'] = (df['Injection_Temperature'] / df['Injection_Pressure']) / df['Cycle_Time']
df['Machine_Utilization'] = df['Total_Cycle_Time'] / (df['Total_Cycle_Time'] + 10)

# Identify features (X) and target (y)
y = df['Parts_Per_Hour']
X = df.drop('Parts_Per_Hour', axis=1)

# --- Handle Categorical Variables ---
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# --- Handle Missing Values ---
X = X.fillna(X.mean())

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# --- Evaluate Model ---
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.4f}")

# Save coefficients (optional)
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\n--- Model Coefficients ---")
print(coefficients.head())

# Save scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, label='Predicted vs. Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Actual vs Predicted Parts Per Hour')
plt.xlabel('Actual Parts Per Hour')
plt.ylabel('Predicted Parts Per Hour')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('prediction_scatter_plot.png')

# --- Save Model and Feature Order ---
with open('linear_regression_model_all_features.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('trained_features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("\n✅ Model and trained_features.pkl saved successfully.")
