# Student Grade Predictor (Intermediate Beginner Version)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# -------------------------------
# 1. CREATE DATASET
# -------------------------------
np.random.seed(42)
n = 150

study_hours = np.random.uniform(1, 10, n)
attendance = np.random.uniform(50, 100, n)
sleep_hours = np.random.uniform(4, 9, n)
previous_score = np.random.uniform(30, 90, n)

# Simple formula + noise
final_score = (
    7 * study_hours +
    0.3 * attendance +
    1.5 * sleep_hours +
    0.4 * previous_score +
    np.random.normal(0, 5, n)
)

final_score = np.clip(final_score, 0, 100)

# Create DataFrame
df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "sleep_hours": sleep_hours,
    "previous_score": previous_score,
    "final_score": final_score
})

print("\nDataset Preview:\n")
print(df.head())
# -------------------------------
# 2. BASIC ANALYSIS
# -------------------------------
print("\nDataset Info:\n")
print(df.describe())

# -------------------------------
# 3. PREPARE DATA
# -------------------------------
X = df[["study_hours", "attendance", "sleep_hours", "previous_score"]]
y = df["final_score"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. TRAIN MODEL
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# -------------------------------
# 5. PREDICTIONS
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 6. EVALUATION
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error:", round(mae, 2))

# Show coefficients
print("\nFeature Importance:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {round(coef, 2)}")

# -------------------------------
# 7. SIMPLE VISUALIZATION
# -------------------------------
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted")

# Line for reference
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.show()

# -------------------------------
# 8. PREDICTION FUNCTION
# -------------------------------
def predict_grade(study, attend, sleep, prev):
    input_data = np.array([[study, attend, sleep, prev]])
    prediction = model.predict(input_data)[0]
    return round(float(np.clip(prediction, 0, 100)), 2)

# -------------------------------
# 9. TEST CASES
# -------------------------------
print("\nSample Predictions:\n")

students = [
    ("Top Student", 8, 95, 8, 85),
    ("Average Student", 5, 75, 6, 60),
    ("Low Performer", 2, 55, 5, 40)
]

for name, sh, att, sl, ps in students:
    score = predict_grade(sh, att, sl, ps)
    print(f"{name}: {score} / 100")
