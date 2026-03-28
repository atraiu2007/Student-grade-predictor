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
