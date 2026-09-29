"""Prediction services."""
import numpy as np
import pandas as pd

from .data_generator import FEATURES
from .validation import validate_student_input


def predict_score(model, study_hours, attendance, sleep_hours, previous_score):
    values = validate_student_input(study_hours, attendance, sleep_hours, previous_score)
    row = pd.DataFrame([values], columns=FEATURES)
    prediction = float(model.predict(row)[0])
    return round(float(np.clip(prediction, 0, 100)), 2)


def grade_band(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"
