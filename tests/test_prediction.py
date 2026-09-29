import pytest
from sklearn.linear_model import LinearRegression
import pandas as pd

from src.prediction import grade_band, predict_score


def test_grade_band_boundaries():
    assert grade_band(95) == "A"
    assert grade_band(85) == "B"
    assert grade_band(75) == "C"
    assert grade_band(65) == "D"
    assert grade_band(50) == "F"


def test_prediction_range():
    df = pd.DataFrame({
        "study_hours": [2, 5, 8, 10],
        "attendance": [60, 75, 90, 95],
        "sleep_hours": [5, 6, 7, 8],
        "previous_score": [40, 60, 75, 90],
        "final_score": [45, 65, 82, 95],
    })
    model = LinearRegression().fit(df.iloc[:, :4], df.iloc[:, 4])
    score = predict_score(model, 7, 85, 7, 75)
    assert 0 <= score <= 100


def test_invalid_attendance():
    df = pd.DataFrame({
        "study_hours": [2, 5], "attendance": [60, 75],
        "sleep_hours": [5, 6], "previous_score": [40, 60],
        "final_score": [45, 65],
    })
    model = LinearRegression().fit(df.iloc[:, :4], df.iloc[:, 4])
    with pytest.raises(ValueError):
        predict_score(model, 5, 150, 7, 70)
