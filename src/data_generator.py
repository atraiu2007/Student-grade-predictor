"""Dataset generation and loading utilities."""
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ["study_hours", "attendance", "sleep_hours", "previous_score"]
TARGET = "final_score"


def generate_dataset(path: str | Path, n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible synthetic student-performance dataset."""
    rng = np.random.default_rng(seed)
    study_hours = rng.uniform(1, 10, n)
    attendance = rng.uniform(50, 100, n)
    sleep_hours = rng.uniform(4, 9, n)
    previous_score = rng.uniform(30, 90, n)

    final_score = (
        7 * study_hours
        + 0.3 * attendance
        + 1.5 * sleep_hours
        + 0.4 * previous_score
        + rng.normal(0, 5, n)
    )
    final_score = np.clip(final_score, 0, 100)

    df = pd.DataFrame({
        "study_hours": study_hours,
        "attendance": attendance,
        "sleep_hours": sleep_hours,
        "previous_score": previous_score,
        "final_score": final_score,
    })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a dataset from CSV."""
    return pd.read_csv(path)
