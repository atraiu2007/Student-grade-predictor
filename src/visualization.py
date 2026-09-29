"""Visualization functions."""
from pathlib import Path
import matplotlib.pyplot as plt


def save_actual_vs_predicted(actual, predicted, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.scatter(actual, predicted, alpha=0.7)
    low = min(min(actual), min(predicted))
    high = max(max(actual), max(predicted))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.title("Actual vs Predicted Scores")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path
