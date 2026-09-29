"""Analytics and reporting helpers."""
from .data_generator import FEATURES


def dataset_summary(df):
    return {
        "records": len(df),
        "average_score": round(float(df["final_score"].mean()), 2),
        "minimum_score": round(float(df["final_score"].min()), 2),
        "maximum_score": round(float(df["final_score"].max()), 2),
    }


def feature_effects(model):
    """Return model coefficients/importance values in a common dictionary."""
    if hasattr(model, "coef_"):
        return dict(zip(FEATURES, model.coef_))
    return dict(zip(FEATURES, model.feature_importances_))


def model_comparison(results):
    return sorted(
        ((name, round(info["mae"], 3), round(info["r2"], 3)) for name, info in results.items()),
        key=lambda item: item[1],
    )
