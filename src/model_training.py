"""Model training and evaluation."""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .data_generator import FEATURES, TARGET


def train_models(df, random_state=42):
    """Train Linear Regression and Random Forest models and return results."""
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=random_state
        ),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = {
            "model": model,
            "mae": mean_absolute_error(y_test, predictions),
            "r2": r2_score(y_test, predictions),
            "predictions": predictions,
            "actual": y_test,
        }
    return results
