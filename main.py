"""Command-line entry point for Student Grade Predictor."""
from pathlib import Path

from src.analytics import dataset_summary, feature_effects, model_comparison
from src.data_generator import generate_dataset, load_dataset
from src.model_training import train_models
from src.prediction import grade_band, predict_score
from src.visualization import save_actual_vs_predicted

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_data.csv"
OUTPUT_PATH = ROOT / "outputs" / "actual_vs_predicted.png"


def print_report(df, results):
    summary = dataset_summary(df)
    print("\n=== DATASET SUMMARY ===")
    print(f"Records: {summary['records']}")
    print(f"Average final score: {summary['average_score']}")
    print(f"Score range: {summary['minimum_score']} - {summary['maximum_score']}")

    print("\n=== MODEL COMPARISON ===")
    for name, mae, r2 in model_comparison(results):
        print(f"{name:<20} MAE: {mae:<8} R²: {r2}")

    best_name = model_comparison(results)[0][0]
    best = results[best_name]
    print(f"\nSelected model: {best_name}")
    print("Feature effects/importance:")
    for feature, value in feature_effects(best["model"]).items():
        print(f"  {feature:<18} {value:.3f}")
    save_actual_vs_predicted(best["actual"], best["predictions"], OUTPUT_PATH)
    print(f"\nChart saved to: {OUTPUT_PATH}")
    return best["model"]


def ask_for_prediction(model):
    print("\n=== STUDENT SCORE PREDICTION ===")
    try:
        study = float(input("Study hours per day: "))
        attendance = float(input("Attendance (%): "))
        sleep = float(input("Sleep hours per day: "))
        previous = float(input("Previous score (%): "))
        score = predict_score(model, study, attendance, sleep, previous)
        print(f"\nPredicted final score: {score}/100")
        print(f"Predicted grade band: {grade_band(score)}")
    except ValueError as exc:
        print(f"Input error: {exc}")
    except EOFError:
        print("No interactive input supplied. Training/evaluation completed successfully.")


def main():
    print("=" * 50)
    print("        STUDENT GRADE PREDICTOR")
    print("=" * 50)

    if not DATA_PATH.exists():
        df = generate_dataset(DATA_PATH)
        print("Generated a new reproducible dataset.")
    else:
        df = load_dataset(DATA_PATH)
        print("Loaded existing dataset.")

    results = train_models(df)
    model = print_report(df, results)
    ask_for_prediction(model)


if __name__ == "__main__":
    main()
