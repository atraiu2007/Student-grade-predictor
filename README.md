# Student Grade Predictor

A modular Python command-line application that predicts a student's final score from study hours, attendance, sleep hours, and previous academic score. The project demonstrates data generation/loading, preprocessing and validation, supervised regression, model comparison, prediction, analytics, visualization, testing, and Git-based project organization.

## Problem Overview

Students' academic outcomes can be influenced by several measurable factors. This project demonstrates how a small machine-learning pipeline can use study habits and previous academic performance to estimate a final score.

> **Important limitation:** the included dataset is synthetic. It is generated from assumed relationships plus random noise for educational demonstration. The model must not be interpreted as a validated real-world student assessment system.

## Features

1. **Data Management**
   - Reproducible synthetic dataset generation
   - CSV storage and loading
   - Dataset summary

2. **Prediction Engine**
   - Linear Regression
   - Random Forest Regression
   - Input validation
   - Final score prediction and grade-band conversion

3. **Analytics & Evaluation**
   - Mean Absolute Error (MAE)
   - R² score
   - Model comparison
   - Feature coefficients/importance
   - Actual-vs-predicted visualization

4. **Testing**
   - Input validation tests
   - Prediction range tests
   - Grade-band tests
   - Dataset generation tests

## Technologies

- Python 3.10+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pytest

## Project Structure

```text
student-grade-predictor/
├── main.py
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── validation.py
│   ├── model_training.py
│   ├── prediction.py
│   ├── analytics.py
│   └── visualization.py
├── data/
│   └── student_data.csv
├── outputs/
│   └── actual_vs_predicted.png
├── tests/
│   ├── test_data.py
│   └── test_prediction.py
├── docs/
│   ├── architecture.png
│   ├── workflow.png
│   ├── use_case.png
│   ├── class_diagram.png
│   └── sequence_diagram.png
├── README.md
├── statement.md
├── requirements.txt
├── .gitignore
└── Project_Report.pdf
```

## Installation

```bash
python -m venv .venv
```

### macOS/Linux

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

From the repository root:

```bash
python main.py
```

The program will load `data/student_data.csv` if it exists. If the file is missing, it will generate a new deterministic dataset.

The program trains both regression models, prints evaluation metrics, saves the best-model visualization to `outputs/actual_vs_predicted.png`, and asks for a student's inputs.

## Run Tests

```bash
pytest -q
```

## Example Input

```text
Study hours per day: 7
Attendance (%): 85
Sleep hours: 7
Previous score (%): 75
```

The program returns a predicted score between 0 and 100 and a corresponding grade band.

## Evaluation Method

The dataset is split into training and testing subsets using an 80/20 split. Both models are evaluated using:

- **MAE:** average absolute prediction error; lower is better.
- **R²:** proportion of target variance explained by the model; higher is generally better.

The application selects the model with the lower MAE for the final prediction workflow.

## Design Notes

The application is intentionally command-line based so it can be executed from a terminal without a GUI-specific setup. Each major responsibility is separated into a Python module to improve maintainability and testability.

## Limitations

- Dataset is synthetic and based on assumed relationships.
- Only four input features are used.
- Predictions are educational demonstrations rather than validated academic assessments.
- Model performance on the synthetic dataset should not be generalized to real students.

## Future Enhancements

- Replace synthetic data with an ethically sourced real dataset.
- Add additional regression/classification models.
- Add cross-validation and hyperparameter tuning.
- Add a web interface after the command-line version is stable.
- Add explainability methods and fairness checks for real-world deployment.
