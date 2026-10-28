# Heart-Disease Prediction with Logistic Regression

EPFL CS-433 Machine Learning, Project 1 (2026).

This project predicts the BRFSS `_MICHD` target from clinical, demographic, and lifestyle variables. The final system uses class-weighted regularized logistic regression, engineered health-risk features, and a decision threshold selected by stratified cross-validation.

## Team

- Dany Moukaddam - SCIPER 378920
- Alexandre Hatem - SCIPER 381642
- Antoine Bastide - SCIPER 375407

## Setup

The project uses Python with NumPy and Matplotlib only.

Create the virtual environment:
```bash
python -m venv .venv
```

Run the environment:

```bash
# Linux or macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Unzip and place the data files without renaming them:

```text
data/
└── dataset/
    ├── x_train.csv
    ├── y_train.csv
    └── x_test.csv
```

## Run

From the repository root, run:

```bash
python run.py
```

The script performs model selection, refits the selected configuration on all training examples, and writes `submission.csv` with predictions in `{-1, +1}`.

## Pipeline

### Data preparation

1. Select 42 clinical, demographic, and lifestyle variables listed in `run.py`.
2. Convert variable-specific missing and refusal codes to missing values; convert the BRFSS day-count code `88` to zero where applicable.
3. Impute missing values with medians learned from the current training fold.
4. Standardize features with training-fold means and standard deviations, using unit scale for constant features.
5. Add a constant intercept column.

### Feature engineering

Ten features are appended to the selected variables:

- comorbidity burden;
- age × general-health severity;
- age × hypertension;
- age × BMI;
- lifestyle-risk score;
- approximate Life's Essential 8 score;
- age × diabetes;
- diabetes × hypertension;
- age × current-smoking status;
- BMI × diabetes.

The comorbidity feature excludes diagnoses that define `_MICHD`. All engineered features use only the provided dataset.

### Model selection

- Model: class-weighted regularized logistic regression.
- Optimizer: gradient descent with learning rate `0.1`, at most `2,000` iterations, and optional stopping at gradient norm below `1e-5`.
- Validation: stratified five-fold cross-validation with seed `42`.
- Regularization values: `1e-5` and `1e-4`.
- Positive-class weight multipliers: `0.5`, `0.75`, and `1.0`.
- Selection metric: positive-class F1 from pooled out-of-fold predictions.
- Threshold: selected from the distinct out-of-fold probabilities to maximize F1.

The reported run selected `lambda = 1e-4`, positive-class weight multiplier `0.75`, and threshold `0.6513`, obtaining pooled out-of-fold F1 `0.4220`.

## Repository structure

- `implementations.py`: the six required ML methods and the weighted logistic-regression extension.
- `run.py`: data preparation, feature engineering, cross-validation, training, and prediction.
- `helpers.py`: CSV loading and submission-file generation.
- `report/`: LaTeX template and supporting resources.
