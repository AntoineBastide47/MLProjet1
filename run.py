"""Produce the best AIcrowd submission from the raw data.

Usage: python run.py
Reads data/dataset/{x_train,y_train,x_test}.csv and writes submission.csv.
"""

import csv

from helpers import create_csv_submission, load_csv_data
import implementations
import numpy as np

DATA_PATH = "data/dataset"
SUBMISSION_PATH = "submission.csv"

# A focused set of clinical, demographic, and behavior predictors. Survey IDs,
# dates, and sampling/weighting fields are intentionally excluded.
SELECTED_FEATURES = [
    "GENHLTH",
    "PHYSHLTH",
    "MENTHLTH",
    "POORHLTH",
    "HLTHPLN1",
    "PERSDOC2",
    "MEDCOST",
    "CHECKUP1",
    "BPHIGH4",
    "BPMEDS",
    "BLOODCHO",
    "CHOLCHK",
    "TOLDHI2",
    "CVDSTRK3",
    "ASTHMA3",
    "ASTHNOW",
    "CHCSCNCR",
    "CHCOCNCR",
    "CHCCOPD1",
    "HAVARTH3",
    "ADDEPEV2",
    "CHCKIDNY",
    "DIABETE3",
    "SEX",
    "MARITAL",
    "EDUCA",
    "RENTHOM1",
    "EMPLOY1",
    "INCOME2",
    "QLACTLM2",
    "DIFFWALK",
    "SMOKE100",
    "SMOKDAY2",
    "STOPSMK2",
    "USENOW3",
    "EXERANY2",
    "_AGE80",
    "_BMI5",
    "_SMOKER3",
    "_TOTINDA",
    "_FRTLT1",
    "_VEGLT1",
]

# Missing/refused codes are variable-specific in BRFSS.
MISSING_CODES = {
    "GENHLTH": (7, 9),
    "PHYSHLTH": (77, 99),
    "MENTHLTH": (77, 99),
    "POORHLTH": (77, 99),
    "HLTHPLN1": (7, 9),
    "PERSDOC2": (7, 9),
    "MEDCOST": (7, 9),
    "CHECKUP1": (7, 9),
    "BPHIGH4": (7, 9),
    "BPMEDS": (7, 9),
    "BLOODCHO": (7, 9),
    "CHOLCHK": (7, 9),
    "TOLDHI2": (7, 9),
    "CVDSTRK3": (7, 9),
    "ASTHMA3": (7, 9),
    "ASTHNOW": (7, 9),
    "CHCSCNCR": (7, 9),
    "CHCOCNCR": (7, 9),
    "CHCCOPD1": (7, 9),
    "HAVARTH3": (7, 9),
    "ADDEPEV2": (7, 9),
    "CHCKIDNY": (7, 9),
    "DIABETE3": (7, 9),
    "MARITAL": (9,),
    "EDUCA": (9,),
    "RENTHOM1": (7, 9),
    "EMPLOY1": (9,),
    "INCOME2": (77, 99),
    "QLACTLM2": (7, 9),
    "DIFFWALK": (7, 9),
    "SMOKE100": (7, 9),
    "SMOKDAY2": (7, 9),
    "STOPSMK2": (7, 9),
    "USENOW3": (7, 9),
    "EXERANY2": (7, 9),
    "_BMI5": (9999,),
    "_SMOKER3": (9,),
    "_TOTINDA": (9,),
    "_FRTLT1": (9,),
    "_VEGLT1": (9,),
}

# BRFSS day-count questions use 88 to mean "none" rather than missing.
ZERO_CODES = {"PHYSHLTH": (88,), "MENTHLTH": (88,), "POORHLTH": (88,)}


def select_and_clean_features(x, source_names):
    """Select predictors and turn their documented missing codes into NaN."""
    positions = {name: i for i, name in enumerate(source_names)}
    missing = [name for name in SELECTED_FEATURES if name not in positions]
    if missing:
        raise ValueError(f"Missing expected BRFSS columns: {missing}")

    selected = np.asarray(x, dtype=float)[
        :, [positions[n] for n in SELECTED_FEATURES]
    ].copy()
    for i, name in enumerate(SELECTED_FEATURES):
        for code in ZERO_CODES.get(name, ()):
            selected[selected[:, i] == code, i] = 0.0
        for code in MISSING_CODES.get(name, ()):
            selected[selected[:, i] == code, i] = np.nan
    return selected, SELECTED_FEATURES.copy()


def engineer_features(x, feature_names):
    """Append clinically motivated interactions and risk summaries."""
    x = np.asarray(x, dtype=float)
    columns = {name: i for i, name in enumerate(feature_names)}

    def get(name):
        if name not in columns:
            raise ValueError(f"Required BRFSS feature is missing: {name}")
        return x[:, columns[name]].astype(float, copy=True)

    age = get("_AGE80")
    age[(age < 18) | (age > 80)] = np.nan

    bmi = get("_BMI5") / 100.0  # BRFSS _BMI5 stores BMI with 2 implied decimals.
    bmi[(bmi < 12) | (bmi > 80)] = np.nan

    health = get("GENHLTH")
    health[(health < 1) | (health > 5)] = np.nan
    health_severity = health - 1.0  # 0=excellent through 4=poor.

    hypertension_code = get("BPHIGH4")
    hypertension = np.full(x.shape[0], np.nan)
    hypertension[np.isin(hypertension_code, [1, 2])] = 1.0
    hypertension[hypertension_code == 3] = 0.0
    hypertension[hypertension_code == 4] = 0.5  # borderline

    diabetes_code = get("DIABETE3")
    diabetes = np.full(x.shape[0], np.nan)
    diabetes[diabetes_code == 1] = 1.0
    diabetes[diabetes_code == 2] = 0.5  # pregnancy-only
    diabetes[diabetes_code == 3] = 0.0
    diabetes[diabetes_code == 4] = 0.5  # prediabetes/borderline

    smoker_code = get("_SMOKER3")
    smoking_risk = np.full(x.shape[0], np.nan)
    smoking_risk[np.isin(smoker_code, [1, 2])] = 1.0  # current smoker
    smoking_risk[smoker_code == 3] = 0.5  # former smoker
    smoking_risk[smoker_code == 4] = 0.0  # never smoked
    current_smoker = np.full(x.shape[0], np.nan)
    current_smoker[np.isin(smoker_code, [1, 2])] = 1.0
    current_smoker[np.isin(smoker_code, [3, 4])] = 0.0

    # 1=active, 2=inactive; 9/other codes are treated as missing.
    activity_code = get("_TOTINDA")
    activity_risk = np.full(x.shape[0], np.nan)
    activity_risk[activity_code == 1] = 0.0
    activity_risk[activity_code == 2] = 1.0

    fruit_code = get("_FRTLT1")
    fruit_risk = np.full(x.shape[0], np.nan)
    fruit_risk[fruit_code == 1] = 0.0
    fruit_risk[fruit_code == 2] = 1.0

    vegetable_code = get("_VEGLT1")
    vegetable_risk = np.full(x.shape[0], np.nan)
    vegetable_risk[vegetable_code == 1] = 0.0
    vegetable_risk[vegetable_code == 2] = 1.0

    # Count other chronic conditions. _MICHD is the target, so its component
    # diagnoses (heart attack and coronary heart disease) are deliberately omitted.
    comorbidity_names = [
        "CVDSTRK3",
        "ASTHMA3",
        "CHCSCNCR",
        "CHCOCNCR",
        "CHCCOPD1",
        "HAVARTH3",
        "ADDEPEV2",
        "CHCKIDNY",
    ]
    comorbidity_flags = []
    for name in comorbidity_names:
        code = get(name)
        flag = np.full(x.shape[0], np.nan)
        flag[code == 1] = 1.0
        flag[code == 2] = 0.0
        comorbidity_flags.append(flag)
    comorbidity_flags.extend([hypertension, diabetes])
    comorbidity_matrix = np.column_stack(comorbidity_flags)
    known_comorbidities = np.sum(np.isfinite(comorbidity_matrix), axis=1)
    comorbidity_burden = np.nansum(comorbidity_matrix, axis=1)
    comorbidity_burden[known_comorbidities == 0] = np.nan

    obesity_risk = np.where(np.isfinite(bmi), (bmi >= 30).astype(float), np.nan)
    lifestyle_parts = np.column_stack(
        [smoking_risk, activity_risk, fruit_risk, vegetable_risk, obesity_risk]
    )
    known_lifestyle = np.sum(np.isfinite(lifestyle_parts), axis=1)
    lifestyle_risk = np.nansum(lifestyle_parts, axis=1)
    lifestyle_risk[known_lifestyle == 0] = np.nan

    # Approximate LE8 with available BRFSS proxies. Sleep duration and measured
    # BP/lipid/glucose values are absent, so these are explicitly proxy scores.
    fruit_ok = 1.0 - fruit_risk
    vegetable_ok = 1.0 - vegetable_risk
    diet_score = 50.0 * (fruit_ok + vegetable_ok)
    diet_score[~(np.isfinite(fruit_ok) & np.isfinite(vegetable_ok))] = np.nan
    activity_score = 100.0 * (1.0 - activity_risk)
    nicotine_score = 100.0 * (1.0 - smoking_risk)
    bmi_score = np.full(x.shape[0], np.nan)
    bmi_score[(bmi >= 12) & (bmi < 25)] = 100.0
    bmi_score[(bmi >= 25) & (bmi < 30)] = 70.0
    bmi_score[(bmi >= 30) & (bmi < 35)] = 30.0
    bmi_score[(bmi >= 35) & (bmi < 40)] = 15.0
    bmi_score[bmi >= 40] = 0.0

    bp_score = np.full(x.shape[0], np.nan)
    bp_score[hypertension == 0] = 100.0
    bp_score[hypertension == 0.5] = 50.0
    bp_score[hypertension == 1] = 25.0

    cholesterol_code = get("TOLDHI2")
    cholesterol_score = np.full(x.shape[0], np.nan)
    cholesterol_score[cholesterol_code == 1] = 50.0  # self-reported high cholesterol
    cholesterol_score[cholesterol_code == 2] = 100.0

    glucose_score = np.full(x.shape[0], np.nan)
    glucose_score[diabetes == 0] = 100.0
    glucose_score[diabetes == 0.5] = 60.0
    glucose_score[diabetes == 1] = 40.0

    le8_components = np.column_stack(
        [
            diet_score,
            activity_score,
            nicotine_score,
            bmi_score,
            cholesterol_score,
            glucose_score,
            bp_score,
        ]
    )
    le8_count = np.sum(np.isfinite(le8_components), axis=1)
    le8_proxy = np.nansum(le8_components, axis=1) / np.maximum(le8_count, 1)
    le8_proxy[le8_count == 0] = np.nan

    engineered = np.column_stack(
        [
            comorbidity_burden,
            age * health_severity,
            age * hypertension,
            age * bmi,
            lifestyle_risk,
            le8_proxy,
            age * diabetes,
            diabetes * hypertension,
            age * current_smoker,
            bmi * diabetes,
        ]
    )

    return np.column_stack((x, engineered))


def preprocess(x_train, x_apply):
    """Fit preprocessing on x_train and apply it to x_apply."""
    x_train = np.asarray(x_train, dtype=float).copy()
    x_apply = np.asarray(x_apply, dtype=float).copy()

    x_train[~np.isfinite(x_train)] = np.nan
    x_apply[~np.isfinite(x_apply)] = np.nan

    all_missing = np.all(np.isnan(x_train), axis=0)
    medians = np.zeros(x_train.shape[1])
    medians[~all_missing] = np.nanmedian(x_train[:, ~all_missing], axis=0)
    x_train = np.where(np.isnan(x_train), medians, x_train)
    x_apply = np.where(np.isnan(x_apply), medians, x_apply)

    # Standardize using training means and standard deviations.
    centers = x_train.mean(axis=0)
    scales = x_train.std(axis=0)
    scales[scales == 0] = 1.0
    x_train = (x_train - centers) / scales
    x_apply = (x_apply - centers) / scales

    # Add an intercept column.
    x_train = np.column_stack((np.ones(x_train.shape[0]), x_train))
    x_apply = np.column_stack((np.ones(x_apply.shape[0]), x_apply))

    return x_train, x_apply


def _stratified_folds(y, n_folds=5, seed=42):
    rng = np.random.default_rng(seed)
    fold_parts = [[] for _ in range(n_folds)]

    for label in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == label))
        for fold, part in enumerate(np.array_split(indices, n_folds)):
            fold_parts[fold].append(part)

    folds = [np.concatenate(parts) for parts in fold_parts]
    return [
        (np.concatenate([folds[j] for j in range(n_folds) if j != i]), folds[i])
        for i in range(n_folds)
    ]


def classification_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    tn = np.sum((y_true == -1) & (y_pred == -1))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator else 0.0
    accuracy = (tp + tn) / len(y_true)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _fit(y, tx, regularization, positive_weight_scale=1.0):
    y01 = (y == 1).astype(float)
    n_positive = np.sum(y01 == 1)
    n_negative = np.sum(y01 == 0)

    class_weight = {
        0: len(y01) / (2 * n_negative),
        1: positive_weight_scale * len(y01) / (2 * n_positive),
    }

    initial_w = np.zeros(tx.shape[1])
    w, loss, info = implementations.weighted_reg_logistic_regression(
        y01,
        tx,
        regularization,
        initial_w,
        max_iters=2000,
        gamma=0.1,
        early_stop=True,
        class_weight=class_weight,
        return_info=True,
    )
    info["regularized_loss"] = float(loss)
    return w, info


def _sigmoid(scores):
    scores = np.clip(scores, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-scores))


def _best_f1_threshold(y_true, probabilities):
    """Find the best threshold among all distinct out-of-fold probabilities."""
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    sorted_positive = (y_true[order] == 1).astype(np.int64)
    cumulative_tp = np.cumsum(sorted_positive)
    cumulative_fp = np.arange(1, len(y_true) + 1) - cumulative_tp
    total_positive = np.sum(y_true == 1)

    # Only evaluate positions at the end of tied-score groups.
    ends = np.r_[
        np.flatnonzero(sorted_probabilities[:-1] > sorted_probabilities[1:]),
        len(y_true) - 1,
    ]
    tp = cumulative_tp[ends]
    fp = cumulative_fp[ends]
    fn = total_positive - tp
    f1 = np.divide(
        2 * tp,
        2 * tp + fp + fn,
        out=np.zeros_like(tp, dtype=float),
        where=(2 * tp + fp + fn) > 0,
    )
    best = ends[int(np.argmax(f1))]
    threshold = float(sorted_probabilities[best])
    pred = np.where(probabilities >= threshold, 1, -1)
    return threshold, classification_metrics(y_true, pred)


def train(y, x_raw, x_test_raw):
    folds = _stratified_folds(y, n_folds=5)
    # The previous sweep showed little difference among small lambdas and
    # positive-weight multipliers near 0.5-1.0. Recheck that region with more
    # optimization steps because none of the 500-iteration fits converged.
    weight_scales = [0.5, 0.75, 1.0]
    regularizations = [1e-5, 1e-4]
    configs = [(lam, scale) for lam in regularizations for scale in weight_scales]
    oof_probabilities = {
        config: np.empty(len(y), dtype=np.float32) for config in configs
    }
    convergence = {config: [] for config in configs}

    # Preprocess once per fold; all 30 hyperparameter combinations reuse it.
    for fold_number, (train_idx, validation_idx) in enumerate(folds, start=1):
        tx_train, tx_validation = preprocess(x_raw[train_idx], x_raw[validation_idx])
        print(f"Fitting CV fold {fold_number}/5")
        for regularization, weight_scale in configs:
            config = (regularization, weight_scale)
            w, info = _fit(y[train_idx], tx_train, regularization, weight_scale)
            oof_probabilities[config][validation_idx] = _sigmoid(
                tx_validation @ w
            ).astype(np.float32)
            convergence[config].append(info)

    best_config = None
    best_threshold = None
    best_metrics = None
    for config in configs:
        threshold, metrics = _best_f1_threshold(y, oof_probabilities[config])
        convergence_infos = convergence[config]
        converged_folds = sum(info["converged"] for info in convergence_infos)
        mean_iterations = np.mean([info["iterations"] for info in convergence_infos])
        mean_start_loss = np.mean(
            [info["loss_history"][0] for info in convergence_infos]
        )
        mean_end_loss = np.mean(
            [info["loss_history"][-1] for info in convergence_infos]
        )
        print(
            f"lambda={config[0]:g}, positive-weight multiplier={config[1]:g}, "
            f"threshold={threshold:.5f}: accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}; converged folds={converged_folds}/5, "
            f"mean iterations={mean_iterations:.1f}, "
            f"mean loss={mean_start_loss:.5f}->{mean_end_loss:.5f}"
        )
        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_config = config
            best_threshold = threshold
            best_metrics = metrics

    regularization, weight_scale = best_config
    print(
        f"Selected lambda={regularization:g}, positive-weight multiplier="
        f"{weight_scale:g}, threshold={best_threshold:.5f}"
    )

    # Refit preprocessing and the selected model using all training examples.
    tx_full, tx_test = preprocess(x_raw, x_test_raw)
    w, final_info = _fit(y, tx_full, regularization, weight_scale)
    print(
        f"Final fit: iterations={final_info['iterations']}, "
        f"converged={final_info['converged']}, "
        f"final gradient norm={final_info['gradient_norm']:.3g}, "
        f"training loss={final_info['loss_history'][-1]:.5f}"
    )
    return w, best_threshold, tx_test


def predict(tx, w, threshold):
    threshold = np.clip(threshold, 1e-12, 1.0 - 1e-12)
    score_cutoff = np.log(threshold / (1.0 - threshold))
    return np.where(tx @ w >= score_cutoff, 1, -1)


def main():
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_PATH)
    with open(f"{DATA_PATH}/x_train.csv", newline="", encoding="utf-8-sig") as csvfile:
        source_names = next(csv.reader(csvfile))[1:]
    x_train, feature_names = select_and_clean_features(x_train, source_names)
    x_test, _ = select_and_clean_features(x_test, source_names)
    x_train = engineer_features(x_train, feature_names)
    x_test = engineer_features(x_test, feature_names)
    w, threshold, tx_test = train(y_train, x_train, x_test)
    create_csv_submission(
        test_ids,
        predict(tx_test, w, threshold),
        SUBMISSION_PATH,
    )


if __name__ == "__main__":
    main()
