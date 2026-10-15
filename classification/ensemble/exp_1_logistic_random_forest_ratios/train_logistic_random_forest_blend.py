from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT_DIR = Path(__file__).resolve().parent
CLASSIFICATION_DIR = EXPERIMENT_DIR.parents[1]
DATASET_DIR = CLASSIFICATION_DIR / "dataset"
TRAIN_PATH = DATASET_DIR / "vehicle_train_10000.csv"
TEST_PATH = DATASET_DIR / "vehicle_test.csv"
SAMPLE_SUBMISSION_PATH = DATASET_DIR / "vehicle_sample_submission.csv"
CV_RESULTS_PATH = EXPERIMENT_DIR / "grouped_cv_results.csv"
OUTPUT_PATH = EXPERIMENT_DIR / "vehicle_submission_logistic_rf_ratio_blend.csv"
TARGET_COLUMN = "vehicle_class"
ID_COLUMN = "vehicle_id"
RANDOM_STATE = 24
RF_WEIGHTS = (0.90, 0.75, 0.50, 0.25)


def make_ratio_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.copy()
    weight_tonnes = frame["vehicle_weight_kg"] / 1000
    displacement = frame["engine_displacement_l"]
    features["horsepower_per_tonne"] = frame["horsepower_hp"] / weight_tonnes
    features["torque_per_tonne"] = frame["torque_nm"] / weight_tonnes
    features["horsepower_per_liter"] = frame["horsepower_hp"] / displacement
    features["torque_per_liter"] = frame["torque_nm"] / displacement
    features["cylinders_per_liter"] = frame["engine_cylinders"] / displacement
    features["cargo_liters_per_tonne"] = frame["cargo_capacity_l"] / weight_tonnes
    features["battery_kwh_per_tonne"] = frame["battery_capacity_kwh"] / weight_tonnes
    features["wheelbase_to_length"] = frame["wheelbase_mm"] / frame["vehicle_length_mm"]
    features["width_to_length"] = frame["vehicle_width_mm"] / frame["vehicle_length_mm"]
    features["height_to_width"] = frame["vehicle_height_mm"] / frame["vehicle_width_mm"]
    features["highway_minus_city_efficiency"] = (
        frame["highway_fuel_efficiency_kmpl"] - frame["city_fuel_efficiency_kmpl"]
    )
    features["highway_to_city_efficiency"] = (
        frame["highway_fuel_efficiency_kmpl"] / frame["city_fuel_efficiency_kmpl"]
    )
    return features


def make_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_logistic_regression():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
    )


def aligned_probabilities(model, features, classes):
    probabilities = model.predict_proba(features)
    aligned = np.zeros((len(features), len(classes)))
    for index, label in enumerate(model.classes_):
        aligned[:, np.flatnonzero(classes == label)[0]] = probabilities[:, index]
    return aligned


def estimate_test_weights(unique_train_features, test_features):
    combined = pd.concat([unique_train_features, test_features], ignore_index=True)
    domains = np.concatenate(
        [np.zeros(len(unique_train_features), dtype=int), np.ones(len(test_features), dtype=int)]
    )
    model = ExtraTreesClassifier(
        n_estimators=160,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    probabilities = np.asarray(
        cross_val_predict(
            model,
            combined,
            domains,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            method="predict_proba",
            n_jobs=1,
        )
    )[:, 1]
    p_test = np.clip(probabilities[: len(unique_train_features)], 0.01, 0.99)
    weights = p_test / (1 - p_test) * len(unique_train_features) / len(test_features)
    weights = np.clip(weights, 0.2, 20.0)
    weights /= weights.mean()
    return weights, roc_auc_score(domains, probabilities)


def add_score(rows, name, probabilities, target, weights, classes):
    predictions = classes[np.argmax(probabilities, axis=1)]
    rows.append(
        {
            "candidate": name,
            "accuracy": accuracy_score(target, predictions),
            "test_weighted_accuracy": accuracy_score(
                target,
                predictions,
                sample_weight=weights,
            ),
            "test_weighted_macro_f1": f1_score(
                target,
                predictions,
                labels=classes,
                average="macro",
                sample_weight=weights,
            ),
        }
    )


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    feature_columns = [
        column
        for column in train.select_dtypes(include="number").columns
        if column not in {ID_COLUMN, TARGET_COLUMN}
    ]
    raw_train = cast(pd.DataFrame, train[feature_columns].reset_index(drop=True))
    raw_test = cast(pd.DataFrame, test[feature_columns].reset_index(drop=True))
    feature_sets = {
        "raw": raw_train,
        "ratios": make_ratio_features(raw_train),
    }
    test_feature_sets = {
        "raw": raw_test,
        "ratios": make_ratio_features(raw_test),
    }
    target = train[TARGET_COLUMN].to_numpy()
    classes = np.sort(np.unique(target))
    groups = pd.MultiIndex.from_frame(raw_train).factorize()[0]
    _, unique_indices = np.unique(groups, return_index=True)
    unique_indices = np.sort(unique_indices)
    unique_mask = np.zeros(len(train), dtype=bool)
    unique_mask[unique_indices] = True
    unique_target = target[unique_indices]
    weights, domain_auc = estimate_test_weights(
        feature_sets["ratios"].iloc[unique_indices],
        test_feature_sets["ratios"],
    )
    print(f"Unique feature rows: {len(unique_indices):,}; domain OOF AUC={domain_auc:.4f}")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = {
        feature_name: {
            model_name: np.zeros((len(unique_indices), len(classes)))
            for model_name in ("random_forest", "logistic_regression")
        }
        for feature_name in feature_sets
    }
    for fold, (fit_indices, valid_indices) in enumerate(
        cv.split(raw_train, target, groups=groups),
        start=1,
    ):
        valid_unique = valid_indices[unique_mask[valid_indices]]
        positions = np.searchsorted(unique_indices, valid_unique)
        print(f"Fitting validation fold {fold}/5")
        for feature_name, features in feature_sets.items():
            rf = make_random_forest().fit(features.iloc[fit_indices], target[fit_indices])
            lr = make_logistic_regression().fit(features.iloc[fit_indices], target[fit_indices])
            oof[feature_name]["random_forest"][positions] = aligned_probabilities(
                rf,
                features.iloc[valid_unique],
                classes,
            )
            oof[feature_name]["logistic_regression"][positions] = aligned_probabilities(
                lr,
                features.iloc[valid_unique],
                classes,
            )

    score_rows = []
    for feature_name, model_probabilities in oof.items():
        rf_probabilities = model_probabilities["random_forest"]
        lr_probabilities = model_probabilities["logistic_regression"]
        add_score(
            score_rows,
            f"{feature_name}_random_forest",
            rf_probabilities,
            unique_target,
            weights,
            classes,
        )
        add_score(
            score_rows,
            f"{feature_name}_logistic_regression",
            lr_probabilities,
            unique_target,
            weights,
            classes,
        )
        for rf_weight in RF_WEIGHTS:
            blended = rf_weight * rf_probabilities + (1 - rf_weight) * lr_probabilities
            add_score(
                score_rows,
                f"{feature_name}_rf_{rf_weight:.2f}_logistic_{1-rf_weight:.2f}",
                blended,
                unique_target,
                weights,
                classes,
            )

    validation = pd.DataFrame(score_rows).sort_values(
        "test_weighted_accuracy",
        ascending=False,
    )
    validation.to_csv(CV_RESULTS_PATH, index=False)
    print("Duplicate-grouped validation results:")
    print(validation.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    ratio_blends = validation[
        validation["candidate"].str.startswith("ratios_rf_")
    ]
    best_blend = ratio_blends.iloc[0]
    rf_weight = float(best_blend["candidate"].split("_rf_")[1].split("_")[0])
    final_rf = make_random_forest().fit(feature_sets["ratios"], target)
    final_lr = make_logistic_regression().fit(feature_sets["ratios"], target)
    rf_test_probabilities = aligned_probabilities(
        final_rf,
        test_feature_sets["ratios"],
        classes,
    )
    lr_test_probabilities = aligned_probabilities(
        final_lr,
        test_feature_sets["ratios"],
        classes,
    )
    predictions = classes[
        np.argmax(rf_weight * rf_test_probabilities + (1 - rf_weight) * lr_test_probabilities, axis=1)
    ]

    prediction_by_id = pd.Series(predictions, index=test[ID_COLUMN].to_numpy())
    sample_ids = pd.Series(sample[ID_COLUMN].to_numpy())
    submission_classes = sample_ids.map(prediction_by_id)
    if submission_classes.isna().any() or len(submission_classes) != len(test):
        raise ValueError("Submission IDs do not align with test predictions.")
    submission = pd.DataFrame(
        {
            ID_COLUMN: sample_ids,
            TARGET_COLUMN: submission_classes,
        }
    )
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Selected ratio blend: {best_blend['candidate']}")
    print(f"Wrote {OUTPUT_PATH}; classes={submission_classes.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
