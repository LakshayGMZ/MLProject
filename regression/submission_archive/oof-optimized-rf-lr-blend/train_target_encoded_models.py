"""Train Random Forest and Linear Regression with 5-fold target encodings.

Run after preprocess.py:
    .venv/bin/python regression/experiment_2/train_target_encoded_models.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DIRECTORY = Path(__file__).resolve().parent
REGRESSION = DIRECTORY.parents[1]
DATA_DIRECTORY = REGRESSION / "experiment_2" / "data"
OUTPUT_DIRECTORY = DIRECTORY / "generated"
FOLDS = 5
RANDOM_STATE = 42
SMOOTHING = 10.0
MIN_HIERARCHY_COUNT = 3

# Ordered from most specific to broadest. Every group contains travel class and
# stops where relevant, so a one-stop Business flight is not mixed with a
# zero-stop Economy flight.
GROUPS = [
    ("flight_class_stops", ["flight", "flight_class", "stops"]),
    ("airline_route_class_stops", ["airline", "route", "flight_class", "stops"]),
    ("route_class_stops", ["route", "flight_class", "stops"]),
    ("airline_class_stops", ["airline", "flight_class", "stops"]),
    ("airline_class", ["airline", "flight_class"]),
    ("class_stops", ["flight_class", "stops"]),
]


def group_key(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    return data[columns].fillna("__MISSING__").astype(str).agg("\x1f".join, axis=1)


def mapping(
    features: pd.DataFrame, target: pd.Series, columns: list[str], global_mean: float
) -> pd.DataFrame:
    table = pd.DataFrame({"key": group_key(features, columns), "target": target.to_numpy()})
    stats = table.groupby("key")["target"].agg(["mean", "count"])
    stats["smoothed_mean"] = (
        stats["count"] * stats["mean"] + SMOOTHING * global_mean
    ) / (stats["count"] + SMOOTHING)
    return stats


def encoded_from_fit(
    fit_x: pd.DataFrame, fit_y: pd.Series, transform_x: pd.DataFrame
) -> pd.DataFrame:
    """Map target statistics from fit_x only onto transform_x."""
    result = transform_x.copy()
    global_mean = float(fit_y.mean())
    hierarchy = pd.Series(global_mean, index=transform_x.index, dtype=float)

    # Individual encodings are features in their own right.
    group_stats: list[tuple[list[str], pd.DataFrame]] = []
    for name, columns in GROUPS:
        stats = mapping(fit_x, fit_y, columns, global_mean)
        keys = group_key(transform_x, columns)
        result[f"te_{name}"] = keys.map(stats["smoothed_mean"]).fillna(global_mean)
        group_stats.append((columns, stats))

    # Hierarchical prior: broad groups are overwritten only by supported,
    # more-specific groups. This is more stable than trusting a one-row mean.
    for columns, stats in reversed(group_stats):
        keys = group_key(transform_x, columns)
        candidate = keys.map(stats["smoothed_mean"])
        count = keys.map(stats["count"]).fillna(0)
        hierarchy.loc[(count >= MIN_HIERARCHY_COUNT) & candidate.notna()] = candidate.loc[
            (count >= MIN_HIERARCHY_COUNT) & candidate.notna()
        ]
    result["te_hierarchical_price"] = hierarchy
    return result


def cross_fitted_encode(
    train_x: pd.DataFrame, train_y: pd.Series, other_x: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OOF-encode train rows and use full train mappings for other_x."""
    splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    encoded_train = train_x.copy()
    encoded_columns = [f"te_{name}" for name, _ in GROUPS] + ["te_hierarchical_price"]
    for column in encoded_columns:
        encoded_train[column] = np.nan

    for fit_indices, valid_indices in splitter.split(train_x):
        fold_encoded = encoded_from_fit(
            train_x.iloc[fit_indices], train_y.iloc[fit_indices], train_x.iloc[valid_indices]
        )
        encoded_train.loc[train_x.index[valid_indices], encoded_columns] = fold_encoded[
            encoded_columns
        ].to_numpy()

    encoded_other = encoded_from_fit(train_x, train_y, other_x)
    return encoded_train, encoded_other


def model_pipeline(features: pd.DataFrame, model: object, scale_numeric: bool) -> Pipeline:
    categorical = features.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = features.columns.difference(categorical).tolist()
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    return Pipeline([
        ("preprocessor", ColumnTransformer([
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
            ("numeric", Pipeline(numeric_steps), numeric),
        ])),
        ("regressor", model),
    ])


def report(name: str, y: pd.Series, prediction: np.ndarray) -> dict[str, float | str]:
    result: dict[str, float | str] = {
        "model": name,
        "rmse": root_mean_squared_error(y, prediction),
        "mae": mean_absolute_error(y, prediction),
        "r2": r2_score(y, prediction),
    }
    print(f"{name}: RMSE={result['rmse']:,.2f}, MAE={result['mae']:,.2f}, R²={result['r2']:.4f}")
    return result


def optimise_weights(prediction_matrix: np.ndarray, target: pd.Series) -> np.ndarray:
    """Find non-negative weights summing to one from OOF predictions only."""
    start = np.full(prediction_matrix.shape[1], 1 / prediction_matrix.shape[1])
    scale = target.std()
    result = minimize(
        lambda weights: np.mean(((target.to_numpy() - prediction_matrix @ weights) / scale) ** 2),
        start,
        method="SLSQP",
        bounds=[(0, 1)] * len(start),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1},
    )
    return result.x if result.success else start


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIRECTORY / "train_preprocessed.csv")
    test = pd.read_csv(DATA_DIRECTORY / "test_preprocessed.csv")
    y = train.pop("price").astype(float)
    x = train.drop(columns="id")
    x_test = test.drop(columns="id")

    # These are the final training/test features. Train target encodings are
    # out-of-fold; test encodings are calculated from all training targets.
    x_encoded, x_test_encoded = cross_fitted_encode(x, y, x_test)
    x_encoded.to_csv(OUTPUT_DIRECTORY / "train_target_encoded_oof.csv", index=False)
    x_test_encoded.to_csv(OUTPUT_DIRECTORY / "test_target_encoded.csv", index=False)

    splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models = {
        "random_forest": (RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=0.8,
            n_jobs=-1, random_state=RANDOM_STATE
        ), False),
        "linear_regression": (LinearRegression(), True),
    }
    scores = []
    oof_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    for name, (estimator, scale_numeric) in models.items():
        oof = np.zeros(len(x_encoded))
        for fit_indices, valid_indices in splitter.split(x_encoded):
            # Nested cross-fitting: the outer validation fold never contributes
            # target values to either its own encodings or model training.
            fold_train, fold_valid = cross_fitted_encode(
                x.iloc[fit_indices], y.iloc[fit_indices], x.iloc[valid_indices]
            )
            pipeline = model_pipeline(fold_train, estimator, scale_numeric)
            pipeline.fit(fold_train, y.iloc[fit_indices])
            oof[valid_indices] = pipeline.predict(fold_valid)
        oof_predictions[name] = oof
        scores.append(report(name, y, oof))

        final_model = model_pipeline(x_encoded, estimator, scale_numeric)
        final_model.fit(x_encoded, y)
        test_prediction = np.maximum(final_model.predict(x_test_encoded), 0)
        test_predictions[name] = test_prediction
        pd.DataFrame({"id": test["id"], "price": test_prediction}).to_csv(
            OUTPUT_DIRECTORY / f"submission_{name}_target_encoded.csv", index=False
        )

    model_names = list(oof_predictions)
    oof_matrix = np.column_stack([oof_predictions[name] for name in model_names])
    weights = optimise_weights(oof_matrix, y)
    ensemble_oof = oof_matrix @ weights
    ensemble_score = report("rf_linear_ensemble", y, ensemble_oof)
    ensemble_score["rf_weight"] = weights[model_names.index("random_forest")]
    ensemble_score["linear_weight"] = weights[model_names.index("linear_regression")]
    scores.append(ensemble_score)

    pd.DataFrame({"price": y, **oof_predictions}).to_csv(
        OUTPUT_DIRECTORY / "oof_model_predictions.csv", index=False
    )
    ensemble_test = np.column_stack([test_predictions[name] for name in model_names]) @ weights
    pd.DataFrame({"id": test["id"], "price": np.maximum(ensemble_test, 0)}).to_csv(
        DIRECTORY / "submission_rf_linear_ensemble.csv", index=False
    )
    score_table = pd.DataFrame(scores)
    score_table.loc[score_table.model != "rf_linear_ensemble", "rf_weight"] = weights[
        model_names.index("random_forest")
    ]
    score_table.loc[score_table.model != "rf_linear_ensemble", "linear_weight"] = weights[
        model_names.index("linear_regression")
    ]
    score_table.to_csv(OUTPUT_DIRECTORY / "five_fold_scores.csv", index=False)
    print(f"Saved outputs to {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
