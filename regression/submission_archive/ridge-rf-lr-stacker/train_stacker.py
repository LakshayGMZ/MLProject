"""Fit a leakage-safe Ridge stacker on OOF base-model predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DIRECTORY = Path(__file__).resolve().parent
REGRESSION = DIRECTORY.parents[1]
DATA = REGRESSION / "experiment_2" / "data"
OUTPUT = DIRECTORY / "generated"
RANDOM_STATE = 42


def stack_features(metadata: pd.DataFrame, rf: pd.Series, linear: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rf_prediction": rf.to_numpy(),
            "linear_prediction": linear.to_numpy(),
            "prediction_gap": rf.to_numpy() - linear.to_numpy(),
            "flight_class": metadata["flight_class"].to_numpy(),
            "stops": metadata["stops"].to_numpy(),
            "duration_was_missing": metadata["duration_was_missing"].to_numpy(),
            "days_left": metadata["days_left"].to_numpy(),
            "duration": metadata["duration"].to_numpy(),
        }
    )


def pipeline(features: pd.DataFrame) -> Pipeline:
    categorical = features.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = features.columns.difference(categorical).tolist()
    return Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        (
                            "categorical",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            categorical,
                        ),
                        (
                            "numeric",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            numeric,
                        ),
                    ]
                ),
            ),
            ("regressor", Ridge()),
        ]
    )


def main() -> None:
    errors = pd.read_csv(OUTPUT / "oof_error_rows.csv")
    train = pd.read_csv(DATA / "train_preprocessed.csv")
    test = pd.read_csv(DATA / "test_preprocessed.csv")
    rf_test = pd.read_csv(OUTPUT / "submission_random_forest_target_encoded.csv")
    linear_test = pd.read_csv(OUTPUT / "submission_linear_regression_target_encoded.csv")

    if not (len(errors) == len(train) and rf_test["id"].equals(linear_test["id"])):
        raise ValueError("OOF rows or base submission IDs do not align.")

    y = errors["price"]
    x = stack_features(train, errors["rf_prediction"], errors["linear_prediction"])
    x_test = stack_features(test, rf_test["price"], linear_test["price"])
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline(x),
        {"regressor__alpha": [0.01, 0.1, 1, 10, 100, 1_000]},
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    )
    oof = cross_val_predict(search, x, y, cv=cv, n_jobs=1)
    print(f"Stacker OOF RMSE: {root_mean_squared_error(y, oof):,.2f}")
    print(f"Stacker OOF MAE: {mean_absolute_error(y, oof):,.2f}")
    print(f"Stacker OOF R²: {r2_score(y, oof):.4f}")

    search.fit(x, y)
    prediction = np.maximum(search.predict(x_test), 0)
    pd.DataFrame({"id": rf_test["id"], "price": prediction}).to_csv(
        DIRECTORY / "submission_ridge_stacker.csv", index=False
    )
    pd.DataFrame(
        [{"best_alpha": search.best_params_["regressor__alpha"], "best_cv_rmse": -search.best_score_}]
    ).to_csv(OUTPUT / "stacker_score.csv", index=False)
    print(f"Selected Ridge alpha: {search.best_params_['regressor__alpha']}")
    print(f"Saved {OUTPUT / 'submission_ridge_stacker.csv'}")


if __name__ == "__main__":
    main()
