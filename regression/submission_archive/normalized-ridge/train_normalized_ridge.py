"""Experiment 5: five-fold, fully standardized numeric Ridge regression."""
from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "experiment_2" / "data"
OUTPUT_DIR = Path(__file__).resolve().parent
SEED = 42


def make_pipeline(features: pd.DataFrame) -> Pipeline:
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
                                    # Fit only on the model-training rows in every CV fold.
                                    ("standardize", StandardScaler()),
                                ]
                            ),
                            numeric,
                        ),
                    ]
                ),
            ),
            ("ridge", Ridge()),
        ]
    )


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Non-target interaction and nonlinear features for the linear model."""
    x = data.copy()
    x["class_stops"] = x["flight_class"] + "__" + x["stops"]
    x["route_class_stops"] = x["route"] + "__" + x["flight_class"] + "__" + x["stops"]
    x["airline_class_stops"] = x["airline"] + "__" + x["flight_class"] + "__" + x["stops"]
    x["days_left_bucket"] = pd.cut(x["days_left"], [-1, 3, 7, 14, 30, np.inf], labels=["0_3", "4_7", "8_14", "15_30", "31_plus"]).astype(str)
    x["log_duration"] = np.log1p(x["duration"])
    x["log_days_left"] = np.log1p(x["days_left"])
    x["duration_squared"] = x["duration"] ** 2
    x["days_left_squared"] = x["days_left"] ** 2
    x["duration_days_left"] = x["duration"] * x["days_left"]
    return x


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train_preprocessed.csv")
    test = pd.read_csv(DATA_DIR / "test_preprocessed.csv")
    y = train["price"].astype(float)
    train_features = add_features(train.drop(columns=["id", "price"]))
    test_features = add_features(test.drop(columns="id"))

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=SEED + 1)
    search = GridSearchCV(
        make_pipeline(train_features),
        {"ridge__alpha": [0.01, 0.1, 1, 10, 30, 100, 300, 1_000, 3_000]},
        scoring="neg_root_mean_squared_error",
        cv=inner_cv,
        n_jobs=-1,
    )
    oof = cross_val_predict(search, train_features, y, cv=outer_cv, n_jobs=1)
    print(f"Normalized Ridge OOF RMSE: {root_mean_squared_error(y, oof):,.2f}")
    print(f"Normalized Ridge OOF MAE: {mean_absolute_error(y, oof):,.2f}")
    print(f"Normalized Ridge OOF R²: {r2_score(y, oof):.4f}")

    search.fit(train_features, y)
    prediction = np.maximum(search.predict(test_features), 0)
    pd.DataFrame({"id": test["id"], "price": prediction}).to_csv(
        OUTPUT_DIR / "submission_normalized_ridge.csv", index=False
    )
    pd.DataFrame(
        [{
            "best_alpha": search.best_params_["ridge__alpha"],
            "inner_cv_rmse": -search.best_score_,
            "outer_oof_rmse": root_mean_squared_error(y, oof),
            "outer_oof_mae": mean_absolute_error(y, oof),
            "outer_oof_r2": r2_score(y, oof),
        }]
    ).to_csv(OUTPUT_DIR / "normalized_ridge_score.csv", index=False)
    print(f"Selected alpha: {search.best_params_['ridge__alpha']}")
    print(f"Saved {OUTPUT_DIR / 'submission_normalized_ridge.csv'}")


if __name__ == "__main__":
    main()
