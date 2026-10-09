"""Recreate the original Random Forest + Extra Trees submission."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

REGRESSION = Path(__file__).resolve().parents[2]
ARCHIVE = Path(__file__).resolve().parent


def features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.drop(columns="id", errors="ignore").copy()
    x["route"] = x["source_city"].astype(str) + "__" + x["destination_city"].astype(str)
    x["flight_prefix"] = x["flight"].str.split().str[0]
    x["duration_missing"] = x["duration"].isna().astype("int8")
    legs = x["stops"].map({"zero": 1, "one": 2, "two_or_more": 3})
    x["duration_per_leg"] = x["duration"] / legs
    return x


def model(x: pd.DataFrame, estimator: object) -> Pipeline:
    categorical = x.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = x.columns.difference(categorical).tolist()
    return Pipeline([
        ("preprocessor", ColumnTransformer([
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
        ])),
        ("regressor", estimator),
    ])


def main() -> None:
    train = pd.read_csv(REGRESSION / "dataset" / "train.csv")
    test = pd.read_csv(REGRESSION / "dataset" / "test.csv")
    x = features(train.drop(columns="price"))
    y = train["price"].astype(float)
    x_test = features(test)
    forest = model(x, RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, max_features=1.0,
        n_jobs=-1, random_state=42,
    ))
    extra = model(x, ExtraTreesRegressor(
        n_estimators=300, min_samples_leaf=3, max_features=1.0,
        n_jobs=-1, random_state=43,
    ))
    forest.fit(x, y)
    extra.fit(x, y)
    prediction = np.maximum(0, 0.5 * forest.predict(x_test) + 0.5 * extra.predict(x_test))
    recreated = pd.DataFrame({"id": test["id"], "price": prediction})
    reference = pd.read_csv(ARCHIVE / "submission_random_forest.old.csv")
    recreated.to_csv(ARCHIVE / "submission_random_forest.csv", index=False)
    assert recreated["id"].equals(reference["id"])
    assert np.allclose(recreated["price"], reference["price"], rtol=0, atol=1e-8)
    print("Regenerated submission; exact match to .old.csv verified.")


if __name__ == "__main__":
    main()
