"""Recreate the tuned ordinary Linear Regression submission."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

REGRESSION = Path(__file__).resolve().parents[2]
ARCHIVE = Path(__file__).resolve().parent
SEED = 42


def features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.drop(columns="id", errors="ignore").copy()
    x["route"] = x["source_city"].astype(str) + "__" + x["destination_city"].astype(str)
    x["flight_prefix"] = x["flight"].str.split().str[0]
    x["duration_missing"] = x["duration"].isna().astype("int8")
    legs = x["stops"].map({"zero": 1, "one": 2, "two_or_more": 3})
    x["duration_per_leg"] = x["duration"] / legs
    return x


def preprocessor(x: pd.DataFrame, degree: int) -> ColumnTransformer:
    categorical = x.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = x.columns.difference(categorical).tolist()
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ]
    if degree == 2:
        numeric_steps.append(("polynomial", PolynomialFeatures(2, include_bias=False)))
    return ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
        ("numeric", Pipeline(numeric_steps), numeric),
    ])


def main() -> None:
    train = pd.read_csv(REGRESSION / "dataset" / "train.csv")
    test = pd.read_csv(REGRESSION / "dataset" / "test.csv")
    x = features(train.drop(columns="price"))
    y = train["price"].astype(float)
    x_test = features(test)
    estimator = Pipeline([
        ("preprocessor", preprocessor(x, 1)),
        ("regressor", LinearRegression()),
    ])
    search = GridSearchCV(
        estimator,
        {"preprocessor": [preprocessor(x, 1), preprocessor(x, 2)]},
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=2, shuffle=True, random_state=SEED + 1),
        n_jobs=-1,
    ).fit(x, y)
    prediction = np.maximum(search.predict(x_test), 0)
    recreated = pd.DataFrame({"id": test["id"], "price": prediction})
    reference = pd.read_csv(ARCHIVE / "submission_linear.old.csv")
    recreated.to_csv(ARCHIVE / "submission_linear.csv", index=False)
    assert recreated["id"].equals(reference["id"])
    max_difference = np.abs(recreated["price"] - reference["price"]).max()
    assert max_difference < 15.0  # small solver/version drift; model recipe is unchanged
    print("Regenerated submission; max price delta vs .old.csv:", round(float(max_difference), 6))


if __name__ == "__main__":
    main()
