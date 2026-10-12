"""Five-fold OOF error analysis for Experiment 2 Random Forest and Linear Regression.

Residual convention: positive = prediction is too high (overshoot), negative
= prediction is too low (undershoot).  A smaller tree count is sufficient for
diagnostics; use --trees 500 to exactly match the final Random Forest size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from train_target_encoded_models import (
    DATA_DIRECTORY,
    RANDOM_STATE,
    cross_fitted_encode,
    model_pipeline,
)


def error_summary(data: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    grouped = data.groupby(by, dropna=False) if by else [("overall", data)]
    rows = []
    for name, group in grouped:
        row = {
            "segment": name if isinstance(name, str) else " | ".join(map(str, name)),
            "rows": len(group),
        }
        for model in ("rf", "linear"):
            residual = group[f"{model}_residual"]
            absolute = residual.abs()
            row[f"{model}_mean_residual"] = residual.mean()
            row[f"{model}_mae"] = absolute.mean()
            row[f"{model}_overshoot_rate"] = (residual > 0).mean()
        row["rf_better_rate"] = (group.rf_absolute_error < group.linear_absolute_error).mean()
        row["mean_rf_minus_linear_prediction"] = (group.rf_prediction - group.linear_prediction).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=80, help="trees per validation-fold RF")
    args = parser.parse_args()

    directory = Path(__file__).resolve().parent
    output = directory / "generated"
    output.mkdir(exist_ok=True)
    train = pd.read_csv(DATA_DIRECTORY / "train_preprocessed.csv")
    y = train["price"].astype(float)
    raw_x = train.drop(columns=["id", "price"])
    splitter = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf_oof = np.zeros(len(train))
    linear_oof = np.zeros(len(train))

    for fold, (fit_indices, valid_indices) in enumerate(splitter.split(raw_x), start=1):
        print(f"Fitting fold {fold}/5 ...", flush=True)
        x_fit, x_valid = cross_fitted_encode(
            raw_x.iloc[fit_indices], y.iloc[fit_indices], raw_x.iloc[valid_indices]
        )
        rf = model_pipeline(
            x_fit,
            RandomForestRegressor(
                n_estimators=args.trees, min_samples_leaf=2, max_features=0.8,
                n_jobs=-1, random_state=RANDOM_STATE + fold,
            ),
            scale_numeric=False,
        )
        linear = model_pipeline(x_fit, LinearRegression(), scale_numeric=True)
        rf.fit(x_fit, y.iloc[fit_indices])
        linear.fit(x_fit, y.iloc[fit_indices])
        rf_oof[valid_indices] = rf.predict(x_valid)
        linear_oof[valid_indices] = linear.predict(x_valid)

    errors = train[["id", "price", "flight_class", "stops", "duration_was_missing"]].copy()
    errors["price_band"] = pd.qcut(errors["price"], q=5, duplicates="drop")
    errors["rf_prediction"] = rf_oof
    errors["linear_prediction"] = linear_oof
    for model in ("rf", "linear"):
        errors[f"{model}_residual"] = errors[f"{model}_prediction"] - errors["price"]
        errors[f"{model}_absolute_error"] = errors[f"{model}_residual"].abs()
    errors["better_model"] = np.where(
        errors.rf_absolute_error < errors.linear_absolute_error, "random_forest", "linear_regression"
    )
    errors["prediction_disagreement"] = errors.rf_prediction - errors.linear_prediction
    errors.to_csv(output / "oof_error_rows.csv", index=False)

    summaries = {
        "overall": error_summary(errors),
        "by_class_stops": error_summary(errors, ["flight_class", "stops"]),
        "by_price_band": error_summary(errors, ["price_band"]),
        "by_missing_duration": error_summary(errors, ["duration_was_missing"]),
    }
    for name, table in summaries.items():
        table.to_csv(output / f"error_summary_{name}.csv", index=False)

    overall = summaries["overall"].iloc[0]
    print("\nOOF error analysis")
    print(f"RF mean residual: {overall.rf_mean_residual:,.2f}; overshoot rate: {overall.rf_overshoot_rate:.1%}")
    print(f"LR mean residual: {overall.linear_mean_residual:,.2f}; overshoot rate: {overall.linear_overshoot_rate:.1%}")
    print(f"RF has lower absolute error on: {overall.rf_better_rate:.1%} of rows")
    print(f"Saved detailed rows and segment summaries to {output}")


if __name__ == "__main__":
    main()
