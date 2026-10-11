"""Create a conservative blend from the existing Experiment 2 submissions."""

from pathlib import Path
import argparse

import pandas as pd


ARCHIVE = Path(__file__).resolve().parent
OUTPUT = ARCHIVE / "generated"
DEFAULT_RF_WEIGHT = 0.90


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf-weight", type=float, default=DEFAULT_RF_WEIGHT)
    parser.add_argument("--output", default="submission_rf60_lr40_ensemble.csv")
    args = parser.parse_args()
    if not 0 <= args.rf_weight <= 1:
        raise ValueError("--rf-weight must be between 0 and 1.")
    linear_weight = 1 - args.rf_weight

    rf = pd.read_csv(OUTPUT / "submission_random_forest_target_encoded.csv")
    linear = pd.read_csv(OUTPUT / "submission_linear_regression_target_encoded.csv")
    if not rf["id"].equals(linear["id"]):
        raise ValueError("Random Forest and Linear Regression submission IDs differ.")

    ensemble = rf[["id"]].copy()
    ensemble["price"] = args.rf_weight * rf["price"] + linear_weight * linear["price"]
    output = ARCHIVE / args.output
    ensemble.to_csv(output, index=False)
    print(f"Saved {output}")
    print(f"Weights: Random Forest={args.rf_weight:.0%}, Linear Regression={linear_weight:.0%}")


if __name__ == "__main__":
    main()
