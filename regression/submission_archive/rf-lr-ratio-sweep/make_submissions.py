"""Generate RF/LR blends for the requested Linear Regression percentages."""
from pathlib import Path

import numpy as np
import pandas as pd

REGRESSION = Path(__file__).resolve().parents[2]
COMPONENTS = (
    REGRESSION
    / "submission_archive"
    / "oof-optimized-rf-lr-blend"
    / "generated"
)
OUTPUT = Path(__file__).resolve().parent
LR_PERCENTAGES = [8.0, 9.5, 10.0, 10.5, 11.0, 11.5]


def main() -> None:
    rf = pd.read_csv(COMPONENTS / "submission_random_forest_target_encoded.csv")
    linear = pd.read_csv(COMPONENTS / "submission_linear_regression_target_encoded.csv")
    if not rf["id"].equals(linear["id"]):
        raise ValueError("RF and Linear Regression prediction IDs do not align.")

    for lr_percent in LR_PERCENTAGES:
        lr_weight = lr_percent / 100
        rf_weight = 1 - lr_weight
        submission = rf[["id"]].copy()
        submission["price"] = rf_weight * rf["price"] + lr_weight * linear["price"]
        if not np.isfinite(submission["price"]).all() or (submission["price"] < 0).any():
            raise ValueError(f"Invalid predictions for LR weight {lr_percent}%.")
        rf_percent = 100 - lr_percent
        filename = f"submission_rf_{str(rf_percent).replace('.', '_')}_lr_{str(lr_percent).replace('.', '_')}.csv"
        submission.to_csv(OUTPUT / filename, index=False)
        print(f"{filename}: RF {rf_weight:.1%} / LR {lr_percent:g}%")


if __name__ == "__main__":
    main()
