from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold


MODEL_DIR = Path(__file__).resolve().parent
CLASSIFICATION_DIR = MODEL_DIR.parents[1]
DATASET_DIR = CLASSIFICATION_DIR / "dataset"
TRAIN_PATH = DATASET_DIR / "vehicle_train_10000.csv"
TEST_PATH = DATASET_DIR / "vehicle_test.csv"
SAMPLE_SUBMISSION_PATH = DATASET_DIR / "vehicle_sample_submission.csv"
OUTPUT_PATH = MODEL_DIR / "vehicle_submission_random_forest.csv"
TARGET_COLUMN = "vehicle_class"
ID_COLUMN = "vehicle_id"
RANDOM_STATE = 24


def make_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
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
    X = cast(pd.DataFrame, train[feature_columns])
    y = train[TARGET_COLUMN].to_numpy()
    X_test = cast(pd.DataFrame, test[feature_columns])

    groups = pd.MultiIndex.from_frame(X).factorize()[0]
    _, unique_indices = np.unique(groups, return_index=True)
    unique_indices = np.sort(unique_indices)
    unique_mask = np.zeros(len(train), dtype=bool)
    unique_mask[unique_indices] = True
    oof_predictions = np.empty(len(unique_indices), dtype=y.dtype)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for fold, (fit_indices, valid_indices) in enumerate(cv.split(X, y, groups=groups), start=1):
        valid_unique = valid_indices[unique_mask[valid_indices]]
        model = make_model().fit(X.iloc[fit_indices], y[fit_indices])
        predictions = model.predict(X.iloc[valid_unique])
        oof_predictions[np.searchsorted(unique_indices, valid_unique)] = predictions
        print(f"Validation fold {fold}/5 complete")

    unique_target = y[unique_indices]
    print(f"Training rows: {len(train):,}; unique feature vectors: {len(unique_indices):,}")
    print(f"Duplicate-grouped OOF accuracy: {accuracy_score(unique_target, oof_predictions):.4f}")
    print(
        "Duplicate-grouped OOF macro F1: "
        f"{f1_score(unique_target, oof_predictions, average='macro'):.4f}"
    )

    final_model = make_model().fit(X, y)
    prediction_by_id = pd.Series(
        final_model.predict(X_test),
        index=test[ID_COLUMN].to_numpy(),
    )
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
    print(f"Wrote {OUTPUT_PATH}; class counts={submission_classes.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
