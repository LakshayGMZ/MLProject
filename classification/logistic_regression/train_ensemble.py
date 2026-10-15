"""
Ensemble: Logistic Regression + Random Forest for Vehicle Classification.

Generates multiple submissions with different blending weights for the
probability outputs of both models and uploads each to Kaggle.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "classification" / "dataset"
TARGET = "vehicle_class"
ID = "vehicle_id"

# ── Blending ratios (RF weight, LR weight) ──────────────────────────────────
BLEND_CONFIGS = [
    {"rf_weight": 0.3, "lr_weight": 0.7, "label": "RF30_LR70"},
    {"rf_weight": 0.5, "lr_weight": 0.5, "label": "RF50_LR50"},
    {"rf_weight": 0.7, "lr_weight": 0.3, "label": "RF70_LR30"},
    {"rf_weight": 0.9, "lr_weight": 0.1, "label": "RF90_LR10"},
]


# ── Feature engineering (reused from train_logistic.py) ──────────────────────
def engineer_features(frame):
    features = frame.drop(columns=[ID, TARGET], errors="ignore").copy()
    features["power_to_weight"] = frame["horsepower_hp"] / frame["vehicle_weight_kg"]
    features["torque_to_weight"] = frame["torque_nm"] / frame["vehicle_weight_kg"]
    features["length_to_width"] = frame["vehicle_length_mm"] / frame["vehicle_width_mm"]
    features["height_to_width"] = frame["vehicle_height_mm"] / frame["vehicle_width_mm"]
    features["fuel_efficiency_gap"] = (
        frame["highway_fuel_efficiency_kmpl"] - frame["city_fuel_efficiency_kmpl"]
    )
    features["braking_per_weight"] = frame["braking_distance_m"] / frame["vehicle_weight_kg"]
    features["engine_power_density"] = frame["horsepower_hp"] / frame["engine_displacement_l"]
    return features


# ── Model factories ──────────────────────────────────────────────────────────
def make_lr_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=3000),
    )


def make_rf_model():
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


# ── Blended prediction via soft voting ───────────────────────────────────────
def blend_predict(lr_proba, rf_proba, lr_weight, rf_weight):
    """Weighted average of class probabilities, then argmax."""
    blended = lr_weight * lr_proba + rf_weight * rf_proba
    return np.argmax(blended, axis=1)


def main():
    # ── Load data ────────────────────────────────────────────────────────────
    train = pd.read_csv(DATA_DIR / "vehicle_train_10000.csv")
    test = pd.read_csv(DATA_DIR / "vehicle_test.csv")
    sample = pd.read_csv(DATA_DIR / "vehicle_sample_submission.csv")

    train_part, val_part = train_test_split(
        train, test_size=0.2, random_state=42, stratify=train[TARGET],
    )

    X_train = engineer_features(train_part)
    y_train = train_part[TARGET]
    X_val = engineer_features(val_part)
    y_val = val_part[TARGET]
    X_full = engineer_features(train)
    y_full = train[TARGET]
    X_test = engineer_features(test)

    # ── Train on split for validation ────────────────────────────────────────
    lr_val = make_lr_model()
    lr_val.fit(X_train, y_train)
    lr_val_acc = accuracy_score(y_val, lr_val.predict(X_val))
    print(f"Logistic Regression  validation accuracy: {lr_val_acc:.4f}")

    rf_val = make_rf_model()
    rf_val.fit(X_train, y_train)
    rf_val_acc = accuracy_score(y_val, rf_val.predict(X_val))
    print(f"Random Forest        validation accuracy: {rf_val_acc:.4f}")

    lr_val_proba = lr_val.predict_proba(X_val)
    rf_val_proba = rf_val.predict_proba(X_val)

    print("\n── Ensemble validation results ─────────────────────────────")
    for cfg in BLEND_CONFIGS:
        preds = blend_predict(lr_val_proba, rf_val_proba, cfg["lr_weight"], cfg["rf_weight"])
        acc = accuracy_score(y_val, preds)
        print(f"  {cfg['label']:>12s}  accuracy: {acc:.4f}")

    # ── Retrain on full training set for submission ──────────────────────────
    print("\n── Retraining on full training data ───────────────────────")
    lr_full = make_lr_model()
    lr_full.fit(X_full, y_full)

    rf_full = make_rf_model()
    rf_full.fit(X_full, y_full)

    lr_test_proba = lr_full.predict_proba(X_test)
    rf_test_proba = rf_full.predict_proba(X_test)

    # ── Generate & write submission files ────────────────────────────────────
    submission_dir = ROOT / "submissions"
    submission_dir.mkdir(exist_ok=True)

    for cfg in BLEND_CONFIGS:
        test_preds = blend_predict(lr_test_proba, rf_test_proba, cfg["lr_weight"], cfg["rf_weight"])
        predictions = pd.Series(test_preds, index=test[ID])

        sub = sample.copy()
        sub[TARGET] = sub[ID].map(predictions)
        if sub[TARGET].isna().any():
            raise ValueError("Some sample submission IDs have no matching test prediction")

        filename = f"submission_{cfg['label']}.csv"
        output_path = submission_dir / filename
        sub.to_csv(output_path, index=False)
        print(f"  Wrote {len(sub)} predictions → {output_path}")

    print("\nDone. Submission files are in:", submission_dir)


if __name__ == "__main__":
    main()

