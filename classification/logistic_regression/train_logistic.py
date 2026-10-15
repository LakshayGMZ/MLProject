from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "classification" / "dataset"
TARGET = "vehicle_class"
ID = "vehicle_id"


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


def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=3000),
    )


def main():
    train = pd.read_csv(DATA_DIR / "vehicle_train_10000.csv")
    test = pd.read_csv(DATA_DIR / "vehicle_test.csv")
    sample = pd.read_csv(DATA_DIR / "vehicle_sample_submission.csv")

    train_part, validation_part = train_test_split(
        train,
        test_size=0.2,
        random_state=42,
        stratify=train[TARGET],
    )
    validation_model = make_model()
    validation_model.fit(
        engineer_features(train_part),
        train_part[TARGET],
    )
    validation_predictions = validation_model.predict(engineer_features(validation_part))
    print(
        "Stratified validation accuracy:",
        f"{accuracy_score(validation_part[TARGET], validation_predictions):.4f}",
    )

    model = make_model()
    model.fit(engineer_features(train), train[TARGET])
    predictions = pd.Series(model.predict(engineer_features(test)), index=test[ID])

    submission = sample.copy()
    submission[TARGET] = submission[ID].map(predictions)
    if submission[TARGET].isna().any():
        raise ValueError("Some sample submission IDs have no matching test prediction")

    output_path = ROOT / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} predictions to {output_path}")


if __name__ == "__main__":
    main()