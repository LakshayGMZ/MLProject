"""Create cleaned train/test files for Experiment 2.

The lookup tables are fitted from observed values in train.csv only.  This is
safe for a future model: test labels are never used, and the same transform is
applied to both datasets.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "experiment_2" / "data"
MISSING_AIRLINE = "Unknown Airline"


def mode_or_default(values: pd.Series, default: str) -> str:
    mode = values.dropna().mode()
    return mode.iloc[0] if not mode.empty else default


def make_lookup(observed: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Median observed duration for each group, preserving NaN-free groups."""
    return observed.groupby(columns, dropna=False)["duration"].median()


def lookup_duration(frame: pd.DataFrame, lookup: pd.Series, columns: list[str]) -> pd.Series:
    if len(columns) == 1:
        return frame[columns[0]].map(lookup)
    keys = pd.MultiIndex.from_frame(frame[columns])
    return pd.Series(keys.map(lookup), index=frame.index)


def add_features_and_clean(
    frame: pd.DataFrame,
    prefix_to_airline: pd.Series,
    airline_fallback: str,
    duration_lookups: list[tuple[str, list[str], pd.Series]],
    global_duration: float,
) -> pd.DataFrame:
    data = frame.copy()
    data["flight_prefix"] = data["flight"].str.split().str[0]

    # Airline codes are deterministic in the observed training data.  Preserve
    # a source column so future analysis can distinguish observed from imputed.
    data["airline_was_missing"] = data["airline"].isna().astype("int8")
    inferred_airline = data["flight_prefix"].map(prefix_to_airline).fillna(airline_fallback)
    data["airline_imputation_source"] = "observed"
    data.loc[data["airline"].isna(), "airline_imputation_source"] = "flight_prefix"
    data["airline"] = data["airline"].fillna(inferred_airline)

    data["duration_was_missing"] = data["duration"].isna().astype("int8")
    data["duration_imputation_source"] = "observed"
    missing = data["duration"].isna()
    for source_name, group_columns, lookup in duration_lookups:
        candidates = lookup_duration(data, lookup, group_columns)
        use_candidate = missing & candidates.notna()
        data.loc[use_candidate, "duration"] = candidates.loc[use_candidate]
        data.loc[use_candidate, "duration_imputation_source"] = source_name
        missing = data["duration"].isna()
    data.loc[missing, "duration"] = global_duration
    data.loc[missing, "duration_imputation_source"] = "global_median"

    # Useful model-ready interaction features.  They do not use price.
    data["route"] = data["source_city"] + "__" + data["destination_city"]
    data["route_stops"] = data["route"] + "__" + data["stops"]
    data["route_class"] = data["route"] + "__" + data["flight_class"]
    data["time_pair"] = data["departure_time"] + "__" + data["arrival_time"]
    data["flight_class_pair"] = data["flight"] + "__" + data["flight_class"]
    data["days_left_class"] = data["days_left"].astype(str) + "__" + data["flight_class"]
    legs = data["stops"].map({"zero": 1, "one": 2, "two_or_more": 3})
    data["duration_per_leg"] = data["duration"] / legs
    return data


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(INPUT_DIR / "train.csv")
    test = pd.read_csv(INPUT_DIR / "test.csv")

    train_prefix = train.assign(flight_prefix=train["flight"].str.split().str[0])
    prefix_to_airline = train_prefix.groupby("flight_prefix")["airline"].agg(
        lambda values: mode_or_default(values, MISSING_AIRLINE)
    )
    airline_fallback = mode_or_default(train["airline"], MISSING_AIRLINE)

    observed_duration = train.dropna(subset=["duration"])
    hierarchy = [
        ("same_flight_route_stops", ["flight", "source_city", "destination_city", "stops"]),
        ("same_flight", ["flight"]),
        ("same_route_stops", ["source_city", "destination_city", "stops"]),
        ("same_route", ["source_city", "destination_city"]),
        ("same_stops", ["stops"]),
    ]
    duration_lookups = [
        (name, group_columns, make_lookup(observed_duration, group_columns))
        for name, group_columns in hierarchy
    ]
    global_duration = float(observed_duration["duration"].median())

    cleaned_train = add_features_and_clean(
        train, prefix_to_airline, airline_fallback, duration_lookups, global_duration
    )
    cleaned_test = add_features_and_clean(
        test, prefix_to_airline, airline_fallback, duration_lookups, global_duration
    )
    cleaned_train.to_csv(OUTPUT_DIR / "train_preprocessed.csv", index=False)
    cleaned_test.to_csv(OUTPUT_DIR / "test_preprocessed.csv", index=False)

    summary = pd.DataFrame(
        {
            "dataset": ["train", "test"],
            "rows": [len(cleaned_train), len(cleaned_test)],
            "airline_missing_before": [train.airline.isna().sum(), test.airline.isna().sum()],
            "duration_missing_before": [train.duration.isna().sum(), test.duration.isna().sum()],
            "airline_missing_after": [cleaned_train.airline.isna().sum(), cleaned_test.airline.isna().sum()],
            "duration_missing_after": [cleaned_train.duration.isna().sum(), cleaned_test.duration.isna().sum()],
        }
    )
    summary.to_csv(OUTPUT_DIR / "preprocessing_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nDuration imputation sources:")
    for name, data in [("train", cleaned_train), ("test", cleaned_test)]:
        print(f"{name}: {data['duration_imputation_source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
