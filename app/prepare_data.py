"""
Data preparation script - turns each recorded event file into ordered windows
and splits train/test by EVENT FILE (never by row), so rows from the same
event cannot appear in both train and test.

Usage:
    python prepare_data.py

This script:
1. Finds event CSVs in data_out/ (one recording per file, e.g. fault_L1_40_1.csv).
   Each file is one ordered recording: nominal -> fault -> nominal.
2. Each file gets ONE label, taken from its 'class' column (the file's most common value).
3. Builds sliding windows of WINDOW_ROWS consecutive rows inside each file.
   Windows never cross file boundaries. Each window becomes one training sample,
   with its columns flattened as <feature>_t<step>.
4. Splits the event files per class: TRAIN_FRACTION of files go to train, the rest to test.
5. Saves to data_out/train_data.csv and data_out/test_data.csv

Ready for training!
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from pathlib import Path

import numpy as np
import pandas as pd

from capstone_fault_ai.app import user_params

WINDOW_ROWS = 10        # consecutive rows per sample (10 rows x 0.2 s = 2 s)
STRIDE_ROWS = 5         # step between window starts (overlap = WINDOW_ROWS - STRIDE_ROWS)
TRAIN_FRACTION = 0.8    # fraction of event FILES per class going to train
RANDOM_SEED = 42


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns, excluding class and meter timestamp metadata."""
    cols = [c for c in df.columns if c != "class"]
    cols = [c for c in cols if not c.endswith(("_meter_timestamp", "_meter_timestamp_utc"))]
    numeric = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    return numeric


def _windows_from_file(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Return array of shape (n_windows, WINDOW_ROWS * n_features), order preserved."""
    values = df[features].to_numpy(dtype=float)
    starts = range(0, len(values) - WINDOW_ROWS + 1, STRIDE_ROWS)
    return np.stack([values[s:s + WINDOW_ROWS].reshape(-1) for s in starts]) if len(starts) else np.empty((0, 0))


def _split_files_by_class(file_labels: dict[Path, int]) -> tuple[list[Path], list[Path]]:
    """Split event FILES within each class, so one file is never in both sets."""
    rng = np.random.default_rng(RANDOM_SEED)
    train_files: list[Path] = []
    test_files: list[Path] = []
    for label in sorted(set(file_labels.values())):
        files = [f for f, l in file_labels.items() if l == label]
        files = [files[i] for i in rng.permutation(len(files))]
        if len(files) == 1:
            print(f"  WARNING: class {label} has only 1 file; it goes to train only.")
            train_files.extend(files)
            continue
        n_train = min(max(1, round(TRAIN_FRACTION * len(files))), len(files) - 1)
        train_files.extend(files[:n_train])
        test_files.extend(files[n_train:])
    return train_files, test_files


def main():
    data_dir = Path(user_params.COLLECT_OUT_CSV).parent

    print("\n" + "=" * 60)
    print("DATA PREPARATION (windowed, split by event file)")
    print("=" * 60)
    print(f"Looking for event CSV files in: {data_dir}\n")

    csv_files = sorted([
        f for f in data_dir.glob("*.csv")
        if f.name not in ["train_data.csv", "test_data.csv"]
        and f.name.startswith(tuple(user_params.CLASS_LABEL_NAMES.values()))
    ])

    if not csv_files:
        print("No event files found.")
        print("Expected files like: normal_1.csv, fault_L1_40_1.csv, etc.")
        return 1

    # Load each file, read its label, and build windows
    feature_names: list[str] | None = None
    file_labels: dict[Path, int] = {}
    file_locations: dict[Path, str] = {}
    file_windows: dict[Path, np.ndarray] = {}

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if "class" not in df.columns:
            print(f"  Skipping {csv_file.name}: no 'class' column.")
            continue
        features = _feature_columns(df)
        if feature_names is None:
            feature_names = features
        elif features != feature_names:
            # Keep only columns shared with the first file, in the same order
            features = [c for c in feature_names if c in features]
            if len(features) != len(feature_names):
                print(f"  Skipping {csv_file.name}: feature columns do not match earlier files.")
                continue
        label = int(df["class"].mode().iloc[0])
        has_location = "fault_location" in df.columns and bool(df["fault_location"].notna().any())
        if has_location:
            location = str(df["fault_location"].mode().iloc[0])
        else:
            location = "none"
        windows = _windows_from_file(df, feature_names)
        if len(windows) == 0:
            print(f"  Skipping {csv_file.name}: fewer than {WINDOW_ROWS} rows.")
            continue
        file_labels[csv_file] = label
        file_locations[csv_file] = location
        file_windows[csv_file] = windows
        print(f"  {csv_file.name}: class {label}, location {location}, {len(df)} rows -> {len(windows)} windows")

    if not file_labels or feature_names is None:
        print("\nNo usable event files.")
        return 1

    train_files, test_files = _split_files_by_class(file_labels)

    # reshape(-1) on (WINDOW_ROWS, n_features) is step-major: all features at t0, then t1, ...
    flat_names = [f"{c}_t{i}" for i in range(WINDOW_ROWS) for c in feature_names]

    def build(files: list[Path]) -> pd.DataFrame:
        parts = []
        for f in files:
            w = file_windows[f]
            part = pd.DataFrame(w, columns=flat_names)
            part["class"] = file_labels[f]
            part["location"] = file_locations[f]
            parts.append(part)
        if not parts:
            return pd.DataFrame(columns=flat_names + ["class", "location"])
        out = pd.concat(parts, ignore_index=True)
        return out.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    train_df = build(train_files)
    test_df = build(test_files)

    print(f"\nTrain event files: {[f.name for f in train_files]}")
    print(f"Test event files:  {[f.name for f in test_files]}")
    print(f"Train windows: {len(train_df)}  Test windows: {len(test_df)}")
    print(f"Train class counts: {train_df['class'].value_counts().to_dict()}")
    print(f"Test class counts:  {test_df['class'].value_counts().to_dict()}")

    train_file = data_dir / "train_data.csv"
    test_file = data_dir / "test_data.csv"
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"\nSaved {train_file.name} and {test_file.name} to {data_dir}")
    print("\n" + "=" * 60)
    print("Ready to train! Run:")
    print("  python -m capstone_fault_ai.app.train")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
