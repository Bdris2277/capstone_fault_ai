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
from capstone_fault_ai.core.config import load_meters_config

WINDOW_ROWS = 10        # consecutive rows per sample (10 rows x 0.2 s = 2 s)
STRIDE_ROWS = 5         # step between window starts (overlap = WINDOW_ROWS - STRIDE_ROWS)
TRAIN_FRACTION = 0.8    # fraction of event FILES per class going to train
RANDOM_SEED = 42
FAULT_WINDOW_KEEP_FRACTION = 0.50  # keep top anomaly windows from fault files
FAULT_WINDOW_KEEP_MIN = 6          # minimum windows to keep for any fault file
FAULT_EVENT_INDEX_GAP = 4          # split selected fault windows into new pseudo-event when index jump > gap
NORMAL_CHUNK_WINDOWS = 80          # split long normal files into chunks to avoid one huge normal event
MIN_FAULT_TRAIN_WINDOWS = 30       # safeguard: keep at least this many windows/class in train for fault classes
MIN_FAULT_TEST_WINDOWS = 10        # target minimum windows/class in test when possible
WINDOW_TIME_FEATURE_NAMES = ["window_start_s", "window_dt_mean_s", "window_dt_std_s"]


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


def _window_starts(n_rows: int) -> np.ndarray:
    return np.arange(0, max(0, n_rows - WINDOW_ROWS + 1), STRIDE_ROWS, dtype=int)


def _window_time_features(
    df: pd.DataFrame,
    *,
    nominal_dt_s: float,
) -> np.ndarray:
    """Per-window time features from meter UTC timestamps.

    Returns columns: window_start_s, window_dt_mean_s, window_dt_std_s.
    """
    starts = _window_starts(len(df))
    if len(starts) == 0:
        return np.empty((0, len(WINDOW_TIME_FEATURE_NAMES)), dtype=float)

    ts_cols = [c for c in df.columns if c.endswith("_meter_timestamp_utc")]
    if not ts_cols:
        base = np.arange(len(df), dtype=float) * nominal_dt_s
    else:
        # Use first timestamp column with enough valid values.
        base = None
        for col in ts_cols:
            ts = pd.to_datetime(df[col], utc=True, errors="coerce")
            valid = ts.notna()
            if int(valid.sum()) >= 2:
                sec = ts.astype("int64").to_numpy(dtype=float) / 1e9
                sec[~valid.to_numpy()] = np.nan
                first = float(np.nanmin(sec))
                base = sec - first
                break
        if base is None:
            base = np.arange(len(df), dtype=float) * nominal_dt_s

    out: list[list[float]] = []
    for s in starts.tolist():
        w = base[s:s + WINDOW_ROWS]
        valid_mask = np.isfinite(w)
        if int(valid_mask.sum()) >= 2:
            diffs = np.diff(w[valid_mask])
            diffs = diffs[(diffs > 0.0) & (diffs < 5.0)]
            if len(diffs) > 0:
                dt_mean = float(np.mean(diffs))
                dt_std = float(np.std(diffs))
            else:
                dt_mean = float(nominal_dt_s)
                dt_std = 0.0
            start_s = float(np.nanmin(w))
        else:
            dt_mean = float(nominal_dt_s)
            dt_std = 0.0
            start_s = float(s * nominal_dt_s)
        out.append([start_s, dt_mean, dt_std])
    return np.asarray(out, dtype=float)


def _fault_window_indices(df: pd.DataFrame, features: list[str], total_windows: int) -> np.ndarray:
    """Select likely-fault windows by anomaly score, so nominal edges are not mislabeled."""
    if total_windows <= 0:
        return np.asarray([], dtype=int)

    signal_cols = [
        c for c in features
        if ("_V_" in c or "_A_" in c or "_W_" in c or "_VA_" in c)
    ]
    if not signal_cols:
        signal_cols = features

    vals = df[signal_cols].to_numpy(dtype=float)
    baseline = np.median(vals, axis=0)
    denom = np.maximum(np.abs(baseline), 1e-6)
    row_score = np.mean(np.abs((vals - baseline) / denom), axis=1)

    starts = np.arange(0, len(vals) - WINDOW_ROWS + 1, STRIDE_ROWS)
    if len(starts) == 0:
        return np.asarray([], dtype=int)

    win_scores = np.asarray([
        float(np.mean(row_score[s:s + WINDOW_ROWS]))
        for s in starts
    ])

    keep_n = min(total_windows, max(FAULT_WINDOW_KEEP_MIN, round(total_windows * FAULT_WINDOW_KEEP_FRACTION)))
    top_idx = np.argsort(win_scores)[-keep_n:]
    return np.sort(top_idx)


def _split_files_by_class(
    file_labels: dict[str, int],
    file_windows: dict[str, np.ndarray],
) -> tuple[list[str], list[str]]:
    """Split event units within each class, keeping enough fault windows in train."""
    rng = np.random.default_rng(RANDOM_SEED)
    train_files: list[str] = []
    test_files: list[str] = []

    for label in sorted(set(file_labels.values())):
        files = [f for f, l in file_labels.items() if l == label]
        files = [files[i] for i in rng.permutation(len(files))]

        if len(files) == 1:
            print(f"  WARNING: class {label} has only 1 file; it goes to train only.")
            train_files.extend(files)
            continue

        # Build class-specific targets using available windows.
        file_sizes = {f: len(file_windows[f]) for f in files}
        total_windows = int(sum(file_sizes.values()))
        base_target_train = round(total_windows * TRAIN_FRACTION)
        if label == 0:
            target_train_windows = max(base_target_train, 1)
            min_test_windows = 1
        else:
            target_train_windows = min(total_windows - 1, max(base_target_train, MIN_FAULT_TRAIN_WINDOWS))
            min_test_windows = min(MIN_FAULT_TEST_WINDOWS, total_windows - 1)

        # Add largest chunks first until train target is met.
        by_size_desc = sorted(files, key=lambda f: (file_sizes[f], f), reverse=True)
        class_train: list[str] = []
        class_test: list[str] = []
        train_windows = 0

        for f in by_size_desc:
            remaining = len(by_size_desc) - (len(class_train) + len(class_test))
            must_leave_for_test = (remaining == 1 and len(class_test) == 0)
            if must_leave_for_test:
                class_test.append(f)
                continue

            if train_windows < target_train_windows:
                class_train.append(f)
                train_windows += file_sizes[f]
            else:
                class_test.append(f)

        if not class_test:
            smallest_train = min(class_train, key=lambda f: file_sizes[f])
            class_train.remove(smallest_train)
            class_test.append(smallest_train)
            train_windows -= file_sizes[smallest_train]

        test_windows = int(sum(file_sizes[f] for f in class_test))

        # If fault class test windows are too tiny, move one smallest train chunk to test.
        if label != 0 and test_windows < min_test_windows and len(class_train) > 1:
            move = min(class_train, key=lambda f: file_sizes[f])
            class_train.remove(move)
            class_test.append(move)
            train_windows -= file_sizes[move]
            test_windows += file_sizes[move]

        if label != 0 and train_windows < MIN_FAULT_TRAIN_WINDOWS:
            print(
                f"  WARNING: class {label} has only {train_windows} train windows "
                f"(<{MIN_FAULT_TRAIN_WINDOWS})."
            )

        train_files.extend(class_train)
        test_files.extend(class_test)

        print(
            f"  class {label} split: events train/test={len(class_train)}/{len(class_test)} "
            f"windows train/test={train_windows}/{test_windows}"
        )

    return train_files, test_files


def _split_selected_windows_into_events(
    windows: np.ndarray,
    selected_idx: np.ndarray,
    *,
    max_index_gap: int,
) -> list[np.ndarray]:
    """Split selected fault windows into pseudo-events based on gaps in selected window indices."""
    if len(windows) == 0:
        return []
    if len(windows) != len(selected_idx):
        return [windows]

    chunks: list[np.ndarray] = []
    start = 0
    for i in range(1, len(selected_idx)):
        if int(selected_idx[i]) - int(selected_idx[i - 1]) > max_index_gap:
            chunks.append(windows[start:i])
            start = i
    chunks.append(windows[start:])
    return [c for c in chunks if len(c) > 0]


def _chunk_windows_fixed_size(windows: np.ndarray, *, chunk_size: int) -> list[np.ndarray]:
    """Split windows into fixed-size chunks (used for long normal recordings)."""
    if len(windows) == 0:
        return []
    if chunk_size <= 0 or len(windows) <= chunk_size:
        return [windows]
    return [windows[i:i + chunk_size] for i in range(0, len(windows), chunk_size) if len(windows[i:i + chunk_size]) > 0]


def main():
    meters_cfg = load_meters_config(user_params.METERS_JSON)
    nominal_dt_s = float(meters_cfg.poll_interval_s)

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
    file_labels: dict[str, int] = {}
    file_locations: dict[str, str] = {}
    file_windows: dict[str, np.ndarray] = {}
    file_window_time: dict[str, np.ndarray] = {}
    event_sources: dict[str, str] = {}

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
        windows_all = _windows_from_file(df, feature_names)
        time_all = _window_time_features(df, nominal_dt_s=nominal_dt_s)
        if len(windows_all) == 0:
            print(f"  Skipping {csv_file.name}: fewer than {WINDOW_ROWS} rows.")
            continue

        event_chunks: list[np.ndarray]
        if label != 0:
            keep_idx = _fault_window_indices(df, feature_names, len(windows_all))
            kept = windows_all[keep_idx]
            kept_time = time_all[keep_idx]
            if len(kept) == 0:
                print(f"  Skipping {csv_file.name}: no fault windows selected.")
                continue
            event_chunks = _split_selected_windows_into_events(
                kept,
                keep_idx,
                max_index_gap=FAULT_EVENT_INDEX_GAP,
            )
            time_chunks = _split_selected_windows_into_events(
                kept_time,
                keep_idx,
                max_index_gap=FAULT_EVENT_INDEX_GAP,
            )
        else:
            event_chunks = _chunk_windows_fixed_size(windows_all, chunk_size=NORMAL_CHUNK_WINDOWS)
            time_chunks = _chunk_windows_fixed_size(time_all, chunk_size=NORMAL_CHUNK_WINDOWS)

        total_windows_after = 0
        for i, (chunk, tchunk) in enumerate(zip(event_chunks, time_chunks), start=1):
            event_id = f"{csv_file.name}::evt{i:03d}"
            file_labels[event_id] = label
            file_locations[event_id] = location
            file_windows[event_id] = chunk
            file_window_time[event_id] = tchunk
            event_sources[event_id] = csv_file.name
            total_windows_after += len(chunk)

        print(
            f"  {csv_file.name}: class {label}, location {location}, "
            f"{len(df)} rows -> {total_windows_after} windows across {len(event_chunks)} events"
        )

    if not file_labels or feature_names is None:
        print("\nNo usable event files.")
        return 1

    train_files, test_files = _split_files_by_class(file_labels, file_windows)

    # reshape(-1) on (WINDOW_ROWS, n_features) is step-major: all features at t0, then t1, ...
    flat_names = [f"{c}_t{i}" for i in range(WINDOW_ROWS) for c in feature_names]

    def build(files: list[str]) -> pd.DataFrame:
        parts = []
        for f in files:
            w = file_windows[f]
            part = pd.DataFrame(w, columns=flat_names)
            t = file_window_time[f]
            for j, name in enumerate(WINDOW_TIME_FEATURE_NAMES):
                part[name] = t[:, j]
            part["class"] = file_labels[f]
            part["location"] = file_locations[f]
            part["source_event"] = f
            part["source_file"] = event_sources[f]
            parts.append(part)
        if not parts:
            return pd.DataFrame(
                columns=flat_names + WINDOW_TIME_FEATURE_NAMES + ["class", "location", "source_event", "source_file"]
            )
        out = pd.concat(parts, ignore_index=True)
        return out.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    train_df = build(train_files)
    test_df = build(test_files)

    train_by_class = pd.Series([file_labels[f] for f in train_files]).value_counts().sort_index().to_dict()
    test_by_class = pd.Series([file_labels[f] for f in test_files]).value_counts().sort_index().to_dict()
    print(f"\nTrain event units: {len(train_files)} by class {train_by_class}")
    print(f"Test event units:  {len(test_files)} by class {test_by_class}")
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
