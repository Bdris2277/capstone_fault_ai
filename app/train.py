from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from capstone_fault_ai.app import user_params


def _numeric_feature_frame(df: pd.DataFrame, *, context: str) -> pd.DataFrame:
    features = df.drop(columns=[c for c in ("class", "location") if c in df.columns])
    timestamp_cols = [
        c for c in features.columns
        if c.endswith("_meter_timestamp") or c.endswith("_meter_timestamp_utc")
    ]
    if timestamp_cols:
        features = features.drop(columns=timestamp_cols)
        print(f"[{context}] Ignoring timestamp metadata columns: {', '.join(timestamp_cols)}")
    numeric = features.select_dtypes(include=[np.number])
    dropped = [c for c in features.columns if c not in numeric.columns]
    if dropped:
        print(f"[{context}] Ignoring non-numeric columns: {', '.join(dropped)}")
    if numeric.shape[1] == 0:
        raise ValueError(f"[{context}] No numeric feature columns found (besides labels).")
    return numeric


def _oversample_minority_classes(
    x: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int,
    target_ratio_to_majority: float,
    skip_label: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample minority classes up to ratio*majority_count using replacement."""
    if len(y) == 0:
        return x, y

    rng = np.random.default_rng(random_state)
    labels, counts = np.unique(y, return_counts=True)
    major_count = int(counts.max())
    target_count = max(1, int(round(major_count * target_ratio_to_majority)))

    all_indices: list[np.ndarray] = []
    for label, count in zip(labels.tolist(), counts.tolist()):
        idx = np.where(y == label)[0]
        all_indices.append(idx)
        if skip_label is not None and int(label) == int(skip_label):
            continue
        if count < target_count:
            need = target_count - int(count)
            extra = rng.choice(idx, size=need, replace=True)
            all_indices.append(extra)

    final_idx = np.concatenate(all_indices)
    rng.shuffle(final_idx)
    return x[final_idx], y[final_idx]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train RandomForest model from labeled CSVs.")
    parser.add_argument("--train_csv", default=user_params.TRAIN_CSV)
    parser.add_argument("--test_csv", default=user_params.TEST_CSV)
    parser.add_argument("--out_dir", default=user_params.ARTIFACTS_DIR)
    parser.add_argument("--target", choices=["class", "location"], default="class")
    parser.add_argument("--n_estimators", type=int, default=int(getattr(user_params, "RF_N_ESTIMATORS", 300)))
    parser.add_argument(
        "--max_depth",
        type=int,
        default=int(getattr(user_params, "RF_MAX_DEPTH", 0)),
        help="0 means unlimited depth.",
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=int(getattr(user_params, "RF_MIN_SAMPLES_LEAF", 1)),
    )
    parser.add_argument("--random_state", type=int, default=int(getattr(user_params, "RF_RANDOM_STATE", 42)))
    parser.add_argument(
        "--oversample_ratio",
        type=float,
        default=float(getattr(user_params, "RF_OVERSAMPLE_RATIO", 0.6)),
        help="For class target: upsample each fault class to this ratio of majority class windows (0 disables).",
    )
    args = parser.parse_args(argv)

    if not Path(args.train_csv).exists() or not Path(args.test_csv).exists():
        print("Missing train/test CSVs.")
        print("Expected by default:")
        print(f" - {Path(args.train_csv)}")
        print(f" - {Path(args.test_csv)}")
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    if "class" not in train_df.columns or "class" not in test_df.columns:
        raise ValueError("Expected a 'class' column in both train and test CSVs.")

    x_train_df = _numeric_feature_frame(train_df, context="train")
    x_test_df = _numeric_feature_frame(test_df, context="test")

    if list(x_test_df.columns) != list(x_train_df.columns):
        x_test_df = x_test_df.reindex(columns=x_train_df.columns)
        if x_test_df.isna().any().any():
            raise ValueError("Test CSV is missing one or more numeric feature columns present in train CSV.")

    prefix = ""
    location_to_id: dict[str, int] = {}

    if args.target == "location":
        if "location" not in train_df.columns or "location" not in test_df.columns:
            raise ValueError("Expected a 'location' column in both CSVs for --target location.")
        location_names = sorted(set(train_df["location"].astype(str)) | set(test_df["location"].astype(str)))
        location_to_id = {name: i for i, name in enumerate(location_names)}
        y_train = train_df["location"].astype(str).map(location_to_id).values.astype(np.int64)
        y_test = test_df["location"].astype(str).map(location_to_id).values.astype(np.int64)
        prefix = "location_"
        (out_dir / "location_labels.json").write_text(json.dumps(location_to_id, indent=2), encoding="utf-8")
        print(f"Location targets: {location_to_id}")
    else:
        y_train = train_df["class"].values.astype(np.int64)
        y_test = test_df["class"].values.astype(np.int64)
        num_classes = len(user_params.FAULT_TYPES)
        if y_train.min() < 0 or y_test.min() < 0 or y_train.max() >= num_classes or y_test.max() >= num_classes:
            raise ValueError(
                f"Class labels must be in range [0, {num_classes - 1}] based on user_params.FAULT_TYPES."
            )

    model = RandomForestClassifier(
        n_estimators=int(args.n_estimators),
        max_depth=None if int(args.max_depth) <= 0 else int(args.max_depth),
        min_samples_leaf=int(args.min_samples_leaf),
        class_weight="balanced_subsample",
        random_state=int(args.random_state),
        n_jobs=-1,
    )

    x_train = x_train_df.values
    x_test = x_test_df.values

    if args.target == "class" and float(args.oversample_ratio) > 0:
        before = pd.Series(y_train).value_counts().sort_index().to_dict()
        x_train, y_train = _oversample_minority_classes(
            x_train,
            y_train,
            random_state=int(args.random_state),
            target_ratio_to_majority=float(args.oversample_ratio),
            skip_label=0,
        )
        after = pd.Series(y_train).value_counts().sort_index().to_dict()
        print(f"Class counts before oversampling: {before}")
        print(f"Class counts after oversampling:  {after}")

    print(f"Training RandomForest on {len(x_train)} samples, {x_train.shape[1]} features...")
    model.fit(x_train, y_train)

    joblib.dump(model, out_dir / f"{prefix}rf_model.pkl")
    (out_dir / f"{prefix}feature_columns.json").write_text(
        json.dumps(list(x_train_df.columns), indent=2),
        encoding="utf-8",
    )

    preds = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_test)
        confs = probs.max(axis=1) * 100.0
    else:
        confs = np.full(shape=(len(preds),), fill_value=np.nan, dtype=float)

    if args.target == "location":
        id_to_name = {v: k for k, v in location_to_id.items()}
        pred_values = np.asarray([id_to_name.get(int(p), str(p)) for p in preds], dtype=object)
    else:
        pred_values = preds

    pred_col = f"predicted_{args.target}"
    test_out = test_df.copy()
    test_out.insert(0, pred_col, pred_values)
    test_out.insert(1, "confidence_pct", [f"{c:.2f}" if np.isfinite(c) else "" for c in confs])
    test_out.to_csv(out_dir / f"{prefix}test_with_predictions.csv", index=False)

    actual = test_out[args.target].astype(str)
    predicted = test_out[pred_col].astype(str)
    mismatched = test_out[actual != predicted]
    mismatched.to_csv(out_dir / f"{prefix}mislabeled_rows.csv", index=False)

    accuracy = float((actual == predicted).mean() * 100.0) if len(test_out) else 0.0
    print(f"Saved artifacts to: {out_dir}")
    print(f"Accuracy ({args.target}): {accuracy:.2f}%")
    print(f"Mislabeled rows: {len(mismatched)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
