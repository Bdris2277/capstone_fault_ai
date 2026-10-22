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
from sklearn.preprocessing import QuantileTransformer

from capstone_fault_ai.Model.model import build_fault_mlp
from capstone_fault_ai.Model.train_eval import predict_classes, train_one_epoch
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
        raise ValueError(f"[{context}] No numeric feature columns found (besides 'class').")
    return numeric


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Fault MLP from labeled CSVs.")
    parser.add_argument("--train_csv", default=user_params.TRAIN_CSV)
    parser.add_argument("--test_csv", default=user_params.TEST_CSV)
    parser.add_argument("--out_dir", default=user_params.ARTIFACTS_DIR)
    parser.add_argument("--epochs", type=int, default=int(user_params.EPOCHS))
    parser.add_argument("--batch_size", type=int, default=int(user_params.BATCH_SIZE))
    parser.add_argument("--target", choices=["class", "location"], default="class",
                        help="'class' = fault type, 'location' = which meter the fault hit.")
    parser.add_argument("--lr", type=float, default=float(user_params.LR))
    args = parser.parse_args(argv)

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError:
        print("PyTorch is required. Install torch on your Jetson (JetPack-specific build).")
        return 2

    if not Path(args.train_csv).exists() or not Path(args.test_csv).exists():
        print("Missing train/test CSVs.")
        print("Expected by default:")
        print(f" - {Path(args.train_csv)}")
        print(f" - {Path(args.test_csv)}")
        print("Collect data first (app/collect.py) or use app/run.py which can auto-split collected.csv.")
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    if "class" not in train_df.columns or "class" not in test_df.columns:
        raise ValueError("Expected a 'class' column in both train and test CSVs.")

    x_test_df = _numeric_feature_frame(test_df, context="test")
    prefix = ""
    location_to_id: dict[str, int] = {}

    if args.target == "location":
        if "location" not in train_df.columns or "location" not in test_df.columns:
            raise ValueError("Expected a 'location' column in both CSVs for --target location.")
        location_names = sorted(set(train_df["location"].astype(str)) | set(test_df["location"].astype(str)))
        location_to_id = {name: i for i, name in enumerate(location_names)}
        y_train = train_df["location"].astype(str).map(location_to_id).values.astype(np.int64)
        y_test = test_df["location"].astype(str).map(location_to_id).values.astype(np.int64)
        num_classes = len(location_names)
        prefix = "location_"
        (out_dir / "location_labels.json").write_text(json.dumps(location_to_id, indent=2))
        print(f"Location targets: {location_to_id}")
    else:
        y_train = train_df["class"].values.astype(np.int64)
        y_test = test_df["class"].values.astype(np.int64)
        num_classes = len(user_params.FAULT_TYPES)
        if y_train.min() < 0 or y_test.min() < 0 or y_train.max() >= num_classes or y_test.max() >= num_classes:
            raise ValueError(
                f"Class labels must be in range [0, {num_classes - 1}] based on user_params.FAULT_TYPES."
                f"Class labels must be in range [0, {num_classes - 1}] based on user_params.FAULT_TYPES."
        )

    if list(x_test_df.columns) != list(x_train_df.columns):
        x_test_df = x_test_df.reindex(columns=x_train_df.columns)
        if x_test_df.isna().any().any():
            raise ValueError("Test CSV is missing one or more numeric feature columns present in train CSV.")

    x_train_raw = x_train_df.values
    x_test_raw = x_test_df.values

    scaler = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(train_df)))
    x_train = scaler.fit_transform(x_train_raw)
    x_test = scaler.transform(x_test_raw)
f"{prefix}
    joblib.dump(scaler, out_dir / "scaler.pkl")

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_fault_mlp(input_size=x_train.shape[1], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        loss, acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        print(f"Epoch {epoch + 1}/{args.epochs} - loss={loss:.4f} acc={acc:.2f}%")
f"{prefix}model.pth")

    preds = predict_classes(model=model, loader=test_loader, device=device)
    if args.target == "location":
        id_to_name = {v: k for k, v in location_to_id.items()}
        pred_values = [id_to_name.get(int(p), str(p)) for p in preds]
    else:
        pred_values = preds
    pred_col = f"predicted_{args.target}"
    test_out = test_df.copy()
    test_out.insert(0, pred_col, pred_values)
    test_out.to_csv(out_dir / f"{prefix}test_with_predictions.csv", index=False)

    mismatched = test_out[test_out[args.target].astype(str) != test_out[pred_col].astype(str)]
    mismatched.to_csv(out_dir / f"{prefix}t["class"] != test_out["predicted_class"]]
    mismatched.to_csv(out_dir / "mislabeled_rows.csv", index=False)
    print(f"Saved artifacts to: {out_dir}")
    print(f"Mislabeled rows: {len(mismatched)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
