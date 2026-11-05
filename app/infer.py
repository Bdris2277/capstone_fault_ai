from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

import argparse
import csv
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from capstone_fault_ai.app import user_params
from capstone_fault_ai.core.config import load_meters_config, load_pipeline_config
from capstone_fault_ai.core.modbus import poll_holding_registers
from capstone_fault_ai.core.preprocess import PerUnitState, select_features, to_per_unit


@dataclass
class MeterRuntime:
    per_unit: PerUnitState
    recent_features: deque[np.ndarray]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-meter Modbus polling + live inference (Jetson).")
    parser.add_argument("--meters_json", default=user_params.METERS_JSON)
    parser.add_argument("--pipeline_json", default=user_params.PIPELINE_JSON)
    parser.add_argument(
        "--meter_ids",
        default="",
        help="Comma-separated meter IDs to poll (default: all meters from meters_json).",
    )
    parser.add_argument("--artifacts_dir", default=user_params.ARTIFACTS_DIR)
    parser.add_argument("--out_csv", default=user_params.INFER_OUT_CSV)
    parser.add_argument("--target", choices=["class", "location"], default="class")
    args = parser.parse_args(argv)

    try:
        from pymodbus.client import ModbusTcpClient
    except ModuleNotFoundError as exc:
        missing = str(exc).split()[-1]
        print(f"Missing dependency: {missing}. Install requirements on the Jetson.")
        return 2

    meters_cfg = load_meters_config(args.meters_json)
    meters_cfg = type(meters_cfg)(
        poll_interval_s=meters_cfg.poll_interval_s,
        meters=[m for m in meters_cfg.meters if getattr(m, "enabled", True)],
    )

    if not meters_cfg.meters:
        print("No enabled meters found in meters config.")
        return 2

    if str(args.meter_ids).strip():
        selected_ids = {s.strip() for s in str(args.meter_ids).split(",") if s.strip()}
        known_ids = {m.id for m in meters_cfg.meters}
        missing = sorted(selected_ids - known_ids)
        if missing:
            print(f"Unknown meter_ids in --meter_ids: {', '.join(missing)}")
            print(f"Known meter IDs: {', '.join(sorted(known_ids))}")
            return 2
        meters_cfg = type(meters_cfg)(
            poll_interval_s=meters_cfg.poll_interval_s,
            meters=[m for m in meters_cfg.meters if m.id in selected_ids],
        )
        if not meters_cfg.meters:
            print("No enabled meters left after applying --meter_ids filter.")
            return 2
    pipeline = load_pipeline_config(args.pipeline_json)

    artifacts_dir = Path(args.artifacts_dir)
    prefix = "location_" if args.target == "location" else ""
    model_path = artifacts_dir / f"{prefix}rf_model.pkl"
    feature_cols_path = artifacts_dir / f"{prefix}feature_columns.json"
    if not model_path.exists() or not feature_cols_path.exists():
        print(
            f"Missing artifacts in {artifacts_dir}. Train first "
            f"(creates {prefix}rf_model.pkl and {prefix}feature_columns.json)."
        )
        return 2

    model = joblib.load(model_path)
    feature_columns = json.loads(feature_cols_path.read_text(encoding="utf-8"))
    if not isinstance(feature_columns, list) or not feature_columns:
        print(f"Invalid feature column metadata in {feature_cols_path}")
        return 2

    base_feature_count = len(pipeline.features.names)
    if len(feature_columns) % base_feature_count != 0:
        print(
            "Feature metadata is incompatible with current pipeline: "
            f"metadata_columns={len(feature_columns)}, base_features={base_feature_count}."
        )
        return 2
    window_rows = len(feature_columns) // base_feature_count
    if window_rows <= 0:
        print("Invalid window size derived from feature metadata.")
        return 2

    location_id_to_name: dict[int, str] = {}
    if args.target == "location":
        labels_path = artifacts_dir / "location_labels.json"
        if not labels_path.exists():
            print(f"Missing location label mapping at {labels_path}. Retrain location model first.")
            return 2
        location_to_id = json.loads(labels_path.read_text(encoding="utf-8"))
        location_id_to_name = {int(v): str(k) for k, v in location_to_id.items()}

    input_size = len(pipeline.features.names)
    model_features = getattr(model, "n_features_in_", None)
    if model_features is not None and int(model_features) != len(feature_columns):
        print(
            "Feature mismatch between model and saved feature columns: "
            f"model={int(model_features)}, metadata={len(feature_columns)}. Retrain required."
        )
        return 2

    print(f"Loaded RandomForest target='{args.target}' from: {model_path}")
    print(f"Window rows: {window_rows}  Base features per row: {input_size}")

    runtimes = {
        m.id: MeterRuntime(per_unit=PerUnitState(), recent_features=deque(maxlen=window_rows))
        for m in meters_cfg.meters
    }

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_path.exists()
    print(f"Writing predictions to: {out_path}")

    clients: dict[str, ModbusTcpClient] = {}
    try:
        for meter in meters_cfg.meters:
            client = ModbusTcpClient(meter.ip, port=meter.port, timeout=meter.timeout_s)
            if not client.connect():
                print(f"[{meter.id}] connect failed ({meter.ip}:{meter.port})")
                continue
            clients[meter.id] = client

        if not clients:
            print("Failed to connect to all selected meters.")
            return 2

        with open(out_path, "a", newline="", encoding="utf-8") as out_file:
            writer = csv.writer(out_file)
            if not file_exists:
                writer.writerow(
                    [
                        "ts_unix_s",
                        "meter_id",
                        *pipeline.features.names,
                        "predicted_value",
                        "predicted_label",
                        "predicted_fault_location",
                        "confidence_pct",
                        "target",
                    ]
                )

            try:
                while True:
                    loop_start = time.monotonic()
                    ts = time.time()

                    for meter in meters_cfg.meters:
                        if meter.id not in clients:
                            continue

                        raw = poll_holding_registers(
                            clients[meter.id],
                            ip=meter.ip,
                            unit_id=meter.unit_id,
                            address=pipeline.modbus.address,
                            count=pipeline.modbus.count,
                            float_endianness=pipeline.decode.float_endianness,
                        )
                        if raw is None:
                            continue

                        features = select_features(raw, drop_indices_0_based=pipeline.features.drop_indices_0_based)
                        if features.shape[0] != input_size:
                            print(f"[{meter.id}] feature_size={features.shape[0]} expected={input_size}")
                            continue
                        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

                        if pipeline.per_unit.enabled:
                            baseline = runtimes[meter.id].per_unit.update_and_get_baseline(
                                features,
                                baseline_samples=pipeline.per_unit.baseline_samples,
                            )
                            if baseline is None:
                                continue
                            features = to_per_unit(
                                features,
                                per_unit_feature_count=pipeline.per_unit.per_unit_feature_count,
                                baseline=baseline,
                            )

                        meter_runtime = runtimes[meter.id]
                        meter_runtime.recent_features.append(features.astype(float))
                        if len(meter_runtime.recent_features) < window_rows:
                            continue

                        window_vec = np.concatenate(list(meter_runtime.recent_features), axis=0).reshape(1, -1)
                        pred_value = model.predict(window_vec)[0]
                        if hasattr(model, "predict_proba"):
                            conf = float(np.max(model.predict_proba(window_vec)) * 100.0)
                        else:
                            conf = float("nan")

                        if args.target == "location":
                            pred_label = location_id_to_name.get(int(pred_value), str(pred_value))
                            predicted_fault_location = pred_label
                        else:
                            pred_label = user_params.FAULT_TYPES.get(int(pred_value), f"class_{int(pred_value)}")
                            predicted_fault_location = meter.id

                        writer.writerow(
                            [
                                f"{ts:.3f}",
                                meter.id,
                                *[f"{v:.6g}" for v in features.tolist()],
                                pred_value,
                                pred_label,
                                predicted_fault_location,
                                "" if np.isnan(conf) else f"{conf:.2f}",
                                args.target,
                            ]
                        )
                        conf_txt = "n/a" if np.isnan(conf) else f"{conf:.2f}%"
                        print(
                            f"[{args.target}] meter={meter.id} pred={pred_label} "
                            f"(value={pred_value}) conf={conf_txt}"
                        )
                        if args.target == "class" and int(pred_value) != 0:
                            print(
                                f"ALERT: fault-like pattern at source meter '{meter.id}' "
                                f"-> predicted type '{pred_label}'"
                            )
                        out_file.flush()

                    elapsed = time.monotonic() - loop_start
                    sleep_s = meters_cfg.poll_interval_s - elapsed
                    if sleep_s > 0:
                        time.sleep(sleep_s)
            except KeyboardInterrupt:
                print("\nStopping live inference.")

    finally:
        for client in clients.values():
            try:
                client.close()
            except (OSError, RuntimeError, AttributeError) as exc:
                print(f"Failed to close meter client cleanly: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
