from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from capstone_fault_ai.core.config import load_meters_config, load_pipeline_config
from capstone_fault_ai.core.modbus import poll_holding_registers
from capstone_fault_ai.core.preprocess import PerUnitState, select_features, to_per_unit
from capstone_fault_ai.ml.model import build_fault_mlp
from capstone_fault_ai.app import user_params


@dataclass
class MeterRuntime:
    per_unit: PerUnitState


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
    args = parser.parse_args(argv)

    try:
        import torch
        from pymodbus.client import ModbusTcpClient
    except ModuleNotFoundError as exc:
        missing = str(exc).split()[-1]
        print(f"Missing dependency: {missing}. Install requirements on the Jetson.")
        return 2

    meters_cfg = load_meters_config(args.meters_json)
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
    pipeline = load_pipeline_config(args.pipeline_json)

    artifacts_dir = Path(args.artifacts_dir)
    if not (artifacts_dir / "scaler.pkl").exists() or not (artifacts_dir / "model.pth").exists():
        print(f"Missing artifacts in {artifacts_dir}. Train first (creates scaler.pkl and model.pth).")
        return 2
    scaler = joblib.load(artifacts_dir / "scaler.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_size = len(pipeline.features.names)
    model = build_fault_mlp(input_size=input_size, num_classes=5)
    model.load_state_dict(torch.load(artifacts_dir / "model.pth", map_location=device))
    model.to(device)
    model.eval()

    runtimes = {m.id: MeterRuntime(per_unit=PerUnitState()) for m in meters_cfg.meters}

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_path.exists()

    clients: dict[str, ModbusTcpClient] = {}
    try:
        for meter in meters_cfg.meters:
            client = ModbusTcpClient(meter.ip, port=meter.port, timeout=meter.timeout_s)
            client.connect()
            clients[meter.id] = client

        with open(out_path, "a", newline="", encoding="utf-8") as out_file:
            writer = csv.writer(out_file)
            if not file_exists:
                writer.writerow(
                    ["ts_unix_s", "meter_id", *pipeline.features.names, "predicted_class", "confidence_pct"]
                )

            while True:
                loop_start = time.monotonic()
                ts = time.time()

                for meter in meters_cfg.meters:
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

                    scaled = scaler.transform(features.reshape(1, -1))
                    input_tensor = torch.tensor(scaled, dtype=torch.float32).to(device)

                    with torch.no_grad():
                        logits = model(input_tensor)
                        probs = torch.softmax(logits, dim=1)
                        conf, pred = probs.max(1)

                    writer.writerow(
                        [
                            f"{ts:.3f}",
                            meter.id,
                            *[f"{v:.6g}" for v in features.tolist()],
                            int(pred.item()),
                            f"{(conf.item() * 100):.2f}",
                        ]
                    )
                    out_file.flush()

                elapsed = time.monotonic() - loop_start
                sleep_s = meters_cfg.poll_interval_s - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)

    finally:
        for client in clients.values():
            try:
                client.close()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
