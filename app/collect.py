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

import numpy as np

from capstone_fault_ai.core.config import load_meters_config, load_pipeline_config
from capstone_fault_ai.core.modbus import poll_holding_registers
from capstone_fault_ai.core.preprocess import PerUnitState, select_features, to_per_unit
from capstone_fault_ai.app import user_params


@dataclass
class MeterRuntime:
    per_unit: PerUnitState


def _open_writer(path: Path, *, header: list[str], append: bool) -> tuple[csv.writer, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    file = open(path, "a" if append else "w", newline="", encoding="utf-8")
    writer = csv.writer(file)
    if (not append) or (not file_exists):
        writer.writerow(header)
        file.flush()
    return writer, file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Poll Modbus-TCP meters and write labeled training rows (class + features) to CSV."
    )
    parser.add_argument("--meters_json", default=user_params.METERS_JSON)
    parser.add_argument("--pipeline_json", default=user_params.PIPELINE_JSON)
    parser.add_argument(
        "--meter_ids",
        default="",
        help="Comma-separated meter IDs to poll (default: all meters from meters_json).",
    )
    parser.add_argument(
        "--class_label",
        type=int,
        default=None,
        help="Integer class ID to write into the CSV. Defaults to app/user_params.py:COLLECT_CLASS_LABEL.",
    )
    parser.add_argument("--out_csv", default=user_params.COLLECT_OUT_CSV)
    parser.add_argument("--duration_s", type=float, default=30.0, help="How long to collect data (seconds).")
    parser.add_argument("--append", action="store_true", help="Append to out_csv if it exists (keep header).")
    parser.add_argument(
        "--include_meter_id",
        action="store_true",
        help="Add a meter_id column to the output (train.py will ignore non-numeric columns).",
    )
    parser.add_argument(
        "--split_by_meter",
        action="store_true",
        help="Write one CSV per meter under out_dir (ignores out_csv filename).",
    )
    parser.add_argument(
        "--out_dir",
        default="data_out",
        help="Used with --split_by_meter; files will be named <meter_id>.csv",
    )
    args = parser.parse_args(argv)

    try:
        from pymodbus.client import ModbusTcpClient
    except ModuleNotFoundError:
        print("Missing dependency: pymodbus. Install requirements first.")
        return 2

    if args.class_label is None:
        args.class_label = int(user_params.COLLECT_CLASS_LABEL)

    meters_cfg = load_meters_config(args.meters_json)
    pipeline = load_pipeline_config(args.pipeline_json)

    selected_ids: set[str] | None = None
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

    if pipeline.modbus.count % 2 != 0:
        raise ValueError("pipeline.modbus.count must be even (register pairs decode into float32 values).")

    feature_names = pipeline.features.names
    input_size = len(feature_names)
    header = ["class", "meter_id", *feature_names] if args.include_meter_id else ["class", *feature_names]

    runtimes = {m.id: MeterRuntime(per_unit=PerUnitState()) for m in meters_cfg.meters}

    out_csv = Path(args.out_csv)
    out_dir = Path(args.out_dir)

    writers: dict[str, csv.writer] = {}
    files: dict[str, object] = {}

    clients: dict[str, ModbusTcpClient] = {}
    try:
        for meter in meters_cfg.meters:
            client = ModbusTcpClient(meter.ip, port=meter.port, timeout=meter.timeout_s)
            client.connect()
            clients[meter.id] = client

        if args.split_by_meter:
            for meter in meters_cfg.meters:
                path = out_dir / f"{meter.id}.csv"
                writer, file = _open_writer(path, header=header, append=args.append)
                writers[meter.id] = writer
                files[meter.id] = file
        else:
            writer, file = _open_writer(out_csv, header=header, append=args.append)
            for meter in meters_cfg.meters:
                writers[meter.id] = writer
            files["_shared"] = file

        deadline = time.monotonic() + max(float(args.duration_s), 0.0)
        rows_written = 0

        while time.monotonic() < deadline:
            loop_start = time.monotonic()

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

                if not np.isfinite(features).all():
                    continue

                if args.include_meter_id:
                    row = [int(args.class_label), meter.id, *[f"{v:.6g}" for v in features.tolist()]]
                else:
                    row = [int(args.class_label), *[f"{v:.6g}" for v in features.tolist()]]
                writers[meter.id].writerow(row)
                rows_written += 1

            for file in files.values():
                try:
                    file.flush()
                except Exception:
                    pass

            elapsed = time.monotonic() - loop_start
            sleep_s = meters_cfg.poll_interval_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

        print(f"Wrote {rows_written} rows.")
        if args.split_by_meter:
            print(f"Output dir: {out_dir}")
        else:
            print(f"Output file: {out_csv}")

    finally:
        for client in clients.values():
            try:
                client.close()
            except Exception:
                pass
        for file in files.values():
            try:
                file.close()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
