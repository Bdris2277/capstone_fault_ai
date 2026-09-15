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
    """Tracks per-meter state during data collection."""
    per_unit: PerUnitState


def _get_next_filename(base_path: Path) -> Path:
    """Auto-increment filename: collected.csv → collected1.csv → collected2.csv → etc."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If base file doesn't exist, use it
    if not base_path.exists():
        return base_path
    
    # Otherwise, find the next numbered version
    stem = base_path.stem
    suffix = base_path.suffix
    counter = 1
    while (base_path.parent / f"{stem}{counter}{suffix}").exists():
        counter += 1
    return base_path.parent / f"{stem}{counter}{suffix}"


def _create_csv_writer(path: Path, header: list[str]) -> tuple[csv.writer, object]:
    """Open a CSV file and return the writer and file object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(file)
    writer.writerow(header)
    file.flush()
    return writer, file


def _process_meter_data(meter, raw_values, pipeline, runtime, class_label, include_meter_id: bool):
    """Extract and validate meter data, return a CSV row or None if invalid."""
    # Select features and check size
    features = select_features(raw_values, drop_indices_0_based=pipeline.features.drop_indices_0_based)
    if features.shape[0] != len(pipeline.features.names):
        return None
    
    # Apply per-unit normalization if enabled
    if pipeline.per_unit.enabled:
        baseline = runtime.per_unit.update_and_get_baseline(
            features, baseline_samples=pipeline.per_unit.baseline_samples
        )
        if baseline is None:
            return None
        features = to_per_unit(
            features,
            per_unit_feature_count=pipeline.per_unit.per_unit_feature_count,
            baseline=baseline,
        )
    
    # Skip invalid data
    if not np.isfinite(features).all():
        return None
    
    # Format row
    feature_strs = [f"{v:.6g}" for v in features.tolist()]
    if include_meter_id:
        return [int(class_label), meter.id, *feature_strs]
    else:
        return [int(class_label), *feature_strs]


def main(argv: list[str] | None = None) -> int:
    """
    Poll Modbus meters and collect labeled sensor data to CSV.
    
    Usage:
        python collect.py --class_label 0 --duration_s 30
    """
    # Import early so it's available for the whole function
    try:
        from pymodbus.client import ModbusTcpClient
    except ModuleNotFoundError:
        print("ERROR: pymodbus not installed. Run: pip install -r requirements.txt")
        return 2
    
    parser = argparse.ArgumentParser(description="Collect labeled meter data to CSV.")
    parser.add_argument("--meters_json", default=user_params.METERS_JSON, help="Meter config file")
    parser.add_argument("--pipeline_json", default=user_params.PIPELINE_JSON, help="Pipeline config file")
    parser.add_argument("--class_label", type=int, default=None, help="Class label for this data (0, 1, 2, etc)")
    parser.add_argument("--out_csv", default=user_params.COLLECT_OUT_CSV, help="Output CSV file")
    parser.add_argument("--duration_s", type=float, default=30.0, help="Collect duration in seconds")
    parser.add_argument("--include_meter_id", action="store_true", help="Add meter_id column to CSV")
    args = parser.parse_args(argv)

    # Set default class label
    if args.class_label is None:
        args.class_label = int(user_params.COLLECT_CLASS_LABEL)

    # Load configuration
    meters_cfg = load_meters_config(args.meters_json)
    pipeline = load_pipeline_config(args.pipeline_json)

    # Filter to only enabled meters
    enabled_meters = [m for m in meters_cfg.meters if getattr(m, 'enabled', True)]
    if not enabled_meters:
        print("ERROR: No enabled meters in meters.json (set 'enabled': true)")
        return 2
    
    print(f"Collecting from {len(enabled_meters)} meter(s): {', '.join(m.id for m in enabled_meters)}\n")
    
    # Replace meters list with only enabled ones
    meters_cfg.meters = enabled_meters

    # Validate pipeline
    if pipeline.modbus.count % 2 != 0:
        print("ERROR: pipeline.modbus.count must be even")
        return 2

    # Prepare CSV header
    feature_names = pipeline.features.names
    if args.include_meter_id:
        header = ["class", "meter_id", *feature_names]
    else:
        header = ["class", *feature_names]

    # Prepare output file
    out_csv = Path(args.out_csv)
    out_csv = _get_next_filename(out_csv)

    # Track per-meter state
    runtimes = {m.id: MeterRuntime(per_unit=PerUnitState()) for m in meters_cfg.meters}

    # Connect to all meters
    clients: dict[str, ModbusTcpClient] = {}
    try:
        for meter in meters_cfg.meters:
            client = ModbusTcpClient(meter.ip, port=meter.port, timeout=meter.timeout_s)
            if not client.connect():
                print(f"WARNING: Failed to connect to {meter.id} ({meter.ip}:{meter.port})")
            clients[meter.id] = client

        # Open CSV writer
        writer, csv_file = _create_csv_writer(out_csv, header)

        # Collect data for specified duration
        deadline = time.monotonic() + args.duration_s
        rows_written = 0

        while time.monotonic() < deadline:
            loop_start = time.monotonic()

            for meter in meters_cfg.meters:
                # Poll meter data
                raw_values = poll_holding_registers(
                    clients[meter.id],
                    ip=meter.ip,
                    unit_id=meter.unit_id,
                    address=pipeline.modbus.address,
                    count=pipeline.modbus.count,
                    float_endianness=pipeline.decode.float_endianness,
                )
                if raw_values is None:
                    continue

                # Process and format data
                row = _process_meter_data(
                    meter, raw_values, pipeline,
                    runtimes[meter.id],
                    args.class_label,
                    args.include_meter_id
                )
                if row is None:
                    continue

                # Write row
                writer.writerow(row)
                rows_written += 1

            # Flush and pace polling
            csv_file.flush()
            elapsed = time.monotonic() - loop_start
            sleep_s = meters_cfg.poll_interval_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

        print(f"✓ Wrote {rows_written} rows to {out_csv}")
        return 0

    finally:
        # Cleanup
        for client in clients.values():
            try:
                client.close()
            except Exception:
                pass
        try:
            csv_file.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
