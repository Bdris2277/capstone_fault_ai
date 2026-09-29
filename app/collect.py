"""
Simple meter data collection script.
Collects data from all enabled meters, organizes into CSV, saves to data_out/.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import numpy as np

from capstone_fault_ai.core.config import load_meters_config, load_pipeline_config
from capstone_fault_ai.core.modbus import poll_holding_registers
from capstone_fault_ai.core.preprocess import PerUnitState, select_features, to_per_unit
from capstone_fault_ai.app import user_params


@dataclass
class MeterRuntime:
    per_unit: PerUnitState


def collect_data(duration_s: float = 60.0, class_label: int = 0):
    """
    Collect data from all enabled meters.
    
    Args:
        duration_s: How long to collect (seconds)
        class_label: Class label for this data
    """
    try:
        from pymodbus.client import ModbusTcpClient
    except ModuleNotFoundError:
        print("ERROR: pymodbus not installed")
        return 2

    # Load configs
    meters_cfg = load_meters_config(user_params.METERS_JSON)
    pipeline = load_pipeline_config(user_params.PIPELINE_JSON)

    # Filter to enabled meters
    meters = [m for m in meters_cfg.meters if getattr(m, 'enabled', True)]
    if not meters:
        print("ERROR: No enabled meters")
        return 2

    print(f"Collecting from: {', '.join(m.id for m in meters)}")
    print(f"Duration: {duration_s}s\n")

    # Setup CSV file with class label name
    class_name = user_params.CLASS_LABEL_NAMES.get(class_label, f"class_{class_label}")
    out_dir = Path(user_params.COLLECT_OUT_CSV).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find next auto-increment number for this class
    counter = 1
    while True:
        out_file = out_dir / f"{class_name}_{counter}.csv"
        if not out_file.exists():
            break
        counter += 1

    # CSV header - one meter's features per column group
    header = ["class"]
    for meter in meters:
        header.extend([f"{meter.id}_{fname}" for fname in pipeline.features.names])

    # Open CSV
    csvfile = open(out_file, "w", newline="", encoding="utf-8")
    writer = csv.writer(csvfile)
    writer.writerow(header)

    # Connect to meters
    clients = {}
    for meter in meters:
        client = ModbusTcpClient(meter.ip, port=meter.port, timeout=meter.timeout_s)
        connected = client.connect()
        if not connected:
            print(f"❌ Failed to connect to {meter.id} ({meter.ip}:{meter.port})")
        else:
            print(f"✓ Connected to {meter.id}")
        clients[meter.id] = client

    print()

    # Collect data
    runtimes = {m.id: MeterRuntime(per_unit=PerUnitState()) for m in meters}
    deadline = time.monotonic() + duration_s
    rows = 0
    polls = 0

    try:
        while time.monotonic() < deadline:
            loop_start = time.monotonic()
            row_data = [class_label]
            row_complete = True

            # Collect from all meters for this cycle
            for meter in meters:
                polls += 1
                # Get data
                raw = poll_holding_registers(
                    clients[meter.id],
                    ip=meter.ip,
                    unit_id=meter.unit_id,
                    address=pipeline.modbus.address,
                    count=pipeline.modbus.count,
                    float_endianness=pipeline.decode.float_endianness,
                )
                if raw is None:
                    print(f"  [{meter.id}] ❌ No response from meter")
                    row_complete = False
                    break

                print(f"  [{meter.id}] Got raw data, shape: {raw.shape if hasattr(raw, 'shape') else len(raw)}")

                # Extract features
                features = select_features(raw, drop_indices_0_based=pipeline.features.drop_indices_0_based)
                if features.shape[0] != len(pipeline.features.names):
                    print(f"  [{meter.id}] ❌ Feature size mismatch: got {features.shape[0]}, expected {len(pipeline.features.names)}")
                    row_complete = False
                    break

                # Check for invalid raw features and replace NaN with 0
                if not np.isfinite(features).all():
                    print(f"  [{meter.id}] Raw features contain NaN/Inf, replacing with 0: {features}")
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    print(f"  [{meter.id}] Features OK (no NaN/Inf)")

                # Apply normalization if enabled
                if pipeline.per_unit.enabled:
                    baseline = runtimes[meter.id].per_unit.update_and_get_baseline(
                        features, baseline_samples=pipeline.per_unit.baseline_samples
                    )
                    if baseline is None:
                        print(f"  [{meter.id}] Waiting for baseline samples...")
                        row_complete = False
                        break
                    print(f"  [{meter.id}] Normalizing with baseline")
                    features = to_per_unit(
                        features,
                        per_unit_feature_count=pipeline.per_unit.per_unit_feature_count,
                        baseline=baseline,
                    )

                # Skip invalid data after normalization
                if not np.isfinite(features).all():
                    print(f"  [{meter.id}] After normalization: NaN/Inf: {features}")
                    row_complete = False
                    break

                # Add this meter's features to the row
                row_data.extend([f"{v:.6g}" for v in features.tolist()])
                print(f"  [{meter.id}] Data OK")

            # Only write row if all meters contributed
            if row_complete:
                writer.writerow(row_data)
                rows += 1
                print(f"✓ Row {rows} written\n")
            else:
                print(f"✗ Row incomplete, skipping\n")

            csvfile.flush()

            # Pace polling
            elapsed = time.monotonic() - loop_start
            sleep_s = meters_cfg.poll_interval_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    finally:
        csvfile.close()
        for client in clients.values():
            try:
                client.close()
            except Exception:
                pass

    print(f"✓ Collected {rows} rows")
    print(f"✓ Saved to: {out_file}\n")
    return 0


if __name__ == "__main__":
    import sys
    
    # Simple args: duration and class label
    duration = 60.0
    label = 0
    
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
        except ValueError:
            pass
    
    if len(sys.argv) > 2:
        try:
            label = int(sys.argv[2])
        except ValueError:
            pass
    
    exit_code = collect_data(duration_s=duration, class_label=label)
    raise SystemExit(exit_code)

