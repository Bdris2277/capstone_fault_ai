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

    # Setup CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = Path(user_params.COLLECT_OUT_CSV).parent / f"collected_{timestamp}.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # CSV header
    header = ["class", "meter_id", *pipeline.features.names]

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
                    print(f"  [{meter.id}] No response from meter")
                    continue

                # Extract features
                features = select_features(raw, drop_indices_0_based=pipeline.features.drop_indices_0_based)
                if features.shape[0] != len(pipeline.features.names):
                    print(f"  [{meter.id}] Feature size mismatch: got {features.shape[0]}, expected {len(pipeline.features.names)}")
                    continue

                # Apply normalization if enabled
                if pipeline.per_unit.enabled:
                    baseline = runtimes[meter.id].per_unit.update_and_get_baseline(
                        features, baseline_samples=pipeline.per_unit.baseline_samples
                    )
                    if baseline is None:
                        continue
                    features = to_per_unit(
                        features,
                        per_unit_feature_count=pipeline.per_unit.per_unit_feature_count,
                        baseline=baseline,
                    )

                # Skip invalid data
                if not np.isfinite(features).all():
                    print(f"  [{meter.id}] Invalid data (NaN or Inf)")
                    continue

                # Write row
                row = [class_label, meter.id, *[f"{v:.6g}" for v in features.tolist()]]
                writer.writerow(row)
                rows += 1
                print(f"  [{meter.id}] Row {rows}")

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

