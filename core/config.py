from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class MeterConfig:
    id: str
    ip: str
    enabled: bool = True
    port: int = 502
    unit_id: int = 1
    timeout_s: float = 3.0


@dataclass(frozen=True)
class MetersConfig:
    poll_interval_s: float
    meters: list[MeterConfig]


@dataclass(frozen=True)
class ModbusConfig:
    address: int = 1
    count: int = 48


@dataclass(frozen=True)
class DecodeConfig:
    float_endianness: str = "big"  # "big" or "little"


@dataclass(frozen=True)
class FeaturesConfig:
    drop_indices_0_based: list[int]
    names: list[str]


@dataclass(frozen=True)
class PerUnitConfig:
    enabled: bool = True
    per_unit_feature_count: int = 8
    baseline_samples: int = 1


@dataclass(frozen=True)
class MeterTimestampConfig:
    enabled: bool = False
    address: int = 483
    count: int = 4
    format: str = "pac4200_unix_u32"  # "pac4200_unix_u32", "u64_be", or "u64_le"


@dataclass(frozen=True)
class PipelineConfig:
    modbus: ModbusConfig
    decode: DecodeConfig
    features: FeaturesConfig
    per_unit: PerUnitConfig
    meter_timestamp: MeterTimestampConfig


def _read_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _require_keys(obj: dict[str, Any], *, keys: Iterable[str], context: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise ValueError(f"Missing keys in {context}: {', '.join(missing)}")


def load_meters_config(path: str | Path) -> MetersConfig:
    raw = _read_json(path)
    _require_keys(raw, keys=["poll_interval_s", "meters"], context=str(path))

    meters: list[MeterConfig] = []
    for meter in raw["meters"]:
        _require_keys(meter, keys=["id", "ip"], context=f"{path} meters[]")
        meters.append(
            MeterConfig(
                id=str(meter["id"]),
                ip=str(meter["ip"]),
                enabled=bool(meter.get("enabled", True)),
                port=int(meter.get("port", 502)),
                unit_id=int(meter.get("unit_id", 1)),
                timeout_s=float(meter.get("timeout_s", 3.0)),
            )
        )

    enabled_meters = [m for m in meters if m.enabled]
    if not enabled_meters:
        known = ", ".join([m.id for m in meters]) if meters else "(none)"
        raise ValueError(f"No enabled meters in {path}. Known meter IDs: {known}")

    return MetersConfig(poll_interval_s=float(raw["poll_interval_s"]), meters=enabled_meters)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    raw = _read_json(path)
    _require_keys(raw, keys=["modbus", "decode", "features", "per_unit"], context=str(path))

    features = raw["features"]
    _require_keys(features, keys=["drop_indices_0_based", "names"], context=f"{path} features")

    per_unit = raw["per_unit"]
    _require_keys(per_unit, keys=["enabled", "per_unit_feature_count", "baseline_samples"], context=f"{path} per_unit")

    meter_ts_raw = raw.get("meter_timestamp", {})

    return PipelineConfig(
        modbus=ModbusConfig(address=int(raw["modbus"].get("address", 1)), count=int(raw["modbus"].get("count", 48))),
        decode=DecodeConfig(float_endianness=str(raw["decode"].get("float_endianness", "big"))),
        features=FeaturesConfig(
            drop_indices_0_based=[int(i) for i in features["drop_indices_0_based"]],
            names=[str(n).strip() for n in features["names"]],
        ),
        per_unit=PerUnitConfig(
            enabled=bool(per_unit["enabled"]),
            per_unit_feature_count=int(per_unit["per_unit_feature_count"]),
            baseline_samples=int(per_unit["baseline_samples"]),
        ),
        meter_timestamp=MeterTimestampConfig(
            enabled=bool(meter_ts_raw.get("enabled", False)),
            address=int(meter_ts_raw.get("address", 483)),
            count=int(meter_ts_raw.get("count", 4)),
            format=str(meter_ts_raw.get("format", "pac4200_unix_u32")).strip().lower(),
        ),
    )
