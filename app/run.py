from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

import shutil
import json
from pathlib import Path
import subprocess
import sys

from capstone_fault_ai.app import user_params


def _prompt(text: str, *, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{text}{suffix}: ").strip()
    return value if value else (default or "")


def _prompt_int(text: str, *, default: int) -> int:
    while True:
        raw = _prompt(text, default=str(default))
        try:
            return int(raw)
        except ValueError:
            print("Enter an integer.")


def _prompt_float(text: str, *, default: float) -> float:
    while True:
        raw = _prompt(text, default=str(default))
        try:
            return float(raw)
        except ValueError:
            print("Enter a number.")


def _prompt_yes_no(text: str, *, default: bool) -> bool:
    default_str = "y" if default else "n"
    while True:
        raw = _prompt(f"{text} (y/n)", default=default_str).lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Enter y or n.")


def _ensure_default_configs() -> None:
    examples = Path("config_examples")
    if not examples.exists():
        return

    mapping = {
        examples / "meters.example.json": Path("meters.json"),
        examples / "pipeline.example.json": Path("pipeline.json"),
    }

    created: list[Path] = []
    for src, dst in mapping.items():
        if src.exists() and not dst.exists():
            shutil.copyfile(src, dst)
            created.append(dst)

    if created:
        print("Created default config files:")
        for path in created:
            print(f" - {path}")
        print("Edit them (especially meter IPs), then run again.\n")


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2)
        file.write("\n")


def _ensure_configs_exist() -> None:
    _ensure_default_configs()
    missing: list[str] = []
    if not Path("meters.json").exists():
        missing.append("meters.json")
    if not Path("pipeline.json").exists():
        missing.append("pipeline.json")
    if missing:
        print("Missing config file(s): " + ", ".join(missing))
        print("Create them by copying from config_examples/ and editing them.\n")


def _install_offline_deps() -> None:
    req = Path(getattr(user_params, "OFFLINE_REQUIREMENTS", "offline/requirements_jetson.txt"))
    wheels = Path(getattr(user_params, "OFFLINE_WHEELS_DIR", "offline/wheels"))

    if not req.exists():
        print(f"Offline requirements file not found: {req}")
        return
    if not wheels.exists():
        print(f"Offline wheels folder not found: {wheels}")
        print("Create it and copy wheels into it, then run Setup again.")
        return

    print("\nInstalling dependencies from offline bundle...")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(wheels),
        "-r",
        str(req),
    ]
    try:
        subprocess.run(cmd, check=True)
        print("Offline dependency install complete.\n")
    except subprocess.CalledProcessError as exc:
        print("\nOffline dependency install failed.")
        print(f"Command: {' '.join(cmd)}")
        print("Make sure the correct Jetson (aarch64) wheels are present in offline/wheels/.")
        print(f"pip exit code: {exc.returncode}\n")


def _setup_dependencies() -> None:
    print("\nSetup (dependencies only)\n")
    if bool(getattr(user_params, "AUTO_INSTALL_OFFLINE_DEPS", False)):
        _install_offline_deps()
    else:
        print("AUTO_INSTALL_OFFLINE_DEPS is disabled in app/user_params.py.\n")


def _load_or_setup_if_missing() -> None:
    _ensure_configs_exist()


def _labels_path() -> Path:
    return Path("labels.json")


def _load_labels() -> dict[str, int]:
    path = _labels_path()
    if not path.exists():
        return {}
    raw = _read_json(path)
    labels: dict[str, int] = {}
    for k, v in raw.items():
        try:
            labels[str(k)] = int(v)
        except Exception:
            continue
    return labels


def _save_labels(labels: dict[str, int]) -> None:
    _write_json(_labels_path(), dict(sorted(labels.items(), key=lambda kv: kv[1])))


def _pick_class_label() -> int:
    labels = _load_labels()
    if labels:
        print("\nKnown labels:")
        for name, idx in sorted(labels.items(), key=lambda kv: kv[1]):
            print(f" - {idx}: {name}")
        print()

    name = _prompt("Label name (e.g. normal, fault_a)", default="normal")
    if name in labels:
        return labels[name]

    next_id = (max(labels.values()) + 1) if labels else 0
    labels[name] = next_id
    _save_labels(labels)
    print(f"Saved label '{name}' as class {next_id} (labels.json).")
    return next_id


def _run_collect() -> int:
    from capstone_fault_ai.app.collect import main as collect_main

    _load_or_setup_if_missing()

    meters_json = user_params.METERS_JSON
    pipeline_json = user_params.PIPELINE_JSON
    duration_s = _prompt_float("duration_s", default=30.0)

    class_label = int(user_params.COLLECT_CLASS_LABEL)
    out_csv = user_params.COLLECT_OUT_CSV
    append = bool(user_params.COLLECT_APPEND_IF_EXISTS) and Path(out_csv).exists()
    split = False

    argv: list[str] = [
        "--meters_json",
        meters_json,
        "--pipeline_json",
        pipeline_json,
        "--class_label",
        str(class_label),
        "--duration_s",
        str(duration_s),
    ]
    if split:
        out_dir = _prompt("out_dir", default="data_out")
        argv += ["--split_by_meter", "--out_dir", out_dir]
    else:
        argv += ["--out_csv", out_csv]

    if bool(user_params.COLLECT_INCLUDE_METER_ID) and (not split):
        argv.append("--include_meter_id")

    if append:
        argv.append("--append")

    return int(collect_main(argv))


def _run_train() -> int:
    from capstone_fault_ai.app.train import main as train_main

    collected_csv = user_params.COLLECTED_CSV
    train_csv = user_params.TRAIN_CSV
    test_csv = user_params.TEST_CSV
    out_dir = user_params.ARTIFACTS_DIR

    epochs = int(user_params.EPOCHS)
    batch_size = int(user_params.BATCH_SIZE)
    lr = float(user_params.LR)

    if not Path(train_csv).exists() or not Path(test_csv).exists():
        if Path(collected_csv).exists():
            import pandas as pd
            from sklearn.model_selection import train_test_split

            df = pd.read_csv(collected_csv)
            if "class" not in df.columns:
                raise ValueError("collected_csv must contain a 'class' column.")
            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["class"] if df["class"].nunique() > 1 else None,
            )
            Path(train_csv).parent.mkdir(parents=True, exist_ok=True)
            Path(test_csv).parent.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(train_csv, index=False)
            test_df.to_csv(test_csv, index=False)
            print(f"Wrote {train_csv} and {test_csv}.")
        else:
            print("Missing train/test CSVs. Collect data first, or provide existing train/test paths.")
            return 2

    argv = [
        "--train_csv",
        train_csv,
        "--test_csv",
        test_csv,
        "--out_dir",
        out_dir,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--lr",
        str(lr),
    ]
    return int(train_main(argv))


def _run_infer() -> int:
    from capstone_fault_ai.app.infer import main as infer_main

    _load_or_setup_if_missing()

    meters_json = user_params.METERS_JSON
    pipeline_json = user_params.PIPELINE_JSON
    artifacts_dir = user_params.ARTIFACTS_DIR
    out_csv = user_params.INFER_OUT_CSV

    argv = [
        "--meters_json",
        meters_json,
        "--pipeline_json",
        pipeline_json,
        "--artifacts_dir",
        artifacts_dir,
        "--out_csv",
        out_csv,
    ]
    return int(infer_main(argv))


def main() -> int:
    while True:
        print("Capstone Fault AI - one-click runner")
        print("  0) Setup (dependencies)")
        print("  1) Collect labeled data")
        print("  2) Train model")
        print("  3) Live inference")
        print("  4) Quit")
        choice = _prompt("Choose", default="1")

        if choice == "0":
            _setup_dependencies()
        elif choice == "1":
            _run_collect()
        elif choice == "2":
            _run_train()
        elif choice == "3":
            _run_infer()
        elif choice == "4":
            return 0
        else:
            print("Pick 0-4.")

        print()


if __name__ == "__main__":
    raise SystemExit(main())
