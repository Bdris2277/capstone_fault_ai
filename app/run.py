from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

import json
from pathlib import Path

from capstone_fault_ai.app import user_params


def _prompt(text: str, *, default: str | None = None) -> str:
    """Prompt user for input with optional default."""
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{text}{suffix}: ").strip()
    return value if value else (default or "")


def _prompt_int(text: str, *, default: int) -> int:
    """Prompt user for integer input."""
    while True:
        raw = _prompt(text, default=str(default))
        try:
            return int(raw)
        except ValueError:
            print("Enter an integer.")


def _prompt_float(text: str, *, default: float) -> float:
    """Prompt user for float input."""
    while True:
        raw = _prompt(text, default=str(default))
        try:
            return float(raw)
        except ValueError:
            print("Enter a number.")


def _labels_path() -> Path:
    return Path("labels.json")


def _load_labels() -> dict[str, int]:
    """Load class labels from labels.json."""
    path = _labels_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {str(k): int(v) for k, v in raw.items()}
    except Exception:
        return {}


def _save_labels(labels: dict[str, int]) -> None:
    """Save class labels to labels.json."""
    path = _labels_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(labels.items(), key=lambda kv: kv[1])), f, indent=2)
        f.write("\n")


def _pick_class_label() -> int:
    """Show known labels and let user pick or create one."""
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
    print(f"Saved label '{name}' as class {next_id}.")
    return next_id


def _run_collect() -> int:
    """Run collect.py with user prompts."""
    from capstone_fault_ai.app.collect import main as collect_main

    duration_s = _prompt_float("Collection duration (seconds)", default=30.0)
    class_label = _pick_class_label()

    argv = [
        "--meters_json",
        user_params.METERS_JSON,
        "--pipeline_json",
        user_params.PIPELINE_JSON,
        "--class_label",
        str(class_label),
        "--duration_s",
        str(duration_s),
        "--out_csv",
        user_params.COLLECT_OUT_CSV,
    ]

    if bool(user_params.COLLECT_INCLUDE_METER_ID):
        argv.append("--include_meter_id")

    return int(collect_main(argv))


def _run_train() -> int:
    """Run train.py."""
    from capstone_fault_ai.app.train import main as train_main

    collected_csv = user_params.COLLECTED_CSV
    train_csv = user_params.TRAIN_CSV
    test_csv = user_params.TEST_CSV
    out_dir = user_params.ARTIFACTS_DIR

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
            print(f"ERROR: Missing {collected_csv}. Collect data first.")
            return 2

    argv = [
        "--train_csv",
        train_csv,
        "--test_csv",
        test_csv,
        "--out_dir",
        out_dir,
        "--epochs",
        str(user_params.EPOCHS),
        "--batch_size",
        str(user_params.BATCH_SIZE),
        "--lr",
        str(user_params.LR),
    ]
    return int(train_main(argv))


def _run_infer() -> int:
    """Run infer.py."""
    from capstone_fault_ai.app.infer import main as infer_main

    argv = [
        "--meters_json",
        user_params.METERS_JSON,
        "--pipeline_json",
        user_params.PIPELINE_JSON,
        "--artifacts_dir",
        user_params.ARTIFACTS_DIR,
        "--out_csv",
        user_params.INFER_OUT_CSV,
    ]
    return int(infer_main(argv))


def main() -> int:
    """Main menu."""
    while True:
        print("\nCapstone Fault AI")
        print("  1) Collect labeled data")
        print("  2) Train model")
        print("  3) Live inference")
        print("  4) Quit")
        choice = _prompt("Choose", default="1")

        if choice == "1":
            _run_collect()
        elif choice == "2":
            _run_train()
        elif choice == "3":
            _run_infer()
        elif choice == "4":
            return 0
        else:
            print("Pick 1-4.")


if __name__ == "__main__":
    raise SystemExit(main())
