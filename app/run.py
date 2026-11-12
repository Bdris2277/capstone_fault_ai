"""
Main application launcher.
Interactive menu is the default behavior.
Use --auto for one-click training pipeline.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

import argparse

from capstone_fault_ai.app import user_params
from capstone_fault_ai.app.collect import collect_data
from capstone_fault_ai.app.infer import main as infer_main
from capstone_fault_ai.app.prepare_data import main as prepare_main
from capstone_fault_ai.app.train import main as train_main


def show_menu():
    """Display menu and get user choice."""
    print("\n" + "="*50)
    print("AI FAULT DETECTION")
    print("="*50)
    print("1. Collect Data")
    print("2. Prepare Data (window + split)")
    print("3. Train Fault-Type Model (RandomForest)")
    print("4. Train Location Model (RandomForest)")
    print("5. Run Live Inference (Fault Type)")
    print("6. Run Live Inference (Location)")
    print("0. Exit")
    print("="*50)
    
    choice = input("What do you want to do? (0-6): ").strip()
    return choice


def run_full_training_pipeline() -> int:
    """Prepare data, train class model, then train location model."""
    print("\n" + "=" * 60)
    print("ONE-CLICK TRAINING PIPELINE")
    print("=" * 60)

    print("\n[1/3] Preparing data...")
    rc = prepare_main()
    if rc != 0:
        print(f"Prepare step failed with exit code {rc}.")
        return rc

    print("\n[2/3] Training fault-type model...")
    rc = train_main(["--target", "class"])
    if rc != 0:
        print(f"Class training failed with exit code {rc}.")
        return rc

    print("\n[3/3] Training location model...")
    rc = train_main(["--target", "location"])
    if rc != 0:
        print(f"Location training failed with exit code {rc}.")
        return rc

    print("\nDone. Artifacts written to:")
    print(f"  {user_params.ARTIFACTS_DIR}")
    print("=" * 60 + "\n")
    return 0


def main(argv: list[str] | None = None):
    """Main application loop."""
    parser = argparse.ArgumentParser(
        description="Run interactive menu (default) or one-click training pipeline."
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run one-click pipeline: prepare -> train class -> train location.",
    )
    parser.add_argument(
        "--collect-first",
        action="store_true",
        help="With --auto, collect once using user_params before prepare/train.",
    )
    args = parser.parse_args(argv)

    if args.auto:
        if args.collect_first:
            duration = user_params.COLLECT_DURATION_S
            label = user_params.COLLECT_CLASS_LABEL
            ms_tag = getattr(user_params, "COLLECT_FILENAME_MS_TAG", None)
            print("\nCollecting before training...")
            rc = collect_data(duration_s=duration, class_label=label, filename_ms_tag=ms_tag)
            if rc != 0:
                return rc
        return run_full_training_pipeline()

    while True:
        choice = show_menu()

        if choice == "0":
            print("\nGoodbye.")
            return 0

        if choice == "1":
            duration = user_params.COLLECT_DURATION_S
            label = user_params.COLLECT_CLASS_LABEL
            ms_tag = getattr(user_params, "COLLECT_FILENAME_MS_TAG", None)

            print("\n" + "="*50)
            print("METER DATA COLLECTION")
            print("="*50)
            print(f"Duration: {duration}s")
            print(f"Class label: {label}")
            print(f"Filename ms tag: {ms_tag}")
            print("="*50 + "\n")

            rc = collect_data(duration_s=duration, class_label=label, filename_ms_tag=ms_tag)
            print(f"Collection finished with code: {rc}")

        elif choice == "2":
            print("\nPreparing data...")
            rc = prepare_main()
            print(f"Prepare finished with code: {rc}")

        elif choice == "3":
            print("\nTraining fault-type model...")
            rc = train_main(["--target", "class"])
            print(f"Class training finished with code: {rc}")

        elif choice == "4":
            print("\nTraining location model...")
            rc = train_main(["--target", "location"])
            print(f"Location training finished with code: {rc}")

        elif choice == "5":
            print("\nStarting live inference (fault type)...")
            rc = infer_main(["--target", "class"])
            print(f"Fault-type inference ended with code: {rc}")

        elif choice == "6":
            print("\nStarting live inference (location)...")
            rc = infer_main(["--target", "location"])
            print(f"Location inference ended with code: {rc}")

        else:
            print("\nInvalid choice")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

