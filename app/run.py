"""
Main application launcher.
Shows menu and runs selected operation.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

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
    print("="*50)
    
    choice = input("What do you want to do? (1-6): ").strip()
    return choice


def main():
    """Main application loop."""
    choice = show_menu()
    
    if choice == "1":
        duration = user_params.COLLECT_DURATION_S
        label = user_params.COLLECT_CLASS_LABEL
        
        print("\n" + "="*50)
        print("METER DATA COLLECTION")
        print("="*50)
        print(f"Duration: {duration}s")
        print(f"Class label: {label}")
        print("="*50 + "\n")
        
        collect_data(duration_s=duration, class_label=label)
    
    elif choice == "2":
        print("\nPreparing data...")
        raise SystemExit(prepare_main())
    
    elif choice == "3":
        print("\nTraining fault-type model...")
        raise SystemExit(train_main(["--target", "class"]))

    elif choice == "4":
        print("\nTraining location model...")
        raise SystemExit(train_main(["--target", "location"]))

    elif choice == "5":
        print("\nStarting live inference (fault type)...")
        raise SystemExit(infer_main(["--target", "class"]))

    elif choice == "6":
        print("\nStarting live inference (location)...")
        raise SystemExit(infer_main(["--target", "location"]))
    
    else:
        print("\nInvalid choice")


if __name__ == "__main__":
    main()

