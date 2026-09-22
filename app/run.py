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


def show_menu():
    """Display menu and get user choice."""
    print("\n" + "="*50)
    print("AI FAULT DETECTION")
    print("="*50)
    print("1. Collect Data")
    print("2. Train Model")
    print("3. Run Inference")
    print("="*50)
    
    choice = input("What do you want to do? (1-3): ").strip()
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
        print("\nTrain Model - coming soon")
    
    elif choice == "3":
        print("\nRun Inference - coming soon")
    
    else:
        print("\nInvalid choice")


if __name__ == "__main__":
    main()

