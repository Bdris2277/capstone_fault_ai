from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

import json
from pathlib import Path
from capstone_fault_ai.app import user_params


# ============================================================================
# UTILITIES
# ============================================================================

def ask(prompt: str, default: str = "") -> str:
    """Ask user a question."""
    suffix = f" [{default}]" if default else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer if answer else default


def ask_float(prompt: str, default: float) -> float:
    """Ask user for a number."""
    while True:
        try:
            return float(ask(prompt, str(default)))
        except ValueError:
            print("Enter a number.")


# ============================================================================
# CLASS LABELS (normal, fault_a, etc)
# ============================================================================

def get_class_label() -> int:
    """Ask user to pick or create a class label."""
    path = Path("labels.json")
    
    # Load existing labels
    labels = {}
    if path.exists():
        try:
            labels = json.loads(path.read_text())
        except Exception:
            pass
    
    # Show existing labels
    if labels:
        print("\nExisting labels:")
        for name, id in sorted(labels.items(), key=lambda x: x[1]):
            print(f"  {id}: {name}")
        print()
    
    # Get label from user
    name = ask("Label name (e.g. normal, fault_a)", default="normal")
    
    # Return existing label ID
    if name in labels:
        return labels[name]
    
    # Create new label
    new_id = max(labels.values()) + 1 if labels else 0
    labels[name] = new_id
    path.write_text(json.dumps(labels, indent=2) + "\n")
    print(f"Created label '{name}' = {new_id}\n")
    return new_id


# ============================================================================
# MAIN COMMANDS
# ============================================================================

def cmd_collect():
    """Collect data from meters."""
    from capstone_fault_ai.app.collect import main as collect_main
    
    duration = ask_float("Duration (seconds)", default=30.0)
    label = get_class_label()
    
    collect_main([
        "--meters_json", user_params.METERS_JSON,
        "--pipeline_json", user_params.PIPELINE_JSON,
        "--class_label", str(label),
        "--duration_s", str(duration),
        "--out_csv", user_params.COLLECT_OUT_CSV,
    ])


def cmd_train():
    """Train the model."""
    from capstone_fault_ai.app.train import main as train_main
    
    # Auto-split data if needed
    collected = Path(user_params.COLLECTED_CSV)
    train_path = Path(user_params.TRAIN_CSV)
    test_path = Path(user_params.TEST_CSV)
    
    if not train_path.exists() or not test_path.exists():
        if collected.exists():
            print("Splitting data into train/test...")
            import pandas as pd
            from sklearn.model_selection import train_test_split
            
            df = pd.read_csv(collected)
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            
            train_path.parent.mkdir(parents=True, exist_ok=True)
            test_path.parent.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            print(f"  {train_path}")
            print(f"  {test_path}\n")
        else:
            print(f"ERROR: No collected data at {collected}\n")
            return
    
    train_main([
        "--train_csv", str(train_path),
        "--test_csv", str(test_path),
        "--out_dir", user_params.ARTIFACTS_DIR,
        "--epochs", str(user_params.EPOCHS),
        "--batch_size", str(user_params.BATCH_SIZE),
        "--lr", str(user_params.LR),
    ])


def cmd_infer():
    """Run live inference."""
    from capstone_fault_ai.app.infer import main as infer_main
    
    infer_main([
        "--meters_json", user_params.METERS_JSON,
        "--pipeline_json", user_params.PIPELINE_JSON,
        "--artifacts_dir", user_params.ARTIFACTS_DIR,
        "--out_csv", user_params.INFER_OUT_CSV,
    ])


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    """Main menu loop."""
    while True:
        print("\n" + "="*40)
        print("Capstone Fault AI")
        print("="*40)
        print("1) Collect labeled data")
        print("2) Train model")
        print("3) Live inference")
        print("4) Quit")
        print()
        
        choice = ask("Choose", default="1")
        
        if choice == "1":
            cmd_collect()
        elif choice == "2":
            cmd_train()
        elif choice == "3":
            cmd_infer()
        elif choice == "4":
            print("Goodbye!\n")
            return
        else:
            print("Invalid choice.\n")


if __name__ == "__main__":
    main()
