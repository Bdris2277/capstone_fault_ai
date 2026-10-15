"""
Data preparation script - combines collected CSVs and splits into train/test.

Usage:
    python prepare_data.py

This script:
1. Finds all CSV files in data_out/ (normal_1.csv, fault_1.csv, etc.)
2. Combines them into one file
3. Shuffles randomly
4. Splits 80% train, 20% test
5. Saves to data_out/train_data.csv and data_out/test_data.csv

Ready for training!
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from pathlib import Path
import pandas as pd
from capstone_fault_ai.app import user_params


def main():
    data_dir = Path(user_params.COLLECT_OUT_CSV).parent
    
    print("\n" + "="*60)
    print("DATA PREPARATION")
    print("="*60)
    print(f"Looking for CSV files in: {data_dir}\n")
    
    # Find all CSV files except train/test splits
    csv_files = sorted([
        f for f in data_dir.glob("*.csv")
        if f.name not in ["train_data.csv", "test_data.csv"]
        and f.name.startswith(tuple(user_params.CLASS_LABEL_NAMES.values()))
    ])
    
    if not csv_files:
        print("❌ No data files found!")
        print(f"   Expected files like: normal_1.csv, normal_2.csv, fault_1.csv, etc.")
        return 1
    
    print(f"Found {len(csv_files)} data files:")
    for f in csv_files:
        print(f"  ✓ {f.name}")
    print()
    
    # Combine all CSVs
    all_data = []
    total_rows = 0
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_data.append(df)
        total_rows += len(df)
        print(f"  Loaded {csv_file.name}: {len(df)} rows")
    
    print(f"\nTotal rows: {total_rows}")
    
    # Combine
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined: {len(combined_df)} rows")
    
    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Shuffled")
    
    # Split 80/20
    split_idx = int(len(combined_df) * 0.8)
    train_df = combined_df[:split_idx]
    test_df = combined_df[split_idx:]
    
    # Save
    train_file = data_dir / "train_data.csv"
    test_file = data_dir / "test_data.csv"
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\n✓ Saved {train_file.name}: {len(train_df)} rows")
    print(f"✓ Saved {test_file.name}: {len(test_df)} rows")
    print("\n" + "="*60)
    print("Ready to train! Run:")
    print("  python -m capstone_fault_ai.app.train")
    print("="*60 + "\n")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
