"""
Simple launcher for meter data collection.
Just run this to collect data from all enabled meters.

Settings are in app/user_params.py:
  - COLLECT_DURATION_S: How long to collect (seconds)
  - COLLECT_CLASS_LABEL: Class label for this data
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from capstone_fault_ai.app import user_params
from capstone_fault_ai.app.collect import collect_data


def main():
    """Collect data with hardcoded settings."""
    duration = user_params.COLLECT_DURATION_S
    label = user_params.COLLECT_CLASS_LABEL
    
    print("\n" + "="*50)
    print("METER DATA COLLECTION")
    print("="*50)
    print(f"Duration: {duration}s")
    print(f"Class label: {label}")
    print("="*50 + "\n")
    
    collect_data(duration_s=duration, class_label=label)


if __name__ == "__main__":
    main()

