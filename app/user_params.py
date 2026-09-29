from __future__ import annotations

"""
CONFIGURATION FILE - Easy setup for everyone!

Edit ONLY the values in STEP 1, 2, and 3 below.
Everything else is automatic. No programming knowledge needed!
"""

# ============================================================================
# STEP 1: Define your fault types (what should we detect)
# ============================================================================
# Change the names to match what you're testing.
# Example: 0=healthy, 1=under_voltage, 2=over_voltage

FAULT_TYPES = {
    0: "normal",           # The good/healthy state
    1: "fault",            # First fault type
    2: "other_fault",      # Second fault type (add more if needed)
}

# ============================================================================
# STEP 2: Collection settings (how long to collect data)
# ============================================================================
COLLECT_DURATION_S = 10.0      # Seconds (10 = collect for 10 seconds)
COLLECT_CLASS_LABEL = 0        # Which fault type are you collecting now?
                               # Change to 1, 2, etc. when you test different faults

# ============================================================================
# STEP 3: Training settings (how should the model learn)
# ============================================================================
EPOCHS = 50              # Training cycles (higher = better, slower)
BATCH_SIZE = 64          # Samples per cycle (64 is typical)
LR = 1e-3                # Learning speed (0.001 is typical)

# ============================================================================
# CONFIG FILES (don't change these)
# ============================================================================
METERS_JSON = "meters.json"
PIPELINE_JSON = "pipeline.json"

# Data paths (automatic)
COLLECT_OUT_CSV = "data_out/collected.csv"
TRAIN_CSV = "data_out/train_data.csv"
TEST_CSV = "data_out/test_data.csv"
ARTIFACTS_DIR = "capstone_fault_ai_artifacts"
INFER_OUT_CSV = "capstone_fault_ai_artifacts/predictions_live.csv"

# Offline Jetson setup
AUTO_INSTALL_OFFLINE_DEPS = True
OFFLINE_REQUIREMENTS = "offline/requirements_jetson.txt"
OFFLINE_WHEELS_DIR = "offline/wheels"

# Internal use (for compatibility)
CLASS_LABEL_NAMES = FAULT_TYPES
COLLECTED_CSV = COLLECT_OUT_CSV
COLLECT_APPEND_IF_EXISTS = True
COLLECT_INCLUDE_METER_ID = True
