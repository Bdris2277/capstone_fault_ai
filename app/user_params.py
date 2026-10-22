from __future__ import annotations

"""
CONFIGURATION FILE 

Edit ONLY the values in STEP 1, 2, and 3 below.
Everything else is automatic.
"""

# ============================================================================
# STEP 1: Define your fault types & Location (what should we detect and where)
# ============================================================================
# Change the names to match what you're testing.
# Example: 0=healthy, 1=under_voltage, 2=over_voltage

# Fault location options (set COLLECT_FAULT_LOCATION to one meter ID):
#   - Wind
#   - BalancePoint
#   - TransmissionLineEnd
#   - ConsumerII
#   - PhotoVoltaic
# For class 0 (normal), location is ignored and saved as "none" automatically.

COLLECT_FAULT_LOCATION = "PhotoVoltaic"  # Fault location label to store in collected CSV
FAULT_TYPES = {
    0:  "normal",           # The good/healthy state

    1:  "fault_L1_0",       # Voltage Sag Line 1 0%
    2:  "fault_L1_20",      # Voltage Sag Line 1 20%
    3:  "fault_L1_40",      # Voltage Sag Line 1 40%
    4:  "fault_L1_60",      # Voltage Sag Line 1 60%
    5:  "fault_L1_80",      # Voltage Sag Line 1 80%
    
    6:  "fault_L2_0",       # Voltage Sag Line 2 0%
    7:  "fault_L2_20",      # Voltage Sag Line 2 20%
    8:  "fault_L2_40",      # Voltage Sag Line 2 40%
    9:  "fault_L2_60",      # Voltage Sag Line 2 60%
    10: "fault_L2_80",      # Voltage Sag Line 2 80%

    11: "fault_L3_0",       # Voltage Sag Line 3 0%
    12: "fault_L3_20",      # Voltage Sag Line 3 20%
    13: "fault_L3_40",      # Voltage Sag Line 3 40%
    14: "fault_L3_60",      # Voltage Sag Line 3 60%
    15: "fault_L3_80",      # Voltage Sag Line 3 80%
}

# ============================================================================
# STEP 2: Collection settings (how long to collect data)
# ============================================================================
COLLECT_DURATION_S = 180.0      # Seconds E.g(10 = collect for 10.0 seconds)
COLLECT_CLASS_LABEL = 2         # Which fault type are you collecting now?
                                # Change to 1, 2, etc. when you test different faults

# ============================================================================
# STEP 3: Training settings (how should the model learn)
# ============================================================================
EPOCHS = 50              # Training cycles (higher = better, slower)
BATCH_SIZE = 64          # Samples per cycle (64 is typical)
LR = 1e-3                # Learning speed (0.001 is typical)

# ============================================================================
# CONFIG FILES (don't change anything below this line PLEASE)
# ============================================================================
METERS_JSON = "Config_Files/meters.json"
PIPELINE_JSON = "Config_Files/pipeline.json"

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
