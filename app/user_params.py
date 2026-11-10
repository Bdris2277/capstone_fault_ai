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

FAULT_TYPES = {
    0:  "normal",           # The good/healthy state

    1:  "fault_L1_0",       # Voltage Sag Line 1 0%
    2:  "fault_L1_20",      # Voltage Sag Line 1 20%
    3:  "fault_L1_40",      # Voltage Sag Line 1 40%
    4:  "fault_L1_60",      # Voltage Sag Line 1 60%
    5:  "fault_L1_100",      # Voltage Sag Line 1 80%
    
    6:  "fault_L2_0",       # Voltage Sag Line 2 0%
    7:  "fault_L2_20",      # Voltage Sag Line 2 20%
    8:  "fault_L2_40",      # Voltage Sag Line 2 40%
    9:  "fault_L2_60",      # Voltage Sag Line 2 60%
    10: "fault_L2_100",      # Voltage Sag Line 2 80%

    11: "fault_L3_0",       # Voltage Sag Line 3 0%
    12: "fault_L3_20",      # Voltage Sag Line 3 20%
    13: "fault_L3_40",      # Voltage Sag Line 3 40%
    14: "fault_L3_60",      # Voltage Sag Line 3 60%
    15: "fault_L3_100",      # Voltage Sag Line 3 80%
}

# ============================================================================
# STEP 2: Collection settings (how long to collect data)
# ============================================================================
COLLECT_DURATION_S = 120.0      # Seconds E.g(10 = collect for 10.0 seconds)

COLLECT_CLASS_LABEL = 5
COLLECT_FILENAME_MS_TAG = 200   # Optional filename suffix in ms (e.g. fault_L1_20_1_200.csv). Set to None to disable.
COLLECT_FAULT_LOCATION = "PhotoVoltaic"  # Fault location label to store in collected CSV

# ============================================================================
# STEP 3: Training settings (how should the model learn)
# ============================================================================
# RandomForest settings (used by app/train.py)
RF_N_ESTIMATORS = 400      # More trees = usually better, but slower training
RF_MAX_DEPTH = 0           # 0 = unlimited depth (stronger, slower)
RF_MIN_SAMPLES_LEAF = 1    # Higher = smoother model, faster, less overfit
RF_OVERSAMPLE_RATIO = 0.8  # Fault-class upsampling ratio vs majority class (0 disables)
RF_RANDOM_STATE = 42

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
