from __future__ import annotations

# Edit this file to match your environment.
# Goal: after initial setup, you can click "Run" on scripts with minimal prompts.

# Config files (usually don’t change often)
METERS_JSON = "meters.json"
PIPELINE_JSON = "pipeline.json"

# Offline install (Jetson / no internet)
# Put wheels in OFFLINE_WHEELS_DIR and list deps in OFFLINE_REQUIREMENTS.
AUTO_INSTALL_OFFLINE_DEPS = True
OFFLINE_REQUIREMENTS = "offline/requirements_jetson.txt"
OFFLINE_WHEELS_DIR = "offline/wheels"

# Data collection defaults
COLLECT_OUT_CSV = "data_out/collected.csv"
COLLECT_APPEND_IF_EXISTS = True
COLLECT_INCLUDE_METER_ID = True

# Label you are currently collecting (change this when you switch conditions)
# Example mapping (you decide): 0=normal, 1=fault_a, 2=fault_b, ...
COLLECT_CLASS_LABEL = 0

# Train defaults
COLLECTED_CSV = "data_out/collected.csv"
TRAIN_CSV = "data_out/train_data.csv"
TEST_CSV = "data_out/test_data.csv"
ARTIFACTS_DIR = "capstone_fault_ai_artifacts"
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3

# Inference defaults
INFER_OUT_CSV = "capstone_fault_ai_artifacts/predictions_live.csv"
