# Capstone Fault AI

Multi-meter Modbus fault data collection, windowed dataset prep, RandomForest training, and live inference.

## What it does
- Collects synchronized features from multiple PAC4200 meters (`app/collect.py`).
- Builds event windows and splits train/test by file (`app/prepare_data.py`).
- Trains RandomForest classifiers for:
  - fault type (`--target class`)
  - fault location (`--target location`)
- Runs live inference with rolling windows (`app/infer.py`).

## Project layout
- `app/` runnable scripts (`collect`, `prepare_data`, `train`, `infer`, `run`)
- `core/` config/modbus/preprocess helpers
- `Config_Files/` meters and pipeline config
- `data_out/` raw captures + prepared train/test CSVs
- `capstone_fault_ai_artifacts/` trained model artifacts and prediction outputs

## Quick start
1. Create environment and install deps:
   - `python3 -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Configure:
   - `Config_Files/meters.json`
   - `Config_Files/pipeline.json`
   - `app/user_params.py`
3. Collect event files (normal and fault runs) into `data_out/`.
4. Prepare data:
   - `python -m capstone_fault_ai.app.prepare_data`
5. Train:
   - Fault type: `python -m capstone_fault_ai.app.train --target class`
   - Fault location: `python -m capstone_fault_ai.app.train --target location`
6. Infer live:
   - Fault type: `python -m capstone_fault_ai.app.infer --target class`
   - Fault location: `python -m capstone_fault_ai.app.infer --target location`

## Interactive launcher
Run `python -m capstone_fault_ai.app.run` and select from menu options.

## Notes
- `prepare_data` windows are sequence-aware and split by event file to avoid leakage.
- Fault windows are filtered to keep disturbance-heavy segments from nominal→fault→nominal recordings.
- For good detection, gather many fault events; severe class imbalance will otherwise bias toward `normal`.
