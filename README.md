# Capstone Fault AI (Your Version)

This folder is a clean, standalone implementation inspired by the prior project, without modifying their code.

## What it does
- Trains a small neural network on labeled CSV data (`class` + feature columns)
- Saves a scaler (`.pkl`) and model weights (`.pth`)
- Polls **multiple Modbus-TCP meters (multiple IPs)** and runs live inference on a Jetson

## Folder layout
- `app/` CLI entrypoints
- `core/` config, Modbus, preprocessing utilities
- `ml/` model + train/eval code
- `config_examples/` example JSON configs

## Jetson quickstart
1) Create a venv and install deps:
   - `python3 -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`
   - Install `torch` using the PyTorch-for-Jetson instructions for your JetPack version

2) One-click flow (recommended):
   - `python app/run.py`
   - Choose `0) Setup (dependencies)` once (installs from `offline/wheels/`)
   - Edit `meters.json` + `pipeline.json` directly (copy from `config_examples/` if needed)
   - Choose `1) Collect` for each condition (writes `data_out/collected.csv`); enable/disable meters in `meters.json`
   - Choose `2) Train` (auto-splits into `data_out/train_data.csv` + `data_out/test_data.csv` if needed)
   - Choose `3) Live inference` (writes `capstone_fault_ai_artifacts/predictions_live.csv`)
2) Copy and edit configs:
   - `capstone_fault_ai/config_examples/meters.example.json` → `meters.json`
   - `capstone_fault_ai/config_examples/pipeline.example.json` → `pipeline.json`
3) Collect labeled training data from meters (writes `class` + feature columns):
   - `python app/collect.py --meters_json meters.json --pipeline_json pipeline.json --class_label 0 --duration_s 30 --out_csv data_out/train_data.csv`
   - Optional: split per meter: `--split_by_meter --out_dir data_out`
4) Train:
   - `python app/train.py --train_csv data_out/train_data.csv --test_csv data_out/test_data.csv --out_dir capstone_fault_ai_artifacts`
5) Run live inference (writes an append-only CSV):
   - `python app/infer.py --meters_json meters.json --pipeline_json pipeline.json --artifacts_dir capstone_fault_ai_artifacts`

## One-click (PyCharm)
- Run `python app/run.py` and use the menu to collect, train, and infer without remembering arguments.

## Run individual files (PyCharm)
- Edit defaults in `app/user_params.py`.
- Run `app/collect.py` (it only needs `meters.json` + `pipeline.json`; change label via `app/user_params.py:COLLECT_CLASS_LABEL`).
- Run `app/train.py` (uses `app/user_params.py` paths/hyperparams).
- Run `app/infer.py` (uses `app/user_params.py` paths and writes `INFER_OUT_CSV`).

## Choose meters
- Enable/disable meters in `meters.json` using `"enabled": true/false`.
- Optional: `app/collect.py` / `app/infer.py` support `--meter_ids meter_1,meter_3` to poll only a subset.

## Jetson offline dependencies
- Put Jetson wheels in `offline/wheels/` and edit `offline/requirements_jetson.txt`.
- `Setup` in `app/run.py` auto-runs an offline install if `app/user_params.py:AUTO_INSTALL_OFFLINE_DEPS` is enabled.
- Details: `offline/README_JETSON_OFFLINE.md`.
