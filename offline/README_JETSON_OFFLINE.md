# Jetson Offline Setup (no internet)

This project supports installing Python dependencies from a local “wheelhouse” folder so a Jetson can run it without internet.

## 1) Build an offline wheelhouse on an internet machine

On a machine that *does* have internet, download the wheels you need into `offline/wheels/`.

Example (run from the repo root):

```bash
python -m pip download -r offline/requirements_jetson.txt -d offline/wheels
```

Important:
- The Jetson needs **Linux aarch64** wheels. Download/build wheels that match your Jetson’s Python version and architecture.
- `torch` is JetPack-specific. Get the correct wheel for your JetPack version and put it into `offline/wheels/`. You can optionally add a `torch==...` line to `offline/requirements_jetson.txt`.

## 2) Copy the repo to the Jetson

Copy the entire repo folder, including:
- `offline/requirements_jetson.txt`
- `offline/wheels/` (your wheel files)

## 3) Install dependencies on the Jetson (offline)

Option A (automatic):
- Run `python app/run.py`
- Choose `0) Setup / configure`
- If `app/user_params.py:AUTO_INSTALL_OFFLINE_DEPS = True`, Setup runs:
  - `python -m pip install --no-index --find-links offline/wheels -r offline/requirements_jetson.txt`

Option B (manual):

```bash
python -m pip install --no-index --find-links offline/wheels -r offline/requirements_jetson.txt
```

If installation fails, it usually means a wheel is missing for Jetson’s Python/architecture.

