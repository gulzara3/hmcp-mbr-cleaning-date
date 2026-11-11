# Reproducibility

## Environment

| Item | Setting used for the paper |
|---|---|
| Hardware | NVIDIA L4 GPU (Google Colab), bfloat16 mixed precision |
| Python | 3.11 |
| Key packages | PyTorch 2.x, scikit-learn ≥ 1.3, XGBoost ≥ 2.0, SHAP, SciPy, pandas, py7zr |
| Seed | 42 (`SEED` in notebook §1); 3 seeds for the proposed model, 1 for learned baselines |

The notebook installs any missing package on first run, so `requirements.txt` is needed only for local use.
A CPU run is possible but slow, so a GPU is strongly recommended.

## Runtime and resuming

- **Fingerprinted outputs.** Every trained model, prediction set and statistic is cached in `results/cache` and
  `results/models`. Checkpoints carry fingerprints of both the architecture and the data, so a changed configuration
  never reloads a stale model.
- **Stage file.** Finished stages are recorded in `results/stages.json`. If a session ends, *Run all* resumes from the
  last finished stage.
- **Persistence on Colab.** Everything is stored on Google Drive under `MyDrive/PCNet_paper`. Set the environment
  variable `HMCP_DRIVE_DIR` to use another folder.
- **Run time.** With all checkpoints cached, a rerun of the predictions, statistics, engineering layer and figures takes
  about 15 minutes on an L4.
- **Fast check.** Set `FAST_MODE = True` in notebook §1 for a 2-fold smoke test.

## Data integrity

`scripts/download_data.py` and the notebook accept a dataset file only if both of these hold:
- its size is within 15 % of the published size;
- for `.7z` archives, its signature is valid and the archive opens.

This rules out truncated downloads and HTML error pages.

## Expected results

A full run should reproduce the paper's tables, up to GPU non-determinism in the third decimal:
- validity: 100 % for HMCP and < 0.1 % for the soft-penalty and unconstrained networks;
- HMCP date MAE: 4.27 d;
- day-level p-values against linear extrapolation, the gradient-boosted regressor and persistence.
