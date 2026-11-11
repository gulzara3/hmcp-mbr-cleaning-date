# Results

The pipeline writes all outputs here (or to Google Drive on Colab). This folder is ignored by git except for this file.

| Path | Content |
|---|---|
| `data/` | verified dataset archives and extracted CSV files |
| `models/` | fingerprinted model checkpoints per fold and seed |
| `cache/` | cached predictions and statistics, used to resume |
| `tables/` | CSV tables T1–T11: validity and accuracy, learned correction, stability, ablation, per-period results, envelope agreement, significance, operating fingerprint, kPa forecast, temperature |
| `figures/` | diagnostic figures |
| `figures_final/` | 600-dpi publication figures (PNG, PDF, SVG) from notebook §12 |
| `stages.json` | finished pipeline stages |
| `HMCP_results_<timestamp>.zip` | packaged tables, figures and manifest |
