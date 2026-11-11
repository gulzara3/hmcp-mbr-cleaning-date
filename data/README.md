# Data

The pipeline uses the **Ghent University BIOMATH lab-scale sidestream crossflow MBR** dataset:

> W. Naessens, T. Maere, D. Van Hauwermeiren, I. Nopens (2017). *Lab-scale membrane bioreactor (MBR) high-frequency
> online data* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.821585

| File | Content | Size |
|---|---|---|
| `Metadata.csv` | sensor descriptions | 4 kB |
| `Settings_MBR.7z` | controller settings | 21 kB |
| `Data_MBR_Period_1.7z` … `Data_MBR_Period_4.7z` | 1 Hz records of the four operating periods (770 days in total) | 19 MB – 383 MB |

The dataset is not redistributed in this repository. You can get it in either of two ways:

- **Automatically.** The notebook downloads and verifies the files on first run.
- **Manually.** Run `python scripts/download_data.py` from the repository root. The files are saved in `results/data/`,
  where the notebook looks for them. If Zenodo blocks scripted downloads on your network, download the files in a
  browser and place them in that folder.

The signals used include:
- feed (PS2), concentrate (PS3) and permeate (PS4) pressure;
- permeate flux and mixed-liquor temperature;
- dissolved oxygen, pH and reactor level;
- membrane and biological phase indicators.
