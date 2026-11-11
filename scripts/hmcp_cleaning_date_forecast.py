# %% [markdown]
# # Physics-constrained deep learning ensures a valid chemical-cleaning date for membrane bioreactors from routine pressure data
# ## With every result translated into the units a membrane engineer works in: kPa, kPa/day, LMH/bar and resistance shares
#
# **Target** *Water Research* · **Data** Ghent BIOMATH sidestream crossflow MBR, Zenodo 10.5281/zenodo.821585, 770 days, four operating regimes
# **Hardware** NVIDIA L4, BF16 · **Persistence** Google Drive, fingerprinted checkpoints, resumes from any interruption
#
# ---
# ### The contribution
# A hard-monotone cumulative parameterisation (HMCP) of irreversible fouling resistance gives a cleaning-date forecast that
#
# 1. **always exists**: a unique threshold crossing on 100 % of ≈ 43,000 windows, where soft-penalty and unconstrained
#    networks produce one on fewer than 0.1 %;
# 2. **is accurate**: significantly better than linear extrapolation, a gradient-boosted RUL regressor and persistence,
#    and statistically indistinguishable from power-law extrapolation, under a **day-level** block bootstrap with Holm
#    correction (Section 6.4);
# 3. **is safe to learn**: a learned correction is adopted only when validation supports it. Without this guard the
#    trained head is worse than linear extrapolation; with it the guard picked the better option on every informative fold.
#
# ### What a process engineer gets from this notebook (Section 8)
# * An **operating fingerprint** for each regime: resistance-in-series budget (membrane, reversible, irreversible shares),
#   hydraulic reversibility, backwash frequency, permeability in LMH/bar, irreversible fouling rate in **kPa/day**, and the
#   size of one degradation step in **kPa**.
# * The **14-day irreversible TMP rise** forecast in **kPa**, scored on all windows, censored or not.
# * A descriptive check of how irreversible fouling responds to **temperature** over 770 days.
#
# ### How the accuracy arises, stated transparently
# The HMCP head is initialised at a monotone blend of linear and power-law extrapolation, weighted toward whichever is
# better on the **training** windows, trained with an L2 anchor to that initialisation, and a validation guard deploys
# either the trained head or the untrained one. Section 6.1 reports all three per fold. Much of the accuracy comes from the
# parameterisation itself: a physically consistent monotone trajectory places the discrete first crossing better than either
# classical extrapolation alone. The contribution is that parameterisation, which makes the forecast valid by construction
# and lets a learned correction in safely.
#
# ### Sections
# | Section | Evidence |
# |---|---|
# | 3 | Self-supervised irreversible-resistance monitor from crossflow TMP, validated across three transducers and two envelope methods |
# | 6 | Validity and accuracy against four references, five forward-chaining folds |
# | 6.1 | Learned correction: untrained seed vs trained head vs deployed |
# | 6.4 | **Day-level inference**: block bootstrap, Holm correction, guard test |
# | 8 | **Process-engineering layer**: fingerprint, kPa forecast accuracy, temperature dependence |
# | 9 | Operational availability and decision stability on daily origins |
#
# ### Prognostic target
# $$\mathrm{RUL}_\delta(d_0)=\min\{k\ge1:\;R_{\rm irrev}(d_0+k)-R_{\rm irrev}(d_0)\ge\delta\},\qquad \delta = 10\,\%\text{ of period range},\quad \text{censored at 14 d}$$
#
# ### Resume
# Architecture and data fingerprints are unchanged from the previous run, so every trained model reloads from Drive.
# Predictions, statistics, the engineering layer and the figures are recomputed: roughly 15 minutes on an L4.

# %% [markdown]
# ## 1. Environment and Seeding

# %%
"""Section 1: environment, hardware, reproducibility, plotting."""
import os, subprocess, sys


def _pip(pkg): subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


for _pkg, _mod in [("py7zr", "py7zr"), ("python-docx", "docx"), ("shap", "shap"), ("scikit-learn>=1.3", "sklearn"), ("xgboost>=2.0", "xgboost")]:
    try: __import__(_mod)
    except ImportError: _pip(_pkg)

import glob, json, math, random, shutil, time, urllib.request, warnings, zipfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb, shap
warnings.filterwarnings("ignore")

SEED = 42
FAST_MODE = False
PERIODS = [1, 2, 3, 4]
N_FOLDS = 2 if FAST_MODE else 5
# ── Google Drive as the single source of truth ───────────────────────────────
DRIVE_ROOT = None
try:
    from google.colab import drive as _drive
    _drive.mount("/content/drive", force_remount=False)
    DRIVE_ROOT = Path(os.environ.get("HMCP_DRIVE_DIR", "/content/drive/MyDrive/PCNet_paper")); DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Drive mounted. Project root: {DRIVE_ROOT}")
except Exception as _exc:  # noqa: BLE001
    print(f"Drive not available ({type(_exc).__name__}); falling back to local ./results (NOT persistent).")
_LOCAL_ROOT = Path.cwd().parent if Path.cwd().name in ("notebooks", "scripts") else Path.cwd()   # repo root
OUT = (DRIVE_ROOT / "results") if DRIVE_ROOT else Path(os.environ.get("HMCP_OUT", _LOCAL_ROOT / "results"))
CACHE = OUT / "cache"
for _s in ("figures", "models", "data", "tables", "cache"): (OUT / _s).mkdir(parents=True, exist_ok=True)


def cache_get(key):
    p = CACHE / f"{key}.json"
    if p.exists():
        try: return json.loads(p.read_text())
        except Exception: return None  # noqa: BLE001
    return None


def cache_put(key, obj):
    (CACHE / f"{key}.json").write_text(json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def npz_get(key):
    p = CACHE / f"{key}.npz"
    return dict(np.load(p, allow_pickle=False)) if p.exists() else None


def npz_put(key, **arrs):
    np.savez_compressed(CACHE / f"{key}.npz", **{k: np.asarray(v) for k, v in arrs.items()})


STAGE_FILE = OUT / "stages.json"
STAGES = json.loads(STAGE_FILE.read_text()) if STAGE_FILE.exists() else {}


def stage_done(name): return bool(STAGES.get(name))


def stage_mark(name):
    STAGES[name] = datetime.now().isoformat(timespec="seconds"); STAGE_FILE.write_text(json.dumps(STAGES, indent=1))


print(f"Completed stages so far: {list(STAGES) or 'none'}")


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed); os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


seed_everything()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BF16_OK = DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if BF16_OK else torch.float32
if DEVICE.type == "cuda":
    _p = torch.cuda.get_device_properties(0); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    print(f"GPU {_p.name} | {_p.total_memory/1e9:.1f} GB | BF16 {BF16_OK}")
N_WORKERS = min(8, os.cpu_count() or 2)
# ── Publication figure system ────────────────────────────────────────────────
# Okabe-Ito palette: colourblind-safe under deuteranopia, protanopia and tritanopia,
# and legible in greyscale. Semantic names keep colour meaning consistent across figures.
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#111111",
     "grey": "#7F7F7F", "light": "#D9D9D9"}
SEM = {"ours": C["blue"], "physics": C["red"], "baseline": C["grey"], "reference": C["orange"],
       "good": C["green"], "bad": C["red"], "neutral": C["sky"], "accent": C["purple"]}
PAL = [C["blue"], C["orange"], C["green"], C["red"], C["purple"], C["sky"], C["yellow"], C["grey"]]

# Column widths in inches (Elsevier / Water Research specification)
W1, W15, W2 = 3.54, 5.51, 7.28

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "lines.linewidth": 1.2, "lines.markersize": 3.6,
    "legend.frameon": False, "legend.handlelength": 1.6, "legend.columnspacing": 1.0,
    "axes.axisbelow": True,
})
sns.set_palette(PAL)


def panel(ax, letter, title=""):
    """Bold panel letter outside the axes, optional plain title beside it."""
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="bottom", ha="right")
    if title:
        ax.set_title(title, loc="left", pad=4)
    return ax


def finish(name, fig=None, tight=True):
    """Save PNG, PDF and SVG at 300 dpi."""
    fig = fig or plt.gcf()
    if tight:
        fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / "figures" / f"{name}.{ext}")
    plt.show()
    print(f"  saved figures/{name}.{{png,pdf,svg}}")


savefig = finish


def barh_labeled(ax, names, vals, colors=None, fmt="{:.3f}", xerr=None):
    """Horizontal bars with the value printed at the end of each bar."""
    vals = np.asarray(vals, float)
    colors = colors or [SEM["baseline"]] * len(names)
    b = ax.barh(list(names), vals, color=colors, edgecolor=C["black"], linewidth=0.5,
                height=0.72, xerr=xerr,
                error_kw=dict(elinewidth=0.7, capsize=2, ecolor=C["black"]))
    finite = vals[np.isfinite(vals)]
    span = (np.nanmax(finite) - min(0.0, np.nanmin(finite))) if len(finite) else 1.0
    if xerr is None:
        hi = np.zeros(len(vals))
    else:
        _xe = np.asarray(xerr, float)
        hi = np.nan_to_num(_xe[1] if _xe.ndim == 2 else _xe)          # place labels beyond the upper error bar
    for r, v, h in zip(b, vals, hi):
        if not np.isfinite(v):
            continue
        ax.text(v + h + (span or 1.0) * 0.02, r.get_y() + r.get_height() / 2, fmt.format(v),
                va="center", ha="left", fontsize=7)
    ax.margins(x=0.18)
    return b


def annotate_value(ax, x, y, text, dx=6, dy=6, color=None, weight="normal"):
    """Offset text annotation in points, for labelling a specific coordinate."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=7, color=color or C["black"], fontweight=weight)


def shade_periods(ax, frame, xcol="time_days", pcol="period"):
    """Alternating shading and dotted separators marking operating periods."""
    for i, p in enumerate(sorted(frame[pcol].unique())):
        m = frame[pcol] == p
        if not m.any():
            continue
        x0, x1 = frame.loc[m, xcol].iloc[0], frame.loc[m, xcol].iloc[-1]
        if i % 2:
            ax.axvspan(x0, x1, color=C["light"], alpha=0.30, lw=0, zorder=0)
        if i:
            ax.axvline(x0, color=C["grey"], ls=":", lw=0.7, zorder=1)

print("Section 1 complete.")

# %% [markdown]
# ## 2. Resilient Data Ingestion
# Three Zenodo endpoints with retry and resume, a persistent Drive cache, and a manual-upload fallback.

# %%
"""Section 2: verified fetch. Every file is checked against its known size and format before use."""
RECORD = "821585"
# Published sizes on the Zenodo record, used to reject truncated downloads.
KNOWN_BYTES = {"Metadata.csv": 4_300, "Settings_MBR.7z": 21_000,
               "Data_MBR_Period_1.7z": 18_700_000, "Data_MBR_Period_2.7z": 39_900_000,
               "Data_MBR_Period_3.7z": 382_800_000, "Data_MBR_Period_4.7z": 238_800_000}
SEVEN_ZIP_MAGIC = b"7z\xbc\xaf\x27\x1c"
URL_VARIANTS = ["https://zenodo.org/records/{rec}/files/{name}?download=1",
                "https://zenodo.org/api/records/{rec}/files/{name}/content",
                "https://zenodo.org/record/{rec}/files/{name}?download=1"]
MAX_ATTEMPTS, BACKOFF_S = 3, [20, 45, 90]
DRIVE_CACHE = (DRIVE_ROOT / "ghent_mbr_cache") if DRIVE_ROOT else None
if DRIVE_CACHE: DRIVE_CACHE.mkdir(parents=True, exist_ok=True); print(f"Data cache: {DRIVE_CACHE}")
LOCAL = OUT / "data"
_HDR = {"User-Agent": "Mozilla/5.0 (X11; Linux) Gecko/20100101 Firefox/120.0", "Accept": "*/*"}


def file_ok(p: Path, name: str) -> bool:
    """A file is accepted only if its size is within 15 % of the published size AND its format is right."""
    if not p.exists(): return False
    n = p.stat().st_size; exp = KNOWN_BYTES.get(name)
    if exp and not (0.85 * exp <= n <= 1.15 * exp): return False
    with open(p, "rb") as fh: head = fh.read(16)
    if name.endswith(".7z"):
        if not head.startswith(SEVEN_ZIP_MAGIC): return False
        try:
            import py7zr
            with py7zr.SevenZipFile(p, "r") as z: z.getnames()
        except Exception:  # noqa: BLE001
            return False
        return True
    return n > 100 and b"<html" not in head.lower()


def _download(url: str, dst: Path) -> None:
    """Fresh download to dst. No resume: partial files from another mirror must never be spliced."""
    if dst.exists(): dst.unlink()
    req = urllib.request.Request(url, headers=_HDR)
    with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as fh:
        clen = r.headers.get("Content-Length")
        while True:
            b = r.read(1 << 20)
            if not b: break
            fh.write(b)
    if clen and dst.stat().st_size != int(clen):
        raise IOError(f"truncated: got {dst.stat().st_size} of {clen} bytes")


def fetch(name: str) -> Optional[Path]:
    dst = LOCAL / name
    # 1. valid local copy
    if file_ok(dst, name): print(f"  cached (local, verified) {name}"); return dst
    if dst.exists(): dst.unlink(); print(f"  purged corrupt local {name}")
    # 2. valid Drive copy
    if DRIVE_CACHE:
        dc = DRIVE_CACHE / name
        if file_ok(dc, name): shutil.copy(dc, dst); print(f"  cached (Drive, verified) {name}"); return dst
        if dc.exists(): dc.unlink(); print(f"  purged corrupt Drive {name}")
    # 3. mirrors, each attempt a fresh download, each result verified
    part = LOCAL / (name + ".part")
    for url_t in URL_VARIANTS:
        url = url_t.format(rec=RECORD, name=name)
        for attempt in range(MAX_ATTEMPTS):
            try:
                print(f"  fetch {name} [{'api' if '/api/' in url else 'web'}] attempt {attempt + 1}")
                _download(url, part)
                if not file_ok(part, name):
                    raise IOError(f"verification failed ({part.stat().st_size/1e6:.1f} MB, expected ~{KNOWN_BYTES.get(name, 0)/1e6:.1f} MB)")
                part.rename(dst); print(f"        ok {dst.stat().st_size/1e6:.1f} MB, verified")
                if DRIVE_CACHE: shutil.copy(dst, DRIVE_CACHE / name)
                return dst
            except Exception as exc:  # noqa: BLE001
                w = BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)]
                print(f"        {type(exc).__name__}: {str(exc)[:70]} -> retry in {w}s"); time.sleep(w)
        if part.exists(): part.unlink()
    # 4. manual upload
    print(f"\n  Could not obtain a valid {name} from Zenodo.")
    print(f"  Download it in your browser from https://zenodo.org/records/{RECORD} and either")
    print(f"    (a) place it in Drive at {DRIVE_CACHE / name if DRIVE_CACHE else LOCAL / name}  and rerun this cell, or")
    print(f"    (b) upload it now when prompted.")
    try:
        from google.colab import files as _files
        for fname, data in _files.upload().items():
            if fname == name:
                dst.write_bytes(data)
                if file_ok(dst, name):
                    if DRIVE_CACHE: shutil.copy(dst, DRIVE_CACHE / name)
                    print(f"        uploaded and verified {dst.stat().st_size/1e6:.1f} MB"); return dst
                print("        uploaded file failed verification"); dst.unlink()
    except ImportError: pass
    return None


def load_period(period: int) -> Optional[pd.DataFrame]:
    import py7zr
    arc = fetch(f"Data_MBR_Period_{period}.7z")
    if arc is None: return None
    ext = LOCAL / f"ghent_p{period}"
    if not ext.exists() or not any(ext.rglob("*.[cC][sS][vV]")):
        with py7zr.SevenZipFile(arc, "r") as z: z.extractall(ext)
    frames = []
    for cp in glob.glob(str(ext / "**" / "*.[cC][sS][vV]"), recursive=True):
        for sep in (";", ","):
            try:
                if pd.read_csv(cp, sep=sep, nrows=5, on_bad_lines="skip").shape[1] >= 5:
                    d = pd.read_csv(cp, sep=sep, on_bad_lines="skip", low_memory=False); d["period"] = period; frames.append(d); break
            except Exception: continue  # noqa: BLE001
    if not frames: return None
    raw = pd.concat(frames, ignore_index=True); print(f"  Period {period}: {len(raw):,} rows"); return raw


meta_path = fetch("Metadata.csv")
if meta_path is None:
    print("  Metadata.csv unavailable; it is documentary only and the physics constants are hard-coded from it.")
_per = [p for p in (load_period(p) for p in PERIODS) if p is not None]
if not _per: raise RuntimeError(f"No valid Ghent archive. Place the .7z files in {DRIVE_CACHE or LOCAL} and rerun.")
RAW = pd.concat(_per, ignore_index=True)
print(f"\nRaw record: {len(RAW):,} rows at 1 Hz, periods {sorted(RAW['period'].unique())}")

# %% [markdown]
# ## 3. Physics-Correct Preprocessing
# ### 3.1 Column mapping, crossflow TMP at 1 Hz, filtration-only aggregation
# TMP and resistance are computed **per second during filtration only** (membrane phase ≥ 0.5),
# then block-averaged. Backwash samples, where TMP is negative by design, no longer contaminate
# the resistance signal. Backwash events are counted from falling edges within each period.

# %%
"""Section 3.1: crossflow TMP, filtration-only resistance, 5-min aggregation."""
DT_MIN, FACTOR = 5, 300
A_MEMBRANE = 0.17  # m^2, from Metadata.csv


def find_col(df, *keys):
    for c in df.columns:
        if all(k in c.lower() for k in keys): return c
    return None


COLS = {"phase_mem": find_col(RAW, "phase", "membrane"), "phase_bio": find_col(RAW, "aerobic"), "temp": find_col(RAW, "temp"),
        "ph_an": find_col(RAW, "phan"), "ph_a": find_col(RAW, "pha "), "do": find_col(RAW, "do "), "level": find_col(RAW, "level"),
        "ps2": find_col(RAW, "ps2"), "ps3": find_col(RAW, "ps3"), "ps4": find_col(RAW, "ps4"), "flux": find_col(RAW, "flux")}
if COLS["ph_a"] is None: COLS["ph_a"] = find_col(RAW, "pha")
print(json.dumps(COLS, indent=2))
assert all(COLS[k] for k in ("ps2", "ps3", "ps4", "flux", "phase_mem")), "Essential columns missing."


def vogel_viscosity(t_c): return 2.414e-5 * 10.0 ** (247.8 / (t_c + 133.15))


_num = lambda c: pd.to_numeric(RAW[c], errors="coerce").ffill().bfill().values.astype(np.float64)
ps2, ps3, ps4 = _num(COLS["ps2"]), _num(COLS["ps3"]), _num(COLS["ps4"])
flux_1hz = np.abs(_num(COLS["flux"]))
temp_1hz = _num(COLS["temp"]) if COLS["temp"] else np.full(len(RAW), 20.0)
phase_1hz = _num(COLS["phase_mem"])
period_1hz = RAW["period"].values

tmp_kpa = 0.5 * (ps2 + ps3) - ps4          # crossflow TMP
dp_axial_kpa = ps2 - ps3                    # feed-channel pressure drop
filt = phase_1hz >= 0.5
mu = vogel_viscosity(temp_1hz)
j_ms = np.clip(flux_1hz, 1e-3, None) / 3.6e6
r_tot_1hz = np.where(filt, np.clip(tmp_kpa, 0, None) * 1e3 * A_MEMBRANE / (mu * j_ms), np.nan)  # kPa->Pa

falls = np.zeros(len(RAW), np.int32)
for _p in np.unique(period_1hz):
    idx = np.where(period_1hz == _p)[0]; ph = phase_1hz[idx]
    falls[idx[1:]] = ((ph[:-1] >= 0.5) & (ph[1:] < 0.5)).astype(np.int32)

n_blocks = len(RAW) // FACTOR
blk = np.repeat(np.arange(n_blocks), FACTOR)


def blockmean(x, mask=None):
    x = x[: n_blocks * FACTOR].astype(np.float64)
    if mask is not None:
        x = np.where(mask[: n_blocks * FACTOR], x, np.nan)
    s = pd.Series(x).groupby(blk)
    return s.mean().values


df = pd.DataFrame({
    "period": pd.Series(period_1hz[: n_blocks * FACTOR]).groupby(blk).agg(lambda v: v.iloc[0]).values,
    "TMP_kPa": blockmean(tmp_kpa, filt), "dP_axial_kPa": blockmean(dp_axial_kpa, filt),
    "PS2_kPa": blockmean(ps2, filt), "PS3_kPa": blockmean(ps3, filt), "PS4_kPa": blockmean(ps4, filt),
    "flux_LMH": blockmean(flux_1hz, filt), "temperature_C": blockmean(temp_1hz),
    "R_total": blockmean(r_tot_1hz), "filt_frac": blockmean(filt.astype(float)),
    "membrane_phase": blockmean(phase_1hz),
    "is_backwash": (pd.Series(falls[: n_blocks * FACTOR]).groupby(blk).sum().values >= 1).astype(int),
})
for k, c in (("DO_mgL", "do"), ("level_cm", "level"), ("pH_an", "ph_an"), ("pH_a", "ph_a"), ("bio_phase", "phase_bio")):
    df[k] = blockmean(_num(COLS[c])) if COLS[c] else np.nan
df["period"] = df["period"].astype(int)
df = df.interpolate(limit=12).ffill().bfill()
df["time_days"] = np.arange(len(df)) * DT_MIN / 60 / 24
naive_bw = int((df["membrane_phase"] < 0.5).sum())
print(f"\n5-min frame: {len(df):,} rows, {df['time_days'].iloc[-1]:.0f} days")
print(f"Backwashes transition-count {int(df['is_backwash'].sum()):,} | naive {naive_bw:,}")
print(f"TMP (crossflow) median {df['TMP_kPa'].median():.2f} kPa, p05 {df['TMP_kPa'].quantile(.05):.2f}, p95 {df['TMP_kPa'].quantile(.95):.2f}, negative frac {(df['TMP_kPa']<0).mean():.3f}")
print(f"R_total median {df['R_total'].median():.3e} m^-1 | corr(R_total, flux) = {df[['R_total','flux_LMH']].corr().iloc[0,1]:+.3f}  (should be small)")

# %% [markdown]
# ### 3.2 Robust slow envelope and resistance decomposition (Eqs. 4 to 7, revised)
# The irreversible envelope is built from **one value per day**, the 10th percentile of filtration
# $R_{\text{total}}$, which removes the diurnal flux cycle and is immune to single spikes. Isotonic
# regression (PAV) is then fitted **per period**. A second envelope from the rolling-24 h median of
# post-backwash minima is computed for validation, and both are compared against a naive feed-side-only
# decomposition to show why the crossflow TMP matters.

# %%
"""Section 3.2: daily-quantile envelope, isotonic per period, validation envelopes."""
STEPS_PER_DAY = 24 * 60 // DT_MIN


def envelope_daily_q(r_tot, q=10):
    """Daily q-th percentile of R_total, interpolated back to 5-min grid."""
    day = np.arange(len(r_tot)) // STEPS_PER_DAY
    s = pd.Series(r_tot).groupby(day).quantile(q / 100.0)
    centers = s.index.values * STEPS_PER_DAY + STEPS_PER_DAY // 2
    return np.interp(np.arange(len(r_tot)), centers, s.values)


def envelope_postbw_median(r_tot, bw, win=STEPS_PER_DAY):
    """Rolling-24h median of post-backwash 1-h minima (v2 method, made robust)."""
    idx = np.where(bw == 1)[0]
    if len(idx) < 3: return np.full(len(r_tot), np.nan)
    b = np.array([np.nanmin(r_tot[k:k + 12]) for k in idx])
    s = pd.Series(b, index=idx).rolling(max(3, win // 8), min_periods=1).median()
    return np.interp(np.arange(len(r_tot)), s.index.values, s.values)


def decompose(r_tot, bw, periods, method="daily_q"):
    R_irr, R_rev, R_m = np.zeros_like(r_tot), np.zeros_like(r_tot), {}
    for p in np.unique(periods):
        m = periods == p; rt = r_tot[m]
        env = envelope_daily_q(rt) if method == "daily_q" else envelope_postbw_median(rt, bw[m])
        env = pd.Series(env).ffill().bfill().values
        r_m = float(np.nanpercentile(env, 5))
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        ri = iso.fit_transform(np.arange(len(env), dtype=float), np.clip(env - r_m, 0, None))
        R_irr[m] = ri; R_m[int(p)] = r_m
        R_rev[m] = np.clip(rt - r_m - ri, 0, None)
    return R_rev, R_irr, R_m


per = df["period"].values; bw = df["is_backwash"].values
R_TOT = df["R_total"].values
df["R_rev"], df["R_irrev"], R_M = decompose(R_TOT, bw, per, "daily_q")
_, R_irr_pbw, _ = decompose(R_TOT, bw, per, "postbw")
# feed-side-only (v2) decomposition for comparison
r_feed = np.clip(df["PS2_kPa"].values, 0, None) * 1e3 * A_MEMBRANE / (vogel_viscosity(df["temperature_C"].values) * np.clip(df["flux_LMH"].values, 1e-3, None) / 3.6e6)
_, R_irr_feed, _ = decompose(r_feed, bw, per, "daily_q")
ENV_VALID = pd.DataFrame({"daily_q10 (primary)": df["R_irrev"], "post-BW median": R_irr_pbw, "feed-side-only (v2)": R_irr_feed}).corr()
steps = int((np.diff(df["R_irrev"]) > 0).sum())
print(f"Irreversible envelope: {steps:,} increasing steps over {df['time_days'].iloc[-1]:.0f} days | R_m by period {{{', '.join(f'{k}: {v:.2e}' for k, v in R_M.items())}}}")
print(f"R_irrev end-of-record by period: {[f'{df.loc[per==p,'R_irrev'].iloc[-1]:.2e}' for p in np.unique(per)]}")
print("\nEnvelope cross-validation (correlation):\n", ENV_VALID.round(3).to_string())


# %% [markdown]
# ### 3.3 Features, and the daily slow series
# pH and DO are median-smoothed because the metadata records that they were rarely calibrated.
# A multi-scale trend block makes explicit the information a flattened-window tree exploits.
# The irreversible envelope is additionally aggregated to **one value per day**, which is the
# resolution at which the prognostic branch operates.

# %%
"""Section 3.3: features and daily aggregation."""
for c in ("pH_an", "pH_a", "DO_mgL"):
    df[c] = df[c].rolling(12, min_periods=1, center=True).median()
    df[f"d_{c}"] = df[c].diff().fillna(0)
df["permeability"] = df["flux_LMH"] / df["TMP_kPa"].clip(lower=0.1)
df["dR_dt"] = df["R_total"].diff().fillna(0) / (DT_MIN / 60)
df["R_roll_mean"] = df["R_total"].rolling(12, min_periods=1).mean()
df["R_roll_std"] = df["R_total"].rolling(12, min_periods=1).std().fillna(0)
df["dP_axial_roll"] = df["dP_axial_kPa"].rolling(12, min_periods=1).mean()
for w, tag in ((12, "1h"), (72, "6h"), (288, "24h"), (2016, "7d")):
    df[f"slope_{tag}"] = df["R_total"].rolling(w, min_periods=max(3, w // 6)).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True).fillna(0)
    df[f"dR_{tag}"] = df["R_total"].diff(w).fillna(0)
df["Rirr_slope_7d"] = df["R_irrev"].rolling(2016, min_periods=288).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True).fillna(0)
df["R_ewm_fast"] = df["R_total"].ewm(span=288, adjust=False).mean()
df["R_ewm_slow"] = df["R_total"].ewm(span=2016, adjust=False).mean()
df["R_ewm_ratio"] = df["R_ewm_fast"] / df["R_ewm_slow"].replace(0, np.nan)
FEATS = ["TMP_kPa", "dP_axial_kPa", "dP_axial_roll", "flux_LMH", "temperature_C", "permeability", "dR_dt",
         "R_roll_mean", "R_roll_std", "level_cm", "DO_mgL", "d_DO_mgL", "pH_an", "d_pH_an", "pH_a", "d_pH_a",
         "membrane_phase", "bio_phase", "slope_1h", "slope_6h", "slope_24h", "slope_7d",
         "dR_1h", "dR_6h", "dR_24h", "dR_7d", "Rirr_slope_7d", "R_ewm_fast", "R_ewm_slow", "R_ewm_ratio"]
FEATS = [f for f in FEATS if f in df and df[f].notna().mean() > 0.5]
for c in FEATS: df[c] = df[c].fillna(df[c].median())
FLUX_IDX = FEATS.index("flux_LMH")          # used by the Task-S flux gradient
print(f"Features ({len(FEATS)}); flux channel at index {FLUX_IDX}")

STEPS_PER_DAY = 24 * 60 // DT_MIN
df["day"] = (np.arange(len(df)) // STEPS_PER_DAY).astype(int)
DAILY = df.groupby("day").agg(R_irrev=("R_irrev", "median"), R_total=("R_total", "median"),
                              flux=("flux_LMH", "mean"), temp=("temperature_C", "mean"),
                              TMP=("TMP_kPa", "median"), period=("period", "first")).reset_index()
print(f"Daily series: {len(DAILY)} days across periods {sorted(DAILY['period'].unique())}")

# %% [markdown]
# ### 3.4 Prognostic target: time to the next degradation step
#
# Let $\delta_p$ be 10 % of the range of $R_{\rm irrev}$ within operating period $p$. For an origin day $d_0$,
# $$\mathrm{RUL}_\delta(d_0)=\min\{k\ge 1: R_{\rm irrev}(d_0+k)-R_{\rm irrev}(d_0)\ge\delta_p\},$$
# right-censored at the horizon. Every window whose resistance is still growing is a prognostic instance; there is
# no overdue state. For the economics of Section 9 the **absolute cleaning date** is obtained by running the same
# predicted trajectory to a TMP-equivalent threshold $R_{\rm crit}$. Both $\delta$ and the horizon are swept in
# Section 10 as a sensitivity analysis.

# %%
"""Section 3.4: delta-step RUL labels (dense, no overdue state) and the absolute cleaning threshold."""
H_DAYS_PRIMARY, H_DAYS_SWEEP = 14, [14]
DELTA_FRAC_PRIMARY, DELTA_SWEEP = 0.10, [0.10]

# degradation step per period
DELTA = {}
for p, g in DAILY.groupby("period"):
    rng_ = float(np.nanpercentile(g["R_irrev"], 99) - np.nanpercentile(g["R_irrev"], 1))
    DELTA[int(p)] = max(rng_, 1e-12)
print("R_irrev range per period:", {k: f"{v:.3e}" for k, v in DELTA.items()})

# absolute cleaning threshold (TMP-equivalent), used only for the chained cleaning date in Section 9
RCRIT = {}
for p, g in DAILY.groupby("period"):
    m5 = df["period"] == p
    tmp_lim = float(np.nanpercentile(df.loc[m5, "TMP_kPa"], 95))
    j_ref = float(np.nanmedian(df.loc[m5, "flux_LMH"])); t_ref = float(np.nanmedian(df.loc[m5, "temperature_C"]))
    RCRIT[int(p)] = max(tmp_lim * 1e3 * A_MEMBRANE / (vogel_viscosity(t_ref) * max(j_ref, 1e-3) / 3.6e6) - R_M[int(p)], 1e-12)


def rul_step_labels(delta_frac: float, h_days: int):
    """Per day: (RUL in days capped at h_days, censored flag). Steps never span a period boundary."""
    ri = DAILY["R_irrev"].values; per_d = DAILY["period"].values; n_d = len(ri)
    rul = np.full(n_d, float(h_days)); cens = np.ones(n_d, bool)
    for i in range(n_d):
        thr = ri[i] + delta_frac * DELTA[int(per_d[i])]
        for k in range(1, h_days + 1):
            j = i + k
            if j >= n_d or per_d[j] != per_d[i]: break
            if ri[j] >= thr: rul[i] = float(k); cens[i] = False; break
    return rul, cens


RUL_TAB = {}
for dfrac in DELTA_SWEEP:
    for h in H_DAYS_SWEEP:
        RUL_TAB[(dfrac, h)] = rul_step_labels(dfrac, h)
r0, c0 = RUL_TAB[(DELTA_FRAC_PRIMARY, H_DAYS_PRIMARY)]
print(f"\nPrimary target: delta = {100*DELTA_FRAC_PRIMARY:.0f} % of period range, horizon {H_DAYS_PRIMARY} d")
print(f"  uncensored days {100*(~c0).mean():.1f} %   median RUL (uncensored) {np.median(r0[~c0]):.1f} d   "
      f"IQR [{np.percentile(r0[~c0], 25):.0f}, {np.percentile(r0[~c0], 75):.0f}] d")

PRIMARY_MODE = f"delta{int(100*DELTA_FRAC_PRIMARY)}"

# %% [markdown]
# ### 3.5 Sequence construction
# Each window carries (10 tensors): the 24 h input; the 4 h reversible level (Task A);
# the daily irreversible increments over the horizon (Task P); the censored RUL and its flag;
# the Task-A binary label and its dense auxiliary target; and the anchor $R_{\text{irrev}}(t_0)$
# with the applicable $R_{\text{crit}}$; and the trailing 7-day slope $s_{\rm lin}$ that seeds the hybrid head.

# %%
"""Section 3.5: windows and folds."""
SEQ_LEN, STRIDE, H_A = 288, 3, 48
X_all = df[FEATS].values.astype(np.float32)
R_REV, R_IRR, R_TOT = df["R_rev"].values, df["R_irrev"].values, df["R_total"].values
DAY_OF_STEP = df["day"].values
RI_DAILY, PER_DAILY = DAILY["R_irrev"].values, DAILY["period"].values


def fwd_inc(x, h):
    o = np.full(len(x), np.nan); o[: len(x) - h] = x[h:] - x[:-h]; return o


@dataclass
class Fold:
    k: int; tr: tuple; va: tuple; te: tuple; fs: MinMaxScaler; S: float
    tau_a: float; pos: dict; h_days: int; delta_frac: float
    tr_s: int = 0; va_s: int = 0; te_s: int = 0; te_e: int = 0     # absolute 5-min indices of each split


def hist_slope(d0, days=7):
    """Observed daily slope of R_irrev over the trailing window, in physical units; >= 0."""
    lo = max(0, d0 - days); h = RI_DAILY[lo:d0 + 1]
    if len(h) < 3: return 0.0
    return max(0.0, float(np.polyfit(np.arange(len(h)), h, 1)[0]))


def powerlaw_increments(d0, h_days, days=7):
    """Increments implied by a power-law fit R - R0 = c t^n over the trailing window, physical units, >= 0.

    Falls back to the linear slope when the fit is degenerate (flat or too short history)."""
    lo = max(0, d0 - days); h = RI_DAILY[lo:d0 + 1]; T = len(h) - 1
    s_lin = hist_slope(d0, days)
    if T < 2: return np.full(h_days, s_lin)
    y = h - h[0]; t = np.arange(len(h), dtype=float)
    m = (y > 0) & (t > 0)
    if m.sum() < 2: return np.full(h_days, s_lin)
    try:
        n_, logc = np.polyfit(np.log(t[m]), np.log(y[m]), 1)
        n_ = float(np.clip(n_, 0.05, 3.0)); c_ = float(np.exp(logc))
        fut = c_ * ((T + np.arange(1, h_days + 1)) ** n_ - T ** n_)
        inc = np.diff(np.concatenate([[0.0], fut]))
        return np.clip(inc, 0.0, None)
    except Exception:  # noqa: BLE001
        return np.full(h_days, s_lin)


def _windows(lo, hi, X_s, S, ya, h_days, rul, cens, delta_frac, tally=None):
    """Returns 12 tensors per window. rc is the *step target level* anchor + delta, so the generic
    first-crossing machinery yields RUL_delta directly; rca is the absolute cleaning threshold for Section 9."""
    xs, yrev, ydaily, la, ac, an, rl, cs, rcv, sl, pw, rca = ([] for _ in range(12))
    need = h_days * STEPS_PER_DAY
    for i in range(lo, hi - SEQ_LEN - max(H_A, need), STRIDE):
        t0 = i + SEQ_LEN - 1; d0 = DAY_OF_STEP[t0]
        if d0 + h_days >= len(RI_DAILY) or PER_DAILY[d0 + h_days] != PER_DAILY[d0]: continue
        base = RI_DAILY[d0]; p_ = int(PER_DAILY[d0])
        inc = np.maximum.accumulate(np.clip(RI_DAILY[d0 + 1: d0 + 1 + h_days] - base, 0, None))
        if len(inc) != h_days: continue
        xs.append(X_s[i:i + SEQ_LEN]); yrev.append(R_REV[t0 + 1:t0 + 1 + H_A] / S)
        ydaily.append(inc / S); an.append(base / S)
        rcv.append((base + delta_frac * DELTA[p_]) / S); rca.append(RCRIT[p_] / S)
        la.append(ya[t0]); ac.append((R_TOT[t0 + H_A] - R_TOT[t0]) / S)
        rl.append(rul[d0]); cs.append(float(cens[d0])); sl.append(hist_slope(d0) / S)
        pw.append(powerlaw_increments(d0, h_days) / S)
    if not xs:
        e = np.zeros((0, SEQ_LEN, X_s.shape[1]), np.float32); v = np.zeros(0, np.float32); z = np.zeros((0, h_days), np.float32)
        return (e, np.zeros((0, H_A), np.float32), z, v, v, v, v, v, v, v, z, v)
    return tuple(np.asarray(a, np.float32) for a in (xs, yrev, ydaily, la, ac, an, rl, cs, rcv, sl, pw, rca))


MIN_VAL_UNCENS = 0.05


def build_fold(k, tr_s, te_s, te_e, h_days=H_DAYS_PRIMARY, delta_frac=DELTA_FRAC_PRIMARY):
    rul, cens = RUL_TAB[(delta_frac, h_days)]
    best_frac, best_unc = 0.75, -1.0
    for frac in (0.75, 0.65, 0.55):
        va_try = tr_s + int(frac * (te_s - tr_s))
        _fs = MinMaxScaler().fit(X_all[tr_s:va_try]); _va = _windows(va_try, te_s, _fs.transform(X_all), 1.0, np.zeros(n, np.float32), h_days, rul, cens, delta_frac)
        unc = (1 - _va[7].mean()) if len(_va[0]) else 0.0
        if unc > best_unc + 1e-9: best_frac, best_unc = frac, unc
        if unc >= MIN_VAL_UNCENS: break
    va_s = tr_s + int((best_frac if best_unc > 0 else 0.75) * (te_s - tr_s))
    fs = MinMaxScaler().fit(X_all[tr_s:va_s]); X_s = fs.transform(X_all)
    S = float(np.nanpercentile((R_REV + R_IRR)[tr_s:va_s], 99)) + 1e-9
    inc_a = fwd_inc(R_TOT, H_A); ia = inc_a[tr_s:va_s]; ia = ia[np.isfinite(ia) & (ia > 0)]
    tau_a = float(np.percentile(ia, 75)) if len(ia) > 20 else float("inf")
    ya = (inc_a > tau_a).astype(np.float32)
    tr, va, te = (_windows(a, b, X_s, S, ya, h_days, rul, cens, delta_frac) for a, b in ((tr_s, va_s), (va_s, te_s), (te_s, te_e)))
    m = lambda a: float(a.mean()) if len(a) else float("nan")
    pos = {"A_train": m(tr[3]), "A_val": m(va[3]), "A_test": m(te[3]),
           "uncens_train": 1 - m(tr[7]), "uncens_val": 1 - m(va[7]), "uncens_test": 1 - m(te[7]),
           "val_frac": round((va_s - tr_s) / max(te_s - tr_s, 1), 2),
           "median_rul_test": float(np.median(te[6][te[7] < 0.5])) if len(te[6]) and (te[7] < 0.5).any() else float("nan")}
    return Fold(k, tr, va, te, fs, S, tau_a, pos, h_days, delta_frac, tr_s, va_s, te_s, te_e)


n = len(df)
_b = np.round(np.array([0.0, 0.30, 0.44, 0.58, 0.72, 0.86, 1.0] if not FAST_MODE else [0.0, 0.5, 0.75, 1.0]) * n).astype(int)
FOLDS = [(0, _b[k], _b[k + 1]) for k in range(1, len(_b) - 1)]
FOLD_DATA = [f for f in (build_fold(k + 1, *fd) for k, fd in enumerate(FOLDS)) if len(f.tr[0]) and len(f.va[0]) and len(f.te[0])]
if not FOLD_DATA:
    raise RuntimeError(f"No fold contains a complete window: each test block must exceed {(SEQ_LEN + H_DAYS_PRIMARY * STEPS_PER_DAY) / STEPS_PER_DAY:.0f} days "
                       f"(1 d context + {H_DAYS_PRIMARY} d horizon). Record is {len(df) / STEPS_PER_DAY:.0f} days; reduce H_DAYS_PRIMARY or N_FOLDS.")
print(f"\n{'fold':>4} {'train':>6} {'val':>5} {'test':>5} | val% | uncensored tr/va/te | median RUL te")
for f in FOLD_DATA:
    print(f"{f.k:>4} {len(f.tr[0]):>6} {len(f.va[0]):>5} {len(f.te[0]):>5} | {100*f.pos['val_frac']:>3.0f}% | "
          f"{f.pos['uncens_train']:.3f} / {f.pos['uncens_val']:.3f} / {f.pos['uncens_test']:.3f} | {f.pos['median_rul_test']:>6.1f} d")
import hashlib as _hl
DATA_SIG = _hl.sha1(f"{len(df)}|{STRIDE}|{SEQ_LEN}|{sorted(df['period'].unique().tolist())}|{H_DAYS_PRIMARY}|{DELTA_FRAC_PRIMARY}".encode()).hexdigest()[:6]
_tot = sum(len(f.te[0]) for f in FOLD_DATA)
print(f"\nData geometry: {len(df):,} rows over {df['time_days'].iloc[-1]:.0f} days, periods {sorted(df['period'].unique().tolist())}, "
      f"stride {STRIDE}, {_tot:,} test windows in total  [signature {DATA_SIG}]")
if _tot < 20000:
    print("WARNING: far fewer windows than the full 770-day record at stride 3 yields (~43,000 test windows). "
          "Check PERIODS, STRIDE, and that all four archives loaded; earlier caches will not be reused because the data signature differs.")
_thin = [f.k for f in FOLD_DATA if f.pos["uncens_test"] * len(f.te[0]) < 50]
if _thin: print(f"NOTE folds {_thin} have fewer than 50 uncensored test windows; their prognostic metrics carry wide intervals.")


def make_loader(arrs, shuffle, balance=False, bs=128):
    ds = TensorDataset(*[torch.from_numpy(a) for a in arrs]); sampler = None
    if balance and arrs[3].sum() > 0 and (1 - arrs[3]).sum() > 0:
        y = arrs[3]; w = np.where(y == 1, 1.0 / y.sum(), 1.0 / (1 - y).sum())
        sampler = WeightedRandomSampler(torch.from_numpy(w).double(), len(w), replacement=True); shuffle = False
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, sampler=sampler, drop_last=shuffle or sampler is not None,
                      num_workers=N_WORKERS, pin_memory=DEVICE.type == "cuda", persistent_workers=N_WORKERS > 0,
                      prefetch_factor=4 if N_WORKERS > 0 else None)

# %% [markdown]
# ## 4. Architecture: monotone prognostic head
#
# ```
#   X (288 x F) ─► encoder {tcn_lstm | lstm | gru | transformer | tcn} ─► H_seq
#                                     │
#                      per-task queries q_rev , q_prog , q_cls
#                                     │
#            h_k = [ z_k ‖ h_last ‖ x_last ‖ x_last − x_{-1h} ]
#                                     │
#      ┌──────────────┬───────────────┴──────────┬──────────────┐
#   R_rev (4 h)   ΔR_irrev daily (HMCP, H_d)   warning logit   ΔR_total (aux)
#                        │
#          RUL = min{ k : anchor + Σ_{j≤k} δ_j ≥ R_crit }
# ```
#
# **The structural argument.** Because $\hat\Delta(k)=\sum_{j\le k}\text{Softplus}(\delta_j)$ is
# non-decreasing, the set $\{k : \text{anchor}+\hat\Delta(k)\ge R_{\text{crit}}\}$ is an upper
# interval, so its minimum is a **unique, well-defined cleaning date**. For an unconstrained head the
# trajectory may cross, retreat and cross again, in which case no single date exists. We measure this
# directly as a **validity rate**, and it is the metric on which the physics constraint is decisive
# rather than incremental.
#
# **Conformal RUL interval.** Applying the conformal band to the trajectory and reading off where the
# upper and lower envelopes cross $R_{\text{crit}}$ gives $[\text{RUL}_{\text{lo}},
# \text{RUL}_{\text{hi}}]$. Monotonicity guarantees both endpoints exist.
#
# **Loss weighting, corrected.** v5 drove the objective to −4.75 because `exp(-lambda)` in the Kendall
# weighting scales inversely with each task's own loss magnitude, so a term on a $10^{-4}$ scale
# received roughly 200 times the weight of the focal term. Each loss is now divided by a detached
# running scale before weighting, and lambda is clamped to $[-3,3]$.

# %%
"""Section 4: encoder-agnostic model with a monotone prognostic head."""
from sklearn.linear_model import LogisticRegression

SWA_ON = True


@dataclass
class Config:
    n_feat: int = len(FEATS); hidden: int = 128; layers: int = 2; dropout: float = 0.30; heads: int = 4
    encoder: str = "tcn_lstm"            # tcn_lstm | lstm | gru | transformer | tcn
    h_a: int = H_A; h_days: int = H_DAYS_PRIMARY
    tcn_channels: int = 64; tcn_dilations: tuple = (1, 2, 4, 8, 16)
    per_task_query: bool = True; skip_to_head: bool = True; aux_head: bool = True; uncertainty_weighting: bool = True
    hard_mono: bool = True; hybrid: bool = True; softplus_rev: bool = True
    lam_cls: float = 1.0; lam_mass: float = 0.3; lam_mono: float = 0.0; lam_smooth: float = 0.05; lam_aux: float = 1.0; lam_rul: float = 0.5; lam_anchor: float = 50.0; mix_init: float = 0.0   # +3 -> linear seed, -3 -> power-law seed, 0 -> mean
    focal_alpha: float = 0.75; focal_gamma: float = 2.0; grad_ckpt: bool = True; name: str = "PC-Net-HMCP"


class TCNStem(nn.Module):
    def __init__(self, n_in, ch, dilations, dropout):
        super().__init__(); self.blocks = nn.ModuleList(); c = n_in
        for d in dilations:
            self.blocks.append(nn.ModuleDict({"conv": nn.Conv1d(c, ch, 3, dilation=d),
                                              "proj": nn.Conv1d(c, ch, 1) if c != ch else nn.Identity(),
                                              "norm": nn.GroupNorm(1, ch), "drop": nn.Dropout(dropout),
                                              "pad": nn.ConstantPad1d((2 * d, 0), 0.0)})); c = ch
        self.out_dim = c

    def forward(self, x):
        h = x.transpose(1, 2)
        for b in self.blocks:
            h = b["drop"](F.gelu(b["norm"](b["conv"](b["pad"](h)) + b["proj"](h))))
        return h.transpose(1, 2)


class Encoder(nn.Module):
    """Five interchangeable encoders, all causal, all returning (B, L, hidden)."""

    def __init__(self, cfg):
        super().__init__(); self.kind = cfg.encoder; d = cfg.hidden
        if self.kind in ("tcn_lstm", "tcn"):
            self.tcn = TCNStem(cfg.n_feat, cfg.tcn_channels, cfg.tcn_dilations, cfg.dropout)
        if self.kind == "tcn_lstm":
            self.rnn = nn.LSTM(cfg.tcn_channels + cfg.n_feat, d, cfg.layers, batch_first=True, dropout=cfg.dropout if cfg.layers > 1 else 0.0)
        elif self.kind == "lstm":
            self.rnn = nn.LSTM(cfg.n_feat, d, cfg.layers, batch_first=True, dropout=cfg.dropout if cfg.layers > 1 else 0.0)
        elif self.kind == "gru":
            self.rnn = nn.GRU(cfg.n_feat, d, cfg.layers, batch_first=True, dropout=cfg.dropout if cfg.layers > 1 else 0.0)
        elif self.kind == "transformer":
            self.inp = nn.Linear(cfg.n_feat, d); self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, cfg.dropout, batch_first=True, norm_first=True, activation="gelu")
            self.tr = nn.TransformerEncoder(layer, cfg.layers)
            self.mask = None
        elif self.kind == "tcn":
            self.proj = nn.Linear(cfg.tcn_channels, d)
        else:
            raise ValueError(f"unknown encoder {self.kind}")

    def forward(self, x):
        if self.kind == "tcn_lstm":
            return self.rnn(torch.cat([self.tcn(x), x], -1))[0]
        if self.kind in ("lstm", "gru"):
            return self.rnn(x)[0]
        if self.kind == "transformer":
            L = x.shape[1]
            if self.mask is None or self.mask.shape[0] != L:
                self.mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
            return self.tr(self.inp(x) + self.pos[:, :L], mask=self.mask)
        return self.proj(self.tcn(x))


class MultiQueryPooler(nn.Module):
    def __init__(self, d, heads, tasks=("rev", "prog", "cls")):
        super().__init__(); self.h, self.dk, self.tasks = heads, d // heads, tasks
        self.q = nn.ParameterDict({t: nn.Parameter(torch.randn(1, 1, d) / math.sqrt(d)) for t in tasks})
        self.k, self.v = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.o = nn.ModuleDict({t: nn.Linear(d, d) for t in tasks})
        self.norm = nn.ModuleDict({t: nn.LayerNorm(d) for t in tasks})

    def forward(self, hs, return_attn=False):
        b, L, d = hs.shape
        k = self.k(hs).view(b, L, self.h, self.dk).transpose(1, 2); v = self.v(hs).view(b, L, self.h, self.dk).transpose(1, 2)
        out, attn = {}, {}
        for t in self.tasks:
            q = self.q[t].expand(b, -1, -1).view(b, 1, self.h, self.dk).transpose(1, 2)
            if return_attn:
                a = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.dk), -1); z = a @ v; attn[t] = a.mean(1).squeeze(1)
            else:
                z = F.scaled_dot_product_attention(q, k, v)
            out[t] = self.norm[t](self.o[t](z.transpose(1, 2).reshape(b, d)))
        return out, (attn if return_attn else None)


class MonotoneProgHead(nn.Module):
    """Dual-base hard-monotone hybrid head (N2).

    Two physics references are supplied per window: the linear increments s_lin·1 and the power-law
    increments p_k fitted on the same trailing history. The head learns a per-window mixing weight
    w = sigmoid(m) and a per-step multiplicative correction:

        base_k  = w · s_lin + (1 − w) · p_k
        delta_k = softplus(a_k + c) · base_k + softplus(r_k − 12),     c = ln(e − 1)

    At initialisation w = sigmoid(mix_init): +3 starts at the linear extrapolation, -3 at the power law,
    0 at their mean. The caller sets mix_init from which reference is better on the training windows. Every term is non-negative, so the cumulative sum is
    non-decreasing for every input. `hard=False` is the unconstrained control on the same inputs.
    """

    C0 = math.log(math.e - 1.0)

    def __init__(self, d_in, h_days, do, hard, hybrid=True, mix_init=0.0):
        super().__init__(); self.hard, self.hybrid, self.h = hard, hybrid, h_days
        d = d_in + 1 + h_days
        self.scale = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(do), nn.Linear(256, h_days))
        self.resid = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(do), nn.Linear(256, h_days))
        self.mix = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
        if hard:
            nn.init.zeros_(self.scale[-1].weight); nn.init.constant_(self.scale[-1].bias, 0.0)
            nn.init.zeros_(self.resid[-1].weight); nn.init.constant_(self.resid[-1].bias, -12.0)
            nn.init.zeros_(self.mix[-1].weight); nn.init.constant_(self.mix[-1].bias, float(mix_init))

    def forward(self, z, s_lin, pw):
        zz = torch.cat([z, s_lin[:, None], pw], -1)
        a, r, m = self.scale(zz), self.resid(zz), self.mix(zz)
        if not self.hard:
            return a + r
        if self.hybrid:
            w = torch.sigmoid(m)                                     # (B, 1)
            base = w * s_lin[:, None] + (1 - w) * pw                 # (B, H)
            inc = F.softplus(a + self.C0) * base + F.softplus(r)
        else:
            inc = F.softplus(a) + F.softplus(r)
        return torch.cumsum(inc, 1)


class LevelHead(nn.Module):
    def __init__(self, d_in, h, do, softplus):
        super().__init__(); self.sp = softplus
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Dropout(do), nn.Linear(256, h))

    def forward(self, z):
        o = self.net(z); return F.softplus(o) if self.sp else o


class PCNet(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        self.enc = Encoder(cfg)
        tasks = ("rev", "prog", "cls") if cfg.per_task_query else ("shared",)
        self.pool = MultiQueryPooler(cfg.hidden, cfg.heads, tasks)
        self.skip_dim = (cfg.hidden + 2 * cfg.n_feat) if cfg.skip_to_head else 0
        d_head = cfg.hidden + self.skip_dim
        self.head_rev = LevelHead(d_head, cfg.h_a, cfg.dropout, cfg.softplus_rev)
        self.head_prog = MonotoneProgHead(d_head, cfg.h_days, cfg.dropout, cfg.hard_mono, cfg.hybrid, cfg.mix_init)
        self.head_cls = nn.Sequential(nn.Linear(d_head, 256), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_aux = nn.Sequential(nn.Linear(d_head, 256), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(256, 1)) if cfg.aux_head else None
        if cfg.uncertainty_weighting: self.log_var = nn.Parameter(torch.zeros(5))   # rev, prog, cls, aux, rul

    def _encode(self, x): return self.enc(x)

    def forward(self, x, s_lin, pw, return_attn=False, return_z=False):
        hs = grad_checkpoint(self._encode, x, use_reentrant=False) if (self.cfg.grad_ckpt and self.training) else self._encode(x)
        pooled, attn = self.pool(hs, return_attn)
        skip = torch.cat([hs[:, -1], x[:, -1], x[:, -1] - x[:, -13]], -1) if self.cfg.skip_to_head else None

        def hin(t):
            z = pooled.get(t, pooled.get("shared"))
            return torch.cat([z, skip], -1) if skip is not None else z

        rr, dp = self.head_rev(hin("rev")), self.head_prog(hin("prog"), s_lin, pw)
        zc = hin("cls"); lg = self.head_cls(zc).squeeze(-1)
        aux = self.head_aux(zc).squeeze(-1) if self.head_aux is not None else torch.zeros_like(lg)
        ex = {"aux": aux}
        if return_attn and attn is not None: ex["attn"] = attn.get("prog", next(iter(attn.values())))
        if return_z: ex["z"] = zc
        return rr, dp, lg, ex


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def focal_loss(logit, y, a, g):
    p = torch.sigmoid(logit); pt = torch.where(y == 1, p, 1 - p)
    at = torch.where(y == 1, torch.full_like(p, a), torch.full_like(p, 1 - a))
    return (-at * (1 - pt).pow(g) * torch.log(pt.clamp_min(1e-7))).mean()


class LossScaler:
    """Detached exponential running mean of each loss, used to normalise before task weighting."""

    def __init__(self, keys, momentum=0.99):
        self.m = {k: None for k in keys}; self.mom = momentum

    def norm(self, k, v):
        x = float(v.detach())
        self.m[k] = x if self.m[k] is None else self.mom * self.m[k] + (1 - self.mom) * x
        return v / max(self.m[k], 1e-8)


def soft_rul(dp, anchor, rcrit, h_days, rel_tau=0.05):
    """Differentiable expected first-crossing time of the threshold (Fix 3).

    With level L_k = anchor + dp_k and p_k = sigma((L_k - R_crit)/tau), the expected waiting time is
    1 + sum_k prod_{j<=k} (1 - p_j). tau scales with the remaining gap so the sharpness is invariant
    to units. The value lies in [1, h_days + 1] and is exact in the hard-sigmoid limit.
    """
    gap = (rcrit - anchor).clamp_min(1e-6)
    tau = (rel_tau * gap).clamp_min(1e-5)[:, None]
    p = torch.sigmoid((anchor[:, None] + dp - rcrit[:, None]) / tau)          # (B, H)
    surv = torch.cumprod(1.0 - p, dim=1)                                      # P(not crossed by k)
    return 1.0 + surv.sum(1)


def composite_loss(model, scaler, rr, dp, lg, aux, yr, yd, ya, yac, cfg, an=None, rc=None, rul=None, cens=None):
    """Scale-normalised, uncertainty-weighted multi-task loss with physics penalties and a soft-RUL term."""
    rr, dp, lg, aux = rr.float(), dp.float(), lg.float(), aux.float()
    raw = {"rev": F.mse_loss(rr, yr), "prog": F.mse_loss(dp, yd),
           "cls": focal_loss(lg, ya, cfg.focal_alpha, cfg.focal_gamma),
           "aux": F.smooth_l1_loss(aux, yac) if cfg.aux_head else torch.zeros((), device=rr.device)}
    if cfg.lam_rul > 0 and an is not None:
        sr = soft_rul(dp, an.float(), rc.float(), cfg.h_days)
        unc = (cens.float() < 0.5)
        l_unc = F.smooth_l1_loss(sr[unc], rul.float()[unc]) if unc.any() else torch.zeros((), device=rr.device)
        l_cen = F.relu(float(cfg.h_days) - sr[~unc]).mean() if (~unc).any() else torch.zeros((), device=rr.device)
        raw["rul"] = l_unc + 0.25 * l_cen                                    # censored: must not cross early
    else:
        raw["rul"] = torch.zeros((), device=rr.device)
    nrm = {k: scaler.norm(k, v) for k, v in raw.items()}
    if cfg.uncertainty_weighting and hasattr(model, "log_var"):
        lv = model.log_var.clamp(-3.0, 3.0)
        data = sum(torch.exp(-lv[i]) * w * nrm[k] for i, (k, w) in enumerate(
            (("rev", 1.0), ("prog", 1.0), ("cls", cfg.lam_cls), ("aux", cfg.lam_aux), ("rul", cfg.lam_rul)))) + 0.5 * lv.sum()
    else:
        data = nrm["rev"] + nrm["prog"] + cfg.lam_cls * nrm["cls"] + cfg.lam_aux * nrm["aux"] + cfg.lam_rul * nrm["rul"]
    l_mass = F.mse_loss(rr.mean(1), dp[:, 0] / max(STEPS_PER_DAY / cfg.h_a, 1.0))
    l_mono = F.relu(dp[:, :-1] - dp[:, 1:]).mean()
    l_smooth = (rr[:, 1:] - rr[:, :-1]).pow(2).mean()
    tot = data + cfg.lam_mass * l_mass + cfg.lam_mono * l_mono + cfg.lam_smooth * l_smooth
    return tot, {k: float(v.detach()) for k, v in raw.items()} | {"mass": float(l_mass), "mono": float(l_mono), "smooth": float(l_smooth)}


import hashlib


def arch_fingerprint(cfg) -> str:
    """Short hash of everything that determines tensor shapes AND the data geometry (record length, stride,
    periods, horizon). Baked into every checkpoint and cache key, so nothing from a different architecture
    or a different window set can be loaded by mistake."""
    key = json.dumps({"cfg": asdict(cfg), "n_feat": len(FEATS), "seq": SEQ_LEN, "h_a": H_A,
                      "tuple": 12, "head": "anchored_bestseed_epoch0guard_v5", "data": DATA_SIG}, sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:8]


ARCH = arch_fingerprint(Config())
print(f"architecture fingerprint {ARCH}  (checkpoints and caches from other architectures are ignored)")
for _e in ("tcn_lstm", "lstm", "gru", "transformer", "tcn"):
    print(f"  encoder {_e:12s} params {count_params(PCNet(Config(encoder=_e))):,}")

# %% [markdown]
# ## 5. RUL extraction, prognostic metrics, and training

# %%
"""Section 5: RUL from trajectories, PHM metrics, trainer."""


def traj_to_rul(traj: np.ndarray, anchor: np.ndarray, rcrit: np.ndarray, h_days: int) -> Tuple[np.ndarray, np.ndarray]:
    """First crossing of R_crit. Returns (RUL in days, valid flag).

    A prediction is *valid* only if the trajectory is non-decreasing, so the crossing set is an
    upper interval and the cleaning date is unique. HMCP satisfies this by construction.
    """
    lvl = anchor[:, None] + traj
    above = lvl >= rcrit[:, None]
    mono = np.all(np.diff(traj, axis=1) >= -1e-9, axis=1)
    first = np.where(above.any(1), above.argmax(1) + 1, h_days).astype(float)
    # a non-monotone trajectory that re-crosses has no unique date
    recross = np.array([len(np.flatnonzero(np.diff(a.astype(int)) != 0)) > 1 for a in above])
    return first, mono & ~recross


def alpha_lambda(true, pred, alpha=0.20):
    """Standard PHM alpha-lambda accuracy: fraction of predictions within +/- alpha of true RUL."""
    ok = np.abs(pred - true) <= alpha * np.maximum(true, 1e-9)
    return float(ok.mean())


def prog_metrics(true, pred, valid, cens, h_days, alpha=0.20):
    """Prognostic metrics on uncensored windows; validity and coverage on all windows."""
    u = (cens < 0.5) & valid
    out = {"validity_rate": float(valid.mean()), "n_uncensored": int((cens < 0.5).sum())}
    if u.sum() < 5:
        out.update({k: float("nan") for k in ("MAE_days", "RMSE_days", "Bias_days", "alpha_lambda_20", "alpha_lambda_10")}); out["n_scored"] = int(u.sum())
        return out
    t, p = true[u], pred[u]
    out.update({"MAE_days": float(np.mean(np.abs(p - t))), "RMSE_days": float(np.sqrt(np.mean((p - t) ** 2))),
                "Bias_days": float(np.mean(p - t)), "alpha_lambda_20": alpha_lambda(t, p, 0.20),
                "alpha_lambda_10": alpha_lambda(t, p, 0.10), "n_scored": int(u.sum())})
    return out


BLEND_PRIOR, BLEND_SHRINK = 0.35, 0.5


def _rank01(x):
    r = stats.rankdata(x, method="average"); return (r - r.min()) / max(r.max() - r.min(), 1e-9)


def _blend_feats(o): return np.column_stack([_rank01(o["p"][0]), _rank01(o["aux"][0])])


def fit_blend(o_val):
    y = o_val["ya"]
    if len(np.unique(y)) < 2: return {"kind": "fixed", "w": BLEND_PRIOR}
    X = _blend_feats(o_val)
    try:
        lr = LogisticRegression(C=1.0, max_iter=400, class_weight="balanced").fit(X, y)
        if roc_auc_score(y, lr.decision_function(X)) < roc_auc_score(y, X[:, 0]): raise ValueError
        w = float(np.clip(lr.coef_[0, 1] / max(abs(lr.coef_[0, 0]) + abs(lr.coef_[0, 1]), 1e-9), 0, 1))
        return {"kind": "logistic", "model": lr, "w": w}
    except Exception:  # noqa: BLE001
        ws = np.linspace(0, 1, 21)
        sc = [roc_auc_score(y, (1 - w) * X[:, 0] + w * X[:, 1]) for w in ws]
        return {"kind": "fixed", "w": BLEND_SHRINK * float(ws[int(np.argmax(sc))]) + (1 - BLEND_SHRINK) * BLEND_PRIOR}


def blend_scoreA(o, bl):
    X = _blend_feats(o)
    return bl["model"].decision_function(X) if bl["kind"] == "logistic" else (1 - bl["w"]) * X[:, 0] + bl["w"] * X[:, 1]


@torch.no_grad()
def predict(model, loader, mc=1, extras=False):
    model.eval()
    if mc > 1:
        for m in model.modules():
            if isinstance(m, nn.Dropout): m.train()
    K = ("rr", "dp", "p", "aux"); acc = {k: [] for k in K}
    tgt = {k: [] for k in ("yr", "yd", "ya", "yac", "an", "rul", "cens", "rc", "sl", "pw", "rca")}
    Z, At = [], []
    for xb, yr, yd, ya, yac, an, rul, cens, rc, sl, pw, rca in loader:
        xb, slb, pwb = xb.to(DEVICE, non_blocking=True), sl.to(DEVICE, non_blocking=True), pw.to(DEVICE, non_blocking=True); s = {k: [] for k in K}
        for _ in range(mc):
            with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=BF16_OK):
                rr, dp, lg, ex = model(xb, slb, pwb, return_attn=extras, return_z=extras)
            s["rr"].append(rr.float()); s["dp"].append(dp.float()); s["p"].append(torch.sigmoid(lg.float())); s["aux"].append(ex["aux"].float())
        for k in K: acc[k].append(torch.stack(s[k]).cpu())
        for k, v in zip(("yr", "yd", "ya", "yac", "an", "rul", "cens", "rc", "sl", "pw", "rca"), (yr, yd, ya, yac, an, rul, cens, rc, sl, pw, rca)): tgt[k].append(v)
        if extras:
            Z.append(ex["z"].float().cpu())
            if ex.get("attn") is not None: At.append(ex["attn"].float().cpu())
    o = {k: torch.cat(v, 1).numpy() for k, v in acc.items()}
    o.update({k: torch.cat(v).numpy() for k, v in tgt.items()})
    if extras:
        o["z"] = torch.cat(Z).numpy()
        if At: o["attn"] = torch.cat(At).numpy()
    return o


def rul_from_output(o, h_days, mc_idx=0):
    return traj_to_rul(o["dp"][mc_idx], o["an"], o["rc"], h_days)


def prog_val_metric(model_or_out, loader=None, h_days=None):
    """Prognostic validation loss, lower is better, and always finite.

    Primary criterion is RUL MAE on uncensored windows. When a validation slice contains too few
    uncensored windows for that to be defined, we fall back to the trajectory RMSE rescaled to day
    units so the two are comparable in magnitude. Without this fallback the checkpoint is never
    written and downstream loading fails.
    """
    o = model_or_out if isinstance(model_or_out, dict) else predict(model_or_out, loader)
    pred, valid = rul_from_output(o, h_days)
    m = prog_metrics(o["rul"], pred, valid, o["cens"], h_days)
    mae = m["MAE_days"]
    if not np.isfinite(mae):
        mae = float(np.sqrt(np.mean((o["yd"] - o["dp"][0]) ** 2))) * float(h_days)
        m["fallback_used"] = True
    else:
        m["fallback_used"] = False
    return float(mae), m


def val_score_prog(model, loader, h_days):
    return -prog_val_metric(model, loader, h_days)[0]


def val_score_A(o):
    y = o["ya"]
    if len(np.unique(y)) < 2: return 0.0
    s = blend_scoreA(o, fit_blend(o)); return 0.5 * (roc_auc_score(y, s) + average_precision_score(y, s))


def cosine_warmup(opt, warm, total, floor=0.05):
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))


def train_model(model, cfg, trn, val, epochs=30, lr=1e-4, patience=10, tag="m", verbose=True):
    """Keeps separate best-P (prognostics) and best-A (excursion) checkpoints.

    Resume: if this tag's history and both checkpoints already exist in Drive, training is skipped
    and the stored paths are returned. A run interrupted mid-way is restarted from scratch for that
    tag only; every other completed tag is reused.
    """
    tag = f"{tag}_{arch_fingerprint(cfg)}"          # architecture-stamped: old checkpoints cannot collide
    hist_p = OUT / "models" / f"{tag}_history.json"
    ckP0, ckA0 = OUT / "models" / f"{tag}_P.pt", OUT / "models" / f"{tag}_A.pt"
    if hist_p.exists() and ckP0.exists() and ckA0.exists():
        h = json.loads(hist_p.read_text())
        if h.get("complete"):
            if verbose: print(f"  [{tag}] resumed from Drive (best RUL MAE {h.get('bestP', float('nan')):.3f} d)")
            if h.get("ckP_is_seed") and (OUT / "models" / f"{tag}_SEED.pt").exists(): ckP_saved = OUT / "models" / f"{tag}_SEED.pt"
            elif h.get("ckP_is_swa") and (OUT / "models" / f"{tag}_S.pt").exists(): ckP_saved = OUT / "models" / f"{tag}_S.pt"
            else: ckP_saved = ckP0
            return h, ckP_saved, ckA0
    model.to(DEVICE)
    # Seed anchor: the prognostic head starts at the physics seed (mean of linear and power-law extrapolation).
    # An L2 penalty toward that initialisation keeps training from discarding it; checkpoint selection includes
    # epoch 0, so the deployed head is never worse on validation than the seed.
    _anchor = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith("head_prog.")}
    _n_anchor = max(sum(p.numel() for p in _anchor.values()), 1)
    ck0 = OUT / "models" / f"{tag}_SEED.pt"; torch.save(model.state_dict(), ck0)   # the physics seed, untrained
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, fused=DEVICE.type == "cuda")
    steps = epochs * max(1, len(trn)); sched = cosine_warmup(opt, int(0.05 * steps), steps)
    scaler = LossScaler(("rev", "prog", "cls", "aux", "rul"))
    hist = {"train": [], "maeP": [], "scoreA": [], "parts": []}
    bestP, bestA, epP, epA, bad = 1e9, -1e9, 0, 0, 0
    ckP, ckA = OUT / "models" / f"{tag}_P.pt", OUT / "models" / f"{tag}_A.pt"
    swa = torch.optim.swa_utils.AveragedModel(model) if SWA_ON else None
    swa_start = int(0.70 * epochs); t0 = time.time(); ep = 0
    for ep in range(epochs):
        model.train(); tl, pa = [], []
        for xb, yr, yd, ya, yac, an, rul, cens, rc, sl, pw, rca in trn:
            xb, yr, yd, ya, yac, an, rul, cens, rc, sl, pw = (t.to(DEVICE, non_blocking=True) for t in (xb, yr, yd, ya, yac, an, rul, cens, rc, sl, pw))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=BF16_OK):
                rr, dp, lg, ex = model(xb, sl, pw)
            loss, parts = composite_loss(model, scaler, rr, dp, lg, ex["aux"], yr, yd, ya, yac, cfg, an, rc, rul, cens)
            if cfg.lam_anchor > 0 and _anchor:
                l_anc = sum(((p - _anchor[n]) ** 2).sum() for n, p in model.named_parameters() if n in _anchor) / _n_anchor
                loss = loss + cfg.lam_anchor * l_anc; parts["anchor"] = float(l_anc)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tl.append(loss.item()); pa.append(parts)
        if swa is not None and ep >= swa_start: swa.update_parameters(model)
        ov = predict(model, val)
        maeP, mP = prog_val_metric(ov, h_days=cfg.h_days)
        if ep == 0:
            # guarantee both checkpoints exist before any comparison can fail
            torch.save(model.state_dict(), ckP); torch.save(model.state_dict(), ckA)
            if mP.get("fallback_used") and verbose:
                print(f"  [{tag}] note: validation has {mP['n_uncensored']} uncensored windows; "
                      f"using trajectory-RMSE fallback for checkpoint selection")
        sA = val_score_A(ov)
        hist["train"].append(float(np.mean(tl))); hist["maeP"].append(float(maeP)); hist["scoreA"].append(float(sA))
        hist["parts"].append({k: float(np.mean([q[k] for q in pa])) for k in pa[0]})
        imp = False
        if maeP <= bestP: bestP, epP, imp = maeP, ep, True; torch.save(model.state_dict(), ckP)
        if sA >= bestA: bestA, epA, imp = sA, ep, True; torch.save(model.state_dict(), ckA)
        bad = 0 if imp else bad + 1
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  [{tag}] ep {ep:3d} loss {hist['train'][-1]:.4f} | RUL MAE {maeP:.3f} d (best {bestP:.3f}@{epP}) | A {sA:.4f} (best {bestA:.4f}@{epA})")
        if bad >= patience:
            if verbose: print(f"  [{tag}] early stop @ {ep}")
            break
    if swa is not None and ep >= swa_start:
        torch.optim.swa_utils.update_bn(trn, swa, device=DEVICE); sw = swa.module
        m_swa, _ = prog_val_metric(sw, val, cfg.h_days)
        ckS = OUT / "models" / f"{tag}_S.pt"; torch.save(sw.state_dict(), ckS)
        if np.isfinite(m_swa) and m_swa < bestP: bestP, ckP = m_swa, ckS
        hist["swa_maeP"] = float(m_swa) if np.isfinite(m_swa) else None
    # ── Seed guard ──────────────────────────────────────────────────────────────
    # Compare the best trained checkpoint against the untrained physics seed on validation and deploy the
    # better of the two. Training can therefore never leave the model worse than the classical reference it
    # was initialised from. Selection uses validation data only.
    model.load_state_dict(torch.load(ck0, map_location=DEVICE, weights_only=True))
    mae_seed, _ = prog_val_metric(model, val, cfg.h_days)
    hist["seed_val_mae"], hist["trained_val_mae"] = float(mae_seed), float(bestP)
    if mae_seed < bestP:
        ckP, bestP = ck0, mae_seed; hist["deployed"] = "physics_seed"
        if verbose: print(f"  [{tag}] seed guard: deploying the untrained seed (val MAE {mae_seed:.3f} < trained {hist['trained_val_mae']:.3f})")
    else:
        hist["deployed"] = "trained"
        if verbose: print(f"  [{tag}] seed guard: deploying the trained head (val MAE {bestP:.3f} <= seed {mae_seed:.3f})")
    hist.update({"time_s": time.time() - t0, "best_epoch_P": epP, "best_epoch_A": epA, "bestP": bestP, "bestA": bestA,
                 "complete": True, "ckP_is_swa": str(ckP).endswith("_S.pt"), "ckP_is_seed": str(ckP).endswith("_SEED.pt")})
    (OUT / "models" / f"{tag}_history.json").write_text(json.dumps(hist))
    for _ck in (ckP, ckA):
        if not Path(_ck).exists():
            torch.save(model.state_dict(), _ck)   # last resort: current weights
    return hist, ckP, ckA


_GRID = np.linspace(0.01, 0.99, 99)


def calibrate_threshold(p, y):
    if len(np.unique(y)) < 2: return float(np.median(p))
    g = np.unique(np.concatenate([_GRID, np.percentile(p, np.arange(1, 100))]))
    return float(g[int(np.argmax([f1_score(y, (p > t).astype(int), zero_division=0) for t in g]))])


def cls_metrics(p, y, thr):
    if len(np.unique(y)) < 2: return {k: np.nan for k in ("ROC-AUC", "PR-AUC", "F1@calib", "F1*", "Bal.Acc")}
    pred = (p > thr).astype(int); g = np.unique(np.concatenate([_GRID, np.percentile(p, np.arange(1, 100))]))
    return {"ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p), "F1@calib": f1_score(y, pred, zero_division=0),
            "F1*": max(f1_score(y, (p > t).astype(int), zero_division=0) for t in g), "Bal.Acc": balanced_accuracy_score(y, pred)}


def bootstrap_ci(fn, *arrs, n_boot=500, seed=SEED):
    rng = np.random.default_rng(seed); idx = np.arange(len(arrs[0])); v = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        try: v.append(fn(*[a[b] for a in arrs]))
        except Exception: continue  # noqa: BLE001
    v = np.array([x for x in v if np.isfinite(x)])
    if not len(v): return float("nan"), float("nan"), float("nan")
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def paired_bootstrap_p(a, b, n_boot=2000, stat=lambda e: np.mean(np.abs(e))):
    """Two-sided paired bootstrap on an error-vector statistic (default MAE)."""
    rng = np.random.default_rng(SEED); idx = np.arange(len(a)); d = []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True); d.append(stat(a[s]) - stat(b[s]))
    d = np.array(d); return float(2 * min((d <= 0).mean(), (d >= 0).mean()))


def delong_p(p1, p2, y):
    def st(p):
        pos, neg = p[y == 1], p[y == 0]
        return np.array([np.mean(x > neg) + 0.5 * np.mean(x == neg) for x in pos]), np.array([np.mean(pos > x) + 0.5 * np.mean(pos == x) for x in neg])
    if len(np.unique(y)) < 2: return float("nan")
    a10, a01 = st(p1); b10, b01 = st(p2)
    s = np.cov(np.stack([a10, b10])) / len(a10) + np.cov(np.stack([a01, b01])) / len(a01); var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    return 1.0 if var <= 0 else float(2 * (1 - stats.norm.cdf(abs((a10.mean() - b10.mean()) / np.sqrt(var)))))


print("Sections 4-5 defined.")
import re

# %% [markdown]
# ## 6. Experiment: validity and accuracy under forward-chaining cross-validation
# Baselines fall into two groups. **Validity controls** share the encoder and inputs but replace HMCP by a soft
# penalty or by nothing; they exist to show that validity is a property of the parameterisation. **Accuracy
# references** are the extrapolations an engineer would do by hand, computed from the identical per-window seeds the
# model receives, plus a gradient-boosted regressor of RUL and persistence.

# %%
"""Section 6: cross-validated comparison."""
EPOCHS = 8 if FAST_MODE else 30
N_SEEDS = 2 if FAST_MODE else 3
SEEDS = [SEED + 101 * i for i in range(N_SEEDS)]
ENCODERS_MAIN = "tcn_lstm"
PROG_MODELS = ("PC-Net-HMCP", "Soft-mono deep", "Unconstrained deep")
REF_MODELS = ("Linear 7d", "Power law", "XGBoost RUL", "Persistence")
ALL_P = (*PROG_MODELS, *REF_MODELS)


def ens_predict(ckpts, cfg, loader, mc=1, extras=False):
    ckpts = [c for c in ckpts if Path(c).exists()]
    if not ckpts: raise RuntimeError("No checkpoints found; check the uncensored fractions in Section 3.5.")
    outs = []
    for ck in ckpts:
        m = PCNet(cfg)
        try: m.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True))
        except RuntimeError as exc: raise RuntimeError(f"Checkpoint {Path(ck).name} does not match fingerprint {arch_fingerprint(cfg)}.") from exc
        m.to(DEVICE); outs.append(predict(m, loader, mc=mc, extras=extras and ck is ckpts[-1])); del m
    torch.cuda.empty_cache()
    o = dict(outs[-1])
    for k in ("rr", "dp"): o[k] = np.mean([q[k] for q in outs], 0)
    for k in ("p", "aux"): o[k] = np.stack([np.mean([_rank01(q[k][0]) for q in outs], 0)])
    return o


def ref_rul(fold, split="te"):
    """Reference forecasts from the SAME per-window seeds the model receives."""
    arrs = getattr(fold, split); an, rc, rul, sl, pwi = arrs[5], arrs[8], arrs[6], arrs[9], arrs[10]
    H = fold.h_days
    lin, _ = traj_to_rul(np.arange(1, H + 1)[None, :] * sl[:, None], an, rc, H)
    pw, _ = traj_to_rul(np.cumsum(pwi, 1), an, rc, H)
    return {"Linear 7d": lin, "Power law": pw, "Persistence": np.full(len(an), float(H)), "true": rul}


def best_seed_for(fold):
    """Which classical reference is better on the TRAINING windows decides the seed (+3 linear, -3 power)."""
    rf = ref_rul(fold, "tr"); t = fold.tr[6]; c = fold.tr[7]
    ml = prog_metrics(t, rf["Linear 7d"], np.ones(len(t), bool), c, fold.h_days)["MAE_days"]
    mp = prog_metrics(t, rf["Power law"], np.ones(len(t), bool), c, fold.h_days)["MAE_days"]
    if not np.isfinite(ml) and not np.isfinite(mp): return 0.0, "mean"
    return (3.0, "linear") if (np.isfinite(ml) and (not np.isfinite(mp) or ml <= mp)) else (-3.0, "power")


def seed_rul(arrs, h_days, mix):
    """Closed-form prediction of the UNTRAINED HMCP head.

    At initialisation the final layers of the scale, residual and mixing networks have zero weights, so the
    head's output does not depend on the encoder: a = 0, r = -12, m = mix_init. The increments are therefore
        delta_k = softplus(c) * (w * s_lin + (1 - w) * p_k) + softplus(-12),  softplus(c) = 1, w = sigmoid(mix_init)
    and the prediction is exactly computable from the per-window seeds. This is what 'physics_seed' deployment
    means, and it lets the learned correction be isolated without any additional training.
    """
    an, rc, sl, pwi = arrs[5], arrs[8], arrs[9], arrs[10]
    w = 1.0 / (1.0 + np.exp(-mix))
    inc = np.logaddexp(0.0, math.log(math.e - 1.0)) * (w * sl[:, None] + (1 - w) * pwi) + np.logaddexp(0.0, -12.0)
    return traj_to_rul(np.cumsum(inc, 1), an, rc, h_days)


CV_P = {k: [] for k in ALL_P}; CV_VALID = {k: [] for k in PROG_MODELS}
LEARN = []            # per fold: seed-only, trained-only and deployed MAE on test
CV_SEEDONLY, CV_TRAINONLY, CV_TRAINVALID = [], [], []
CV_END = {k: [] for k in ("true", "PC-Net-HMCP", "Physics seed only", "Linear 7d", "Power law")}   # 14-day irreversible rise, m^-1
CV_TRUE, CV_CENS, CV_PERIOD, PER_FOLD, FINAL = [], [], [], [], {}
CV_TRAJ = {k: [] for k in PROG_MODELS}          # per-window predicted RUL kept for the stability analysis
for fd in FOLD_DATA:
    mix, seed_name = best_seed_for(fd)
    print(f"\n===== Fold {fd.k}: train {len(fd.tr[0])} / val {len(fd.va[0])} / test {len(fd.te[0])}  | seed from {seed_name} =====")
    print(f"      uncensored  train {fd.pos['uncens_train']:.3f} | val {fd.pos['uncens_val']:.3f} | test {fd.pos['uncens_test']:.3f}   val width {100*fd.pos['val_frac']:.0f}%")
    trn, val, tst = make_loader(fd.tr, True, True), make_loader(fd.va, False), make_loader(fd.te, False)
    row = {"fold": fd.k, "n_test": len(fd.te[0]), "uncens_test": fd.pos["uncens_test"], "seed": seed_name}
    _t0s = fd.te_s + np.arange(len(fd.te[0])) * STRIDE + SEQ_LEN - 1
    CV_PERIOD.append(PER_DAILY[DAY_OF_STEP[np.clip(_t0s, 0, len(DAY_OF_STEP) - 1)]])
    variants = {"PC-Net-HMCP": Config(encoder=ENCODERS_MAIN, h_days=fd.h_days, mix_init=mix),
                "Soft-mono deep": Config(encoder=ENCODERS_MAIN, h_days=fd.h_days, hard_mono=False, lam_mono=0.2, mix_init=mix),
                "Unconstrained deep": Config(encoder=ENCODERS_MAIN, h_days=fd.h_days, hard_mono=False, lam_mono=0.0, softplus_rev=False,
                                             lam_mass=0.0, lam_smooth=0.0, lam_anchor=0.0, mix_init=mix)}
    for vname, vcfg in variants.items():
        _ck = f"fold{fd.k}_{vname.replace(' ', '')}_pred_{arch_fingerprint(vcfg)}"
        _cached = npz_get(_ck)
        if _cached is not None and len(_cached["pred"]) != len(fd.te[0]): _cached = None
        if _cached is not None and vname != "PC-Net-HMCP":
            pred, valid = _cached["pred"], _cached["valid"].astype(bool); ot = {"rul": _cached["rul"], "cens": _cached["cens"]}
        else:
            cks = []
            for si, sd in enumerate(SEEDS if vname == "PC-Net-HMCP" else SEEDS[:1]):
                seed_everything(sd)
                h, cp, ca = train_model(PCNet(vcfg), vcfg, trn, val, EPOCHS, tag=f"f{fd.k}_{vname[:8].replace(' ', '')}_s{si}",
                                        verbose=(fd.k == FOLD_DATA[-1].k and si == 0 and vname == "PC-Net-HMCP"))
                cks.append(cp)
                if si == 0 and vname == "PC-Net-HMCP": h_main = h; row["deployed"] = h.get("deployed", "trained")
            ot = ens_predict(cks, vcfg, tst); pred, valid = rul_from_output(ot, fd.h_days)
            npz_put(_ck, pred=pred, valid=valid, rul=ot["rul"], cens=ot["cens"])
            if vname == "PC-Net-HMCP": FINAL_OT, FINAL_CKS, FINAL_CFG = ot, cks, vcfg
        met = prog_metrics(ot["rul"], pred, valid, ot["cens"], fd.h_days)
        for k, v in met.items(): row[f"{vname}|{k}"] = v
        if vname == "PC-Net-HMCP":
            # isolate the learned correction: untrained seed vs trained checkpoints, same test windows
            ps, vs_ = seed_rul(fd.te, fd.h_days, vcfg.mix_init)
            _tk = f"fold{fd.k}_HMCP_trainedonly_{arch_fingerprint(vcfg)}"
            _tc = npz_get(_tk)
            if _tc is not None and len(_tc["pred"]) == len(fd.te[0]):
                pt_, vt_ = _tc["pred"], _tc["valid"].astype(bool)
            else:
                _fp = arch_fingerprint(vcfg)
                tr_cks = [OUT / "models" / f"f{fd.k}_PC-Net-H_s{si}_{_fp}_P.pt" for si in range(len(SEEDS))]
                tr_cks = [c for c in tr_cks if c.exists()]
                if tr_cks:
                    ot_tr = ens_predict(tr_cks, vcfg, tst); pt_, vt_ = rul_from_output(ot_tr, fd.h_days)
                else:
                    pt_, vt_ = pred, valid
                npz_put(_tk, pred=pt_, valid=vt_)
            m_seed = prog_metrics(ot["rul"], ps, vs_, ot["cens"], fd.h_days)
            m_tr = prog_metrics(ot["rul"], pt_, vt_, ot["cens"], fd.h_days)
            LEARN.append({"fold": fd.k, "seed_ref": seed_name, "deployed": row.get("deployed", "trained"),
                          "MAE_seed_only": m_seed["MAE_days"], "MAE_trained_only": m_tr["MAE_days"], "MAE_deployed": met["MAE_days"],
                          "a20_seed_only": m_seed["alpha_lambda_20"], "a20_trained_only": m_tr["alpha_lambda_20"],
                          "validity_trained_only": m_tr["validity_rate"], "n_scored": met["n_scored"]})
            CV_SEEDONLY.append(ps); CV_TRAINONLY.append(pt_); CV_TRAINVALID.append(vt_)
            # end-of-horizon irreversible-resistance rise in physical units, for the kPa analysis of Section 8
            _w = 1.0 / (1.0 + np.exp(-vcfg.mix_init))
            _seed_traj = np.cumsum(_w * fd.te[9][:, None] + (1 - _w) * fd.te[10] + np.logaddexp(0.0, -12.0), 1)
            CV_END["true"].append(fd.te[2][:, -1] * fd.S)
            CV_END["PC-Net-HMCP"].append(ot["dp"][0][:, -1] * fd.S)
            CV_END["Physics seed only"].append(_seed_traj[:, -1] * fd.S)
            CV_END["Linear 7d"].append(fd.te[9] * fd.h_days * fd.S)
            CV_END["Power law"].append(fd.te[10].sum(1) * fd.S)
            print(f"  {'  of which':20s} seed only {m_seed['MAE_days']:.3f} d | trained only {m_tr['MAE_days']:.3f} d | deployed: {row.get('deployed', 'trained')}")
        CV_P[vname].append(pred); CV_VALID[vname].append(valid); CV_TRAJ[vname].append(np.where(valid, pred, np.nan))
        print(f"  {vname:20s} MAE {met['MAE_days']:.3f} d | a-20 {met['alpha_lambda_20']:.3f} | validity {met['validity_rate']:.3f}")
    refs = ref_rul(fd, "te")
    Xtr, Xva, Xte = (a[0].reshape(len(a[0]), -1) for a in (fd.tr, fd.va, fd.te))
    seed_everything(SEED)
    xr = xgb.XGBRegressor(n_estimators=800, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6, tree_method="hist",
                          device="cuda" if DEVICE.type == "cuda" else "cpu", early_stopping_rounds=60, random_state=SEED)
    xr.fit(Xtr, fd.tr[6], eval_set=[(Xva, fd.va[6])], verbose=False)
    refs["XGBoost RUL"] = np.clip(xr.predict(Xte), 1, fd.h_days)
    for rn in REF_MODELS:
        m = prog_metrics(fd.te[6], refs[rn], np.ones(len(refs[rn]), bool), fd.te[7], fd.h_days)
        for k, v in m.items(): row[f"{rn}|{k}"] = v
        CV_P[rn].append(refs[rn])
        print(f"  {rn:20s} MAE {m['MAE_days']:.3f} d | a-20 {m['alpha_lambda_20']:.3f}")
    CV_TRUE.append(fd.te[6]); CV_CENS.append(fd.te[7]); PER_FOLD.append(row)
    if fd.k == FOLD_DATA[-1].k:
        FINAL = {"fd": fd, "cfg": FINAL_CFG, "cks": FINAL_CKS, "ot": FINAL_OT, "hist": h_main, "loaders": (trn, val, tst), "refs": refs}
    torch.cuda.empty_cache()
stage_mark("cv_complete")

PER_FOLD_DF = pd.DataFrame(PER_FOLD).set_index("fold")
TRUE, CENS = np.concatenate(CV_TRUE), np.concatenate(CV_CENS)
for _nm in list(CV_P):
    if len(np.concatenate(CV_P[_nm])) != len(TRUE): raise RuntimeError(f"{_nm}: stale cache; delete results/cache and rerun.")
RES_P = {}
for nm in ALL_P:
    pr = np.concatenate(CV_P[nm]); vd = np.concatenate(CV_VALID[nm]) if nm in CV_VALID else np.ones(len(pr), bool)
    m = prog_metrics(TRUE, pr, vd, CENS, FOLD_DATA[0].h_days); u = (CENS < 0.5) & vd
    _, lo, hi = bootstrap_ci(lambda t, p: np.mean(np.abs(p - t)), TRUE[u], pr[u]); m["MAE CI"] = f"[{lo:.3f}, {hi:.3f}]"
    m["fold_mean_MAE"] = PER_FOLD_DF[f"{nm}|MAE_days"].mean(); m["fold_sd_MAE"] = PER_FOLD_DF[f"{nm}|MAE_days"].std(); RES_P[nm] = m
_u_h = (CENS < 0.5) & np.concatenate(CV_VALID["PC-Net-HMCP"])
P_P = {}
for nm in ALL_P:
    if nm == "PC-Net-HMCP": continue
    vd = np.concatenate(CV_VALID[nm]) if nm in CV_VALID else np.ones(len(TRUE), bool); u = _u_h & vd
    if u.sum() > 10:
        P_P[f"HMCP_vs_{nm}"] = paired_bootstrap_p(np.abs(np.concatenate(CV_P["PC-Net-HMCP"])[u] - TRUE[u]), np.abs(np.concatenate(CV_P[nm])[u] - TRUE[u]))
UNCENS_POOL = float(np.mean(CENS < 0.5)); _best_ref = min(("Linear 7d", "Power law"), key=lambda k: RES_P[k]["MAE_days"])
print("\n=== VALIDITY AND ACCURACY, pooled CV ===\n" + pd.DataFrame(RES_P).T.to_string())

# ── pooled learned-correction summary ──
LEARN_DF = pd.DataFrame(LEARN).set_index("fold")
_so, _to, _tv = np.concatenate(CV_SEEDONLY), np.concatenate(CV_TRAINONLY), np.concatenate(CV_TRAINVALID)
_m_so = prog_metrics(TRUE, _so, np.ones(len(_so), bool), CENS, FOLD_DATA[0].h_days)
_m_to = prog_metrics(TRUE, _to, _tv, CENS, FOLD_DATA[0].h_days)
LEARN_POOL = {"Physics seed only (untrained HMCP)": {"MAE_days": _m_so["MAE_days"], "alpha_lambda_20": _m_so["alpha_lambda_20"], "validity_rate": _m_so["validity_rate"]},
              "Trained HMCP only (no guard)": {"MAE_days": _m_to["MAE_days"], "alpha_lambda_20": _m_to["alpha_lambda_20"], "validity_rate": _m_to["validity_rate"]},
              "Deployed (validation guard)": {k: RES_P["PC-Net-HMCP"][k] for k in ("MAE_days", "alpha_lambda_20", "validity_rate")}}
LEARN_POOL_DF = pd.DataFrame(LEARN_POOL).T; LEARN_POOL_DF.index.name = "HMCP variant"
_u = (CENS < 0.5)
P_LEARN = {"seed_vs_linear": paired_bootstrap_p(np.abs(_so[_u] - TRUE[_u]), np.abs(np.concatenate(CV_P["Linear 7d"])[_u] - TRUE[_u])),
           "deployed_vs_seed": paired_bootstrap_p(np.abs(np.concatenate(CV_P["PC-Net-HMCP"])[_u] - TRUE[_u]), np.abs(_so[_u] - TRUE[_u]))}
print("\n=== LEARNED CORRECTION ===\n" + LEARN_DF.round(3).to_string())
print("\n" + LEARN_POOL_DF.round(3).to_string())
print("p-values:", {k: f"{v:.3g}" for k, v in P_LEARN.items()})
print(f"\nPooled: {len(TRUE):,} windows, {100*UNCENS_POOL:.1f} % uncensored; PC-Net {RES_P['PC-Net-HMCP']['MAE_days']:.3f} d vs best classical ({_best_ref}) {RES_P[_best_ref]['MAE_days']:.3f} d")
print("p-values (paired bootstrap on MAE):", {k: f"{v:.3g}" for k, v in P_P.items()})

# ── N4: per-period ──
PERIOD_POOL = np.concatenate(CV_PERIOD); _pr_h = np.concatenate(CV_P["PC-Net-HMCP"]); _vd_h = np.concatenate(CV_VALID["PC-Net-HMCP"])
RES_PERIOD = {}
for p_ in sorted(np.unique(PERIOD_POOL)):
    m_ = PERIOD_POOL == p_
    mh = prog_metrics(TRUE[m_], _pr_h[m_], _vd_h[m_], CENS[m_], FOLD_DATA[0].h_days)
    RES_PERIOD[int(p_)] = {"n_windows": int(m_.sum()), "uncensored": float((CENS[m_] < 0.5).mean()), "HMCP_MAE": mh["MAE_days"],
                           "HMCP_alpha20": mh["alpha_lambda_20"], "HMCP_validity": mh["validity_rate"],
                           **{f"{rn}_MAE": prog_metrics(TRUE[m_], np.concatenate(CV_P[rn])[m_], np.ones(int(m_.sum()), bool), CENS[m_], FOLD_DATA[0].h_days)["MAE_days"] for rn in ("Linear 7d", "Power law")},
                           "days_in_period": int((DAILY["period"] == p_).sum())}
RES_PERIOD_DF = pd.DataFrame(RES_PERIOD).T; RES_PERIOD_DF.index.name = "Period"
print("\n=== Per-period generalisation ===\n" + RES_PERIOD_DF.round(3).to_string())

# %% [markdown]
# ### 6.4 Day-level statistical inference (reported throughout)
#
# Windows whose origin falls on the same day share one daily label, so they are not independent. Confidence
# intervals and p-values are therefore computed by a **day-block bootstrap**: days are resampled with replacement
# and every window of a selected day enters together. Primary comparisons (PC-Net against each reference) are
# corrected for multiplicity by the **Holm** procedure; the decomposition of the gain (seed, learned correction,
# guard) forms a second family, corrected separately. These values replace the window-level bootstrap in every
# table and figure.

# %%
"""Section 6.4: day-block bootstrap, Holm correction, guard test."""
DAY_POOL = np.concatenate([DAY_OF_STEP[np.clip(fd.te_s + np.arange(len(fd.te[0])) * STRIDE + SEQ_LEN - 1, 0, len(DAY_OF_STEP) - 1)]
                           + 100000 * fd.k for fd in FOLD_DATA])
assert len(DAY_POOL) == len(TRUE), "window/day alignment failed"
B_BOOT = 5000
_rng_boot = np.random.default_rng(SEED)


def block_bootstrap(err_by_model, mask, B=B_BOOT):
    """Mean of each error vector over `mask`, with day-block bootstrap replicates shared across models."""
    days, inv = np.unique(DAY_POOL[mask], return_inverse=True); D = len(days)
    n_day = np.bincount(inv, minlength=D).astype(float)
    sums = {nm: np.bincount(inv, weights=e[mask], minlength=D) for nm, e in err_by_model.items()}
    pick = _rng_boot.integers(0, D, size=(B, D)); den = n_day[pick].sum(1)
    boot = {nm: sums[nm][pick].sum(1) / den for nm in sums}
    point = {nm: float(sums[nm].sum() / n_day.sum()) for nm in sums}
    return point, boot, D, int(n_day.sum())


def p_two_sided(boot, a, b, B=B_BOOT):
    d = boot[a] - boot[b]
    return max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / B)


def holm(pdict):
    """Holm step-down adjusted p-values."""
    items = sorted(pdict.items(), key=lambda kv: kv[1]); m = len(items); out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p)); out[k] = running
    return out


_U = CENS < 0.5
SERIES = {nm: np.concatenate(CV_P[nm]) for nm in ("PC-Net-HMCP", "Linear 7d", "Power law", "XGBoost RUL", "Persistence")}
SERIES["Physics seed only"] = np.concatenate(CV_SEEDONLY)
SERIES["Trained head (no guard)"] = np.concatenate(CV_TRAINONLY)
_err = {nm: np.abs(p - TRUE) for nm, p in SERIES.items()}
BLK_PT, BLK_BOOT, N_DAYS, N_SCORED = block_bootstrap(_err, _U)
BLOCK_DF = pd.DataFrame({nm: {"MAE_days": BLK_PT[nm], "CI_low": float(np.percentile(BLK_BOOT[nm], 2.5)),
                              "CI_high": float(np.percentile(BLK_BOOT[nm], 97.5))} for nm in SERIES}).T.sort_values("MAE_days")
BLOCK_DF.index.name = "Model"
P_PRIMARY = {f"PC-Net vs {r}": p_two_sided(BLK_BOOT, "PC-Net-HMCP", r) for r in ("Linear 7d", "Power law", "XGBoost RUL", "Persistence")}
P_DECOMP = {"Physics seed vs Linear 7d": p_two_sided(BLK_BOOT, "Physics seed only", "Linear 7d"),
            "PC-Net vs Physics seed": p_two_sided(BLK_BOOT, "PC-Net-HMCP", "Physics seed only"),
            "PC-Net vs Trained head (no guard)": p_two_sided(BLK_BOOT, "PC-Net-HMCP", "Trained head (no guard)")}
P_PRIMARY_HOLM, P_DECOMP_HOLM = holm(P_PRIMARY), holm(P_DECOMP)
SIG_TABLE = pd.DataFrame([{"family": fam, "comparison": k, "p_raw": v, "p_holm": adj[k], "significant_0.05": bool(adj[k] < 0.05)}
                          for fam, raw, adj in (("primary", P_PRIMARY, P_PRIMARY_HOLM), ("decomposition", P_DECOMP, P_DECOMP_HOLM))
                          for k, v in raw.items()]).set_index("comparison")
print(f"Day-block bootstrap: {N_DAYS} independent days, {N_SCORED:,} scored windows, {B_BOOT} resamples\n")
print(BLOCK_DF.round(3).to_string()); print("\n" + SIG_TABLE.round(4).to_string())
for nm in ("PC-Net-HMCP", "Linear 7d", "Power law", "XGBoost RUL", "Persistence"):   # replace window-level CIs downstream
    RES_P[nm]["MAE CI"] = f"[{BLOCK_DF.loc[nm, 'CI_low']:.3f}, {BLOCK_DF.loc[nm, 'CI_high']:.3f}]"
P_P = {f"HMCP_vs_{k.replace('PC-Net vs ', '')}": v for k, v in P_PRIMARY.items()}
cache_put("block_bootstrap", {"table": BLOCK_DF.to_dict(orient="index"),
                              "significance": SIG_TABLE.reset_index().to_dict(orient="records"), "n_days": N_DAYS})

# %% [markdown]
# ## 8. Process-engineering layer: results in the units a membrane engineer uses
#
# Resistance is the natural variable for the model, but plant decisions are taken in **kPa of transmembrane
# pressure, kPa per day of fouling and LMH per bar of permeability**. Every resistance is converted back to its
# **TMP-equivalent** at the operating period's median flux and temperature by inverting the Darcy relation of
# Section 3, $\Delta\mathrm{TMP} = \Delta R\,\mu(T_{\rm ref})\,J_{\rm ref}/A$. Three products follow:
#
# 1. an **operating fingerprint** for each period: resistance-in-series budget, hydraulic reversibility, backwash
#    frequency, permeability, irreversible fouling rate in kPa/day, and the size of one degradation step in kPa;
# 2. the **14-day irreversible TMP rise** forecast in kPa, scored on **all** windows, censored or not, with
#    day-block confidence intervals: a trajectory-level accuracy measure with far more statistical power than the
#    date error;
# 3. a descriptive check of the **temperature dependence** of irreversible fouling, reported whatever its outcome.

# %%
"""Section 8: operating fingerprint, physical-unit forecast accuracy, temperature dependence."""
REF_COND = {}
for p_ in sorted(df["period"].unique()):
    m_ = df["period"] == p_
    REF_COND[int(p_)] = {"J_ms": float(np.nanmedian(df.loc[m_, "flux_LMH"])) / 3.6e6,
                         "mu": float(vogel_viscosity(float(np.nanmedian(df.loc[m_, "temperature_C"])))),
                         "J_LMH": float(np.nanmedian(df.loc[m_, "flux_LMH"])), "T_C": float(np.nanmedian(df.loc[m_, "temperature_C"]))}


def r_to_kpa(dR, periods):
    """Invert R = TMP*A/(mu*J) of Section 3 at each window's period reference conditions."""
    periods = np.atleast_1d(periods)
    j = np.array([REF_COND[int(p)]["J_ms"] for p in periods]); mu = np.array([REF_COND[int(p)]["mu"] for p in periods])
    return np.asarray(dR, dtype=float) * mu * j / (A_MEMBRANE * 1e3)


# ── 8.1 operating fingerprint per period ──
FP = {}
for p_ in sorted(df["period"].unique()):
    p_ = int(p_); m_ = df["period"] == p_; dd = DAILY[DAILY["period"] == p_]; ndays = max(len(dd), 1)
    rm, rr, ri = float(R_M[p_]), float(np.nanmean(df.loc[m_, "R_rev"])), float(np.nanmean(df.loc[m_, "R_irrev"]))
    tot = rm + rr + ri
    # Period-pure days only: a day index can straddle a period boundary, so days with fewer than 90 % of their
    # samples inside period p are excluded before the first-to-last irreversible-resistance rate is taken.
    _pure = df.loc[m_].groupby("day")["R_irrev"].agg(["median", "size"])
    _pure = _pure[_pure["size"] >= 0.9 * STEPS_PER_DAY]["median"]
    rate_r = float((_pure.iloc[-1] - _pure.iloc[0]) / max(len(_pure) - 1, 1)) if len(_pure) > 1 else float("nan")
    FP[p_] = {"days": ndays, "median_TMP_kPa": float(np.nanmedian(df.loc[m_, "TMP_kPa"])),
              "median_flux_LMH": REF_COND[p_]["J_LMH"], "median_T_C": REF_COND[p_]["T_C"],
              "median_permeability_LMH_per_bar": float(np.nanmedian(df.loc[m_, "permeability"])) * 100.0,
              "R_m_share_%": 100 * rm / tot, "R_rev_share_%": 100 * rr / tot, "R_irrev_share_%": 100 * ri / tot,
              "hydraulic_reversibility_%": 100 * rr / max(rr + ri, 1e-30),
              "backwashes_per_day": float(df.loc[m_, "is_backwash"].sum()) / ndays,
              "irrev_fouling_rate_kPa_per_day": float(r_to_kpa(rate_r, [p_])[0]),
              "irrev_fouling_rate_m-1_per_day": rate_r,
              "degradation_step_kPa": float(r_to_kpa(DELTA_FRAC_PRIMARY * DELTA[p_], [p_])[0])}
FP_DF = pd.DataFrame(FP).T; FP_DF.index.name = "Period"
print("=== Operating fingerprint per period ===\n" + FP_DF.round(3).to_string())

# ── 8.2 14-day irreversible TMP rise in kPa, all windows, day-block CIs ──
PER_ALL = np.concatenate(CV_PERIOD)
END = {k: np.concatenate(v) for k, v in CV_END.items()}
assert len(END["true"]) == len(DAY_POOL), "trajectory/day alignment failed"
KPA = {k: r_to_kpa(v, PER_ALL) for k, v in END.items()}
KPA["Persistence"] = np.zeros_like(KPA["true"])
_KM = ("PC-Net-HMCP", "Physics seed only", "Linear 7d", "Power law", "Persistence")
_kerr = {nm: np.abs(KPA[nm] - KPA["true"]) for nm in _KM}
K_PT, K_BOOT, K_DAYS, K_N = block_bootstrap(_kerr, np.ones(len(DAY_POOL), bool))
KPA_DF = pd.DataFrame({nm: {"MAE_kPa": K_PT[nm], "CI_low": float(np.percentile(K_BOOT[nm], 2.5)),
                            "CI_high": float(np.percentile(K_BOOT[nm], 97.5)),
                            "bias_kPa": float(np.mean(KPA[nm] - KPA["true"]))} for nm in _KM}).T.sort_values("MAE_kPa")
KPA_DF["mean_observed_rise_kPa"] = float(np.mean(KPA["true"])); KPA_DF.index.name = "Model"
P_KPA = {f"PC-Net vs {r}": p_two_sided(K_BOOT, "PC-Net-HMCP", r) for r in ("Linear 7d", "Power law", "Persistence", "Physics seed only")}
P_KPA_HOLM = holm(P_KPA)
print(f"\n=== 14-day irreversible TMP rise (kPa), all {K_N:,} windows, {K_DAYS} days ===\n" + KPA_DF.round(4).to_string())
print("p (raw / Holm):", {k: f"{v:.4f} / {P_KPA_HOLM[k]:.4f}" for k, v in P_KPA.items()})

# ── 8.3 temperature dependence of irreversible fouling (descriptive) ──
WK = []
for p_, dd in DAILY.groupby("period"):
    dd = dd.reset_index(drop=True)
    for s0 in range(0, len(dd) - 6, 7):
        blk = dd.iloc[s0:s0 + 7]
        rate = float(blk["R_irrev"].iloc[-1] - blk["R_irrev"].iloc[0]) / 6.0
        WK.append({"period": int(p_), "T_C": float(blk["temp"].mean()), "rate_kPa_per_day": float(r_to_kpa(rate, [int(p_)])[0])})
WK_DF = pd.DataFrame(WK)
if len(WK_DF) > 5:
    _sp = stats.spearmanr(WK_DF["T_C"], WK_DF["rate_kPa_per_day"]); _rho, _prho = float(_sp[0]), float(_sp[1])
else:
    _rho, _prho = float("nan"), float("nan")
TEMP_DEP = {"spearman_rho": _rho, "p_value": _prho, "n_weeks": int(len(WK_DF)),
            "T_range_C": [float(WK_DF["T_C"].min()), float(WK_DF["T_C"].max())] if len(WK_DF) else [float("nan")] * 2}
print(f"\nTemperature vs weekly irreversible fouling rate: rho = {_rho:.3f}, p = {_prho:.3g}, n = {len(WK_DF)} weeks "
      f"({'significant' if _prho < 0.05 else 'no significant association'} at 0.05)")
cache_put("engineering", {"fingerprint": FP_DF.to_dict(orient="index"), "kpa": KPA_DF.to_dict(orient="index"),
                          "p_kpa": P_KPA, "p_kpa_holm": P_KPA_HOLM, "temp": TEMP_DEP})

# %% [markdown]
# ## 9. Decision stability: can the signal be scheduled against?
#
# For scheduling, the day-to-day behaviour of the recommendation matters as much as its average error. Three
# operational metrics are computed on consecutive daily origins within each test block:
#
# * **Usable-day fraction**: share of days on which a unique cleaning date exists at all.
# * **Countdown consistency**: median $|\\Delta\\mathrm{RUL}+1|$ between consecutive days; a perfect countdown decrements by one.
# * **Schedule reversals per week**: how often the recommended date moves later by more than two days from one day to the next.
#
# Windows at a fixed daily cadence are selected from the stride-3 test set (one origin every 96 windows).

# %%
"""Section 7: decision-stability metrics on daily origins."""
STEP_DAY = STEPS_PER_DAY // STRIDE


def stability(pred_daily, valid_daily, h_days, cens_daily=None):
    """Operational behaviour of the daily recommendation.

    usable_day_fraction  share of days on which a unique cleaning date exists.
    countdown_consistency median |Delta RUL + 1| over consecutive ACTIVE days.
    reversals_per_week    how often the recommended date moves later by more than two days, per seven active days.

    A day is ACTIVE when the method's own recommendation is below the horizon, i.e. it is actually naming a
    date. Days where a method returns the horizon carry no scheduling information and are excluded; requiring
    the ground truth to be uncensored as well (previous version) left only 11 to 50 pairs, too few to compare.
    """
    p = np.where(valid_daily, pred_daily, np.nan)
    usable = float(np.mean(valid_daily))
    act = np.isfinite(p) & (p < h_days - 1e-9)
    pair = act[:-1] & act[1:]
    d = np.diff(p)
    consistency = float(np.nanmedian(np.abs(d[pair] + 1.0))) if pair.any() else float("nan")
    reversals = float(np.sum(d[pair] > 2.0) / max(pair.sum(), 1) * 7.0) if pair.any() else float("nan")
    return {"usable_day_fraction": usable, "countdown_consistency_days": consistency, "reversals_per_week": reversals,
            "n_days": int(len(p)), "n_active_pairs": int(pair.sum())}


STAB = {}
for nm in (*PROG_MODELS, "Linear 7d", "Power law", "XGBoost RUL"):
    rows = []
    for i, fd in enumerate(FOLD_DATA):
        pr = CV_P[nm][i]; vd = CV_VALID[nm][i] if nm in CV_VALID else np.ones(len(pr), bool)
        idx = np.arange(0, len(pr), STEP_DAY)
        if len(idx) >= 7: rows.append(stability(pr[idx], vd[idx], fd.h_days, fd.te[7][idx]))
    if rows:
        df_s = pd.DataFrame(rows)
        STAB[nm] = {"usable_day_fraction": float(df_s["usable_day_fraction"].mean()),
                    "countdown_consistency_days": float(np.nanmean(df_s["countdown_consistency_days"])),
                    "reversals_per_week": float(np.nanmean(df_s["reversals_per_week"])),
                    "n_days": int(df_s["n_days"].sum()), "n_active_pairs": int(df_s["n_active_pairs"].sum())}
STAB_DF = pd.DataFrame(STAB).T
print("=== DECISION STABILITY on daily origins ===\n" + STAB_DF.round(3).to_string())
cache_put(f"stability_{ARCH}", STAB)

# example countdown stretch on the final fold, for Figure 3
fd = FINAL["fd"]; _i0 = 0; _n = min(35, len(CV_P["PC-Net-HMCP"][-1]) // STEP_DAY)
IDX_DAILY = np.arange(_i0, _i0 + _n * STEP_DAY, STEP_DAY)
COUNTDOWN = {"day": np.arange(_n), "true": fd.te[6][IDX_DAILY], "cens": fd.te[7][IDX_DAILY]}
for nm in ("PC-Net-HMCP", "Soft-mono deep", "Unconstrained deep", "Linear 7d", "Power law"):
    pr = CV_P[nm][-1][IDX_DAILY]; vd = CV_VALID[nm][-1][IDX_DAILY] if nm in CV_VALID else np.ones(_n, bool)
    COUNTDOWN[nm] = np.where(vd, pr, np.nan)

# %% [markdown]
# ## 9.1 Ablation: is it the head?

# %%
"""Section 6.3: four configurations on the final fold, single seed, cached."""
ABL_EPOCHS = 6 if FAST_MODE else 20
trn_ld, val_ld, tst_ld = FINAL["loaders"]
_mix = FINAL["cfg"].mix_init
CONFIGS = {"Full (HMCP)": Config(encoder=ENCODERS_MAIN, h_days=fd.h_days, mix_init=_mix),
           "Soft monotonicity penalty": Config(encoder=ENCODERS_MAIN, h_days=fd.h_days, hard_mono=False, lam_mono=0.2, mix_init=_mix),
           "No monotonicity": Config(encoder=ENCODERS_MAIN, h_days=fd.h_days, hard_mono=False, lam_mono=0.0, mix_init=_mix),
           "Encoder: transformer": Config(encoder="transformer", h_days=fd.h_days, mix_init=_mix)}
ABL = {}
for nm, c in CONFIGS.items():
    _key = f"abl_{re.sub('[^a-z0-9]+', '_', nm.lower()).strip('_')}_{arch_fingerprint(c)}_stab2"
    if (_c := cache_get(_key)): ABL[nm] = _c; print(f"  cached {nm}"); continue
    seed_everything(SEED)
    _, cp, _ca = train_model(PCNet(c), c, trn_ld, val_ld, ABL_EPOCHS, tag=f"abl_{re.sub('[^a-z0-9]+', '_', nm.lower()).strip('_')}", verbose=False)
    m = PCNet(c); m.load_state_dict(torch.load(cp, map_location=DEVICE, weights_only=True)); m.to(DEVICE)
    ot = predict(m, tst_ld); pr, vd = rul_from_output(ot, c.h_days)
    met = prog_metrics(ot["rul"], pr, vd, ot["cens"], c.h_days); met["params"] = count_params(m)
    idx = np.arange(0, len(pr), STEP_DAY); met.update({f"stab_{k}": v for k, v in stability(pr[idx], vd[idx], c.h_days, fd.te[7][idx]).items()})
    ABL[nm] = met; cache_put(_key, met)
    print(f"  {nm:26s} MAE {met['MAE_days']:.3f} d | validity {met['validity_rate']:.3f} | usable days {met['stab_usable_day_fraction']:.3f} | reversals/wk {met['stab_reversals_per_week']:.2f}")
    del m; torch.cuda.empty_cache()
ABL_DF = pd.DataFrame(ABL).T

# %% [markdown]
# ## 10. Publication figures
# Okabe-Ito palette, serif type, journal column widths, 300 dpi PNG plus vector PDF and SVG. Blue is the proposed
# model, grey a learned baseline, orange and purple the classical references, green a satisfied constraint, red a violated one.

# %%
"""Section 8: figures."""
h_d = FOLD_DATA[0].h_days
P_ORDER = sorted((k for k in RES_P if np.isfinite(RES_P[k]["MAE_days"])), key=lambda k: RES_P[k]["MAE_days"])
P_COL = {n: (SEM["ours"] if n == "PC-Net-HMCP" else SEM["physics"] if "deep" in n else SEM["reference"] if n == "Linear 7d" else SEM["accent"] if n == "Power law" else SEM["baseline"]) for n in RES_P}
NICE = {"PC-Net-HMCP": "PC-Net (ours)", "Soft-mono deep": "Soft-penalty net", "Unconstrained deep": "Unconstrained net"}
nice = lambda n: NICE.get(n, n)

# ══ Figure 0 | Graphical abstract ══
fig = plt.figure(figsize=(W2, 4.6)); gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.65, wspace=0.36)
axA = fig.add_subplot(gs[0, :2]); _sg = df.iloc[: min(STEPS_PER_DAY * 45, len(df))]
axA.fill_between(_sg["time_days"], 0, _sg["R_rev"], color=SEM["ours"], alpha=0.40, lw=0, label=r"$R_{\rm rev}$ (backwash removes)")
axA.fill_between(_sg["time_days"], _sg["R_rev"], _sg["R_rev"] + _sg["R_irrev"], color=SEM["physics"], alpha=0.45, lw=0, label=r"$R_{\rm irrev}$ (needs CIP)")
axA.set_ylabel(r"Resistance (m$^{-1}$)"); axA.set_xlabel("Time (days)"); axA.legend(loc="upper left", ncol=2)
panel(axA, "a", "Physics-constrained decomposition of measured fouling")
axB = fig.add_subplot(gs[0, 2]); _vn = list(CV_VALID); _vv = [RES_P[n]["validity_rate"] * 100 for n in _vn]
barh_labeled(axB, [nice(n) for n in _vn], _vv, colors=[SEM["good"] if v > 99 else SEM["bad"] for v in _vv], fmt="{:.0f}%")
axB.set_xlim(0, 118); axB.set_xlabel("Usable cleaning dates (%)"); panel(axB, "b", "Can a date be produced?")
axC = fig.add_subplot(gs[1, 0])
barh_labeled(axC, [nice(n) for n in P_ORDER], [RES_P[n]["MAE_days"] for n in P_ORDER], colors=[P_COL[n] for n in P_ORDER], fmt="{:.2f}")
axC.set_xlabel("Date error (days)"); panel(axC, "c", "Accuracy vs references")
axD = fig.add_subplot(gs[1, 1]); cd = COUNTDOWN
axD.plot(cd["day"], cd["PC-Net-HMCP"], "o-", color=SEM["ours"], ms=2.8, lw=1.3, label="PC-Net (ours)")
axD.plot(cd["day"], cd["Unconstrained deep"], "x", color=SEM["bad"], ms=3.5, label="Unconstrained (valid days only)")
axD.step(cd["day"], np.where(cd["cens"] < 0.5, cd["true"], np.nan), where="mid", color=C["black"], lw=1.0, ls="--", label="True")
axD.set_xlabel("Day"); axD.set_ylabel("Days to next step"); axD.set_ylim(0, h_d + 1); axD.legend(fontsize=6, loc="upper right")
panel(axD, "d", "A countdown you can schedule against")
axE = fig.add_subplot(gs[1, 2]); _sn = [n for n in ("PC-Net-HMCP", "Linear 7d", "Power law", "Unconstrained deep") if n in STAB]
barh_labeled(axE, [nice(n) for n in _sn], [STAB[n]["usable_day_fraction"] * 100 for n in _sn], colors=[P_COL[n] for n in _sn], fmt="{:.0f}%")
axE.set_xlim(0, 118); axE.set_xlabel("Days with a usable recommendation (%)"); panel(axE, "e", "Operational availability")
finish("Fig0_graphical_abstract", fig, tight=False)

# ══ Figure 1 | Measured dynamics and the target ══
seg = df.iloc[: min(STEPS_PER_DAY * 60, len(df))]
fig, ax = plt.subplots(3, 1, figsize=(W15, 5.4), sharex=True)
ax[0].plot(seg["time_days"], seg["TMP_kPa"], color=C["black"], lw=0.45); shade_periods(ax[0], seg); ax[0].set_ylabel("TMP (kPa)")
panel(ax[0], "a", r"Crossflow TMP $=\frac{1}{2}(P_{\rm feed}+P_{\rm conc})-P_{\rm perm}$, filtration samples only")
ax[1].fill_between(seg["time_days"], 0, seg["R_rev"], color=SEM["ours"], alpha=0.40, lw=0, label=r"$R_{\rm rev}$")
ax[1].fill_between(seg["time_days"], seg["R_rev"], seg["R_rev"] + seg["R_irrev"], color=SEM["physics"], alpha=0.45, lw=0, label=r"$R_{\rm irrev}$")
ax[1].set_ylabel(r"Resistance (m$^{-1}$)"); ax[1].legend(loc="upper left", ncol=2); panel(ax[1], "b", "Isotonic decomposition, per operating period")
_d0 = DAILY[DAILY["day"] <= int(seg["day"].max())]; _rt, _ct = RUL_TAB[(DELTA_FRAC_PRIMARY, h_d)]
ax[2].step(_d0["day"], _rt[: len(_d0)], where="post", color=SEM["accent"], lw=1.1)
ax[2].fill_between(_d0["day"], 0, h_d, where=_ct[: len(_d0)], color=C["light"], alpha=0.45, step="post", lw=0, label="censored")
ax[2].set_ylabel(r"True RUL$_\delta$ (days)"); ax[2].set_xlabel("Time (days)"); ax[2].legend(loc="upper right")
panel(ax[2], "c", rf"Ground truth: days to the next {int(100*DELTA_FRAC_PRIMARY)} % degradation step, censored at {h_d} d")
finish("Fig1_dynamics_and_target", fig)

# ══ Figure 2 | Envelope validation ══
fig, ax = plt.subplots(1, 2, figsize=(W15, 2.7))
for (lab, y_, col) in (("daily-q10 (primary)", df["R_irrev"], SEM["ours"]), ("post-backwash median", R_irr_pbw, SEM["reference"]), ("single gauge (naive)", R_irr_feed, SEM["baseline"])):
    ax[0].plot(df["time_days"], y_, lw=0.8, color=col, label=lab)
shade_periods(ax[0], df); ax[0].set_xlabel("Time (days)"); ax[0].set_ylabel(r"$R_{\rm irrev}$ (m$^{-1}$)"); ax[0].legend(fontsize=6.5)
panel(ax[0], "a", "Three ways to extract the irreversible envelope")
sns.heatmap(ENV_VALID, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1, ax=ax[1], cbar=False, annot_kws={"size": 8}); ax[1].tick_params(axis="x", rotation=25)
panel(ax[1], "b", "Agreement between methods"); finish("Fig2_envelope_validation", fig)

# ══ Figure 3 | Validity, accuracy, and the countdown ══
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.6))
_vn = list(CV_VALID); _vv = [RES_P[n]["validity_rate"] * 100 for n in _vn]
barh_labeled(ax[0], [nice(n) for n in _vn], _vv, colors=[SEM["good"] if v > 99 else SEM["bad"] for v in _vv], fmt="{:.1f}%")
ax[0].set_xlim(0, 118); ax[0].set_xlabel("Windows with a unique cleaning date (%)"); panel(ax[0], "a", "Validity")
_err = np.array([[RES_P[n]["MAE_days"] - float(RES_P[n]["MAE CI"].strip("[]").split(",")[0]) for n in P_ORDER],
                 [float(RES_P[n]["MAE CI"].strip("[]").split(",")[1]) - RES_P[n]["MAE_days"] for n in P_ORDER]])
barh_labeled(ax[1], [nice(n) for n in P_ORDER], [RES_P[n]["MAE_days"] for n in P_ORDER], colors=[P_COL[n] for n in P_ORDER], fmt="{:.2f}", xerr=np.abs(_err))
ax[1].set_xlabel("RUL MAE (days), 95% CI"); panel(ax[1], "b", "Accuracy")
cd = COUNTDOWN
ax[2].step(cd["day"], np.where(cd["cens"] < 0.5, cd["true"], np.nan), where="mid", color=C["black"], lw=1.0, ls="--", label="True")
ax[2].plot(cd["day"], cd["PC-Net-HMCP"], "o-", color=SEM["ours"], ms=2.8, lw=1.3, label="PC-Net (ours)")
ax[2].plot(cd["day"], cd["Linear 7d"], "s-", color=SEM["reference"], ms=2.4, lw=0.9, alpha=0.9, label="Linear 7 d")
ax[2].plot(cd["day"], cd["Unconstrained deep"], "x", color=SEM["bad"], ms=3.5, label="Unconstrained (valid days only)")
ax[2].set_xlabel("Day"); ax[2].set_ylabel("Days to next step"); ax[2].set_ylim(0, h_d + 1); ax[2].legend(fontsize=6, loc="upper right")
panel(ax[2], "c", "Daily recommendation, final fold"); finish("Fig3_validity_accuracy_countdown", fig)

# ══ Figure 4 | Decision stability ══
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.9)); _sn = list(STAB)
plt.subplots_adjust(wspace=0.55)
barh_labeled(ax[0], [nice(n) for n in _sn], [STAB[n]["usable_day_fraction"] * 100 for n in _sn], colors=[P_COL[n] for n in _sn], fmt="{:.0f}%")
ax[0].set_xlim(0, 118); ax[0].set_xlabel("Days with a usable\nrecommendation (%)", fontsize=7); panel(ax[0], "a", "Availability")
_ok = [n for n in _sn if np.isfinite(STAB[n]["countdown_consistency_days"])]
barh_labeled(ax[1], [f"{nice(n)}\n(n={STAB[n].get('n_active_pairs', 0)})" for n in _ok], [STAB[n]["countdown_consistency_days"] for n in _ok], colors=[P_COL[n] for n in _ok], fmt="{:.2f}")
ax[1].set_xlabel("Countdown consistency (days)\nactive days only, lower is better", fontsize=7); panel(ax[1], "b", "Countdown consistency")
_ok = [n for n in _sn if np.isfinite(STAB[n]["reversals_per_week"])]
barh_labeled(ax[2], [f"{nice(n)}\n(n={STAB[n].get('n_active_pairs', 0)})" for n in _ok], [STAB[n]["reversals_per_week"] for n in _ok], colors=[P_COL[n] for n in _ok], fmt="{:.3f}")
ax[2].set_xlabel("Schedule reversals per week\nlower is better", fontsize=7); panel(ax[2], "c", "Reversals")
finish("Fig4_decision_stability", fig, tight=False)

# ══ Figure 4b | Learned correction ══
fig, ax = plt.subplots(1, 2, figsize=(W15, 2.7))
_f = LEARN_DF.index.values; _x = np.arange(len(_f)); _bw = 0.26
for j, (col_, lab_, c_) in enumerate((("MAE_seed_only", "Physics seed (untrained)", SEM["reference"]),
                                       ("MAE_trained_only", "Trained head", SEM["baseline"]),
                                       ("MAE_deployed", "Deployed (guard)", SEM["ours"]))):
    ax[0].bar(_x + (j - 1) * _bw, LEARN_DF[col_], _bw, color=c_, edgecolor=C["black"], linewidth=0.5, label=lab_)
for xi, d_ in enumerate(LEARN_DF["deployed"]):
    ax[0].text(xi, np.nanmax(LEARN_DF.loc[_f[xi], ["MAE_seed_only", "MAE_trained_only", "MAE_deployed"]].values.astype(float)) * 1.04,
               "T" if d_ == "trained" else "S", ha="center", fontsize=7, fontweight="bold", color=SEM["ours"])
ax[0].set_xticks(_x); ax[0].set_xticklabels([f"Fold {k}" for k in _f]); ax[0].set_ylabel("RUL MAE (days)"); ax[0].legend(fontsize=6.3, loc="upper left"); ax[0].margins(y=0.2)
panel(ax[0], "a", "Per fold (T = trained deployed, S = seed deployed)")
_lp = LEARN_POOL_DF.copy(); _lp.loc["Linear 7d"] = [RES_P["Linear 7d"]["MAE_days"], RES_P["Linear 7d"]["alpha_lambda_20"], 1.0]
_lp.loc["Power law"] = [RES_P["Power law"]["MAE_days"], RES_P["Power law"]["alpha_lambda_20"], 1.0]
_lp = _lp.sort_values("MAE_days", ascending=False)
_cols = [SEM["ours"] if "Deployed" in i else SEM["reference"] if ("seed" in i or "Linear" in i) else SEM["accent"] if "Power" in i else SEM["baseline"] for i in _lp.index]
barh_labeled(ax[1], [i.replace(" (untrained HMCP)", "\n(untrained HMCP)").replace(" (no guard)", "\n(no guard)").replace(" (validation guard)", "\n(validation guard)") for i in _lp.index],
             _lp["MAE_days"], colors=_cols, fmt="{:.2f}")
ax[1].set_xlabel("Pooled RUL MAE (days)"); panel(ax[1], "b", "Where the accuracy comes from")
finish("Fig4b_learned_correction", fig)

# ══ Figure 5 | Per-period generalisation ══
fig, ax = plt.subplots(1, 2, figsize=(W15, 2.6)); _pp = RES_PERIOD_DF.index.astype(int).tolist(); _x = np.arange(len(_pp)); _bw = 0.26
for j, (col_, lab_, c_) in enumerate((("HMCP_MAE", "PC-Net (ours)", SEM["ours"]), ("Linear 7d_MAE", "Linear 7 d", SEM["reference"]), ("Power law_MAE", "Power law", SEM["accent"]))):
    ax[0].bar(_x + (j - 1) * _bw, RES_PERIOD_DF[col_], _bw, color=c_, edgecolor=C["black"], linewidth=0.5, label=lab_)
    for xi, v in enumerate(RES_PERIOD_DF[col_]):
        if np.isfinite(v): ax[0].text(xi + (j - 1) * _bw, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
ax[0].set_xticks(_x); ax[0].set_xticklabels([f"Period {p}\n{int(RES_PERIOD_DF.loc[p, 'days_in_period'])} d" for p in _pp]); ax[0].set_ylabel("RUL MAE (days)"); ax[0].legend(fontsize=6.5); ax[0].margins(y=0.25)
panel(ax[0], "a", "Accuracy by operating regime")
ax[1].bar(_x, RES_PERIOD_DF["HMCP_validity"] * 100, 0.5, color=[SEM["good"] if v > 0.99 else SEM["bad"] for v in RES_PERIOD_DF["HMCP_validity"]], edgecolor=C["black"], linewidth=0.5)
for xi, v in enumerate(RES_PERIOD_DF["HMCP_validity"] * 100): ax[1].text(xi, v + 1.5, f"{v:.0f}%", ha="center", fontsize=7)
ax[1].set_xticks(_x); ax[1].set_xticklabels([f"Period {p}" for p in _pp]); ax[1].set_ylim(0, 108); ax[1].set_ylabel("Valid dates (%)")
panel(ax[1], "b", "Validity by regime"); finish("Fig5_regime_generalisation", fig)

# ══ Figure 6 | Ablation ══
_ord = ABL_DF.sort_values("MAE_days", ascending=False).index
_col = [SEM["ours"] if o.startswith("Full") else SEM["baseline"] for o in _ord]
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.5))
barh_labeled(ax[0], _ord, ABL_DF.loc[_ord, "validity_rate"] * 100, colors=[SEM["good"] if v > 0.99 else SEM["bad"] for v in ABL_DF.loc[_ord, "validity_rate"]], fmt="{:.0f}%")
ax[0].set_xlim(0, 118); ax[0].set_xlabel("Valid dates (%)"); panel(ax[0], "a", "Validity is the head's doing")
barh_labeled(ax[1], _ord, ABL_DF.loc[_ord, "MAE_days"], colors=_col, fmt="{:.2f}"); ax[1].set_xlabel("RUL MAE (days)"); ax[1].set_yticklabels([]); panel(ax[1], "b", "Accuracy")
barh_labeled(ax[2], _ord, ABL_DF.loc[_ord, "stab_usable_day_fraction"] * 100, colors=[SEM["good"] if v > 0.99 else SEM["bad"] for v in ABL_DF.loc[_ord, "stab_usable_day_fraction"]], fmt="{:.0f}%")
ax[2].set_xlim(0, 118); ax[2].set_xlabel("Usable days (%)"); ax[2].set_yticklabels([]); panel(ax[2], "c", "Availability")
finish("Fig6_ablation", fig)

# ══ Figure 7 | Operating fingerprint (process engineering) ══
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.9))
plt.subplots_adjust(wspace=0.5)
_pp = FP_DF.index.astype(int).tolist(); _y = np.arange(len(_pp)); _left = np.zeros(len(_pp))
for col_, lab_, c_ in (("R_m_share_%", r"$R_m$ (membrane)", C["light"]), ("R_rev_share_%", r"$R_{\rm rev}$ (backwash removes)", SEM["ours"]),
                       ("R_irrev_share_%", r"$R_{\rm irrev}$ (needs CIP)", SEM["physics"])):
    v = FP_DF[col_].values.astype(float)
    ax[0].barh(_y, v, left=_left, color=c_, edgecolor=C["black"], linewidth=0.5, label=lab_)
    for yi, (l_, w_) in enumerate(zip(_left, v)):
        if w_ >= 8: ax[0].text(l_ + w_ / 2, yi, f"{w_:.0f}%", ha="center", va="center", fontsize=6.5)
    _left += v
ax[0].set_yticks(_y); ax[0].set_yticklabels([f"Period {p}" for p in _pp]); ax[0].set_xlim(0, 100); ax[0].set_xlabel("Share of total resistance (%)")
ax[0].legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False, handlelength=1.0, columnspacing=0.8)
panel(ax[0], "a", "Resistance-in-series budget")
_r = FP_DF["irrev_fouling_rate_kPa_per_day"].values.astype(float)
b = ax[1].bar(_y, _r, 0.55, color=SEM["physics"], edgecolor=C["black"], linewidth=0.5)
for xi, (v, bw, rv) in enumerate(zip(_r, FP_DF["backwashes_per_day"], FP_DF["hydraulic_reversibility_%"])):
    ax[1].text(xi, v, f"{v:.3f}\n{bw:.0f} BW/d\n{rv:.0f}% rev.", ha="center", va="bottom", fontsize=6)
ax[1].set_xticks(_y); ax[1].set_xticklabels([f"P{p}" for p in _pp]); ax[1].set_ylabel("Irreversible rate (kPa/day)"); ax[1].margins(y=0.45)
panel(ax[1], "b", "Fouling rate, TMP-equivalent")
if len(WK_DF):
    for p_, c_ in zip(sorted(WK_DF["period"].unique()), PAL):
        s_ = WK_DF[WK_DF["period"] == p_]
        ax[2].scatter(s_["T_C"], s_["rate_kPa_per_day"], s=14, color=c_, edgecolor=C["black"], linewidth=0.3, label=f"P{p_}", alpha=0.85)
    ax[2].legend(fontsize=6, loc="lower right", framealpha=0.9)
ax[2].set_xlabel("Weekly mean temperature (°C)"); ax[2].set_ylabel("Weekly rate (kPa/day)")
ax[2].text(0.03, 0.95, rf"Spearman $\rho$ = {TEMP_DEP['spearman_rho']:.2f}" + f"\np = {TEMP_DEP['p_value']:.2g}, n = {TEMP_DEP['n_weeks']}",
           transform=ax[2].transAxes, va="top", fontsize=6.5, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C["light"], lw=0.6, alpha=0.9), zorder=5)
panel(ax[2], "c", "Temperature dependence")
finish("Fig7_operating_fingerprint", fig, tight=False)

# ══ Figure 8 | The forecast in kPa ══
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.6))
_ko = KPA_DF.sort_values("MAE_kPa", ascending=False)
_kerrbar = np.vstack([_ko["MAE_kPa"] - _ko["CI_low"], _ko["CI_high"] - _ko["MAE_kPa"]]).astype(float)
_kcol = [SEM["ours"] if i == "PC-Net-HMCP" else SEM["reference"] if i == "Linear 7d" else SEM["accent"] if i == "Power law" else SEM["neutral"] if "seed" in i else SEM["baseline"] for i in _ko.index]
barh_labeled(ax[0], [nice(i) for i in _ko.index], _ko["MAE_kPa"].values, colors=_kcol, fmt="{:.3f}", xerr=np.abs(_kerrbar))
ax[0].set_xlabel("MAE of 14-day irreversible TMP rise (kPa)"); panel(ax[0], "a", "Trajectory accuracy, all windows")
fd_f = FINAL["fd"]; ot_f = FINAL["ot"]; per_f = CV_PERIOD[-1]; _uf = np.flatnonzero(fd_f.te[7] < 0.5)
_i = int(_uf[len(_uf) // 2]) if len(_uf) else 0; _p = [per_f[_i]]; k = np.arange(1, fd_f.h_days + 1)
ax[1].plot(k, r_to_kpa(ot_f["dp"][0][_i] * fd_f.S, _p * fd_f.h_days), "o-", color=SEM["ours"], ms=2.5, lw=1.4, label="PC-Net (ours)")
ax[1].plot(k, r_to_kpa(np.arange(1, fd_f.h_days + 1) * fd_f.te[9][_i] * fd_f.S, _p * fd_f.h_days), "s--", color=SEM["reference"], ms=2.2, lw=1.0, label="Linear 7 d")
ax[1].plot(k, r_to_kpa(fd_f.te[2][_i] * fd_f.S, _p * fd_f.h_days), color=C["black"], lw=1.0, ls=":", label="Observed")
_step = float(r_to_kpa((fd_f.te[8][_i] - fd_f.te[5][_i]) * fd_f.S, _p)[0])
ax[1].axhline(_step, color=SEM["physics"], ls="--", lw=1.0)
ax[1].text(fd_f.h_days, _step, f"one degradation step = {_step:.2f} kPa", va="bottom", ha="right", fontsize=6.3, color=SEM["physics"])
ax[1].set_xlabel("Days ahead"); ax[1].set_ylabel("Irreversible TMP rise (kPa)"); ax[1].legend(fontsize=6, loc="lower right"); ax[1].margins(y=0.12)
panel(ax[1], "b", "One forecast, in kPa")
_lim = float(np.nanpercentile(np.concatenate([KPA["true"], KPA["PC-Net-HMCP"]]), 99)) * 1.05 + 1e-9
hb = ax[2].hexbin(KPA["true"], KPA["PC-Net-HMCP"], gridsize=28, extent=(0, _lim, 0, _lim), cmap="Blues", mincnt=1, bins="log", linewidths=0)
ax[2].plot([0, _lim], [0, _lim], color=C["black"], lw=0.8, ls="--"); ax[2].set_xlim(0, _lim); ax[2].set_ylim(0, _lim)
cb = fig.colorbar(hb, ax=ax[2], pad=0.02, fraction=0.05); cb.ax.tick_params(labelsize=6); cb.set_label("windows (log scale)", fontsize=6.5)
ax[2].set_xlabel("Observed 14-day rise (kPa)"); ax[2].set_ylabel("PC-Net forecast (kPa)")
panel(ax[2], "c", "Forecast vs observed, all windows")
finish("Fig8_forecast_in_kPa", fig)

# %% [markdown]
# ## 11. Tables and export

# %%
"""Section 9: Word tables, manifest, Drive export."""
from docx import Document
from docx.shared import Pt


def df_to_docx(doc, frame, title, fmt="{:.3f}"):
    doc.add_heading(title, level=2); t = doc.add_table(rows=1, cols=len(frame.columns) + 1); t.style = "Light Grid Accent 1"
    hdr = t.rows[0].cells; hdr[0].text = frame.index.name or ""
    for j, c in enumerate(frame.columns): hdr[j + 1].text = str(c)
    for idx, row in frame.iterrows():
        cells = t.add_row().cells; cells[0].text = str(idx)
        for j, v in enumerate(row): cells[j + 1].text = fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)
    for r in t.rows:
        for c in r.cells:
            for p_ in c.paragraphs:
                for run in p_.runs: run.font.size = Pt(8)
    doc.add_paragraph()


CAPABILITY = pd.DataFrame({
    "PC-Net (ours)": {"Unique cleaning date guaranteed": "yes, by construction", "Monotone R_irrev": "yes (0 % violations)",
                      "Usable every day": f"{100*STAB['PC-Net-HMCP']['usable_day_fraction']:.0f} %", "Countdown consistency (d)": f"{STAB['PC-Net-HMCP']['countdown_consistency_days']:.2f}", "Reversals per week": f"{STAB['PC-Net-HMCP']['reversals_per_week']:.3f}",
                      "Multi-step physical trajectory": f"yes ({h_d} daily steps)", "Learns from data": "yes"},
    "Soft / unconstrained net": {"Unique cleaning date guaranteed": "no", "Monotone R_irrev": "no (~50 % violations)",
                                 "Usable every day": f"{100*STAB['Unconstrained deep']['usable_day_fraction']:.0f} %", "Countdown consistency (d)": "undefined",
                                 "Multi-step physical trajectory": "may be non-physical", "Learns from data": "yes", "Reversals per week": "undefined"},
    "Classical extrapolation": {"Unique cleaning date guaranteed": "yes", "Monotone R_irrev": "yes", "Usable every day": f"{100*STAB['Linear 7d']['usable_day_fraction']:.0f} %",
                                "Countdown consistency (d)": f"{STAB['Linear 7d']['countdown_consistency_days']:.2f}", "Multi-step physical trajectory": "yes", "Learns from data": "no",
                                "Reversals per week": f"{STAB['Linear 7d']['reversals_per_week']:.3f}"}})
CAPABILITY.index.name = "Property"
doc = Document(); doc.add_heading("Hard-Monotone Cleaning-Date Forecast for MBRs: Results", 1)
doc.add_paragraph(f"Ghent BIOMATH sidestream crossflow MBR, periods {PERIODS}, {df['time_days'].iloc[-1]:.0f} days. TMP = (PS2+PS3)/2 − PS4, A = {A_MEMBRANE} m². "
                  f"Target: days to the next {int(100*DELTA_FRAC_PRIMARY)} % degradation step, horizon {h_d} d. Encoder {ENCODERS_MAIN}, {len(FEATS)} features, "
                  f"{SEQ_LEN*DT_MIN/60:.0f} h context. {len(FOLD_DATA)}-fold forward-chaining CV, {N_SEEDS}-seed ensemble. "
                  f"{len(TRUE):,} test windows, {100*UNCENS_POOL:.1f} % uncensored. All selection on validation folds only.")
t1 = pd.DataFrame(RES_P).T[["validity_rate", "MAE_days", "MAE CI", "RMSE_days", "Bias_days", "alpha_lambda_20", "alpha_lambda_10", "fold_mean_MAE", "fold_sd_MAE"]]; t1.index.name = "Model"
df_to_docx(doc, t1, "Table 1. Validity and accuracy of the cleaning-date signal, pooled CV")
doc.add_paragraph(f"Confidence intervals are day-block bootstrap ({N_DAYS} independent days, {B_BOOT} resamples).")
_sig = SIG_TABLE.copy(); df_to_docx(doc, _sig, "Table 1b. Day-level significance with Holm correction", "{:.4f}")
LEARN_POOL_DF.index.name = "HMCP variant"; df_to_docx(doc, LEARN_POOL_DF, "Table 2. Where the accuracy comes from: untrained seed, trained head, deployed choice (pooled)")
doc.add_paragraph("Day-level p-values for this decomposition are in Table 1b (decomposition family).")
_fp = FP_DF.copy(); df_to_docx(doc, _fp, "Table 9. Operating fingerprint per period (process-engineering units)")
_kp = KPA_DF.copy(); df_to_docx(doc, _kp, "Table 10. Accuracy of the 14-day irreversible TMP rise forecast, kPa, all windows", "{:.4f}")
doc.add_paragraph("Day-level p-values (raw / Holm): " + "; ".join(f"{k} = {v:.4f} / {P_KPA_HOLM[k]:.4f}" for k, v in P_KPA.items())
                  + f". Temperature vs weekly irreversible rate: Spearman rho = {TEMP_DEP['spearman_rho']:.3f}, p = {TEMP_DEP['p_value']:.3g}, n = {TEMP_DEP['n_weeks']} weeks.")
_ld = LEARN_DF.copy(); _ld.index.name = "Fold"; df_to_docx(doc, _ld, "Table 2b. Learned correction per fold")
t2 = STAB_DF.copy(); t2.index.name = "Model"; df_to_docx(doc, t2, "Table 2c. Decision stability on daily origins (active days: recommendation below the horizon)")
df_to_docx(doc, CAPABILITY, "Table 3. Capability comparison", "{}")
t4 = ABL_DF[["validity_rate", "MAE_days", "alpha_lambda_20", "stab_usable_day_fraction", "stab_reversals_per_week", "params"]].copy(); t4.index.name = "Configuration"
df_to_docx(doc, t4, "Table 4. Ablation, final fold")
df_to_docx(doc, RES_PERIOD_DF, "Table 5. Generalisation across operating regimes")
t6 = ENV_VALID.copy(); t6.index.name = "Envelope"; df_to_docx(doc, t6, "Table 6. Irreversible-envelope cross-validation", "{:.4f}")
t7 = PER_FOLD_DF[[c for c in PER_FOLD_DF.columns if "MAE_days" in c or "validity" in c or c in ("n_test", "uncens_test", "seed", "deployed")]]; df_to_docx(doc, t7, "Table 7. Per-fold results")
t8 = pd.DataFrame({"Value": {"Days": round(float(df["time_days"].iloc[-1]), 1), "5-min rows": len(df), "Daily rows": len(DAILY), "Backwashes": int(df["is_backwash"].sum()),
                   "Features": len(FEATS), "Context (h)": SEQ_LEN * DT_MIN / 60, "Horizon (days)": h_d, "delta (% of period range)": int(100 * DELTA_FRAC_PRIMARY),
                   "Encoder": ENCODERS_MAIN, "Seeds": N_SEEDS, "Folds": len(FOLD_DATA), "Params": count_params(PCNet(FINAL["cfg"])),
                   "GPU": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU", "BF16": BF16_OK}}); t8.index.name = "Item"
df_to_docx(doc, t8, "Table 8. Run summary", "{}")
out_docx = OUT / "tables" / "cleaning_date_signal_results.docx"; doc.save(out_docx)
for nm, fr in (("T1_validity_accuracy", t1), ("T2_learned_correction_pooled", LEARN_POOL_DF), ("T2b_learned_correction_perfold", LEARN_DF), ("T2c_stability", t2), ("T3_capability", CAPABILITY), ("T4_ablation", t4), ("T5_periods", RES_PERIOD_DF), ("T6_envelope", t6), ("T7_perfold", t7), ("T1b_significance", SIG_TABLE), ("T1c_block_bootstrap", BLOCK_DF),
               ("T9_operating_fingerprint", FP_DF), ("T10_forecast_kPa", KPA_DF), ("T11_weekly_temperature", WK_DF)): fr.to_csv(OUT / "tables" / f"{nm}.csv")
print(f"Word tables -> {out_docx}")

# ── release DataLoader worker processes before packaging ──
# Each loader keeps persistent worker processes, each holding a copy of the window tensors. Packaging while they
# are alive can exhaust host RAM, and the OOM killer then takes a worker, which PyTorch reports as a crash.
import gc
for _n in [k for k, v in list(globals().items()) if isinstance(v, torch.utils.data.DataLoader)]:
    del globals()[_n]
if isinstance(globals().get("FINAL"), dict): FINAL.pop("loaders", None)
gc.collect(); torch.cuda.empty_cache()

stamp = datetime.now().strftime("%Y%m%d_%H%M%S"); zip_name = f"HMCP_results_{stamp}.zip"
manifest = {"version": "final-engineering-v1", "significance": SIG_TABLE.reset_index().to_dict(orient="records"), "n_independent_days": N_DAYS,
            "operating_fingerprint": FP_DF.to_dict(orient="index"), "forecast_kPa": KPA_DF.to_dict(orient="index"),
            "p_kPa": P_KPA, "p_kPa_holm": P_KPA_HOLM, "temperature_dependence": TEMP_DEP, "exported_at": stamp, "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU", "periods": PERIODS,
            "days": float(df["time_days"].iloc[-1]), "horizon_days": h_d, "delta_frac": DELTA_FRAC_PRIMARY, "n_seeds": N_SEEDS, "folds": len(FOLD_DATA),
            "n_windows": int(len(TRUE)), "uncensored_pooled": UNCENS_POOL, "data_signature": DATA_SIG, "arch": ARCH,
            "validity_accuracy": {k: {kk: v[kk] for kk in ("validity_rate", "MAE_days", "alpha_lambda_20")} for k, v in RES_P.items()}, "p_values": P_P,
            "stability": STAB, "per_period": RES_PERIOD, "learned_correction_pooled": LEARN_POOL, "learned_correction_per_fold": LEARN_DF.to_dict(orient="index"), "learned_correction_p": P_LEARN, "ablation": ABL_DF[["validity_rate", "MAE_days", "stab_usable_day_fraction"]].to_dict(),
            "envelope_validation": ENV_VALID.to_dict(), "seed_choice_per_fold": PER_FOLD_DF["seed"].to_dict(), "deployed_per_fold": (PER_FOLD_DF["deployed"].to_dict() if "deployed" in PER_FOLD_DF else {}), "config": asdict(FINAL["cfg"])}
(OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
# package only what a manuscript needs: tables, figures, manifest (cache and checkpoints stay in Drive)
zip_path = (Path("/content") if Path("/content").exists() else OUT.parent) / zip_name
keep = [OUT / "run_manifest.json", OUT / "stages.json"] + sorted((OUT / "tables").glob("*")) + sorted((OUT / "figures").glob("*"))
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for p in keep:
        if p.is_file(): zf.write(p, arcname=p.relative_to(OUT.parent))
shutil.copy(zip_path, OUT / zip_name)
print(f"Packaged {sum(p.is_file() for p in keep)} files -> {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB); copy saved to {OUT / zip_name}")
try:
    from google.colab import files as colab_files; colab_files.download(str(zip_path)); print("Browser download started.")
except ImportError: print("Not in Colab; archive saved locally.")
stage_mark("all_complete")

print("\n" + "=" * 78 + "\nCOMPLETE\n" + "=" * 78)
print(f"DAY-LEVEL INFERENCE ({N_DAYS} days):"); print(SIG_TABLE.round(4).to_string())
print("\nENGINEERING:"); print(FP_DF[["median_TMP_kPa", "median_permeability_LMH_per_bar", "R_irrev_share_%", "hydraulic_reversibility_%",
                                     "irrev_fouling_rate_kPa_per_day", "degradation_step_kPa"]].round(3).to_string())
print("\n" + KPA_DF.round(4).to_string())
print(f"Temperature: rho = {TEMP_DEP['spearman_rho']:.3f}, p = {TEMP_DEP['p_value']:.3g}, n = {TEMP_DEP['n_weeks']} weeks")
print(f"VALIDITY   PC-Net {100*RES_P['PC-Net-HMCP']['validity_rate']:.1f} % | soft {100*RES_P['Soft-mono deep']['validity_rate']:.1f} % | unconstrained {100*RES_P['Unconstrained deep']['validity_rate']:.1f} %")
for k in P_ORDER: print(f"ACCURACY   {nice(k):22s} MAE {RES_P[k]['MAE_days']:.3f} d {RES_P[k]['MAE CI']}  a-20 {RES_P[k]['alpha_lambda_20']:.3f}")
print("p-values:", {k: f"{v:.3g}" for k, v in P_P.items()})
for k, v in LEARN_POOL.items(): print(f"SOURCE     {k:36s} MAE {v['MAE_days']:.3f} d  a-20 {v['alpha_lambda_20']:.3f}  validity {100*v['validity_rate']:.1f} %")
print("           p:", {k: f"{v:.3g}" for k, v in P_LEARN.items()})
for k, v in STAB.items(): print(f"STABILITY  {nice(k):22s} usable {100*v['usable_day_fraction']:5.1f} %  consistency {v['countdown_consistency_days']:.2f} d  reversals/wk {v['reversals_per_week']:.3f}  (active pairs {v.get('n_active_pairs',0)})")
if "deployed" in PER_FOLD_DF: print("DEPLOYED  ", PER_FOLD_DF["deployed"].to_dict())
print(f"\nEverything is saved under: {OUT}")

# %% [markdown]
# ## 12. Final publication figures
# Redraws every manuscript figure from the results in memory (no retraining). Output: `results/figures_final/` as 600-dpi PNG, PDF and SVG.

# %%
"""Publication figures, redrawn cleanly from the results already in memory.

Run AFTER the Section 8 correction cell (period-pure fouling rates). No retraining.
Every figure uses constrained layout, so titles, labels and legends cannot overlap.
Output: figures_final/ as 600-dpi PNG, vector PDF and SVG.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FIGDIR = OUT / "figures_final"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ── style ────────────────────────────────────────────────────────────────────
INK, GRID = "#1a1a1a", "#e6e6e6"
COL = {"ours": "#0072B2", "linear": "#E69F00", "power": "#CC79A7", "seed": "#56B4E9",
       "learned": "#7F7F7F", "fail": "#D55E00", "ok": "#009E73", "rev": "#0072B2",
       "irr": "#D55E00", "mem": "#BDBDBD", "cens": "#EFEFEF"}
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.linewidth": 0.7, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "figure.constrained_layout.h_pad": 0.04, "figure.constrained_layout.w_pad": 0.04,
})
W1, W15, W2 = 3.54, 5.51, 7.28          # Elsevier single, 1.5 and double column (inches)
NAME = {"PC-Net-HMCP": "PC-Net (ours)", "Soft-mono deep": "Soft-penalty net", "Unconstrained deep": "Unconstrained net",
        "Linear 7d": "Linear 7 d", "Power law": "Power law", "XGBoost RUL": "XGBoost RUL", "Persistence": "Persistence",
        "Physics seed only": "Physics seed", "Trained head (no guard)": "Trained head, no guard"}
MCOL = {"PC-Net-HMCP": COL["ours"], "Linear 7d": COL["linear"], "Power law": COL["power"], "XGBoost RUL": COL["learned"],
        "Persistence": "#A6A6A6", "Physics seed only": COL["seed"], "Soft-mono deep": COL["fail"], "Unconstrained deep": COL["fail"]}


def title(ax, letter, text):
    ax.set_title(rf"$\mathbf{{{letter}}}$   {text}", loc="left", pad=5)


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    plt.show(); plt.close(fig)
    print(f"  {name}")


def hbars(ax, labels, values, colors, fmt="{:.2f}", err=None, xmax=None, note=None):
    """Horizontal bars, value printed beyond the bar (and beyond its error bar), room reserved on the right."""
    values = np.asarray(values, float); y = np.arange(len(labels))
    plot_v = np.nan_to_num(values)
    ax.barh(y, plot_v, color=colors, edgecolor=INK, linewidth=0.5, height=0.66,
            xerr=err, error_kw=dict(elinewidth=0.7, capsize=2.2, ecolor=INK))
    hi = np.zeros(len(values)) if err is None else np.nan_to_num(np.asarray(err)[1])
    top = np.nanmax(plot_v + hi) if len(values) else 1.0
    for yi, v, h in zip(y, values, hi):
        txt = (note or {}).get(yi) or (fmt.format(v) if np.isfinite(v) else "no valid date")
        ax.text((v if np.isfinite(v) else 0) + h + top * 0.045, yi, txt, va="center", ha="left", fontsize=7,
                color=INK if np.isfinite(v) else COL["fail"], style="normal" if np.isfinite(v) else "italic")
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(0, xmax if xmax else top * 1.32)
    ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)


def ci_of(name):
    lo, hi = [float(x) for x in RES_P[name]["MAE CI"].strip("[]").split(",")]
    return lo, hi


def p_label(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ── data prepared once ───────────────────────────────────────────────────────
fd = FINAL["fd"]; H = fd.h_days
_idx = np.arange(0, len(fd.te[6]), STEP_DAY)
_cens_d = fd.te[7][_idx]
_L = int(min(35, len(_idx)))
_s0 = max(range(0, len(_idx) - _L + 1), key=lambda s: int((_cens_d[s:s + _L] < 0.5).sum()))
SEL = _idx[_s0:_s0 + _L]                                   # the stretch with the most days that have a true date
CD_TRUE, CD_CENS = fd.te[6][SEL], fd.te[7][SEL]
CD = {nm: np.where(CV_VALID[nm][-1][SEL] if nm in CV_VALID else True, CV_P[nm][-1][SEL], np.nan)
      for nm in ("PC-Net-HMCP", "Linear 7d", "Power law", "Unconstrained deep")}
PHOLM = SIG_TABLE["p_holm"].to_dict()


def draw_countdown(ax, compact=False):
    d = np.arange(len(SEL))
    for i, c in enumerate(CD_CENS):
        if c >= 0.5: ax.axvspan(i - 0.5, i + 0.5, color=COL["cens"], lw=0, zorder=0)
    ax.step(d, np.where(CD_CENS < 0.5, CD_TRUE, np.nan), where="mid", color=INK, lw=1.2, ls="--", label="Observed", zorder=3)
    ax.plot(d, CD["PC-Net-HMCP"], "o-", color=COL["ours"], ms=2.6, lw=1.3, label="PC-Net (ours)", zorder=4)
    if not compact:
        ax.plot(d, CD["Linear 7d"], "s-", color=COL["linear"], ms=2.2, lw=0.9, alpha=0.9, label="Linear 7 d", zorder=2)
    nv = int(np.isfinite(CD["Unconstrained deep"]).sum())
    ax.set_xlim(-0.5, len(SEL) - 0.5); ax.set_ylim(0, H + 1.5)
    ax.set_xlabel("Day of test stretch"); ax.set_ylabel("Days to next step")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=COL["cens"], edgecolor="none")); labels.append("Censored (no step within 14 d)")
    handles.append(plt.Line2D([], [], marker="x", ls="none", color=COL["fail"], ms=4)); labels.append(f"Unconstrained net: a date on {nv}/{len(SEL)} days")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=6.3, handlelength=1.6)


print("Writing figures to", FIGDIR)

# ════ Figure 1 | Graphical abstract ═════════════════════════════════════════
fig = plt.figure(figsize=(W2, 5.0), layout="constrained")
gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1])
a = fig.add_subplot(gs[0, :2]); seg = df.iloc[: min(STEPS_PER_DAY * 45, len(df))]
a.fill_between(seg["time_days"], 0, seg["R_rev"] / 1e11, color=COL["rev"], alpha=0.45, lw=0, label=r"$R_{\rm rev}$, removed by backwash")
a.fill_between(seg["time_days"], seg["R_rev"] / 1e11, (seg["R_rev"] + seg["R_irrev"]) / 1e11, color=COL["irr"], alpha=0.5, lw=0,
               label=r"$R_{\rm irrev}$, requires chemical cleaning")
a.set_xlabel("Time (days)"); a.set_ylabel(r"Resistance ($\times10^{11}$ m$^{-1}$)"); a.legend(loc="upper right", fontsize=6.8)
title(a, "a", "Fouling decomposed from routine pressure data")
b = fig.add_subplot(gs[0, 2]); vn = ["Unconstrained deep", "Soft-mono deep", "PC-Net-HMCP"]
hbars(b, [NAME[n] for n in vn], [100 * RES_P[n]["validity_rate"] for n in vn],
      [COL["ok"] if RES_P[n]["validity_rate"] > 0.99 else COL["fail"] for n in vn], fmt="{:.0f} %", xmax=135)
b.set_xlabel("Windows with a cleaning date (%)"); title(b, "b", "Does a date exist?")
c = fig.add_subplot(gs[1, 0]); order = ["Persistence", "XGBoost RUL", "Power law", "Linear 7d", "PC-Net-HMCP"]
hbars(c, [NAME[n] for n in order], [RES_P[n]["MAE_days"] for n in order], [MCOL[n] for n in order])
c.set_xlabel("Cleaning-date error (days)"); title(c, "c", "Accuracy")
d = fig.add_subplot(gs[1, 1]); draw_countdown(d, compact=True); title(d, "d", "Daily countdown")
e = fig.add_subplot(gs[1, 2]); pp = FP_DF.index.astype(int).tolist(); yy = np.arange(len(pp)); left = np.zeros(len(pp))
for col_, lab_, c_ in (("R_m_share_%", "Membrane", COL["mem"]), ("R_rev_share_%", "Reversible", COL["rev"]), ("R_irrev_share_%", "Irreversible", COL["irr"])):
    v = FP_DF[col_].values.astype(float); e.barh(yy, v, left=left, color=c_, edgecolor=INK, linewidth=0.5, height=0.66, label=lab_)
    for yi, (l_, w_) in enumerate(zip(left, v)):
        if w_ >= 12: e.text(l_ + w_ / 2, yi, f"{w_:.0f}", ha="center", va="center", fontsize=6.5, color="white" if c_ != COL["mem"] else INK)
    left += v
e.set_yticks(yy); e.set_yticklabels([f"P{p}" for p in pp]); e.set_xlim(0, 100); e.grid(axis="y", visible=False); e.tick_params(axis="y", length=0)
e.set_xlabel("Share of total resistance (%)"); e.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=6.3, handlelength=1.0, columnspacing=0.8)
title(e, "e", "Resistance budget")
save(fig, "Fig1_graphical_abstract")

# ════ Figure 2 | Measured dynamics and the prognostic target ════════════════
seg = df.iloc[: min(STEPS_PER_DAY * 60, len(df))]
fig, ax = plt.subplots(3, 1, figsize=(W15, 5.6), sharex=True, layout="constrained")
ax[0].plot(seg["time_days"], seg["TMP_kPa"], color=INK, lw=0.4); shade_periods(ax[0], seg); ax[0].set_ylabel("TMP (kPa)")
title(ax[0], "a", r"Crossflow TMP $=\frac{1}{2}(P_{\rm feed}+P_{\rm conc})-P_{\rm perm}$, filtration samples")
ax[1].fill_between(seg["time_days"], 0, seg["R_rev"] / 1e11, color=COL["rev"], alpha=0.45, lw=0, label=r"$R_{\rm rev}$")
ax[1].fill_between(seg["time_days"], seg["R_rev"] / 1e11, (seg["R_rev"] + seg["R_irrev"]) / 1e11, color=COL["irr"], alpha=0.5, lw=0, label=r"$R_{\rm irrev}$")
shade_periods(ax[1], seg); ax[1].set_ylabel(r"Resistance ($\times10^{11}$ m$^{-1}$)")
ax[1].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.1)
title(ax[1], "b", "Isotonic decomposition per period")
_d0 = DAILY[DAILY["day"] <= int(seg["day"].max())]; _rt, _ct = RUL_TAB[(DELTA_FRAC_PRIMARY, H)]
ax[2].fill_between(_d0["day"], 0, H, where=_ct[: len(_d0)].astype(bool), color=COL["cens"], step="post", lw=0, label="Censored")
ax[2].step(_d0["day"], _rt[: len(_d0)], where="post", color=COL["power"], lw=1.2, label=r"RUL$_\delta$")
ax[2].set_ylim(0, H + 1.5); ax[2].set_ylabel("Days to next step"); ax[2].set_xlabel("Time (days)")
ax[2].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.1)
title(ax[2], "c", rf"Target: days to next {int(100 * DELTA_FRAC_PRIMARY)} % rise in $R_{{\rm irrev}}$")
save(fig, "Fig2_dynamics_and_target")

# ════ Figure 3 | Envelope validation ═══════════════════════════════════════
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9), layout="constrained", gridspec_kw={"width_ratios": [1.6, 1]})
for lab_, y_, c_, lw_ in (("Daily 10th percentile (primary)", df["R_irrev"], COL["ours"], 1.1),
                          ("Post-backwash median", R_irr_pbw, COL["linear"], 1.0), ("Single pressure gauge", R_irr_feed, COL["learned"], 0.9)):
    ax[0].plot(df["time_days"], np.asarray(y_) / 1e11, color=c_, lw=lw_, label=lab_)
shade_periods(ax[0], df); ax[0].set_xlabel("Time (days)"); ax[0].set_ylabel(r"$R_{\rm irrev}$ ($\times10^{11}$ m$^{-1}$)")
ax[0].legend(loc="upper left", fontsize=6.6); title(ax[0], "a", "Three extractions of the irreversible envelope")
lab3 = ["Daily q10", "Post-BW median", "Single gauge"]
im = ax[1].imshow(ENV_VALID.values.astype(float), cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        v = float(ENV_VALID.values[i, j]); ax[1].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else INK)
ax[1].set_xticks(range(3)); ax[1].set_yticks(range(3)); ax[1].set_xticklabels(lab3, rotation=30, ha="right"); ax[1].set_yticklabels(lab3)
ax[1].grid(False); ax[1].tick_params(length=0)
for s_ in ax[1].spines.values(): s_.set_visible(False)
cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03); cb.set_label("Pearson r", fontsize=7); cb.ax.tick_params(labelsize=6.5)
title(ax[1], "b", "Agreement")
save(fig, "Fig3_envelope_validation")

# ════ Figure 4 | Validity, accuracy, daily countdown ═══════════════════════
fig, ax = plt.subplots(1, 3, figsize=(W2, 3.0), layout="constrained", gridspec_kw={"width_ratios": [0.9, 1.15, 1.25]})
hbars(ax[0], [NAME[n] for n in vn], [100 * RES_P[n]["validity_rate"] for n in vn],
      [COL["ok"] if RES_P[n]["validity_rate"] > 0.99 else COL["fail"] for n in vn], fmt="{:.1f} %", xmax=140)
ax[0].set_xlabel("Windows with a cleaning date (%)"); title(ax[0], "a", "Validity")
lo_hi = np.array([[RES_P[n]["MAE_days"] - ci_of(n)[0], ci_of(n)[1] - RES_P[n]["MAE_days"]] for n in order]).T
_notes = {yi: f"{RES_P[n]['MAE_days']:.2f}" + (f"   {p_label(PHOLM[f'PC-Net vs {n}'])}" if f"PC-Net vs {n}" in PHOLM else "")
          for yi, n in enumerate(order)}
hbars(ax[1], [NAME[n] for n in order], [RES_P[n]["MAE_days"] for n in order], [MCOL[n] for n in order], err=lo_hi, note=_notes)
ax[1].set_xlim(0, max(ci_of(n)[1] for n in order) * 1.75)
ax[1].set_xlabel("Cleaning-date MAE (days), 95 % CI"); title(ax[1], "b", "Accuracy; p vs PC-Net (Holm)")
draw_countdown(ax[2]); title(ax[2], "c", "Daily recommendation, final fold")
save(fig, "Fig4_validity_accuracy_countdown")

# ════ Figure 5 | Learned correction and the validation guard ═══════════════
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1]})
folds = LEARN_DF.index.tolist(); x = np.arange(len(folds)); bw = 0.26
for j, (col_, lab_, c_) in enumerate((("MAE_seed_only", "Physics seed (untrained)", COL["seed"]),
                                       ("MAE_trained_only", "Trained head", COL["learned"]), ("MAE_deployed", "Deployed (guard)", COL["ours"]))):
    ax[0].bar(x + (j - 1) * bw, np.nan_to_num(LEARN_DF[col_].values.astype(float)), bw, color=c_, edgecolor=INK, linewidth=0.5, label=lab_)
tick = []
for f_ in folds:
    r = LEARN_DF.loc[f_]
    tick.append(f"Fold {f_}\n" + ("no scored\nwindows" if not np.isfinite(float(r["MAE_deployed"])) else ("trained" if r["deployed"] == "trained" else "seed")))
ax[0].set_xticks(x); ax[0].set_xticklabels(tick, fontsize=6.6); ax[0].set_ylabel("Cleaning-date MAE (days)")
ax[0].set_ylim(0, np.nanmax(LEARN_DF[["MAE_seed_only", "MAE_trained_only", "MAE_deployed"]].values.astype(float)) * 1.28)
ax[0].legend(loc="upper left", fontsize=6.5, ncol=1); ax[0].grid(axis="x", visible=False)
title(ax[0], "a", "Per fold: which head the guard deployed")
pool_order = ["Trained head (no guard)", "Power law", "Linear 7d", "Physics seed only", "PC-Net-HMCP"]
pool_val = {"Trained head (no guard)": LEARN_POOL_DF.loc["Trained HMCP only (no guard)", "MAE_days"],
            "Physics seed only": LEARN_POOL_DF.loc["Physics seed only (untrained HMCP)", "MAE_days"],
            "PC-Net-HMCP": RES_P["PC-Net-HMCP"]["MAE_days"], "Linear 7d": RES_P["Linear 7d"]["MAE_days"], "Power law": RES_P["Power law"]["MAE_days"]}
pg = PHOLM.get("PC-Net vs Trained head (no guard)")
_gn = {0: f"{pool_val['Trained head (no guard)']:.2f}   vs guard: {p_label(pg)}"} if pg is not None else None
hbars(ax[1], [NAME[n] if n != "PC-Net-HMCP" else "Deployed (guard)" for n in pool_order], [pool_val[n] for n in pool_order],
      [COL["learned"], COL["power"], COL["linear"], COL["seed"], COL["ours"]], note=_gn)
ax[1].set_xlim(0, max(pool_val.values()) * 1.75)
ax[1].set_xlabel("Pooled cleaning-date MAE (days)"); title(ax[1], "b", "Where the accuracy comes from")
save(fig, "Fig5_learned_correction_and_guard")

# ════ Figure 6 | Generalisation across operating regimes ═══════════════════
fig, ax = plt.subplots(1, 2, figsize=(W15, 2.7), layout="constrained", gridspec_kw={"width_ratios": [1.5, 1]})
pp = RES_PERIOD_DF.index.astype(int).tolist(); x = np.arange(len(pp)); bw = 0.25
for j, (col_, lab_, c_) in enumerate((("HMCP_MAE", "PC-Net (ours)", COL["ours"]), ("Linear 7d_MAE", "Linear 7 d", COL["linear"]), ("Power law_MAE", "Power law", COL["power"]))):
    v = RES_PERIOD_DF[col_].values.astype(float)
    ax[0].bar(x + (j - 1) * bw, v, bw, color=c_, edgecolor=INK, linewidth=0.5, label=lab_)
    for xi, vv in enumerate(v): ax[0].text(xi + (j - 1) * bw, vv + 0.08, f"{vv:.2f}", ha="center", va="bottom", fontsize=6.2)
ax[0].set_xticks(x); ax[0].set_xticklabels([f"Period {p}\n({int(RES_PERIOD_DF.loc[p, 'days_in_period'])} d)" for p in pp])
ax[0].set_ylabel("Cleaning-date MAE (days)"); ax[0].set_ylim(0, np.nanmax(RES_PERIOD_DF[["HMCP_MAE", "Linear 7d_MAE", "Power law_MAE"]].values.astype(float)) * 1.3)
ax[0].legend(loc="upper right", ncol=3, fontsize=6.3, handlelength=1.0, columnspacing=0.8); ax[0].grid(axis="x", visible=False)
title(ax[0], "a", "Accuracy by operating regime")
vv = RES_PERIOD_DF["HMCP_validity"].values.astype(float) * 100
ax[1].bar(x, vv, 0.5, color=[COL["ok"] if v_ > 99 else COL["fail"] for v_ in vv], edgecolor=INK, linewidth=0.5)
for xi, v_ in enumerate(vv): ax[1].text(xi, v_ + 2, f"{v_:.0f} %", ha="center", fontsize=7)
ax[1].set_xticks(x); ax[1].set_xticklabels([f"Period {p}" for p in pp]); ax[1].set_ylim(0, 115); ax[1].set_ylabel("Windows with a date (%)")
ax[1].grid(axis="x", visible=False); title(ax[1], "b", "Validity by regime")
save(fig, "Fig6_regime_generalisation")

# ════ Figure 7 | Ablation ══════════════════════════════════════════════════
abl = ["No monotonicity", "Soft monotonicity penalty", "Encoder: transformer", "Full (HMCP)"]
abl = [a_ for a_ in abl if a_ in ABL_DF.index]
lab_abl = {"No monotonicity": "No monotonicity", "Soft monotonicity penalty": "Soft penalty", "Encoder: transformer": "HMCP, transformer encoder", "Full (HMCP)": "HMCP, TCN-LSTM (full)"}
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), layout="constrained")
vals = ABL_DF.loc[abl, "validity_rate"].values.astype(float) * 100
hbars(ax[0], [lab_abl[a_] for a_ in abl], vals, [COL["ok"] if v > 99 else COL["fail"] for v in vals], fmt="{:.0f} %", xmax=140)
ax[0].set_xlabel("Windows with a cleaning date (%)"); title(ax[0], "a", "Validity comes from the head")
hbars(ax[1], [lab_abl[a_] for a_ in abl], ABL_DF.loc[abl, "MAE_days"].values.astype(float),
      [COL["ours"] if a_ == "Full (HMCP)" else COL["learned"] for a_ in abl])
ax[1].set_yticklabels([]); ax[1].set_xlabel("Cleaning-date MAE, final fold (days)"); title(ax[1], "b", "Accuracy does not depend on the encoder")
save(fig, "Fig7_ablation")

# ════ Figure 8 | Operating fingerprint ═════════════════════════════════════
fig, ax = plt.subplots(1, 3, figsize=(W2, 3.0), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1, 1.1]})
pp = FP_DF.index.astype(int).tolist(); yy = np.arange(len(pp)); left = np.zeros(len(pp))
for col_, lab_, c_ in (("R_m_share_%", r"Membrane $R_m$", COL["mem"]), ("R_rev_share_%", r"Reversible $R_{\rm rev}$", COL["rev"]),
                       ("R_irrev_share_%", r"Irreversible $R_{\rm irrev}$", COL["irr"])):
    v = FP_DF[col_].values.astype(float)
    ax[0].barh(yy, v, left=left, color=c_, edgecolor=INK, linewidth=0.5, height=0.66, label=lab_)
    for yi, (l_, w_) in enumerate(zip(left, v)):
        if w_ >= 9: ax[0].text(l_ + w_ / 2, yi, f"{w_:.0f} %", ha="center", va="center", fontsize=6.4, color="white" if c_ != COL["mem"] else INK)
    left += v
ax[0].set_yticks(yy); ax[0].set_yticklabels([f"Period {p}" for p in pp]); ax[0].set_xlim(0, 100); ax[0].grid(axis="y", visible=False)
ax[0].tick_params(axis="y", length=0); ax[0].set_xlabel("Share of total resistance (%)")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.3, handlelength=1.0, columnspacing=0.8)
title(ax[0], "a", "Resistance-in-series budget")
r_ = FP_DF["irrev_fouling_rate_kPa_per_day"].values.astype(float)
ax[1].bar(yy, r_, 0.55, color=COL["irr"], edgecolor=INK, linewidth=0.5)
_top = np.nanmax(r_)
for xi, (v, bwd, rv) in enumerate(zip(r_, FP_DF["backwashes_per_day"], FP_DF["hydraulic_reversibility_%"])):
    ax[1].text(xi, v + _top * 0.03, f"{v:.3f}", ha="center", va="bottom", fontsize=6.8, fontweight="bold")
    ax[1].text(xi, v + _top * 0.13, f"{bwd:.0f} BW/d\n{rv:.0f} % rev.", ha="center", va="bottom", fontsize=5.9, color="#444444", linespacing=1.1)
ax[1].set_xticks(yy); ax[1].set_xticklabels([f"P{p}" for p in pp], fontsize=7.5)
ax[1].set_ylabel("Irreversible fouling rate (kPa/day)"); ax[1].set_ylim(0, _top * 1.6); ax[1].grid(axis="x", visible=False)
title(ax[1], "b", "Fouling rate, TMP-equivalent")
lo_t, hi_t = np.percentile(WK_DF["T_C"], [1, 99]); lo_r, hi_r = np.percentile(WK_DF["rate_kPa_per_day"], [0, 98])
pad_t = 0.08 * (hi_t - lo_t + 1e-9)
inside = WK_DF["T_C"].between(lo_t - pad_t, hi_t + pad_t) & (WK_DF["rate_kPa_per_day"] <= hi_r * 1.1)
for p_, c_ in zip(sorted(WK_DF["period"].unique()), (COL["ours"], COL["linear"], COL["ok"], COL["power"])):
    s_ = WK_DF[(WK_DF["period"] == p_) & inside]
    ax[2].scatter(s_["T_C"], s_["rate_kPa_per_day"], s=13, color=c_, edgecolor=INK, linewidth=0.3, alpha=0.85, label=f"P{p_}")
ax[2].set_xlim(lo_t - pad_t, hi_t + pad_t); ax[2].set_ylim(min(0, lo_r) - 0.02 * hi_r, hi_r * 1.12)
ax[2].set_xlabel("Weekly mean temperature (°C)"); ax[2].set_ylabel("Weekly irreversible rate (kPa/day)")
n_out = int((~inside).sum())
ax[2].text(0.03, 0.97, rf"within-period $\rho$ = {TEMP_DEP['spearman_rho']:.2f}" + f"\np = {TEMP_DEP['p_value']:.3f}, n = {TEMP_DEP['n_weeks']} weeks"
           + (f"\n{n_out} week(s) beyond axes" if n_out else ""), transform=ax[2].transAxes, va="top", fontsize=6.3,
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.6))
ax[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=6.3, handletextpad=0.2, columnspacing=0.9)
title(ax[2], "c", "Temperature association")
save(fig, "Fig8_operating_fingerprint")

# ════ Figure 9 | The forecast in kPa ═══════════════════════════════════════
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.9), layout="constrained", gridspec_kw={"width_ratios": [1.05, 1, 1]})
ko = ["Power law", "Physics seed only", "Linear 7d", "Persistence", "PC-Net-HMCP"]
ko = [k for k in ko if k in KPA_DF.index]
kerr = np.array([[KPA_DF.loc[k, "MAE_kPa"] - KPA_DF.loc[k, "CI_low"], KPA_DF.loc[k, "CI_high"] - KPA_DF.loc[k, "MAE_kPa"]] for k in ko]).T
hbars(ax[0], [NAME.get(k, k) for k in ko], KPA_DF.loc[ko, "MAE_kPa"].values, [MCOL.get(k, COL["learned"]) for k in ko], fmt="{:.2f}", err=kerr)
ax[0].set_xlabel("MAE of 14-day irreversible rise (kPa)"); title(ax[0], "a", "Trajectory error, all windows")
ot_f = FINAL["ot"]; per_f = CV_PERIOD[-1]; S = fd.S
obs_end = r_to_kpa(fd.te[2][:, -1] * S, per_f); pred_end = r_to_kpa(ot_f["dp"][0][:, -1] * S, per_f)
cand = np.flatnonzero((fd.te[7] < 0.5) & (obs_end > 0))
if len(cand):
    e_ = np.abs(pred_end[cand] - obs_end[cand]); i = int(cand[np.argsort(e_)[len(e_) // 2]])      # median-error window
else:
    i = int(len(obs_end) // 2)
k = np.arange(1, H + 1); pw_ = [per_f[i]] * H
ax[1].plot(k, r_to_kpa(fd.te[2][i] * S, pw_), color=INK, lw=1.2, ls="--", label="Observed")
ax[1].plot(k, r_to_kpa(ot_f["dp"][0][i] * S, pw_), "o-", color=COL["ours"], ms=2.5, lw=1.3, label="PC-Net (ours)")
ax[1].plot(k, r_to_kpa(np.arange(1, H + 1) * fd.te[9][i] * S, pw_), "s-", color=COL["linear"], ms=2.2, lw=0.9, label="Linear 7 d")
step_kpa = float(r_to_kpa((fd.te[8][i] - fd.te[5][i]) * S, [per_f[i]])[0])
ax[1].axhline(step_kpa, color=COL["irr"], lw=0.9, ls=":")
ax[1].text(1.2, step_kpa, f"one degradation step ({step_kpa:.2f} kPa)", va="bottom", fontsize=6.2, color=COL["irr"])
ax[1].set_xlabel("Days ahead"); ax[1].set_ylabel("Irreversible TMP rise (kPa)"); ax[1].legend(loc="upper left", fontsize=6.3)
ax[1].set_ylim(bottom=min(0, ax[1].get_ylim()[0]), top=max(ax[1].get_ylim()[1], step_kpa * 1.35))
title(ax[1], "b", "A median-error forecast")
xv, yv = KPA["true"], KPA["PC-Net-HMCP"]; lim = float(np.nanpercentile(np.concatenate([xv, yv]), 99.5)) * 1.05 + 1e-9
hb = ax[2].hexbin(xv, yv, gridsize=30, extent=(0, lim, 0, lim), cmap="Blues", mincnt=1, bins="log", linewidths=0)
ax[2].plot([0, lim], [0, lim], color=INK, lw=0.8, ls="--"); ax[2].set_xlim(0, lim); ax[2].set_ylim(0, lim)
cb = fig.colorbar(hb, ax=ax[2], fraction=0.046, pad=0.03); cb.set_label("Windows (log)", fontsize=6.8); cb.ax.tick_params(labelsize=6.2)
ax[2].set_xlabel("Observed rise (kPa)"); ax[2].set_ylabel("PC-Net forecast (kPa)"); ax[2].grid(False)
title(ax[2], "c", "Forecast vs observed")
save(fig, "Fig9_forecast_in_kPa")

# ════ Figure S1 | Decision stability (supplementary) ════════════════════════
sn = [n for n in ("PC-Net-HMCP", "Linear 7d", "Power law", "XGBoost RUL", "Soft-mono deep", "Unconstrained deep") if n in STAB][::-1]
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.7), layout="constrained")
hbars(ax[0], [NAME[n] for n in sn], [100 * STAB[n]["usable_day_fraction"] for n in sn],
      [COL["ok"] if STAB[n]["usable_day_fraction"] > 0.99 else COL["fail"] for n in sn], fmt="{:.0f} %", xmax=140)
ax[0].set_xlabel("Days with a usable date (%)"); title(ax[0], "a", "Availability")
for j, (key, lab_, fmt_) in enumerate((("countdown_consistency_days", "Countdown consistency (days)", "{:.2f}"),
                                       ("reversals_per_week", "Schedule reversals per week", "{:.2f}")), start=1):
    vals_ = [STAB[n][key] for n in sn]
    notes = {yi: (fmt_.format(v) + f"  (n = {STAB[n].get('n_active_pairs', 0)})") for yi, (n, v) in enumerate(zip(sn, vals_)) if np.isfinite(v)}
    hbars(ax[j], [NAME[n] for n in sn], vals_, [MCOL.get(n, COL["learned"]) for n in sn], note=notes)
    ax[j].set_yticklabels([]); ax[j].set_xlabel(lab_ + "\n(lower is better)")
    ax[j].set_xlim(0, np.nanmax(np.asarray(vals_, float)) * 1.9)
title(ax[1], "b", "Consistency"); title(ax[2], "c", "Reversals")
save(fig, "FigS1_decision_stability")
print(f"\nDone. {len(list(FIGDIR.glob('*.png')))} figures in {FIGDIR}")
