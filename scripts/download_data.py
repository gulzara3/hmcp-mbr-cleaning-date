"""Download and verify the Ghent BIOMATH sidestream MBR dataset (Zenodo record 821585).

Usage
-----
    python scripts/download_data.py                 # downloads into ./results/data (where the notebook looks)
    python scripts/download_data.py --dest D:/mbr   # any other folder
    python scripts/download_data.py --extract       # also unpacks the .7z archives (needs: pip install py7zr)

Every file is checked against its published size and, for .7z archives, its file signature, so a truncated or
HTML error page is never accepted. Files that are already present and valid are skipped.
If Zenodo blocks scripted downloads on your network, download the files manually from
https://doi.org/10.5281/zenodo.821585 into the destination folder; the notebook will find them there.
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

RECORD = "821585"
FILES = {  # approximate published sizes in bytes (accepted within +/-15 %)
    "Metadata.csv": 4_300,
    "Settings_MBR.7z": 21_000,
    "Data_MBR_Period_1.7z": 18_700_000,
    "Data_MBR_Period_2.7z": 39_900_000,
    "Data_MBR_Period_3.7z": 382_800_000,
    "Data_MBR_Period_4.7z": 238_800_000,
}
URLS = [
    "https://zenodo.org/records/{rec}/files/{name}?download=1",
    "https://zenodo.org/api/records/{rec}/files/{name}/content",
]
SEVEN_ZIP_MAGIC = b"7z\xbc\xaf\x27\x1c"
HEADERS = {"User-Agent": "hmcp-mbr-cleaning-date/1.0 (dataset download)", "Accept": "*/*"}


def is_valid(path: Path, name: str) -> bool:
    if not path.exists():
        return False
    size, expected = path.stat().st_size, FILES[name]
    if not 0.85 * expected <= size <= 1.15 * expected:
        return False
    with open(path, "rb") as fh:
        head = fh.read(16)
    if name.endswith(".7z"):
        return head.startswith(SEVEN_ZIP_MAGIC)
    return b"<html" not in head.lower()


def fetch(url: str, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    if total and tmp.stat().st_size != total:
        tmp.unlink(missing_ok=True)
        raise IOError("truncated download")
    tmp.replace(dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="results/data", help="destination folder (default: ./results/data)")
    ap.add_argument("--extract", action="store_true", help="unpack the .7z archives after download")
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    failed = []
    for name in FILES:
        target = dest / name
        if is_valid(target, name):
            print(f"[ok]   {name} already present")
            continue
        print(f"[get]  {name}")
        for attempt, tmpl in enumerate(URLS * 2, 1):
            try:
                fetch(tmpl.format(rec=RECORD, name=name), target)
                if is_valid(target, name):
                    break
                raise IOError("file failed the size/format check")
            except Exception as exc:  # noqa: BLE001
                print(f"       attempt {attempt} failed: {exc}")
                time.sleep(min(10 * attempt, 60))
        else:
            failed.append(name)

    if args.extract:
        try:
            import py7zr
        except ImportError:
            print("Install py7zr to extract:  pip install py7zr")
            return 1
        for name in FILES:
            if name.endswith(".7z") and is_valid(dest / name, name):
                print(f"[unz]  {name}")
                with py7zr.SevenZipFile(dest / name, "r") as z:
                    z.extractall(dest / name[:-3])

    if failed:
        print("\nCould not download:", ", ".join(failed))
        print(f"Download them manually from https://doi.org/10.5281/zenodo.{RECORD} into {dest.resolve()}")
        return 1
    print(f"\nAll files verified in {dest.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
