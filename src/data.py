"""Data loading: the original course file and the official Météo-France series.

Two sources, both monthly mean of daily maximum temperature (°C):
  * data/original/data2.csv  - the 24 values given with the ISEP assignment (2023-2024)
  * Météo-France "Données climatologiques de base - mensuelles", field TX,
    station PARIS-MONTSOURIS (NUM_POSTE 75114001). See data/README.md.
"""
from pathlib import Path
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
ORIGINAL_CSV = DATA / "original" / "data2.csv"
MONTSOURIS_CSV = DATA / "montsouris_tx_monthly.csv"

STATION = "75114001"  # PARIS-MONTSOURIS
MF_BASE = "https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/MENS/"
MF_FILES = ["MENSQ_75_previous-1950-2024.csv.gz", "MENSQ_75_latest-2025-2026.csv.gz"]
KEEP = ["NUM_POSTE", "NOM_USUEL", "AAAAMM", "TX", "QTX", "NBTX"]

# Values stated in the assignment PDF (IF1205_projetDS-25.pdf, Q13 and Q17).
TRUTH_2025_PDF = [7.5, 8.6, 14.6, 20.0]
# Values hard-coded in the original code after commit 51648b7 ("真实温度改正").
TRUTH_2025_CODE = [8.0, 10.0, 14.0, 18.0]


def download_raw(force=False):
    """Download the two département-75 monthly files into data/raw/."""
    RAW.mkdir(parents=True, exist_ok=True)
    for name in MF_FILES:
        target = RAW / name
        if force or not target.exists():
            urllib.request.urlretrieve(MF_BASE + name, target)


def read_raw_dept75():
    """All stations of département 75, selected columns only."""
    frames = [pd.read_csv(RAW / f, sep=";", usecols=KEEP, dtype={"NUM_POSTE": str})
              for f in MF_FILES]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["AAAAMM"].astype(str), format="%Y%m")
    return df


def build_montsouris_csv():
    """Extract Montsouris TX, keeping only months with every day present
    (this drops the current, still-running month)."""
    df = read_raw_dept75()
    m = df[df["NUM_POSTE"] == STATION].sort_values("date")
    complete = m["NBTX"] == m["date"].dt.days_in_month
    m = m[complete & m["TX"].notna()]
    out = m[["date", "TX", "QTX", "NBTX"]]
    out.to_csv(MONTSOURIS_CSV, index=False, date_format="%Y-%m")
    return out


def load_montsouris():
    """Monthly TX series (pd.Series indexed by month start, freq MS)."""
    df = pd.read_csv(MONTSOURIS_CSV, parse_dates=["date"])
    s = df.set_index("date")["TX"].asfreq("MS")
    assert s.notna().all(), "gap in the Montsouris series"
    return s


def load_original():
    """The 24 values of data2.csv as a monthly Series (2023-01 .. 2024-12)."""
    df = pd.read_csv(ORIGINAL_CSV)
    idx = pd.date_range("2023-01-01", periods=len(df), freq="MS")
    return pd.Series(df["Temperature_maximale"].to_numpy(float), index=idx, name="TX")


if __name__ == "__main__":
    download_raw()
    s = build_montsouris_csv()
    print(f"wrote {MONTSOURIS_CSV.relative_to(ROOT)}: {len(s)} months, "
          f"{s['date'].min():%Y-%m} .. {s['date'].max():%Y-%m}")
