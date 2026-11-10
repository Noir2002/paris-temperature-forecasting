"""Stage 1: check the course data and the 2025 'truth' against Météo-France.

Run:  python -m src.verify_data      (needs data/raw/, see `python -m src.data`)
"""
import pandas as pd

from .data import (STATION, TRUTH_2025_CODE, TRUTH_2025_PDF, load_original,
                   read_raw_dept75)


def main():
    raw = read_raw_dept75()
    tx = raw.pivot_table(index="date", columns=["NUM_POSTE", "NOM_USUEL"], values="TX")
    original = load_original()

    print("1) data2.csv (2023-2024) vs TX of every station in département 75")
    sub = tx.loc[original.index].dropna(axis=1, how="all")
    for col in sub.columns:
        diff = (sub[col] - original).abs()
        print(f"   {col[0]} {col[1]:<24} exact matches {int((diff < 1e-9).sum()):2d}/24"
              f"   max |diff| {diff.max():.1f} °C")

    print("\n2) Jan-Apr 2025, TX by station (°C)")
    months = pd.date_range("2025-01-01", periods=4, freq="MS")
    table = tx.loc[months].dropna(axis=1, how="all").T
    table.columns = ["Jan", "Feb", "Mar", "Apr"]
    table.loc[("assignment PDF", ""), :] = TRUTH_2025_PDF
    table.loc[("original code", ""), :] = TRUTH_2025_CODE
    print(table.to_string())

    print("\n3) Montsouris 1991-2020 mean of monthly TX (°C)")
    mt = tx[STATION].iloc[:, 0]
    normals = mt["1991":"2020"].groupby(mt["1991":"2020"].index.month).agg(["mean", "count"])
    print(normals.round(2).T.to_string())


if __name__ == "__main__":
    main()
