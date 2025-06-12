#!/usr/bin/env python3
"""combined.py

Concatenate every CSV that matches the pattern
    data/processed/<name>/<name>.csv
into a single file:
    data/combined/data.csv

The script streams each source CSV in 1‑Mio‑row chunks so memory stays low.
Typos in the historic column names (protocl/classfication) are fixed on‑the‑fly.

Usage
-----
python combined.py

You can optionally supply a different root directory:
python combined.py --root other/processed_dir --out other/combined/data.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import pandas as pd

CHUNK = 1_000_000  # rows per read_csv chunk


def fix_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy columns to canonical names."""
    return df.rename(
        columns={"protocl": "protocol", "classfication": "classification"}
    )


def find_source_csvs(proc_root: Path):
    """Yield CSV paths that exactly match processed/<name>/<name>.csv."""
    for sub in sorted(proc_root.iterdir()):
        if not sub.is_dir():
            continue
        target = sub / f"{sub.name}.csv"
        if target.exists():
            yield target


def combine(proc_root: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first_write = True
    for csv_path in find_source_csvs(proc_root):
        print(f"→ adding {csv_path.relative_to(proc_root.parent)}")
        for chunk in pd.read_csv(csv_path, chunksize=CHUNK):
            chunk = fix_cols(chunk)
            chunk.to_csv(
                out_path,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )
            first_write = False
    print(f"✓ combined CSV written to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Combine processed CSVs into one.")
    p.add_argument("--root", default="data/processed", help="processed dir root")
    p.add_argument("--out", default="data/combined/data.csv", help="output CSV")
    args = p.parse_args()

    combine(Path(args.root).expanduser(), Path(args.out).expanduser())


if __name__ == "__main__":
    main()
