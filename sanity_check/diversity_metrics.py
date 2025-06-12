#!/usr/bin/env python3
"""diversity_metrics.py (fast version)

Estimate how much diversity each CSV adds without building a giant in‑memory
DataFrame.  Designed for IoT packet logs with millions of rows.

Quick summary printed as a GitHub‑style table (requires *tabulate*; falls back
to pandas plain text).

Usage
-----
python diversity_metrics.py path/to/processed_dir [-r] [--sample 50000]

Metrics
-------
ΔEntropy  : change in Shannon entropy of *classification* counts
ΔGini     : change in Gini impurity of the same counts
χ² p      : Pearson χ² p‑value old vs new classification counts
Jaccard   : similarity of unique (src,dst) pairs (0 → new pairs, 1 → no new)
KS src p  : Kolmogorov–Smirnov p‑value, source‑port dist (uses sampling)
KS dst p  : Kolmogorov–Smirnov p‑value, dest‑port  dist (uses sampling)

Speed tricks
------------
* No growing DataFrame; we keep Counters / sets / lists.
* Ports for KS are *sampled* (default 50 k) to bound cost.
* (src,dst) pairs are hashed to a 32‑bit int to reduce set overhead.
* pandas reads via **pyarrow** engine when available.
"""

import argparse
from pathlib import Path
from collections import Counter
from typing import List, Set

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, entropy

try:
    from tabulate import tabulate
    _USE_TABULATE = True
except ImportError:
    _USE_TABULATE = False

# -----------------------------------------------------------------------------
# Helper metrics
# -----------------------------------------------------------------------------

def shannon(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    p = np.fromiter(counts.values(), dtype=float)
    p /= total
    return entropy(p, base=2)


def gini(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return 1.0 - sum((n / total) ** 2 for n in counts.values())


def jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

# -----------------------------------------------------------------------------
# Core analysis
# -----------------------------------------------------------------------------

def analyse(csv_files: List[Path], sample_size: int):
    """Return list of dicts with diversity metrics for each added file."""

    # cumulative state (no big DataFrame!)
    class_counter: Counter = Counter()
    pair_hashes: Set[int] = set()
    src_list: List[int] = []
    dst_list: List[int] = []

    rows = []

    for csv_path in csv_files:
        df = pd.read_csv(
            csv_path,
            engine="pyarrow" if pd.__version__ >= "2" else "c",  # fast parse
            usecols=["protocl", "src", "dst", "classfication"],
            dtype={
                "protocl": "uint16",
                "protocol": "uint16",
                "src": "uint16",
                "dst": "uint16",
            },
        )
        # normalise column names
        df.rename(columns={"protocl": "protocol", "classfication": "classification"}, inplace=True)

        # snapshot previous state
        prev_class = class_counter.copy()
        prev_pairs = pair_hashes.copy()
        prev_src = np.asarray(src_list, dtype=np.uint16)
        prev_dst = np.asarray(dst_list, dtype=np.uint16)

        # --- update cumulative structures ------------------------------------
        class_counter.update(df["classification"].value_counts().to_dict())

        # hash (src,dst) into 32‑bit int to save memory
        pair_ids = (df["src"].to_numpy(dtype=np.uint32) << np.uint32(16)) | \
            df["dst"].to_numpy(dtype=np.uint32)


        # extend port lists (keep small ints)
        src_list.extend(df["src"].tolist())
        dst_list.extend(df["dst"].tolist())

        # --- metrics ----------------------------------------------------------
        # χ² classification
        chi_p = np.nan
        if prev_class:
            all_classes = list(set(prev_class) | set(df["classification"].unique()))
            old = [prev_class.get(c, 0) for c in all_classes]
            new = [df["classification"].value_counts().get(c, 0) for c in all_classes]
            _, chi_p, _, _ = chi2_contingency([old, new])

        # entropy & gini deltas
        delta_entropy = shannon(class_counter) - shannon(prev_class)
        delta_gini = gini(class_counter) - gini(prev_class)

        # Jaccard on pair hashes
        jc = jaccard(prev_pairs, pair_hashes)

        # KS tests on sampled ports
        ks_src_p = ks_dst_p = np.nan
        if prev_src.size:
            new_src = df["src"].to_numpy(dtype=np.uint16)
            new_dst = df["dst"].to_numpy(dtype=np.uint16)
            if prev_src.size > sample_size:
                prev_src_sample = np.random.choice(prev_src, sample_size, replace=False)
            else:
                prev_src_sample = prev_src
            if new_src.size > sample_size:
                new_src_sample = np.random.choice(new_src, sample_size, replace=False)
            else:
                new_src_sample = new_src
            if prev_dst.size > sample_size:
                prev_dst_sample = np.random.choice(prev_dst, sample_size, replace=False)
            else:
                prev_dst_sample = prev_dst
            if new_dst.size > sample_size:
                new_dst_sample = np.random.choice(new_dst, sample_size, replace=False)
            else:
                new_dst_sample = new_dst

            ks_src_p = ks_2samp(prev_src_sample, new_src_sample).pvalue
            ks_dst_p = ks_2samp(prev_dst_sample, new_dst_sample).pvalue

        rows.append(
            {
                "File": csv_path.name,
                "Rows": len(df),
                "ΔEntropy": round(delta_entropy, 4),
                "ΔGini": round(delta_gini, 4),
                "χ² p": f"{chi_p:.3g}" if not np.isnan(chi_p) else "NA",
                "Jaccard": round(jc, 3),
                "KS src p": f"{ks_src_p:.3g}" if not np.isnan(ks_src_p) else "NA",
                "KS dst p": f"{ks_dst_p:.3g}" if not np.isnan(ks_dst_p) else "NA",
            }
        )
    return rows

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate diversity contribution of each CSV (fast version).")
    ap.add_argument("csv_dir", help="Directory containing CSV files")
    ap.add_argument("-r", "--recursive", action="store_true", help="Recursively search csv_dir")
    ap.add_argument("--sample", type=int, default=50_000, help="Sample size for KS tests (default 50k)")
    args = ap.parse_args()

    root = Path(args.csv_dir)
    pattern = "**/*.csv" if args.recursive else "*.csv"
    csv_files = sorted(root.glob(pattern))
    if not csv_files:
        print("No CSV files found.")
        return

    table_rows = analyse(csv_files, args.sample)

    if _USE_TABULATE:
        print(tabulate(table_rows, headers="keys", tablefmt="github", floatfmt=".4f"))
    else:
        print(pd.DataFrame(table_rows).to_string(index=False))

    print(
        "\nLegend:\n  • p-values (χ², KS) < 0.05 → new file significantly shifts distribution (GOOD)"
        "\n  • Positive ΔEntropy or ΔGini → richer mix; near 0 → little new info"
        "\n  • Jaccard close to 0 → many unseen (src,dst) pairs; close to 1 → redundant."
    )

if __name__ == "__main__":
    main()
