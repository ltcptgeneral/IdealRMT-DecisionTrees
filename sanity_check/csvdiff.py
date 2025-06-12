#!/usr/bin/env python3
"""
csvdiff.py file1.csv file2.csv
Streams both files; prints the first differing line or
‘No differences found’. Uses O(1) memory.
"""

import sys
from itertools import zip_longest
from pathlib import Path

def open_checked(p: str):
    print(p)
    path = Path(p)
    try:
        return path.open("r", newline=""), path
    except FileNotFoundError:
        sys.exit(f"Error: {path} not found")

def human(n: int) -> str:
    return f"{n:,}"

def main(a_path: str, b_path: str) -> None:
    fa, a = open_checked(a_path)
    fb, b = open_checked(b_path)

    with fa, fb:
        for idx, (ra, rb) in enumerate(zip_longest(fa, fb), 1):
            if ra != rb:
                print(f"Files differ at line {human(idx)}")
                if ra is None:
                    print(f"{a} ended early")
                elif rb is None:
                    print(f"{b} ended early")
                else:
                    print(f"{a}: {ra.rstrip()}")
                    print(f"{b}: {rb.rstrip()}")
                return
    print("No differences found")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: csvdiff.py file1.csv file2.csv")
    main(sys.argv[1], sys.argv[2])
