#!/usr/bin/env python3
"""Range‑to‑Prefix evaluation tool

This script keeps the original logic intact while letting you choose
which expansion strategy to run via a command‑line flag.

Example:
    $ python rmt_selectable.py --mode naive
    $ python rmt_selectable.py --mode priority --input mytree.json --output result.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------------------
field_width = {
    "src": 16,
    "dst": 16,
    "protocol": 8,
}

# ---------------------------------------------------------------------------
# Helper routines (unchanged)
# ---------------------------------------------------------------------------

def int_to_bin(i, width):
    return bin(i)[2:].zfill(width)


def increment_dc(pfx):
    idx = pfx.find("*")
    if idx == -1:
        idx = len(pfx)
    idx -= 1
    return pfx[:idx] + "*" + pfx[idx + 1 :]


def can_merge(pfx_a, pfx_b):
    pfx_a = pfx_a.replace("*", "")
    pfx_b = pfx_b.replace("*", "")
    return pfx_a[:-1] == pfx_b[:-1] and pfx_a[-1] != pfx_b[-1]


def merge(pfx_a, prefixes):
    pfx_a = increment_dc(pfx_a)
    prefixes[-1] = pfx_a

    for i in range(len(prefixes) - 2, -1, -1):
        if can_merge(prefixes[i], prefixes[i + 1]):
            prefixes.pop()
            pfx = increment_dc(prefixes[i])
            prefixes[i] = pfx


def convert_range(lower, upper, width):
    prefixes = []
    prefix = int_to_bin(lower, width)
    prefixes.append(prefix)
    norm_upper = min(upper, 2 ** width - 1)
    for i in range(lower + 1, norm_upper + 1):
        prefix = int_to_bin(i, width)
        if can_merge(prefix, prefixes[-1]):
            merge(prefix, prefixes)
        else:
            prefixes.append(prefix)
    return prefixes

# ---------------------------------------------------------------------------
# RMT construction strategies (logic preserved)
# ---------------------------------------------------------------------------

def worst_case_rmt(tree):
    rmt = []
    step = 0

    tcam_bits = 0
    ram_bits = 0

    for layer in layers:
        num_ranges = len(layers[layer])
        # assume that each range requires all of 2*k prefixes when performing prefix expansion
        # therefore there are 2*k * R for R ranges and width k
        num_prefixes = 2 * field_width[layer] * num_ranges
        prefix_width = field_width[layer]

        tcam = {
            "id": f"{layer}_range",
            "step": step,
            "match": "ternary",
            "entries": num_prefixes,
            "key_size": prefix_width,
        }
        tcam_bits += num_prefixes * prefix_width

        # assume basic pointer reuse for metadata storage
        ram = {
            "id": f"{layer}_meta",
            "step": step,
            "match": "exact",
            "method": "index",
            "key_size": math.ceil(math.log2(num_ranges)),
            "data_size": len(classes),
        }
        ram_bits += num_ranges * len(classes)

        rmt.append(tcam)
        rmt.append(ram)

        step += 1

    return rmt, tcam_bits, ram_bits


def naive_rmt(tree):
    rmt = []
    step = 0

    tcam_bits = 0
    ram_bits = 0

    for layer in layers:
        num_prefixes = 0
        prefix_width = field_width[layer]
        # for each range in the layer, convert the ranges to prefixes using naive range expansion
        for r in layers[layer]:
            if r["min"] is None:
                r["min"] = 0
            elif r["max"] is None:
                r["max"] = 2 ** prefix_width
            prefixes = convert_range(r["min"], r["max"], prefix_width)
            r["prefixes"] = prefixes
            num_prefixes += len(prefixes)
            tcam_bits += len(prefixes) * prefix_width

        tcam = {
            "id": f"{layer}_range",
            "step": step,
            "match": "ternary",
            "entries": num_prefixes,
            "key_size": prefix_width,
            "ranges": layers[layer],
        }

        num_ranges = len(layers[layer])
        # assume no pointer reuse for metadata storage
        ram = {
            "id": f"{layer}_meta",
            "step": step,
            "match": "exact",
            "method": "index",
            "key_size": math.ceil(math.log2(num_ranges)),
            "data_size": len(classes),
        }
        ram_bits += num_ranges * len(classes)

        rmt.append(tcam)
        rmt.append(ram)

        step += 1

    return rmt, tcam_bits, ram_bits


def priority_aware(tree):
    rmt = []
    step = 0

    tcam_bits = 0
    ram_bits = 0

    for layer in layers:
        num_prefixes = 0
        prefix_width = field_width[layer]
        # for each range, run the regular prefix expansion, and also the prefix expansion setting the minimum to 0
        # then check which set of prefixes would be better
        # we will assume the ranges are already disjoint and in the correct order
        for r in layers[layer]:
            if r["min"] is None:
                r["min"] = 0
            elif r["max"] is None:
                r["max"] = 2 ** prefix_width
            regular_prefixes = convert_range(r["min"], r["max"], prefix_width)
            zero_start_prefixes = convert_range(0, r["max"], prefix_width)

            if len(regular_prefixes) <= len(zero_start_prefixes):
                pfx_type = "exact"
                prefixes = regular_prefixes
            else:
                pfx_type = "zero"
                prefixes = zero_start_prefixes

            r["prefixes"] = prefixes
            r["prefix_type"] = pfx_type
            num_prefixes += len(prefixes)
            tcam_bits += len(prefixes) * prefix_width

        tcam = {
            "id": f"{layer}_range",
            "step": step,
            "match": "ternary",
            "entries": num_prefixes,
            "key_size": prefix_width,
            "ranges": layers[layer],
        }

        num_ranges = len(layers[layer])
        # assume no pointer reuse for metadata storage
        ram = {
            "id": f"{layer}_meta",
            "step": step,
            "match": "exact",
            "method": "index",
            "key_size": math.ceil(math.log2(num_ranges)),
            "data_size": len(classes),
        }
        ram_bits += num_ranges * len(classes)

        rmt.append(tcam)
        rmt.append(ram)

        step += 1

    return rmt, tcam_bits, ram_bits

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RMT memory usage for different range‑to‑prefix strategies.")
    parser.add_argument("--mode", choices=["worst", "naive", "priority"], default="worst", help="Strategy to use")
    parser.add_argument("--input", default="compressed_tree.json", help="Input tree JSON file")
    parser.add_argument("--output", default=None, help="Output RMT JSON file (defaults to <mode>_rmt.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Keep the original variable names so the functions stay unchanged
    global layers, classes

    try:
        with open(args.input) as f:
            tree = json.load(f)
    except FileNotFoundError:
        sys.exit(f"Input file '{args.input}' not found.")

    layers = tree["layers"]
    classes = tree["classes"]

    if args.mode == "worst":
        rmt, tcam_bits, ram_bits = worst_case_rmt(tree)
        default_out = "worst_case_rmt.json"
    elif args.mode == "naive":
        rmt, tcam_bits, ram_bits = naive_rmt(tree)
        default_out = "naive_rmt.json"
    else:  # priority
        rmt, tcam_bits, ram_bits = priority_aware(tree)
        default_out = "priority_aware.json"

    out_file = args.output or default_out

    with open(out_file, "w") as f:
        json.dump(rmt, f, indent=4)

    #! command python3 ideal-rmt-simulator/sim.py {out_file}
    print(f"Output written to {out_file}")
    print(f"TCAM bits: {tcam_bits}")
    print(f"RAM bits:  {ram_bits}")


if __name__ == "__main__":
    main()
