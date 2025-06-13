#!/usr/bin/env python3
"""Batch‑compress decision‑tree JSON files.

This script preserves the original logic but loops over every *.json file
in results/tree and drops a corresponding compressed file in
results/compressed_tree.

Example:
    $ python compress_trees_batch.py
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path

INPUT_DIR = Path("results/tree")
OUTPUT_DIR = Path("results/compressed_tree")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class SetEncoder(json.JSONEncoder):
    def default(self, obj):  # type: ignore[override]
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


# helper function given a range and value x returns if x is in the range

def is_in_range(x: int, lower: int | None, upper: int | None) -> bool:  # noqa: N803
    if lower is None and upper is None:
        return True
    if lower is None:
        return x <= upper  # type: ignore[operator]
    if upper is None:
        return x > lower
    return x <= upper and x > lower  # type: ignore[operator]


for tree_path in INPUT_DIR.glob("*.json"):
    with tree_path.open() as f:
        tree = json.load(f)

    paths = tree["paths"]

    # First cleanup the tree by rounding the decision points to integer values
    path_ids: set[int] = set()
    path_classes = tree["classes"]

    # assign ids and round thresholds
    for idx, path in enumerate(paths):
        path["id"] = idx
        path_ids.add(idx)
        for condition in path["conditions"]:
            operation = condition["operation"]
            if operation == "<=":
                condition["value"] = math.floor(condition["value"])
            else:
                condition["value"] = math.floor(condition["value"])

    # Find all breakpoints for each feature and create a set of disjoint ranges
    breakpoints: dict[str, list[int]] = defaultdict(set)  # type: ignore[assignment]
    for path in paths:
        for condition in path["conditions"]:
            feature = condition["feature"]
            value = condition["value"]
            breakpoints[feature].add(value)

    # sort breakpoint lists
    for feature in breakpoints:
        points = list(breakpoints[feature])
        points.sort()
        breakpoints[feature] = points  # type: ignore[assignment]

    # collapse all paths to ranges for each feature
    for path in paths:
        compressed: dict[str, dict[str, int | None]] = {}
        for feature in breakpoints:
            compressed[feature] = {"min": None, "max": None}

        for condition in path["conditions"]:
            feature = condition["feature"]
            operation = condition["operation"]
            value = condition["value"]
            if operation == "<=" and compressed[feature]["max"] is None:
                compressed[feature]["max"] = value
            elif operation == ">" and compressed[feature]["min"] is None:
                compressed[feature]["min"] = value
            elif operation == "<=" and value < compressed[feature]["max"]:  # type: ignore[operator]
                compressed[feature]["max"] = value
            elif operation == ">" and value > compressed[feature]["min"]:  # type: ignore[operator]
                compressed[feature]["min"] = value

        path["compressed"] = compressed

    # create buckets for each feature, where each is a list of sets
    buckets_id: dict[str, list[set[int]]] = {}
    buckets_class: dict[str, list[set[str]]] = {}
    for feature in breakpoints:
        num_points = len(breakpoints[feature])
        buckets_id[feature] = [set() for _ in range(num_points + 1)]
        buckets_class[feature] = [set() for _ in range(num_points + 1)]

    # fill buckets
    for path in paths:
        for feature_name, feature in path["compressed"].items():
            lower = feature["min"]
            upper = feature["max"]
            pid = path["id"]
            cls = path["classification"]

            for idx, bp in enumerate(breakpoints[feature_name]):
                if is_in_range(bp, lower, upper):
                    buckets_id[feature_name][idx].add(pid)
                    buckets_class[feature_name][idx].add(cls)
            # last bucket (> last breakpoint)
            if is_in_range(bp + 1, lower, upper):
                buckets_id[feature_name][-1].add(pid)
                buckets_class[feature_name][-1].add(cls)

    # combine breakpoints and buckets to one representation
    compressed_layers: dict[str, list[dict[str, object]]] = defaultdict(list)
    for feature_name in buckets_id:
        lower = None
        upper = breakpoints[feature_name][0]
        compressed_layers[feature_name].append(
            {
                "min": lower,
                "max": upper,
                "paths": buckets_id[feature_name][0],
                "classes": buckets_class[feature_name][0],
            }
        )
        for i in range(1, len(buckets_id[feature_name]) - 1):
            lower = breakpoints[feature_name][i - 1]
            upper = breakpoints[feature_name][i]
            compressed_layers[feature_name].append(
                {
                    "min": lower,
                    "max": upper,
                    "paths": buckets_id[feature_name][i],
                    "classes": buckets_class[feature_name][i],
                }
            )
        lower = breakpoints[feature_name][-1]
        upper = None
        compressed_layers[feature_name].append(
            {
                "min": lower,
                "max": upper,
                "paths": buckets_id[feature_name][-1],
                "classes": buckets_class[feature_name][-1],
            }
        )

    path_to_class = {path["id"]: path["classification"] for path in paths}

    compressed_tree = {
        "paths": list(path_ids),
        "classes": path_classes,
        "layers": compressed_layers,
        "path_to_class": path_to_class,
    }

    out_path = OUTPUT_DIR / tree_path.name.replace("tree", "compressed_tree")
    with out_path.open("w") as f_out:
        json.dump(compressed_tree, f_out, indent=4, cls=SetEncoder)

    # print(f"Wrote {out_path.relative_to(Path.cwd())}")
