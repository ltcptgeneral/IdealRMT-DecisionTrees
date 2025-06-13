#!/usr/bin/env python3
"""
Train a decision tree, optionally “nudge” its split thresholds, and
export the result as JSON.

Usage examples
--------------
# plain training, no nudging
python build_tree.py --input data/combined/data.csv --output tree.json

# nudge every internal threshold, keeping only the top-2 bits
python build_tree.py --input data/combined/data.csv --output tree.json \
                     --nudge --bits 2
"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, _tree

# ----------------------------------------------------------------------
# 1. command-line arguments
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input",  "-i", help="CSV file with protocol,src,dst,label", default="../data/combined/data.csv")
parser.add_argument("--output", "-o", help="Path for the exported JSON tree", default="tree.json")
parser.add_argument("--depth",  "-d", type=int, default=5,
                    help="Max depth of the decision tree (default: 5)")
parser.add_argument("--nudge",  action="store_true",
                    help="Enable threshold nudging")
parser.add_argument("--bits",   type=int, default=2,
                    help="Number of bits to keep when nudging (default: 2)")
args = parser.parse_args()

# ----------------------------------------------------------------------
# 2. helper functions
# ----------------------------------------------------------------------
def nudge_threshold_max_n_bits(threshold: float, n_bits: int) -> int:
    """Remove n bits from each"""
    threshold = math.floor(threshold)
    if n_bits == 0:
        return threshold
    
    mask = pow(2, 32) - 1 ^ ((1 << n_bits) - 1)
    nudged_value = threshold & mask
    if threshold & (1 << (n_bits - 1)):
        nudged_value += (1 << (n_bits))
            
    return nudged_value

def apply_nudging(tree: _tree.Tree, node_idx: int, n_bits: int) -> None:
    """Post-order traversal that nudges every internal node’s threshold."""
    flag = False
    if tree.children_left[node_idx] != -1:
        apply_nudging(tree, tree.children_left[node_idx], n_bits)
        flag = True
    if tree.children_right[node_idx] != -1:
        apply_nudging(tree, tree.children_right[node_idx], n_bits)
        flag = True
    if flag:    # internal node
        tree.threshold[node_idx] = nudge_threshold_max_n_bits(
            tree.threshold[node_idx], n_bits
        )

# output the tree
def get_lineage(tree, feature_names):
    data = {"features": {}, "paths": [], "classes": list(tree.classes_)}

    thresholds = tree.tree_.threshold
    features   = [feature_names[i] for i in tree.tree_.feature]
    left       = tree.tree_.children_left
    right      = tree.tree_.children_right
    value      = tree.tree_.value

    # -------- helper to climb up from a leaf to the root -----------
    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = [child]          # leaf marker (an int)
        if child in left:
            parent = np.where(left == child)[0].item()
            split  = "l"
        elif child in right:
            parent = np.where(right == child)[0].item()
            split  = "r"
        else:                          # should never happen
            return lineage

        lineage.append((parent, split, thresholds[parent], features[parent]))
        if parent == 0:
            return list(reversed(lineage))
        return recurse(left, right, parent, lineage)

    leaf_ids = np.where(left == -1)[0]             # indices of all leaves
    for path_id, leaf in enumerate(leaf_ids):
        clause = []

        for node in recurse(left, right, leaf):
            if not isinstance(node, tuple):        # skip the leaf marker
                continue

            direction, threshold, feature = node[1], node[2], node[3]
            if direction == "l":
                clause.append(
                    {"feature": feature, "operation": "<=", "value": threshold}
                )
            else:
                clause.append(
                    {"feature": feature, "operation": ">",  "value": threshold}
                )

        class_idx = int(np.argmax(value[leaf][0]))  # use the leaf itself
        data["paths"].append(
            {"conditions": clause, "classification": class_idx, "id": path_id}
        )

    # collect all thresholds per feature
    for i, feat in enumerate(features):
        if tree.tree_.feature[i] != _tree.TREE_UNDEFINED:
            data["features"].setdefault(feat, []).append(thresholds[i])

    return data


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

# ----------------------------------------------------------------------
# 3. load data
# ----------------------------------------------------------------------
df = pd.read_csv(args.input)
X = df.iloc[:, :3].to_numpy()
Y = df.iloc[:, 3].to_numpy()

print(f"dataset size: {len(X)}")

# ----------------------------------------------------------------------
# 4. train the tree
# ----------------------------------------------------------------------
dt = DecisionTreeClassifier(max_depth=args.depth)
dt.fit(X, Y)
print("train accuracy (before nudging):",
      accuracy_score(Y, dt.predict(X)))

if args.nudge:
    nudged_tree = copy.deepcopy(dt.tree_)
    apply_nudging(nudged_tree, 0, args.bits)
    dt.tree_ = nudged_tree
    print(f"nudging enabled, removed bottom {args.bits} bit(s) per threshold")

    print("train accuracy (after  nudging):",
        accuracy_score(Y, dt.predict(X)))

# ----------------------------------------------------------------------
# 5. export
# ----------------------------------------------------------------------
lineage = get_lineage(dt, df.columns[:3])

output_path = Path(args.output)
output_path.write_text(json.dumps(lineage, indent=4, cls=SetEncoder))
print(f"Wrote tree to {output_path.resolve()}")
