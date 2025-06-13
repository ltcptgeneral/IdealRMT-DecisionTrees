import json
from pathlib import Path

for file in Path("results/compressed_tree/").glob("*.json"):
    with open(file, "r") as f:
        s = json.load(f)
        print(max(s["paths"])+1)