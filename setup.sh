#!/usr/bin/env bash
# Creates the directory layout:
#   data/
#     tar/
#     pcap/
#     processed/

set -euo pipefail

root="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$root"/data/{tar,pcap,processed,combined}

echo "Directory structure ready under $root/data/"
