#!/usr/bin/env bash
# Usage: extract_all.sh SOURCE_DIR TARGET_DIR
# For every .tar, .tar.gz, .tgz, .tar.bz2, .tar.xz in SOURCE_DIR:
#   1. Create TARGET_DIR/<name>/
#   2. If TARGET_DIR/<name>/<name>.pcap already exists, skip the archive.
#   3. Otherwise, extract the archive into its own folder.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 SOURCE_DIR TARGET_DIR" >&2
  exit 1
fi

src_dir="$1"
dst_dir="$2"
mkdir -p "$dst_dir"

# Strip common extensions to recover the base name
strip_ext() {
  local n="$1"
  n=${n%.tar.gz}; n=${n%.tgz}; n=${n%.tar.bz2}; n=${n%.tar.xz}; n=${n%.tar}
  echo "$n"
}

shopt -s nullglob
for archive in "$src_dir"/*.tar{,.gz,.bz2,.xz} "$src_dir"/*.tgz; do
  base=$(basename "$archive")
  name=$(strip_ext "$base")
  out_dir="$dst_dir/$name"
  key_file="$out_dir/$name.pcap"

  if [[ -f "$key_file" ]]; then
    echo "Skipping $archive  —  $key_file already present"
    continue
  fi

  echo "Extracting $archive into $out_dir"
  mkdir -p "$out_dir"

  case "$archive" in
    *.tar)          tar -xf "$archive" -C "$out_dir" ;;
    *.tar.gz|*.tgz) tar -xzf "$archive" -C "$out_dir" ;;
    *.tar.bz2)      tar -xjf "$archive" -C "$out_dir" ;;
    *.tar.xz)       tar -xJf "$archive" -C "$out_dir" ;;
    *)              echo "Unknown type: $archive" ;;
  esac
done

echo "All archives processed."
