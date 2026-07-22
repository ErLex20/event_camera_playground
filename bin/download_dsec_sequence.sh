#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bin/download_dsec_sequence.sh <sequence>

Downloads the DSEC left events, forward optical-flow event GT, and forward
optical-flow timestamps for one sequence, then extracts them into:

  logs/dsec/<sequence>/events_left/
  logs/dsec/<sequence>/optical_flow_forward/

Example:
  bin/download_dsec_sequence.sh zurich_city_02_c

Environment:
  DSEC_ROOT       Output root. Default: <workspace>/logs/dsec
  DSEC_KEEP_ZIPS  Set to 1 to keep downloaded zip files. Default: 0
  WGET_ARGS       Extra arguments passed to wget.
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

case "${1:-}" in
  -h | --help )
    usage
    exit 0
    ;;
esac

seq="$1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$script_dir/.." && pwd)"
dsec_root="${DSEC_ROOT:-$workspace/logs/dsec}"
download_page="https://dsec.ifi.uzh.ch/dsec-datasets/download/"
keep_zips="${DSEC_KEEP_ZIPS:-0}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command not found: $1" >&2
    exit 1
  fi
}

require_cmd wget
require_cmd unzip
require_cmd find
require_cmd grep
require_cmd sed

case "$seq" in
  *[!A-Za-z0-9_]* | "" )
    echo "error: invalid sequence name '$seq'" >&2
    exit 1
    ;;
esac

target="$dsec_root/$seq"
events_dir="$target/events_left"
flow_dir="$target/optical_flow_forward"
download_dir="$target/_downloads"
tmp_dir="$target/.extract_tmp"

mkdir -p "$events_dir" "$flow_dir" "$download_dir"
rm -rf "$tmp_dir"
mkdir -p "$tmp_dir"

cleanup() {
  rm -rf "$tmp_dir"
  if [[ "$keep_zips" != "1" ]]; then
    rm -rf "$download_dir"
  fi
}
trap cleanup EXIT

echo "Fetching DSEC download page..."
page_html="$tmp_dir/download.html"
wget -q -O "$page_html" "$download_page"

find_url() {
  local pattern="$1"
  local url
  url="$(grep -Eo 'https://download\.ifi\.uzh\.ch/[^"]+' "$page_html" \
    | sed 's/&amp;/\&/g' \
    | grep -E "/${seq}/${pattern}$" \
    | head -n 1 || true)"
  printf '%s\n' "$url"
}

events_url="$(find_url "${seq}_events_left\\.zip")"
flow_url="$(find_url "${seq}_optical_flow_forward_event\\.zip")"
timestamps_url="$(find_url "${seq}_optical_flow_forward_timestamps\\.txt")"

missing=0
if [[ -z "$events_url" ]]; then
  echo "error: could not find events_left.zip link for '$seq'" >&2
  missing=1
fi
if [[ -z "$flow_url" ]]; then
  echo "error: could not find optical_flow_forward_event.zip link for '$seq'" >&2
  missing=1
fi
if [[ -z "$timestamps_url" ]]; then
  echo "error: could not find optical_flow_forward_timestamps.txt link for '$seq'" >&2
  missing=1
fi
if [[ "$missing" -ne 0 ]]; then
  echo "       Check that the sequence has forward optical-flow GT on $download_page" >&2
  exit 1
fi

download_file() {
  local url="$1"
  local out="$2"
  local wget_extra=()
  if [[ -n "${WGET_ARGS:-}" ]]; then
    read -r -a wget_extra <<< "$WGET_ARGS"
  fi
  if [[ -s "$out" ]]; then
    if [[ "$out" == *.zip ]]; then
      if unzip -tq "$out" >/dev/null 2>&1; then
        echo "Using existing $(basename "$out")"
        return
      fi
      echo "Resuming incomplete $(basename "$out")"
    else
      echo "Using existing $(basename "$out")"
      return
    fi
  else
    echo "Downloading $(basename "$out")"
  fi
  wget -c "${wget_extra[@]}" -O "$out" "$url"
}

events_zip="$download_dir/${seq}_events_left.zip"
flow_zip="$download_dir/${seq}_optical_flow_forward_event.zip"
timestamps_file="$flow_dir/${seq}_optical_flow_forward_timestamps.txt"

download_file "$events_url" "$events_zip"
download_file "$flow_url" "$flow_zip"
download_file "$timestamps_url" "$timestamps_file"

extract_zip_to_tmp() {
  local zip_path="$1"
  local dst="$2"
  rm -rf "$dst"
  mkdir -p "$dst"
  echo "Extracting $(basename "$zip_path")"
  unzip -q -o "$zip_path" -d "$dst"
}

move_first_match() {
  local src_root="$1"
  local name="$2"
  local dst="$3"
  local found
  found="$(find "$src_root" -type f -name "$name" | head -n 1)"
  if [[ -z "$found" ]]; then
    echo "error: '$name' was not found after extracting $(basename "$src_root")" >&2
    exit 1
  fi
  mv -f "$found" "$dst"
}

events_tmp="$tmp_dir/events_left"
flow_tmp="$tmp_dir/optical_flow_forward"
extract_zip_to_tmp "$events_zip" "$events_tmp"
extract_zip_to_tmp "$flow_zip" "$flow_tmp"

move_first_match "$events_tmp" "events.h5" "$events_dir/events.h5"
move_first_match "$events_tmp" "rectify_map.h5" "$events_dir/rectify_map.h5"

echo "Installing optical-flow PNGs"
find "$flow_tmp" -type f -name '*.png' -exec mv -f {} "$flow_dir/" \;

png_count="$(find "$flow_dir" -maxdepth 1 -type f -name '*.png' | wc -l)"
if [[ "$png_count" -eq 0 ]]; then
  echo "error: no optical-flow PNGs were found after extracting $(basename "$flow_zip")" >&2
  exit 1
fi

echo "Done."
echo "  events:     $events_dir"
echo "  flow:       $flow_dir ($png_count PNGs)"
echo "  timestamps: $timestamps_file"
