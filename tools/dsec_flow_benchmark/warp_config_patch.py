#!/usr/bin/env python3
"""Patch event_detector_cpp.yaml flow_* keys in place, preserving comments.

Usage: warp_config_patch.py CONFIG key=value [key=value ...]
String values (paths, warp mode) are written quoted so YAML 1.1 does not parse
'off'/'on' as booleans. Numeric values are written bare.
"""
from __future__ import annotations

import re
import sys

# flow_warp_compensation MUST be quoted (YAML 1.1 reads bare 'off' as boolean).
# flow_save_output_dir is intentionally left bare: run_flow_test.sh extracts it
# with a sed that would capture surrounding quotes into the path.
STRING_KEYS = {"flow_warp_compensation", "flow_warp_gt_dir"}


def main(argv: list[str]) -> int:
  config = argv[1]
  overrides = dict(kv.split("=", 1) for kv in argv[2:])
  with open(config, "r", encoding="utf-8") as handle:
    text = handle.read()

  for key, value in overrides.items():
    rendered = f'"{value}"' if key in STRING_KEYS else value
    # Replace the value region between 'key:' and an optional trailing comment.
    pattern = re.compile(rf"^(\s*{re.escape(key)}:)[^#\n]*(#.*)?$", re.M)
    if not pattern.search(text):
      raise SystemExit(f"key not found in config: {key}")
    text = pattern.sub(lambda m: f"{m.group(1)} {rendered}      {m.group(2) or ''}".rstrip(), text)

  with open(config, "w", encoding="utf-8") as handle:
    handle.write(text)
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
