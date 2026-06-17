"""Magnitude-correlation stats: predicted vs DSEC-GT flow speed.

Reports, over GT-valid pixels (median across frames):
  ratio_med  : median(pred_speed) / median(gt_speed)   (1.0 == unbiased)
  slope      : least-squares slope pred=slope*gt through origin on speed
  pearson_r  : correlation of pred_speed vs gt_speed
  cos_med    : median cosine similarity of flow vectors (direction agreement)
A good result wants ratio_med and slope near 1.0 (no under-estimation),
pearson_r and cos_med high.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from .flow_io import discover_flow_files, load_flow


def _idx(p: str) -> str:
  m = re.findall(r"(\d+)", os.path.basename(p))
  return m[-1] if m else os.path.basename(p)


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--gt-dir", required=True)
  ap.add_argument("--pred-dir", required=True)
  ap.add_argument("--output-json", default=None)
  args = ap.parse_args(argv)

  gts = {_idx(str(p)): p for p in discover_flow_files(args.gt_dir)}
  ratios, slopes, rs, coss = [], [], [], []
  for pr in discover_flow_files(args.pred_dir):
    k = _idx(str(pr))
    if k not in gts:
      continue
    g = load_flow(gts[k])
    r = load_flow(pr)
    v = g.valid & np.isfinite(r.flow[..., 0]) & np.isfinite(r.flow[..., 1])
    if v.sum() < 100:
      continue
    gs = np.hypot(g.flow[..., 0], g.flow[..., 1])[v]
    rs_ = np.hypot(r.flow[..., 0], r.flow[..., 1])[v]
    ratios.append(np.median(rs_) / max(np.median(gs), 1e-6))
    slopes.append(float(np.dot(rs_, gs) / max(np.dot(gs, gs), 1e-6)))
    if gs.std() > 1e-6 and rs_.std() > 1e-6:
      rs.append(float(np.corrcoef(rs_, gs)[0, 1]))
    gv = g.flow[v]
    rv = r.flow[v]
    num = (gv * rv).sum(-1)
    den = np.linalg.norm(gv, axis=-1) * np.linalg.norm(rv, axis=-1)
    ok = den > 1e-6
    coss.append(float(np.median(num[ok] / den[ok])))

  out = {
    "frames": len(ratios),
    "ratio_med": float(np.median(ratios)) if ratios else float("nan"),
    "slope": float(np.median(slopes)) if slopes else float("nan"),
    "pearson_r": float(np.median(rs)) if rs else float("nan"),
    "cos_med": float(np.median(coss)) if coss else float("nan"),
  }
  print(json.dumps(out, indent=2))
  if args.output_json:
    Path(args.output_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
