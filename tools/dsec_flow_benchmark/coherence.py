"""Tile-direction coherence metric for MomentFlow predictions.

Quantifies the "random colored tiles" defect: on the final tile grid, among
tiles whose flow speed is significant, how far does each tile's direction
deviate from its 4-neighbour mean direction. High deviation == incoherent /
random hues in the flow visualization. Reported in degrees, averaged over
frames (lower is better).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .flow_io import discover_flow_files, load_flow


def tile_grid(flow: np.ndarray, tiles: int) -> np.ndarray:
  """Average flow into a tiles x tiles grid (mean over each tile block)."""
  h, w, _ = flow.shape
  ys = np.linspace(0, h, tiles + 1).astype(int)
  xs = np.linspace(0, w, tiles + 1).astype(int)
  grid = np.zeros((tiles, tiles, 2), dtype=np.float32)
  for j in range(tiles):
    for i in range(tiles):
      block = flow[ys[j]:ys[j + 1], xs[i]:xs[i + 1], :]
      grid[j, i, 0] = float(np.nanmean(block[..., 0])) if block.size else 0.0
      grid[j, i, 1] = float(np.nanmean(block[..., 1])) if block.size else 0.0
  return grid


def frame_disorder(grid: np.ndarray, speed_floor: float) -> tuple[float, float]:
  """Mean neighbour-direction deviation [deg] over significant-speed tiles."""
  tiles = grid.shape[0]
  spd = np.hypot(grid[..., 0], grid[..., 1])
  thr = max(speed_floor, 0.15 * float(np.nanpercentile(spd, 95)) if np.isfinite(spd).any() else speed_floor)
  active = spd > thr
  devs: list[float] = []
  for j in range(tiles):
    for i in range(tiles):
      if not active[j, i]:
        continue
      nx = ny = 0.0
      n = 0
      for dj, di in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        jj, ii = j + dj, i + di
        if 0 <= jj < tiles and 0 <= ii < tiles and active[jj, ii]:
          nx += grid[jj, ii, 0]
          ny += grid[jj, ii, 1]
          n += 1
      if n == 0:
        continue
      v = grid[j, i]
      a = np.hypot(*v)
      b = np.hypot(nx, ny)
      if a < 1e-6 or b < 1e-6:
        continue
      cos = np.clip((v[0] * nx + v[1] * ny) / (a * b), -1.0, 1.0)
      devs.append(float(np.degrees(np.arccos(cos))))
  if not devs:
    return 0.0, 0.0
  return float(np.mean(devs)), float(active.mean())


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--pred-dir", required=True)
  ap.add_argument("--tiles", type=int, default=32)
  ap.add_argument("--speed-floor", type=float, default=0.5)
  ap.add_argument("--output-json", default=None)
  args = ap.parse_args(argv)

  files = discover_flow_files(args.pred_dir)
  disorders, actives = [], []
  for f in files:
    flow = load_flow(f).flow
    d, a = frame_disorder(tile_grid(flow, args.tiles), args.speed_floor)
    disorders.append(d)
    actives.append(a)
  result = {
    "frames": len(files),
    "tile_disorder_deg": float(np.median(disorders)) if disorders else 0.0,
    "tile_disorder_deg_mean": float(np.mean(disorders)) if disorders else 0.0,
    "active_tile_fraction": float(np.median(actives)) if actives else 0.0,
  }
  print(json.dumps(result, indent=2))
  if args.output_json:
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
