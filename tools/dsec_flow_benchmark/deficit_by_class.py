from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .flow_io import load_flow


BINS = (
  ("0-50", 0.0, 50.0),
  ("50-100", 50.0, 100.0),
  ("100-200", 100.0, 200.0),
  ("200-400", 200.0, 400.0),
  (">400", 400.0, math.inf),
)
CLASSES = ("fallback", "aperture", "full")
COLORS = {
  "fallback": "#c44e52",
  "aperture": "#dd8452",
  "full": "#4c72b0",
}


@dataclass
class TileSample:
  mode: str
  bin_label: str
  solve_class: str
  gain: float
  deficit: float


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--gt-dir", type=Path, required=True)
  parser.add_argument("--coarse-run-dir", type=Path, required=True)
  parser.add_argument("--gt-run-dir", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  return parser.parse_args()


def gt_tile_velocity(gt_path: Path, dt_s: float, tiles: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  frame = load_flow(gt_path)
  flow = frame.flow
  valid = frame.valid
  img_h, img_w = flow.shape[:2]
  rows = min(flow.shape[0], img_h)
  cols = min(flow.shape[1], img_w)
  valid = valid[:rows, :cols]
  ys, xs = np.nonzero(valid)
  n_tiles = tiles * tiles
  acc_x = np.zeros(n_tiles, dtype=np.float64)
  acc_y = np.zeros(n_tiles, dtype=np.float64)
  counts = np.zeros(n_tiles, dtype=np.int64)
  if ys.size:
    tx = np.minimum((xs.astype(np.float32) * tiles / max(1, img_w)).astype(np.int64), tiles - 1)
    ty = np.minimum((ys.astype(np.float32) * tiles / max(1, img_h)).astype(np.int64), tiles - 1)
    k = ty * tiles + tx
    inv_dt = 1.0 / dt_s
    np.add.at(acc_x, k, flow[ys, xs, 0].astype(np.float64) * inv_dt)
    np.add.at(acc_y, k, flow[ys, xs, 1].astype(np.float64) * inv_dt)
    np.add.at(counts, k, 1)
  gt_vx = np.zeros(n_tiles, dtype=np.float64)
  gt_vy = np.zeros(n_tiles, dtype=np.float64)
  ok = counts > 0
  gt_vx[ok] = acc_x[ok] / counts[ok]
  gt_vy[ok] = acc_y[ok] / counts[ok]
  return gt_vx, gt_vy, counts


def speed_bin(speed: float) -> str:
  for label, lo, hi in BINS:
    if lo <= speed < hi:
      return label
  return BINS[-1][0]


def read_mode(mode: str, run_dir: Path, gt_dir: Path) -> list[TileSample]:
  debug_dir = run_dir / "debug_tiles"
  paths = sorted(debug_dir.glob("*.csv"))
  if not paths:
    raise FileNotFoundError(f"no tile-debug CSVs in {debug_dir}")
  samples: list[TileSample] = []
  gt_cache: dict[tuple[int, float, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
  for path in paths:
    with path.open("r", newline="") as handle:
      rows = list(csv.DictReader(handle))
    if not rows:
      continue
    file_index = int(rows[0]["file_index"])
    from_us = int(rows[0]["from_us"])
    to_us = int(rows[0]["to_us"])
    dt_s = (to_us - from_us) * 1e-6
    tiles = max(max(int(r["tx"]) for r in rows), max(int(r["ty"]) for r in rows)) + 1
    cache_key = (file_index, dt_s, tiles)
    if cache_key not in gt_cache:
      gt_cache[cache_key] = gt_tile_velocity(gt_dir / f"{file_index:06d}.png", dt_s, tiles)
    gt_vx, gt_vy, gt_count = gt_cache[cache_key]
    for row in rows:
      tx = int(row["tx"])
      ty = int(row["ty"])
      k = ty * tiles + tx
      if gt_count[k] <= 0:
        continue
      cls = row["solve_class"]
      if cls not in CLASSES:
        continue
      pred_vx = float(row["vx"])
      pred_vy = float(row["vy"])
      speed_pred = math.hypot(pred_vx, pred_vy)
      speed_gt = math.hypot(float(gt_vx[k]), float(gt_vy[k]))
      gain = speed_pred / speed_gt if speed_gt > 1e-9 else math.nan
      samples.append(
        TileSample(
          mode=mode,
          bin_label=speed_bin(speed_gt),
          solve_class=cls,
          gain=gain,
          deficit=max(0.0, speed_gt - speed_pred),
        )
      )
  return samples


def summarize(samples: list[TileSample]) -> list[dict[str, object]]:
  total_deficit_by_mode: dict[str, float] = defaultdict(float)
  n_by_mode_bin: dict[tuple[str, str], int] = defaultdict(int)
  grouped: dict[tuple[str, str, str], list[TileSample]] = defaultdict(list)
  modes = sorted({s.mode for s in samples})
  for sample in samples:
    total_deficit_by_mode[sample.mode] += sample.deficit
    n_by_mode_bin[(sample.mode, sample.bin_label)] += 1
    grouped[(sample.mode, sample.bin_label, sample.solve_class)].append(sample)

  rows: list[dict[str, object]] = []
  for mode in modes:
    total_deficit = total_deficit_by_mode[mode]
    for bin_label, _, _ in BINS:
      bin_n = n_by_mode_bin[(mode, bin_label)]
      for cls in CLASSES:
        items = grouped[(mode, bin_label, cls)]
        gains = [s.gain for s in items if math.isfinite(s.gain)]
        summed_deficit = sum(s.deficit for s in items)
        rows.append(
          {
            "mode": mode,
            "bin": bin_label,
            "solve_class": cls,
            "n_tiles": len(items),
            "pop_frac": (len(items) / bin_n) if bin_n else 0.0,
            "median_gain": float(np.median(gains)) if gains else math.nan,
            "summed_deficit": summed_deficit,
            "deficit_share": (summed_deficit / total_deficit) if total_deficit > 0 else 0.0,
          }
        )
  return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fields = ("mode", "bin", "solve_class", "n_tiles", "pop_frac", "median_gain", "summed_deficit", "deficit_share")
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def plot_mode(mode: str, rows: list[dict[str, object]], path: Path) -> None:
  mode_rows = [r for r in rows if r["mode"] == mode]
  by_key = {(r["bin"], r["solve_class"]): r for r in mode_rows}
  x = np.arange(len(BINS))
  labels = [b[0] for b in BINS]
  fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)

  bottom = np.zeros(len(BINS), dtype=float)
  for cls in CLASSES:
    y = np.array([float(by_key[(b, cls)]["deficit_share"]) for b in labels])
    axes[0].bar(x, y, bottom=bottom, color=COLORS[cls], label=cls)
    bottom += y
  axes[0].set_title(f"{mode}: deficit share")
  axes[0].set_xticks(x, labels, rotation=25)
  axes[0].set_ylabel("share of total deficit")
  axes[0].legend(frameon=False)

  for cls in CLASSES:
    y = np.array([float(by_key[(b, cls)]["median_gain"]) for b in labels])
    axes[1].plot(x, y, marker="o", color=COLORS[cls], label=cls)
  axes[1].set_title("median gain")
  axes[1].set_xticks(x, labels, rotation=25)
  axes[1].set_ylabel("speed_pred / speed_gt")
  axes[1].set_ylim(bottom=0.0)

  bottom = np.zeros(len(BINS), dtype=float)
  for cls in CLASSES:
    y = np.array([float(by_key[(b, cls)]["pop_frac"]) for b in labels])
    axes[2].bar(x, y, bottom=bottom, color=COLORS[cls], label=cls)
    bottom += y
  axes[2].set_title("population")
  axes[2].set_xticks(x, labels, rotation=25)
  axes[2].set_ylabel("fraction within bin")
  axes[2].set_ylim(0.0, 1.0)

  for ax in axes:
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlabel("speed_gt bin (px/s)")
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=180)
  plt.close(fig)


def verdict(rows: list[dict[str, object]]) -> str:
  def class_deficit(mode: str, classes: tuple[str, ...], bins: tuple[str, ...] | None = None) -> float:
    return sum(
      float(r["summed_deficit"])
      for r in rows
      if r["mode"] == mode and r["solve_class"] in classes and (bins is None or r["bin"] in bins)
    )

  parts = []
  high_bins = ("200-400", ">400")
  for mode in ("coarse", "gt"):
    total = class_deficit(mode, CLASSES)
    high_total = class_deficit(mode, CLASSES, high_bins)
    by_cls = {cls: class_deficit(mode, (cls,)) for cls in CLASSES}
    high_by_cls = {cls: class_deficit(mode, (cls,), high_bins) for cls in CLASSES}
    top = max(CLASSES, key=lambda c: by_cls[c])
    high_top = max(CLASSES, key=lambda c: high_by_cls[c])
    parts.append(
      f"{mode}: overall {top} {by_cls[top]:.1f} ({100.0 * by_cls[top] / total:.1f}%), "
      f">=200 {high_top} {high_by_cls[high_top]:.1f} ({100.0 * high_by_cls[high_top] / high_total:.1f}%)"
    )

  coarse_fallback = class_deficit("coarse", ("fallback",))
  gt_fallback = class_deficit("gt", ("fallback",))
  coarse_solved = class_deficit("coarse", ("aperture", "full"))
  gt_solved = class_deficit("gt", ("aperture", "full"))
  fallback_delta = 100.0 * (gt_fallback - coarse_fallback) / coarse_fallback if coarse_fallback else math.nan
  solved_delta = 100.0 * (gt_solved - coarse_solved) / coarse_solved if coarse_solved else math.nan
  lever = "reduce/improve fallback tiles" if gt_fallback >= gt_solved else "improve the solved-tile estimate/routing"
  parts.append(
    f"coarse->gt fallback {coarse_fallback:.1f}->{gt_fallback:.1f} ({fallback_delta:+.1f}%), "
    f"solved {coarse_solved:.1f}->{gt_solved:.1f} ({solved_delta:+.1f}%); dominant residual lever: {lever}."
  )
  return "; ".join(parts)


def main() -> int:
  args = parse_args()
  samples = []
  samples.extend(read_mode("coarse", args.coarse_run_dir, args.gt_dir))
  samples.extend(read_mode("gt", args.gt_run_dir, args.gt_dir))
  rows = summarize(samples)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  write_csv(rows, args.output_dir / "part_deficit_by_class.csv")
  for mode in ("coarse", "gt"):
    plot_mode(mode, rows, args.output_dir / f"part_deficit_by_class_{mode}.png")
  print(verdict(rows))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
