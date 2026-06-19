"""MomentFlow magnitude-compression diagnostics for DSEC runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .flow_io import discover_flow_files, load_flow


DEFAULT_BINS = np.array([0, 2, 5, 10, 15, 20, 30, 40, 60, np.inf], dtype=np.float32)


def flow_pairs(gt_dir: str | Path, pred_dir: str | Path) -> list[tuple[Path, Path]]:
  gt_by_name = {path.name: path for path in discover_flow_files(gt_dir)}
  pairs = [(gt_by_name[pred.name], pred) for pred in discover_flow_files(pred_dir) if pred.name in gt_by_name]
  if not pairs:
    raise RuntimeError(f"No same-name GT/prediction pairs: gt={gt_dir} pred={pred_dir}")
  return pairs


def collect_vectors(
  gt_dir: str | Path,
  pred_dir: str | Path,
  *,
  mask_mode: str = "gt",
) -> tuple[np.ndarray, np.ndarray]:
  gt_chunks: list[np.ndarray] = []
  pred_chunks: list[np.ndarray] = []
  for gt_path, pred_path in flow_pairs(gt_dir, pred_dir):
    gt = load_flow(gt_path, kind="dsec_png")
    pred = load_flow(pred_path)
    valid = gt.valid & np.isfinite(gt.flow).all(axis=-1) & np.isfinite(pred.flow).all(axis=-1)
    if mask_mode == "intersection":
      valid &= pred.valid
    if not valid.any():
      continue
    gt_chunks.append(gt.flow[valid].astype(np.float32, copy=False))
    pred_chunks.append(pred.flow[valid].astype(np.float32, copy=False))
  if not gt_chunks:
    raise RuntimeError(f"No valid pixels: gt={gt_dir} pred={pred_dir}")
  return np.concatenate(gt_chunks, axis=0), np.concatenate(pred_chunks, axis=0)


def mean_epe(gt: np.ndarray, pred: np.ndarray, scale: float = 1.0) -> float:
  return float(np.linalg.norm(scale * pred - gt, axis=1).mean())


def fit_epe_scale(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, float]:
  denom = float(np.sum(pred * pred))
  ls_scale = float(np.sum(pred * gt) / denom) if denom > 1e-12 else 1.0
  gt_mag = np.linalg.norm(gt, axis=1)
  pred_mag = np.linalg.norm(pred, axis=1)
  mag_scale = float(np.median(gt_mag) / max(float(np.median(pred_mag)), 1e-9))
  hi = max(4.0, 3.0 * max(ls_scale, 0.0), 3.0 * max(mag_scale, 0.0))
  lo = 0.0

  def objective(s: float) -> float:
    return mean_epe(gt, pred, s)

  for _ in range(5):
    if objective(hi) < objective(0.9 * hi):
      hi *= 2.0
    else:
      break

  gr = (math.sqrt(5.0) - 1.0) / 2.0
  c = hi - gr * (hi - lo)
  d = lo + gr * (hi - lo)
  fc = objective(c)
  fd = objective(d)
  for _ in range(80):
    if fc < fd:
      hi = d
      d = c
      fd = fc
      c = hi - gr * (hi - lo)
      fc = objective(c)
    else:
      lo = c
      c = d
      fc = fd
      d = lo + gr * (hi - lo)
      fd = objective(d)
  scale = 0.5 * (lo + hi)
  return scale, mean_epe(gt, pred, 1.0), objective(scale), ls_scale


def bin_stats(gt: np.ndarray, pred: np.ndarray, bins: np.ndarray = DEFAULT_BINS) -> list[dict[str, Any]]:
  gt_mag = np.linalg.norm(gt, axis=1)
  pred_mag = np.linalg.norm(pred, axis=1)
  rows: list[dict[str, Any]] = []
  for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (gt_mag >= lo) & (gt_mag < hi) & (gt_mag > 1e-6)
    if not mask.any():
      rows.append({"bin": bin_label(lo, hi), "lo": float(lo), "hi": float(hi), "count": 0})
      continue
    gain = pred_mag[mask] / np.maximum(gt_mag[mask], 1e-6)
    rows.append({
      "bin": bin_label(lo, hi),
      "lo": float(lo),
      "hi": float(hi),
      "count": int(mask.sum()),
      "gain_median": float(np.median(gain)),
      "gain_p25": float(np.percentile(gain, 25)),
      "gain_p75": float(np.percentile(gain, 75)),
      "pred_mag_median": float(np.median(pred_mag[mask])),
      "gt_mag_median": float(np.median(gt_mag[mask])),
    })
  return rows


def bin_label(lo: float, hi: float) -> str:
  if np.isinf(hi):
    return f"{lo:g}+"
  return f"{lo:g}-{hi:g}"


def finite_float(value: Any, default: float = float("nan")) -> float:
  try:
    out = float(value)
  except (TypeError, ValueError):
    return default
  return out if math.isfinite(out) else default


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = sorted({key for row in rows for key in row.keys()})
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def plot_gain(rows: list[dict[str, Any]], path: Path, title: str) -> None:
  xs = []
  gains = []
  labels = []
  lo_err = []
  hi_err = []
  for row in rows:
    if row.get("count", 0) <= 0:
      continue
    lo = row["lo"]
    hi = row["hi"]
    xs.append(lo + 5.0 if np.isinf(hi) else 0.5 * (lo + hi))
    gains.append(row["gain_median"])
    labels.append(row["bin"])
    lo_err.append(row["gain_median"] - row["gain_p25"])
    hi_err.append(row["gain_p75"] - row["gain_median"])
  fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
  ax.errorbar(xs, gains, yerr=[lo_err, hi_err], marker="o", capsize=3)
  ax.axhline(1.0, color="0.3", linewidth=1, linestyle="--")
  ax.set_title(title)
  ax.set_xlabel("|GT displacement| bin [px]")
  ax.set_ylabel("median |pred| / |GT|")
  ax.set_xticks(xs)
  ax.set_xticklabels(labels, rotation=30, ha="right")
  ax.grid(True, alpha=0.25)
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=160)
  plt.close(fig)


def run_part1(args: argparse.Namespace) -> int:
  out_dir = Path(args.out_dir)
  gt, pred = collect_vectors(args.gt_dir, args.pred_dir, mask_mode=args.mask_mode)
  scale, epe_before, epe_after, ls_scale = fit_epe_scale(gt, pred)
  rows = bin_stats(gt, pred)
  departure = next((row["bin"] for row in rows if row.get("count", 0) > 0 and row.get("gain_median", 1.0) < 0.8), None)
  result = {
    "pixels": int(gt.shape[0]),
    "epe_before": epe_before,
    "epe_after_global_scale": epe_after,
    "epe_after_over_before": epe_after / epe_before if epe_before > 0 else float("nan"),
    "global_epe_scale": scale,
    "least_squares_component_scale": ls_scale,
    "gain_departure_below_0p8_bin": departure,
    "bins": rows,
  }
  write_json(out_dir / "part1_global_scale.json", result)
  write_csv(out_dir / "part1_gain_bins.csv", rows)
  plot_gain(rows, out_dir / "part1_gain_by_gt_mag.png", "MomentFlow gain by GT magnitude")
  print(json.dumps(result, indent=2))
  return 0


def read_benchmark_epe(run_dir: Path) -> float:
  path = run_dir / "dense_benchmark_gt.json"
  if not path.exists():
    return float("nan")
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
    return finite_float(data.get("summary", {}).get("epe"))
  except Exception:
    return float("nan")


def parse_detector_log(run_dir: Path) -> dict[str, float]:
  path = run_dir / "detector.log"
  out = {
    "final_full_avg": float("nan"),
    "final_aperture_avg": float("nan"),
    "final_fallback_avg": float("nan"),
    "final_fallback_pct": float("nan"),
    "final_support_fallback_avg": float("nan"),
    "final_reject_fallback_avg": float("nan"),
    "final_no_focus_avg": float("nan"),
    "final_no_improve_avg": float("nan"),
  }
  if not path.exists():
    return out
  text = path.read_text(encoding="utf-8", errors="replace")
  final = [
    tuple(map(int, m.groups()))
    for m in re.finditer(r"tiles_final\(full/aperture/fallback\)=(\d+)/(\d+)/(\d+)", text)
  ]
  if final:
    arr = np.asarray(final, dtype=np.float64)
    out["final_full_avg"] = float(arr[:, 0].mean())
    out["final_aperture_avg"] = float(arr[:, 1].mean())
    out["final_fallback_avg"] = float(arr[:, 2].mean())
    denom = float(arr.sum(axis=1).mean())
    out["final_fallback_pct"] = 100.0 * out["final_fallback_avg"] / denom if denom > 0 else float("nan")
  support = [
    tuple(map(int, m.groups()))
    for m in re.finditer(r"timeaware_fallback\(support/reject\)=\d+/\d+ final=(\d+)/(\d+)", text)
  ]
  if support:
    arr = np.asarray(support, dtype=np.float64)
    out["final_support_fallback_avg"] = float(arr[:, 0].mean())
    out["final_reject_fallback_avg"] = float(arr[:, 1].mean())
  detail = [
    tuple(map(int, m.groups()))
    for m in re.finditer(r"reject_causes\(all no_improve/no_focus\)=\d+/\d+ final=(\d+)/(\d+)", text)
  ]
  if detail:
    arr = np.asarray(detail, dtype=np.float64)
    out["final_no_improve_avg"] = float(arr[:, 0].mean())
    out["final_no_focus_avg"] = float(arr[:, 1].mean())
  return out


def large_ratio(gt: np.ndarray, pred: np.ndarray, threshold: float = 30.0) -> float:
  gt_mag = np.linalg.norm(gt, axis=1)
  pred_mag = np.linalg.norm(pred, axis=1)
  mask = gt_mag >= threshold
  if not mask.any():
    return float("nan")
  return float(np.median(pred_mag[mask] / np.maximum(gt_mag[mask], 1e-6)))


def parse_runs(run_args: list[str]) -> list[tuple[str, Path]]:
  runs = []
  for item in run_args:
    if "=" not in item:
      raise ValueError(f"Run arguments must be NAME=PATH, got {item!r}")
    name, path = item.split("=", 1)
    runs.append((name, Path(path)))
  return runs


def plot_gain_overlay(
  series: list[tuple[str, list[dict[str, Any]]]], path: Path, title: str) -> None:
  """Overlay the gain curves g(|GT|) of several modes on one figure."""
  fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
  for name, rows in series:
    xs, gains = [], []
    for row in rows:
      if row.get("count", 0) <= 0:
        continue
      lo, hi = row["lo"], row["hi"]
      xs.append(lo + 5.0 if np.isinf(hi) else 0.5 * (lo + hi))
      gains.append(row["gain_median"])
    if xs:
      ax.plot(xs, gains, marker="o", label=name)
  ax.axhline(1.0, color="0.3", linewidth=1, linestyle="--")
  ax.set_title(title)
  ax.set_xlabel("|GT displacement| bin centre [px]")
  ax.set_ylabel("median |pred| / |GT|")
  ax.set_ylim(0.0, 1.4)
  ax.grid(True, alpha=0.25)
  ax.legend()
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=160)
  plt.close(fig)


def run_part2(args: argparse.Namespace) -> int:
  out_dir = Path(args.out_dir)
  table: list[dict[str, Any]] = []
  overlay: list[tuple[str, list[dict[str, Any]]]] = []
  for name, run_dir in parse_runs(args.run):
    pred_dir = run_dir / "dense"
    gt, pred = collect_vectors(args.gt_dir, pred_dir, mask_mode="gt")
    bins = bin_stats(gt, pred)
    overlay.append((name, bins))
    row: dict[str, Any] = {
      "config": name,
      "run_dir": str(run_dir),
      "pixels": int(gt.shape[0]),
      "epe": read_benchmark_epe(run_dir),
      "ratio_overall": float(np.median(
        np.linalg.norm(pred, axis=1) / np.maximum(np.linalg.norm(gt, axis=1), 1e-6))),
      "ratio_gt_ge_30": large_ratio(gt, pred, 30.0),
    }
    row.update(parse_detector_log(run_dir))
    for stat in bins:
      row[f"ratio_{stat['bin']}"] = stat.get("gain_median", float("nan"))
      row[f"count_{stat['bin']}"] = stat.get("count", 0)
    table.append(row)

  write_csv(out_dir / "part2_ablation_table.csv", table)
  write_json(out_dir / "part2_ablation_table.json", table)
  write_part2_markdown(out_dir / "part2_ablation_table.md", table)
  plot_gain_overlay(overlay, out_dir / "part2_gain_overlay.png", "Warp-compensation gain curves")
  print_table(table)
  return 0


def write_part2_markdown(path: Path, table: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  cols = ["config", "ratio_overall", "ratio_gt_ge_30", "epe", "final_fallback_pct", "final_no_focus_avg"]
  lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
  for row in table:
    values = []
    for col in cols:
      value = row.get(col, "")
      values.append(f"{value:.4g}" if isinstance(value, float) else str(value))
    lines.append("| " + " | ".join(values) + " |")
  lines.extend(component_attribution(table))
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def component_attribution(table: list[dict[str, Any]]) -> list[str]:
  by_name = {row["config"]: row for row in table}
  if "A" not in by_name:
    return []
  base = by_name["A"]
  lines = ["", "Component deltas versus A: positive ratio delta means recovered magnitude."]
  for name, label in (
    ("B", "lower prior"),
    ("C", "regularizer off"),
    ("D", "EMA off"),
    ("E", "prior+reg+EMA off"),
    ("F", "E plus 8x8 tiles"),
    ("G50", "E plus 50 ms"),
    ("G100", "E plus 100 ms"),
  ):
    row = by_name.get(name)
    if row is None:
      continue
    dr = finite_float(row.get("ratio_overall")) - finite_float(base.get("ratio_overall"))
    dl = finite_float(row.get("ratio_gt_ge_30")) - finite_float(base.get("ratio_gt_ge_30"))
    de = finite_float(row.get("epe")) - finite_float(base.get("epe"))
    lines.append(f"- {name} ({label}): overall ratio {dr:+.3f}, |GT|>=30 ratio {dl:+.3f}, EPE {de:+.3f}")
  return lines


def print_table(table: list[dict[str, Any]]) -> None:
  for row in table:
    print(
      f"{row['config']:>4s} ratio={finite_float(row.get('ratio_overall')):.3f} "
      f"large={finite_float(row.get('ratio_gt_ge_30')):.3f} "
      f"EPE={finite_float(row.get('epe')):.3f} "
      f"fallback={finite_float(row.get('final_fallback_pct')):.1f}%"
    )


def read_tile_rows(debug_dir: str | Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for path in sorted(Path(debug_dir).glob("*.csv")):
    with path.open("r", encoding="utf-8", newline="") as handle:
      for row in csv.DictReader(handle):
        parsed: dict[str, Any] = {
          "source": str(path),
          "frame": path.name.replace(".csv", ".png"),
        }
        for key, value in row.items():
          if key in {"solve_class", "fallback_reason"}:
            parsed[key] = value
          elif key in {"file_index", "from_us", "to_us", "tx", "ty", "solve_class_id", "fallback_reason_id", "cell_count"}:
            parsed[key] = int(float(value))
          else:
            parsed[key] = finite_float(value)
        rows.append(parsed)
  if not rows:
    raise RuntimeError(f"No tile-debug CSV rows found in {debug_dir}")
  return rows


def local_divergence(flow: np.ndarray, valid: np.ndarray, x: int, y: int) -> float:
  h, w = valid.shape
  if not (1 <= x < w - 1 and 1 <= y < h - 1):
    return float("nan")
  if not (valid[y, x - 1] and valid[y, x + 1] and valid[y - 1, x] and valid[y + 1, x]):
    return float("nan")
  du_dx = 0.5 * (float(flow[y, x + 1, 0]) - float(flow[y, x - 1, 0]))
  dv_dy = 0.5 * (float(flow[y + 1, x, 1]) - float(flow[y - 1, x, 1]))
  return du_dx + dv_dy


def annotate_tile_rows(rows: list[dict[str, Any]], gt_dir: str | Path) -> list[dict[str, Any]]:
  cache: dict[str, Any] = {}
  max_tx = max(int(row["tx"]) for row in rows)
  tiles = max_tx + 1
  annotated: list[dict[str, Any]] = []
  for row in rows:
    frame = row["frame"]
    if frame not in cache:
      gt = load_flow(Path(gt_dir) / frame, kind="dsec_png")
      cache[frame] = gt
    gt = cache[frame]
    h, w = gt.valid.shape
    x = int(np.clip(round(row["center_x"]), 0, w - 1))
    y = int(np.clip(round(row["center_y"]), 0, h - 1))
    if not gt.valid[y, x] or not np.isfinite(gt.flow[y, x]).all():
      continue
    xs = np.linspace(0, w, tiles + 1).astype(int)
    ys = np.linspace(0, h, tiles + 1).astype(int)
    x0, x1 = xs[int(row["tx"])], xs[int(row["tx"]) + 1]
    y0, y1 = ys[int(row["ty"])], ys[int(row["ty"]) + 1]
    block_valid = gt.valid[y0:y1, x0:x1] & np.isfinite(gt.flow[y0:y1, x0:x1]).all(axis=-1)
    block = gt.flow[y0:y1, x0:x1]
    if block_valid.any():
      vals = block[block_valid]
      mean = vals.mean(axis=0)
      spread = float(np.sqrt(np.mean(np.sum((vals - mean) ** 2, axis=1))))
    else:
      spread = float("nan")
    dt_s = max(1.0, float(row["to_us"] - row["from_us"])) * 1e-6
    pred_disp = np.array([row["vx"] * dt_s, row["vy"] * dt_s], dtype=np.float32)
    gt_vec = gt.flow[y, x].astype(np.float32, copy=False)
    out = dict(row)
    out.update({
      "gt_u": float(gt_vec[0]),
      "gt_v": float(gt_vec[1]),
      "gt_mag": float(np.linalg.norm(gt_vec)),
      "pred_disp_u": float(pred_disp[0]),
      "pred_disp_v": float(pred_disp[1]),
      "center_epe": float(np.linalg.norm(pred_disp - gt_vec)),
      "gt_spread": spread,
      "gt_divergence": local_divergence(gt.flow, gt.valid, x, y),
      "dt_s": dt_s,
      "tile_id": int(row["ty"]) * tiles + int(row["tx"]),
    })
    annotated.append(out)
  if not annotated:
    raise RuntimeError("No tile-debug rows landed on valid GT tile centers")
  return annotated


def grouped_bin_rates(rows: list[dict[str, Any]], bins: np.ndarray = DEFAULT_BINS) -> list[dict[str, Any]]:
  out = []
  gt_mag = np.asarray([row["gt_mag"] for row in rows], dtype=np.float32)
  for lo, hi in zip(bins[:-1], bins[1:]):
    idx = np.where((gt_mag >= lo) & (gt_mag < hi))[0]
    subset = [rows[i] for i in idx]
    if not subset:
      out.append({"bin": bin_label(lo, hi), "lo": float(lo), "hi": float(hi), "count": 0})
      continue
    fallback = np.asarray([row["solve_class"] == "fallback" for row in subset], dtype=np.float32)
    focus = np.asarray([row["fallback_reason"] == "no_focus" for row in subset], dtype=np.float32)
    reject = np.asarray([row["fallback_reason"] in {"no_focus", "no_improve"} for row in subset], dtype=np.float32)
    out.append({
      "bin": bin_label(lo, hi),
      "lo": float(lo),
      "hi": float(hi),
      "count": len(subset),
      "fallback_rate": float(fallback.mean()),
      "focus_reject_rate": float(focus.mean()),
      "candidate_reject_rate": float(reject.mean()),
    })
  return out


def pearson(xs: list[float] | np.ndarray, ys: list[float] | np.ndarray) -> float:
  x = np.asarray(xs, dtype=np.float64)
  y = np.asarray(ys, dtype=np.float64)
  ok = np.isfinite(x) & np.isfinite(y)
  if ok.sum() < 3 or x[ok].std() <= 1e-12 or y[ok].std() <= 1e-12:
    return float("nan")
  return float(np.corrcoef(x[ok], y[ok])[0, 1])


def plot_h1(rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
  bins = grouped_bin_rates(rows)
  xs = [row["lo"] + 5.0 if np.isinf(row["hi"]) else 0.5 * (row["lo"] + row["hi"]) for row in bins if row["count"] > 0]
  labels = [row["bin"] for row in bins if row["count"] > 0]
  fallback = [row["fallback_rate"] for row in bins if row["count"] > 0]
  focus = [row["focus_reject_rate"] for row in bins if row["count"] > 0]
  reject = [row["candidate_reject_rate"] for row in bins if row["count"] > 0]
  fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
  ax.plot(xs, fallback, marker="o", label="fallback")
  ax.plot(xs, focus, marker="o", label="no_focus")
  ax.plot(xs, reject, marker="o", label="no_focus or no_improve")
  ax.set_xticks(xs)
  ax.set_xticklabels(labels, rotation=30, ha="right")
  ax.set_xlabel("|GT displacement| bin [px]")
  ax.set_ylabel("rate")
  ax.set_ylim(0.0, 1.02)
  ax.grid(True, alpha=0.25)
  ax.legend()
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=160)
  plt.close(fig)
  return bins


def scatter_sample(rows: list[dict[str, Any]], n: int = 30000) -> list[dict[str, Any]]:
  if len(rows) <= n:
    return rows
  rng = np.random.default_rng(7)
  idx = rng.choice(len(rows), size=n, replace=False)
  return [rows[int(i)] for i in idx]


def plot_h2(rows: list[dict[str, Any]], path: Path) -> dict[str, float]:
  solved = [row for row in rows if row["solve_class"] != "fallback"]
  sample = scatter_sample(solved)
  spread = np.asarray([row["gt_spread"] for row in sample], dtype=np.float32)
  err = np.asarray([row["center_epe"] for row in sample], dtype=np.float32)
  phi = np.asarray([row["focus_phi"] for row in sample], dtype=np.float32)
  div = np.asarray([abs(row["gt_divergence"]) for row in sample], dtype=np.float32)
  fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
  sc0 = axes[0].scatter(spread, err, c=div, s=5, alpha=0.35, cmap="viridis")
  axes[0].set_xlabel("within-tile GT spread [px]")
  axes[0].set_ylabel("center EPE [px]")
  axes[0].grid(True, alpha=0.25)
  fig.colorbar(sc0, ax=axes[0], label="|GT divergence|")
  axes[1].scatter(spread, phi, c=div, s=5, alpha=0.35, cmap="viridis")
  axes[1].set_xlabel("within-tile GT spread [px]")
  axes[1].set_ylabel("focus phi")
  axes[1].grid(True, alpha=0.25)
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=160)
  plt.close(fig)
  return {
    "solved_tiles": len(solved),
    "corr_error_vs_spread": pearson([row["gt_spread"] for row in solved], [row["center_epe"] for row in solved]),
    "corr_error_vs_abs_divergence": pearson(
      [abs(row["gt_divergence"]) for row in solved],
      [row["center_epe"] for row in solved]),
    "corr_phi_vs_spread": pearson([row["gt_spread"] for row in solved], [row["focus_phi"] for row in solved]),
  }


def plot_h3(rows: list[dict[str, Any]], path: Path) -> dict[str, float]:
  by_tile: dict[int, list[dict[str, Any]]] = defaultdict(list)
  for row in rows:
    by_tile[int(row["tile_id"])].append(row)
  stats = []
  for tile_id, seq in by_tile.items():
    if len(seq) < 3:
      continue
    vx = np.asarray([row["vx"] for row in seq], dtype=np.float64)
    vy = np.asarray([row["vy"] for row in seq], dtype=np.float64)
    stats.append({
      "tile_id": tile_id,
      "temporal_std_px_s": float(np.sqrt(np.var(vx) + np.var(vy))),
      "fallback_rate": float(np.mean([row["solve_class"] == "fallback" for row in seq])),
      "mean_gt_mag": float(np.mean([row["gt_mag"] for row in seq])),
    })
  x_fallback = [row["fallback_rate"] for row in stats]
  x_gt = [row["mean_gt_mag"] for row in stats]
  y_std = [row["temporal_std_px_s"] for row in stats]
  fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
  axes[0].scatter(x_fallback, y_std, s=10, alpha=0.55)
  axes[0].set_xlabel("tile fallback rate")
  axes[0].set_ylabel("temporal std |v| [px/s]")
  axes[0].grid(True, alpha=0.25)
  axes[1].scatter(x_gt, y_std, s=10, alpha=0.55)
  axes[1].set_xlabel("mean |GT displacement| [px]")
  axes[1].set_ylabel("temporal std |v| [px/s]")
  axes[1].grid(True, alpha=0.25)
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=160)
  plt.close(fig)
  return {
    "tiles": len(stats),
    "corr_std_vs_fallback": pearson(x_fallback, y_std),
    "corr_std_vs_gt_mag": pearson(x_gt, y_std),
  }


def run_tiles(args: argparse.Namespace) -> int:
  out_dir = Path(args.out_dir)
  base_rows = annotate_tile_rows(read_tile_rows(args.debug_dir), args.gt_dir)
  h1_bins = plot_h1(base_rows, out_dir / "part3_h1_fallback_vs_gt_mag.png")
  h2 = plot_h2(base_rows, out_dir / "part3_h2_model_breakdown.png")
  write_csv(out_dir / "part3_h1_bins.csv", h1_bins)

  summary: dict[str, Any] = {
    "baseline_debug_rows": len(base_rows),
    "h1_bins": h1_bins,
    "h2": h2,
  }
  if args.raw_debug_dir:
    raw_rows = annotate_tile_rows(read_tile_rows(args.raw_debug_dir), args.gt_dir)
    summary["raw_debug_rows"] = len(raw_rows)
    summary["h3"] = plot_h3(raw_rows, out_dir / "part3_h3_raw_temporal_instability.png")
  write_json(out_dir / "part3_summary.json", summary)
  write_part3_verdict(out_dir / "part3_verdict.md", summary)
  print(json.dumps(summary, indent=2))
  return 0


def write_part3_verdict(path: Path, summary: dict[str, Any]) -> None:
  h1 = [row for row in summary.get("h1_bins", []) if row.get("count", 0) > 0]
  low = h1[0]["fallback_rate"] if h1 else float("nan")
  high = h1[-1]["fallback_rate"] if h1 else float("nan")
  h2 = summary.get("h2", {})
  h3 = summary.get("h3", {})
  lines = [
    f"H1 gating: fallback rises from {low:.3f} in the lowest non-empty speed bin to {high:.3f} in the highest.",
    f"H2 model: corr(error, spread)={finite_float(h2.get('corr_error_vs_spread')):.3f}, corr(error, |div|)={finite_float(h2.get('corr_error_vs_abs_divergence')):.3f}.",
  ]
  if h3:
    lines.append(
      f"H3 instability: corr(temporal std, fallback)={finite_float(h3.get('corr_std_vs_fallback')):.3f}, "
      f"corr(temporal std, |GT|)={finite_float(h3.get('corr_std_vs_gt_mag')):.3f}."
    )
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="cmd", required=True)

  p1 = sub.add_parser("part1", help="Global scale and gain-by-GT-magnitude diagnosis.")
  p1.add_argument("--gt-dir", required=True)
  p1.add_argument("--pred-dir", required=True)
  p1.add_argument("--out-dir", required=True)
  p1.add_argument("--mask-mode", choices=("gt", "intersection"), default="gt")
  p1.set_defaults(func=run_part1)

  p2 = sub.add_parser("part2", help="Summarize ablation run directories.")
  p2.add_argument("--gt-dir", required=True)
  p2.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR. Use A/B/C/D/E/F/G50/G100 names for attribution text.")
  p2.add_argument("--out-dir", required=True)
  p2.set_defaults(func=run_part2)

  p3 = sub.add_parser("tiles", help="Analyze exported per-tile debug CSVs.")
  p3.add_argument("--gt-dir", required=True)
  p3.add_argument("--debug-dir", required=True, help="Baseline/debug run debug_tiles directory for H1/H2.")
  p3.add_argument("--raw-debug-dir", help="Raw config-E debug_tiles directory for H3.")
  p3.add_argument("--out-dir", required=True)
  p3.set_defaults(func=run_tiles)
  return parser


def main(argv: list[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  return args.func(args)


if __name__ == "__main__":
  raise SystemExit(main())
