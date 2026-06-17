from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .flow_io import load_flow
from .matching import build_records, match_records


def main(argv: list[str] | None = None) -> int:
  parser = build_arg_parser()
  args = parser.parse_args(argv)
  result = analyze(args)
  print_summary(result)
  return 0


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Diagnose upper-frame high-magnitude flow blobs against DSEC GT validity masks.",
  )
  parser.add_argument("--gt-dir", required=True)
  parser.add_argument("--pred-dir", required=True)
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--pred-format", default="auto", choices=("auto", "dsec_png", "npy", "npz", "flo"))
  parser.add_argument("--mask-mode", default="gt", choices=("gt", "intersection"))
  parser.add_argument("--upper-fraction", type=float, default=0.5)
  parser.add_argument("--high-percentile", type=float, default=95.0)
  parser.add_argument("--match", default="auto", choices=("auto", "name", "order", "timestamp"))
  parser.add_argument("--allow-missing", action="store_true")
  return parser


def analyze(args: argparse.Namespace) -> dict[str, Any]:
  out = Path(args.output_dir)
  out.mkdir(parents=True, exist_ok=True)

  pairs = match_records(
    build_records(args.gt_dir, require_timestamps=False),
    build_records(args.pred_dir, require_timestamps=False),
    mode=args.match,
    allow_missing=args.allow_missing,
  )
  if not pairs:
    raise RuntimeError("No matched flow pairs")

  frames = []
  upper_pred_mags = []
  upper_scored_epes = []
  for pair in pairs:
    frame = load_frame(pair.gt.path, pair.pred.path, args.pred_format, args.upper_fraction)
    frames.append(frame)
    upper_pred_mags.append(frame["pred_mag"][frame["upper_finite"]])
    scored = frame["upper_finite"] & scored_valid(frame, args.mask_mode)
    if scored.any():
      upper_scored_epes.append(frame["epe"][scored])

  pred_threshold = percentile_or_nan(upper_pred_mags, args.high_percentile)
  epe_threshold = percentile_or_nan(upper_scored_epes, args.high_percentile)

  h, w = frames[0]["gt_valid"].shape
  density_count = np.zeros((h, w), dtype=np.float32)
  high_pred_count = np.zeros((h, w), dtype=np.float32)
  high_epe_count = np.zeros((h, w), dtype=np.float32)
  upper_count = np.zeros((h, w), dtype=np.float32)

  aggregate = Counter()
  frame_rows: list[dict[str, Any]] = []
  for frame in frames:
    upper = frame["upper"]
    finite = frame["upper_finite"]
    gt_valid = scored_valid(frame, args.mask_mode) & finite
    high_pred = finite & (frame["pred_mag"] >= pred_threshold)
    high_epe = finite & (frame["epe"] >= epe_threshold)

    density_count += gt_valid.astype(np.float32)
    high_pred_count += high_pred.astype(np.float32)
    high_epe_count += high_epe.astype(np.float32)
    upper_count += upper.astype(np.float32)

    row = summarize_frame(frame, high_pred, high_epe, gt_valid)
    frame_rows.append(row)
    aggregate.add(row)

  maps = {
    "gt_valid_density": density_count / np.maximum(upper_count, 1.0),
    "high_pred_frequency": high_pred_count / np.maximum(upper_count, 1.0),
    "high_epe_frequency": high_epe_count / np.maximum(upper_count, 1.0),
  }
  for name, image in maps.items():
    save_heatmap(out / f"{name}.png", image)
  np.savez_compressed(out / "diagnosis_maps.npz", **maps)

  result = {
    "config": {
      "gt_dir": str(Path(args.gt_dir).resolve()),
      "pred_dir": str(Path(args.pred_dir).resolve()),
      "mask_mode": args.mask_mode,
      "upper_fraction": args.upper_fraction,
      "high_percentile": args.high_percentile,
      "pred_magnitude_threshold_px": pred_threshold,
      "epe_threshold_px": epe_threshold,
      "frames": len(frames),
    },
    "aggregate": aggregate.summary(),
    "frames": frame_rows,
  }

  (out / "blob_diagnosis.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
  write_csv(out / "blob_diagnosis_frames.csv", frame_rows)
  return result


def load_frame(gt_path: Path, pred_path: Path, pred_format: str, upper_fraction: float) -> dict[str, Any]:
  gt = load_flow(gt_path, kind="dsec_png")
  pred = load_flow(pred_path, kind=pred_format)
  if gt.flow.shape != pred.flow.shape:
    raise ValueError(f"Shape mismatch: {gt_path} {gt.flow.shape}, {pred_path} {pred.flow.shape}")

  h, w = gt.valid.shape
  upper_h = int(round(np.clip(upper_fraction, 0.0, 1.0) * h))
  upper = np.zeros((h, w), dtype=bool)
  upper[:upper_h, :] = True
  finite = np.isfinite(gt.flow).all(axis=-1) & np.isfinite(pred.flow).all(axis=-1)
  pred_mag = np.linalg.norm(pred.flow, axis=-1).astype(np.float32)
  gt_mag = np.linalg.norm(gt.flow, axis=-1).astype(np.float32)
  epe = np.linalg.norm(pred.flow - gt.flow, axis=-1).astype(np.float32)
  return {
    "frame": gt_path.name,
    "prediction": pred_path.name,
    "gt_valid": gt.valid & finite,
    "pred_valid": pred.valid & finite,
    "upper": upper,
    "upper_finite": upper & finite,
    "pred_mag": pred_mag,
    "gt_mag": gt_mag,
    "epe": epe,
  }


def scored_valid(frame: dict[str, Any], mask_mode: str) -> np.ndarray:
  if mask_mode == "intersection":
    return frame["gt_valid"] & frame["pred_valid"]
  return frame["gt_valid"]


def summarize_frame(
  frame: dict[str, Any],
  high_pred: np.ndarray,
  high_epe: np.ndarray,
  gt_valid: np.ndarray,
) -> dict[str, Any]:
  upper = frame["upper"]
  scored = upper & gt_valid
  high_pred_upper = upper & high_pred
  high_epe_upper = upper & high_epe
  high_pred_scored = high_pred_upper & gt_valid
  high_epe_scored = high_epe_upper & gt_valid
  row = {
    "frame": frame["frame"],
    "prediction": frame["prediction"],
    "upper_pixels": int(upper.sum()),
    "upper_gt_valid_pixels": int(scored.sum()),
    "upper_gt_valid_pct": pct(scored.sum(), upper.sum()),
    "high_pred_pixels": int(high_pred_upper.sum()),
    "high_pred_inside_gt": int(high_pred_scored.sum()),
    "high_pred_outside_gt": int((high_pred_upper & ~gt_valid).sum()),
    "high_pred_inside_gt_pct": pct(high_pred_scored.sum(), high_pred_upper.sum()),
    "high_epe_pixels": int(high_epe_upper.sum()),
    "high_epe_inside_gt": int(high_epe_scored.sum()),
    "high_epe_outside_gt": int((high_epe_upper & ~gt_valid).sum()),
    "high_epe_inside_gt_pct": pct(high_epe_scored.sum(), high_epe_upper.sum()),
  }
  if scored.any():
    row["upper_scored_epe_mean"] = float(frame["epe"][scored].mean())
    row["upper_scored_pred_mag_mean"] = float(frame["pred_mag"][scored].mean())
    row["upper_scored_gt_mag_mean"] = float(frame["gt_mag"][scored].mean())
    row["upper_scored_pred_mag_sum"] = float(frame["pred_mag"][scored].sum())
    row["upper_scored_gt_mag_sum"] = float(frame["gt_mag"][scored].sum())
  else:
    row["upper_scored_epe_mean"] = float("nan")
    row["upper_scored_pred_mag_mean"] = float("nan")
    row["upper_scored_gt_mag_mean"] = float("nan")
    row["upper_scored_pred_mag_sum"] = 0.0
    row["upper_scored_gt_mag_sum"] = 0.0
  if high_pred_scored.any():
    row["high_pred_scored_epe_mean"] = float(frame["epe"][high_pred_scored].mean())
    row["high_pred_scored_epe_sum"] = float(frame["epe"][high_pred_scored].sum())
  else:
    row["high_pred_scored_epe_mean"] = float("nan")
    row["high_pred_scored_epe_sum"] = 0.0
  row["upper_scored_epe_sum"] = float(frame["epe"][scored].sum()) if scored.any() else 0.0
  return row


class Counter:
  def __init__(self) -> None:
    self.values: dict[str, float] = {}

  def add(self, row: dict[str, Any]) -> None:
    for key, value in row.items():
      if key in {"frame", "prediction"}:
        continue
      if key.endswith("_mean") or key.endswith("_pct"):
        continue
      if isinstance(value, (int, float)) and np.isfinite(value):
        self.values[key] = self.values.get(key, 0.0) + float(value)

  def summary(self) -> dict[str, float]:
    total_high_pred = self.values.get("high_pred_pixels", 0.0)
    total_high_epe = self.values.get("high_epe_pixels", 0.0)
    total_upper = self.values.get("upper_pixels", 0.0)
    total_valid = self.values.get("upper_gt_valid_pixels", 0.0)
    scored_epe_sum = self.values.get("upper_scored_epe_sum", 0.0)
    high_pred_epe_sum = self.values.get("high_pred_scored_epe_sum", 0.0)
    summary = dict(self.values)
    summary["upper_gt_valid_pct"] = pct(total_valid, total_upper)
    summary["high_pred_inside_gt_pct"] = pct(self.values.get("high_pred_inside_gt", 0.0), total_high_pred)
    summary["high_pred_outside_gt_pct"] = pct(self.values.get("high_pred_outside_gt", 0.0), total_high_pred)
    summary["high_epe_inside_gt_pct"] = pct(self.values.get("high_epe_inside_gt", 0.0), total_high_epe)
    summary["high_epe_outside_gt_pct"] = pct(self.values.get("high_epe_outside_gt", 0.0), total_high_epe)
    summary["high_pred_scored_epe_share_pct"] = pct(high_pred_epe_sum, scored_epe_sum)
    summary["upper_scored_epe_mean"] = scored_epe_sum / total_valid if total_valid else float("nan")
    summary["upper_scored_pred_mag_mean"] = (
      self.values.get("upper_scored_pred_mag_sum", 0.0) / total_valid if total_valid else float("nan")
    )
    summary["upper_scored_gt_mag_mean"] = (
      self.values.get("upper_scored_gt_mag_sum", 0.0) / total_valid if total_valid else float("nan")
    )
    summary["high_pred_scored_epe_mean"] = (
      high_pred_epe_sum / self.values.get("high_pred_inside_gt", 0.0)
      if self.values.get("high_pred_inside_gt", 0.0)
      else float("nan")
    )
    return summary


def percentile_or_nan(values: list[np.ndarray], percentile: float) -> float:
  arrays = [v for v in values if v.size > 0]
  if not arrays:
    return float("nan")
  return float(np.percentile(np.concatenate(arrays), percentile))


def pct(num: float, den: float) -> float:
  return float(100.0 * num / den) if den else float("nan")


def save_heatmap(path: Path, value: np.ndarray) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  finite = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
  if finite.max() > finite.min():
    norm = (255.0 * (finite - finite.min()) / (finite.max() - finite.min())).astype(np.uint8)
  else:
    norm = np.zeros_like(finite, dtype=np.uint8)
  color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
  color[finite <= 0.0] = 0
  cv2.imwrite(str(path), color)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  fieldnames = sorted({key for row in rows for key in row})
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def print_summary(result: dict[str, Any]) -> None:
  cfg = result["config"]
  agg = result["aggregate"]
  print("Blob diagnosis")
  print(f"  frames: {cfg['frames']}")
  print(f"  high pred threshold: {cfg['pred_magnitude_threshold_px']:.3f} px")
  print(f"  high epe threshold:  {cfg['epe_threshold_px']:.3f} px")
  print(f"  upper GT valid:      {agg['upper_gt_valid_pct']:.3f}%")
  print(
    "  high pred inside/outside GT: "
    f"{agg['high_pred_inside_gt_pct']:.3f}% / {agg['high_pred_outside_gt_pct']:.3f}%"
  )
  print(
    "  high EPE inside/outside GT:  "
    f"{agg['high_epe_inside_gt_pct']:.3f}% / {agg['high_epe_outside_gt_pct']:.3f}%"
  )
  print(f"  high pred scored EPE share: {agg['high_pred_scored_epe_share_pct']:.3f}%")


if __name__ == "__main__":
  raise SystemExit(main())
