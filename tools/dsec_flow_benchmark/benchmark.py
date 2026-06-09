from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .flow_io import FlowFrame, load_flow
from .matching import build_records, match_records
from .metrics import MetricsAccumulator, evaluate_flow_pair
from .visualize import save_epe_image


def evaluate_directory(args: argparse.Namespace) -> dict[str, Any]:
  gt_records = build_records(args.gt_dir, timestamp_file=args.gt_timestamps, require_timestamps=False)
  pred_records = build_records(args.pred_dir, timestamp_file=args.pred_timestamps, require_timestamps=False)
  pairs = match_records(
    gt_records,
    pred_records,
    mode=args.match,
    timestamp_key=args.timestamp_key,
    tolerance_us=args.timestamp_tolerance_us,
    allow_missing=args.allow_missing,
  )
  if args.max_frames is not None:
    pairs = pairs[:args.max_frames]
  if not pairs:
    raise RuntimeError("No matched flow pairs to evaluate")

  accumulator = MetricsAccumulator()
  frame_rows = []
  for index, pair in enumerate(pairs):
    gt = load_flow(pair.gt.path, kind="dsec_png")
    pred = load_flow(pair.pred.path, kind=args.pred_format)
    if pred.flow.shape != gt.flow.shape:
      if not args.resize_pred:
        raise ValueError(
          f"Prediction shape {pred.flow.shape} does not match GT {gt.flow.shape}: {pair.pred.path}. "
          "Use --resize-pred to resize explicitly."
        )
      pred = resize_prediction(pred, target_shape=gt.flow.shape[:2])

    evaluation = evaluate_flow_pair(
      gt,
      pred,
      timing=pair.gt.timing,
      pred_units=args.pred_units,
      mask_mode=args.mask_mode,
    )
    evaluation.metrics["frame"] = pair.gt.path.name
    evaluation.metrics["prediction"] = pair.pred.path.name
    if pair.gt.timing is not None:
      evaluation.metrics["from_us"] = pair.gt.timing.from_us
      evaluation.metrics["to_us"] = pair.gt.timing.to_us
      evaluation.metrics["dt_us"] = pair.gt.timing.dt_us

    accumulator.add(evaluation)
    public_metrics = {
      key: value for key, value in evaluation.metrics.items()
      if not key.startswith("_")
    }
    frame_rows.append(public_metrics)

    if args.error_dir is not None:
      save_epe_image(
        Path(args.error_dir) / pair.gt.path.name,
        evaluation.epe_map,
        evaluation.valid_mask,
        max_error=args.error_max,
      )
    if args.verbose:
      print(
        f"[{index + 1:04d}/{len(pairs):04d}] {pair.gt.path.name} <- {pair.pred.path.name}: "
        f"EPE={evaluation.metrics['epe']:.4f}, "
        f"AE={evaluation.metrics['ae']:.4f}, "
        f"valid={evaluation.metrics['valid_pixels']}"
      )

  result = {
    "config": {
      "gt_dir": str(Path(args.gt_dir).resolve()),
      "pred_dir": str(Path(args.pred_dir).resolve()),
      "match": args.match,
      "pred_units": args.pred_units,
      "mask_mode": args.mask_mode,
      "frames_evaluated": len(frame_rows),
    },
    "summary": accumulator.summary(),
    "frames": frame_rows,
  }

  if args.output_json is not None:
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
  if args.output_csv is not None:
    write_frame_csv(Path(args.output_csv), frame_rows)
  return result


def resize_prediction(pred: FlowFrame, *, target_shape: tuple[int, int]) -> FlowFrame:
  target_h, target_w = target_shape
  src_h, src_w = pred.flow.shape[:2]
  if src_h <= 0 or src_w <= 0:
    raise ValueError(f"Invalid prediction shape: {pred.flow.shape}")

  resized_flow = cv2.resize(pred.flow, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
  resized_flow[..., 0] *= target_w / src_w
  resized_flow[..., 1] *= target_h / src_h
  resized_valid = cv2.resize(
    pred.valid.astype(np.uint8),
    (target_w, target_h),
    interpolation=cv2.INTER_NEAREST,
  ).astype(bool)
  return FlowFrame(
    flow=resized_flow.astype(np.float32, copy=False),
    valid=resized_valid,
    path=pred.path,
    metadata=pred.metadata | {"resized_from": (src_h, src_w)},
  )


def write_frame_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = sorted({key for row in rows for key in row.keys()})
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def print_summary(result: dict[str, Any]) -> None:
  summary = result["summary"]
  print("DSEC flow benchmark")
  print(f"  frames:       {summary.get('frames', 0)}")
  print(f"  valid pixels: {summary.get('valid_pixels', 0)}")
  if summary.get("valid_pixels", 0) == 0:
    return
  print(f"  EPE:          {summary['epe']:.6f}")
  print(f"  AE:           {summary['ae']:.6f} deg")
  print(f"  1PE / 2PE / 3PE: {summary['1pe_pct']:.3f}% / {summary['2pe_pct']:.3f}% / {summary['3pe_pct']:.3f}%")
  print(f"  Outlier 3px+5%: {summary['outlier_3px_5pct']:.3f}%")


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Benchmark raw optical-flow predictions against DSEC/E-RAFT 16-bit PNG ground truth.",
  )
  parser.add_argument("--gt-dir", required=True, help="Directory containing DSEC ground-truth flow PNG files.")
  parser.add_argument("--pred-dir", required=True, help="Directory containing prediction flow files.")
  parser.add_argument("--gt-timestamps", help="Optional GT timestamp file. Defaults to *timestamp* in --gt-dir.")
  parser.add_argument("--pred-timestamps", help="Optional prediction timestamp file.")
  parser.add_argument(
    "--pred-format",
    default="auto",
    choices=("auto", "dsec_png", "npy", "npz", "flo"),
    help="Prediction file format. Default: infer from suffix.",
  )
  parser.add_argument(
    "--pred-units",
    default="displacement",
    choices=("displacement", "px_per_second", "px_per_ms", "px_per_us"),
    help="Prediction units before comparison. event_detector_cpp velocity dumps should use px_per_second.",
  )
  parser.add_argument(
    "--match",
    default="auto",
    choices=("auto", "name", "order", "timestamp"),
    help="How to match GT and prediction files. Default: name, then timestamp, then order.",
  )
  parser.add_argument(
    "--timestamp-key",
    default="from",
    choices=("from", "to", "midpoint"),
    help="Timestamp field used for timestamp matching.",
  )
  parser.add_argument("--timestamp-tolerance-us", type=int, help="Maximum timestamp match error in microseconds.")
  parser.add_argument(
    "--mask-mode",
    default="gt",
    choices=("gt", "intersection"),
    help="Use DSEC GT valid pixels only, or intersect GT and prediction masks.",
  )
  parser.add_argument("--resize-pred", action="store_true", help="Resize predictions to GT resolution explicitly.")
  parser.add_argument("--allow-missing", action="store_true", help="Evaluate matched files and skip missing ones.")
  parser.add_argument("--max-frames", type=int, help="Evaluate only the first N matched frames.")
  parser.add_argument("--output-json", help="Write aggregate and per-frame metrics to JSON.")
  parser.add_argument("--output-csv", help="Write per-frame metrics to CSV.")
  parser.add_argument("--error-dir", help="Optional directory for per-frame endpoint-error heatmaps.")
  parser.add_argument("--error-max", type=float, default=10.0, help="Endpoint error value mapped to max heatmap color.")
  parser.add_argument("--verbose", action="store_true", help="Print per-frame metrics.")
  return parser


def main(argv: list[str] | None = None) -> int:
  parser = build_arg_parser()
  args = parser.parse_args(argv)
  result = evaluate_directory(args)
  print_summary(result)
  return 0
