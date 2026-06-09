from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .flow_io import FlowFrame
from .timestamps import FlowTiming


@dataclass
class FrameEvaluation:
  metrics: dict[str, float | int | str]
  epe_map: np.ndarray
  valid_mask: np.ndarray


@dataclass
class MetricsAccumulator:
  epe_values: list[np.ndarray] = field(default_factory=list)
  ae_values: list[np.ndarray] = field(default_factory=list)
  gt_mag_values: list[np.ndarray] = field(default_factory=list)
  frame_metrics: list[dict[str, float | int | str]] = field(default_factory=list)

  def add(self, evaluation: FrameEvaluation) -> None:
    valid = evaluation.valid_mask
    if valid.any():
      self.epe_values.append(evaluation.epe_map[valid].astype(np.float32, copy=False))
      self.ae_values.append(evaluation.metrics["_ae_values"])
      self.gt_mag_values.append(evaluation.metrics["_gt_mag_values"])
    public_metrics = {
      key: value for key, value in evaluation.metrics.items()
      if not key.startswith("_")
    }
    self.frame_metrics.append(public_metrics)

  def summary(self) -> dict[str, float | int]:
    epe = _concat_or_empty(self.epe_values)
    ae = _concat_or_empty(self.ae_values)
    gt_mag = _concat_or_empty(self.gt_mag_values)
    if epe.size == 0:
      return {"frames": len(self.frame_metrics), "valid_pixels": 0}
    return summarize_errors(epe=epe, ae=ae, gt_mag=gt_mag) | {
      "frames": len(self.frame_metrics),
      "valid_pixels": int(epe.size),
    }


def evaluate_flow_pair(
  gt: FlowFrame,
  pred: FlowFrame,
  *,
  timing: FlowTiming | None = None,
  pred_units: str = "displacement",
  mask_mode: str = "gt",
) -> FrameEvaluation:
  if gt.flow.shape != pred.flow.shape:
    raise ValueError(f"Flow shapes differ: gt {gt.flow.shape}, pred {pred.flow.shape}")
  if gt.valid.shape != pred.valid.shape:
    raise ValueError(f"Valid masks differ: gt {gt.valid.shape}, pred {pred.valid.shape}")

  pred_flow = convert_prediction_units(pred.flow, timing=timing, units=pred_units)
  finite = np.isfinite(gt.flow).all(axis=-1) & np.isfinite(pred_flow).all(axis=-1)
  if mask_mode == "gt":
    valid = gt.valid & finite
  elif mask_mode == "intersection":
    valid = gt.valid & pred.valid & finite
  else:
    raise ValueError(f"Unknown mask mode: {mask_mode}")

  diff = pred_flow - gt.flow
  epe_map = np.linalg.norm(diff, axis=-1).astype(np.float32)
  epe = epe_map[valid]
  gt_vec = gt.flow[valid]
  pred_vec = pred_flow[valid]
  gt_mag = np.linalg.norm(gt_vec, axis=-1).astype(np.float32)
  ae = angular_error_degrees(gt_vec, pred_vec).astype(np.float32)

  metrics = summarize_errors(epe=epe, ae=ae, gt_mag=gt_mag)
  metrics["valid_pixels"] = int(valid.sum())
  metrics["coverage_pct"] = float(100.0 * valid.mean())
  metrics["_ae_values"] = ae
  metrics["_gt_mag_values"] = gt_mag
  return FrameEvaluation(metrics=metrics, epe_map=epe_map, valid_mask=valid)


def convert_prediction_units(
  pred_flow: np.ndarray,
  *,
  timing: FlowTiming | None,
  units: str,
) -> np.ndarray:
  if units == "displacement":
    return pred_flow
  if timing is None:
    raise ValueError(f"Prediction units {units!r} require timestamps for dt scaling")
  if units == "px_per_second":
    return pred_flow * (timing.dt_us * 1e-6)
  if units == "px_per_ms":
    return pred_flow * (timing.dt_us * 1e-3)
  if units == "px_per_us":
    return pred_flow * timing.dt_us
  raise ValueError(f"Unknown prediction units: {units}")


def angular_error_degrees(gt_flow: np.ndarray, pred_flow: np.ndarray) -> np.ndarray:
  gt_flow = gt_flow.astype(np.float64, copy=False)
  pred_flow = pred_flow.astype(np.float64, copy=False)
  numerator = gt_flow[:, 0] * pred_flow[:, 0] + gt_flow[:, 1] * pred_flow[:, 1] + 1.0
  gt_norm = np.sqrt(gt_flow[:, 0] ** 2 + gt_flow[:, 1] ** 2 + 1.0)
  pred_norm = np.sqrt(pred_flow[:, 0] ** 2 + pred_flow[:, 1] ** 2 + 1.0)
  cosine = numerator / np.maximum(gt_norm * pred_norm, 1e-12)
  return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def summarize_errors(
  *,
  epe: np.ndarray,
  ae: np.ndarray,
  gt_mag: np.ndarray,
) -> dict[str, float | int]:
  if epe.size == 0:
    nan = float("nan")
    return {
      "epe": nan,
      "epe_median": nan,
      "epe_rmse": nan,
      "ae": nan,
      "ae_median": nan,
      "1pe_pct": nan,
      "2pe_pct": nan,
      "3pe_pct": nan,
      "outlier_3px_5pct": nan,
    }

  epe64 = epe.astype(np.float64, copy=False)
  ae64 = ae.astype(np.float64, copy=False)
  relative = epe64 / np.maximum(gt_mag.astype(np.float64, copy=False), 1e-9)
  return {
    "epe": float(epe64.mean()),
    "epe_median": float(np.median(epe64)),
    "epe_rmse": float(np.sqrt(np.mean(epe64**2))),
    "ae": float(ae64.mean()),
    "ae_median": float(np.median(ae64)),
    "1pe_pct": float(100.0 * np.mean(epe64 > 1.0)),
    "2pe_pct": float(100.0 * np.mean(epe64 > 2.0)),
    "3pe_pct": float(100.0 * np.mean(epe64 > 3.0)),
    "outlier_3px_5pct": float(100.0 * np.mean((epe64 > 3.0) & (relative > 0.05))),
  }


def _concat_or_empty(values: Iterable[np.ndarray]) -> np.ndarray:
  values = list(values)
  if not values:
    return np.empty((0,), dtype=np.float32)
  return np.concatenate(values, axis=0)
