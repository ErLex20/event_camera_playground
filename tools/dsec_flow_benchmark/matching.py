from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .flow_io import discover_flow_files
from .timestamps import FlowTiming, find_timestamp_file, load_timestamps


@dataclass(frozen=True)
class FlowRecord:
  path: Path
  timing: FlowTiming | None = None

  @property
  def stem(self) -> str:
    return self.path.stem


@dataclass(frozen=True)
class MatchedPair:
  gt: FlowRecord
  pred: FlowRecord


def build_records(
  flow_dir: str | Path,
  *,
  timestamp_file: str | Path | None = None,
  require_timestamps: bool = False,
) -> list[FlowRecord]:
  files = discover_flow_files(flow_dir)
  if not files:
    raise FileNotFoundError(f"No flow files found in {flow_dir}")

  timestamps = None
  if timestamp_file is not None:
    timestamps = load_timestamps(timestamp_file)
  else:
    auto_timestamp = find_timestamp_file(flow_dir)
    timestamps = load_timestamps(auto_timestamp) if auto_timestamp is not None else None

  if timestamps is None:
    if require_timestamps:
      raise FileNotFoundError(f"No timestamp file found for {flow_dir}")
    return [FlowRecord(path=file) for file in files]

  if len(timestamps) == len(files):
    return [FlowRecord(path=file, timing=timing) for file, timing in zip(files, timestamps)]

  by_index = {
    str(timing.index).zfill(6): timing for timing in timestamps
    if timing.index is not None
  }
  if by_index:
    records = []
    missing = []
    for file in files:
      timing = by_index.get(file.stem)
      if timing is None:
        missing.append(file.name)
      records.append(FlowRecord(path=file, timing=timing))
    if not missing:
      return records

  if require_timestamps:
    raise ValueError(
      f"Timestamp count/index mismatch for {flow_dir}: {len(timestamps)} timestamps, {len(files)} files"
    )
  return [FlowRecord(path=file) for file in files]


def match_records(
  gt_records: list[FlowRecord],
  pred_records: list[FlowRecord],
  *,
  mode: str = "auto",
  timestamp_key: str = "from",
  tolerance_us: int | None = None,
  allow_missing: bool = False,
) -> list[MatchedPair]:
  if mode == "auto":
    gt_stems = {record.stem for record in gt_records}
    pred_stems = {record.stem for record in pred_records}
    if gt_stems.issubset(pred_stems) or (allow_missing and bool(gt_stems & pred_stems)):
      mode = "name"
    elif all(record.timing is not None for record in gt_records + pred_records):
      mode = "timestamp"
    else:
      mode = "order"

  if mode == "name":
    by_name = {record.stem: record for record in pred_records}
    pairs = []
    missing = []
    for gt in gt_records:
      pred = by_name.get(gt.stem)
      if pred is None:
        missing.append(gt.path.name)
        continue
      pairs.append(MatchedPair(gt=gt, pred=pred))
    if missing and not allow_missing:
      raise FileNotFoundError(f"Missing prediction files matching ground-truth names: {missing[:10]}")
    return pairs

  if mode == "order":
    if len(gt_records) != len(pred_records) and not allow_missing:
      raise ValueError(
        f"Order matching requires equal counts unless --allow-missing is used: "
        f"{len(gt_records)} ground-truth, {len(pred_records)} predictions"
      )
    return [MatchedPair(gt=gt, pred=pred) for gt, pred in zip(gt_records, pred_records)]

  if mode == "timestamp":
    return _match_by_timestamp(
      gt_records,
      pred_records,
      timestamp_key=timestamp_key,
      tolerance_us=tolerance_us,
      allow_missing=allow_missing,
    )

  raise ValueError(f"Unknown match mode: {mode}")


def _match_by_timestamp(
  gt_records: list[FlowRecord],
  pred_records: list[FlowRecord],
  *,
  timestamp_key: str,
  tolerance_us: int | None,
  allow_missing: bool,
) -> list[MatchedPair]:
  if not all(record.timing is not None for record in gt_records):
    raise ValueError("Timestamp matching requires ground-truth timestamps")
  if not all(record.timing is not None for record in pred_records):
    raise ValueError("Timestamp matching requires prediction timestamps")

  sorted_preds = sorted(pred_records, key=lambda record: record.timing.key(timestamp_key))  # type: ignore[union-attr]
  pred_keys = [record.timing.key(timestamp_key) for record in sorted_preds if record.timing is not None]
  pairs = []
  missing = []
  used: set[int] = set()

  for gt in gt_records:
    target = gt.timing.key(timestamp_key)  # type: ignore[union-attr]
    idx = _nearest_index(pred_keys, target)
    pred = sorted_preds[idx]
    delta = abs(pred.timing.key(timestamp_key) - target)  # type: ignore[union-attr]
    if idx in used or (tolerance_us is not None and delta > tolerance_us):
      missing.append((gt.path.name, delta))
      continue
    used.add(idx)
    pairs.append(MatchedPair(gt=gt, pred=pred))

  if missing and not allow_missing:
    preview = ", ".join(f"{name} (delta {delta} us)" for name, delta in missing[:10])
    raise FileNotFoundError(f"Could not timestamp-match predictions: {preview}")
  return pairs


def _nearest_index(values: list[int], target: int) -> int:
  if not values:
    raise ValueError("No prediction timestamps available")
  lo = 0
  hi = len(values)
  while lo < hi:
    mid = (lo + hi) // 2
    if values[mid] < target:
      lo = mid + 1
    else:
      hi = mid
  candidates = []
  if lo < len(values):
    candidates.append(lo)
  if lo > 0:
    candidates.append(lo - 1)
  return min(candidates, key=lambda idx: abs(values[idx] - target))
