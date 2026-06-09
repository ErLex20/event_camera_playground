from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np


DSEC_FLOW_SCALE = 128.0
DSEC_FLOW_OFFSET = 2**15
SUPPORTED_FLOW_SUFFIXES = {".png", ".npy", ".npz", ".flo"}


@dataclass(frozen=True)
class FlowFrame:
  flow: np.ndarray
  valid: np.ndarray
  path: Path
  metadata: Mapping[str, object] = field(default_factory=dict)


def discover_flow_files(directory: str | Path) -> list[Path]:
  directory = Path(directory)
  if not directory.is_dir():
    raise FileNotFoundError(f"Flow directory does not exist: {directory}")
  files = [
    path for path in directory.iterdir()
    if path.is_file() and path.suffix.lower() in SUPPORTED_FLOW_SUFFIXES
  ]
  return sorted(files, key=lambda p: p.name)


def load_flow(path: str | Path, *, kind: str = "auto") -> FlowFrame:
  path = Path(path)
  if kind == "auto":
    suffix = path.suffix.lower()
    if suffix == ".png":
      return load_dsec_png(path)
    if suffix == ".npy":
      return load_npy(path)
    if suffix == ".npz":
      return load_npz(path)
    if suffix == ".flo":
      return load_flo(path)
    raise ValueError(f"Unsupported flow file suffix: {path.suffix}")
  if kind == "dsec_png":
    return load_dsec_png(path)
  if kind == "npy":
    return load_npy(path)
  if kind == "npz":
    return load_npz(path)
  if kind == "flo":
    return load_flo(path)
  raise ValueError(f"Unsupported flow kind: {kind}")


def load_dsec_png(path: str | Path) -> FlowFrame:
  """Load a DSEC/E-RAFT 3-channel 16-bit optical-flow PNG.

  DSEC defines the channel order as RGB: R=flow_x, G=flow_y, B=valid.
  OpenCV reads PNGs as BGR, so this function reverses the channel order before
  applying the E-RAFT conversion:

      flow = (encoded - 2**15) / 128
  """
  path = Path(path)
  image_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
  if image_bgr is None:
    raise ValueError(f"Could not read flow PNG: {path}")
  if image_bgr.dtype != np.uint16 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
    raise ValueError(
      f"DSEC flow PNG must be HxWx3 uint16, got {image_bgr.dtype} {image_bgr.shape}: {path}"
    )

  image_rgb = image_bgr[..., ::-1]
  encoded = image_rgb.astype(np.float32)
  flow = (encoded[..., :2] - DSEC_FLOW_OFFSET) / DSEC_FLOW_SCALE
  valid = image_rgb[..., 2] == 1
  return FlowFrame(
    flow=flow.astype(np.float32, copy=False),
    valid=valid,
    path=path,
    metadata={"format": "dsec_png"},
  )


def write_dsec_png(path: str | Path, flow: np.ndarray, valid: np.ndarray | None = None) -> None:
  """Write a DSEC-compatible 3-channel 16-bit PNG using OpenCV safely."""
  path = Path(path)
  flow = coerce_flow_array(flow, source=path).flow
  h, w, _ = flow.shape
  if valid is None:
    valid = np.ones((h, w), dtype=bool)
  valid = np.asarray(valid, dtype=bool)
  if valid.shape != (h, w):
    raise ValueError(f"valid mask must have shape {(h, w)}, got {valid.shape}")

  image_rgb = np.zeros((h, w, 3), dtype=np.uint16)
  encoded = np.rint(flow * DSEC_FLOW_SCALE + DSEC_FLOW_OFFSET)
  image_rgb[..., :2] = np.clip(encoded, 0, np.iinfo(np.uint16).max).astype(np.uint16)
  image_rgb[..., 2] = valid.astype(np.uint16)
  image_bgr = image_rgb[..., ::-1]
  path.parent.mkdir(parents=True, exist_ok=True)
  if not cv2.imwrite(str(path), image_bgr):
    raise OSError(f"Could not write DSEC flow PNG: {path}")


def load_npy(path: str | Path) -> FlowFrame:
  path = Path(path)
  array = np.load(path, allow_pickle=False)
  return coerce_flow_array(array, source=path)


def load_npz(path: str | Path) -> FlowFrame:
  path = Path(path)
  with np.load(path, allow_pickle=False) as data:
    flow_key = _first_existing_key(data, ("flow", "uv", "arr_0"))
    valid_key = _first_existing_key(data, ("valid", "mask", "gt_valid_mask"), required=False)
    if flow_key is None:
      raise ValueError(f"NPZ flow file needs one of keys flow, uv, arr_0: {path}")
    valid = data[valid_key] if valid_key is not None else None
    return coerce_flow_array(data[flow_key], valid=valid, source=path)


def load_flo(path: str | Path) -> FlowFrame:
  """Load a Middlebury .flo file."""
  path = Path(path)
  with path.open("rb") as handle:
    magic = np.fromfile(handle, np.float32, count=1)
    if magic.size != 1 or magic[0] != 202021.25:
      raise ValueError(f"Invalid .flo magic number: {path}")
    width = int(np.fromfile(handle, np.int32, count=1)[0])
    height = int(np.fromfile(handle, np.int32, count=1)[0])
    data = np.fromfile(handle, np.float32, count=2 * width * height)
  if data.size != 2 * width * height:
    raise ValueError(f"Truncated .flo file: {path}")
  flow = data.reshape((height, width, 2))
  valid = np.isfinite(flow).all(axis=-1)
  return FlowFrame(flow=flow, valid=valid, path=path, metadata={"format": "flo"})


def coerce_flow_array(
  array: np.ndarray,
  *,
  valid: np.ndarray | None = None,
  source: str | Path | None = None,
) -> FlowFrame:
  path = Path(source) if source is not None else Path("<array>")
  array = np.asarray(array)
  if array.ndim == 4 and array.shape[0] == 1:
    array = array[0]

  if array.ndim != 3:
    raise ValueError(f"Flow array must be 3-D, got shape {array.shape}: {path}")

  inferred_valid = None
  if array.shape[-1] in (2, 3):
    flow = array[..., :2]
    if array.shape[-1] == 3:
      inferred_valid = _looks_like_mask(array[..., 2])
  elif array.shape[0] in (2, 3):
    flow = np.moveaxis(array[:2], 0, -1)
    if array.shape[0] == 3:
      inferred_valid = _looks_like_mask(array[2])
  else:
    raise ValueError(
      "Flow array must have channels as HxWx2, HxWx3, 2xHxW, or 3xHxW; "
      f"got {array.shape}: {path}"
    )

  flow = flow.astype(np.float32, copy=False)
  finite = np.isfinite(flow).all(axis=-1)
  if valid is not None:
    valid_mask = np.asarray(valid, dtype=bool)
  elif inferred_valid is not None:
    valid_mask = inferred_valid
  else:
    valid_mask = finite

  if valid_mask.shape != flow.shape[:2]:
    raise ValueError(f"valid mask shape {valid_mask.shape} does not match flow shape {flow.shape}: {path}")
  return FlowFrame(
    flow=flow,
    valid=valid_mask & finite,
    path=path,
    metadata={"format": path.suffix.lower().lstrip(".") or "array"},
  )


def _first_existing_key(data: Mapping[str, object], keys: tuple[str, ...], *, required: bool = True) -> str | None:
  for key in keys:
    if key in data:
      return key
  if required:
    raise KeyError(f"Missing required keys: {keys}")
  return None


def _looks_like_mask(array: np.ndarray) -> np.ndarray | None:
  arr = np.asarray(array)
  if arr.dtype == np.bool_:
    return arr
  finite = arr[np.isfinite(arr)]
  if finite.size == 0:
    return None
  unique = np.unique(finite)
  if unique.size <= 2 and set(unique.tolist()).issubset({0, 1, 0.0, 1.0}):
    return arr.astype(bool)
  return None
