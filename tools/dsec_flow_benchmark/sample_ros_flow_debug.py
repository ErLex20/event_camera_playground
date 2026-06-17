from __future__ import annotations

import argparse
import atexit
import json
import signal
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Sample event_detector_cpp flow debug topics.")
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--seconds", type=float, default=120.0)
  parser.add_argument("--target-debug-frames", type=int, default=60)
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_arg_parser().parse_args(argv)
  out = Path(args.output_dir)
  out.mkdir(parents=True, exist_ok=True)

  counts = {"dense_debug": 0, "events_debug": 0, "dense_raw": 0, "events_raw": 0}
  stats: list[dict[str, float | int | str]] = []

  def write_summary() -> None:
    (out / "summary.json").write_text(
      json.dumps({"counts": counts, "stats": stats}, indent=2),
      encoding="utf-8",
    )

  atexit.register(write_summary)
  signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
  signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))

  rclpy.init()
  node = rclpy.create_node("flow_debug_sampler")
  bridge = CvBridge()
  qos = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
  )
  keep = {1, 5, 10, 20, 30, 40, 50, 60}

  def debug_cb(name: str):
    def cb(msg: Image) -> None:
      counts[name] += 1
      img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
      hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      value = hsv[..., 2]
      active = value > 0
      row: dict[str, float | int | str] = {"topic": name, "count": counts[name]}
      if active.any():
        vv = value[active].astype(np.float32)
        row |= {
          "v_mean": float(vv.mean()),
          "v_p50": float(np.percentile(vv, 50)),
          "v_p95": float(np.percentile(vv, 95)),
          "active_pct": float(100.0 * active.mean()),
        }
      else:
        row |= {"v_mean": 0.0, "v_p50": 0.0, "v_p95": 0.0, "active_pct": 0.0}
      stats.append(row)
      if counts[name] in keep:
        cv2.imwrite(str(out / f"{name}_{counts[name]:04d}.png"), img)
      if counts[name] % 5 == 0:
        write_summary()
    return cb

  def raw_cb(name: str):
    def cb(msg: Image) -> None:
      counts[name] += 1
      flow = np.asarray(bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"), dtype=np.float32)
      if flow.ndim != 3 or flow.shape[2] < 2:
        return
      finite = np.isfinite(flow[..., 0]) & np.isfinite(flow[..., 1])
      speed = np.linalg.norm(flow[..., :2], axis=-1)
      row: dict[str, float | int | str] = {
        "topic": name,
        "count": counts[name],
        "finite_pct": float(100.0 * finite.mean()),
      }
      if finite.any():
        ss = speed[finite]
        row |= {
          "speed_mean": float(ss.mean()),
          "speed_p50": float(np.percentile(ss, 50)),
          "speed_p95": float(np.percentile(ss, 95)),
          "speed_p99": float(np.percentile(ss, 99)),
          "speed_max": float(ss.max()),
        }
      stats.append(row)
      if counts[name] % 5 == 0:
        write_summary()
    return cb

  node.create_subscription(Image, "/event_detector_cpp/flow_dense_debug", debug_cb("dense_debug"), qos)
  node.create_subscription(Image, "/event_detector_cpp/flow_events_debug", debug_cb("events_debug"), qos)
  node.create_subscription(Image, "/event_detector_cpp/flow_dense", raw_cb("dense_raw"), qos)
  node.create_subscription(Image, "/event_detector_cpp/flow_events", raw_cb("events_raw"), qos)

  start = time.time()
  while time.time() - start < args.seconds and (
    counts["dense_debug"] < args.target_debug_frames or
    counts["events_debug"] < args.target_debug_frames
  ):
    rclpy.spin_once(node, timeout_sec=0.1)

  write_summary()
  make_montages(out)
  node.destroy_node()
  rclpy.shutdown()
  return 0


def make_montages(out: Path) -> None:
  for prefix in ("dense_debug", "events_debug"):
    images = []
    for path in sorted(out.glob(f"{prefix}_*.png")):
      image = cv2.imread(str(path))
      if image is None:
        continue
      images.append(cv2.resize(image, (240, 180), interpolation=cv2.INTER_AREA))
    if not images:
      continue
    rows = []
    for i in range(0, len(images), 4):
      row = images[i:i + 4]
      while len(row) < 4:
        row.append(np.zeros_like(images[0]))
      rows.append(cv2.hconcat(row))
    cv2.imwrite(str(out / f"{prefix}_montage.png"), cv2.vconcat(rows))


if __name__ == "__main__":
  raise SystemExit(main())
