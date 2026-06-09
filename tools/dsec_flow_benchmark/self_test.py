from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from .flow_io import load_dsec_png, write_dsec_png
from .metrics import evaluate_flow_pair


class FlowIoTests(unittest.TestCase):
  def test_dsec_png_round_trip(self) -> None:
    flow = np.zeros((5, 7, 2), dtype=np.float32)
    flow[..., 0] = np.linspace(-2.0, 2.0, 7, dtype=np.float32)
    flow[..., 1] = np.linspace(1.0, -1.0, 5, dtype=np.float32)[:, None]
    valid = np.ones((5, 7), dtype=bool)
    valid[0, 0] = False

    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "000001.png"
      write_dsec_png(path, flow, valid)
      loaded = load_dsec_png(path)

    self.assertEqual(loaded.flow.shape, flow.shape)
    self.assertTrue(np.array_equal(loaded.valid, valid))
    self.assertLessEqual(float(np.max(np.abs(loaded.flow - flow))), 0.5 / 128.0 + 1e-6)

  def test_identical_flow_has_zero_error(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "000001.png"
      flow = np.ones((4, 4, 2), dtype=np.float32)
      valid = np.ones((4, 4), dtype=bool)
      write_dsec_png(path, flow, valid)
      frame = load_dsec_png(path)
      evaluation = evaluate_flow_pair(frame, frame)

    self.assertAlmostEqual(evaluation.metrics["epe"], 0.0, places=7)
    self.assertAlmostEqual(evaluation.metrics["1pe_pct"], 0.0, places=7)


if __name__ == "__main__":
  unittest.main()
