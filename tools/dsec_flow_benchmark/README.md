# DSEC Flow Benchmark

Small local benchmark for comparing raw optical-flow estimates against DSEC/E-RAFT flow PNG ground truth.

## Why The Decoder Looks Like This

DSEC flow PNG files are 3-channel 16-bit images. In RGB channel order:

```python
flow_x = (I[..., 0] - 2**15) / 128.0
flow_y = (I[..., 1] - 2**15) / 128.0
valid = I[..., 2] == 1
```

OpenCV reads PNGs as BGR, so the package reverses the channel order before decoding. This matches the E-RAFT/DSEC convention.

## Run

From the workspace root:

```bash
python3 -m tools.dsec_flow_benchmark \
  --gt-dir logs/dsec/thun_00_a/optical_flow_foward \
  --pred-dir path/to/raw_predictions \
  --output-json tools/dsec_flow_benchmark/results/thun_00_a.json \
  --output-csv tools/dsec_flow_benchmark/results/thun_00_a_frames.csv
```

Prediction files can be DSEC-style `.png`, `.npy`, `.npz`, or Middlebury `.flo`.

Accepted NumPy shapes are `H,W,2`, `2,H,W`, `H,W,3`, or `3,H,W`. For `.npz`, use key `flow`, `uv`, or `arr_0`; optional validity keys are `valid`, `mask`, or `gt_valid_mask`.

## event_detector_cpp Units

The current `event_detector_cpp` `~/flow_image` topic is a BGR visualization, not raw optical flow. Do not benchmark those color images.

The C++ node can now save raw DSEC-format PNG predictions directly to `logs/moment_flow` when `flow_save_enabled` is true in `src/event_detector_cpp/config/event_detector_cpp.yaml`. The default config aligns saved files to:

```text
logs/dsec/thun_00_a/optical_flow_foward/thun_00_a_optical_flow_forward_timestamps.txt
```

When a timestamp file is configured, `event_detector_cpp` saves exactly one prediction per ground-truth row. The estimator still uses `flow_max_window_ms` events from the start of each row; for example, a 10 ms algorithm window produces one saved prediction every ~100 ms. The saved velocity is encoded as displacement over the DSEC ground-truth interval so filenames and units remain comparable.

Compare those saved predictions with:

```bash
python3 -m tools.dsec_flow_benchmark \
  --gt-dir logs/dsec/thun_00_a/optical_flow_foward \
  --pred-dir logs/moment_flow/thun_00_a \
  --output-json logs/moment_flow/thun_00_a/benchmark.json \
  --output-csv logs/moment_flow/thun_00_a/benchmark_frames.csv
```

If you dump the algorithm's raw velocity field in pixels per second, compare it with:

```bash
python3 -m tools.dsec_flow_benchmark \
  --gt-dir logs/dsec/thun_00_a/optical_flow_foward \
  --pred-dir path/to/raw_velocity_predictions \
  --pred-units px_per_second
```

The tool scales velocity predictions by the DSEC ground-truth interval in the timestamp file before computing metrics. DSEC ground truth is displacement from `from_timestamp_us` to `to_timestamp_us`.

## Metrics

The summary reports DSEC-style endpoint error (`EPE`), angular error (`AE`), and `1PE`, `2PE`, `3PE` percentages over pixels where ground-truth flow is valid. `outlier_3px_5pct` is included as an extra KITTI-style diagnostic.

## Analysis Notebook

Open `tools/dsec_flow_benchmark/analyze_moment_flow.ipynb` to inspect the `logs/moment_flow` benchmark outputs. The notebook refreshes stale metric files when prediction PNGs are present, checks DSEC PNG encoding, plots per-frame errors, shows worst-frame flow/error maps, and runs sign/scale/axis diagnostics to separate save-format bugs from estimator errors.

## Self Test

```bash
python3 -m tools.dsec_flow_benchmark.self_test
```
