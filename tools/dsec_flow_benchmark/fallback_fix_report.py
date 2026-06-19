#!/usr/bin/env python3
"""Report for the fallback-deficit experiment (Run1 oracle / Run2 reg / Run3 baseline).

Per speed_gt bin: pooled median |pred|/|GT| tile ratio, fallback population %,
summed magnitude deficit split fallback vs solved. Per run: EPE / sparse EPE /
AE (from run_flow_test benchmark JSONs) and overall >=30px-equivalent pixel
ratio. Emits one CSV table and one overlay figure part_fallback_fix.png.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .deficit_by_class import BINS, CLASSES, read_mode
from .flow_io import discover_flow_files, load_flow

GT_DIR = Path("logs/dsec/thun_00_a/optical_flow_forward")
RUNS = [
    ("run1_oracle", "logs/moment_flow/warp_fb/run1_oracle"),
    ("run2_reg", "logs/moment_flow/warp_fb/run2_reg"),
    ("run3_baseline", "logs/moment_flow/warp_fb/run3_baseline"),
]
OUT = Path("logs/moment_flow/warp_fb/analysis")
BIN_LABELS = [b[0] for b in BINS]


def pixel_ratio_ge(pred_dir: Path, threshold_px: float = 30.0) -> float:
    gt_by = {p.name: p for p in discover_flow_files(GT_DIR)}
    gts, preds = [], []
    for pp in discover_flow_files(pred_dir):
        if pp.name not in gt_by:
            continue
        g = load_flow(gt_by[pp.name], kind="dsec_png")
        p = load_flow(pp)
        v = g.valid & np.isfinite(g.flow).all(-1) & np.isfinite(p.flow).all(-1)
        if v.any():
            gts.append(g.flow[v])
            preds.append(p.flow[v])
    gt = np.concatenate(gts)
    pred = np.concatenate(preds)
    gm = np.linalg.norm(gt, axis=1)
    pm = np.linalg.norm(pred, axis=1)
    m = gm >= threshold_px
    return float(np.median(pm[m] / np.maximum(gm[m], 1e-6))) if m.any() else float("nan")


def bench(run_dir: Path):
    def epe(name, key="epe"):
        try:
            return json.loads((run_dir / name).read_text())["summary"][key]
        except Exception:
            return float("nan")
    return (epe("dense_benchmark_gt.json"), epe("sparse_benchmark_intersection.json"),
            epe("dense_benchmark_gt.json", "ae"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    curves = {}
    for name, rd in RUNS:
        rd = Path(rd)
        samples = read_mode(name, rd, GT_DIR)
        by_bin = defaultdict(list)
        for s in samples:
            by_bin[s.bin_label].append(s)
        dense_epe, sparse_epe, ae = bench(rd)
        ge30 = pixel_ratio_ge(rd / "dense")
        curve = []
        for b in BIN_LABELS:
            items = by_bin.get(b, [])
            n = len(items)
            gains = [s.gain for s in items if math.isfinite(s.gain)]
            n_fb = sum(1 for s in items if s.solve_class == "fallback")
            def_fb = sum(s.deficit for s in items if s.solve_class == "fallback")
            def_sv = sum(s.deficit for s in items if s.solve_class in ("aperture", "full"))
            ratio = float(np.median(gains)) if gains else float("nan")
            curve.append(ratio)
            rows.append({
                "run": name, "bin": b, "n_tiles": n,
                "ratio_median": ratio,
                "fallback_pct": 100.0 * n_fb / n if n else float("nan"),
                "deficit_fallback": def_fb, "deficit_solved": def_sv,
                "dense_epe": dense_epe, "sparse_epe": sparse_epe, "ae": ae,
                "pixel_ratio_ge30px": ge30,
            })
        curves[name] = curve

    # CSV
    import csv as _csv
    fields = ["run", "bin", "n_tiles", "ratio_median", "fallback_pct",
              "deficit_fallback", "deficit_solved", "dense_epe", "sparse_epe", "ae",
              "pixel_ratio_ge30px"]
    with (OUT / "part_fallback_fix.csv").open("w", newline="") as h:
        w = _csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Figure: overlay of pooled-tile ratio vs speed bin
    x = np.arange(len(BIN_LABELS))
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    for name, _ in RUNS:
        ax.plot(x, curves[name], marker="o", label=name)
    ax.axhline(1.0, color="0.3", ls="--", lw=1)
    ax.axhline(0.9, color="0.6", ls=":", lw=1)
    ax.set_xticks(x, BIN_LABELS, rotation=20)
    ax.set_xlabel("speed_gt bin [px/s]")
    ax.set_ylabel("median tile |pred|/|GT|")
    ax.set_title("Fallback-deficit fix: tile ratio vs GT speed")
    ax.set_ylim(0.0, 1.2)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "part_fallback_fix.png", dpi=160)

    # Console table
    hdr = f"{'run':<14}{'bin':>9}{'ratio':>7}{'fb%':>7}{'def_fb':>11}{'def_sv':>11}"
    print(hdr)
    for r in rows:
        print(f"{r['run']:<14}{r['bin']:>9}{r['ratio_median']:>7.3f}"
              f"{r['fallback_pct']:>7.1f}{r['deficit_fallback']:>11.0f}{r['deficit_solved']:>11.0f}")
    print("\nper-run summary:")
    seen = set()
    for r in rows:
        if r["run"] in seen:
            continue
        seen.add(r["run"])
        print(f"  {r['run']:<14} dense_EPE={r['dense_epe']:.3f} sparse_EPE={r['sparse_epe']:.3f} "
              f"AE={r['ae']:.2f} pixel_ratio_|GT|>=30px={r['pixel_ratio_ge30px']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
