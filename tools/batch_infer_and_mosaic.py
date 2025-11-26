#!/usr/bin/env python3
"""
tools/batch_infer_and_mosaic.py

Run infer_and_stitch over all before/after pairs found under a directory (by filename pattern),
then mosaic all per-AOI outputs into a single raster and compute simple metrics.

Usage:
  PYTHONPATH=. python tools/batch_infer_and_mosaic.py \
    --pairs-dir data/raw \
    --before-suffix "_before" \
    --after-suffix "_after" \
    --out-dir outputs/aois \
    --mosaic-out outputs/mosaic_india.tif \
    --threshold 0.5 \
    --tile-size 256 --overlap 32 --batch-size 4
"""
import argparse
import glob
import os
import subprocess
import sys
import time
import csv

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import transform_bounds

def find_pairs(pairs_dir, before_suffix, after_suffix):
    # find files matching *_before*.tif and their after counterpart
    before_files = sorted(glob.glob(os.path.join(pairs_dir, f"*{before_suffix}*.tif")))
    pairs = []
    for bf in before_files:
        # attempt to find the corresponding after by replacing suffix
        base = os.path.basename(bf)
        af = base.replace(before_suffix, after_suffix)
        af_path = os.path.join(pairs_dir, af)
        if os.path.exists(af_path):
            pairs.append((bf, af_path))
        else:
            # fallback: try pattern match using shared prefix up to suffix
            prefix = base.split(before_suffix)[0]
            candidates = sorted(glob.glob(os.path.join(pairs_dir, prefix + "*" + after_suffix + "*.tif")))
            if candidates:
                pairs.append((bf, candidates[0]))
            else:
                print(f"[WARN] No match for before file {bf}")
    return pairs

def run_infer_for_pair(before, after, model, out_dir, tile_size, overlap, batch_size, threshold, verbose=False):
    # produce an output filename based on before filename
    name = os.path.basename(before).replace(".tif", "").replace("_before", "").strip("_")
    out_path = os.path.join(out_dir, f"{name}_pred.tif")
    cmd = [
        sys.executable, "tools/infer_and_stitch.py",
        "--model", model,
        "--before", before,
        "--after", after,
        "--out", out_path,
        "--tile-size", str(tile_size),
        "--overlap", str(overlap),
        "--batch-size", str(batch_size),
    ]
    if threshold is not None:
        cmd += ["--threshold", str(threshold)]
    if verbose:
        cmd.append("--verbose")
    env = os.environ.copy()
    # ensure the repo root is on PYTHONPATH so `train.*` imports work
    env["PYTHONPATH"] = env.get("PYTHONPATH","")
    if env["PYTHONPATH"]:
        env["PYTHONPATH"] = os.path.abspath(".") + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = os.path.abspath(".")

    print("[CMD]", " ".join(cmd))
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"[ERROR] inference failed for {before}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        raise SystemExit(1)
    else:
        print(res.stdout)
    return out_path

def compute_pixel_area_m2(ds):
    # approximate pixel area in square meters by reprojecting image bounds to EPSG:3857
    bounds = ds.bounds
    src_crs = ds.crs
    try:
        minx, miny, maxx, maxy = transform_bounds(src_crs, "EPSG:3857", bounds.left, bounds.bottom, bounds.right, bounds.top, densify_pts=21)
    except Exception:
        # fallback: if transform_bounds fails, approximate using original bounds dimensions
        minx, miny, maxx, maxy = bounds.left, bounds.bottom, bounds.right, bounds.top
    width_m = abs(maxx - minx)
    height_m = abs(maxy - miny)
    px_w = width_m / ds.width
    px_h = height_m / ds.height
    return abs(px_w * px_h)

def mosaic_and_metrics(outputs_list, mosaic_out, metrics_csv, threshold):
    srcs = []
    print("[INFO] opening outputs for mosaic:", outputs_list)
    for p in outputs_list:
        src = rasterio.open(p)
        srcs.append(src)
    if not srcs:
        raise SystemExit("No per-AOI outputs to mosaic")
    mosaic_arr, mosaic_transform = merge(srcs)
    # merge returns array with shape (bands, H, W); we expect 1 band
    if mosaic_arr.shape[0] > 1:
        mosaic_single = mosaic_arr[0]
    else:
        mosaic_single = mosaic_arr[0]
    out_meta = srcs[0].meta.copy()
    out_meta.update({
        "height": mosaic_single.shape[0],
        "width": mosaic_single.shape[1],
        "transform": mosaic_transform,
        "count": 1,
        "dtype": "float32"
    })
    os.makedirs(os.path.dirname(mosaic_out) or ".", exist_ok=True)
    with rasterio.open(mosaic_out, "w", **out_meta) as dst:
        dst.write(mosaic_single.astype("float32"), 1)
    print(f"[OK] wrote mosaic to {mosaic_out}")

    # compute per-AOI metrics (changed pixels and area ha). For area estimate we re-open each and compute pixel area
    rows = []
    for p, src in zip(outputs_list, srcs):
        arr = src.read(1)
        changed = int((arr > threshold).sum()) if threshold is not None else int((arr > 0.5).sum())
        pixel_area = compute_pixel_area_m2(src)
        area_ha = changed * pixel_area / 10000.0
        rows.append((os.path.basename(p), changed, pixel_area, area_ha))
    # write CSV
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aoi_output", "changed_pixels", "pixel_area_m2", "area_ha"])
        w.writerows(rows)
    print(f"[OK] wrote metrics to {metrics_csv}")
    for s in srcs:
        s.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-dir", required=True, help="Directory containing before/after files")
    p.add_argument("--before-suffix", default="_before", help="Suffix token in filename indicating BEFORE image")
    p.add_argument("--after-suffix", default="_after", help="Suffix token in filename indicating AFTER image")
    p.add_argument("--model", required=True, help="Path to fused model .pth")
    p.add_argument("--out-dir", default="outputs/aois", help="Where to store per-AOI outputs")
    p.add_argument("--mosaic-out", default="outputs/mosaic_india.tif", help="Final mosaic output")
    p.add_argument("--metrics-csv", default="outputs/mosaic_metrics.csv", help="CSV summary of metrics")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    pairs = find_pairs(args.pairs_dir, args.before_suffix, args.after_suffix)
    if not pairs:
        print("[ERROR] no before/after pairs found in", args.pairs_dir)
        raise SystemExit(1)
    print(f"[INFO] Found {len(pairs)} pairs to process")

    os.makedirs(args.out_dir, exist_ok=True)
    produced = []
    for before, after in pairs:
        try:
            name = os.path.basename(before)
            out_path = os.path.join(args.out_dir, os.path.basename(before).replace(args.before_suffix, "_pred"))
            out_path = os.path.splitext(out_path)[0] + ".tif"
            if args.dry_run:
                print("[DRY] would run inference for:", before, "->", out_path)
                produced.append(out_path)
                continue
            print(f"[INFO] running inference for: {before}")
            out_path = run_infer_for_pair(before, after, args.model, args.out_dir, args.tile_size, args.overlap, args.batch_size, args.threshold, verbose=args.verbose)
            produced.append(out_path)
        except Exception as e:
            print("[ERROR] failed pair", before, after, ":", e)

    # mosaic and metrics
    if produced:
        mosaic_and_metrics(produced, args.mosaic_out, args.metrics_csv, args.threshold)
    else:
        print("[WARN] no outputs produced; skipping mosaic")

if __name__ == "__main__":
    main()
