#!/usr/bin/env python3
"""
Compute baseline stats (per-band mean/std and simple histograms) over a sample of chips.
"""
import argparse, glob, json
import numpy as np
import rasterio

p = argparse.ArgumentParser()
p.add_argument("--chips-dir", required=True)
p.add_argument("--out", required=True)
p.add_argument("--sample", type=int, default=200)
args = p.parse_args()

files = sorted(glob.glob(args.chips_dir + "/*_before.tif"))
if len(files) == 0:
    raise SystemExit("No chips found")
files = files[:args.sample]
means = []
stds = []
for f in files:
    with rasterio.open(f) as ds:
        arr = ds.read().astype('float32')/10000.0
        means.append(arr.reshape(arr.shape[0], -1).mean(axis=1).tolist())
        stds.append(arr.reshape(arr.shape[0], -1).std(axis=1).tolist())

means = np.array(means)
stds = np.array(stds)
out = {
    "band_mean": means.mean(axis=0).tolist(),
    "band_std": stds.mean(axis=0).tolist(),
    "sample_size": len(files)
}
with open(args.out, "w") as f:
    json.dump(out, f, indent=2)
print("[OK] baseline written to", args.out)
