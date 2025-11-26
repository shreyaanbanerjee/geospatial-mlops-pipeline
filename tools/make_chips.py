#!/usr/bin/env python3
"""
tools/make_chips.py
Chips all before/after/mask triplets found in data/raw into data/chips/ with names:
tile_00001_before.tif, tile_00001_after.tif, tile_00001_mask.tif (if mask present)
"""
import argparse, glob, os
from pathlib import Path
import rasterio
from rasterio.windows import Window
import math

def chip_one(path, out_dir, tile_size=256, overlap=0, prefix="tile"):
    with rasterio.open(path) as ds:
        W = ds.width; H = ds.height
        meta = ds.meta.copy()
        step = tile_size - overlap
        i = 0
        for y in range(0, H, step):
            for x in range(0, W, step):
                w = min(tile_size, W - x)
                h = min(tile_size, H - y)
                if w <= 0 or h <= 0:
                    continue
                win = Window(x, y, w, h)
                data = ds.read(window=win)
                # skip tiny tiles
                if data.shape[1] < tile_size//4 or data.shape[2] < tile_size//4:
                    continue
                out = out_dir / f"{prefix}_{i:05d}.tif"
                m = meta.copy()
                m.update({"height": h, "width": w, "transform": rasterio.windows.transform(win, ds.transform)})
                with rasterio.open(out, "w", **m) as dst:
                    dst.write(data)
                i += 1
    return i

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-dir", default="data/raw")
    p.add_argument("--out-dir", default="data/chips")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--overlap", type=int, default=32)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(str(Path(args.pairs_dir) / "*before*.tif")))
    count = 0
    for f in files:
        name = Path(f).stem
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        n = chip_one(f, out_dir, tile_size=args.tile_size, overlap=args.overlap, prefix=name)
        print("Chipped", f, "=>", n, "tiles")
        count += n
    print("Total tiles:", count)

if __name__ == '__main__':
    main()
