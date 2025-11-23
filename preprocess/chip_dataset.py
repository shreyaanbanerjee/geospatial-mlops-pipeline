#!/usr/bin/env python3
"""
Chip large before/after/mask rasters into tiles (256x256 default).

Produces files:
  out_dir/tile_00000_before.tif
  out_dir/tile_00000_after.tif
  out_dir/tile_00000_mask.tif
"""
import argparse, os
import rasterio
from rasterio.windows import Window

p = argparse.ArgumentParser()
p.add_argument("--before", required=True)
p.add_argument("--after", required=True)
p.add_argument("--mask", required=True)
p.add_argument("--tile-size", type=int, default=256)
p.add_argument("--stride", type=int, default=256)
p.add_argument("--out-dir", required=True)
args = p.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

with rasterio.open(args.before) as bsrc, rasterio.open(args.after) as asrc, rasterio.open(args.mask) as msrc:
    assert bsrc.crs == asrc.crs == msrc.crs, "CRS mismatch"
    assert bsrc.width == asrc.width == msrc.width, "Width mismatch"
    assert bsrc.height == asrc.height == msrc.height, "Height mismatch"

    n=0
    for yi in range(0, bsrc.height - args.tile_size + 1, args.stride):
        for xi in range(0, bsrc.width - args.tile_size + 1, args.stride):
            win = Window(xi, yi, args.tile_size, args.tile_size)
            before = bsrc.read(window=win)
            after  = asrc.read(window=win)
            mask   = msrc.read(1, window=win)
            base = os.path.join(args.out_dir, f"tile_{n:05d}")
            meta = bsrc.profile.copy()
            meta.update(width=args.tile_size, height=args.tile_size, transform=bsrc.window_transform(win))
            with rasterio.open(base + "_before.tif", "w", **meta) as dst:
                dst.write(before)
            with rasterio.open(base + "_after.tif", "w", **meta) as dst:
                dst.write(after)
            meta_mask = meta.copy(); meta_mask.update(count=1, dtype='uint8')
            with rasterio.open(base + "_mask.tif", "w", **meta_mask) as dst:
                dst.write(mask.astype('uint8'), 1)
            n += 1
    print("[OK] wrote", n, "tiles to", args.out_dir)
