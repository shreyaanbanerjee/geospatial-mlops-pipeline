#!/usr/bin/env python3
import glob, os, csv
import numpy as np
import rasterio
from rasterio.merge import merge

outs = sorted(glob.glob("outputs/aois/*_pred*.tif") + glob.glob("outputs/aois/*.tif"))
if not outs:
    print("No AOI outputs found under outputs/aois. Exiting.")
    raise SystemExit(1)

print("Found %d per-AOI outputs. Mosaicking..."%len(outs))
srcs = [rasterio.open(p) for p in outs]
mosaic_arr, mosaic_transform = merge(srcs)
mosaic = mosaic_arr[0] if mosaic_arr.shape[0]>0 else mosaic_arr
out_meta = srcs[0].meta.copy()
out_meta.update({
    "height": mosaic.shape[0],
    "width": mosaic.shape[1],
    "transform": mosaic_transform,
    "count": 1,
    "dtype": "float32"
})
os.makedirs("outputs", exist_ok=True)
mosaic_out = "outputs/mosaic_india_existing.tif"
with rasterio.open(mosaic_out, "w", **out_meta) as dst:
    dst.write(mosaic.astype("float32"), 1)
print("Wrote mosaic:", mosaic_out)

# compute simple per-AOI metrics and write CSV
rows=[]
for p,src in zip(outs, srcs):
    arr = src.read(1)
    tr = src.transform
    try:
        px_area = abs(tr.a * tr.e)
    except Exception:
        px_area = None
    changed = int((arr>0.5).sum())
    area_ha = changed * px_area / 10000.0 if px_area else None
    rows.append([os.path.basename(p), changed, px_area, area_ha])
for s in srcs:
    s.close()
csv_out = "outputs/mosaic_existing_metrics.csv"
with open(csv_out,'w',newline='') as f:
    w=csv.writer(f); w.writerow(["aoi_output","changed_pixels","pixel_area_m2","area_ha"]); w.writerows(rows)
print("Wrote metrics CSV:", csv_out)
