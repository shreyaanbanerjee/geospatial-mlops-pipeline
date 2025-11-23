#!/usr/bin/env python3
"""
Rasterize a GeoJSON polygon (or featurecollection) into a mask aligned with a reference raster.

Usage:
python tools/rasterize_geojson.py --before data/raw/..._before.tif --geojson labels/wayanad.geojson --out data/raw/wayanad_mask.tif
"""
import argparse, json
import rasterio
from rasterio.features import rasterize

p = argparse.ArgumentParser()
p.add_argument("--before", required=True)
p.add_argument("--geojson", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()

with rasterio.open(args.before) as src:
    meta = src.profile.copy()
    meta.update(count=1, dtype='uint8', nodata=0, compress='lzw')
    gj = json.load(open(args.geojson))
    feats = gj.get("features", [gj])
    geoms = [(f["geometry"], 1) for f in feats]
    mask = rasterize(
        geoms,
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype='uint8'
    )
    with rasterio.open(args.out, "w", **meta) as dst:
        dst.write(mask, 1)
print("[OK] wrote", args.out)
