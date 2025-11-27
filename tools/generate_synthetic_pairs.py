#!/usr/bin/env python3
"""
Generate synthetic before/after GeoTIFF pairs from AOI polygons.
Creates:
  data/raw/<aoi>_before_2024-01-01_2024-01-31.tif
  data/raw/<aoi>_after_2024-11-01_2024-11-30.tif
  data/raw/<aoi>_mask.tif

They are small (e.g., 2227x2227 can be large) — default size 1024x1024; adjust TILE_SIZE below.
"""
import os, json, math
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import shape, Polygon, mapping
from shapely.affinity import translate, scale
from rasterio.features import rasterize

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)
AOI_DIR = Path("aoi")
TILE_SIZE = 1024    # pixels (H=W). choose 512/1024. Keep small for speed.
PIXEL_SIZE = 10     # meters per pixel (informational in transform)
CRS = "EPSG:4326"   # keep WGS84 to avoid reproj mishaps for demo

def aoi_centroid_to_origin(aoi_geom, size=TILE_SIZE, pixel=PIXEL_SIZE):
    # create an arbitrary transform: top-left at centroid + offset
    geom = shape(aoi_geom)
    lon, lat = geom.centroid.x, geom.centroid.y
    # create transform with top-left shifted by half-size degrees (coarse)
    # Note: using degree units for simplicity; ok for small demo tiles
    west = lon - 0.05
    north = lat + 0.05
    transform = from_origin(west, north, 0.0001, 0.0001)  # small degrees per pixel
    return transform

def write_tif(path, arr, transform, crs):
    # arr shape (C,H,W)
    C, H, W = arr.shape
    meta = {
        "driver": "GTiff",
        "dtype": arr.dtype.name if hasattr(arr.dtype, 'name') else str(arr.dtype),
        "count": C,
        "width": W,
        "height": H,
        "crs": crs,
        "transform": transform,
        "compress": "LZW",
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(arr)
    print("Wrote", path, "shape", arr.shape)

def create_synthetic_pair(aoi_path: Path, name: str):
    with open(aoi_path) as f:
        feat = json.load(f)
    geom = feat["geometry"]
    transform = aoi_centroid_to_origin(geom)

    H = W = TILE_SIZE
    # generate base 'before' image: 3 bands with textured noise + greenery
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    band1 = (rng.normal(loc=0.2, scale=0.03, size=(H,W)) + 0.1).clip(0,1)  # red-ish
    band2 = (rng.normal(loc=0.6, scale=0.05, size=(H,W)) + 0.2).clip(0,1)  # green-ish
    band3 = (rng.normal(loc=0.3, scale=0.04, size=(H,W)) + 0.1).clip(0,1)  # nir-ish

    before = np.stack([band1, band2, band3]).astype("float32")
    # create mask: small patches that will be deforested in "after"
    # draw a couple of random polygons based on AOI centroid
    poly = shape(geom)
    # rasterize a polygon roughly centered and scaled
    scaled = scale(poly, xfact=0.2, yfact=0.2, origin='center')
    # shift one polygon to simulate scattered clearing
    shifted = translate(scaled, xoff=0.01, yoff=-0.01)
    mask = rasterize([mapping(scaled), mapping(shifted)], out_shape=(H,W), transform=transform, default_value=1, dtype='uint8')
    # make after: dim vegetation where mask==1
    after = before.copy()
    after[1] = np.where(mask==1, after[1]*0.4, after[1])  # reduce green band
    after[2] = np.where(mask==1, after[2]*0.6, after[2])  # reduce nir band slightly

    # scale to reflect original units: store as int16 where values ~ 0..10000 for reflectance
    scale_factor = 10000.0
    before_i = (before * scale_factor).astype("int16")
    after_i = (after * scale_factor).astype("int16")
    mask_i = mask.astype("uint8")

    # file names
    before_name = DATA_DIR / f"{name}_before_2024-01-01_2024-01-31.tif"
    after_name  = DATA_DIR / f"{name}_after_2024-11-01_2024-11-30.tif"
    mask_name   = DATA_DIR / f"{name}_mask.tif"

    write_tif(before_name, before_i, transform, CRS)
    write_tif(after_name, after_i, transform, CRS)
    # mask as single band uint8
    write_tif(mask_name, mask_i[np.newaxis, ...], transform, CRS)

def main():
    aoi_dir = Path("aoi")
    if not aoi_dir.exists():
        print("No aoi/ dir found. Run tools/generate_aois.py first.")
        return
    geojsons = sorted(aoi_dir.glob("india_*.geojson"))
    if not geojsons:
        print("No AOIs found in aoi/. Create them first.")
        return
    for g in geojsons:
        name = g.stem
        print("Generating synthetic pair for:", name)
        create_synthetic_pair(g, name)
    print("Done.")

if __name__ == "__main__":
    main()