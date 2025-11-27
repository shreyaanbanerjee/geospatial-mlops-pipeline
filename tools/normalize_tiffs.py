#!/usr/bin/env python3
"""
Normalize all TIFFs under data/raw/ into a safe (C, H, W) float32 format.
Creates backups under data/raw/backup_normalized/.
"""

import os
import glob
import shutil
import numpy as np
import rasterio
from pathlib import Path

RAW_DIR = Path("data/raw")
BACKUP_DIR = RAW_DIR / "backup_normalized"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

def normalize_array(arr):
    """Normalize TIFF array shape to (C, H, W) float32."""
    arr = np.squeeze(arr)

    # Case: (H, W) -> (1, H, W)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]

    # Case: (H, W, C) -> (C, H, W)
    if arr.ndim == 3 and arr.shape[0] not in (1, 2, 3, 4, 8, 12) and arr.shape[-1] in (1, 2, 3, 4, 8, 12):
        arr = arr.transpose(2, 0, 1)

    # Case: >3 dims (Earth Engine sometimes outputs 1,1,H,W)
    if arr.ndim > 3:
        C = int(np.prod(arr.shape[:-2]))
        H, W = arr.shape[-2], arr.shape[-1]
        arr = arr.reshape((C, H, W))

    return arr.astype("float32")

def normalize_tif(path: Path):
    print(f"[INFO] Normalizing: {path}")

    backup_path = BACKUP_DIR / path.name
    shutil.copy2(path, backup_path)
    print(f"[OK] Backup saved: {backup_path}")

    with rasterio.open(path) as ds:
        arr = ds.read()
        meta = ds.meta.copy()

    arr = normalize_array(arr)
    meta.update({
        "count": arr.shape[0],
        "dtype": "float32",
        "height": arr.shape[1],
        "width": arr.shape[2],
    })

    # Save normalized
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(arr)

    print(f"[OK] Normalized and rewritten: {path} -> shape {arr.shape}")

def main():
    tifs = sorted(glob.glob("data/raw/*.tif") + glob.glob("data/raw/**/*.tif", recursive=True))
    if not tifs:
        print("[ERROR] No TIFFs found in data/raw/")
        return

    print(f"[INFO] Found {len(tifs)} TIFFs")

    for tif in tifs:
        if "backup_normalized" in tif:
            continue
        normalize_tif(Path(tif))

    print("[DONE] Normalization complete.")

if __name__ == "__main__":
    main()