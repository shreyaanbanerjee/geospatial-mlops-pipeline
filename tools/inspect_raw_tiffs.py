
#!/usr/bin/env python3
"""
Inspect all TIFFs under data/raw/ and print:
- shape
- dtype
- CRS
- min/max/mean per band

Useful for debugging issues before running inference.
"""

import glob
import rasterio
import numpy as np

tifs = sorted(glob.glob("data/raw/*.tif") + glob.glob("data/raw/**/*.tif", recursive=True))

if not tifs:
    print("[ERROR] No TIFFs found in data/raw/")
    exit(1)

for p in tifs:
    print("----------------------------------------------------------------")
    print("FILE:", p)
    try:
        with rasterio.open(p) as ds:
            arr = ds.read()
            print(" read() shape:", arr.shape)
            print(" count:", ds.count)
            print(" width, height:", ds.width, ds.height)
            print(" CRS:", ds.crs)
            print(" DTYPE:", ds.dtypes)

            # Band stats
            for i in range(min(ds.count, 4)):
                b = arr[i]
                print(f"  Band {i+1}: min={b.min()} max={b.max()} mean={b.mean():.4f}")

    except Exception as e:
        print("[ERROR]", e)

print("----------------------------------------------------------------")
print("[DONE] inspection complete.")