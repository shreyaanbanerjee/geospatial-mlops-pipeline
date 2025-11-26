import numpy as np
from typing import Tuple
import rasterio


def sliding_windows(image: np.ndarray, tile_size: int, overlap: int):
    """
    Yield windows and tiles.

    Returns:
        window: (row, col, h, w)
        tile_chw: CHW numpy array padded to tile_size
    """
    # Convert CHW -> HWC if needed
    if image.ndim == 3 and image.shape[0] <= 6:
        img = np.transpose(image, (1, 2, 0))
    else:
        img = image

    H, W, C = img.shape
    step = tile_size - overlap

    for r in range(0, max(1, H - overlap), step):
        for c in range(0, max(1, W - overlap), step):
            h = min(tile_size, H - r)
            w = min(tile_size, W - c)

            tile = img[r:r+h, c:c+w, :]

            # pad smaller tiles
            if h < tile_size or w < tile_size:
                pad = np.zeros((tile_size, tile_size, C), dtype=tile.dtype)
                pad[:h, :w, :] = tile
                tile = pad

            # return CHW
            tile_chw = np.transpose(tile, (2, 0, 1))
            yield (r, c, h, w), tile_chw


def stitch_tiles(windows, outputs, out_shape):
    """
    Combine tiles back into full-resolution raster.
    """
    H, W = out_shape
    acc = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)

    for (r, c, h, w), tile in zip(windows, outputs):
        crop = tile[:h, :w]
        acc[r:r+h, c:c+w] += crop
        cnt[r:r+h, c:c+w] += 1

    cnt[cnt == 0] = 1
    return acc / cnt


def read_geotiff(path):
    with rasterio.open(path) as ds:
        arr = ds.read().astype("float32")
        meta = ds.meta.copy()
    return arr, meta


def write_geotiff(path, arr2d, meta):
    """
    Write a single-channel float32 GeoTIFF.
    """
    meta2 = meta.copy()
    meta2.update({"count": 1, "dtype": "float32"})

    with rasterio.open(path, "w", **meta2) as ds:
        ds.write(arr2d.astype("float32"), 1)
