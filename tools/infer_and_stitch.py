#!/usr/bin/env python3
"""
tools/infer_and_stitch.py

Usage:
  python tools/infer_and_stitch.py \
    --model runs/model_fused.pth \
    --before data/raw/india_wayanad_before.tif \
    --after  data/raw/india_wayanad_after.tif \
    --out outputs/wayanad_change.tif \
    --tile-size 256 --overlap 32 --batch-size 8 --threshold 0.5 --cog

Description:
  Tile a large georeferenced before/after pair, run batched model inference,
  stitch probability outputs into a single raster, optionally threshold to
  a binary mask and save a georeferenced GeoTIFF.
"""
import sys
import argparse
import math
import os
import time
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch, "has_mps", False) and torch.has_mps:
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path: str, device: torch.device):
    from train.model.siamese_unet import SiameseUNet

    model = SiameseUNet(in_ch=6)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def generate_windows(h: int, w: int, tile: int, overlap: int):
    """Yield (row_off, col_off, height, width). Pads not considered here."""
    step = tile - overlap
    if step <= 0:
        raise ValueError("tile_size must be greater than overlap")
    for r in range(0, max(1, h - overlap), step):
        for c in range(0, max(1, w - overlap), step):
            hh = min(tile, h - r)
            ww = min(tile, w - c)
            yield (r, c, hh, ww)


def read_window(ds: rasterio.io.DatasetReader, win: Tuple[int, int, int, int]):
    r, c, h, w = win
    window = Window(c, r, w, h)  # col_off, row_off, width, height
    arr = ds.read(window=window)  # returns bands x h x w (CHW)
    return arr


def pad_tile(tile: np.ndarray, tile_size: int):
    # tile: CHW
    bands, h, w = tile.shape
    if h == tile_size and w == tile_size:
        return tile
    out = np.zeros((bands, tile_size, tile_size), dtype=tile.dtype)
    out[:, :h, :w] = tile
    return out


def stitch_outputs(windows: List[Tuple[int, int, int, int]], outputs: List[np.ndarray], H: int, W: int):
    acc = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    for (r, c, h, w), tile in zip(windows, outputs):
        crop = tile[:h, :w]
        acc[r:r+h, c:c+w] += crop
        cnt[r:r+h, c:c+w] += 1.0
    cnt[cnt == 0] = 1.0
    return acc / cnt


def write_geotiff(path: str, arr2d: np.ndarray, meta: dict, cog: bool = False):
    meta2 = meta.copy()
    meta2.update({
        "count": 1,
        "dtype": "float32",
        "compress": "deflate" if cog else meta2.get("compress", None),
        "tiled": True if cog else meta2.get("tiled", False),
    })
    # Choose block size same as tile if cog
    if cog:
        ts = 256
        meta2.update({"blockxsize": ts, "blockysize": ts})
    # remove None values
    meta2 = {k: v for k, v in meta2.items() if v is not None}
    with rasterio.open(path, "w", **meta2) as ds:
        ds.write(arr2d.astype("float32"), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to fused model .pth")
    p.add_argument("--before", required=True, help="Before GeoTIFF path")
    p.add_argument("--after", required=True, help="After GeoTIFF path")
    p.add_argument("--out", required=True, help="Output GeoTIFF path")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=None, help="If set, output binary mask (0/255)")
    p.add_argument("--cog", action="store_true", help="Write tiled, compressed GeoTIFF (not strict COG)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    device = select_device()
    if args.verbose:
        print("Using device:", device)

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}")

    # open before/after and sanity checks
    with rasterio.open(args.before) as db:
        b_meta = db.meta.copy()
        B, H, W = db.count, db.height, db.width
    with rasterio.open(args.after) as da:
        A, H2, W2 = da.count, da.height, da.width

    if (H, W) != (H2, W2):
        raise SystemExit("Before/After spatial dimensions mismatch")
    if B != A:
        print("Warning: before/after band counts differ. Proceeding with min(B,A)")
    bands = min(B, A)
    if args.verbose:
        print(f"Image size: {W}x{H}, bands per image: {bands}")

    model = load_model(args.model, device)

    windows = list(generate_windows(H, W, args.tile_size, args.overlap))
    if args.verbose:
        print(f"Generated {len(windows)} windows (tile={args.tile_size}, overlap={args.overlap})")

    outputs = []
    batch_tiles_b = []
    batch_tiles_a = []
    batch_windows = []
    t0 = time.time()

    # We'll read directly from files to keep memory low
    with rasterio.open(args.before) as db, rasterio.open(args.after) as da:
        for win in tqdm(windows, desc="Windows"):
            # read CHW arrays
            tile_b = read_window(db, win)[:bands, :, :]
            tile_a = read_window(da, win)[:bands, :, :]

            # normalize: if expected >1 scale, try dividing to 0..1 (assume reflectance *10000)
            # This is heuristic; adjust if your chips are differently scaled.
            tile_b = tile_b.astype("float32") / 10000.0
            tile_a = tile_a.astype("float32") / 10000.0

            # pad to tile_size
            tile_b_p = pad_tile(tile_b, args.tile_size)
            tile_a_p = pad_tile(tile_a, args.tile_size)

            batch_tiles_b.append(tile_b_p)
            batch_tiles_a.append(tile_a_p)
            batch_windows.append(win)

            if len(batch_tiles_b) >= args.batch_size:
                # run batch inference
                tb = torch.from_numpy(np.stack(batch_tiles_b, axis=0)).to(device)  # B x C x H x W
                ta = torch.from_numpy(np.stack(batch_tiles_a, axis=0)).to(device)
                with torch.no_grad():
                    out = model(tb, ta)  # B x 1 x H x W
                    out_np = out.squeeze(1).cpu().numpy()  # B x H x W
                # append cropped tiles
                for widx, tile_out in enumerate(out_np):
                    r, c, h, w = batch_windows[widx]
                    outputs.append(tile_out)  # tile_out is tile_size x tile_size
                batch_tiles_b = []
                batch_tiles_a = []
                batch_windows = []

        # flush remaining
        if batch_tiles_b:
            tb = torch.from_numpy(np.stack(batch_tiles_b, axis=0)).to(device)
            ta = torch.from_numpy(np.stack(batch_tiles_a, axis=0)).to(device)
            with torch.no_grad():
                out = model(tb, ta)
                out_np = out.squeeze(1).cpu().numpy()
            for tile_out in out_np:
                outputs.append(tile_out)

    if args.verbose:
        print("Stitching tiles...")
    stitched = stitch_outputs(windows, outputs, H, W)

    # optional thresholding
    if args.threshold is not None:
        mask = (stitched > args.threshold).astype("uint8") * 255
        out_arr = mask.astype("float32")
    else:
        out_arr = stitched.astype("float32")

    # write geotiff
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_geotiff(args.out, out_arr, b_meta, cog=args.cog)

    t1 = time.time()
    print(f"[OK] Wrote {args.out}  runtime={t1-t0:.2f}s  windows={len(windows)}")

    # print a tiny summary
    print("out min/max/mean:", float(out_arr.min()), float(out_arr.max()), float(out_arr.mean()))


if __name__ == "__main__":
    main()
