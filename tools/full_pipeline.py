#!/usr/bin/env python3
"""
tools/full_pipeline.py

One-script demo pipeline:
 - discover AOIs (aoi/*.geojson) OR read explicit AOI list
 - for each AOI:
     * locate before/after GeoTIFFs in data/raw by pattern
     * optionally call ingest/gee_ingest.py to export (flag --fetch-gee)
     * run inference via tools/infer_and_stitch.py (uses PYTHONPATH=.)
     * generate visuals (before RGB, after RGB, heatmap, mask, overlay)
 - mosaic all per-AOI predictions into outputs/mosaic_india.tif
 - produce outputs/mosaic_metrics.csv and outputs/site/index.html
 - static site is saved to outputs/site/ (can be served with python -m http.server)

Usage (demo local-only):
  PYTHONPATH=. python tools/full_pipeline.py \
    --model runs/model_fused.pth \
    --pairs-dir data/raw \
    --out-dir outputs/aois \
    --site-out outputs/site \
    --threshold 0.5 \
    --tile-size 256 --overlap 32 --batch-size 4

Usage (try auto-fetch via GEE, advanced - you need ingest/gee_ingest.py + Drive/GCS):
  PYTHONPATH=. python tools/full_pipeline.py \
    --model runs/model_fused.pth \
    --pairs-dir data/raw \
    --fetch-gee \
    --gee-project radiant-works-474616-t3 \
    --drive-folder EO_Exports \
    --out-dir outputs/aois \
    --site-out outputs/site

Notes:
 - The script expects the repo root added to PYTHONPATH (we call subprocesses with PYTHONPATH set).
 - If you don't want to fetch from GEE, ensure your before/after GeoTIFFs are in data/raw and named with 'before'/'after' tokens so they are auto-paired.
 - The script will call tools/infer_and_stitch.py (so it must exist and be executable).
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.merge import merge
import matplotlib.pyplot as plt
import csv

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # ensure relative paths are repo-root relative


def find_aoi_geojsons(aoi_glob="aoi/*.geojson"):
    files = sorted(glob.glob(aoi_glob))
    return files


def pair_before_after_for_aoi(basename_token: str, pairs_dir: str):
    """
    Try to find before/after files in pairs_dir by searching for filenames that contain
    the basename_token and 'before' or 'after'. Returns tuple (before_path, after_path) or (None,None)
    """
    pd = Path(pairs_dir)
    if not pd.exists():
        return None, None
    # search heuristics
    before_candidates = sorted(glob.glob(str(pd / f"*{basename_token}*before*.tif")) + glob.glob(str(pd / f"*{basename_token}*_before*.tif")))
    after_candidates = sorted(glob.glob(str(pd / f"*{basename_token}*after*.tif")) + glob.glob(str(pd / f"*{basename_token}*_after*.tif")))
    if before_candidates and after_candidates:
        return before_candidates[0], after_candidates[0]
    # fallback: match prefix up to first underscore
    prefix = basename_token.split("_before")[0].split("_after")[0]
    bc = sorted(glob.glob(str(pd / f"{prefix}*before*.tif")) + glob.glob(str(pd / f"{prefix}*_before*.tif")))
    ac = sorted(glob.glob(str(pd / f"{prefix}*after*.tif")) + glob.glob(str(pd / f"{prefix}*_after*.tif")))
    if bc and ac:
        return bc[0], ac[0]
    return None, None


def run_infer_for_pair_python(before: str, after: str, model: str, out_path: str, tile_size: int, overlap: int, batch_size: int, threshold: float, verbose: bool):
    """
    Runs tools/infer_and_stitch.py by spawning a subprocess with PYTHONPATH set to repo root.
    """
    cmd = [
        sys.executable, "tools/infer_and_stitch.py",
        "--model", model,
        "--before", before,
        "--after", after,
        "--out", out_path,
        "--tile-size", str(tile_size),
        "--overlap", str(overlap),
        "--batch-size", str(batch_size),
    ]
    if threshold is not None:
        cmd += ["--threshold", str(threshold)]
    if verbose:
        cmd += ["--verbose"]
    env = os.environ.copy()
    # ensure repo root on PYTHONPATH
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env.get("PYTHONPATH", "")) if env.get("PYTHONPATH") else str(ROOT)
    print("[INF] running:", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print("[ERR] inference subprocess failed:", proc.stderr)
        raise RuntimeError(f"inference failed for {before} -> {out_path}")
    return out_path


def make_rgb_from_geotiff(path: str, out_png: str):
    """Create a simple RGB composite PNG from the first 3 bands (or duplicate single band)."""
    with rasterio.open(path) as ds:
        arr = ds.read().astype("float32")
    C, H, W = arr.shape
    if C >= 3:
        rgb = np.stack([arr[0], arr[1], arr[2]], axis=-1)
    else:
        # duplicate single band
        band = arr[0]
        rgb = np.stack([band, band, band], axis=-1)
    # normalize per-band using 2-98 percentile stretch
    out = np.zeros_like(rgb, dtype=np.uint8)
    for i in range(3):
        band = rgb[..., i].astype(np.float32)
        p2, p98 = np.nanpercentile(band, (2, 98))
        if p98 - p2 <= 0:
            scaled = np.clip(band, 0, 255)
        else:
            scaled = (band - p2) / (p98 - p2)
        out[..., i] = (np.clip(scaled, 0, 1) * 255).astype(np.uint8)
    plt.imsave(out_png, out)
    return out_png


def save_heat_and_mask(pred_tif: str, heat_png: str, mask_png: str, overlay_png: str, before_tif: str = None, threshold=0.5):
    with rasterio.open(pred_tif) as ds:
        arr = ds.read(1).astype("float32")
    # heat
    plt.imsave(heat_png, np.clip(arr, 0, 1), cmap="magma")
    # mask
    mask = (arr > threshold).astype("uint8") * 255
    plt.imsave(mask_png, mask, cmap="gray")
    # overlay (if before available)
    if before_tif and os.path.exists(before_tif):
        with rasterio.open(before_tif) as db:
            b = db.read().astype("float32")
        # create rgb
        C = b.shape[0]
        if C >= 3:
            rgb = np.stack([b[0], b[1], b[2]], axis=-1)
        else:
            rgb = np.stack([b[0], b[0], b[0]], axis=-1)
        # normalize
        out = np.zeros_like(rgb, dtype=np.uint8)
        for i in range(3):
            band = rgb[..., i].astype(np.float32)
            p2, p98 = np.nanpercentile(band, (2, 98))
            if p98 - p2 <= 0:
                scaled = np.clip(band, 0, 255)
            else:
                scaled = (band - p2) / (p98 - p2)
            out[..., i] = (np.clip(scaled, 0, 1) * 255).astype(np.uint8)
        # overlay red where mask==255
        overlay = out.copy()
        red_idx = mask.astype(bool)
        overlay[red_idx] = np.array([255, 0, 0], dtype=np.uint8)
        plt.imsave(overlay_png, overlay)
    else:
        # fallback: save overlay as heatmap
        plt.imsave(overlay_png, np.clip(arr, 0, 1), cmap="magma")


def mosaic_and_write(outputs_list: List[str], mosaic_out: str):
    srcs = [rasterio.open(p) for p in outputs_list]
    mosaic_arr, mosaic_transform = merge(srcs)
    mosaic_single = mosaic_arr[0]
    out_meta = srcs[0].meta.copy()
    out_meta.update({
        "height": mosaic_single.shape[0],
        "width": mosaic_single.shape[1],
        "transform": mosaic_transform,
        "count": 1,
        "dtype": "float32"
    })
    os.makedirs(os.path.dirname(mosaic_out) or ".", exist_ok=True)
    with rasterio.open(mosaic_out, "w", **out_meta) as dst:
        dst.write(mosaic_single.astype("float32"), 1)
    for s in srcs:
        s.close()
    return mosaic_out


def compute_metrics_for_outputs(outputs_list: List[str], threshold: float, metrics_csv: str):
    rows = []
    for p in outputs_list:
        with rasterio.open(p) as ds:
            arr = ds.read(1)
            tr = ds.transform
            try:
                px_area = abs(tr.a * tr.e)
            except Exception:
                px_area = None
            changed = int((arr > threshold).sum())
            area_ha = changed * px_area / 10000.0 if px_area else None
            rows.append([os.path.basename(p), changed, px_area, area_ha])
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aoi_output", "changed_pixels", "pixel_area_m2", "area_ha"])
        w.writerows(rows)
    return metrics_csv


def build_static_site(site_dir: str, per_aoi_info: List[dict], mosaic_path: str, metrics_csv: str):
    os.makedirs(site_dir, exist_ok=True)
    # copy visuals into site dir for simpler linking
    site_assets = Path(site_dir) / "assets"
    if site_assets.exists():
        shutil.rmtree(site_assets)
    site_assets.mkdir(parents=True, exist_ok=True)
    # copy files and build rows for HTML
    rows_html = []
    for info in per_aoi_info:
        # copy files to site assets
        def copy_to_site(p):
            if not p:
                return None
            dst = site_assets / Path(p).name
            shutil.copy(p, dst)
            return f"assets/{dst.name}"

        before = copy_to_site(info.get("before_png"))
        after = copy_to_site(info.get("after_png"))
        heat = copy_to_site(info.get("heat_png"))
        mask = copy_to_site(info.get("mask_png"))
        overlay = copy_to_site(info.get("overlay_png"))
        pred_tif = info.get("pred_tif")
        if pred_tif and Path(pred_tif).exists():
            dst_pred = site_assets / Path(pred_tif).name
            shutil.copy(pred_tif, dst_pred)
            pred_link = f"assets/{dst_pred.name}"
        else:
            pred_link = None
        rows_html.append({
            "name": info.get("name"),
            "before": before,
            "after": after,
            "heat": heat,
            "mask": mask,
            "overlay": overlay,
            "pred": pred_link,
            "metrics": info.get("metrics", {})
        })
    # copy mosaic and metrics CSV
    mosaic_link = None
    if mosaic_path and os.path.exists(mosaic_path):
        dst = site_assets / Path(mosaic_path).name
        shutil.copy(mosaic_path, dst)
        mosaic_link = f"assets/{dst.name}"
    if metrics_csv and os.path.exists(metrics_csv):
        dstm = site_assets / Path(metrics_csv).name
        shutil.copy(metrics_csv, dstm)
        metrics_link = f"assets/{dstm.name}"
    else:
        metrics_link = None

    # minimal HTML page with thumbnails and links
    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>India Deforestation - Visuals</title>",
        "<style>body{font-family:Arial} .grid{display:flex;flex-wrap:wrap} .card{width:320px;padding:8px;margin:6px;border:1px solid #ddd} img{max-width:100%}</style>",
        "</head><body>",
        "<h1>India Deforestation - Visuals</h1>",
    ]
    if mosaic_link:
        html.append(f"<h2>Mosaic</h2><a href='{mosaic_link}' target='_blank'><img src='{mosaic_link}' style='max-width:800px'></a>")
    html.append("<h2>Per-AOI thumbnails</h2><div class='grid'>")
    for r in rows_html:
        html.append("<div class='card'>")
        html.append(f"<h3>{r['name']}</h3>")
        if r['overlay']:
            html.append(f"<a href='{r['overlay']}' target='_blank'><img src='{r['overlay']}'></a>")
        elif r['heat']:
            html.append(f"<a href='{r['heat']}' target='_blank'><img src='{r['heat']}'></a>")
        if r['pred']:
            html.append(f"<p><a href='{r['pred']}' target='_blank'>Download prediction TIFF</a></p>")
        html.append("<p>Metrics:</p><ul>")
        for k,v in (r['metrics'].items() if r['metrics'] else []):
            html.append(f"<li>{k}: {v}</li>")
        html.append("</ul></div>")
    html.append("</div>")
    if metrics_link:
        html.append(f"<p><a href='{metrics_link}'>Download metrics CSV</a></p>")
    html.append("</body></html>")
    with open(Path(site_dir) / "index.html", "w") as f:
        f.write("\n".join(html))
    return Path(site_dir) / "index.html"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to fused model .pth")
    p.add_argument("--pairs-dir", default="data/raw", help="Where before/after TIFFs live")
    p.add_argument("--out-dir", default="outputs/aois", help="Where per-AOI pred TIFFs will be written")
    p.add_argument("--site-out", default="outputs/site", help="Where to write the static site")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--fetch-gee", action="store_true", help="If set, attempt to call ingest/gee_ingest.py for AOIs without local files (requires configuration)")
    p.add_argument("--gee-project", default=None, help="GEE project (if using fetch)")
    p.add_argument("--drive-folder", default=None, help="Drive folder name used by ingest/export (if using fetch)")
    p.add_argument("--aoi-glob", default="aoi/*.geojson", help="Glob for AOI geojsons")
    args = p.parse_args()

    aoi_files = find_aoi_geojsons(args.aoi_glob)
    if not aoi_files:
        print("[WARN] No AOI geojsons found with", args.aoi_glob)
        # fallback: attempt to pair all before/after in pairs-dir by name pattern
        be_files = sorted(glob.glob(str(Path(args.pairs_dir) / "*before*.tif")))
        aoi_files = []
        if not be_files:
            print("[ERROR] No AOI geojsons and no before files found. Put files in data/raw or create aoi/*.geojson")
            return
        else:
            # create pseudo-AOIs from names
            for bf in be_files:
                aoi_files.append(bf)  # store the before path as identifier

    os.makedirs(args.out_dir, exist_ok=True)
    per_aoi_info = []
    produced_preds = []

    for aoi in aoi_files:
        try:
            if aoi.endswith(".geojson"):
                name = Path(aoi).stem
                # try to pair using name token
                before, after = pair_before_after_for_aoi(name, args.pairs_dir)
            else:
                # aoi is actually a before path in fallback case
                before = aoi
                after = aoi.replace("before", "after")
                name = Path(before).stem

            if (not before) or (not after) or (not os.path.exists(before)) or (not os.path.exists(after)):
                print(f"[INFO] before/after missing for AOI {name}")
                if args.fetch_gee:
                    # attempt to call ingest/gee_ingest.py --- this is a naive call, expecting the ingest script to submit exports to Drive
                    print("[INFO] Attempting to run ingest/gee_ingest.py for", name)
                    # construct minimal command - you may need to adapt this to your ingest script signature
                    cmd = [sys.executable, "ingest/gee_ingest.py", "--aoi", aoi, "--before", "2024-01-01", "2024-01-31", "--after", "2024-11-01", "2024-11-30", "--name", name]
                    if args.gee_project:
                        cmd += ["--project", args.gee_project]
                    if args.drive_folder:
                        cmd += ["--drive-folder", args.drive_folder]
                    print("[INF] running fetch command:", " ".join(cmd))
                    if not args.dry_run:
                        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        print(proc.stdout)
                        if proc.returncode != 0:
                            print("[WARN] ingest/gee_ingest.py failed:", proc.stderr)
                        else:
                            print("[INFO] ingest submitted exports; you must download them with tools/download_from_drive.py (not automated here).")
                    else:
                        print("[DRY] would run ingest command:", " ".join(cmd))
                else:
                    print("[WARN] Skipping AOI (no local before/after):", name)
                    continue

            # Now ensure local before/after exist (search again)
            if not before or not os.path.exists(before):
                before, after = pair_before_after_for_aoi(name, args.pairs_dir)
            if not before or not after or not os.path.exists(before) or not os.path.exists(after):
                print("[WARN] Still missing before/after for", name, "— skipping")
                continue

            # compose out path
            pred_name = f"{name}_pred.tif"
            pred_out = str(Path(args.out_dir) / pred_name)

            if args.dry_run:
                print("[DRY] would infer:", before, "->", pred_out)
            else:
                # ensure output dir
                os.makedirs(os.path.dirname(pred_out) or ".", exist_ok=True)
                # run inference
                try:
                    run_infer_for_pair_python(before, after, args.model, pred_out, args.tile_size, args.overlap, args.batch_size, args.threshold, args.verbose)
                except Exception as e:
                    print("[ERR] inference failed for", name, ":", e)
                    continue

            # generate visuals
            visuals_dir = Path(args.site_out) / "visuals"
            visuals_dir.mkdir(parents=True, exist_ok=True)
            before_png = str(visuals_dir / f"{name}_before.png")
            after_png = str(visuals_dir / f"{name}_after.png")
            heat_png = str(visuals_dir / f"{name}_heat.png")
            mask_png = str(visuals_dir / f"{name}_mask.png")
            overlay_png = str(visuals_dir / f"{name}_overlay.png")
            # create rgb pnGs from before/after
            try:
                make_rgb_from_geotiff(before, before_png)
                make_rgb_from_geotiff(after, after_png)
            except Exception as e:
                print("[WARN] failed to create RGBs:", e)
            # create heat/mask/overlay from pred_out
            try:
                save_heat_and_mask(pred_out, heat_png, mask_png, overlay_png, before_tif=before, threshold=args.threshold)
            except Exception as e:
                print("[WARN] failed to create heat/mask/overlay:", e)
            # collect info
            # compute simple metrics for this AOI
            metrics = {}
            try:
                with rasterio.open(pred_out) as ds:
                    arr = ds.read(1)
                    tr = ds.transform
                    px_area = abs(tr.a * tr.e) if tr else None
                    changed = int((arr > args.threshold).sum())
                    area_ha = changed * px_area / 10000.0 if px_area else None
                    metrics = {"changed_pixels": changed, "pixel_area_m2": px_area, "area_ha": area_ha}
            except Exception:
                metrics = {}
            per_aoi_info.append({
                "name": name,
                "before_png": before_png if os.path.exists(before_png) else None,
                "after_png": after_png if os.path.exists(after_png) else None,
                "heat_png": heat_png if os.path.exists(heat_png) else None,
                "mask_png": mask_png if os.path.exists(mask_png) else None,
                "overlay_png": overlay_png if os.path.exists(overlay_png) else None,
                "pred_tif": pred_out,
                "metrics": metrics
            })
            produced_preds.append(pred_out)
        except Exception as e:
            print("[ERR] Unexpected error processing AOI", aoi, e)

    # mosaic produced preds that exist
    existing_preds = [p for p in produced_preds if os.path.exists(p)]
    mosaic_out = str(Path(args.site_out) / "mosaic_india.tif")
    metrics_csv = str(Path(args.site_out) / "mosaic_metrics.csv")
    if existing_preds:
        print("[INF] Mosaicking", len(existing_preds), "predictions")
        mosaic_path = mosaic_and_write(existing_preds, mosaic_out)
        compute_metrics_for_outputs(existing_preds, args.threshold, metrics_csv)
    else:
        mosaic_path = None
        print("[WARN] No predictions were produced; no mosaic created")

    # build static site
    site_index = build_static_site(args.site_out, per_aoi_info, mosaic_path, metrics_csv)
    print("[OK] site index:", site_index)
    print("To serve site locally: python -m http.server --directory", args.site_out)

if __name__ == "__main__":
    main()