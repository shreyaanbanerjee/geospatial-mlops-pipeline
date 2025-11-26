#!/usr/bin/env python3
"""
tools/full_pipeline_proxy.py

A robust demo pipeline that will generate per-AOI prediction TIFFs and a static site even
if you don't yet have a model or Drive exports.

Modes:
 - Normal: tries to run inference by calling tools/infer_and_stitch.py (if model & tool exist).
 - Proxy-only (--proxy-only): compute a proxy prediction from before/after (NDVI delta or absolute diff).
 - Dry-run (--dry-run): will not run model inference but will create proxy predictions for visuals.

Usage (proxy-only, recommended for a quick demo):
PYTHONPATH=. python tools/full_pipeline_proxy.py --pairs-dir data/raw --out-dir outputs/aois --site-out outputs/site --proxy-only
"""
import argparse, glob, os, shutil, subprocess, sys, csv
from pathlib import Path
from typing import List
import numpy as np
import rasterio
from rasterio.merge import merge
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

def find_aoi_geojsons(aoi_glob="aoi/*.geojson"):
    return sorted(glob.glob(aoi_glob))

def pair_before_after_for_aoi(basename_token: str, pairs_dir: str):
    pd = Path(pairs_dir)
    if not pd.exists(): return None, None
    before_candidates = sorted(glob.glob(str(pd / f"*{basename_token}*before*.tif")) + glob.glob(str(pd / f"*{basename_token}*_before*.tif")))
    after_candidates = sorted(glob.glob(str(pd / f"*{basename_token}*after*.tif")) + glob.glob(str(pd / f"*{basename_token}*_after*.tif")))
    if before_candidates and after_candidates:
        return before_candidates[0], after_candidates[0]
    prefix = basename_token.split("_before")[0].split("_after")[0]
    bc = sorted(glob.glob(str(pd / f"{prefix}*before*.tif")) + glob.glob(str(pd / f"{prefix}*_before*.tif")))
    ac = sorted(glob.glob(str(pd / f"{prefix}*after*.tif")) + glob.glob(str(pd / f"{prefix}*_after*.tif")))
    if bc and ac: return bc[0], ac[0]
    return None, None

def compute_proxy_pred_before_after(before_tif: str, after_tif: str):
    """Compute a proxy prediction array (float32 0..1) given before/after geotiffs."""
    with rasterio.open(before_tif) as db:
        b = db.read().astype('float32')
        meta = db.meta.copy()
    with rasterio.open(after_tif) as da:
        a = da.read().astype('float32')
    # If there are >=4 bands assume band order [B,G,R,NIR] (Sentinel-2 style) and compute NDVI delta
    pred = None
    try:
        if b.shape[0] >= 4 and a.shape[0] >= 4:
            def ndvi(x):
                nir = x[3].astype('float32')
                red = x[2].astype('float32')
                denom = (nir + red)
                denom[denom == 0] = 1e-6
                return (nir - red) / denom
            ndvib = ndvi(b); ndvia = ndvi(a)
            # magnitude of change
            pd = np.abs(ndvia - ndvib)
            # clip and normalize robustly
            p2, p98 = np.nanpercentile(pd, (2,98))
            pred = np.clip((pd - p2) / (p98 - p2 + 1e-9), 0.0, 1.0)
        else:
            # fallback: normalized absolute difference on first band
            diff = np.abs(a[0].astype('float32') - b[0].astype('float32'))
            p2, p98 = np.nanpercentile(diff, (2,98))
            pred = np.clip((diff - p2) / (p98 - p2 + 1e-9), 0.0, 1.0)
    except Exception:
        # ultimate fallback: simple abs diff on mean of bands
        try:
            mb = np.nanmean(b, axis=0)
            ma = np.nanmean(a, axis=0)
            diff = np.abs(ma - mb)
            p2, p98 = np.nanpercentile(diff, (2,98))
            pred = np.clip((diff - p2) / (p98 - p2 + 1e-9), 0.0, 1.0)
        except Exception:
            # create a small zero array
            C,H,W = b.shape
            pred = np.zeros((H,W), dtype='float32')
    return pred.astype('float32'), meta

def write_pred_tif(pred_arr: np.ndarray, meta: dict, out_path: str):
    # write a single-band float32 geotiff using the meta from before image
    m = meta.copy()
    m.update({"count": 1, "dtype": "float32"})
    # ensure nodata not set incorrectly
    if "nodata" in m and m["nodata"] is None:
        m.pop("nodata", None)
    with rasterio.open(out_path, "w", **m) as dst:
        dst.write(pred_arr.astype('float32')[np.newaxis,...], 1)
    return out_path

def make_rgb_from_geotiff(path: str, out_png: str):
    with rasterio.open(path) as ds:
        arr = ds.read().astype("float32")
    C,H,W = arr.shape
    if C >= 3:
        rgb = np.stack([arr[0], arr[1], arr[2]], axis=-1)
    else:
        rgb = np.stack([arr[0], arr[0], arr[0]], axis=-1)
    out = np.zeros_like(rgb, dtype=np.uint8)
    for i in range(3):
        b = rgb[...,i]
        p2,p98 = np.nanpercentile(b, (2,98))
        if p98 - p2 <= 0:
            scaled = np.clip(b, 0, 255)
        else:
            scaled = (b - p2) / (p98 - p2)
        out[...,i] = (np.clip(scaled,0,1)*255).astype(np.uint8)
    plt.imsave(out_png, out)
    return out_png

def save_heat_and_mask(pred_tif: str, heat_png: str, mask_png: str, overlay_png: str, before_tif: str=None, threshold=0.5):
    with rasterio.open(pred_tif) as ds:
        arr = ds.read(1).astype('float32')
        meta = ds.meta
    plt.imsave(heat_png, np.clip(arr,0,1), cmap='magma')
    mask = (arr > threshold).astype('uint8') * 255
    plt.imsave(mask_png, mask, cmap='gray')
    if before_tif and os.path.exists(before_tif):
        with rasterio.open(before_tif) as db:
            b = db.read().astype('float32')
        C = b.shape[0]
        if C >= 3:
            rgb = np.stack([b[0], b[1], b[2]], axis=-1)
        else:
            rgb = np.stack([b[0], b[0], b[0]], axis=-1)
        out = np.zeros_like(rgb, dtype=np.uint8)
        for i in range(3):
            band = rgb[...,i]
            p2,p98 = np.nanpercentile(band,(2,98))
            if p98 - p2 <= 0:
                scaled = np.clip(band,0,255)
            else:
                scaled = (band - p2) / (p98 - p2)
            out[...,i] = (np.clip(scaled,0,1)*255).astype(np.uint8)
        overlay = out.copy()
        overlay[mask.astype(bool)] = np.array([255,0,0], dtype=np.uint8)
        plt.imsave(overlay_png, overlay)
    else:
        plt.imsave(overlay_png, np.clip(arr,0,1), cmap='magma')

def run_infer_for_pair_python(before, after, model, out_path, tile_size, overlap, batch_size, threshold, verbose):
    # attempt to call tools/infer_and_stitch.py; returns out_path if created
    cmd = [sys.executable, "tools/infer_and_stitch.py", "--model", model, "--before", before, "--after", after, "--out", out_path, "--tile-size",str(tile_size), "--overlap", str(overlap), "--batch-size", str(batch_size)]
    if threshold is not None:
        cmd += ["--threshold", str(threshold)]
    if verbose: cmd += ["--verbose"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env.get("PYTHONPATH","")) if env.get("PYTHONPATH") else str(ROOT)
    print("[CMD]", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print("[WARN] infer_and_stitch returned non-zero:", proc.stderr)
        return None
    return out_path if os.path.exists(out_path) else None

def mosaic_and_write(outputs_list: List[str], mosaic_out: str):
    srcs = [rasterio.open(p) for p in outputs_list]
    mosaic_arr, mosaic_transform = merge(srcs)
    mosaic_single = mosaic_arr[0]
    out_meta = srcs[0].meta.copy()
    out_meta.update({"height":mosaic_single.shape[0],"width":mosaic_single.shape[1],"transform":mosaic_transform,"count":1,"dtype":"float32"})
    os.makedirs(os.path.dirname(mosaic_out) or ".", exist_ok=True)
    with rasterio.open(mosaic_out,"w",**out_meta) as dst:
        dst.write(mosaic_single.astype('float32'),1)
    for s in srcs: s.close()
    return mosaic_out

def compute_metrics_for_outputs(outputs_list: List[str], threshold: float, metrics_csv: str):
    rows=[]
    for p in outputs_list:
        with rasterio.open(p) as ds:
            arr = ds.read(1)
            tr = ds.transform
            try:
                px_area = abs(tr.a * tr.e)
            except Exception:
                px_area = None
            changed = int((arr>threshold).sum())
            area_ha = changed * px_area / 10000.0 if px_area else None
            rows.append([os.path.basename(p), changed, px_area, area_ha])
    with open(metrics_csv,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["aoi_output","changed_pixels","pixel_area_m2","area_ha"])
        w.writerows(rows)
    return metrics_csv

def build_static_site(site_dir, per_aoi_info, mosaic_path, metrics_csv):
    os.makedirs(site_dir, exist_ok=True)
    site_assets = Path(site_dir)/"assets"
    if site_assets.exists(): shutil.rmtree(site_assets)
    site_assets.mkdir(parents=True, exist_ok=True)
    rows_html=[]
    for info in per_aoi_info:
        def copy_to_site(p):
            if not p: return None
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
        rows_html.append({"name": info.get("name"), "before": before, "after": after, "heat": heat, "mask": mask, "overlay": overlay, "pred": pred_link, "metrics": info.get("metrics",{})})
    mosaic_link=None
    if mosaic_path and os.path.exists(mosaic_path):
        dst = site_assets / Path(mosaic_path).name
        shutil.copy(mosaic_path, dst)
        mosaic_link = f"assets/{dst.name}"
    metrics_link = None
    if metrics_csv and os.path.exists(metrics_csv):
        dstm = site_assets / Path(metrics_csv).name
        shutil.copy(metrics_csv, dstm)
        metrics_link = f"assets/{dstm.name}"
    html = ["<!doctype html>","<html><head><meta charset='utf-8'><title>India Deforestation - Visuals</title>","<style>body{font-family:Arial} .grid{display:flex;flex-wrap:wrap} .card{width:320px;padding:8px;margin:6px;border:1px solid #ddd} img{max-width:100%}</style>","</head><body>","<h1>India Deforestation - Visuals</h1>"]
    if mosaic_link: html.append(f"<h2>Mosaic</h2><a href='{mosaic_link}' target='_blank'><img src='{mosaic_link}' style='max-width:800px'></a>")
    html.append("<h2>Per-AOI thumbnails</h2><div class='grid'>")
    for r in rows_html:
        html.append("<div class='card'>"); html.append(f"<h3>{r['name']}</h3>")
        if r['overlay']: html.append(f"<a href='{r['overlay']}' target='_blank'><img src='{r['overlay']}'></a>")
        elif r['heat']: html.append(f"<a href='{r['heat']}' target='_blank'><img src='{r['heat']}'></a>")
        if r['pred']: html.append(f"<p><a href='{r['pred']}' target='_blank'>Download prediction TIFF</a></p>")
        html.append("<p>Metrics:</p><ul>")
        for k,v in (r['metrics'].items() if r['metrics'] else []):
            html.append(f"<li>{k}: {v}</li>")
        html.append("</ul></div>")
    html.append("</div>")
    if metrics_link: html.append(f"<p><a href='{metrics_link}'>Download metrics CSV</a></p>")
    html.append("</body></html>")
    with open(Path(site_dir)/"index.html","w") as f:
        f.write("\n".join(html))
    return Path(site_dir)/"index.html"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-dir", default="data/raw")
    p.add_argument("--out-dir", default="outputs/aois")
    p.add_argument("--site-out", default="outputs/site")
    p.add_argument("--model", default=None)
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--proxy-only", action="store_true", help="Compute proxy predictions from before/after and skip heavy model inference")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # find AOIs: prefer aoi/*.geojson, else fallback to before files in pairs-dir
    aoi_files = find_aoi_geojsons("aoi/*.geojson")
    if not aoi_files:
        be_files = sorted(glob.glob(str(Path(args.pairs_dir)/"*before*.tif")))
        aoi_files = be_files

    os.makedirs(args.out_dir, exist_ok=True)
    per_aoi_info=[]
    produced_preds=[]

    for aoi in aoi_files:
        try:
            if aoi.endswith(".geojson"):
                name = Path(aoi).stem
                before, after = pair_before_after_for_aoi(name, args.pairs_dir)
            else:
                before = aoi
                after = aoi.replace("before","after")
                name = Path(before).stem
            if not before or not after or not os.path.exists(before) or not os.path.exists(after):
                print("[WARN] missing before/after for", name, "- skipping")
                continue
            pred_name = f"{name}_pred.tif"
            pred_out = str(Path(args.out_dir)/pred_name)

            # if proxy-only or dry-run, compute proxy pred and write as GeoTIFF
            pred_created = None
            if args.proxy_only or args.dry_run:
                try:
                    pred_arr, meta = compute_proxy_pred_before_after(before, after)
                    write_pred_tif(pred_arr, meta, pred_out)
                    pred_created = pred_out
                    print("[INFO] wrote proxy pred to", pred_out)
                except Exception as e:
                    print("[WARN] failed to create proxy pred for", name, e)
            else:
                # attempt to run real inference if model path exists and infer script present
                if args.model and Path("tools/infer_and_stitch.py").exists():
                    got = run_infer_for_pair_python(before, after, args.model, pred_out, args.tile_size, args.overlap, args.batch_size, args.threshold, args.verbose)
                    if got:
                        pred_created = got
                # fallback to proxy if inference didn't produce pred
                if not pred_created:
                    try:
                        pred_arr, meta = compute_proxy_pred_before_after(before, after)
                        write_pred_tif(pred_arr, meta, pred_out)
                        pred_created = pred_out
                        print("[INFO] fallback proxy pred written to", pred_out)
                    except Exception as e:
                        print("[ERR] could not create pred for", name, e)
                        continue

            # generate visuals (PNGs)
            visuals_dir = Path(args.site_out)/"visuals"
            visuals_dir.mkdir(parents=True, exist_ok=True)
            before_png = str(visuals_dir/f"{name}_before.png")
            after_png = str(visuals_dir/f"{name}_after.png")
            heat_png = str(visuals_dir/f"{name}_heat.png")
            mask_png = str(visuals_dir/f"{name}_mask.png")
            overlay_png = str(visuals_dir/f"{name}_overlay.png")

            try:
                make_rgb_from_geotiff(before, before_png)
                make_rgb_from_geotiff(after, after_png)
            except Exception as e:
                print("[WARN] make_rgb failed for", name, e)

            try:
                save_heat_and_mask(pred_created, heat_png, mask_png, overlay_png, before_tif=before, threshold=args.threshold)
            except Exception as e:
                print("[WARN] save_heat_and_mask failed for", name, e)

            # metrics
            metrics={}
            try:
                with rasterio.open(pred_created) as ds:
                    arr = ds.read(1)
                    tr = ds.transform
                    px_area = abs(tr.a * tr.e) if tr else None
                    changed = int((arr>args.threshold).sum())
                    area_ha = changed * px_area / 10000.0 if px_area else None
                    metrics={"changed_pixels":changed,"pixel_area_m2":px_area,"area_ha":area_ha}
            except Exception as e:
                print("[WARN] metrics compute failed for", name, e)

            per_aoi_info.append({"name":name,"before_png":before_png if os.path.exists(before_png) else None,"after_png":after_png if os.path.exists(after_png) else None,"heat_png":heat_png if os.path.exists(heat_png) else None,"mask_png":mask_png if os.path.exists(mask_png) else None,"overlay_png":overlay_png if os.path.exists(overlay_png) else None,"pred_tif":pred_created,"metrics":metrics})
            produced_preds.append(pred_created)
        except Exception as e:
            print("[ERR] unexpected error for aoi", aoi, e)

    existing_preds = [p for p in produced_preds if p and os.path.exists(p)]
    mosaic_out = str(Path(args.site_out)/"mosaic_india.tif")
    metrics_csv = str(Path(args.site_out)/"mosaic_metrics.csv")
    if existing_preds:
        print("[INF] mosaicking", len(existing_preds), "predictions")
        mosaic_path = mosaic_and_write(existing_preds, mosaic_out)
        compute_metrics_for_outputs(existing_preds, args.threshold, metrics_csv)
    else:
        mosaic_path = None
        print("[WARN] no preds found; skipping mosaic")

    site_index = build_static_site(args.site_out, per_aoi_info, mosaic_path, metrics_csv)
    print("[OK] site index:", site_index)
    print("Serve with: python -m http.server --directory", args.site_out)

if __name__ == "__main__":
    main()
