#!/usr/bin/env python3
"""
tools/full_pipeline_proxy.py
Robust demo pipeline: compute proxy predictions from before/after, create visuals,
mosaic, metrics and a simple static site.

Usage:
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

def _safe_read_squeeze(path: str):
    with rasterio.open(path) as ds:
        arr = ds.read()  # typical shape (bands, H, W)
        meta = ds.meta.copy()
    # remove any accidental singleton dims
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    # ensure float32 for calculations
    return arr.astype('float32'), meta

def pair_before_after_for_aoi(basename_token: str, pairs_dir: str):
    pd = Path(pairs_dir)
    if not pd.exists():
        return None, None
    before_candidates = sorted(glob.glob(str(pd / f"*{basename_token}*before*.tif")) + glob.glob(str(pd / f"*{basename_token}*_before*.tif")))
    after_candidates = sorted(glob.glob(str(pd / f"*{basename_token}*after*.tif")) + glob.glob(str(pd / f"*{basename_token}*_after*.tif")))
    if before_candidates and after_candidates:
        return before_candidates[0], after_candidates[0]
    prefix = basename_token.split("_before")[0].split("_after")[0]
    bc = sorted(glob.glob(str(pd / f"{prefix}*before*.tif")) + glob.glob(str(pd / f"{prefix}*_before*.tif")))
    ac = sorted(glob.glob(str(pd / f"{prefix}*after*.tif")) + glob.glob(str(pd / f"{prefix}*_after*.tif")))
    if bc and ac:
        return bc[0], ac[0]
    return None, None

def compute_proxy_pred_before_after(before_tif: str, after_tif: str):
    """Return (pred_2d: HxW float32 in [0,1], write_meta dict)"""
    b, meta = _safe_read_squeeze(before_tif)
    a, _meta2 = _safe_read_squeeze(after_tif)

    try:
        # prefer NDVI delta if >=4 bands (assume B,G,R,NIR)
        if b.shape[0] >= 4 and a.shape[0] >= 4:
            def ndvi(x):
                nir = x[3]; red = x[2]
                denom = (nir + red)
                denom[denom == 0] = 1e-6
                return (nir - red) / denom
            ndvib = ndvi(b); ndvia = ndvi(a)
            pd = np.abs(ndvia - ndvib)
            p2, p98 = np.nanpercentile(pd, (2,98))
            pred = np.clip((pd - p2) / (p98 - p2 + 1e-9), 0.0, 1.0)
        else:
            # fallback: normalized absolute difference on first band
            diff = np.abs(a[0] - b[0])
            p2, p98 = np.nanpercentile(diff, (2,98))
            pred = np.clip((diff - p2) / (p98 - p2 + 1e-9), 0.0, 1.0)
    except Exception:
        # ultimate fallback: mean-of-bands diff
        mb = np.nanmean(b, axis=0)
        ma = np.nanmean(a, axis=0)
        diff = np.abs(ma - mb)
        p2, p98 = np.nanpercentile(diff, (2,98))
        pred = np.clip((diff - p2) / (p98 - p2 + 1e-9), 0.0, 1.0)

    if pred.ndim == 3:
        pred = pred[0]
    pred = pred.astype('float32')
    write_meta = meta.copy()
    write_meta.update({"count": 1, "dtype": "float32"})
    if "nodata" in write_meta and write_meta["nodata"] is None:
        write_meta.pop("nodata", None)
    return pred, write_meta

def write_pred_tif(pred_arr: np.ndarray, meta: dict, out_path: str):
    """Write single-band GeoTIFF; ensure shape and meta match."""
    if pred_arr.ndim != 2:
        pred_arr = np.squeeze(pred_arr)
        if pred_arr.ndim != 2:
            raise ValueError("pred_arr must be 2D after squeeze")
    m = meta.copy()
    m.update({"count": 1, "dtype": "float32"})
    # ensure width/height consistent
    H, W = pred_arr.shape
    m["height"] = H; m["width"] = W
    if "transform" not in m or m["transform"] is None:
        from rasterio.transform import Affine
        m["transform"] = Affine.translation(0,0)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with rasterio.open(out_path, "w", **m) as dst:
        dst.write(pred_arr.astype('float32')[np.newaxis, ...], 1)
    return out_path

def make_rgb_from_geotiff(path: str, out_png: str):
    with rasterio.open(path) as ds:
        arr = ds.read().astype('float32')
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[0] >= 3:
            arr = np.stack([arr[0], arr[1], arr[2]], axis=-1)
        else:
            arr = np.stack([arr[0], arr[0], arr[0]], axis=-1)
    out = np.zeros_like(arr, dtype=np.uint8)
    for i in range(3):
        band = arr[..., i].astype('float32')
        p2, p98 = np.nanpercentile(band, (2,98))
        if p98 - p2 <= 0:
            scaled = np.clip(band, 0, 255)
        else:
            scaled = (band - p2) / (p98 - p2)
        out[..., i] = (np.clip(scaled, 0, 1) * 255).astype(np.uint8)
    plt.imsave(out_png, out)
    return out_png

def save_heat_and_mask(pred_tif: str, heat_png: str, mask_png: str, overlay_png: str, before_tif: str=None, threshold=0.5):
    if not pred_tif or not os.path.exists(pred_tif):
        raise FileNotFoundError("pred_tif missing")
    with rasterio.open(pred_tif) as ds:
        arr = ds.read(1).astype('float32')
    plt.imsave(heat_png, np.clip(arr, 0, 1), cmap='magma')
    mask = (arr > threshold).astype('uint8') * 255
    plt.imsave(mask_png, mask, cmap='gray')
    if before_tif and os.path.exists(before_tif):
        with rasterio.open(before_tif) as db:
            b = db.read().astype('float32')
        b = np.squeeze(b)
        if b.ndim == 2:
            rgb = np.stack([b,b,b], axis=-1)
        else:
            rgb = np.stack([b[0], b[1] if b.shape[0]>1 else b[0], b[2] if b.shape[0]>2 else b[0]], axis=-1)
        out = np.zeros_like(rgb, dtype=np.uint8)
        for i in range(3):
            band = rgb[..., i].astype('float32')
            p2,p98 = np.nanpercentile(band,(2,98))
            if p98 - p2 <= 0:
                scaled = np.clip(band, 0, 255)
            else:
                scaled = (band - p2)/(p98 - p2)
            out[...,i] = (np.clip(scaled,0,1)*255).astype(np.uint8)
        overlay = out.copy()
        overlay[mask.astype(bool)] = np.array([255,0,0], dtype=np.uint8)
        plt.imsave(overlay_png, overlay)
    else:
        plt.imsave(overlay_png, np.clip(arr,0,1), cmap='magma')

def run_infer_for_pair_python(before, after, model, out_path, tile_size, overlap, batch_size, threshold, verbose):
    cmd = [sys.executable, "tools/infer_and_stitch.py", "--model", model, "--before", before, "--after", after, "--out", out_path, "--tile-size", str(tile_size), "--overlap", str(overlap), "--batch-size", str(batch_size)]
    if threshold is not None:
        cmd += ["--threshold", str(threshold)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env.get("PYTHONPATH","")) if env.get("PYTHONPATH") else str(ROOT)
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print("[WARN] infer_and_stitch failed:", proc.stderr)
        return None
    return out_path if os.path.exists(out_path) else None

def mosaic_and_write(outputs_list: List[str], mosaic_out: str):
    srcs = [rasterio.open(p) for p in outputs_list]
    mosaic_arr, mosaic_transform = merge(srcs)
    mosaic_single = mosaic_arr[0]
    out_meta = srcs[0].meta.copy()
    out_meta.update({"height": mosaic_single.shape[0], "width": mosaic_single.shape[1], "transform": mosaic_transform, "count": 1, "dtype": "float32"})
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

def build_static_site(site_dir, per_aoi_info, mosaic_path, metrics_csv):
    os.makedirs(site_dir, exist_ok=True)
    site_assets = Path(site_dir)/"assets"
    if site_assets.exists(): shutil.rmtree(site_assets)
    site_assets.mkdir(parents=True, exist_ok=True)
    rows_html = []
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
        pred_link = None
        if pred_tif and Path(pred_tif).exists():
            dst_pred = site_assets / Path(pred_tif).name
            shutil.copy(pred_tif, dst_pred)
            pred_link = f"assets/{dst_pred.name}"
        rows_html.append({"name": info.get("name"), "before": before, "after": after, "heat": heat, "mask": mask, "overlay": overlay, "pred": pred_link, "metrics": info.get("metrics", {})})
    mosaic_link = None
    if mosaic_path and os.path.exists(mosaic_path):
        dst = site_assets / Path(mosaic_path).name
        shutil.copy(mosaic_path, dst)
        mosaic_link = f"assets/{dst.name}"
    metrics_link = None
    if metrics_csv and os.path.exists(metrics_csv):
        dstm = site_assets / Path(metrics_csv).name
        shutil.copy(metrics_csv, dstm)
        metrics_link = f"assets/{dstm.name}"
    html = ["<!doctype html>","<html><head><meta charset='utf-8'><title>India Deforestation - Visuals</title>",
            "<style>body{font-family:Arial} .grid{display:flex;flex-wrap:wrap} .card{width:320px;padding:8px;margin:6px;border:1px solid #ddd} img{max-width:100%}</style>",
            "</head><body>","<h1>India Deforestation - Visuals</h1>"]
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
    p.add_argument("--proxy-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    aoi_files = sorted(glob.glob("aoi/*.geojson"))
    if not aoi_files:
        be_files = sorted(glob.glob(str(Path(args.pairs_dir) / "*before*.tif")))
        aoi_files = be_files

    os.makedirs(args.out_dir, exist_ok=True)
    per_aoi_info = []
    produced_preds = []
    for aoi in aoi_files:
        try:
            if aoi.endswith(".geojson"):
                name = Path(aoi).stem
                before, after = pair_before_after_for_aoi(name, args.pairs_dir)
            else:
                before = aoi
                after = aoi.replace("before", "after")
                name = Path(before).stem
            if not before or not after or not os.path.exists(before) or not os.path.exists(after):
                print("[WARN] missing before/after for", name, "- skipping")
                continue
            pred_name = f"{name}_pred.tif"
            pred_out = str(Path(args.out_dir)/pred_name)
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
                if args.model and Path("tools/infer_and_stitch.py").exists():
                    got = run_infer_for_pair_python(before, after, args.model, pred_out, args.tile_size, args.overlap, args.batch_size, args.threshold, args.verbose)
                    if got:
                        pred_created = got
                if not pred_created:
                    try:
                        pred_arr, meta = compute_proxy_pred_before_after(before, after)
                        write_pred_tif(pred_arr, meta, pred_out)
                        pred_created = pred_out
                        print("[INFO] fallback proxy pred written to", pred_out)
                    except Exception as e:
                        print("[ERR] could not create pred for", name, e)
                        continue
            visuals_dir = Path(args.site_out) / "visuals"
            visuals_dir.mkdir(parents=True, exist_ok=True)
            before_png = str(visuals_dir / f"{name}_before.png")
            after_png = str(visuals_dir / f"{name}_after.png")
            heat_png = str(visuals_dir / f"{name}_heat.png")
            mask_png = str(visuals_dir / f"{name}_mask.png")
            overlay_png = str(visuals_dir / f"{name}_overlay.png")
            try:
                make_rgb_from_geotiff(before, before_png)
                make_rgb_from_geotiff(after, after_png)
            except Exception as e:
                print("[WARN] make_rgb failed for", name, e)
            try:
                save_heat_and_mask(pred_created, heat_png, mask_png, overlay_png, before_tif=before, threshold=args.threshold)
            except Exception as e:
                print("[WARN] save_heat_and_mask failed for", name, e)
            metrics = {}
            try:
                with rasterio.open(pred_created) as ds:
                    arr = ds.read(1)
                    tr = ds.transform
                    px_area = abs(tr.a * tr.e) if tr else None
                    changed = int((arr > args.threshold).sum())
                    area_ha = changed * px_area / 10000.0 if px_area else None
                    metrics = {"changed_pixels": changed, "pixel_area_m2": px_area, "area_ha": area_ha}
            except Exception as e:
                print("[WARN] metrics compute failed for", name, e)
            per_aoi_info.append({"name": name, "before_png": before_png if os.path.exists(before_png) else None, "after_png": after_png if os.path.exists(after_png) else None, "heat_png": heat_png if os.path.exists(heat_png) else None, "mask_png": mask_png if os.path.exists(mask_png) else None, "overlay_png": overlay_png if os.path.exists(overlay_png) else None, "pred_tif": pred_created, "metrics": metrics})
            produced_preds.append(pred_created)
        except Exception as e:
            print("[ERR] unexpected error for aoi", aoi, e)

    existing_preds = [p for p in produced_preds if p and os.path.exists(p)]
    mosaic_out = str(Path(args.site_out) / "mosaic_india.tif")
    metrics_csv = str(Path(args.site_out) / "mosaic_metrics.csv")
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
