#!/usr/bin/env python3
"""
gey_ingest.py - simplified, robust Earth Engine ingestion script.

Usage example:
python ingest/gee_ingest.py \
  --project your-gcp-project \
  --aoi aoi/india_wayanad.geojson \
  --before 2024-01-01 2024-01-31 \
  --after  2024-11-01 2024-11-30 \
  --name india_wayanad \
  --drive-folder EO_Exports \
  --cloud-pct 60 --scale 10 --crs EPSG:4326

Requires earthengine-api installed and authenticated.
"""
import argparse
import json
import ee
import sys

def init_ee(project, service_account=None, key_file=None):
    if service_account and key_file:
        creds = ee.ServiceAccountCredentials(service_account, key_file)
        ee.Initialize(credentials=creds, project=project)
    else:
        ee.Initialize(project=project)

def load_aoi(path):
    with open(path, "r") as f:
        gj = json.load(f)
    # support FeatureCollection or single Feature
    feat = gj.get("features", [gj])[0]
    geom = ee.Geometry(feat["geometry"])
    return geom

def sentinel_composite(aoi, start, end, cloud_pct):
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(lambda i: i.clip(aoi))
    )
    size = col.size().getInfo()
    if size == 0:
        raise RuntimeError(f"No scenes found for {start}..{end} (cloud<{cloud_pct})")
    # median composite and select common bands
    img = col.median().select(["B2","B3","B4","B8","B11","B12"])
    return img

def export_to_drive(img, region, name, folder, scale, crs):
    desc = name
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=desc,
        folder=folder,
        fileNamePrefix=desc,
        scale=scale,
        crs=crs,
        region=region,
        maxPixels=1e13
    )
    task.start()
    print("[INFO] EARTH ENGINE TASK:", task.id)
    return task.id

def export_to_gcs(img, region, name, bucket, prefix, scale, crs):
    path = f"{prefix}/{name}"
    task = ee.batch.Export.image.toCloudStorage(
        image=img,
        description=name,
        bucket=bucket,
        fileNamePrefix=path,
        scale=scale,
        crs=crs,
        region=region,
        maxPixels=1e13
    )
    task.start()
    print("[INFO] EARTH ENGINE TASK:", task.id)
    return task.id

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--service-account", default=None)
    p.add_argument("--key-file", default=None)
    p.add_argument("--aoi", required=True)
    p.add_argument("--before", nargs=2, required=True)
    p.add_argument("--after", nargs=2, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--drive-folder", default=None)
    p.add_argument("--gcs-bucket", default=None)
    p.add_argument("--gcs-prefix", default="exports")
    p.add_argument("--scale", type=int, default=10)
    p.add_argument("--crs", default="EPSG:4326")
    p.add_argument("--cloud-pct", type=int, default=60)
    return p.parse_args()

def main():
    args = parse_args()
    init_ee(args.project, args.service_account, args.key_file)
    print("[INFO] Loading AOI:", args.aoi)
    aoi = load_aoi(args.aoi)
    before_start, before_end = args.before
    after_start, after_end = args.after

    print("[INFO] Building BEFORE composite:", before_start, before_end)
    before_img = sentinel_composite(aoi, before_start, before_end, args.cloud_pct)
    print("[INFO] Building AFTER composite:", after_start, after_end)
    after_img = sentinel_composite(aoi, after_start, after_end, args.cloud_pct)

    fname_before = f"{args.name}_before_{before_start}_{before_end}"
    fname_after  = f"{args.name}_after_{after_start}_{after_end}"

    if args.drive_folder:
        export_to_drive(before_img, aoi, fname_before, args.drive_folder, args.scale, args.crs)
        export_to_drive(after_img,  aoi, fname_after,  args.drive_folder, args.scale, args.crs)
    elif args.gcs_bucket:
        export_to_gcs(before_img, aoi, fname_before, args.gcs_bucket, args.gcs_prefix, args.scale, args.crs)
        export_to_gcs(after_img,  aoi, fname_after,  args.gcs_bucket, args.gcs_prefix, args.scale, args.crs)
    else:
        print("[ERR] Specify --drive-folder or --gcs-bucket", file=sys.stderr)
        sys.exit(2)
    print("[DONE] Submitted BEFORE and AFTER tasks.")

if __name__ == "__main__":
    main()
