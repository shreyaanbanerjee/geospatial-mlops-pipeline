#!/usr/bin/env python3
"""
ingest/gee_ingest.py

Programmatically fetches Sentinel-2 L2A images from Google Earth Engine for an AOI and
exports TWO cloud-masked mosaics ("before" and "after") as GeoTIFFs to:
- Google Drive (default), or
- Google Cloud Storage (if --gcs-bucket is provided).

Usage (Google Drive export):
  python ingest/gee_ingest.py \
    --aoi aoi.geojson \
    --before 2024-06-01 2024-06-15 \
    --after  2024-11-01 2024-11-15 \
    --name amazon_demo \
    --drive-folder EO_Exports

Usage (Google Cloud Storage export):
  python ingest/gee_ingest.py \
    --aoi aoi.geojson \
    --before 2024-06-01 2024-06-15 \
    --after  2024-11-01 2024-11-15 \
    --name amazon_demo \
    --gcs-bucket your-gcs-bucket \
    --gcs-prefix tiles/amazon

Notes
- Authenticate once: `earthengine authenticate`
- S2 L2A (Surface Reflectance) + SCL cloud/shadow masking.
- Creates median mosaic for each window, then masks clouds.
"""

import argparse
import json
import time
import ee

S2 = 'COPERNICUS/S2_SR'
# Keep only these SCL (scene classification) classes as valid surface:
# 4 = Vegetation, 5 = Not Vegetated, 6 = Water, 7 = Unclassified, 11 = Snow/Ice
VALID_SCL = [4, 5, 6, 7, 11]
SELECT_BANDS = ['B2','B3','B4','B8','B11','B12']  # Blue, Green, Red, NIR, SWIR1, SWIR2

def load_aoi(geojson_path):
    with open(geojson_path, 'r') as f:
        gj = json.load(f)
    return ee.Geometry(gj['features'][0]['geometry'])

def scl_mask(img):
    """Mask clouds/shadows using SCL band (keep VALID_SCL)."""
    scl = img.select('SCL')
    valid = None
    for v in VALID_SCL:
        mask_v = scl.eq(v)
        valid = mask_v if valid is None else valid.Or(mask_v)
    return img.updateMask(valid)

def median_mosaic(collection):
    return collection.median()

def get_mosaic(aoi, start_date, end_date, cloud_pct=60):
    coll = (ee.ImageCollection(S2)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct)))
    # Median mosaic, then mask using SCL
    mosaic = median_mosaic(coll).updateMask(ee.ImageCollection(S2)
                                            .filterBounds(aoi)
                                            .filterDate(start_date, end_date)
                                            .median()
                                            .select('SCL').neq(0))
    mosaic = scl_mask(mosaic).select(SELECT_BANDS)
    return mosaic.clip(aoi)

def export_image_to_drive(image, description, folder, scale=10, crs='EPSG:4326'):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        scale=scale,
        crs=crs,
        maxPixels=1e13
    )
    task.start()
    return task

def export_image_to_gcs(image, description, bucket, prefix="", scale=10, crs='EPSG:4326'):
    path = f"{prefix.rstrip('/')}/{description}"
    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=description,
        bucket=bucket,
        fileNamePrefix=path,
        scale=scale,
        crs=crs,
        maxPixels=1e13
    )
    task.start()
    return task

def wait_for_tasks(tasks):
    while any(t.status()['state'] in ('READY','RUNNING') for t in tasks):
        states = [t.status()['state'] for t in tasks]
        print(f"[INFO] Task states: {states}")
        time.sleep(30)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aoi', required=True, help='AOI GeoJSON (Polygon/MultiPolygon)')
    parser.add_argument('--before', nargs=2, required=True, metavar=('START','END'),
                        help='Before window: YYYY-MM-DD YYYY-MM-DD')
    parser.add_argument('--after', nargs=2, required=True, metavar=('START','END'),
                        help='After window: YYYY-MM-DD YYYY-MM-DD')
    parser.add_argument('--name', required=True, help='Base name for exports (e.g., amazon_demo)')
    parser.add_argument('--drive-folder', default='EO_Exports', help='Google Drive folder name')
    parser.add_argument('--gcs-bucket', default=None, help='GCS bucket name (if exporting to GCS)')
    parser.add_argument('--gcs-prefix', default='', help='Prefix/path inside GCS bucket')
    parser.add_argument('--scale', type=int, default=10, help='Export resolution in meters (S2 native 10m)')
    parser.add_argument('--crs', default='EPSG:4326', help='Target CRS for export')
    parser.add_argument('--cloud-pct', type=int, default=60, help='CLOUDY_PIXEL_PERCENTAGE upper bound')
    args = parser.parse_args()

    ee.Initialize()
    aoi = load_aoi(args.aoi)

    print("[INFO] Preparing mosaics…")
    before_img = get_mosaic(aoi, args.before[0], args.before[1], args.cloud_pct)
    after_img  = get_mosaic(aoi, args.after[0],  args.after[1],  args.cloud_pct)

    before_desc = f"{args.name}_before_{args.before[0]}_{args.before[1]}"
    after_desc  = f"{args.name}_after_{args.after[0]}_{args.after[1]}"

    tasks = []
    if args.gcs_bucket:
        print(f"[INFO] Exporting to GCS: gs://{args.gcs_bucket}/{args.gcs_prefix}")
        tasks.append(export_image_to_gcs(before_img, before_desc, args.gcs_bucket,
                                         args.gcs_prefix, args.scale, args.crs))
        tasks.append(export_image_to_gcs(after_img, after_desc, args.gcs_bucket,
                                         args.gcs_prefix, args.scale, args.crs))
    else:
        print(f"[INFO] Exporting to Google Drive folder: {args.drive_folder}")
        tasks.append(export_image_to_drive(before_img, before_desc, args.drive_folder,
                                           args.scale, args.crs))
        tasks.append(export_image_to_drive(after_img, after_desc, args.drive_folder,
                                           args.scale, args.crs))

    print("[INFO] Tasks started. Waiting…")
    wait_for_tasks(tasks)
    print("[DONE] Exports complete. Check Drive or GCS for outputs.")

if __name__ == '__main__':
    main()
